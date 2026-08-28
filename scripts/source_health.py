#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Layer 1 of the daily quality check: source health.

A feed or query returning zero items looks identical whether the source is
genuinely quiet, dead, frozen, or has simply collapsed — the only way to tell
them apart is that source's own history. This module:

  1. Harvests per-source yield from journald (the runner already logs
     ``Found N articles from <feed>`` / ``Found N articles for query: <q>``
     at INFO, plus parsing/scan failures at WARNING — no pipeline
     instrumentation needed, following the precedent set by
     ``scripts/brave_usage_report.py``).
  2. Keeps a rolling history in ``logs/source-health.jsonl``.
  3. Detects three distinct kinds of source rot against that history, plus
     Brave query dead-weight and NWS zone unreachability.

See references/quality_monitoring_design.md ("Layer 1 — Source health") for
the design this implements.
"""

import argparse
import json
import logging
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import feedparser
import requests
import yaml

from scripts.quality_findings import CRITICAL, INFO, WARN, Finding, sort_findings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HISTORY_PATH = "logs/source-health.jsonl"

# ---------------------------------------------------------------------------
# Journald line parsing
# ---------------------------------------------------------------------------

# Matches the `%(levelname)s: %(message)s` logging format every scanner uses,
# as rendered by `journalctl -o short-iso`:
#   2026-08-25T06:15:23-0700 eric-NUC7i7BNHX local-briefing[2251581]: INFO: <msg>
LINE_RE = re.compile(
    r"^(?P<ts>\S+)\s+\S+\s+(?P<ident>[a-zA-Z0-9_.-]+)\[\d+\]:\s+"
    r"(?P<level>[A-Z]+):\s+(?P<msg>.*)$"
)

FEED_YIELD_RE = re.compile(r"^Found (\d+) articles from (.+)$")
QUERY_YIELD_RE = re.compile(r"^Found (\d+) articles for query: (.+)$")
FEED_ERROR_RE = re.compile(r"^Feed (?:parsing issue|scan failed) for (.+?): (.+)$")
ALERTS_YIELD_RE = re.compile(r"^Found (\d+) active alerts for zones (.+)$")
ALERTS_ERROR_RE = re.compile(r"^NWS alerts fetch failed: (.+)$")

# The alerts fetch covers all configured zones in a single call, so all
# alerts_zone records share one synthetic name rather than one per zone.
ALERTS_ZONE_NAME = "alerts"


def _pipeline_name(unit: str) -> str:
    """Map a systemd tag (``atlas-briefing``) to a pipeline name (``atlas``)."""
    suffix = "-briefing"
    return unit[: -len(suffix)] if unit.endswith(suffix) else unit


def _parse_lines(lines: Iterable[str], units: Sequence[str]) -> List[Dict[str, Any]]:
    """Parse raw journald lines into flat per-log-line events."""
    pipeline_map = {u: _pipeline_name(u) for u in units}
    events: List[Dict[str, Any]] = []
    for line in lines:
        match = LINE_RE.match(line)
        if not match:
            continue
        ident = match.group("ident")
        pipeline = pipeline_map.get(ident)
        if pipeline is None:
            continue
        ts = match.group("ts")
        msg = match.group("msg")

        m = FEED_YIELD_RE.match(msg)
        if m:
            events.append(
                {"ts": ts, "pipeline": pipeline, "kind": "feed",
                 "name": m.group(2).strip(), "yield": int(m.group(1)), "hard_error": None}
            )
            continue

        m = QUERY_YIELD_RE.match(msg)
        if m:
            events.append(
                {"ts": ts, "pipeline": pipeline, "kind": "query",
                 "name": m.group(2).strip(), "yield": int(m.group(1)), "hard_error": None}
            )
            continue

        m = ALERTS_YIELD_RE.match(msg)
        if m:
            events.append(
                {"ts": ts, "pipeline": pipeline, "kind": "alerts_zone",
                 "name": ALERTS_ZONE_NAME, "yield": int(m.group(1)), "hard_error": None}
            )
            continue

        m = FEED_ERROR_RE.match(msg)
        if m:
            events.append(
                {"ts": ts, "pipeline": pipeline, "kind": "feed",
                 "name": m.group(1).strip(), "yield": None, "hard_error": m.group(2).strip()}
            )
            continue

        m = ALERTS_ERROR_RE.match(msg)
        if m:
            events.append(
                {"ts": ts, "pipeline": pipeline, "kind": "alerts_zone",
                 "name": ALERTS_ZONE_NAME, "yield": None, "hard_error": m.group(1).strip()}
            )
            continue

    return events


def _normalize_ts(ts: str) -> str:
    """Best-effort normalize a timestamp to ISO-8601 with a colon offset."""
    try:
        return datetime.fromisoformat(ts).isoformat()
    except ValueError:
        return ts


def _merge_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge same-day, same-source log events into one record per run.

    A single scanner run can log both a "Feed parsing issue" (or "Feed scan
    failed") warning *and* a "Found N articles" line for the same feed —
    the parse issue doesn't necessarily abort extraction. Both are folded
    into one record per (day, pipeline, kind, name) so the history file
    matches the one-record-per-source-per-run schema.
    """
    groups: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str, str, str]] = []
    for e in events:
        date_part = e["ts"][:10]
        key = (date_part, e["pipeline"], e["kind"], e["name"])
        if key not in groups:
            groups[key] = {
                "ts": e["ts"],
                "pipeline": e["pipeline"],
                "kind": e["kind"],
                "name": e["name"],
                "yield": None,
                "hard_error": None,
                "newest_entry": None,
            }
            order.append(key)
        rec = groups[key]
        if e["ts"] < rec["ts"]:
            rec["ts"] = e["ts"]
        if e["yield"] is not None:
            rec["yield"] = e["yield"]
        if e["hard_error"] is not None:
            rec["hard_error"] = e["hard_error"]

    records = [groups[k] for k in order]
    for r in records:
        r["ts"] = _normalize_ts(r["ts"])
    return records


def harvest_journal(
    since: str = "-1d",
    units: Sequence[str] = ("atlas-briefing", "local-briefing"),
    timeout: int = 180,
) -> List[Dict[str, Any]]:
    """
    Harvest per-source yield records from journald.

    Graceful degradation is a hard rule in this repo: if journalctl is
    missing, times out, or returns nothing usable, log a warning and return
    an empty list. Never raise.
    """
    cmd = ["journalctl", "--since", since]
    for unit in units:
        cmd += ["-t", unit]
    cmd += ["-o", "short-iso", "--no-pager"]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        logger.warning("journalctl not found; returning empty source-health harvest")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("journalctl timed out after %ss; returning empty source-health harvest", timeout)
        return []
    except OSError as e:
        logger.warning("journalctl failed to launch: %s", e)
        return []

    if proc.returncode != 0:
        logger.warning("journalctl exited %s: %s", proc.returncode, (proc.stderr or "").strip()[:200])
        return []

    if not proc.stdout:
        return []

    events = _parse_lines(proc.stdout.splitlines(), units)
    records = _merge_events(events)
    logger.info("harvested %d source-health record(s) from journald since %s", len(records), since)
    return records


# ---------------------------------------------------------------------------
# History file (logs/source-health.jsonl)
# ---------------------------------------------------------------------------


def _record_key(record: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """The identity of a history record: (ts, pipeline, kind, name).

    Overlapping ``--since`` harvest windows (``-30d`` then ``-3d`` then
    ``-2d``) re-harvest the same runs from journald; this key is what makes
    re-appending them a no-op instead of an inflated streak.
    """
    return (record.get("ts"), record.get("pipeline"), record.get("kind"), record.get("name"))


def _existing_keys(path: Path) -> Set[Tuple[Any, Any, Any, Any]]:
    keys: Set[Tuple[Any, Any, Any, Any]] = set()
    if not path.exists():
        return keys
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(_record_key(rec))
    return keys


def append_history(records: Iterable[Dict[str, Any]], path: str = HISTORY_PATH) -> int:
    """Append records to the history JSONL, one JSON object per line.

    Idempotent on ``(ts, pipeline, kind, name)``: a record already on disk
    (or already written earlier in this same batch) is silently skipped
    rather than duplicated, so re-harvesting an overlapping journald window
    never inflates a streak.
    """
    records = list(records)
    if not records:
        return 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    seen = _existing_keys(p)
    written = 0
    with p.open("a") as f:
        for r in records:
            key = _record_key(r)
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(r, sort_keys=True) + "\n")
            written += 1
    return written


def _resolve_since(since: Optional[str]) -> Optional[datetime]:
    """Resolve a ``--since``-style value (``-30d``, ``-12h``, or an ISO date)."""
    if not since:
        return None
    since = since.strip()
    m = re.match(r"^-(\d+)([dh])$", since)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = timedelta(days=n) if unit == "d" else timedelta(hours=n)
        return datetime.now(timezone.utc) - delta
    try:
        dt = datetime.fromisoformat(since)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        logger.warning("could not parse --since value %r; ignoring filter", since)
        return None


def load_history(path: str = HISTORY_PATH, since: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load history records, skipping malformed lines, optionally filtered by ``since``.

    Defensively de-duplicates on ``(ts, pipeline, kind, name)`` while
    reading, so a file that already accumulated duplicate lines (from
    before ``append_history`` became idempotent, or from any other source)
    stops skewing streak/median calculations without anyone hand-editing
    the file. The first occurrence of a key wins; later duplicates are
    dropped.
    """
    p = Path(path)
    if not p.exists():
        return []

    cutoff = _resolve_since(since)
    seen: Set[Tuple[Any, Any, Any, Any]] = set()
    records: List[Dict[str, Any]] = []
    with p.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed history line %d in %s", lineno, path)
                continue
            key = _record_key(rec)
            if key in seen:
                continue
            seen.add(key)
            if cutoff is not None:
                rec_dt = _safe_parse_ts(rec.get("ts", ""))
                if rec_dt is not None and rec_dt < cutoff:
                    continue
            records.append(rec)
    return records


def _safe_parse_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _ts_sort_key(record: Dict[str, Any]) -> datetime:
    dt = _safe_parse_ts(record.get("ts", ""))
    return dt if dt is not None else datetime.min.replace(tzinfo=timezone.utc)


def _group_by_pipeline_name(
    history: List[Dict[str, Any]], kind: str
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Group history records of one ``kind`` by (pipeline, name), sorted oldest-first."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in history:
        if r.get("kind") != kind:
            continue
        key = (r.get("pipeline", ""), r.get("name", ""))
        groups[key].append(r)
    for key in groups:
        groups[key].sort(key=_ts_sort_key)
    return dict(groups)


# ---------------------------------------------------------------------------
# Live feed probing
# ---------------------------------------------------------------------------


def _newest_entry_date(entries: List[Any]) -> Optional[datetime]:
    best: Optional[datetime] = None
    for entry in entries:
        for key in ("published_parsed", "updated_parsed"):
            t = entry.get(key) if hasattr(entry, "get") else None
            if t:
                try:
                    dt = datetime(*t[:6], tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    continue
                if best is None or dt > best:
                    best = dt
                break
    return best


def probe_feed(name: str, url: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Live-check one feed. Never raises — failures are captured in ``error``.
    """
    result: Dict[str, Any] = {
        "name": name,
        "url": url,
        "status": None,
        "content_type": None,
        "entries": 0,
        "newest_entry": None,
        "error": None,
    }
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "atlas-morning-briefing/source-health"},
        )
        result["status"] = resp.status_code
        result["content_type"] = resp.headers.get("Content-Type")
        parsed = feedparser.parse(resp.content)
        entries = list(getattr(parsed, "entries", None) or [])
        result["entries"] = len(entries)
        newest = _newest_entry_date(entries)
        if newest is not None:
            result["newest_entry"] = newest.isoformat()
    except requests.RequestException as e:
        result["error"] = str(e)
    except Exception as e:  # pragma: no cover - defensive, probe_feed must never raise
        result["error"] = str(e)
    return result


def probe_feeds(feeds: Sequence[Dict[str, str]], timeout: int = 20) -> List[Dict[str, Any]]:
    """Live-check a list of ``{"name": ..., "url": ...}`` feed dicts."""
    results = []
    for feed in feeds:
        name = feed.get("name", "")
        url = feed.get("url", "")
        results.append(probe_feed(name, url, timeout=timeout))
    return results


# ---------------------------------------------------------------------------
# Rot detection
# ---------------------------------------------------------------------------

DEFAULT_RULES: Dict[str, Any] = {
    "stale_after_days": 90,
    "zero_run_threshold": 7,
    "median_window": 30,
    "dead_url_streak": 3,
    "query_window": 14,
    "query_mean_threshold": 1.0,
    "query_min_runs": 3,
    # Per-feed override, e.g. {"Lilian Weng": {"stale_after_days": 400}}
    "feed_overrides": {},
}


def _trailing_error_streak(records: List[Dict[str, Any]]) -> int:
    """
    Count consecutive trailing (most-recent-first) runs that both errored
    and produced nothing.

    A run only counts toward the dead-URL streak when ``hard_error`` is set
    AND ``yield`` is falsy (0 or None). The harvester merges a same-run bozo
    warning with its "Found N articles" line into one record, so a feed can
    carry a cosmetic parser warning (e.g. an encoding-declaration mismatch)
    on every run while still delivering content — that is a live feed with
    a noisy log, not a dead one. Any run with a nonzero yield is a success
    and resets the streak, even if that same run also logged an error.
    """
    streak = 0
    for r in reversed(records):
        if r.get("hard_error") and not r.get("yield"):
            streak += 1
        else:
            break
    return streak


def _probe_is_dead(probe: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    if probe.get("error"):
        return True, str(probe["error"])
    status = probe.get("status")
    if status != 200:
        return True, f"HTTP {status}"
    content_type = (probe.get("content_type") or "").lower()
    if "html" in content_type:
        return True, f"content-type {probe.get('content_type')}"
    if not probe.get("entries"):
        return True, "0 parsed entries"
    return False, None


def _has_yield_in_window(records: List[Dict[str, Any]], window: int) -> bool:
    """True if any of the trailing ``window`` records (oldest-first input)
    recorded a nonzero yield.

    Used to tell "dead" from "quiet with a cosmetic warning": a feed that
    logs a hard_error on every run (e.g. an encoding-declaration mismatch)
    but still delivered content somewhere in recent memory is degraded, not
    dead, even if its most recent runs happen to string together an
    error+zero streak past the threshold.
    """
    return any(r.get("yield") for r in records[-window:])


def _detect_dead_url(
    history: List[Dict[str, Any]], probes: List[Dict[str, Any]], rules: Dict[str, Any]
) -> List[Finding]:
    findings: List[Finding] = []
    streak_threshold = rules["dead_url_streak"]
    median_window = rules["median_window"]
    probes_by_name = {p.get("name"): p for p in probes}
    feed_groups = _group_by_pipeline_name(history, kind="feed")

    seen_names = set()
    for (pipeline, name), records in feed_groups.items():
        seen_names.add(name)
        streak = _trailing_error_streak(records)
        # A raw streak past the threshold is only *confirmed* dead-by-history
        # when there is also no successful delivery anywhere in the trailing
        # window -- otherwise it's a live feed having a quiet spell that
        # happens to also carry a noisy cosmetic warning on every run.
        streak_confirmed = streak >= streak_threshold and not _has_yield_in_window(records, median_window)
        probe = probes_by_name.get(name)
        probe_bad, probe_reason = _probe_is_dead(probe) if probe else (False, None)

        if not streak_confirmed and not probe_bad:
            continue

        severity = CRITICAL if streak_confirmed else WARN
        reason = probe_reason or (records[-1].get("hard_error") if records else None) or "repeated fetch errors"
        streak_note = f" ({streak} consecutive runs with errors)" if streak else ""
        findings.append(
            Finding(
                severity=severity,
                code="feed-dead-url",
                message=f"{name}: {reason}{streak_note}",
                source=name,
                pipeline=pipeline,
                detail={"consecutive_error_runs": streak, "probe": probe},
            )
        )

    # Probed feeds with no history yet still deserve a first-sighting WARN.
    for name, probe in probes_by_name.items():
        if name in seen_names:
            continue
        probe_bad, probe_reason = _probe_is_dead(probe)
        if not probe_bad:
            continue
        findings.append(
            Finding(
                severity=WARN,
                code="feed-dead-url",
                message=f"{name}: {probe_reason}",
                source=name,
                pipeline="",
                detail={"consecutive_error_runs": 0, "probe": probe},
            )
        )

    return findings


def _detect_frozen_feed(probes: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Finding]:
    findings: List[Finding] = []
    overrides = rules.get("feed_overrides") or {}
    default_days = rules["stale_after_days"]
    now = datetime.now(timezone.utc)

    for probe in probes:
        name = probe.get("name", "")
        is_dead, _ = _probe_is_dead(probe)
        if is_dead:
            continue  # Mode A owns dead/unparseable feeds.
        newest = probe.get("newest_entry")
        if not newest:
            continue
        newest_dt = _safe_parse_ts(newest)
        if newest_dt is None:
            continue
        threshold_days = overrides.get(name, {}).get("stale_after_days", default_days)
        age_days = (now - newest_dt).days
        if age_days >= threshold_days:
            findings.append(
                Finding(
                    severity=WARN,
                    code="feed-frozen",
                    message=(
                        f"{name}: newest entry {newest_dt.date().isoformat()} "
                        f"({age_days}d old, threshold {threshold_days}d)"
                    ),
                    source=name,
                    pipeline="",
                    detail={"newest_entry": newest, "age_days": age_days, "threshold_days": threshold_days},
                )
            )
    return findings


def _detect_yield_collapse(history: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Finding]:
    findings: List[Finding] = []
    zero_run_threshold = rules["zero_run_threshold"]
    median_window = rules["median_window"]

    for (pipeline, name), records in _group_by_pipeline_name(history, kind="feed").items():
        numeric = [r for r in records if r.get("yield") is not None]
        if not numeric:
            continue

        streak = 0
        for r in reversed(numeric):
            if r["yield"] == 0:
                streak += 1
            else:
                break
        if streak < zero_run_threshold:
            continue

        window = numeric[-median_window:]
        yields = [r["yield"] for r in window]
        median_yield = statistics.median(yields)
        if median_yield < 1:
            continue  # Genuinely quiet source (e.g. Lilian Weng) — no alarm.

        findings.append(
            Finding(
                severity=WARN,
                code="feed-yield-collapse",
                message=(
                    f"{name}: {streak} consecutive zero-yield runs despite a trailing "
                    f"median yield of {median_yield:g} over the last {len(window)} runs"
                ),
                source=name,
                pipeline=pipeline,
                detail={"zero_streak": streak, "median_yield": median_yield, "window": len(window)},
            )
        )
    return findings


def _detect_query_dead_weight(history: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Finding]:
    findings: List[Finding] = []
    query_window = rules["query_window"]
    threshold = rules["query_mean_threshold"]
    min_runs = rules["query_min_runs"]

    for (pipeline, name), records in _group_by_pipeline_name(history, kind="query").items():
        numeric = [r["yield"] for r in records if r.get("yield") is not None]
        if len(numeric) < min_runs:
            continue
        window = numeric[-query_window:]
        mean_yield = sum(window) / len(window)
        if mean_yield >= threshold:
            continue
        findings.append(
            Finding(
                severity=INFO,
                code="query-dead-weight",
                message=f"{name}: mean yield {mean_yield:.2f} over the last {len(window)} runs — pure quota burn",
                source=name,
                pipeline=pipeline,
                detail={"mean_yield": mean_yield, "window": len(window)},
            )
        )
    return findings


def _detect_zone_unreachable(history: List[Dict[str, Any]]) -> List[Finding]:
    findings: List[Finding] = []
    for (pipeline, name), records in _group_by_pipeline_name(history, kind="alerts_zone").items():
        if not records:
            continue
        latest = records[-1]
        hard_error = latest.get("hard_error")
        if not hard_error:
            continue  # Zero active alerts (or a healthy fetch) is normal, never a finding.
        findings.append(
            Finding(
                severity=WARN,
                code="zone-unreachable",
                message=f"NWS alerts fetch failing: {hard_error}",
                source=name,
                pipeline=pipeline,
                detail={"hard_error": hard_error},
            )
        )
    return findings


def detect_rot(
    history: List[Dict[str, Any]],
    probes: Optional[List[Dict[str, Any]]] = None,
    rules: Optional[Dict[str, Any]] = None,
    live_sources: Optional[Set[str]] = None,
) -> List[Finding]:
    """
    Run all Layer 1 detection rules and return sorted findings.

    ``probes`` (from :func:`probe_feeds`) are only needed for Modes A/B live
    checks; Modes C, query dead-weight, and zone-unreachable work from
    ``history`` alone.

    ``live_sources``, when given, is the set of feed ``name`` values still
    configured (across all pipelines being checked). A feed deleted or
    renamed in config.yaml has no way to recover on its own -- its
    historical error records just keep firing every morning until someone
    hand-edits the log file -- so any history-derived ``feed``-kind record
    whose name isn't in ``live_sources`` is dropped before detection runs.
    Leaving this ``None`` (the default) disables the filter entirely,
    preserving prior behavior for callers that don't pass it. This only
    filters *history*; a live probe result stands on its own regardless,
    since a caller only probes feeds it still has configured.
    """
    merged_rules = {**DEFAULT_RULES, **(rules or {})}
    probes = probes or []

    if live_sources is not None:
        history = [
            r for r in history
            if r.get("kind") != "feed" or r.get("name") in live_sources
        ]

    findings: List[Finding] = []
    findings.extend(_detect_dead_url(history, probes, merged_rules))
    findings.extend(_detect_frozen_feed(probes, merged_rules))
    findings.extend(_detect_yield_collapse(history, merged_rules))
    findings.extend(_detect_query_dead_weight(history, merged_rules))
    findings.extend(_detect_zone_unreachable(history))
    return sort_findings(findings)


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------


def build_report(history: List[Dict[str, Any]]) -> str:
    """Markdown table of the most recent record per (pipeline, kind, source)."""
    latest: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in history:
        key = (r.get("pipeline", ""), r.get("kind", ""), r.get("name", ""))
        if key not in latest or _ts_sort_key(r) >= _ts_sort_key(latest[key]):
            latest[key] = r

    lines = [
        "# Source Health",
        "",
        f"_{len(latest)} source(s) tracked, {len(history)} total record(s)._",
        "",
        "| Pipeline | Kind | Source | Last yield | Last error | Last seen |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(latest):
        r = latest[key]
        pipeline, kind, name = key
        yield_val = r.get("yield")
        yield_display = "—" if yield_val is None else str(yield_val)
        error_display = r.get("hard_error") or ""
        lines.append(
            f"| {pipeline} | {kind} | {name} | {yield_display} | {error_display} | {r.get('ts', '')} |"
        )
    return "\n".join(lines) + "\n"


def _load_config(config_path: str) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        logger.warning("config file not found: %s", config_path)
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        logger.warning("could not parse config file %s: %s", config_path, e)
        return {}


def _load_rules(config_path: str) -> Dict[str, Any]:
    config = _load_config(config_path)
    return (config.get("quality_check") or {}).get("source_health") or {}


def _load_feeds(config_path: str) -> List[Dict[str, str]]:
    config = _load_config(config_path)
    return config.get("blog_feeds") or []


def _live_source_names(feeds: List[Dict[str, str]]) -> Set[str]:
    return {f.get("name") for f in feeds if f.get("name")}


def _print_findings(findings: List[Finding]) -> None:
    if not findings:
        print("No findings — all sources healthy.")
        return
    for f in findings:
        pipeline_prefix = f"[{f.pipeline}] " if f.pipeline else ""
        print(f"[{f.severity}] {f.code} {pipeline_prefix}{f.message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer 1 quality check: source health")
    parser.add_argument(
        "--since",
        default="-1d",
        help=(
            "journalctl --since window for harvesting, e.g. -1d, -12h (default: -1d). "
            "Use the --since=-3d form (with '=') — a space before a negative value "
            "is parsed by argparse as an unknown option, e.g. '--since -3d' fails."
        ),
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--history", default=HISTORY_PATH, help="Path to the source-health history JSONL")
    parser.add_argument("--probe", action="store_true", help="Live-probe every feed in --config")
    parser.add_argument("--report", action="store_true", help="Print a markdown table of current source health")
    parser.add_argument("--timeout", type=int, default=180, help="journalctl subprocess timeout (seconds)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    rules = _load_rules(args.config)

    if args.report:
        history = load_history(args.history)
        print(build_report(history))
        return 0

    feeds = _load_feeds(args.config)
    # An empty/missing config is treated as "no filter" rather than "filter
    # everything out" -- graceful degradation, same as the rest of this repo.
    live_sources = _live_source_names(feeds) or None

    if args.probe:
        logger.info("probing %d feed(s) from %s", len(feeds), args.config)
        probes = probe_feeds(feeds)
        history = load_history(args.history)
        findings = detect_rot(history, probes=probes, rules=rules, live_sources=live_sources)
        _print_findings(findings)
        return 0

    records = harvest_journal(since=args.since, timeout=args.timeout)
    appended = append_history(records, path=args.history)
    logger.info("appended %d record(s) to %s", appended, args.history)

    history = load_history(args.history)
    findings = detect_rot(history, rules=rules, live_sources=live_sources)
    _print_findings(findings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
