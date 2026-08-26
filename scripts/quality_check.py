#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Daily quality check orchestrator.

Runs three independently-skippable layers over the morning briefing
pipelines (default: ``config.yaml`` as pipeline "atlas", ``config_local.yaml``
as pipeline "local"):

  Layer 1 -- source health (journald harvest -> history -> rot detection),
             implemented in ``scripts/source_health.py``.
  Layer 2 -- report invariants, a deterministic scan of today's rendered
             briefing markdown, implemented in ``scripts/report_invariants.py``.
  Layer 3 -- an LLM judge that scores the rendered markdown against a
             six-dimension rubric (this module).

All three layers report the shared ``Finding`` type from
``scripts/quality_findings.py``. The orchestrator merges, sorts, digests,
and routes alerts (CRITICAL sends an email immediately, through the same
delivery path the morning briefing itself uses; WARN/INFO land in the
digest only), then returns an exit code cron can read.

See references/quality_monitoring_design.md for the design this implements.

Graceful degradation is a hard rule here, same as the pipeline itself
(GEMINI.md): a missing sibling module, a failed LLM call, or malformed judge
output must degrade to a Finding, never raise past this module's boundary.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from scripts.quality_findings import (
    CRITICAL,
    INFO,
    SEVERITY_ORDER,
    WARN,
    Finding,
    counts_by_severity,
    sort_findings,
    worst_severity,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# The six Layer 3 rubric dimensions, in the order the design doc lists them.
RUBRIC_DIMENSIONS = (
    "tier_1_share",
    "lead_alignment",
    "actionability",
    "locality",
    "specificity",
    "freshness",
)

DEFAULT_HISTORY_PATH = "logs/source-health.jsonl"
DEFAULT_SCORES_PATH = "logs/quality-scores.jsonl"
DEFAULT_ALERTS_PATH = "logs/quality-alerts.json"

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict quality auditor for a daily briefing. You score, you do "
    "not edit or rewrite. You always answer with machine-readable JSON only."
)


class InvalidJudgeDimension(ValueError):
    """Raised when quality_check.judge.dimensions in a pipeline config names
    a dimension the rubric doesn't define. Caught at the call site and
    turned into a reported Finding -- never left to crash the run."""


# One-line rubric description per dimension, keyed the same as
# RUBRIC_DIMENSIONS. Used to build the prompt for exactly the dimensions a
# given pipeline is configured to score.
_DIMENSION_RUBRIC: Dict[str, str] = {
    "tier_1_share": (
        "2 = most items are daily-life/neighborhood-relevant; "
        "0 = dominated by policy and market items."
    ),
    "lead_alignment": (
        "2 = the executive summary opens on what changes for the reader; "
        "0 = it opens on a development or investment story."
    ),
    "actionability": (
        "2 = items carry a date, place, cost, or decision; "
        "0 = analysis with no handle for action."
    ),
    "locality": (
        "2 = everything is in or about the area; "
        "0 = contains items from elsewhere."
    ),
    "specificity": "2 = names, numbers, streets; 0 = generic filler.",
    "freshness": "2 = nothing already past; 0 = advertises expired events.",
}


def resolve_judge_dimensions(config: Dict[str, Any]) -> List[str]:
    """Which rubric dimensions this pipeline's judge should score.

    Reads ``config["quality_check"]["judge"]["dimensions"]``. A briefing
    like Atlas (defense/AI/space, deliberately not neighborhood-local) has
    no meaningful answer for "tier_1_share" or "locality" -- scoring it on
    those forever reports a permanently-red number that teaches the reader
    to ignore the whole section, and drags detect_quality_regression's
    baseline around for no reason. A pipeline opts into a narrower rubric
    by listing exactly the dimensions that apply to it.

    When the key is absent, this returns all six dimensions -- today's
    behavior, unchanged, so an unconfigured briefing never silently changes
    what it's scored on.

    Raises InvalidJudgeDimension (a ValueError) if the config names a
    dimension the rubric doesn't define, so a typo is rejected loudly
    rather than silently asking the model to score something undefined.
    """
    judge_cfg = (config.get("quality_check") or {}).get("judge") or {}
    configured = judge_cfg.get("dimensions")
    if not configured:
        return list(RUBRIC_DIMENSIONS)

    dims: List[str] = []
    seen = set()
    for raw in configured:
        dim = str(raw).strip()
        if dim and dim not in seen:
            seen.add(dim)
            dims.append(dim)

    unknown = [d for d in dims if d not in RUBRIC_DIMENSIONS]
    if unknown:
        raise InvalidJudgeDimension(
            f"quality_check.judge.dimensions names unknown dimension(s) {unknown!r}; "
            f"valid dimensions are {list(RUBRIC_DIMENSIONS)}"
        )

    return dims or list(RUBRIC_DIMENSIONS)


# ---------------------------------------------------------------------------
# Config / filename resolution -- imports the pipeline's own source of
# truth (scripts/briefing_runner.py) lazily, so an unrelated import error
# in that module (it also pulls in feedparser, reportlab, etc.) degrades to
# a reported finding instead of killing the whole check, same principle as
# the sibling-module lazy imports below.
#
# These two rules used to be reimplemented locally here. That was a live
# hazard: if briefing_runner's ${VAR} interpolation or {yyyy}/{mm}/{dd}
# filename pattern ever changed, this checker would keep computing the old
# path, fail to find a briefing that was written perfectly well, and report
# a confident but false "briefing-missing" CRITICAL. Importing the runner's
# own load_config / format_briefing_filename / DEFAULT_FILE_NAMING makes
# that class of bug structurally impossible instead of merely unlikely.
# ---------------------------------------------------------------------------


def pipeline_name_from_path(path: str) -> str:
    """Derive a pipeline name from a config path.

    ``config.yaml`` -> ``atlas``, ``config_local.yaml`` -> ``local``,
    anything else falls back to its filename stem.
    """
    stem = Path(path).stem
    if stem == "config":
        return "atlas"
    if stem.startswith("config_"):
        rest = stem[len("config_"):]
        return rest or stem
    return stem


def locate_briefing_path(config: Dict[str, Any], today: date) -> Path:
    """Where today's rendered markdown for this pipeline should live.

    Delegates filename formatting to ``briefing_runner.format_briefing_filename``
    -- the exact rule that named the file -- rather than a second copy that
    can drift from it.
    """
    from scripts.briefing_runner import DEFAULT_FILE_NAMING, format_briefing_filename

    output_dir = config.get("output_dir", "briefings")
    file_naming = config.get("file_naming", DEFAULT_FILE_NAMING)
    filename = format_briefing_filename(file_naming, today)
    return Path(output_dir) / f"{filename}.md"


def _lazy_import(module_name: str, attr: str) -> Callable[..., Any]:
    """Import ``attr`` from ``module_name`` at call time.

    Used for the two sibling modules (``source_health``, ``report_invariants``)
    that are being written in parallel and may not exist yet -- an ImportError
    here must be catchable by the caller's own try/except, never take down
    the whole run.
    """
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


# ---------------------------------------------------------------------------
# LLM client construction (mirrors briefing_runner.py's chain, read-only,
# never raises -- returns None on any failure so the judge degrades cleanly).
# ---------------------------------------------------------------------------


def build_llm_client(config: Dict[str, Any]) -> Optional[Any]:
    """Build the same opencode -> gemini -> openrouter fallback chain the
    runner builds, for the judge's own use. Never raises; returns None if no
    backend can be constructed or something goes wrong importing one.
    """
    try:
        gemini_config = config.get("gemini", config.get("bedrock", {})) or {}
        openrouter_config = config.get("openrouter", {}) or {}
        opencode_config = config.get("opencode", {}) or {}

        if opencode_config.get("enabled"):
            from scripts.opencode_client import OpencodeClient

            opencode_client = OpencodeClient(opencode_config)
            fallback_clients = []
            if gemini_config.get("enabled"):
                from scripts.gemini_client import GeminiCLIClient

                fallback_clients.append(GeminiCLIClient(gemini_config))
            if openrouter_config.get("enabled"):
                from scripts.openrouter_client import OpenRouterClient

                fallback_clients.append(OpenRouterClient(openrouter_config))
            if fallback_clients:
                from scripts.composite_client import CompositeClient

                timeout = config.get("llm", {}).get(
                    "fallback_timeout_seconds",
                    (config.get("composite", {}) or {}).get("timeout_seconds", 240),
                )
                return CompositeClient([opencode_client] + fallback_clients, timeout=timeout)
            return opencode_client

        if gemini_config.get("enabled"):
            from scripts.gemini_client import GeminiCLIClient

            return GeminiCLIClient(gemini_config)

        return None
    except Exception as e:  # pragma: no cover - defensive, exercised via judge-skipped tests
        logger.warning("Could not build LLM client for quality judge: %s", e)
        return None


# ---------------------------------------------------------------------------
# Layer 3 -- LLM judge
# ---------------------------------------------------------------------------


def _build_judge_prompt(sanitized_markdown: str, domain: str, dimensions: Sequence[str]) -> str:
    """Build the judge prompt for exactly the requested dimensions.

    Only the configured dimensions are asked for -- the model is never
    asked to score something the pipeline's own config says doesn't apply,
    and never asked for more than what actually counts toward the total.
    """
    count_word = "dimension" if len(dimensions) == 1 else "dimensions"
    rubric_lines = "\n".join(
        f"- {dim}: {_DIMENSION_RUBRIC[dim]}" for dim in dimensions
    )
    shape = ", ".join(f'"{dim}": {{"score": 0, "why": "..."}}' for dim in dimensions)
    return (
        f"Score this rendered daily {domain} briefing on the following "
        f"{len(dimensions)} {count_word}, each on a 0-2 integer scale, with "
        "a short one-line justification per dimension:\n\n"
        f"{rubric_lines}\n\n"
        "Respond with ONLY strict JSON, no prose, no markdown code fences, "
        "in exactly this shape:\n"
        f"{{{shape}}}"
        "\n\nBRIEFING:\n<<<\n" + sanitized_markdown + "\n>>>"
    )


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> Optional[str]:
    """Pull a JSON object out of text that may wrap it in prose or fences."""
    if not text:
        return None
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def parse_judge_response(
    text: Optional[str], dimensions: Sequence[str] = RUBRIC_DIMENSIONS
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Defensively parse the judge's response into per-dimension scores.

    Validates exactly ``dimensions`` -- no more, no less. Every dimension in
    ``dimensions`` must be present with a valid 0-2 score or the whole parse
    fails (returns None). A dimension the model volunteers beyond what was
    asked for is ignored rather than treated as a parse failure or included
    in the result -- we didn't ask for it, so it doesn't count toward the
    total either.

    Returns None on any failure to find/parse JSON or if a required
    dimension/score is missing or out of range -- callers treat None as
    "degrade to judge-skipped", never raise.
    """
    if not text:
        return None
    candidate = _extract_json_object(text)
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    parsed: Dict[str, Dict[str, Any]] = {}
    for dim in dimensions:
        entry = data.get(dim)
        if isinstance(entry, dict):
            score = entry.get("score")
            why = entry.get("why") or entry.get("justification") or entry.get("reason") or ""
        else:
            score = entry
            why = ""
        try:
            score = int(score)
        except (TypeError, ValueError):
            return None
        if score not in (0, 1, 2):
            return None
        parsed[dim] = {"score": score, "why": str(why)[:300]}
    return parsed


def judge_briefing(
    markdown: str,
    config: Dict[str, Any],
    pipeline: str,
    today: date,
    client: Any,
) -> Tuple[List[Finding], Optional[Dict[str, Any]]]:
    """Run the Layer 3 rubric judge on one pipeline's rendered briefing.

    Returns (findings, score_record). ``score_record`` is None whenever the
    dimension config is invalid, the call failed, or the response couldn't
    be parsed -- in each of those cases findings contains exactly one
    finding and no record is produced or persisted.
    """
    from scripts.intelligence import _sanitize_prompt_input

    try:
        dimensions = resolve_judge_dimensions(config)
    except InvalidJudgeDimension as e:
        return [Finding(WARN, "judge-config-invalid", str(e), pipeline=pipeline)], None

    profile = config.get("briefing_profile", {}) or {}
    domain = profile.get("domain", "AI and technology")

    sanitized = _sanitize_prompt_input(markdown, max_length=12000)
    prompt = _build_judge_prompt(sanitized, domain, dimensions)

    try:
        raw = client.invoke(prompt, tier="medium", system_prompt=_JUDGE_SYSTEM_PROMPT)
    except Exception as e:
        logger.warning("Quality judge invoke failed for pipeline '%s': %s", pipeline, e)
        raw = None

    if not raw:
        return (
            [Finding(INFO, "judge-skipped", "LLM judge call failed or returned nothing", pipeline=pipeline)],
            None,
        )

    parsed = parse_judge_response(raw, dimensions)
    if parsed is None:
        return (
            [Finding(INFO, "judge-skipped", "Could not parse judge output as valid JSON", pipeline=pipeline)],
            None,
        )

    # Denominator reflects only the dimensions actually scored for this
    # pipeline -- 8/12 vs 8/8 is the difference between a meaningful trend
    # and a meaningless one, so a briefing scored on a narrower rubric never
    # reports against the full six-dimension ceiling it was never asked
    # about.
    total = sum(v["score"] for v in parsed.values())
    max_total = len(dimensions) * 2
    record = {
        "ts": datetime.now().astimezone().isoformat(),
        "pipeline": pipeline,
        "date": today.isoformat(),
        "scores": parsed,
        "total": total,
        "max_total": max_total,
        "notes": "",
    }
    return [], record


def append_score_record(record: Dict[str, Any], path: str = DEFAULT_SCORES_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_score_history(pipeline: str, path: str = DEFAULT_SCORES_PATH) -> List[Dict[str, Any]]:
    """All recorded judge scores for one pipeline, oldest first."""
    p = Path(path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("pipeline") == pipeline:
            records.append(rec)
    records.sort(key=lambda r: (r.get("date") or "", r.get("ts") or ""))
    return records


def _dim_score(record: Dict[str, Any], dim: str) -> Optional[float]:
    entry = (record.get("scores") or {}).get(dim)
    val = entry.get("score") if isinstance(entry, dict) else entry
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def detect_quality_regression(
    history: List[Dict[str, Any]], pipeline: str, threshold: float = 1.0
) -> List[Finding]:
    """Alert only on a sustained drop: mean of the last 3 runs is at least
    ``threshold`` lower than the mean of the 10 runs before that.

    A single low score among an otherwise-normal history will not move a
    3-run mean by a full point against a 10-run baseline, so this stays
    silent on noise by construction. Needs at least 13 runs of history.
    """
    if len(history) < 13:
        return []

    recent = history[-3:]
    baseline = history[-13:-3]

    findings: List[Finding] = []
    for dim in RUBRIC_DIMENSIONS:
        recent_scores = [s for s in (_dim_score(r, dim) for r in recent) if s is not None]
        baseline_scores = [s for s in (_dim_score(r, dim) for r in baseline) if s is not None]
        if len(recent_scores) < 3 or len(baseline_scores) < 10:
            continue
        recent_mean = sum(recent_scores) / len(recent_scores)
        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        drop = baseline_mean - recent_mean
        if drop >= threshold:
            findings.append(
                Finding(
                    WARN,
                    "quality-regression",
                    f"{dim} dropped {drop:.1f} pts over the last 3 runs "
                    f"(baseline {baseline_mean:.1f} -> recent {recent_mean:.1f})",
                    source=dim,
                    pipeline=pipeline,
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_checks(
    configs: Dict[str, Dict[str, Any]],
    *,
    since: str = "-2d",
    deep: bool = False,
    no_judge: bool = False,
    today: Optional[date] = None,
    dry_run: bool = False,
    history_path: str = DEFAULT_HISTORY_PATH,
    scores_path: str = DEFAULT_SCORES_PATH,
    journal_timeout: int = 180,
    probe_timeout: int = 20,
    harvest_journal: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    append_history: Optional[Callable[..., int]] = None,
    load_history: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    probe_feeds: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    detect_rot: Optional[Callable[..., List[Finding]]] = None,
    check_report: Optional[Callable[..., List[Finding]]] = None,
    build_client: Optional[Callable[[Dict[str, Any]], Any]] = None,
    judge_fn: Optional[Callable[..., Tuple[List[Finding], Optional[Dict[str, Any]]]]] = None,
    score_history_loader: Optional[Callable[..., List[Dict[str, Any]]]] = None,
) -> Tuple[List[Finding], Dict[str, Dict[str, Any]]]:
    """Run all three layers and return (sorted findings, judge score records).

    Every layer callable is injectable for testing; when omitted the real
    implementation is imported lazily so an ImportError in one sibling
    module can't take the others down. Nothing here touches disk when
    ``dry_run`` is True except reading what's already there.
    """
    today = today or date.today()
    findings: List[Finding] = []

    # ---- Layer 1: source health --------------------------------------
    try:
        _harvest = harvest_journal or _lazy_import("scripts.source_health", "harvest_journal")
        _append_hist = append_history or _lazy_import("scripts.source_health", "append_history")
        _load_hist = load_history or _lazy_import("scripts.source_health", "load_history")
        _detect_rot = detect_rot or _lazy_import("scripts.source_health", "detect_rot")

        records = _harvest(since=since, timeout=journal_timeout)

        if dry_run:
            history = records
        else:
            _append_hist(records, path=history_path)
            history = _load_hist(path=history_path)

        probes = None
        if deep:
            _probe = probe_feeds or _lazy_import("scripts.source_health", "probe_feeds")
            probes = []
            for pipeline, config in configs.items():
                try:
                    probes.extend(_probe(config.get("blog_feeds", []) or [], timeout=probe_timeout))
                except Exception as e:
                    findings.append(
                        Finding(WARN, "feed-probe-failed", str(e), source="probe_feeds", pipeline=pipeline)
                    )

        findings.extend(_detect_rot(history, probes=probes))
    except Exception as e:
        findings.append(Finding(WARN, "source-health-unavailable", f"Layer 1 (source health) failed: {e}", source="layer1"))

    # ---- Layer 2: report invariants ------------------------------------
    markdowns: Dict[str, Optional[str]] = {}
    _check_report = None
    try:
        _check_report = check_report or _lazy_import("scripts.report_invariants", "check_report")
    except Exception as e:
        findings.append(Finding(WARN, "report-invariants-unavailable", f"Layer 2 (report invariants) unavailable: {e}", source="layer2"))

    for pipeline, config in configs.items():
        try:
            path = locate_briefing_path(config, today)
        except Exception as e:
            findings.append(
                Finding(
                    WARN,
                    "briefing-runner-unavailable",
                    f"Could not resolve briefing path for pipeline '{pipeline}': {e}",
                    source="layer2",
                    pipeline=pipeline,
                )
            )
            markdowns[pipeline] = None
            continue
        if not path.exists():
            findings.append(
                Finding(
                    CRITICAL,
                    "briefing-missing",
                    f"No briefing found for pipeline '{pipeline}' at {path}",
                    source=str(path),
                    pipeline=pipeline,
                )
            )
            markdowns[pipeline] = None
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            findings.append(
                Finding(
                    CRITICAL,
                    "briefing-missing",
                    f"Could not read briefing for pipeline '{pipeline}': {e}",
                    source=str(path),
                    pipeline=pipeline,
                )
            )
            markdowns[pipeline] = None
            continue

        markdowns[pipeline] = text
        if _check_report is not None:
            try:
                findings.extend(_check_report(text, config, today=today, pipeline=pipeline))
            except Exception as e:
                findings.append(
                    Finding(WARN, "report-invariants-unavailable", f"check_report failed for '{pipeline}': {e}", pipeline=pipeline)
                )

    # ---- Layer 3: LLM judge --------------------------------------------
    judge_records: Dict[str, Dict[str, Any]] = {}
    if not no_judge:
        _build_client = build_client or build_llm_client
        _judge = judge_fn or judge_briefing
        _load_score_hist = score_history_loader or load_score_history

        client = None
        try:
            for config in configs.values():
                candidate = _build_client(config)
                if candidate is not None and getattr(candidate, "available", False):
                    client = candidate
                    break
        except Exception as e:
            logger.warning("Building the LLM client for the quality judge raised: %s", e)
            client = None

        if client is None:
            findings.append(Finding(INFO, "judge-skipped", "No LLM client available for the quality judge", source="layer3"))
        else:
            for pipeline, config in configs.items():
                text = markdowns.get(pipeline)
                if not text:
                    continue
                try:
                    jf, record = _judge(text, config, pipeline, today, client)
                except Exception as e:
                    jf, record = (
                        [Finding(INFO, "judge-skipped", f"Judge raised an exception: {e}", pipeline=pipeline)],
                        None,
                    )
                findings.extend(jf)
                if record is not None:
                    judge_records[pipeline] = record
                    try:
                        prior = _load_score_hist(pipeline=pipeline, path=scores_path)
                    except Exception:
                        prior = []
                    findings.extend(detect_quality_regression(prior + [record], pipeline))
                    if not dry_run:
                        try:
                            append_score_record(record, path=scores_path)
                        except Exception as e:
                            logger.warning("Could not append judge score record: %s", e)

    return sort_findings(findings), judge_records


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


def render_digest(findings: List[Finding], judge_records: Dict[str, Dict[str, Any]], today: date) -> str:
    """A phone-skimmable markdown digest: headline, one findings table, then
    per-pipeline judge scores. No walls of text.
    """
    findings = sort_findings(findings)
    counts = counts_by_severity(findings)
    worst = worst_severity(findings) or "OK"

    lines = [f"# Quality Digest -- {today.isoformat()}", ""]
    headline = f"**{worst}** -- " + ", ".join(f"{counts[s]} {s}" for s in SEVERITY_ORDER)
    lines.append(headline)
    lines.append("")

    if findings:
        lines.append("| Severity | Pipeline | Code | Source | Message |")
        lines.append("|---|---|---|---|---|")
        for f in findings:
            msg = (f.message or "").replace("|", "\\|").replace("\n", " ")
            src = (f.source or "-").replace("|", "\\|")
            lines.append(f"| {f.severity} | {f.pipeline or '-'} | {f.code} | {src} | {msg} |")
        lines.append("")
    else:
        lines.append("No findings.")
        lines.append("")

    if judge_records:
        lines.append("## Judge scores")
        lines.append("")
        for pipeline in sorted(judge_records):
            record = judge_records[pipeline]
            scores = record.get("scores") or {}
            # Denominator reflects the dimensions this pipeline was actually
            # scored on (config-driven, e.g. Atlas skips tier_1_share/
            # locality) -- max_total is written by judge_briefing; fall
            # back to len(scores)*2 for any older record written before
            # that field existed, and 12 only if even that's unavailable.
            max_total = record.get("max_total")
            if max_total is None:
                max_total = len(scores) * 2 or 12
            lines.append(f"### {pipeline} -- total {record.get('total', '-')}/{max_total}")
            lines.append("")
            lines.append("| Dimension | Score | Note |")
            lines.append("|---|---|---|")
            # Only the dimensions actually scored -- a dimension this
            # pipeline isn't configured for has no row at all, rather than
            # a blank "-" implying it was considered and found wanting.
            for dim in RUBRIC_DIMENSIONS:
                if dim not in scores:
                    continue
                entry = scores.get(dim, {})
                score = entry.get("score", "-") if isinstance(entry, dict) else entry
                why = entry.get("why", "") if isinstance(entry, dict) else ""
                why = str(why).replace("|", "\\|")
                lines.append(f"| {dim} | {score} | {why} |")
            lines.append("")

    return "\n".join(lines)


def digest_path_for(today: date) -> str:
    return f"logs/quality-digest-{today.isoformat()}.md"


def write_digest(content: str, today: date, path: Optional[str] = None) -> Path:
    p = Path(path) if path else Path(digest_path_for(today))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Alert routing
# ---------------------------------------------------------------------------

_SEVERITY_LABEL: Dict[str, str] = {CRITICAL: "critical", WARN: "warning", INFO: "info"}

_EMAIL_RE = re.compile(r"^[^@\s,]+@[^@\s,]+\.[^@\s,]+$")


def _looks_like_email(addr: str) -> bool:
    """Cheap sanity filter, not full RFC validation -- just enough to drop
    an unresolved ``${VAR}`` placeholder or a stray blank rather than hand
    it to SMTP as a destination."""
    return bool(_EMAIL_RE.match(addr))


def build_alert_subject(findings: List[Finding], today: date) -> str:
    """Subject line for the alert email.

    Leads with severity and counts so it is triageable from a phone lock
    screen, e.g. "[CRITICAL] Briefing quality: 3 critical, 1 warning --
    2026-08-25". A run with nothing to report (the daily_digest opt-in on a
    clean day) gets an OK-flavored subject instead, e.g. "[OK] Briefing
    quality: no findings -- 2026-08-25".
    """
    worst = worst_severity(findings) or "OK"
    counts = counts_by_severity(findings)
    parts = [f"{counts[s]} {_SEVERITY_LABEL[s]}" for s in SEVERITY_ORDER if counts.get(s)]
    summary = ", ".join(parts) if parts else "no findings"
    return f"[{worst}] Briefing quality: {summary} — {today.isoformat()}"


def resolve_alert_recipients(config: Dict[str, Any]) -> List[str]:
    """Resolve alert-email recipients for one pipeline config.

    Fallback chain:
      1. ``config["quality_check"]["alert_email"]["recipients"]``
      2. ``config["email_recipients"]`` -- the pipeline's own delivery list

    Either may be a single comma-separated string, a list of addresses, or
    a list mixing comma-separated strings with plain addresses -- the same
    shape ``email_distributor.distribute()`` normalizes around line 436.
    The result is deduped (first-seen order preserved) and filtered to
    entries that look like an address, so an unresolved ``${VAR:-default}``
    placeholder or a stray blank never becomes a send target.
    """
    qc_cfg = (config.get("quality_check") or {}).get("alert_email") or {}
    raw = qc_cfg.get("recipients")
    if not raw:
        raw = config.get("email_recipients")
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]

    seen = set()
    out: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        for piece in item.split(","):
            addr = piece.strip()
            if addr and _looks_like_email(addr) and addr not in seen:
                seen.add(addr)
                out.append(addr)
    return out


def _all_alert_recipients(configs: Dict[str, Dict[str, Any]]) -> List[str]:
    """Union of resolve_alert_recipients() across every pipeline config,
    deduped -- one shared digest email covers every pipeline in the run."""
    seen = set()
    out: List[str] = []
    for cfg in configs.values():
        for addr in resolve_alert_recipients(cfg):
            if addr not in seen:
                seen.add(addr)
                out.append(addr)
    return out


def _daily_digest_enabled(configs: Dict[str, Dict[str, Any]]) -> bool:
    """True if any pipeline opts into ``quality_check.alert_email.daily_digest``
    -- send the digest every run regardless of severity, not just on CRITICAL."""
    return any(
        bool(((cfg.get("quality_check") or {}).get("alert_email") or {}).get("daily_digest"))
        for cfg in configs.values()
    )


def send_alert_email(subject: str, markdown_body: str, recipients: List[str]) -> bool:
    """Send the quality digest by email, reusing the exact delivery path
    the morning briefing itself uses (``scripts.email_distributor.EmailDistributor``)
    so there is one delivery mechanism to keep working, not two.

    Graceful degradation is a hard requirement here, same as the pipeline
    itself: no recipients, missing GMAIL_USER/GMAIL_APP_PASSWORD, an import
    error, or an SMTP failure must all log and return False -- never raise.
    A quality checker that crashes because it could not report is worse
    than one that stays quiet.
    """
    if not recipients:
        logger.warning("No alert email recipients configured; digest not sent: %s", subject)
        return False

    sender_email = os.environ.get("GMAIL_USER")
    sender_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender_email or not sender_password:
        logger.warning("GMAIL_USER/GMAIL_APP_PASSWORD not set; alert email not sent: %s", subject)
        return False

    try:
        from scripts.email_distributor import EmailDistributor

        distributor = EmailDistributor(sender_email=sender_email, sender_password=sender_password)
        results = distributor.send_html_email(
            recipients=recipients, markdown_content=markdown_body, subject=subject
        )
        return bool(results) and all(results.values())
    except Exception as e:
        logger.error("Quality alert email failed: %s", e)
        return False


def load_alert_state(path: str = DEFAULT_ALERTS_PATH) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_alert_state(state: Dict[str, Any], path: str = DEFAULT_ALERTS_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def route_alerts(
    findings: List[Finding],
    *,
    digest_markdown: str,
    configs: Dict[str, Dict[str, Any]],
    today: date,
    state: Optional[Dict[str, Any]] = None,
    path: str = DEFAULT_ALERTS_PATH,
    send_fn: Optional[Callable[[str, str, List[str]], bool]] = None,
    now: Optional[datetime] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Route findings to a single alert email, deduped within a 24h window.

    A CRITICAL finding pages (subject to the existing 24h dedupe keyed on
    ``Finding.dedupe_key`` -- a dead feed pages once, then becomes a
    standing digest item). WARN/INFO never page on their own. When any
    pipeline config sets ``quality_check.alert_email.daily_digest``, the
    digest is sent every run regardless of severity, with an OK-flavored
    subject on a clean run.

    At most one email is sent per call: if several CRITICALs are present,
    or the critical path and daily_digest would both fire, they share the
    single send -- one email per run, not one per finding.

    In dry-run mode this computes nothing new to persist: it neither sends
    nor writes the alert-state file.
    """
    now = now or datetime.now().astimezone()
    state = dict(state) if state is not None else load_alert_state(path)
    _send = send_fn or send_alert_email
    changed = False
    should_send = False

    for f in findings:
        if f.severity != CRITICAL:
            continue
        key = f.dedupe_key
        entry = state.get(key)

        should_push = True
        if entry and entry.get("last_alerted"):
            try:
                last_dt = datetime.fromisoformat(entry["last_alerted"])
                if (now - last_dt) < timedelta(hours=24):
                    should_push = False
            except ValueError:
                pass

        if dry_run:
            continue

        first_seen = entry.get("first_seen") if entry else now.isoformat()
        count = (entry.get("count", 0) if entry else 0) + 1
        last_alerted = entry.get("last_alerted") if entry else None

        if should_push:
            should_send = True
            last_alerted = now.isoformat()

        state[key] = {"first_seen": first_seen, "last_alerted": last_alerted, "count": count}
        changed = True

    if dry_run:
        return state

    if not should_send and _daily_digest_enabled(configs):
        should_send = True

    if should_send:
        subject = build_alert_subject(findings, today)
        recipients = _all_alert_recipients(configs)
        _send(subject, digest_markdown, recipients)

    if changed:
        save_alert_state(state, path)

    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily quality check for the morning briefing pipelines.")
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        default=None,
        help="Pipeline config file (repeatable). Default: config.yaml and config_local.yaml",
    )
    parser.add_argument("--since", default="-2d", help="journald window for the source-health harvest")
    parser.add_argument("--deep", action="store_true", help="also live-probe blog_feeds (weekly run)")
    parser.add_argument("--no-judge", action="store_true", help="skip the Layer 3 LLM judge")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write nothing, notify nothing")
    parser.add_argument("--date", default=None, help="override 'today' as YYYY-MM-DD, for replaying a past day")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    today = date.today()
    if args.date:
        try:
            today = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error("Invalid --date '%s', expected YYYY-MM-DD", args.date)
            return 2

    config_paths = args.configs or ["config.yaml", "config_local.yaml"]

    # Uses briefing_runner's own load_config -- the source of truth for
    # ${VAR:-default} interpolation -- imported lazily so a broken runner
    # import degrades to a clean exit-2 rather than an unrelated traceback.
    #
    # Deliberate choice: load_config() itself calls sys.exit(2) on a missing
    # or malformed config file. We keep that as the effective outcome (a
    # config the checker can't even parse is an operational failure of the
    # monitoring tool, not a "reader got a broken briefing" CRITICAL, so it
    # must not send an alert email) but catch the SystemExit here so main()
    # keeps its "always returns an int" contract instead of hard-killing the
    # interpreter -- needed for tests and any programmatic caller.
    configs: Dict[str, Dict[str, Any]] = {}
    try:
        from scripts.briefing_runner import load_config as _load_runner_config

        for cp in config_paths:
            pipeline = pipeline_name_from_path(cp)
            configs[pipeline] = _load_runner_config(cp)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 2
        logger.error("Failed to load pipeline configs (runner load_config exited with code %s)", code)
        return code or 2
    except Exception as e:
        logger.error("Failed to load pipeline configs: %s", e)
        return 2

    try:
        findings, judge_records = run_checks(
            configs,
            since=args.since,
            deep=args.deep,
            no_judge=args.no_judge,
            today=today,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error("Quality check failed: %s", e)
        return 2

    try:
        digest = render_digest(findings, judge_records, today)
        print(digest)
        if not args.dry_run:
            write_digest(digest, today)
        route_alerts(
            findings,
            digest_markdown=digest,
            configs=configs,
            today=today,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error("Digest rendering / alert routing failed: %s", e)
        return 2

    return 1 if worst_severity(findings) == CRITICAL else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
