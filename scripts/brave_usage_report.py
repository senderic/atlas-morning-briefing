#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Survey Brave Search API usage and search terms.

Primary source is systemd journald: every Brave news/search request logs a
DEBUG "GET /res/v1/news/search?q=..." line from the atlas-briefing pipeline.
Optionally hits the Brave API once to read the X-RateLimit-* headers for
current rolling-month usage (--live-check).

Report is markdown on stdout or written to --output.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote_plus

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/news/search"

REQUEST_RE = re.compile(
    r'^(?P<ts>\S+)\s+\S+\s+(?P<ident>[a-z-]+)\[\d+\]:\s+DEBUG: '
    r'https://api\.search\.brave\.com:443 "GET /res/v1/news/search\?'
    r'q=(?P<q>[^&"]+)&count=\d+&freshness=\w+ HTTP/1\.1" (?P<status>\d+)'
)


def fetch_journal(since: str, until: str) -> list[dict]:
    cmd = [
        "journalctl",
        "-o", "short-iso",
        "--since", since,
        "--until", until,
        "--no-pager",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("journalctl failed: %s", proc.stderr.strip())
        return []

    entries = []
    for line in proc.stdout.splitlines():
        if "GET /res/v1/news/search" not in line:
            continue
        match = REQUEST_RE.match(line)
        if not match:
            logger.debug("unparsed line: %s", line[:200])
            continue
        entries.append(
            {
                "ts": match.group("ts"),
                "ident": match.group("ident"),
                "query": unquote_plus(match.group("q")),
                "status": int(match.group("status")),
            }
        )
    return entries


def load_base_queries() -> dict:
    base = {"main": set(), "local": set()}
    for path, key in (("config.yaml", "main"), ("config_local.yaml", "local")):
        if Path(path).exists():
            cfg = yaml.safe_load(Path(path).read_text())
            base[key] = set(cfg.get("news_queries", []))
    return base


def classify(entry: dict, base: dict) -> tuple[str, str]:
    source = "main" if entry["ident"] == "atlas-briefing" else "local"
    if entry["ident"] not in ("atlas-briefing", "local-briefing"):
        if entry["query"] in base["main"]:
            source = "main"
        elif entry["query"] in base["local"]:
            source = "local"
        else:
            source = "unknown"

    origin = "base"
    if entry["query"] not in base["main"] | base["local"]:
        origin = "dynamic"
    return source, origin


def live_check() -> dict:
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return {"error": "BRAVE_API_KEY not set"}
    try:
        resp = requests.get(
            BRAVE_SEARCH_URL,
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            params={"q": "technology news", "count": 1, "freshness": "pm"},
            timeout=30,
        )
    except requests.RequestException as e:
        return {"error": f"request failed: {e}"}

    headers = resp.headers
    return {
        "status": resp.status_code,
        "limit": headers.get("X-RateLimit-Limit"),
        "remaining": headers.get("X-RateLimit-Remaining"),
        "reset": headers.get("X-RateLimit-Reset"),
    }


def build_report(entries: list[dict], since: str, until: str, live: dict | None) -> str:
    total = len(entries)
    ok = sum(1 for e in entries if e["status"] == 200)
    by_day = defaultdict(list)
    queries = Counter()
    source_count = Counter()
    origin_count = Counter()

    for e in entries:
        day = e["ts"][:10]
        by_day[day].append(e)
        queries[e["query"]] += 1

    base = load_base_queries()
    for e in entries:
        source, origin = classify(e, base)
        source_count[source] += 1
        origin_count[origin] += 1

    lines = [
        f"# Brave Search API Usage Report",
        "",
        f"**Range:** {since} → {until}",
        f"**Total requests:** {total}  (HTTP 200: {ok}, failed: {total - ok})",
        f"**Distinct queries:** {len(queries)}",
        "",
        "## Per-day requests",
        "",
        "| Date | Requests | Distinct queries |",
        "|------|----------|------------------|",
    ]
    for day in sorted(by_day):
        day_queries = {e["query"] for e in by_day[day]}
        lines.append(f"| {day} | {len(by_day[day])} | {len(day_queries)} |")

    lines += [
        "",
        "## Query frequency",
        "",
        "| # | Query | Count |",
        "|---|-------|-------|",
    ]
    for i, (query, count) in enumerate(queries.most_common(), 1):
        lines.append(f"| {i} | {query} | {count} |")

    lines += [
        "",
        "## Source split",
        "",
        f"- **Main briefing** (atlas-briefing): {source_count.get('main', 0)}",
        f"- **Local briefing** (local-briefing): {source_count.get('local', 0)}",
        f"- **Unknown source**: {source_count.get('unknown', 0)}",
        "",
        "## Query origin",
        "",
        f"- **Base config queries** (config.yaml + config_local.yaml): {origin_count.get('base', 0)}",
        f"- **LLM-dynamic queries**: {origin_count.get('dynamic', 0)}",
        "",
        "## Caveats",
        "",
        "- Counts reflect only requests captured by journald within the range;",
        "  earlier runs before journald persistence (or after rotation) are missing.",
        "- Each `news/search` call counts as one Brave API request regardless of",
        "  results returned.",
    ]

    if live:
        lines += ["", "## Live rate-limit check", ""]
        if "error" in live:
            lines.append(f"- **Error:** {live['error']}")
        else:
            burst_limit, monthly_limit = parse_rate_pair(live.get("limit")) or (None, None)
            burst_rem, monthly_rem = parse_rate_pair(live.get("remaining")) or (None, None)
            reset_secs = parse_rate_pair(live.get("reset")) or (None, None)
            reset_at = datetime.now() + timedelta(seconds=reset_secs[1]) if reset_secs and reset_secs[1] else None
            lines.append(f"- **Status:** HTTP {live.get('status')}")
            lines.append(
                f"- **Raw headers:** `X-RateLimit-Limit: {live.get('limit')}`, "
                f"`X-RateLimit-Remaining: {live.get('remaining')}`, "
                f"`X-RateLimit-Reset: {live.get('reset')}`"
            )
            if burst_limit is not None:
                lines.append(f"- **Burst limit:** {burst_limit}/sec, **remaining:** {burst_rem}/sec")
            if monthly_limit:
                used = monthly_limit - (monthly_rem or 0)
                lines.append(f"- **Monthly limit:** {monthly_limit}, **remaining:** {monthly_rem}")
                lines.append(f"- **Monthly used (est.):** {used}")
            else:
                lines.append(
                    "- **Monthly quota:** not exposed by the API for this account "
                    "(header value is `0`), so per-month usage cannot be read this way."
                )
            if reset_at:
                lines.append(f"- **Monthly window resets:** {reset_at.isoformat(timespec='minutes')}")
            lines.append(
                "- **Note:** the monthly window is rolling (~31 days), not calendar-month,"
                " so it reflects current usage, not historical August."
            )

    return "\n".join(lines) + "\n"


def parse_rate_pair(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        parts = [int(p.strip()) for p in value.split(",")]
    except ValueError:
        return None
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Survey Brave Search API usage")
    parser.add_argument("--since", default=None, help="Start date (e.g. 2026-08-01), default: 1st of current month")
    parser.add_argument("--until", default=None, help="End date (e.g. 2026-08-16), default: now")
    parser.add_argument("--output", type=Path, default=None, help="Write markdown report to this file")
    parser.add_argument("--live-check", action="store_true", help="Fire one Brave request and read rate-limit headers")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    now = datetime.now()
    since = args.since or f"{now.year:04d}-{now.month:02d}-01"
    until = args.until or now.strftime("%Y-%m-%d")

    entries = fetch_journal(since, until)
    logger.info("parsed %d brave requests in range", len(entries))

    live = live_check() if args.live_check else None
    if live and "error" not in live:
        logger.info("live check: monthly remaining %s", parse_rate_pair(live.get("remaining"))[1] if parse_rate_pair(live.get("remaining")) else "n/a")

    report = build_report(entries, since, until, live)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        logger.info("wrote report to %s", args.output)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())