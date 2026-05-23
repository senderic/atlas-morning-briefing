#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Diagnostic: report what's accumulated in .atlas-state.json for the
Saturday weekly deep dive ("This Week in AI" section).

Use case: it's Friday night, you want to know if tomorrow's report
will have substance. The runner only emits the section if (a) it's
Saturday, (b) intelligence is available, AND (c) weekly_items is
non-empty. This script tells you what's in (c).

Usage:
    uv run scripts/check_weekly_state.py
    uv run scripts/check_weekly_state.py --state-path /custom/path/state.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path(".atlas-state.json"),
        help="Path to atlas state file (default: ./.atlas-state.json)",
    )
    args = parser.parse_args()

    if not args.state_path.exists():
        print(f"FAIL  State file not found: {args.state_path}")
        print()
        print("The runner writes this on every successful briefing. If "
              "it's missing, the cron either hasn't run yet OR runs from "
              "a different working directory. Try:")
        print(f"  find ~ -name '.atlas-state.json' -not -path '*/\\.*/' 2>/dev/null")
        return 2

    try:
        state = json.loads(args.state_path.read_text())
    except json.JSONDecodeError as e:
        print(f"FAIL  State file is corrupt JSON: {e}")
        return 2

    weekly_items = state.get("weekly_items", [])

    print("=" * 60)
    print(f"Atlas weekly-state diagnostic")
    print(f"State file: {args.state_path}")
    print(f"Today:      {date.today()} (weekday {date.today().weekday()}, "
          f"{'SATURDAY — report fires today!' if date.today().weekday() == 5 else 'not Saturday'})")
    print("=" * 60)

    if not weekly_items:
        print()
        print("EMPTY  weekly_items is []")
        print()
        print("Tomorrow's 'This Week in AI' section will NOT render (the")
        print("runner skips it when the list is empty). Either the cron")
        print("hasn't run this week, OR a previous Saturday cleared the")
        print("list and no weekday runs have happened since.")
        return 1

    by_date = defaultdict(list)
    by_type = Counter()
    for item in weekly_items:
        d = item.get("date", "unknown")
        t = item.get("type", "unknown")
        by_date[d].append(item)
        by_type[t] += 1

    print()
    print(f"Total weekly_items: {len(weekly_items)}")
    print(f"Breakdown by type:  {dict(by_type)}")
    print()
    print(f"Items per day (last 7 days):")
    today = date.today()
    for i in range(7, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        items = by_date.get(d, [])
        marker = "  " if items else "✗ "
        print(f"  {marker}{d} ({(today - timedelta(days=i)).strftime('%a')}): "
              f"{len(items)} items")

    extra_dates = sorted(d for d in by_date if d != "unknown"
                         and d < (today - timedelta(days=7)).isoformat())
    if extra_dates:
        print()
        print(f"WARN  Found {len(extra_dates)} item(s) older than 7 days "
              f"({extra_dates[0]} ... {extra_dates[-1]}).")
        print("      The runner doesn't auto-prune. They'll be included")
        print("      in the deep dive prompt, which may dilute the focus.")

    print()
    print("Sample items (first 5):")
    for item in weekly_items[:5]:
        title = item.get("title", "")[:75]
        print(f"  - [{item.get('date', '?')}] [{item.get('type', '?'):4}] {title}")

    print()
    if len(weekly_items) < 6:
        print(f"THIN   {len(weekly_items)} items is sparse. The Heavy-tier")
        print(f"       prompt asks for 3 themes + analysis + predictions — ")
        print(f"       with this few items the LLM will be stretched.")
        return 1
    elif len(weekly_items) < 15:
        print(f"OK     {len(weekly_items)} items is decent (3-5 items/day for")
        print(f"       a few weekdays). Section should render with substance.")
    else:
        print(f"RICH   {len(weekly_items)} items — plenty of material for a")
        print(f"       strong 'This Week in AI' synthesis.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
