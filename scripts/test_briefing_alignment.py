#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Alignment test: run the briefing pipeline and verify the output
structure matches Atlas-Briefing-2026.05.16-main.md (the reference report
saved at the repo root).

This exercises the actual code paths used by cron — Gemini CLI auto-detect,
data fetch, LLM enrichment, rendering — but skips email delivery, so it's
safe to run on demand. Useful as a sanity check after touching anything in
scripts/.

Usage
-----

    # Real run — exercises gemini-cli + APIs end-to-end. Costs ~$0.02 in
    # Gemini credit and takes 2-5 minutes depending on network/LLM latency.
    # This is the right mode for "did my changes break anything" checks.
    uv run scripts/test_briefing_alignment.py

    # Fast structural smoke test — uses synthetic data, makes zero API
    # calls, finishes in under a second. Catches markdown-template
    # regressions but won't catch issues in the LLM-enrichment path.
    uv run scripts/test_briefing_alignment.py --mock

Outputs
-------

    logs/alignment-{ts}.log          — full DEBUG log of the run
    logs/alignment-report-{ts}.txt   — pass/fail breakdown by section
    Atlas-Briefing-{date}*.md        — the generated briefing (live mode)

Exit codes: 0 = all required sections present, 1 = some missing,
2 = generation itself failed.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REFERENCE_BRIEFING = REPO_ROOT / "Atlas-Briefing-2026.05.16-main.md"


# Patterns must appear somewhere in the generated markdown. These mirror
# the exact section headers and structural elements visible in
# Atlas-Briefing-2026.05.16-main.md so if the reference structure ever
# changes intentionally, update this list.
#
# Items tagged "live-only" are demoted to OPTIONAL in --mock mode since
# they depend on real Gemini calls happening (the Usage Summary section
# is correctly suppressed when no calls were made).
REQUIRED_PATTERNS: List[Tuple[str, str, bool]] = [
    # (label, regex, live_only)
    ("Title heading",                 r"^# Atlas Morning Briefing\s*$",                              False),
    ("Date/time header line",         r"^\*[A-Z][a-z]+, [A-Z][a-z]+ \d{1,2}, \d{4} \| \d{1,2}:\d{2} (AM|PM)", False),
    ("Executive Summary heading",     r"^## Executive Summary\s*$",                                  False),
    ("Financial Market Overview",     r"^## Financial Market Overview\s*$",                          False),
    ("Stock table header row",        r"^\| Ticker \| Price \| Change \| Driver \|",                 False),
    ("AI & Tech News heading",        r"^## AI & Tech News\s*$",                                     False),
    ("Top Papers heading",            r"^## Top Papers\s*$",                                         False),
    ("Star rating present",           r"[★☆]{5}",                                                    False),
    ("Repro score present",           r"\*\*Repro:.*\d+/\d+",                                        False),
    ("Blog Updates heading",          r"^## Blog Updates\s*$",                                       False),
    ("Gemini Usage Summary heading",  r"^## Gemini Usage Summary\s*$",                               True),
    ("Usage summary table header",    r"^\| Tier \| Success \| Failures \|",                         True),
]

# Optional patterns: we report on them but don't fail the test.
OPTIONAL_PATTERNS: List[Tuple[str, str]] = [
    ("This Week in AI (Saturday-only)",        r"^## This Week in AI\s*$"),
    ("User-name suffix [RIGHT]for X[/RIGHT]",  r"\[RIGHT\]for .+\[/RIGHT\]"),
    ("Source Information blurb",               r"^#### Source Information\s*$"),
    ("Editorial intro non-empty",              r"## Executive Summary\s*\n\n[^\s]"),
]


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Drop pre-existing handlers so re-runs in the same session don't double-log
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
    ))
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(stream_handler)

    return logging.getLogger("alignment")


def run_live_briefing(logger: logging.Logger) -> Path:
    """Invoke briefing_runner.py --dry-run as a subprocess and return the
    path to the generated markdown."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "briefing_runner.py"),
        "--config", str(REPO_ROOT / "config.yaml"),
        "--dry-run",
        "--log-level", "DEBUG",
    ]
    logger.info("Running live briefing: %s", " ".join(cmd))
    logger.info("(this takes 2-5 minutes; the briefing runner's own logs "
                "are streamed below)")

    # Stream subprocess output through our logger so it lands in the
    # alignment log file too
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in result.stdout.splitlines():
        logger.debug("[briefing] %s", line)

    if result.returncode == 2:
        raise RuntimeError(
            f"briefing_runner.py exited with code 2 (total failure). "
            f"Check the alignment log for details."
        )
    if result.returncode == 1:
        logger.warning("briefing_runner.py exited with code 1 (partial "
                       "failure) — markdown was likely still generated, "
                       "continuing with alignment check")

    # Find today's generated markdown (filename pattern set in config.yaml)
    today = datetime.datetime.now()
    glob_pattern = f"Atlas-Briefing-{today.year}.{today.month:02d}.{today.day:02d}*.md"
    candidates = sorted(REPO_ROOT.glob(glob_pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No briefing markdown found matching {glob_pattern} in "
            f"{REPO_ROOT}. The runner may have failed before writing the "
            f"file — check the alignment log."
        )
    return candidates[-1]


def run_mock_briefing(logger: logging.Logger) -> Path:
    """Generate a briefing using synthetic data — no API calls. Exercises
    only the markdown-rendering code path."""
    from scripts.briefing_runner import BriefingRunner, load_config

    logger.info("Running mock briefing (no API calls)")
    config = load_config(str(REPO_ROOT / "config.yaml"))
    runner = BriefingRunner(config, dry_run=True)

    stocks = [
        {"symbol": "NVDA", "current_price": 142.80, "percent_change": -3.20,
         "news_correlation": "AI chip export controls"},
        {"symbol": "MSFT", "current_price": 421.94, "percent_change": 3.06,
         "news_correlation": "AI optimism resilience"},
        {"symbol": "PLTR", "current_price": 134.01, "percent_change": 0.21,
         "news_correlation": "Defense innovation interest"},
        {"symbol": "GOOGL", "current_price": 396.80, "percent_change": -1.06,
         "news_correlation": "Broad tech selloff"},
        {"symbol": "SPY", "current_price": 739.19, "percent_change": -1.20,
         "news_correlation": "Broad market weakness"},
    ]

    news = [
        {"title": "Mock: AI in military operations expands",
         "url": "https://example.com/news1",
         "brief_summary": "European defense forces are integrating AI into "
                          "battlefield decision systems.",
         "author_blurb": "Example News is a fictional outlet used here to "
                         "exercise the Source Information renderer."},
        {"title": "Mock: Chip export controls tighten",
         "url": "https://example.com/news2",
         "brief_summary": "New restrictions limit high-end accelerator "
                          "exports to additional regions.",
         "author_blurb": "Example Wire is a fictional outlet used here to "
                         "exercise the Source Information renderer."},
    ]

    top_papers = [
        {"title": "Mock: Formal Verification of Agentic Workflows",
         "authors": ["A. Researcher", "B. Coauthor"],
         "arxiv_url": "http://arxiv.org/abs/9999.00001",
         "brief_summary": "Introduces a formally verifiable workflow "
                          "architecture for multi-step agents.",
         "score_combined": 4,
         "repro_total": 15, "repro_verdict": "Medium",
         "reproduction_difficulty": "M",
         "author_blurb": "The mock authors are placeholders to exercise "
                         "the paper Source Information renderer."},
        {"title": "Mock: Closed-Loop Planning for Autonomous Systems",
         "authors": ["C. Investigator"],
         "arxiv_url": "http://arxiv.org/abs/9999.00002",
         "brief_summary": "Closed-loop value estimation improves planner "
                          "selection over pure imitation learning.",
         "score_combined": 4,
         "repro_total": 13, "repro_verdict": "Hard",
         "reproduction_difficulty": "L",
         "author_blurb": "Mock placeholder for the author blurb renderer."},
    ]

    blogs = [
        {"title": "Mock: Building Real-Time Voice Agents",
         "source": "Example ML Blog",
         "link": "https://example.com/blog1",
         "score_combined": 4,
         "brief_summary": "How to build low-latency voice agents on a "
                          "fictional inference stack.",
         "author_blurb": "Mock placeholder for the blog Source Information "
                         "renderer."},
    ]

    synthesis = {
        "editorial_intro": (
            "This is a mock executive summary used by the alignment test. "
            "It exists to verify the markdown rendering pipeline produces "
            "an Executive Summary section that resembles the reference "
            "briefing's structure."
        ),
    }

    markdown = runner.generate_markdown_briefing(
        papers=top_papers,
        blogs=blogs,
        stocks=stocks,
        news=news,
        top_papers=top_papers,
        synthesis=synthesis,
        market_trend="Mock market trend: mixed session with tech selloff.",
    )

    out_path = REPO_ROOT / "logs" / "mock-briefing.md"
    out_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote mock briefing: %s", out_path)
    return out_path


def check_alignment(markdown: str, logger: logging.Logger,
                    mock_mode: bool = False) -> Tuple[int, int, List[str]]:
    """Walk REQUIRED and OPTIONAL pattern lists, return (passed, failed,
    report_lines). Optional misses are reported but never fail the test.
    In mock mode, live-only required patterns are skipped (they depend on
    real Gemini calls happening)."""
    report = []
    bar = "=" * 70
    report.append(bar)
    report.append("ATLAS BRIEFING ALIGNMENT REPORT")
    report.append(f"Reference: {REFERENCE_BRIEFING.name}")
    report.append(f"Mode: {'mock' if mock_mode else 'live'}")
    report.append(bar)
    report.append("")

    passed = failed = 0

    report.append("REQUIRED structural elements:")
    for label, pattern, live_only in REQUIRED_PATTERNS:
        if mock_mode and live_only:
            report.append(f"  [SKIP] {label} (live-only, skipped in mock mode)")
            continue
        ok = bool(re.search(pattern, markdown, re.MULTILINE))
        mark = "PASS" if ok else "FAIL"
        report.append(f"  [{mark}] {label}")
        if ok:
            passed += 1
        else:
            failed += 1
            logger.warning("Missing required pattern (%s): %r", label, pattern)

    report.append("")
    report.append("OPTIONAL elements (informational only):")
    for label, pattern in OPTIONAL_PATTERNS:
        ok = bool(re.search(pattern, markdown, re.MULTILINE))
        mark = "PRESENT" if ok else "absent "
        report.append(f"  [{mark}] {label}")

    report.append("")
    report.append("Quantitative comparison vs reference:")

    word_count = len(markdown.split())
    ref_word_count = (len(REFERENCE_BRIEFING.read_text().split())
                      if REFERENCE_BRIEFING.exists() else 0)
    report.append(f"  Word count: generated={word_count}, "
                  f"reference={ref_word_count} "
                  f"(generated should be at least ~40% of reference, "
                  f"i.e. ~{int(ref_word_count * 0.4)})")

    source_blurbs = len(re.findall(r"^#### Source Information\s*$",
                                   markdown, re.MULTILINE))
    report.append(f"  Source Information blurbs: {source_blurbs} "
                  f"(reference has 10: 5 news + 2 blogs + 3 papers)")

    stock_rows = len(re.findall(r"^\| \*\*[A-Z]{1,5}\*\* \|",
                                markdown, re.MULTILINE))
    report.append(f"  Stock ticker rows: {stock_rows} "
                  f"(reference has 16)")

    news_items = len(re.findall(r"^\*\*\[.+?\]\(http", markdown, re.MULTILINE))
    report.append(f"  Linked headlines (news + blogs combined): {news_items} "
                  f"(reference has 7: 5 news + 2 blogs)")

    paper_entries = len(re.findall(r"^### \d+\. \[", markdown, re.MULTILINE))
    report.append(f"  Top Paper entries: {paper_entries} "
                  f"(reference has 3)")

    expected_total = sum(
        1 for _, _, live_only in REQUIRED_PATTERNS
        if not (mock_mode and live_only)
    )
    report.append("")
    report.append(bar)
    if failed == 0:
        report.append(f"VERDICT: ALIGNED — {passed}/{expected_total} "
                      f"required elements present")
    else:
        report.append(f"VERDICT: MISALIGNED — {failed} required element(s) "
                      f"missing, {passed} present")
    report.append(bar)

    return passed, failed, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate generated briefing against the reference report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Skip API calls; generate briefing from synthetic data",
    )
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = REPO_ROOT / "logs" / f"alignment-{ts}.log"
    report_path = REPO_ROOT / "logs" / f"alignment-report-{ts}.txt"

    logger = setup_logging(log_path)
    logger.info("=" * 70)
    logger.info("Atlas Briefing Alignment Test")
    logger.info("Mode: %s", "mock" if args.mock else "live")
    logger.info("Log:  %s", log_path)
    logger.info("=" * 70)

    if not REFERENCE_BRIEFING.exists():
        logger.warning("Reference briefing not found at %s — quantitative "
                       "comparisons will be skipped", REFERENCE_BRIEFING)

    try:
        if args.mock:
            md_path = run_mock_briefing(logger)
        else:
            md_path = run_live_briefing(logger)
    except Exception as e:
        logger.error("Briefing generation failed: %s", e, exc_info=True)
        return 2

    logger.info("Generated briefing: %s", md_path)
    markdown = md_path.read_text(encoding="utf-8")

    passed, failed, report_lines = check_alignment(markdown, logger,
                                                   mock_mode=args.mock)
    report_text = "\n".join(report_lines)
    report_path.write_text(report_text + "\n", encoding="utf-8")

    print()
    print(report_text)
    print()
    print(f"Generated briefing : {md_path}")
    print(f"Full log file      : {log_path}")
    print(f"Alignment report   : {report_path}")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
