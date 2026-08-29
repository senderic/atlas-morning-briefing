#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Layer 2 of the daily quality check: deterministic report invariants.

A pure scan of a *rendered* briefing markdown string against the config that
produced it. No LLM, no network, no file I/O beyond what the caller hands in
(the CLI entry point is the only place that touches disk). Every check here
corresponds to a defect that actually shipped to a reader -- see
``references/quality_monitoring_design.md``, "Layer 2 -- Report invariants".

Three other modules already implement pieces of this logic, and rather than
mirror their rules with a second hand-maintained copy, this module imports
the actual objects those rules are made of. A check whose copy has drifted
from the rule it claims to verify would keep reporting green while testing
nothing real, so:

- ``scripts/intelligence.py``'s compiled ``_TRAILING_RATIONALE_RE`` *is*
  the leaked-filtering-rationale pattern -- ``check_scaffolding_leak`` uses
  that exact regex object, not a copy of it.
- ``scripts/text_similarity.py`` is the shared home for the
  Jaccard-over-content-words headline measure (``headline_terms``,
  ``jaccard``, ``DEFAULT_SIMILARITY_THRESHOLD``); ``briefing_runner.py``'s
  own dedup delegates to it too, so ``check_near_duplicates`` and the
  pipeline's ``deduplicate_similar_news`` are provably measuring the same
  thing.
- ``scripts/geo_filter.py``'s public ``is_local`` and private ``_host_matches``
  are imported directly for the same reason -- they are the actual filtering
  logic, not just a pattern to mirror, so importing them is the only way
  "blocked" and "out-of-area" mean the same thing here as they do in the
  pipeline.

Usage:
    python3 scripts/report_invariants.py --config config_local.yaml \\
        --report briefings/local/Local-Briefing-2026.08.25.md
"""

import argparse
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Allow `python3 scripts/report_invariants.py ...` to resolve `scripts.*`
# imports even when the repo root isn't already on sys.path (mirrors
# briefing_runner.py's own bootstrap).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.geo_filter import _host_matches, is_local
from scripts.intelligence import _TRAILING_RATIONALE_RE as _SCAFFOLDING_LEAK_RE
from scripts.quality_findings import CRITICAL, INFO, WARN, Finding, sort_findings
from scripts.text_similarity import DEFAULT_SIMILARITY_THRESHOLD, headline_terms, jaccard

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown structure helpers
# ---------------------------------------------------------------------------

# A top-level section heading, as every renderer in briefing_runner.py emits
# it: "## <heading text>\n\n". Sub-headings ("#### Source Information") use
# more hashes and are deliberately not matched.
_HEADING_RE = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)

# One rendered item. Every list-section renderer opens an item with a bold
# title, but the shape after the closing "**" differs by section:
#
#   news / happenings:  **[Title](url)**\n<summary>\n\n
#   blogs (linked):      **[Title](url)** *(Source Name)* \u2605\u2605\u2605\u2605\u2606\n<summary>\n\n
#   blogs (no link):     **Title** *(Source Name)* \u2605\u2605\u2605\u2606\u2606\n<summary>\n\n
#
# _render_blogs appends the source and star-rating on the *same line* as the
# closing "**" (see briefing_runner.py's ``_render_blogs``:
# ``f"**[{title}]({link})** *({source})*{score_tag}\n"``, or
# ``f"**{title}** *({source})*{score_tag}\n"`` when there is no link). A
# version of this pattern that demanded a bare newline immediately after
# "**" missed every blog item -- the whole blogs section was invisible to
# this module's blocked-source, out-of-area, near-duplicate and
# thin-section checks, and looked like a false "empty" section rather than
# announcing a bug.
#
# The trailer is matched by its exact shape -- " *(Source)*" plus an
# optional " " + stars/circles -- not by "whatever comes before the next
# newline". That distinction matters: prose elsewhere in a rendered
# briefing routinely opens a line with inline bold ("**Watch:** the
# **Midway Rising certification**...") or renders a markdown table row
# ("| **Total** | ..."), and a wildcard trailer would swallow those as
# fake items too. Requiring the literal " *(...)*..." shape is what keeps
# this pattern matching only what _render_blogs actually emits.
#
# The no-link ("plain") title form is handled by a second pattern,
# _PLAIN_ITEM_RE, below -- and deliberately requires the source trailer
# rather than allowing a bare "**Title**\n". A bare bold line immediately
# followed by a newline is common outside the list sections too (e.g. the
# Gemini usage summary's "**Model fallback activity:**"), and is never how
# a real link-less blog item renders -- _render_blogs always appends the
# source trailer, link or no link -- so a link-less item only counts as a
# rendered item when that trailer is present.
_BLOG_TRAILER = r" \*\([^)\n]*\)\*(?: [\u2605\u2606]+)?"

_LINKED_ITEM_RE = re.compile(
    r"\*\*\[(?P<title>[^\]]*)\]\((?P<url>https?://[^\s\)]+)\)\*\*"
    rf"(?:{_BLOG_TRAILER})?\n"
    r"(?P<body>.*?)(?=\n{2,}|\n##\s|\Z)",
    re.DOTALL,
)
# (?!\[) keeps this from also matching a linked item's own "**...**" span
# a second time (e.g. starting at the *closing* "**" of a
# "**[Title](url)**" head and misreading what follows as a bare-bold
# title) -- a real plain title never starts with "[".
_PLAIN_ITEM_RE = re.compile(
    rf"\*\*(?P<title>(?!\[)[^*\n]+)\*\*(?:{_BLOG_TRAILER})\n"
    r"(?P<body>.*?)(?=\n{2,}|\n##\s|\Z)",
    re.DOTALL,
)


def _parse_headings(markdown: str) -> List[Tuple[str, int, int]]:
    """(heading text, body start offset, body end offset) for every '## ' heading."""
    matches = list(_HEADING_RE.finditer(markdown))
    sections = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append((m.group(1).strip(), body_start, body_end))
    return sections


def _section_body(markdown: str, heading_text: str) -> Optional[str]:
    """The text between a named heading and the next heading, or None if absent."""
    if not heading_text:
        return None
    for text, start, end in _parse_headings(markdown):
        if text == heading_text:
            return markdown[start:end]
    return None


def _iter_rendered_items(markdown: str) -> Iterator[Tuple[str, str, str]]:
    """
    Yield (title, url, body text) for every rendered item, in document
    order, whether it is a markdown link (``**[Title](url)**``, optionally
    with a blog source trailer) or plain bold text with a source trailer
    and no link (``**Title** *(Source)*``, as ``_render_blogs`` emits when
    an article has no URL). ``url`` is ``""`` for a link-less item.
    """
    matches = list(_LINKED_ITEM_RE.finditer(markdown)) + list(_PLAIN_ITEM_RE.finditer(markdown))
    matches.sort(key=lambda m: m.start())
    for m in matches:
        url = m.groupdict().get("url") or ""
        yield m.group("title").strip(), url.strip(), m.group("body").strip()


def _url_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# Check 1 -- section presence
# ---------------------------------------------------------------------------


def check_sections_present(
    markdown: str,
    config: Dict[str, Any],
    sections_with_data: Optional[Sequence[str]] = None,
    pipeline: str = "",
) -> List[Finding]:
    """
    A section named in ``section_order`` renders no heading at all, even
    though the caller says it had data to render.

    A section legitimately renders nothing when its input was empty (the
    renderers all early-return on an empty list), so this only flags a
    section the caller explicitly marked as having data. When
    ``sections_with_data`` is None, the check is skipped entirely -- the
    caller didn't tell us which sections had data, so we can't tell "no
    heading, correctly" from "no heading, bug".
    """
    if sections_with_data is None:
        return []

    section_order = config.get("section_order") or []
    headings_cfg = config.get("section_headings") or {}
    rendered = {text for text, _, _ in _parse_headings(markdown)}
    data_set = set(sections_with_data)

    findings = []
    for section in section_order:
        if section not in data_set:
            continue
        heading = headings_cfg.get(section, section)
        if heading not in rendered:
            findings.append(
                Finding(
                    severity=CRITICAL,
                    code="section-missing",
                    message=(
                        f"Section '{section}' had data but no '## {heading}' "
                        f"heading rendered"
                    ),
                    source=section,
                    pipeline=pipeline,
                    detail={"section": section, "heading": heading},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Check 2 -- section order
# ---------------------------------------------------------------------------


def check_section_order(
    markdown: str, config: Dict[str, Any], pipeline: str = ""
) -> List[Finding]:
    """
    Headings appear in a different order than ``section_order`` configures.

    Sections absent from the render (no data) are ignored -- comparing only
    the subset of expected headings that actually rendered, in the order
    they rendered, against that same subset in configured order.
    """
    section_order = config.get("section_order") or []
    headings_cfg = config.get("section_headings") or {}
    expected_texts = [headings_cfg.get(s, s) for s in section_order]

    rendered = [text for text, _, _ in _parse_headings(markdown)]
    expected_present = [h for h in expected_texts if h in rendered]
    actual_present = [h for h in rendered if h in expected_present]

    if expected_present and actual_present != expected_present:
        return [
            Finding(
                severity=WARN,
                code="section-order",
                message=(
                    f"Sections rendered out of configured order: expected "
                    f"{expected_present}, got {actual_present}"
                ),
                pipeline=pipeline,
                detail={"expected": expected_present, "actual": actual_present},
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Check 3 -- stale events (happenings section only)
# ---------------------------------------------------------------------------

# Date parsing lives in scripts.event_dates so the pipeline's happenings
# filter and this check cannot drift on what counts as a calendar date.
from scripts.event_dates import (  # noqa: F401  (re-exported for tests)
    _ARROW_RANGE_RE,
    _DASH_RANGE_RE,
    _MONTHS,
    _SINGLE_DATE_RE,
    _STALE_WINDOW_DAYS,
    _extract_dates_from_line,
    _mask,
    _month_num,
    _safe_date,
)


def check_stale_events(
    markdown: str,
    config: Dict[str, Any],
    today: Optional[date] = None,
    pipeline: str = "",
) -> List[Finding]:
    """
    An explicit calendar date in the happenings section that is already in
    the past.

    Only the happenings section is scanned -- an alerts window legitimately
    starts in the past and a news story legitimately references one, so
    scanning the whole document would make this check useless. For a date
    range, staleness is decided by the end of the range. A bare month with
    no day ("in September") is never treated as a date.
    """
    today = today or date.today()
    heading = (config.get("section_headings") or {}).get("happenings")
    body = _section_body(markdown, heading) if heading else None
    if body is None:
        return []

    findings = []
    seen_dates = set()
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for d in _extract_dates_from_line(line, today):
            if d >= today or d in seen_dates:
                continue
            seen_dates.add(d)
            findings.append(
                Finding(
                    severity=CRITICAL,
                    code="stale-event",
                    message=(
                        f"Happenings section references {d.isoformat()}, "
                        f'already past: "{line}"'
                    ),
                    source=heading,
                    pipeline=pipeline,
                    detail={"date": d.isoformat(), "line": line},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Check 4 -- scaffolding leak
# ---------------------------------------------------------------------------

# _SCAFFOLDING_LEAK_RE *is* scripts.intelligence._TRAILING_RATIONALE_RE (see
# the import above) -- not a copy of its pattern. Importing the compiled
# regex object is what guarantees this check and _strip_trailing_rationale
# can never quietly drift apart.


def check_scaffolding_leak(
    markdown: str, config: Optional[Dict[str, Any]] = None, pipeline: str = ""
) -> List[Finding]:
    """
    Reader-facing copy containing leaked model meta-commentary.

    Matches "Dropped:" / "Excluded:" / "Omitted:" / "Not selected:" /
    "Rejected:" / "Skipped:" only when they start a sentence or a line, so
    legitimate prose ("Charges were dropped: the DA declined to file")
    survives untouched.
    """
    findings = []
    seen_lines = set()
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line or line in seen_lines:
            continue
        m = _SCAFFOLDING_LEAK_RE.search(line)
        if m:
            seen_lines.add(line)
            findings.append(
                Finding(
                    severity=CRITICAL,
                    code="scaffolding-leak",
                    message=(
                        "Model filtering rationale leaked into reader-facing "
                        f'copy: "{line}"'
                    ),
                    pipeline=pipeline,
                    detail={"marker": line[m.start(): m.end()].strip()},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Check 5 -- blocked source
# ---------------------------------------------------------------------------


def check_blocked_sources(
    markdown: str, config: Dict[str, Any], pipeline: str = ""
) -> List[Finding]:
    """A rendered link whose host matches ``geo_filter.blocked_sources``."""
    blocked = [b for b in (config.get("geo_filter") or {}).get("blocked_sources", []) if str(b).strip()]
    if not blocked:
        return []

    findings = []
    seen_hosts = set()
    for title, url, _body in _iter_rendered_items(markdown):
        host = _url_host(url)
        if host and _host_matches(host, blocked) and host not in seen_hosts:
            seen_hosts.add(host)
            findings.append(
                Finding(
                    severity=WARN,
                    code="blocked-source",
                    message=f'Item from blocked source {host} rendered: "{title}"',
                    source=host,
                    pipeline=pipeline,
                    detail={"url": url, "title": title},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Check 6 -- out of area
# ---------------------------------------------------------------------------


def check_out_of_area(
    markdown: str, config: Dict[str, Any], pipeline: str = ""
) -> List[Finding]:
    """
    A rendered item with no ``geo_filter.place_terms`` match and whose host
    is not in ``trusted_sources``.

    Only runs when ``geo_filter.enabled`` is true; a blocked-source item is
    skipped here even though ``is_local`` would also reject it, because that
    case is already reported (with the right code) by
    ``check_blocked_sources``.
    """
    geo_cfg = config.get("geo_filter") or {}
    if not geo_cfg.get("enabled"):
        return []
    place_terms = [t for t in geo_cfg.get("place_terms", []) if str(t).strip()]
    if not place_terms:
        return []
    trusted_sources = geo_cfg.get("trusted_sources", [])
    blocked_sources = geo_cfg.get("blocked_sources", [])

    findings = []
    seen = set()
    for title, url, body in _iter_rendered_items(markdown):
        host = _url_host(url)
        if _host_matches(host, blocked_sources):
            continue
        item = {"title": title, "description": body, "url": url}
        if not is_local(item, place_terms, trusted_sources, blocked_sources=blocked_sources):
            key = (host, title)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                Finding(
                    severity=WARN,
                    code="out-of-area",
                    message=(
                        f'Item has no local reference and is not from a '
                        f'trusted source: "{title}" ({host or "no host"})'
                    ),
                    source=host,
                    pipeline=pipeline,
                    detail={"url": url, "title": title},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Check 7 -- near duplicates
# ---------------------------------------------------------------------------

# headline_terms / jaccard / DEFAULT_SIMILARITY_THRESHOLD *are*
# scripts.text_similarity's (see the import above) -- not a copy of them.
# briefing_runner.BriefingRunner._headline_terms now delegates to the same
# module, so this check and the pipeline's own dedup are provably measuring
# the same thing. Threshold calibrated on live Brave results: cross-outlet
# retellings of the same story score 0.23-0.47, distinct stories score
# 0.00-0.17.


def check_near_duplicates(
    markdown: str, config: Optional[Dict[str, Any]] = None, pipeline: str = ""
) -> List[Finding]:
    """Two rendered item headings with Jaccard overlap of content words >= threshold."""
    config = config or {}
    threshold = float(
        (config.get("news_similarity_dedup") or {}).get("threshold", DEFAULT_SIMILARITY_THRESHOLD)
    )
    items = [(title, headline_terms(title)) for title, _url, _body in _iter_rendered_items(markdown)]

    findings = []
    seen_pairs = set()
    for i in range(len(items)):
        title_i, terms_i = items[i]
        if not terms_i:
            continue
        for j in range(i + 1, len(items)):
            title_j, terms_j = items[j]
            if not terms_j:
                continue
            overlap = jaccard(terms_i, terms_j)
            if overlap >= threshold:
                key = tuple(sorted((title_i, title_j)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                findings.append(
                    Finding(
                        severity=WARN,
                        code="near-duplicate",
                        message=(
                            f"Near-duplicate headlines (overlap {overlap:.2f}): "
                            f'"{title_i}" / "{title_j}"'
                        ),
                        pipeline=pipeline,
                        detail={"overlap": round(overlap, 3), "titles": [title_i, title_j]},
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# Check 8 -- placeholder text
# ---------------------------------------------------------------------------

_PLACEHOLDER_PATTERNS = ("your-email@", "YOUR_NAME", "example.com", "your-sender@")


def check_placeholder_text(
    markdown: str, config: Optional[Dict[str, Any]] = None, pipeline: str = ""
) -> List[Finding]:
    """Unfilled template placeholders leaking into the rendered output."""
    findings = []
    seen_patterns = set()
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for pattern in _PLACEHOLDER_PATTERNS:
            if pattern in raw_line and pattern not in seen_patterns:
                seen_patterns.add(pattern)
                findings.append(
                    Finding(
                        severity=WARN,
                        code="placeholder-text",
                        message=f'Placeholder text "{pattern}" found in rendered output: "{line}"',
                        pipeline=pipeline,
                        detail={"pattern": pattern},
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# Check 9 -- degraded content
# ---------------------------------------------------------------------------

# The pipeline's own LLM-fallback copy (briefing_runner.py's editorial-intro
# fallback is the verbatim first entry here -- see the Aug 26 Executive
# Summary this check exists to catch). Deliberately short and specific:
# a false CRITICAL every morning is worse than a missed one, so the default
# list only holds phrasing the pipeline itself is known to emit on a
# degraded LLM step, not generic words like "unavailable" that ordinary
# news prose says all the time.
_DEFAULT_DEGRADED_MARKERS = (
    "synthesis unavailable",
    "unavailable for today's briefing",
    "summary unavailable",
    "unable to generate",
)


def _is_usage_appendix_heading(heading: str) -> bool:
    """
    True for the LLM cost/usage appendix headings -- ``## Gemini Usage
    Summary``, ``## OpenRouter Usage Summary``, ``## Opencode Usage
    Summary`` (gemini_client.py / openrouter_client.py / opencode_client.py)
    and ``## API Key Rotation Summary`` (composite_client.py, or
    gemini_client.py's own non-composite rendering of the same table).

    This is operational metadata about the pipeline's own LLM calls -- it
    routinely says things like "Failures" and can legitimately reference a
    tier having zero successful calls -- not reader-facing briefing
    content, so it is out of scope for check_degraded_content regardless
    of what it says. Matched structurally by the heading shape every one
    of those renderers emits, rather than an enumerated list of provider
    names, so a future backend's usage-summary heading is skipped the same
    way without this function needing an update.
    """
    text = heading.strip().lower()
    return text.endswith("usage summary") or text == "api key rotation summary"


def check_degraded_content(
    markdown: str, config: Dict[str, Any], pipeline: str = ""
) -> List[Finding]:
    """
    A known LLM-fallback placeholder rendered into the briefing.

    Every ``[LLM]`` step in the pipeline has a deterministic fallback --
    when a heavy-tier call times out or every backend fails, the pipeline
    ships a placeholder like "Synthesis unavailable for today's briefing"
    instead of crashing the run. That fallback firing is correct, required
    behavior (see CLAUDE.md). What is a defect is nobody *noticing*: the
    run still exits 0, status.json still records ``errors: []``, and a
    reader gets a briefing with a placeholder where its lead section
    should be. This check is that missing signal, made deterministic and
    free -- no LLM judge required to catch a hard, string-matchable
    default.

    Scoping: markers are matched against each rendered section's own body
    text -- walked one heading at a time via ``_parse_headings``, the same
    structure ``_section_body`` uses -- rather than a single substring
    search over the whole markdown file. That has two effects. First, the
    LLM usage-summary appendix (see ``_is_usage_appendix_heading``) is
    skipped outright -- it is pipeline cost/call metadata, not reader
    content, and irrelevant to whether the briefing itself degraded.
    Second, severity is attributable to *which* section the placeholder
    landed in: a marker phrase that shows up inside an ordinary news
    item's own summary (e.g. quoting some unrelated product's outage
    notice) is scoped to that section and reads as a WARN like anything
    else found there, never a false CRITICAL. Only a match inside the
    executive summary's own body is CRITICAL, because that section is
    always meant to be the pipeline's own synthesis, never a quote from a
    source, and it is the one section every reader reads.

    Reads markers from ``config["quality_check"]["degraded_markers"]``
    (case-insensitive substring match); the key being absent falls back to
    ``_DEFAULT_DEGRADED_MARKERS``, while an explicit empty list disables
    the check entirely.

    Count: at most one Finding per degraded section, no matter how many
    configured markers match its body text -- the reportable fact is "this
    section is degraded", not "N of my configured synonyms happen to
    overlap in this one placeholder". The real Aug 26 placeholder trips
    both "synthesis unavailable" and "unavailable for today's briefing" at
    once; that is one defect, and a digest headline ("3 CRITICAL") that
    inflates by however many synonyms overlap is exactly the kind of noise
    this checker exists to avoid. Every marker that matched is still kept
    in ``detail["markers"]`` for debugging. Two genuinely different
    degraded sections still produce two findings, one each.
    """
    quality_cfg = config.get("quality_check") or {}
    markers_cfg = quality_cfg.get("degraded_markers")
    if markers_cfg is None:
        markers_cfg = _DEFAULT_DEGRADED_MARKERS
    markers = [str(m).strip().lower() for m in markers_cfg if str(m).strip()]
    if not markers:
        return []

    exec_heading = (config.get("section_headings") or {}).get(
        "executive_summary", "Executive Summary"
    )

    findings = []
    for heading, start, end in _parse_headings(markdown):
        if _is_usage_appendix_heading(heading):
            continue
        body = markdown[start:end]
        body_lower = body.lower()
        # A section is degraded or it isn't -- which configured synonym(s)
        # happen to overlap is incidental and must not multiply the
        # reported count. One finding per section, with every marker that
        # matched kept in `detail` for debugging.
        matched = [marker for marker in markers if marker in body_lower]
        if not matched:
            continue
        line = next(
            (
                raw.strip()
                for raw in body.splitlines()
                if any(marker in raw.strip().lower() for marker in matched)
            ),
            body.strip(),
        )
        severity = CRITICAL if heading == exec_heading else WARN
        findings.append(
            Finding(
                severity=severity,
                code="degraded-content",
                message=(
                    f"Degradation placeholder found in '{heading}' "
                    f'section: "{line}"'
                ),
                source=heading,
                pipeline=pipeline,
                detail={"markers": matched, "section": heading, "line": line},
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Check 10 -- thin sections
# ---------------------------------------------------------------------------


def check_thin_sections(
    markdown: str, config: Dict[str, Any], pipeline: str = ""
) -> List[Finding]:
    """
    A section with fewer rendered items than a configured floor.

    Reads floors from ``config["quality_check"]["section_floors"]`` as
    ``{section_name: int}``. Absent config means this check does nothing. A
    section absent from the render entirely is left to
    ``check_sections_present`` -- this only fires on a section that rendered
    but rendered thin.
    """
    floors = (config.get("quality_check") or {}).get("section_floors") or {}
    if not floors:
        return []

    headings_cfg = config.get("section_headings") or {}
    findings = []
    for section, floor in floors.items():
        heading = headings_cfg.get(section, section)
        body = _section_body(markdown, heading)
        if body is None:
            continue
        count = len(list(_iter_rendered_items(body)))
        if count < int(floor):
            findings.append(
                Finding(
                    severity=INFO,
                    code="thin-section",
                    message=(
                        f"Section '{section}' rendered {count} item(s), "
                        f"below configured floor {floor}"
                    ),
                    source=section,
                    pipeline=pipeline,
                    detail={"count": count, "floor": int(floor)},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def check_report(
    markdown: str,
    config: Dict[str, Any],
    today: Optional[date] = None,
    pipeline: str = "",
    sections_with_data: Optional[Sequence[str]] = None,
) -> List[Finding]:
    """Run every Layer 2 check against a rendered briefing and return all findings."""
    today = today or date.today()
    findings: List[Finding] = []
    findings += check_sections_present(
        markdown, config, sections_with_data=sections_with_data, pipeline=pipeline
    )
    findings += check_section_order(markdown, config, pipeline=pipeline)
    findings += check_stale_events(markdown, config, today=today, pipeline=pipeline)
    findings += check_scaffolding_leak(markdown, config, pipeline=pipeline)
    findings += check_degraded_content(markdown, config, pipeline=pipeline)
    findings += check_blocked_sources(markdown, config, pipeline=pipeline)
    findings += check_out_of_area(markdown, config, pipeline=pipeline)
    findings += check_near_duplicates(markdown, config, pipeline=pipeline)
    findings += check_placeholder_text(markdown, config, pipeline=pipeline)
    findings += check_thin_sections(markdown, config, pipeline=pipeline)
    return findings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic invariant scan of a rendered briefing markdown file"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml / config_local.yaml")
    parser.add_argument("--report", required=True, help="Path to a rendered briefing markdown file")
    parser.add_argument("--pipeline", default="", help="Pipeline label attached to each finding")
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    with open(args.report, encoding="utf-8") as f:
        markdown = f.read()

    findings = sort_findings(check_report(markdown, config, pipeline=args.pipeline))

    if not findings:
        print("No report invariant violations found.")
        return 0

    for finding in findings:
        print(f"[{finding.severity}] {finding.code}: {finding.message}")

    return 1 if any(f.severity == CRITICAL for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
