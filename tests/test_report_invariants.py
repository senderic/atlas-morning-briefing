# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Tests for Layer 2 of the daily quality check: deterministic report
invariants (scripts/report_invariants.py).

Several fixtures below are verbatim strings from real production output
(see references/quality_monitoring_design.md) -- the defects they encode
actually shipped to a reader, so the check that catches them is pinned to
the exact text that slipped through.
"""

from datetime import date

from scripts import intelligence, report_invariants
from scripts.quality_findings import CRITICAL, INFO, WARN
from scripts.report_invariants import (
    check_blocked_sources,
    check_degraded_content,
    check_near_duplicates,
    check_out_of_area,
    check_placeholder_text,
    check_report,
    check_scaffolding_leak,
    check_section_order,
    check_sections_present,
    check_stale_events,
    check_thin_sections,
)
from scripts.text_similarity import headlines_match

HAPPENINGS_HEADING = "Pacific Beach Area — Upcoming Happenings"
MINIMAL_HAPPENINGS_CONFIG = {"section_headings": {"happenings": HAPPENINGS_HEADING}}


def _happenings_markdown(body: str) -> str:
    return f"## {HAPPENINGS_HEADING}\n\n{body}\n"


# ---------------------------------------------------------------------------
# check_stale_events -- the most important check
# ---------------------------------------------------------------------------

# Verbatim from production: a weekend-events roundup served three days after
# the weekend it advertised had ended.
STALE_WEEKEND_FIXTURE = (
    "**[The best things to do this weekend in San Diego: Aug. 21-23 - San "
    "Diego Union-Tribune](https://example.org/a)**\n"
    "This weekend's lineup (Aug. 21-23) runs HarborFest, the KPBS San Diego "
    "Book Festival, SAMAFEST, and the TAP-SD Taiwan Festival."
)

# Verbatim from production: a farmers-market listing dated the day after
# today -- must never be flagged as stale.
FRESH_FARMERS_MARKET_FIXTURE = (
    "**[Events Listing | City of San Diego](https://example.org/b)**\n"
    "The weekday Little Italy farmer's market runs Wednesday, Aug. 26, 9:30 "
    "a.m.-1:30 p.m., on three blocks of West Date Street."
)


class TestCheckStaleEvents:
    def test_must_fire_on_past_dated_weekend_range(self):
        md = _happenings_markdown(STALE_WEEKEND_FIXTURE)
        findings = check_stale_events(md, MINIMAL_HAPPENINGS_CONFIG, today=date(2026, 8, 25))
        assert len(findings) == 1
        assert findings[0].code == "stale-event"
        assert findings[0].severity == CRITICAL
        # Range staleness is decided by the end of the range (Aug 23), and
        # the message must name the date and quote the offending line.
        assert findings[0].detail["date"] == "2026-08-23"
        assert "Aug. 21-23" in findings[0].message

    def test_must_not_fire_on_tomorrows_farmers_market(self):
        md = _happenings_markdown(FRESH_FARMERS_MARKET_FIXTURE)
        findings = check_stale_events(md, MINIMAL_HAPPENINGS_CONFIG, today=date(2026, 8, 25))
        assert findings == []

    def test_bare_month_with_no_day_is_never_a_date(self):
        md = _happenings_markdown(
            "**[Fall Calendar](https://example.org/c)**\n"
            "13 Charitable Events to Attend in September"
        )
        findings = check_stale_events(md, MINIMAL_HAPPENINGS_CONFIG, today=date(2026, 8, 25))
        assert findings == []

    def test_arrow_range_staleness_decided_by_end_date(self):
        md = _happenings_markdown(
            "**[Multi-day Festival](https://example.org/d)**\n"
            "Tue Aug 25, 10:00 AM -> Fri Aug 28, 8:00 PM"
        )
        # After the range ends (Aug 29): stale.
        stale = check_stale_events(md, MINIMAL_HAPPENINGS_CONFIG, today=date(2026, 8, 29))
        assert len(stale) == 1
        assert stale[0].detail["date"] == "2026-08-28"

        # Before the range ends (Aug 26): not stale.
        fresh = check_stale_events(md, MINIMAL_HAPPENINGS_CONFIG, today=date(2026, 8, 26))
        assert fresh == []

    def test_only_happenings_section_is_scanned(self):
        # An alerts window legitimately starts in the past; a stray date in
        # some other section must never trip this check.
        md = (
            "## Active Alerts\n\n"
            "**Coastal Flood Advisory**\n"
            "Tue Aug 18, 5:00 AM -> Tue Aug 18, 11:00 AM\n\n"
            f"## {HAPPENINGS_HEADING}\n\n"
            "Nothing dated in this section.\n"
        )
        findings = check_stale_events(md, MINIMAL_HAPPENINGS_CONFIG, today=date(2026, 8, 25))
        assert findings == []

    def test_no_happenings_heading_configured_is_a_no_op(self):
        md = _happenings_markdown(STALE_WEEKEND_FIXTURE)
        assert check_stale_events(md, {}, today=date(2026, 8, 25)) == []


# ---------------------------------------------------------------------------
# check_scaffolding_leak
# ---------------------------------------------------------------------------

# Verbatim from production: the model's own filtering rationale leaked into
# the reader-facing summary.
LEAKED_RATIONALE_FIXTURE = (
    "Skim it to find the beach-adjacent events before they fill up. Dropped: "
    "[1], [2], [7] (Mission Bay funding/planning/policy), [3], [4] "
    "(opinion/climate commentary)."
)

# Legitimate prose using the same words mid-sentence -- must never fire.
LEGITIMATE_DROPPED_PROSE = (
    "Charges were dropped: the DA declined to file. The suit was dropped "
    "last week."
)


class TestCheckScaffoldingLeak:
    def test_must_fire_on_leaked_filtering_rationale(self):
        findings = check_scaffolding_leak(LEAKED_RATIONALE_FIXTURE)
        assert len(findings) == 1
        assert findings[0].code == "scaffolding-leak"
        assert findings[0].severity == CRITICAL
        assert "Dropped:" in findings[0].message

    def test_must_not_fire_on_legitimate_mid_sentence_usage(self):
        assert check_scaffolding_leak(LEGITIMATE_DROPPED_PROSE) == []

    def test_fires_on_line_start_too(self):
        text = "Some lead-in text.\nExcluded: [2], [5] (duplicate coverage)."
        findings = check_scaffolding_leak(text)
        assert len(findings) == 1
        assert findings[0].code == "scaffolding-leak"

    def test_uses_the_imported_intelligence_regex_object(self):
        # This check exists to assert intelligence.py's leak-scrubbing rule
        # is still working. If it silently drifted to a private copy of the
        # pattern, it would keep reporting green while testing a rule
        # nobody enforces any more -- so pin the coupling itself, not just
        # a shared outcome.
        assert report_invariants._SCAFFOLDING_LEAK_RE is intelligence._TRAILING_RATIONALE_RE


# ---------------------------------------------------------------------------
# check_blocked_sources
# ---------------------------------------------------------------------------

# Verbatim from production: a pay-to-publish press release rendered as
# neighborhood news.
BLOCKED_SOURCE_FIXTURE = (
    "**[Seaside Pizza Co. Adds Beer and Wine to Its Pacific Beach Pizza "
    "Takeout Experience](https://openpr.com/news/4612028/seaside)**\n"
    "Seaside Pizza Co. announced an expanded beer and wine menu at its "
    "Pacific Beach takeout location.\n"
)


class TestCheckBlockedSources:
    def test_must_fire_when_host_is_blocked(self):
        config = {"geo_filter": {"blocked_sources": ["openpr.com"]}}
        findings = check_blocked_sources(BLOCKED_SOURCE_FIXTURE, config)
        assert len(findings) == 1
        assert findings[0].code == "blocked-source"
        assert findings[0].severity == WARN
        assert findings[0].source == "openpr.com"

    def test_silent_with_no_blocked_sources_configured(self):
        assert check_blocked_sources(BLOCKED_SOURCE_FIXTURE, {}) == []

    def test_silent_when_host_not_blocked(self):
        config = {"geo_filter": {"blocked_sources": ["some-other-site.com"]}}
        assert check_blocked_sources(BLOCKED_SOURCE_FIXTURE, config) == []


# ---------------------------------------------------------------------------
# check_sections_present
# ---------------------------------------------------------------------------

_SECTIONS_CONFIG = {
    "section_order": ["alerts", "happenings"],
    "section_headings": {"alerts": "Active Alerts", "happenings": "Upcoming Happenings"},
}


class TestCheckSectionsPresent:
    def test_fires_when_heading_missing_but_caller_says_data_existed(self):
        md = "## Active Alerts\n\nSomething.\n"
        findings = check_sections_present(
            md, _SECTIONS_CONFIG, sections_with_data=["alerts", "happenings"]
        )
        assert len(findings) == 1
        assert findings[0].code == "section-missing"
        assert findings[0].severity == CRITICAL
        assert findings[0].source == "happenings"

    def test_silent_when_section_legitimately_had_no_data(self):
        md = "## Active Alerts\n\nSomething.\n"
        findings = check_sections_present(
            md, _SECTIONS_CONFIG, sections_with_data=["alerts"]
        )
        assert findings == []

    def test_skipped_entirely_when_sections_with_data_is_none(self):
        md = "## Active Alerts\n\nSomething.\n"
        assert check_sections_present(md, _SECTIONS_CONFIG) == []
        assert check_sections_present(md, _SECTIONS_CONFIG, sections_with_data=None) == []


# ---------------------------------------------------------------------------
# check_section_order
# ---------------------------------------------------------------------------

_ORDER_CONFIG = {
    "section_order": ["alerts", "happenings", "news"],
    "section_headings": {"alerts": "Alerts", "happenings": "Happenings", "news": "News"},
}


class TestCheckSectionOrder:
    def test_fires_when_headings_are_swapped(self):
        md = "## Happenings\n\nx\n\n## Alerts\n\ny\n\n## News\n\nz\n"
        findings = check_section_order(md, _ORDER_CONFIG)
        assert len(findings) == 1
        assert findings[0].code == "section-order"
        assert findings[0].severity == WARN

    def test_silent_when_order_is_correct(self):
        md = "## Alerts\n\ny\n\n## Happenings\n\nx\n\n## News\n\nz\n"
        assert check_section_order(md, _ORDER_CONFIG) == []

    def test_absent_sections_are_ignored_for_ordering(self):
        # happenings never rendered (no data) -- the remaining two are still
        # in the right relative order, so this must stay silent.
        md = "## Alerts\n\ny\n\n## News\n\nz\n"
        assert check_section_order(md, _ORDER_CONFIG) == []


# ---------------------------------------------------------------------------
# check_out_of_area
# ---------------------------------------------------------------------------


class TestCheckOutOfArea:
    def test_fires_for_untrusted_source_with_no_place_term(self):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["san diego", "pacific beach"],
                "trusted_sources": ["voiceofsandiego.org"],
            }
        }
        md = (
            "**[National Chain Reports Earnings](https://example.org/national)**\n"
            "A national retail chain announced quarterly earnings today.\n"
        )
        findings = check_out_of_area(md, config)
        assert len(findings) == 1
        assert findings[0].code == "out-of-area"
        assert findings[0].severity == WARN

    def test_silent_for_trusted_source_even_with_no_place_term(self):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["pacific beach"],
                "trusted_sources": ["voiceofsandiego.org"],
            }
        }
        md = (
            "**[Local Update](https://voiceofsandiego.org/update)**\n"
            "A story that never mentions a place term.\n"
        )
        assert check_out_of_area(md, config) == []

    def test_no_op_when_geo_filter_disabled(self):
        config = {"geo_filter": {"enabled": False, "place_terms": ["pacific beach"]}}
        md = "**[Random](https://example.org/x)**\nNo place term here.\n"
        assert check_out_of_area(md, config) == []

    def test_blocked_source_not_double_reported_as_out_of_area(self):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["pacific beach"],
                "blocked_sources": ["openpr.com"],
            }
        }
        md = "**[Ad Copy](https://openpr.com/x)**\nNo place term here.\n"
        # Blocked is check_blocked_sources' job -- this check must stay quiet.
        assert check_out_of_area(md, config) == []


# ---------------------------------------------------------------------------
# check_near_duplicates
# ---------------------------------------------------------------------------


class TestCheckNearDuplicates:
    def test_fires_for_syndicated_retellings(self):
        md = (
            "**[Darth Vader Speaks At City Council Meeting](https://a.example/1)**\n"
            "Body one.\n\n"
            "**[City Council Meeting Hears From Darth Vader](https://b.example/2)**\n"
            "Body two.\n"
        )
        findings = check_near_duplicates(md)
        assert len(findings) == 1
        assert findings[0].code == "near-duplicate"
        assert findings[0].severity == WARN

    def test_silent_for_genuinely_distinct_stories(self):
        md = (
            "**[Pacific Beach Parking Rates Rise](https://a.example/1)**\n"
            "Body one.\n\n"
            "**[Mission Bay Park Plan Advances](https://b.example/2)**\n"
            "Body two.\n"
        )
        assert check_near_duplicates(md) == []

    def test_agrees_with_text_similarity_by_construction(self):
        # Don't just assert the check fires/is silent on hand-picked pairs --
        # assert it agrees with scripts.text_similarity.headlines_match,
        # which is the actual shared measure the pipeline also uses. If
        # check_near_duplicates ever reverts to a private tokenizer or
        # threshold, this is the test that catches the drift.
        matching_pair = (
            "Darth Vader Speaks At City Council Meeting",
            "City Council Meeting Hears From Darth Vader",
        )
        distinct_pair = (
            "Pacific Beach Parking Rates Rise",
            "Mission Bay Park Plan Advances",
        )
        assert headlines_match(*matching_pair) is True
        assert headlines_match(*distinct_pair) is False

        def _fires(pair):
            md = (
                f"**[{pair[0]}](https://a.example/1)**\nBody one.\n\n"
                f"**[{pair[1]}](https://b.example/2)**\nBody two.\n"
            )
            return bool(check_near_duplicates(md))

        assert _fires(matching_pair) == headlines_match(*matching_pair)
        assert _fires(distinct_pair) == headlines_match(*distinct_pair)


# ---------------------------------------------------------------------------
# check_placeholder_text
# ---------------------------------------------------------------------------


class TestCheckPlaceholderText:
    def test_fires_on_each_placeholder_pattern(self):
        md = (
            "Contact us at your-email@example.org or set ALERT_CONTACT=YOUR_NAME.\n"
            "Sender fallback is your-sender@example.com.\n"
        )
        findings = check_placeholder_text(md)
        patterns = {f.detail["pattern"] for f in findings}
        assert patterns == {"your-email@", "YOUR_NAME", "example.com", "your-sender@"}
        assert all(f.severity == WARN and f.code == "placeholder-text" for f in findings)

    def test_silent_on_clean_copy(self):
        md = "Contact the newsroom at tips@voiceofsandiego.org.\n"
        assert check_placeholder_text(md) == []


# ---------------------------------------------------------------------------
# check_degraded_content
# ---------------------------------------------------------------------------

# Verbatim from production: the entire Executive Summary of the Aug 26
# briefing, after the heavy-tier LLM timed out across every backend. The
# fallback firing is correct (graceful degradation is a hard requirement);
# nothing noticing was the defect -- status.json still said errors: [] and
# the run was declared a success. See CLAUDE.md and
# references/quality_monitoring_design.md.
EXEC_SUMMARY_PLACEHOLDER_FIXTURE = (
    "*Synthesis unavailable for today's briefing. Please see the individual "
    "sections below for key updates in tech, defense, and research.*"
)

DEGRADED_MIN_CONFIG = {
    "section_headings": {"executive_summary": "Executive Summary", "news": "AI & Tech News"}
}


def _section_markdown(heading, body):
    return f"## {heading}\n\n{body}\n"


class TestCheckDegradedContent:
    def test_real_executive_summary_placeholder_is_critical(self):
        # The verbatim fixture trips two default markers at once
        # ("synthesis unavailable" and "unavailable for today's briefing")
        # -- that is still one degraded section, so exactly one finding,
        # with both matched markers recorded for debugging.
        md = _section_markdown("Executive Summary", EXEC_SUMMARY_PLACEHOLDER_FIXTURE)
        findings = check_degraded_content(md, DEGRADED_MIN_CONFIG)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.severity == CRITICAL
        assert finding.code == "degraded-content"
        assert "Executive Summary" in finding.message
        assert "Synthesis unavailable for today's briefing" in finding.message
        assert set(finding.detail["markers"]) == {
            "synthesis unavailable",
            "unavailable for today's briefing",
        }

    def test_same_placeholder_in_non_executive_section_is_warn(self):
        md = _section_markdown("AI & Tech News", EXEC_SUMMARY_PLACEHOLDER_FIXTURE)
        findings = check_degraded_content(md, DEGRADED_MIN_CONFIG)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.severity == WARN
        assert finding.code == "degraded-content"
        assert "AI & Tech News" in finding.message

    def test_three_overlapping_markers_still_one_finding(self):
        # A section whose text happens to trip three configured markers at
        # once is one degraded section, not three findings -- the count is
        # what a human reads first in the digest headline, and it must not
        # inflate by however many synonyms happen to overlap.
        md = _section_markdown(
            "Executive Summary",
            "Synthesis unavailable for today's briefing: summary unavailable "
            "and the model was unable to generate a lead section.",
        )
        findings = check_degraded_content(md, DEGRADED_MIN_CONFIG)
        assert len(findings) == 1
        assert set(findings[0].detail["markers"]) == {
            "synthesis unavailable",
            "unavailable for today's briefing",
            "summary unavailable",
            "unable to generate",
        }
        assert findings[0].severity == CRITICAL

    def test_two_different_degraded_sections_are_two_findings(self):
        # Two genuinely different degraded sections is still two findings,
        # one each, with the right severity per section.
        md = (
            _section_markdown("Executive Summary", EXEC_SUMMARY_PLACEHOLDER_FIXTURE)
            + _section_markdown("AI & Tech News", "Summary unavailable for this section today.")
        )
        findings = check_degraded_content(md, DEGRADED_MIN_CONFIG)
        assert len(findings) == 2
        by_section = {f.source: f for f in findings}
        assert set(by_section) == {"Executive Summary", "AI & Tech News"}
        assert by_section["Executive Summary"].severity == CRITICAL
        assert by_section["AI & Tech News"].severity == WARN

    def test_silent_on_clean_briefing(self):
        md = _section_markdown(
            "Executive Summary",
            "Two council votes land back to back this week, and a new "
            "funding round reshapes the local defense-tech landscape.",
        )
        assert check_degraded_content(md, DEGRADED_MIN_CONFIG) == []

    def test_silent_on_full_clean_briefing_fixture(self):
        # The same multi-section clean fixture check_report's own
        # clean-report guarantee is built on -- degraded-content adds no
        # findings to it either.
        assert check_degraded_content(CLEAN_MARKDOWN, CLEAN_CONFIG) == []

    def test_custom_markers_from_config_are_honored(self):
        md = _section_markdown("Executive Summary", "*Editorial desk offline for today.*")
        config = {"quality_check": {"degraded_markers": ["editorial desk offline"]}}
        findings = check_degraded_content(md, config)
        assert len(findings) == 1
        assert findings[0].severity == CRITICAL
        assert findings[0].detail["markers"] == ["editorial desk offline"]

    def test_built_in_defaults_apply_when_key_absent(self):
        md = _section_markdown("Executive Summary", "Summary unavailable due to an upstream error.")
        findings = check_degraded_content(md, {})
        assert len(findings) == 1
        assert findings[0].detail["markers"] == ["summary unavailable"]

    def test_explicit_empty_marker_list_disables_the_check(self):
        md = _section_markdown("Executive Summary", EXEC_SUMMARY_PLACEHOLDER_FIXTURE)
        config = {"quality_check": {"degraded_markers": []}}
        assert check_degraded_content(md, config) == []

    def test_case_insensitive_matching(self):
        md = _section_markdown(
            "Executive Summary", "SYNTHESIS UNAVAILABLE today -- see below."
        )
        findings = check_degraded_content(md, DEGRADED_MIN_CONFIG)
        assert len(findings) == 1
        assert findings[0].severity == CRITICAL
        assert findings[0].detail["markers"] == ["synthesis unavailable"]

    def test_news_item_merely_mentioning_unavailable_service_is_not_critical(self):
        # A real outage story legitimately uses phrasing that overlaps a
        # degradation marker ("unable to generate") without the briefing
        # itself having degraded. Scoped to the news section, this is at
        # worst a WARN, never a false CRITICAL.
        md = _section_markdown(
            "AI & Tech News",
            "**[Billing Outage Hits Regional Cloud Host](https://example.org/outage)**\n"
            "The provider's billing dashboard was unable to generate invoices "
            "for several hours Tuesday after a database failover, the company "
            "said in a status update.",
        )
        findings = check_degraded_content(md, DEGRADED_MIN_CONFIG)
        assert all(f.severity != CRITICAL for f in findings)
        if findings:
            assert findings[0].severity == WARN

    def test_usage_summary_appendix_is_skipped_entirely(self):
        md = (
            "## Gemini Usage Summary\n\n"
            "Synthesis unavailable calls: 1 failed attempt on the heavy tier.\n"
        )
        assert check_degraded_content(md, DEGRADED_MIN_CONFIG) == []

    def test_api_key_rotation_summary_appendix_is_skipped(self):
        md = (
            "## API Key Rotation Summary\n\n"
            "Key 2 unable to generate a response after 3 attempts.\n"
        )
        assert check_degraded_content(md, DEGRADED_MIN_CONFIG) == []


# ---------------------------------------------------------------------------
# check_thin_sections
# ---------------------------------------------------------------------------


class TestCheckThinSections:
    def test_fires_below_configured_floor(self):
        config = {
            "section_headings": {"news": "News"},
            "quality_check": {"section_floors": {"news": 3}},
        }
        md = (
            "## News\n\n"
            "**[Item One](https://a.example/1)**\nBody.\n\n"
            "**[Item Two](https://a.example/2)**\nBody.\n"
        )
        findings = check_thin_sections(md, config)
        assert len(findings) == 1
        assert findings[0].code == "thin-section"
        assert findings[0].severity == INFO
        assert findings[0].detail == {"count": 2, "floor": 3}

    def test_silent_when_floor_is_met(self):
        config = {
            "section_headings": {"news": "News"},
            "quality_check": {"section_floors": {"news": 2}},
        }
        md = (
            "## News\n\n"
            "**[Item One](https://a.example/1)**\nBody.\n\n"
            "**[Item Two](https://a.example/2)**\nBody.\n"
        )
        assert check_thin_sections(md, config) == []

    def test_no_op_with_no_config(self):
        md = "## News\n\n**[Item](https://a.example/1)**\nBody.\n"
        assert check_thin_sections(md, {}) == []


# ---------------------------------------------------------------------------
# Blog item shape -- **[Title](url)** *(Source)* ★-rating, and the link-less
# **Title** *(Source)* ★-rating variant _render_blogs also emits.
#
# Regression coverage: _ITEM_RE used to require a bare newline right after
# the closing "**", but every blog item has a source/star-rating trailer on
# that same line, so every blog item silently failed to match. Because
# _iter_rendered_items backs check_blocked_sources, check_out_of_area, and
# check_near_duplicates too, that one regex bug made the entire blogs
# section invisible to all of them, not just check_thin_sections -- a
# press-release URL or an out-of-area item in blogs would have sailed
# through with no finding at all.
# ---------------------------------------------------------------------------


class TestBlogItemShape:
    def test_linked_blog_item_with_source_and_stars_is_extracted_intact(self):
        md = (
            "## Local Sources & Analysis\n\n"
            "**[All the Midway Rising You Can Handle]"
            "(https://voiceofsandiego.org/2026/08/24/all-the-midway-rising-you-can-handle/)** "
            "*(Voice of San Diego)* ★★★★☆\n"
            "A roundup of everything happening around the Midway District redevelopment.\n"
        )
        items = list(report_invariants._iter_rendered_items(md))
        assert len(items) == 1
        title, url, body = items[0]
        assert title == "All the Midway Rising You Can Handle"
        assert url == "https://voiceofsandiego.org/2026/08/24/all-the-midway-rising-you-can-handle/"
        assert body == "A roundup of everything happening around the Midway District redevelopment."

    def test_linkless_blog_item_is_still_counted(self):
        # _render_blogs can emit "**Title** *(Source)*" with no URL at all
        # -- but always with the source trailer, link or no link. Decision:
        # a link-less item counts as a rendered item for thin-section
        # purposes (url == "") *only* when that trailer is present;
        # blocked-source / out-of-area then naturally judge it on
        # title+summary text alone, same as any other item with no host
        # would be judged.
        md = (
            "## Local Sources & Analysis\n\n"
            "**A Local Blog Post With No Link** *(Some Blog)* ★★★☆☆\n"
            "Body text with no URL attached to the headline at all.\n"
        )
        items = list(report_invariants._iter_rendered_items(md))
        assert len(items) == 1
        title, url, body = items[0]
        assert title == "A Local Blog Post With No Link"
        assert url == ""
        assert body == "Body text with no URL attached to the headline at all."

    def test_bare_bold_line_with_no_source_trailer_is_not_an_item(self):
        # Regression guard: a bare "**Bold Line**\n" with no source trailer
        # is common outside the list sections (e.g. the Gemini usage
        # summary's "**Model fallback activity:**", or a markdown table row
        # like "| **Total** | ... |") and must never be miscounted as a
        # rendered item -- that was the shape that produced false
        # out-of-area findings on pipeline boilerplate the first time this
        # was loosened.
        md = (
            "## Local Sources & Analysis\n\n"
            "**[Real Blog Item](https://voiceofsandiego.org/x)** *(Voice of San Diego)* ★★★★☆\n"
            "A genuine blog summary.\n\n"
            "**Model fallback activity:**\n\n"
            "- **medium**: served by a fallback model\n"
        )
        items = list(report_invariants._iter_rendered_items(md))
        assert len(items) == 1
        assert items[0][0] == "Real Blog Item"

    def test_inline_bold_prose_is_not_mistaken_for_an_item(self):
        # A line of prose that happens to *start* with bold text and
        # continues in plain prose (not a "*(Source)*" trailer, not an
        # immediate newline) must not be mistaken for an item head.
        md = (
            "## Executive Summary\n\n"
            "**Watch:** the certification agenda date and the **Flock** "
            "camera vote -- two Council calls back to back.\n"
        )
        assert list(report_invariants._iter_rendered_items(md)) == []

    @staticmethod
    def _blog_item(title, url, source="Voice of San Diego", stars="★★★☆☆", body=None):
        body = body or "Body text mentioning San Diego and Pacific Beach directly."
        return f"**[{title}]({url})** *({source})* {stars}\n{body}\n"

    def test_three_blog_items_do_not_trip_thin_section_floor_of_three(self):
        config = {
            "section_headings": {"blogs": "Local Sources & Analysis"},
            "quality_check": {"section_floors": {"blogs": 3}},
        }
        md = "## Local Sources & Analysis\n\n" + "\n".join(
            self._blog_item(f"Story {i}", f"https://voiceofsandiego.org/story-{i}")
            for i in range(3)
        )
        findings = check_thin_sections(md, config)
        assert findings == []

    def test_blocked_source_inside_blog_item_shape_is_flagged(self):
        config = {"geo_filter": {"blocked_sources": ["openpr.com"]}}
        md = "## Local Sources & Analysis\n\n" + self._blog_item(
            "Seaside Pizza Co. Adds Beer and Wine to Its Pacific Beach Pizza Takeout Experience",
            "https://openpr.com/news/4612028/seaside",
            source="Open PR",
        )
        findings = check_blocked_sources(md, config)
        assert len(findings) == 1
        assert findings[0].code == "blocked-source"
        assert findings[0].source == "openpr.com"

    def test_out_of_area_blog_item_shape_is_flagged(self):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["san diego", "pacific beach"],
                "trusted_sources": ["voiceofsandiego.org"],
            }
        }
        md = "## Local Sources & Analysis\n\n" + self._blog_item(
            "National Chain Reports Quarterly Earnings",
            "https://example.org/national",
            source="Wire Service",
            body="A national retail chain announced quarterly earnings today.",
        )
        findings = check_out_of_area(md, config)
        assert len(findings) == 1
        assert findings[0].code == "out-of-area"

    def test_news_and_happenings_shapes_still_parse_identically(self):
        # No trailer at all -- the pre-existing, still-primary shape.
        md = (
            "**[Little Italy Farmers Market Returns This Wednesday]"
            "(https://voiceofsandiego.org/events/little-italy)**\n"
            "The weekday Little Italy farmer's market runs Wednesday, Aug. 26.\n"
        )
        items = list(report_invariants._iter_rendered_items(md))
        assert len(items) == 1
        title, url, body = items[0]
        assert title == "Little Italy Farmers Market Returns This Wednesday"
        assert url == "https://voiceofsandiego.org/events/little-italy"
        assert body == "The weekday Little Italy farmer's market runs Wednesday, Aug. 26."


# ---------------------------------------------------------------------------
# check_report end to end -- a realistic multi-section briefing, modeled on
# config_local.yaml, and a clean-report guarantee.
# ---------------------------------------------------------------------------

CLEAN_CONFIG = {
    "section_order": ["alerts", "happenings", "news", "blogs"],
    "section_headings": {
        "executive_summary": "Executive Summary",
        "alerts": "Active Alerts",
        "happenings": HAPPENINGS_HEADING,
        "news": "Around You — What Changes This Week",
        "blogs": "Local Sources & Analysis",
        "errors": "Errors",
    },
    "geo_filter": {
        "enabled": True,
        "place_terms": [
            "san diego", "pacific beach", "mission beach", "mission bay", "crown point",
        ],
        "trusted_sources": ["voiceofsandiego.org", "timesofsandiego.com", "kpbs.org"],
        "blocked_sources": ["openpr.com"],
    },
    "news_similarity_dedup": {"enabled": True, "threshold": 0.3},
    "quality_check": {"section_floors": {"news": 2, "happenings": 1}},
}

CLEAN_MARKDOWN = f"""# San Diego Local News Briefing - August 25, 2026

## Executive Summary

Beach parking rates are changing before Labor Day, Mission Bay's park plan
heads to council next month, and Little Italy's midweek market runs on its
usual schedule.

## Active Alerts

**Coastal Flood Advisory** — Minor
Tue Aug 25, 5:00 AM -> Tue Aug 25, 11:00 AM
San Diego County Coastal Areas

Minor tidal flooding is possible along low-lying shoreline roads during the
Tuesday morning high tide.

## {HAPPENINGS_HEADING}

**[Little Italy Farmers Market Returns This Wednesday](https://voiceofsandiego.org/events/little-italy)**
The weekday Little Italy farmer's market runs Wednesday, Aug. 26, 9:30 a.m.-1:30 p.m., a short drive from Pacific Beach, on three blocks of West Date Street.

**[Crown Point Sunset Yoga Meetup](https://timesofsandiego.com/events/crown-point-yoga)**
Join neighbors for sunset yoga at Crown Point Sunday, Aug. 30, 6:00 p.m.-7:00 p.m., a short walk from the Mission Bay bike path.

## Around You — What Changes This Week

**[Pacific Beach Parking Meters Get New Rates Starting September](https://voiceofsandiego.org/2026/08/24/pb-parking-rates)**
The city council approved new parking meter rates for Pacific Beach starting in September, part of a broader push to fund beach lifeguard staffing.

**[Mission Bay Park Master Plan Update Heads to Council](https://timesofsandiego.com/2026/08/24/mission-bay-plan)**
San Diego's Mission Bay park master plan update goes before city council next month, addressing wetland restoration and bike path expansion.

## Local Sources & Analysis

**[The Weekly San Diego Housing Roundup](https://voiceofsandiego.org/2026/08/24/housing-roundup)**
A closer look at San Diego's latest housing permit data and what it means for renters across the city.
"""


class TestCheckReport:
    def test_clean_report_produces_zero_findings(self):
        findings = check_report(
            CLEAN_MARKDOWN,
            CLEAN_CONFIG,
            today=date(2026, 8, 25),
            sections_with_data=["alerts", "happenings", "news", "blogs"],
        )
        assert findings == []

    def test_catches_real_world_defects_together(self):
        dirty_md = CLEAN_MARKDOWN.replace(
            # Swap in the verbatim stale-weekend fixture.
            "**[Little Italy Farmers Market Returns This Wednesday](https://voiceofsandiego.org/events/little-italy)**\n"
            "The weekday Little Italy farmer's market runs Wednesday, Aug. 26, 9:30 a.m.-1:30 p.m., a short drive from Pacific Beach, on three blocks of West Date Street.",
            STALE_WEEKEND_FIXTURE,
        ).replace(
            # Leak a filtering rationale into the news section.
            "The city council approved new parking meter rates for Pacific Beach starting in September, part of a broader push to fund beach lifeguard staffing.",
            "The city council approved new parking meter rates for Pacific Beach. "
            "Dropped: [2], [5] (duplicate coverage of the same vote).",
        ).replace(
            # Slip in a blocked pay-to-publish source in the blogs section.
            "**[The Weekly San Diego Housing Roundup](https://voiceofsandiego.org/2026/08/24/housing-roundup)**\n"
            "A closer look at San Diego's latest housing permit data and what it means for renters across the city.",
            BLOCKED_SOURCE_FIXTURE,
        )

        findings = check_report(
            dirty_md,
            CLEAN_CONFIG,
            today=date(2026, 8, 25),
            sections_with_data=["alerts", "happenings", "news", "blogs"],
        )
        codes = {f.code for f in findings}
        assert "stale-event" in codes
        assert "scaffolding-leak" in codes
        assert "blocked-source" in codes

    def test_section_missing_and_reordered_together(self):
        # happenings never rendered at all despite having data, and news
        # renders before alerts.
        md = (
            "## Around You — What Changes This Week\n\n"
            "**[Pacific Beach Item](https://voiceofsandiego.org/x)**\nBody.\n\n"
            "## Active Alerts\n\n"
            "**Coastal Flood Advisory**\nWindow text.\n"
        )
        findings = check_report(
            md,
            CLEAN_CONFIG,
            today=date(2026, 8, 25),
            sections_with_data=["alerts", "happenings", "news"],
        )
        codes = {f.code for f in findings}
        assert "section-missing" in codes
        assert "section-order" in codes
