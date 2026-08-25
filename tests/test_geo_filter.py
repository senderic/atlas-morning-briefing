# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for the geographic relevance gate (scripts/geo_filter.py)."""

import logging

from scripts.geo_filter import apply_config_filter, filter_by_place, is_local


TERMS = ["san diego", "pacific beach", "california"]


class TestIsLocal:
    def test_matches_place_term_in_title(self):
        assert is_local({"title": "Mission Bay bathrooms close"}, ["mission bay"])

    def test_matches_case_insensitively(self):
        assert is_local({"title": "SAN DIEGO council vote"}, TERMS)

    def test_matches_in_description(self):
        item = {"title": "Council votes on cameras", "description": "The San Diego council..."}
        assert is_local(item, TERMS)

    def test_rejects_out_of_area(self):
        item = {"title": "Bay Area drivers face delays on Hwy 101", "description": "Carquinez"}
        assert not is_local(item, TERMS)

    def test_trusted_source_passes_without_place_name(self):
        item = {"title": "Amazon Goes to the Beach", "source": "voiceofsandiego.org"}
        assert not is_local(item, TERMS)
        assert is_local(item, TERMS, trusted_sources=["voiceofsandiego.org"])

    def test_trusted_source_matches_subdomain(self):
        item = {"title": "Local story", "source": "www.kpbs.org"}
        assert is_local(item, TERMS, trusted_sources=["kpbs.org"])

    def test_falls_back_to_url_host(self):
        item = {"title": "Local story", "url": "https://obrag.org/2026/08/post/"}
        assert is_local(item, TERMS, trusted_sources=["obrag.org"])

    def test_blocked_source_rejected_despite_place_term(self):
        item = {
            "title": (
                "Seaside Pizza Co. Adds Beer and Wine to Its Pacific Beach "
                "Pizza Takeout Experience"
            ),
            "source": "openpr.com",
        }
        assert not is_local(item, TERMS, blocked_sources=["openpr.com"])

    def test_blocked_source_matches_subdomain(self):
        item = {"title": "Pacific Beach news", "source": "www.openpr.com"}
        assert not is_local(item, TERMS, blocked_sources=["openpr.com"])

    def test_blocked_beats_trusted(self):
        item = {"title": "Pacific Beach news", "source": "openpr.com"}
        assert not is_local(
            item,
            TERMS,
            trusted_sources=["openpr.com"],
            blocked_sources=["openpr.com"],
        )

    def test_blocked_source_resolved_from_url(self):
        item = {
            "title": "Pacific Beach pizza shop press release",
            "url": "https://www.openpr.com/news/12345/seaside-pizza.html",
        }
        assert not is_local(item, TERMS, blocked_sources=["openpr.com"])

    def test_non_blocked_source_still_passes(self):
        item = {"title": "Pacific Beach street vendors return", "source": "voiceofsandiego.org"}
        assert is_local(item, TERMS, blocked_sources=["openpr.com"])


class TestFilterByPlace:
    def test_splits_kept_and_dropped(self):
        items = [
            {"title": "San Diego beach parking tickets"},
            {"title": "TxDOT road closures August 22-30"},
            {"title": "Pacific Beach street vendors return"},
        ]
        kept, dropped = filter_by_place(items, TERMS)
        assert [i["title"] for i in kept] == [
            "San Diego beach parking tickets",
            "Pacific Beach street vendors return",
        ]
        assert len(dropped) == 1

    def test_no_terms_is_a_noop(self):
        items = [{"title": "anything"}]
        kept, dropped = filter_by_place(items, [])
        assert kept == items and dropped == []

    def test_blank_terms_ignored(self):
        kept, dropped = filter_by_place([{"title": "Indiana outage"}], ["  ", ""])
        assert len(kept) == 1 and not dropped

    def test_blocked_source_dropped_despite_place_term(self):
        items = [
            {
                "title": (
                    "Seaside Pizza Co. Adds Beer and Wine to Its Pacific Beach "
                    "Pizza Takeout Experience"
                ),
                "source": "openpr.com",
            },
            {"title": "Pacific Beach street vendors return", "source": "voiceofsandiego.org"},
        ]
        kept, dropped = filter_by_place(items, TERMS, blocked_sources=["openpr.com"])
        assert [i["source"] for i in kept] == ["voiceofsandiego.org"]
        assert len(dropped) == 1 and dropped[0]["source"] == "openpr.com"

    def test_blocked_sources_apply_with_no_place_terms(self):
        items = [
            {
                "title": (
                    "Seaside Pizza Co. Adds Beer and Wine to Its Pacific Beach "
                    "Pizza Takeout Experience"
                ),
                "source": "openpr.com",
            },
            {"title": "City council approves new bike lane", "source": "voiceofsandiego.org"},
        ]
        kept, dropped = filter_by_place(items, [], blocked_sources=["openpr.com"])
        assert [i["source"] for i in kept] == ["voiceofsandiego.org"]
        assert len(dropped) == 1 and dropped[0]["source"] == "openpr.com"

    def test_blocked_sources_apply_with_blank_only_place_terms(self):
        items = [
            {"title": "Pacific Beach pizza press release", "source": "openpr.com"},
            {"title": "City council approves new bike lane", "source": "voiceofsandiego.org"},
        ]
        kept, dropped = filter_by_place(items, ["  "], blocked_sources=["openpr.com"])
        assert [i["source"] for i in kept] == ["voiceofsandiego.org"]
        assert len(dropped) == 1 and dropped[0]["source"] == "openpr.com"


class TestApplyConfigFilter:
    def test_disabled_returns_everything(self):
        items = [{"title": "Indiana power outage"}]
        assert apply_config_filter(items, {"geo_filter": {"enabled": False}}) == items

    def test_absent_block_returns_everything(self):
        items = [{"title": "Indiana power outage"}]
        assert apply_config_filter(items, {}) == items

    def test_enabled_filters(self):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["san diego"],
                "trusted_sources": ["kpbs.org"],
            }
        }
        items = [
            {"title": "San Diego heat advisory"},
            {"title": "Indiana still without power"},
            {"title": "Untitled local piece", "source": "kpbs.org"},
        ]
        kept = apply_config_filter(items, config, label="news")
        assert len(kept) == 2
        assert all("Indiana" not in i["title"] for i in kept)

    def test_blocked_sources_dropped_via_config(self):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["san diego", "pacific beach"],
                "blocked_sources": ["openpr.com", "prnewswire.com"],
            }
        }
        items = [
            {
                "title": (
                    "Seaside Pizza Co. Adds Beer and Wine to Its Pacific Beach "
                    "Pizza Takeout Experience"
                ),
                "source": "openpr.com",
            },
            {"title": "San Diego heat advisory", "source": "kpbs.org"},
        ]
        kept = apply_config_filter(items, config, label="news")
        assert [i["source"] for i in kept] == ["kpbs.org"]

    def test_absent_blocked_sources_is_a_noop(self):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["pacific beach"],
                "trusted_sources": ["openpr.com"],
            }
        }
        items = [{"title": "Pacific Beach news", "source": "openpr.com"}]
        assert apply_config_filter(items, config, label="news") == items

    def test_empty_blocked_sources_is_a_noop(self):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["pacific beach"],
                "blocked_sources": [],
            }
        }
        items = [{"title": "Pacific Beach news", "source": "openpr.com"}]
        assert apply_config_filter(items, config, label="news") == items

    def test_blocked_beats_trusted_via_config(self):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["pacific beach"],
                "trusted_sources": ["openpr.com"],
                "blocked_sources": ["openpr.com"],
            }
        }
        items = [{"title": "Pacific Beach news", "source": "openpr.com"}]
        assert apply_config_filter(items, config, label="news") == []

    def test_blocked_sources_apply_with_no_place_terms_key(self):
        config = {
            "geo_filter": {
                "enabled": True,
                "blocked_sources": ["openpr.com", "prnewswire.com"],
            }
        }
        items = [
            {
                "title": (
                    "Seaside Pizza Co. Adds Beer and Wine to Its Pacific Beach "
                    "Pizza Takeout Experience"
                ),
                "source": "openpr.com",
            },
            {"title": "City council approves new bike lane", "source": "voiceofsandiego.org"},
            {"title": "Indiana still without power"},
        ]
        kept = apply_config_filter(items, config, label="news")
        assert [i["source"] for i in kept if "source" in i] == ["voiceofsandiego.org"]
        assert len(kept) == 2
    def test_blocked_drop_logged_as_blocked_not_no_local_reference(self, caplog):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["pacific beach"],
                "blocked_sources": ["openpr.com"],
            }
        }
        items = [
            {
                "title": (
                    "Seaside Pizza Co. Adds Beer and Wine to Its Pacific Beach "
                    "Pizza Takeout Experience"
                ),
                "source": "openpr.com",
            },
        ]
        with caplog.at_level(logging.INFO):
            apply_config_filter(items, config, label="news")
        messages = [r.message for r in caplog.records]
        assert any("from blocked sources" in m for m in messages)
        assert not any("no local reference" in m for m in messages)

    def test_out_of_area_drop_still_logged_as_no_local_reference(self, caplog):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["pacific beach"],
                "blocked_sources": ["openpr.com"],
            }
        }
        items = [
            {"title": "TxDOT road closures August 22-30", "source": "txdot.gov"},
        ]
        with caplog.at_level(logging.INFO):
            apply_config_filter(items, config, label="news")
        messages = [r.message for r in caplog.records]
        assert any("no local reference" in m for m in messages)
        assert not any("from blocked sources" in m for m in messages)

    def test_mixed_drops_report_both_lines_with_correct_counts(self, caplog):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["pacific beach"],
                "blocked_sources": ["openpr.com"],
            }
        }
        items = [
            {
                "title": (
                    "Seaside Pizza Co. Adds Beer and Wine to Its Pacific Beach "
                    "Pizza Takeout Experience"
                ),
                "source": "openpr.com",
            },
            {"title": "TxDOT road closures August 22-30", "source": "txdot.gov"},
            {"title": "Pacific Beach street vendors return", "source": "voiceofsandiego.org"},
        ]
        with caplog.at_level(logging.INFO):
            kept = apply_config_filter(items, config, label="news")
        assert len(kept) == 1
        messages = [r.message for r in caplog.records]
        blocked_lines = [m for m in messages if "from blocked sources" in m]
        no_local_lines = [m for m in messages if "no local reference" in m]
        assert len(blocked_lines) == 1 and "dropped 1/3" in blocked_lines[0]
        assert len(no_local_lines) == 1 and "dropped 1/3" in no_local_lines[0]

    def test_nothing_logged_when_nothing_dropped(self, caplog):
        config = {
            "geo_filter": {
                "enabled": True,
                "place_terms": ["pacific beach"],
                "blocked_sources": ["openpr.com"],
            }
        }
        items = [{"title": "Pacific Beach street vendors return", "source": "voiceofsandiego.org"}]
        with caplog.at_level(logging.INFO):
            kept = apply_config_filter(items, config, label="news")
        assert kept == items
        assert caplog.records == []
