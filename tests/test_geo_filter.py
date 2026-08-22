# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for the geographic relevance gate (scripts/geo_filter.py)."""

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
