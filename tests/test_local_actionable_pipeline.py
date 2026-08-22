# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Tests for the geo-aware local briefing pipeline.

Covers the pieces added to make the local report actionable: per-branch
freshness windows on news queries, the geographic relevance gate, cross-outlet
duplicate collapse, and the deterministic alerts section.
"""

from unittest.mock import MagicMock, patch

import pytest

from scripts.briefing_runner import BriefingRunner
from scripts.interest_graph import Query


@pytest.fixture
def local_config():
    return {
        "arxiv_topics": [],
        "blog_feeds": [],
        "stocks": [],
        "news_queries": [],
        "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
        "num_paper_picks": 2,
        "max_papers": 5,
        "max_blogs": 5,
        "max_news": 15,
        "arxiv_days_back": 3,
        "output_format": "kindle",
        "file_naming": "Local-{yyyy}.{mm}.{dd}",
        "pdf": {"enabled": False},
        "bedrock": {"enabled": False},
        "gemini": {"enabled": False},
        "news_freshness": "pd",
        "geo_filter": {
            "enabled": True,
            "place_terms": ["san diego", "pacific beach"],
            "trusted_sources": ["voiceofsandiego.org"],
        },
        "news_similarity_dedup": {"enabled": True, "threshold": 0.3},
        "alerts": {
            "enabled": True,
            "provider": "nws",
            "zones": ["CAZ043"],
            "max_alerts": 4,
        },
        "section_order": ["alerts", "news"],
        "section_headings": {"alerts": "Active Alerts", "news": "Around You"},
    }


@pytest.fixture
def runner(local_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return BriefingRunner(local_config, dry_run=True)


# ---------- per-branch freshness windows ----------


class TestNewsFreshnessGrouping:
    def _aggregator_calls(self, runner, queries, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        made = []

        def fake_aggregator(api_key, queries, max_results, freshness):
            made.append({"queries": list(queries), "freshness": freshness})
            instance = MagicMock()
            instance.aggregate_all_queries.return_value = [
                {"title": f"San Diego story for {q}", "url": f"https://x/{q}"}
                for q in queries
            ]
            return instance

        with patch("scripts.briefing_runner.NewsAggregator", side_effect=fake_aggregator):
            articles = runner.run_news_aggregation(queries=queries)
        return made, articles

    def test_one_group_per_freshness_window(self, runner, monkeypatch):
        queries = [
            Query("Pacific Beach San Diego", freshness="pw"),
            Query("Crown Point San Diego", freshness="pw"),
            Query("San Diego housing market"),
        ]
        made, articles = self._aggregator_calls(runner, queries, monkeypatch)
        by_freshness = {call["freshness"]: call["queries"] for call in made}
        assert set(by_freshness) == {"pw", "pd"}
        assert by_freshness["pw"] == ["Pacific Beach San Diego", "Crown Point San Diego"]
        assert by_freshness["pd"] == ["San Diego housing market"]
        assert len(articles) == 3

    def test_plain_strings_use_the_configured_default(self, runner, monkeypatch):
        made, _ = self._aggregator_calls(runner, ["San Diego city council"], monkeypatch)
        assert made[0]["freshness"] == "pd"

    def test_duplicate_urls_collapse_across_groups(self, runner, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        instance = MagicMock()
        instance.aggregate_all_queries.return_value = [
            {"title": "Same San Diego story", "url": "https://example.com/a"}
        ]
        with patch("scripts.briefing_runner.NewsAggregator", return_value=instance):
            articles = runner.run_news_aggregation(
                queries=[Query("a", freshness="pw"), Query("b", freshness="pd")]
            )
        assert len(articles) == 1

    def test_no_api_key_returns_empty(self, runner, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        assert runner.run_news_aggregation(queries=["x"]) == []


# ---------- geographic relevance gate ----------


class TestGeoFilterWiring:
    def test_drops_out_of_area_and_counts_it(self, runner):
        items = [
            {"title": "San Diego beach parking tickets"},
            {"title": "Bay Area drivers face major delays on Hwy 101"},
            {"title": "Indiana residents endure ninth day without power"},
        ]
        kept = runner._apply_geo_filter(items, "news")
        assert [i["title"] for i in kept] == ["San Diego beach parking tickets"]
        assert runner.status["geo_filtered_out"] == 2

    def test_trusted_local_outlet_survives(self, runner):
        items = [{"title": "Amazon Goes to the Beach", "source": "voiceofsandiego.org"}]
        assert runner._apply_geo_filter(items, "news") == items

    def test_disabled_filter_is_a_noop(self, local_config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local_config["geo_filter"]["enabled"] = False
        runner = BriefingRunner(local_config, dry_run=True)
        items = [{"title": "Indiana outage"}]
        assert runner._apply_geo_filter(items, "news") == items
        assert runner.status["geo_filtered_out"] == 0


# ---------- cross-outlet duplicate collapse ----------


class TestSimilarNewsDedup:
    VIRAL = [
        {"title": "Darth Vader to San Diego City Council: The Emperor liked Flock cameras",
         "source": "latimes.com"},
        {"title": "'Darth Vader' comes out in favor of Flock cameras at San Diego city council meeting",
         "source": "theguardian.com"},
        {"title": "Man in Darth Vader costume jokingly supports Flock funding at San Diego City Council meeting",
         "source": "wfla.com"},
    ]

    def test_collapses_syndicated_retellings(self, runner):
        assert len(runner.deduplicate_similar_news(self.VIRAL)) == 1

    def test_keeps_distinct_local_stories(self, runner):
        distinct = [
            {"title": "South Mission Beach volleyball can now begin at 6 a.m."},
            {"title": "Mission Bay bathrooms set to close in December unless council finds funding"},
            {"title": "Carlsbad's skepticism remains over Oceanside's beach sand project"},
        ]
        assert len(runner.deduplicate_similar_news(distinct)) == 3

    def test_prefers_the_local_outlet_copy(self, runner):
        news = list(self.VIRAL) + [
            {"title": "Darth Vader tells San Diego City Council the Emperor likes Flock cameras",
             "source": "voiceofsandiego.org"}
        ]
        kept = runner.deduplicate_similar_news(news)
        assert len(kept) == 1
        assert kept[0]["source"] == "voiceofsandiego.org"

    def test_disabled_by_default(self, local_config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        del local_config["news_similarity_dedup"]
        runner = BriefingRunner(local_config, dry_run=True)
        assert len(runner.deduplicate_similar_news(self.VIRAL)) == 3

    def test_untitled_items_are_kept(self, runner):
        assert len(runner.deduplicate_similar_news([{"title": ""}, {"title": ""}])) == 2


# ---------- alerts section ----------


class TestAlertsSection:
    ALERT = {
        "event": "Extreme Heat Warning",
        "severity": "Severe",
        "area": "San Diego County Coastal Areas",
        "onset": "2026-08-21T21:35:00-07:00",
        "expires": "2026-08-28T20:00:00-07:00",
        "instruction": "Drink plenty of fluids and check on relatives and neighbors.",
        "source": "National Weather Service",
    }

    def test_scan_returns_alerts_and_counts_them(self, runner):
        scanner = MagicMock()
        scanner.fetch.return_value = [self.ALERT]
        with patch("scripts.briefing_runner.create_alerts_scanner", return_value=scanner):
            alerts = runner.run_alerts_scan()
        assert alerts == [self.ALERT]
        assert runner.status["alerts_found"] == 1

    def test_scan_disabled_returns_empty(self, runner):
        with patch("scripts.briefing_runner.create_alerts_scanner", return_value=None):
            assert runner.run_alerts_scan() == []

    def test_fetch_failure_is_recorded_not_raised(self, runner):
        scanner = MagicMock()
        scanner.fetch.side_effect = RuntimeError("weather.gov down")
        with patch("scripts.briefing_runner.create_alerts_scanner", return_value=scanner):
            assert runner.run_alerts_scan() == []
        assert any("Alerts scan" in e for e in runner.errors)

    def test_renders_event_window_area_and_instruction(self, runner):
        md = runner._render_alerts([self.ALERT])
        assert "## Active Alerts" in md
        assert "**Extreme Heat Warning** — Severe" in md
        assert "→" in md
        assert "San Diego County Coastal Areas" in md
        assert "Drink plenty of fluids" in md
        assert "National Weather Service" in md

    def test_renders_nothing_when_empty(self, runner):
        assert runner._render_alerts([]) == ""

    def test_respects_max_alerts(self, runner):
        alerts = [dict(self.ALERT, event=f"Alert {i}") for i in range(6)]
        md = runner._render_alerts(alerts)
        assert md.count("**Alert ") == 4

    def test_long_instruction_is_truncated(self, runner):
        alert = dict(self.ALERT, instruction="x" * 500)
        assert "..." in runner._render_alerts([alert])

    def test_unparseable_timestamps_render_verbatim(self, runner):
        alert = dict(self.ALERT, onset="soon", expires="")
        assert "soon" in runner._render_alerts([alert])

    def test_alerts_lead_the_briefing_when_first_in_section_order(self, runner):
        md = runner.generate_markdown_briefing(
            papers=[], blogs=[], stocks=[],
            news=[{"title": "San Diego council votes", "url": "https://x/1"}],
            top_papers=[], alerts=[self.ALERT],
        )
        assert md.index("## Active Alerts") < md.index("## Around You")

    def test_alerts_render_without_the_llm(self, runner):
        """Alerts are deterministic: no intelligence layer involved."""
        assert runner.intelligence.available is False
        md = runner.generate_markdown_briefing(
            papers=[], blogs=[], stocks=[], news=[], top_papers=[], alerts=[self.ALERT],
        )
        assert "Extreme Heat Warning" in md
