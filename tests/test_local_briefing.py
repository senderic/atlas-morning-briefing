# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for local news (San Diego / CA) briefing config and render path.

Verifies config-driven headings, section ordering, feature gates, and
backward compatibility of defaults in the refactored BriefingRunner.
"""

import copy
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from scripts.briefing_runner import BriefingRunner


LOCAL_NEWS_CONFIG = {
    "arxiv_topics": [],
    "stocks": [],
    "blog_feeds": [
        {"name": "Voice of San Diego", "url": "https://voiceofsandiego.org/feed/"},
        {"name": "KPBS Local", "url": "https://www.kpbs.org/news/local.rss"},
        {"name": "Times of San Diego", "url": "https://timesofsandiego.com/feed/"},
        {"name": "CalMatters", "url": "https://calmatters.org/feed/"},
    ],
    "news_queries": [
        "San Diego city council decisions zoning permits",
        "San Diego housing development regulations changes",
        "San Diego business investment opportunities",
        "California state legislation housing property",
    ],
    "happenings_queries": [
        "Pacific Beach San Diego events upcoming",
        "Mission Beach San Diego upcoming events",
    ],
    "max_happenings": 4,
    "happenings_freshness": "pw",
    "interest_profile": [
        {"topic": "San Diego zoning changes", "weight": 1.0},
        {"topic": "California housing policy", "weight": 0.95},
    ],
    "briefing_profile": {
        "domain": "San Diego and California local news",
        "audience": "a San Diego resident",
        "landscape": "the San Diego landscape",
    },
    "state_file_path": ".local-state.json",
    "briefing_title": "San Diego Local News Briefing",
    "file_naming": "Local-Briefing-{yyyy}.{mm}.{dd}",
    "output_dir": "briefings/local",
    "section_order": ["happenings", "news", "blogs"],
    "section_headings": {
        "executive_summary": "Executive Summary",
        "happenings": "Pacific Beach Area — Upcoming Happenings",
        "news": "San Diego / California News",
        "blogs": "Local Sources & Analysis",
        "errors": "Errors",
    },
    "features": {
        "solo_founder_angle": False,
        "agent_cost_optimization": False,
        "weekly_deep_dive": False,
    },
    "epub": {
        "title_format": "Local Briefing - {date}",
        "author": "Atlas Local",
    },
    "snapshot": {"dir": "snapshots/local", "enabled": True},
    "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
    "num_paper_picks": 2,
    "repro_min_score": 12,
    "max_papers": 30,
    "max_blogs": 8,
    "max_news": 15,
    "arxiv_days_back": 3,
    "max_workers": 1,
    "log_level": "DEBUG",
    "pdf": {"enabled": False},
    "bedrock": {"enabled": False},
    "gemini": {"enabled": False},
    "output_format": "kindle",
}

MAIN_CONFIG = {
    "arxiv_topics": ["AI", "Defense"],
    "stocks": [],
    "blog_feeds": [],
    "news_queries": ["AI news"],
    "briefing_profile": {
        "domain": "AI research, tech, defense, and space infrastructure",
        "audience": "an AI researcher, engineer, or defense analyst",
        "landscape": "the AI, defense-tech, and space exploration landscape",
    },
    "state_file_path": ".atlas-state.json",
    "briefing_title": "Atlas Morning Briefing",
    "file_naming": "Atlas-Briefing-{yyyy}.{mm}.{dd}",
    "output_dir": "briefings",
    "section_order": ["stocks", "news", "top_papers", "blogs"],
    "section_headings": {
        "executive_summary": "Executive Summary",
        "stocks": "Financial Market Overview",
        "news": "AI & Tech News",
        "top_papers": "Top Papers",
        "blogs": "Blog Updates",
        "errors": "Errors",
    },
    "features": {
        "solo_founder_angle": True,
        "pipeline_efficiency_play": True,
        "weekly_deep_dive": True,
    },
    "extension_sections": [
        {
            "key": "solo_founder_angle",
            "heading": "Solo Founder Angle",
            "tier": "heavy",
            "persona": "You are a startup scout for a solo founder.",
            "task": "Propose ONE product.",
            "fields": ["**Product:** <one sentence>"],
            "rules": ["Must be buildable alone."],
        },
        {
            "key": "pipeline_efficiency_play",
            "heading": "Pipeline Efficiency Play",
            "tier": "heavy",
            "persona": "You advise an engineer running a cron-driven LLM pipeline.",
            "task": "Propose ONE change.",
            "fields": ["**Play:** <one sentence>"],
            "rules": ["Reference TODAY's signals."],
        },
    ],
    "epub": {
        "title_format": "Morning Briefing - {date}",
        "author": "Atlas",
    },
    "snapshot": {"dir": "snapshots", "enabled": True},
    "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
    "num_paper_picks": 3,
    "repro_min_score": 12,
    "max_papers": 30,
    "max_blogs": 12,
    "max_news": 15,
    "arxiv_days_back": 3,
    "max_workers": 1,
    "log_level": "DEBUG",
    "pdf": {"enabled": False},
    "bedrock": {"enabled": False},
    "gemini": {"enabled": False},
    "output_format": "kindle",
}


# ---------- fixtures ----------


@pytest.fixture
def local_runner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return BriefingRunner(config=LOCAL_NEWS_CONFIG, dry_run=True)


@pytest.fixture
def main_runner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return BriefingRunner(config=MAIN_CONFIG, dry_run=True)


def _sample_news(n: int = 5):
    return [
        {
            "title": f"Local News {i}",
            "url": f"https://news-site.com/local-{i}",
            "description": f"Description for local news {i}",
            "source": "Brave",
            "brief_summary": f"Summary of local news {i}",
        }
        for i in range(n)
    ]


def _sample_blogs(n: int = 4):
    return [
        {
            "title": f"Blog Post {i}",
            "link": f"https://blog-source.com/blog-{i}",
            "source": f"Source {i}",
            "summary": f"Raw summary {i}",
            "brief_summary": f"Brief summary {i}",
            "score_combined": 5 - i,
            "published": datetime.now().isoformat(),
        }
        for i in range(n)
    ]


def _sample_papers(n: int = 3):
    return [
        {
            "title": f"Paper {i}",
            "arxiv_url": f"https://arxiv.org/abs/{i}",
            "summary": f"Abstract {i}",
            "brief_summary": f"LLM summary {i}",
            "score_combined": 4,
            "authors": ["Author A", "Author B"],
            "published": "",
        }
        for i in range(n)
    ]


# ---------- unit: config-driven defaults and overrides ----------


class TestConfigDrivenDefaults:
    """Verify backward-compatible defaults and local overrides."""

    def test_state_file_path_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        minimal = {"bedrock": {"enabled": False}, "gemini": {"enabled": False}}
        runner = BriefingRunner(config=minimal, dry_run=True)
        assert runner.state_file_path == ".atlas-state.json"

    def test_state_file_path_override(self, local_runner):
        assert local_runner.state_file_path == ".local-state.json"

    def test_section_order_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        minimal = {"bedrock": {"enabled": False}, "gemini": {"enabled": False}}
        runner = BriefingRunner(config=minimal, dry_run=True)
        assert runner.section_order == ["stocks", "news", "top_papers", "blogs"]

    def test_section_order_override_local(self, local_runner):
        assert local_runner.section_order == ["happenings", "news", "blogs"]

    def test_feature_gates_default_true(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        minimal = {"bedrock": {"enabled": False}, "gemini": {"enabled": False}}
        runner = BriefingRunner(config=minimal, dry_run=True)
        assert runner.feature_weekly_deep_dive is True
        # No extension_sections declared -> none load.
        assert runner.extension_sections == []

    def test_extension_sections_loaded_for_main(self, main_runner):
        assert [s.key for s in main_runner.extension_sections] == [
            "solo_founder_angle",
            "pipeline_efficiency_play",
        ]

    def test_feature_gates_local_disabled(self, local_runner):
        assert local_runner.feature_weekly_deep_dive is False
        assert local_runner.extension_sections == []

    def test_feature_flag_disables_a_declared_section(self, tmp_path, monkeypatch):
        """features.<key>: false still switches a declared section off."""
        monkeypatch.chdir(tmp_path)
        cfg = copy.deepcopy(MAIN_CONFIG)
        cfg["features"]["pipeline_efficiency_play"] = False
        runner = BriefingRunner(config=cfg, dry_run=True)
        assert [s.key for s in runner.extension_sections] == ["solo_founder_angle"]

    def test_headings_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        minimal = {"bedrock": {"enabled": False}, "gemini": {"enabled": False}}
        runner = BriefingRunner(config=minimal, dry_run=True)
        assert runner._headings == {}

    def test_headings_local(self, local_runner):
        assert local_runner._headings == LOCAL_NEWS_CONFIG["section_headings"]


# ---------- unit: markdown generation with local config ----------


class TestLocalMarkdownRendering:
    def test_local_title(self, local_runner):
        news = _sample_news(5)
        blogs = _sample_blogs(4)
        md = local_runner.generate_markdown_briefing(
            papers=[], blogs=blogs, stocks=[], news=news,
            top_papers=[], synthesis={}, weekly_deep_dive="",
        )
        assert "# San Diego Local News Briefing" in md

    def test_local_news_heading(self, local_runner):
        news = _sample_news(5)
        blogs = _sample_blogs(4)
        md = local_runner.generate_markdown_briefing(
            papers=[], blogs=blogs, stocks=[], news=news,
            top_papers=[],
        )
        assert "## San Diego / California News" in md
        assert "## AI & Tech News" not in md

    def test_local_blogs_heading(self, local_runner):
        news = _sample_news(5)
        blogs = _sample_blogs(4)
        md = local_runner.generate_markdown_briefing(
            papers=[], blogs=blogs, stocks=[], news=news,
            top_papers=[],
        )
        assert "## Local Sources & Analysis" in md
        assert "## Blog Updates" not in md

    def test_local_section_order_respected(self, local_runner):
        news = _sample_news(5)
        blogs = _sample_blogs(4)
        md = local_runner.generate_markdown_briefing(
            papers=[], blogs=blogs, stocks=[], news=news,
            top_papers=[],
        )
        news_idx = md.index("San Diego / California News")
        blogs_idx = md.index("Local Sources & Analysis")
        assert news_idx < blogs_idx

    def test_local_news_only_when_blogs_empty(self, local_runner):
        news = _sample_news(5)
        md = local_runner.generate_markdown_briefing(
            papers=[], blogs=[], stocks=[], news=news,
            top_papers=[],
        )
        assert "San Diego / California News" in md
        assert "Local Sources & Analysis" not in md

    def test_local_extension_sections_not_rendered(self, local_runner):
        news = _sample_news(5)
        blogs = _sample_blogs(4)
        synthesis = {
            "solo_founder_angle": "A solo startup idea",
            "pipeline_efficiency_play": "A pipeline play",
        }
        md = local_runner.generate_markdown_briefing(
            papers=[], blogs=blogs, stocks=[], news=news,
            top_papers=[], synthesis=synthesis,
        )
        assert "Solo Founder Angle" not in md
        assert "Pipeline Efficiency Play" not in md

    def test_main_config_renders_extension_sections(self, main_runner):
        news = _sample_news(5)
        blogs = _sample_blogs(4)
        papers = _sample_papers(3)
        synthesis = {
            "solo_founder_angle": "A solo startup angle",
            "pipeline_efficiency_play": "A pipeline play",
        }
        md = main_runner.generate_markdown_briefing(
            papers=papers, blogs=blogs, stocks=[], news=news,
            top_papers=papers, synthesis=synthesis,
        )
        assert "## Solo Founder Angle" in md
        assert "## Pipeline Efficiency Play" in md
        # Declaration order is render order.
        assert md.index("Solo Founder Angle") < md.index("Pipeline Efficiency Play")

    def test_local_news_renders_articles(self, local_runner):
        news = _sample_news(3)
        md = local_runner.generate_markdown_briefing(
            papers=[], blogs=[], stocks=[], news=news, top_papers=[],
        )
        for i in range(3):
            assert f"Local News {i}" in md
            assert f"Summary of local news {i}" in md
            assert f"https://news-site.com/local-{i}" in md

    def test_local_blogs_renders_with_scores(self, local_runner):
        news = _sample_news(1)
        blogs = _sample_blogs(4)
        md = local_runner.generate_markdown_briefing(
            papers=[], blogs=blogs, stocks=[], news=news, top_papers=[],
        )
        for i in range(3):  # top 3 with score >= 3
            assert f"Blog Post {i}" in md

    def test_local_blogs_no_scores_fallback(self, local_runner):
        news = _sample_news(1)
        blogs = [
            {"title": f"Blog {i}", "link": f"https://example.com/{i}",
             "source": f"Src{i}", "summary": f"Raw {i}"}
            for i in range(6)
        ]
        md = local_runner.generate_markdown_briefing(
            papers=[], blogs=blogs, stocks=[], news=news, top_papers=[],
        )
        assert "Blog 0" in md
        assert "Blog 4" in md

    def test_papers_not_in_local_section_order(self, local_runner):
        papers = _sample_papers(3)
        news = _sample_news(3)
        blogs = _sample_blogs(3)
        md = local_runner.generate_markdown_briefing(
            papers=papers, blogs=blogs, stocks=[], news=news,
            top_papers=papers,
        )
        assert "Top Papers" not in md
        assert "Recent Papers" not in md

    def test_stocks_not_in_local_section_order(self, local_runner):
        stocks = [{"symbol": "AAPL", "name": "Apple", "current_price": 150.0,
                    "change": 2.0, "percent_change": 1.3}]
        news = _sample_news(3)
        blogs = _sample_blogs(3)
        md = local_runner.generate_markdown_briefing(
            papers=[], blogs=blogs, stocks=stocks, news=news,
            top_papers=[],
        )
        assert "Financial Market Overview" not in md

    def test_epub_config_used(self, local_runner):
        epub_cfg = local_runner.config.get("epub", {})
        assert epub_cfg["title_format"] == "Local Briefing - {date}"
        assert epub_cfg["author"] == "Atlas Local"


# ---------- integration: full run with mocked scanners ----------


class TestLocalRunOrchestration:
    def test_local_run_success(self, local_runner, tmp_path):
        news = _sample_news(5)
        blogs = _sample_blogs(4)
        with patch.object(local_runner, "run_arxiv_scan", return_value=[]), \
             patch.object(local_runner, "run_blog_scan", return_value=blogs), \
             patch.object(local_runner, "run_stock_fetch", return_value=[]), \
             patch.object(local_runner, "run_news_aggregation", return_value=news), \
             patch.object(local_runner, "run_happenings_aggregation", return_value=[]):
            rc = local_runner.run()
        assert rc in (0, 1)
        output_dir = tmp_path / "briefings" / "local"
        md_files = list(output_dir.glob("Local-Briefing-*.md"))
        epub_files = list(output_dir.glob("Local-Briefing-*.epub"))
        assert len(md_files) == 1, f"No MD found in {output_dir}: {list(output_dir.iterdir())}"
        assert len(epub_files) == 1

    def test_local_run_no_data_returns_2(self, local_runner):
        with patch.object(local_runner, "run_arxiv_scan", return_value=[]), \
             patch.object(local_runner, "run_blog_scan", return_value=[]), \
             patch.object(local_runner, "run_stock_fetch", return_value=[]), \
             patch.object(local_runner, "run_news_aggregation", return_value=[]), \
             patch.object(local_runner, "run_happenings_aggregation", return_value=[]):
            assert local_runner.run() == 2

    def test_local_run_with_blogs_only(self, local_runner, tmp_path):
        blogs = _sample_blogs(4)
        with patch.object(local_runner, "run_arxiv_scan", return_value=[]), \
             patch.object(local_runner, "run_blog_scan", return_value=blogs), \
             patch.object(local_runner, "run_stock_fetch", return_value=[]), \
             patch.object(local_runner, "run_news_aggregation", return_value=[]), \
             patch.object(local_runner, "run_happenings_aggregation", return_value=[]):
            rc = local_runner.run()
        assert rc in (0, 1)
        md_files = list((tmp_path / "briefings" / "local").glob("Local-Briefing-*.md"))
        assert len(md_files) == 1

    def test_local_run_with_news_only(self, local_runner, tmp_path):
        news = _sample_news(5)
        with patch.object(local_runner, "run_arxiv_scan", return_value=[]), \
             patch.object(local_runner, "run_blog_scan", return_value=[]), \
             patch.object(local_runner, "run_stock_fetch", return_value=[]), \
             patch.object(local_runner, "run_news_aggregation", return_value=news), \
             patch.object(local_runner, "run_happenings_aggregation", return_value=[]):
            rc = local_runner.run()
        assert rc in (0, 1)
        md_files = list((tmp_path / "briefings" / "local").glob("Local-Briefing-*.md"))
        assert len(md_files) == 1

    def test_local_run_features_not_called(self, tmp_path, monkeypatch):
        """With local features disabled, solo/agent LLM calls must not be made."""
        base_config_local = dict(LOCAL_NEWS_CONFIG)
        base_config_local["interest_profile"] = [{"topic": "A", "weight": 1.0}]
        monkeypatch.chdir(tmp_path)
        runner = BriefingRunner(config=base_config_local, dry_run=True)

        runner.intelligence = MagicMock()
        runner.intelligence.available = True
        runner.intelligence.expand_topics.side_effect = lambda t: t
        runner.intelligence.generate_dynamic_queries.side_effect = lambda s, q, today_blogs=None: q
        runner.intelligence.filter_papers_by_relevance.side_effect = lambda p, ip: p
        runner.intelligence.rank_and_summarize_news.side_effect = lambda n, t: n
        runner.intelligence.rank_and_summarize_blogs.side_effect = lambda b, t: b
        runner.intelligence.correlate_stocks_and_news.side_effect = lambda s, n: s
        runner.intelligence.detect_emerging_themes.return_value = []
        runner.intelligence.track_trending.side_effect = lambda p, b, n, s: (s, p, b, n)
        runner.intelligence.score_papers_semantically.return_value = []
        runner.intelligence.summarize_papers.side_effect = lambda p: p
        runner.intelligence.assess_reproduction_feasibility.side_effect = lambda p: p
        runner.intelligence.generate_author_blurbs.side_effect = lambda items, t: items
        runner.intelligence.synthesize_briefing.return_value = {
            "editorial_intro": "Local editorial summary"
        }
        # The local config declares no extension_sections, so the extension
        # generator must never reach the client.
        runner.intelligence.client.invoke.return_value = "SHOULD NOT APPEAR"
        runner.intelligence.detect_entity_mentions.return_value = []
        runner._enrich_papers = MagicMock(side_effect=lambda p, t: p)

        with patch.object(runner, "run_arxiv_scan", return_value=[]), \
             patch.object(runner, "run_blog_scan", return_value=_sample_blogs(4)), \
             patch.object(runner, "run_stock_fetch", return_value=[]), \
             patch.object(runner, "run_news_aggregation", return_value=_sample_news(5)), \
             patch.object(runner, "run_happenings_aggregation", return_value=[]):
            rc = runner.run()

        assert rc in (0, 1)
        runner.intelligence.client.invoke.assert_not_called()
        runner.intelligence.generate_weekly_deep_dive.assert_not_called()

        # Verify the markdown does not contain disabled features
        md_files = list(tmp_path.glob("briefings/local/Local-Briefing-*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text()
        assert "SHOULD NOT APPEAR" not in content
        assert "Solo Founder Angle" not in content
        assert "Pipeline Efficiency Play" not in content
        assert "Local editorial summary" in content
        assert "San Diego / California News" in content
        assert "Local Sources & Analysis" in content

    def test_local_run_saves_state_to_custom_path(self, local_runner, tmp_path):
        news = _sample_news(5)
        blogs = _sample_blogs(4)
        with patch.object(local_runner, "run_arxiv_scan", return_value=[]), \
             patch.object(local_runner, "run_blog_scan", return_value=blogs), \
             patch.object(local_runner, "run_stock_fetch", return_value=[]), \
             patch.object(local_runner, "run_news_aggregation", return_value=news), \
             patch.object(local_runner, "run_happenings_aggregation", return_value=[]):
            local_runner.run()
        state_path = tmp_path / ".local-state.json"
        assert state_path.exists()
        assert "top_news_titles" in state_path.read_text()

    def test_local_run_saves_snapshots_to_custom_dir(self, local_runner, tmp_path):
        news = _sample_news(5)
        blogs = _sample_blogs(4)
        snapshot_dir = tmp_path / "snapshots" / "local"
        with patch.object(local_runner, "run_arxiv_scan", return_value=[]), \
             patch.object(local_runner, "run_blog_scan", return_value=blogs), \
             patch.object(local_runner, "run_stock_fetch", return_value=[]), \
             patch.object(local_runner, "run_news_aggregation", return_value=news), \
             patch.object(local_runner, "run_happenings_aggregation", return_value=[]):
            local_runner.run()
        assert snapshot_dir.exists()
        brave_files = list(snapshot_dir.rglob("brave_news.json"))
        assert len(brave_files) >= 1, f"No brave_news.json found in {snapshot_dir}"

    def test_full_main_config_pipeline_still_works(self, main_runner, tmp_path):
        """Regression: ensure main config still produces expected output."""
        main_runner.intelligence = MagicMock()
        main_runner.intelligence.available = True
        main_runner.intelligence.expand_topics.side_effect = lambda t: t
        main_runner.intelligence.generate_dynamic_queries.side_effect = lambda s, q, today_blogs=None: q
        main_runner.intelligence.filter_papers_by_relevance.side_effect = lambda p, ip: p
        main_runner.intelligence.rank_and_summarize_news.side_effect = lambda n, t: n
        main_runner.intelligence.rank_and_summarize_blogs.side_effect = lambda b, t: b
        main_runner.intelligence.correlate_stocks_and_news.side_effect = lambda s, n: s
        main_runner.intelligence.detect_emerging_themes.return_value = []
        main_runner.intelligence.track_trending.side_effect = lambda p, b, n, s: (s, p, b, n)
        main_runner.intelligence.score_papers_semantically.return_value = []
        main_runner.intelligence.summarize_papers.side_effect = lambda p: p
        main_runner.intelligence.assess_reproduction_feasibility.side_effect = lambda p: p
        main_runner.intelligence.generate_author_blurbs.side_effect = lambda items, t: items
        main_runner.intelligence.synthesize_briefing.return_value = {"editorial_intro": "Main summary"}
        main_runner.intelligence.client.invoke.return_value = "Extension section body"
        main_runner.intelligence.detect_entity_mentions.return_value = []
        main_runner._enrich_papers = MagicMock(side_effect=lambda p, t: p)

        papers = _sample_papers(3)
        with patch.object(main_runner, "run_arxiv_scan", return_value=papers), \
             patch.object(main_runner, "run_blog_scan", return_value=_sample_blogs(3)), \
             patch.object(main_runner, "run_stock_fetch", return_value=[]), \
             patch.object(main_runner, "run_news_aggregation", return_value=_sample_news(3)):
            rc = main_runner.run()

        assert rc in (0, 1)
        # One client call per declared extension section.
        tiers = [
            kwargs.get("tier")
            for _, kwargs in main_runner.intelligence.client.invoke.call_args_list
        ]
        assert tiers == ["heavy", "heavy"]

        md_files = list(tmp_path.glob("briefings/Atlas-Briefing-*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text()
        assert "Atlas Morning Briefing" in content
        assert "## Solo Founder Angle" in content
        assert "## Pipeline Efficiency Play" in content
        assert "Extension section body" in content
        assert "AI & Tech News" in content


class TestShippedLocalConfig:
    """Guards the life-first intent of the checked-in config_local.yaml."""

    @staticmethod
    def _config():
        import yaml
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "config_local.yaml"
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_summaries_are_action_first(self):
        assert self._config()["briefing_profile"]["actionable"] is True

    def test_investment_is_the_last_priority_tier(self):
        priorities = self._config()["briefing_profile"]["priorities"]
        assert len(priorities) >= 2
        assert "investment" in priorities[-1].lower()
        assert not any("investment" in tier.lower() for tier in priorities[:-1])

    def test_daily_life_outweighs_investment_in_interest_profile(self):
        profile = self._config()["interest_profile"]
        weights = {item["topic"].lower(): item["weight"] for item in profile}
        money = [w for topic, w in weights.items()
                 if "investment" in topic or "commercial real estate" in topic]
        assert money, "expected investment topics to still be present"
        top_topic = max(profile, key=lambda item: item["weight"])["topic"].lower()
        assert "investment" not in top_topic
        assert max(money) < max(weights.values())

    def test_happenings_lead_the_content_sections(self):
        order = self._config()["section_order"]
        assert order.index("happenings") < order.index("news") < order.index("blogs")

    def test_alerts_lead_the_section_order(self):
        assert self._config()["section_order"][0] == "alerts"

    def test_alerts_use_coastal_zones_not_the_whole_county(self):
        alerts = self._config()["alerts"]
        assert alerts["enabled"] is True
        assert alerts["zones"] == ["CAZ043", "CAZ243"]
        # CAC073 covers inland deserts and would fire alerts that do not apply
        # at the coast.
        assert "CAC073" not in alerts["zones"]

    def test_geo_filter_and_dedup_are_on(self):
        config = self._config()
        assert config["geo_filter"]["enabled"] is True
        assert "san diego" in config["geo_filter"]["place_terms"]
        assert config["news_similarity_dedup"]["enabled"] is True

    def test_every_graph_query_is_short(self):
        """Long keyword strings drift out of the area; 2-4 words is the shape."""
        config = self._config()

        def walk(node):
            yield node
            for child in node.get("children", []) or []:
                yield from walk(child)

        for root in config["interest_graph"]["roots"]:
            for node in walk(root):
                words = node["query"].split()
                assert len(words) <= 5, f"{node['id']} query is too long: {node['query']}"

    def test_hyperlocal_branches_use_a_weekly_window(self):
        roots = {r["id"]: r for r in self._config()["interest_graph"]["roots"]}
        assert roots["neighborhood"]["freshness"] == "pw"
        assert roots["beach_access"]["freshness"] == "pw"

    def test_condition_queries_are_not_sent_to_news_search(self):
        """Surf/outage/closure status comes from the alerts scanner, not Brave."""
        config = self._config()
        blob = str(config["interest_graph"]).lower()
        for term in ("power outage", "water shutoff", "high surf", "road closure"):
            assert term not in blob

    def test_happenings_refresh_lands_before_the_weekend(self):
        """A Saturday-only fetch left the section advertising a past weekend."""
        days = self._config()["happenings_fetch_weekday"]
        assert isinstance(days, list) and days
        assert 3 in days, "expected a Thursday refresh ahead of the weekend"
        assert days != [5]

    def test_press_release_portals_are_blocked(self):
        blocked = self._config()["geo_filter"]["blocked_sources"]
        assert "openpr.com" in blocked
        assert "prnewswire.com" in blocked
