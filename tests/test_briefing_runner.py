# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for briefing_runner module."""

import pytest
from scripts.briefing_runner import BriefingRunner


@pytest.fixture
def minimal_config():
    return {
        "arxiv_topics": ["Agent Evaluation"],
        "blog_feeds": [],
        "stocks": [],
        "news_queries": [],
        "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
        "num_paper_picks": 2,
        "max_papers": 5,
        "arxiv_days_back": 7,
        "output_format": "kindle",
        "file_naming": "Atlas-Briefing-{yyyy}.{mm}.{dd}",
        "pdf": {"font_size": 10, "line_spacing": 1.5},
        "bedrock": {"enabled": False},
    }


@pytest.fixture
def runner(minimal_config):
    return BriefingRunner(config=minimal_config, dry_run=True)


class TestDeduplicateNewsAndBlogs:
    def test_removes_duplicate_title(self, runner):
        news = [
            {"title": "Big AI News", "url": "http://news.com/1"},
            {"title": "Other News", "url": "http://news.com/2"},
        ]
        blogs = [
            {"title": "Big AI News", "link": "http://blog.com/big-ai"},
        ]
        deduped_news, _ = runner.deduplicate_news_and_blogs(news, blogs)
        assert len(deduped_news) == 1
        assert deduped_news[0]["title"] == "Other News"

    def test_removes_same_domain(self, runner):
        news = [
            {"title": "Anthropic Update", "url": "https://www.anthropic.com/news/update"},
            {"title": "Other News", "url": "http://other.com/1"},
        ]
        blogs = [
            {"title": "Blog Post", "link": "https://www.anthropic.com/blog/post"},
        ]
        deduped_news, _ = runner.deduplicate_news_and_blogs(news, blogs)
        assert len(deduped_news) == 1
        assert deduped_news[0]["title"] == "Other News"

    def test_no_blogs_returns_all_news(self, runner):
        news = [{"title": "News 1", "url": "http://a.com"}, {"title": "News 2", "url": "http://b.com"}]
        deduped_news, _ = runner.deduplicate_news_and_blogs(news, [])
        assert len(deduped_news) == 2

    def test_empty_inputs(self, runner):
        deduped_news, deduped_blogs = runner.deduplicate_news_and_blogs([], [])
        assert deduped_news == []
        assert deduped_blogs == []


class TestGenerateMarkdownBriefing:
    def test_generates_title(self, runner):
        md = runner.generate_markdown_briefing([], [], [], [], [])
        assert "Executive Summary" in md or md == ""  # title removed from markdown body

    def test_includes_stocks(self, runner):
        stocks = [{"symbol": "AMZN", "name": "Amazon", "current_price": 200.0, "change": 5.0, "percent_change": 2.5}]
        md = runner.generate_markdown_briefing([], [], stocks, [], [])
        assert "Financial Market Overview" in md
        assert "AMZN" in md
        assert "$200.00" in md

    def test_includes_stock_correlation(self, runner):
        stocks = [{
            "symbol": "NVDA", "name": "NVIDIA", "current_price": 100.0,
            "change": -5.0, "percent_change": -5.0,
            "news_correlation": "Export controls tightened",
        }]
        md = runner.generate_markdown_briefing([], [], stocks, [], [])
        assert "Export controls tightened" in md

    def test_includes_news(self, runner):
        news = [{"title": "AI Breakthrough", "url": "http://example.com", "source": "Reuters"}]
        md = runner.generate_markdown_briefing([], [], [], news, [])
        assert "AI & Tech News" in md
        assert "AI Breakthrough" in md

    def test_includes_blogs(self, runner):
        blogs = [{"title": "New Post", "source": "Anthropic", "link": "http://a.com", "summary": "Summary text"}]
        md = runner.generate_markdown_briefing([], blogs, [], [], [])
        assert "Blog Updates" in md
        assert "New Post" in md

    def test_includes_top_papers(self, runner):
        top_papers = [{
            "title": "Great Paper",
            "authors": ["Alice"],
            "score": 8.5,
            "score_combined": 4,
            "reproduction_difficulty": "S",
            "score_breakdown": {"has_code": True, "topic_match": 0.9, "recency": 0.95},
            "arxiv_url": "http://arxiv.org/abs/1",
            "pdf_link": "http://arxiv.org/pdf/1",
        }]
        md = runner.generate_markdown_briefing([], [], [], [], top_papers)
        assert "Top Papers" in md
        assert "Great Paper" in md

    def test_includes_paper_brief_summary(self, runner):
        top_papers = [{
            "title": "Paper",
            "authors": [],
            "score": 5.0,
            "score_combined": 4,
            "reproduction_difficulty": "M",
            "score_breakdown": {"has_code": False, "topic_match": 0.5, "recency": 0.5},
            "brief_summary": "This paper proposes a novel method.",
            "relevance_reason": "Directly matches agent evaluation",
            "arxiv_url": "",
            "pdf_link": "",
        }]
        md = runner.generate_markdown_briefing([], [], [], [], top_papers)
        assert "This paper proposes a novel method." in md

    def test_includes_synthesis(self, runner):
        synthesis = {
            "editorial_intro": "Today's briefing highlights a surge in agent evaluation papers.",
        }
        md = runner.generate_markdown_briefing([], [], [], [], [], synthesis)
        assert "Today's briefing highlights" in md
        assert "Executive Summary" in md

    def test_intelligence_badge_when_disabled(self, runner):
        md = runner.generate_markdown_briefing([], [], [], [], [])
        assert "Amazon Bedrock" not in md

    def test_includes_errors(self, runner):
        runner.errors = ["ArXiv scan failed"]
        md = runner.generate_markdown_briefing([], [], [], [], [])
        assert "Errors" in md
        assert "ArXiv scan failed" in md


class TestStatus:
    def test_initial_status(self, runner):
        assert runner.status["papers_found"] == 0
        assert runner.status["intelligence_enabled"] is False
        assert runner.status["pdf_generated"] is False

    def test_save_status(self, runner, tmp_path):
        runner.save_status(str(tmp_path))
        import json
        status_path = tmp_path / "status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        assert "timestamp" in status
        assert "elapsed_seconds" in status


class TestPreflightModelLoading:
    """The runner lets preflight pin a per-tier model, so a bad file is costly.

    A stale .model-availability.json would pin models chosen under yesterday's
    conditions — exactly what the check exists to avoid, since free-model
    availability changes within hours.
    """

    def _config(self, minimal_config, path):
        cfg = dict(minimal_config)
        cfg["preflight_file_path"] = str(path)
        return cfg

    def test_missing_file_returns_empty(self, minimal_config, tmp_path):
        cfg = self._config(minimal_config, tmp_path / "nope.json")
        assert BriefingRunner(config=cfg, dry_run=True)._load_preflight_models() == {}

    def test_fresh_file_is_used(self, minimal_config, tmp_path):
        import json

        p = tmp_path / ".model-availability.json"
        payload = {"openrouter": {"heavy": {"available": True, "tier": "heavy",
                                            "model": "openrouter/x:free"}}}
        p.write_text(json.dumps(payload))
        cfg = self._config(minimal_config, p)
        assert BriefingRunner(config=cfg, dry_run=True)._load_preflight_models() == payload

    def test_stale_file_is_ignored(self, minimal_config, tmp_path):
        import json
        import os
        import time

        p = tmp_path / ".model-availability.json"
        p.write_text(json.dumps({"openrouter": {"heavy": {"available": True}}}))
        runner_cls = BriefingRunner
        stale = time.time() - (runner_cls.PREFLIGHT_MAX_AGE_SECONDS + 60)
        os.utime(p, (stale, stale))
        cfg = self._config(minimal_config, p)
        assert BriefingRunner(config=cfg, dry_run=True)._load_preflight_models() == {}

    def test_corrupt_file_returns_empty(self, minimal_config, tmp_path):
        p = tmp_path / ".model-availability.json"
        p.write_text("{not json")
        cfg = self._config(minimal_config, p)
        assert BriefingRunner(config=cfg, dry_run=True)._load_preflight_models() == {}

    def test_non_object_json_returns_empty(self, minimal_config, tmp_path):
        p = tmp_path / ".model-availability.json"
        p.write_text("[1, 2, 3]")
        cfg = self._config(minimal_config, p)
        assert BriefingRunner(config=cfg, dry_run=True)._load_preflight_models() == {}


class TestBackendChainOrder:
    """Order is cost: the chain is tried front to back, so paid goes last."""

    def _config(self, minimal_config, **overrides):
        cfg = dict(minimal_config)
        cfg.update({
            "openrouter": {"enabled": True, "api_key": "k"},
            "gemini": {"enabled": False},
            "opencode": {"enabled": True},
        })
        cfg.update(overrides)
        return cfg

    def _names(self, cfg):
        runner = BriefingRunner(config=cfg, dry_run=True)
        client = runner.llm_client
        clients = getattr(client, "clients", [client])
        return [type(c).__name__ for c in clients]

    def test_free_openrouter_precedes_paid_opencode_by_default(self, minimal_config):
        names = self._names(self._config(minimal_config))
        assert names.index("OpenRouterClient") < names.index("OpencodeClient")

    def test_priority_list_is_honoured(self, minimal_config):
        cfg = self._config(minimal_config, llm={"backend_priority": ["opencode", "openrouter"]})
        names = self._names(cfg)
        assert names.index("OpencodeClient") < names.index("OpenRouterClient")

    def test_backend_missing_from_priority_is_still_included_last(self, minimal_config):
        """A typo in the priority list must not silently drop a backend."""
        cfg = self._config(minimal_config, llm={"backend_priority": ["openrouter"]})
        names = self._names(cfg)
        assert "OpencodeClient" in names
        assert names[-1] == "OpencodeClient"

    def test_unknown_priority_entry_is_ignored(self, minimal_config):
        cfg = self._config(minimal_config, llm={"backend_priority": ["nope", "openrouter"]})
        assert "OpenRouterClient" in self._names(cfg)

    def test_disabled_backend_is_skipped(self, minimal_config):
        cfg = self._config(minimal_config, opencode={"enabled": False})
        assert "OpencodeClient" not in self._names(cfg)


class TestStatusFileIsPerPipeline:
    """Two pipelines sharing status.json means the second erases the first.

    On 2026-08-28 status.json reported papers_found: 0 from the local pipeline
    (which scans no papers) three minutes after the main run had collected 172.
    """

    def test_default_status_filename(self, minimal_config, tmp_path):
        import json

        BriefingRunner(config=minimal_config, dry_run=True).save_status(str(tmp_path))
        assert (tmp_path / "status.json").exists()

    def test_configured_status_filename_is_used(self, minimal_config, tmp_path):
        cfg = dict(minimal_config, status_file_path="status-local.json")
        BriefingRunner(config=cfg, dry_run=True).save_status(str(tmp_path))
        assert (tmp_path / "status-local.json").exists()
        assert not (tmp_path / "status.json").exists()

    def test_two_pipelines_do_not_overwrite_each_other(self, minimal_config, tmp_path):
        import json

        main = BriefingRunner(
            config=dict(minimal_config, status_file_path="status.json",
                        pipeline_name="main"), dry_run=True)
        main.status["papers_found"] = 172
        main.save_status(str(tmp_path))

        local = BriefingRunner(
            config=dict(minimal_config, status_file_path="status-local.json",
                        pipeline_name="local"), dry_run=True)
        local.status["papers_found"] = 0
        local.save_status(str(tmp_path))

        saved = json.loads((tmp_path / "status.json").read_text())
        assert saved["papers_found"] == 172, "local run clobbered the main run"
        assert saved["pipeline"] == "main"
        assert json.loads((tmp_path / "status-local.json").read_text())["pipeline"] == "local"


class TestSharedChainBuilder:
    """The runner and the quality checker must not drift on backend order.

    They each built their own chain, and did drift: quality_check kept
    opencode (paid) first long after the runner moved to openrouter (free)
    first, so every daily quality run billed the paid backstop.
    """

    def _config(self, minimal_config, **over):
        cfg = dict(minimal_config)
        cfg.update({
            "openrouter": {"enabled": True, "api_key": "k"},
            "gemini": {"enabled": False},
            "opencode": {"enabled": True},
        })
        cfg.update(over)
        return cfg

    def test_runner_and_quality_check_agree_on_order(self, minimal_config):
        from scripts.llm_chain import build_llm_chain
        from scripts.quality_check import build_llm_client

        cfg = self._config(minimal_config)
        expected = [type(c).__name__ for c in build_llm_chain(cfg)]
        judge = build_llm_client(cfg)
        actual = [type(c).__name__ for c in getattr(judge, "clients", [judge])]
        assert actual == expected

    def test_quality_judge_puts_free_before_paid(self, minimal_config):
        from scripts.quality_check import build_llm_client

        judge = build_llm_client(self._config(minimal_config))
        names = [type(c).__name__ for c in getattr(judge, "clients", [judge])]
        assert names.index("OpenRouterClient") < names.index("OpencodeClient")

    def test_quality_judge_returns_none_when_nothing_enabled(self, minimal_config):
        from scripts.quality_check import build_llm_client

        cfg = self._config(minimal_config, openrouter={"enabled": False},
                           opencode={"enabled": False}, gemini={"enabled": False})
        assert build_llm_client(cfg) is None


class TestHappeningsStalenessFilter:
    """Cached happenings must not advertise events that have already passed.

    On 2026-08-28 the local briefing led with three "Aug. 21-23 this weekend"
    items — a week past — because the cache refreshed only on Saturdays.
    A shorter cache window narrows the gap but cannot close it, so dates are
    checked at use time.
    """

    def _runner(self, minimal_config):
        return BriefingRunner(config=minimal_config, dry_run=True)

    def test_drops_items_whose_dates_have_passed(self, minimal_config, monkeypatch):
        import scripts.briefing_runner as br
        from datetime import datetime as real_dt

        class _Now(real_dt):
            @classmethod
            def now(cls, tz=None):
                return real_dt(2026, 8, 28)

        monkeypatch.setattr(br, "datetime", _Now)
        items = [
            {"title": "Weekend roundup", "description": "Events Aug. 21-23 citywide."},
            {"title": "Farmers market", "description": "Market on Aug. 26, 9:30 a.m."},
        ]
        assert self._runner(minimal_config)._drop_past_happenings(items) == []

    def test_keeps_future_and_undated_items(self, minimal_config, monkeypatch):
        import scripts.briefing_runner as br
        from datetime import datetime as real_dt

        class _Now(real_dt):
            @classmethod
            def now(cls, tz=None):
                return real_dt(2026, 8, 28)

        monkeypatch.setattr(br, "datetime", _Now)
        items = [
            {"title": "Labor Day fest", "description": "Runs Aug 30-Sep 2."},
            {"title": "Volleyball rule", "description": "Play can now begin at 6 a.m."},
            {"title": "City events calendar", "description": "Standing listing."},
        ]
        kept = self._runner(minimal_config)._drop_past_happenings(items)
        assert [i["title"] for i in kept] == [
            "Labor Day fest", "Volleyball rule", "City events calendar",
        ]

    def test_range_is_judged_by_its_end_date(self, minimal_config, monkeypatch):
        import scripts.briefing_runner as br
        from datetime import datetime as real_dt

        class _Now(real_dt):
            @classmethod
            def now(cls, tz=None):
                return real_dt(2026, 8, 28)

        monkeypatch.setattr(br, "datetime", _Now)
        # Starts in the past, still running today -> keep.
        items = [{"title": "Fair", "description": "Runs Aug 25-30."}]
        assert len(self._runner(minimal_config)._drop_past_happenings(items)) == 1

    def test_empty_input_is_passed_through(self, minimal_config):
        assert self._runner(minimal_config)._drop_past_happenings([]) == []


class TestHappeningsUrlDedup:
    """The same document under a cosmetic URL variant reached the reader twice."""

    def _runner(self, minimal_config):
        return BriefingRunner(config=minimal_config, dry_run=True)

    def test_trailing_slash_variant_is_collapsed(self, minimal_config):
        items = [
            {"title": "Roundup", "url": "https://sdmag.com/things-to-do/x/"},
            {"title": "Roundup", "url": "https://sdmag.com/things-to-do/x"},
        ]
        assert len(self._runner(minimal_config)._dedupe_happenings_by_url(items)) == 1

    def test_www_variant_is_collapsed(self, minimal_config):
        items = [
            {"title": "Picks", "url": "https://www.kpbs.org/news/y"},
            {"title": "Picks", "url": "https://kpbs.org/news/y"},
        ]
        assert len(self._runner(minimal_config)._dedupe_happenings_by_url(items)) == 1

    def test_first_occurrence_is_the_one_kept(self, minimal_config):
        items = [
            {"title": "first", "url": "https://a.com/p"},
            {"title": "second", "url": "https://a.com/p/"},
        ]
        kept = self._runner(minimal_config)._dedupe_happenings_by_url(items)
        assert [i["title"] for i in kept] == ["first"]

    def test_distinct_urls_are_all_kept(self, minimal_config):
        items = [
            {"title": "a", "url": "https://a.com/1"},
            {"title": "b", "url": "https://a.com/2"},
        ]
        assert len(self._runner(minimal_config)._dedupe_happenings_by_url(items)) == 2

    def test_items_without_a_url_are_not_collapsed_together(self, minimal_config):
        items = [{"title": "a"}, {"title": "b"}]
        assert len(self._runner(minimal_config)._dedupe_happenings_by_url(items)) == 2

    def test_empty_input(self, minimal_config):
        assert self._runner(minimal_config)._dedupe_happenings_by_url([]) == []


class TestBlogWindowIsIndependentOfArxiv:
    """Blogs and papers have different cadences.

    The blog cutoff reused arxiv_days_back, so at arxiv_days_back=3 a weekly
    blogger was invisible on most runs — which the quality checker reported as
    a yield collapse against feeds that were healthy. Measured 2026-08-29 over
    28 feeds: 17 had posted within 3 days, 19 within 7.
    """

    def _days_back_used(self, runner, monkeypatch):
        seen = {}

        class _Scanner:
            def __init__(self, feeds, days_back, max_items):
                seen["days_back"] = days_back

            def scan_all_feeds(self):
                return []

        monkeypatch.setattr("scripts.briefing_runner.BlogScanner", _Scanner)
        runner.run_blog_scan()
        return seen["days_back"]

    def test_blog_days_back_wins_when_set(self, minimal_config, monkeypatch):
        cfg = dict(minimal_config, blog_feeds=[{"name": "x", "url": "u"}],
                   arxiv_days_back=3, blog_days_back=7)
        runner = BriefingRunner(config=cfg, dry_run=True)
        assert self._days_back_used(runner, monkeypatch) == 7

    def test_falls_back_to_arxiv_days_back_when_unset(self, minimal_config, monkeypatch):
        """Existing configs without the new key keep their current behaviour."""
        cfg = dict(minimal_config, blog_feeds=[{"name": "x", "url": "u"}],
                   arxiv_days_back=3)
        cfg.pop("blog_days_back", None)
        runner = BriefingRunner(config=cfg, dry_run=True)
        assert self._days_back_used(runner, monkeypatch) == 3

    def test_defaults_to_seven_when_neither_is_set(self, minimal_config, monkeypatch):
        cfg = dict(minimal_config, blog_feeds=[{"name": "x", "url": "u"}])
        cfg.pop("arxiv_days_back", None)
        cfg.pop("blog_days_back", None)
        runner = BriefingRunner(config=cfg, dry_run=True)
        assert self._days_back_used(runner, monkeypatch) == 7
