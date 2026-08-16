# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for the full BriefingRunner.run() orchestration pipeline.

Mocks scanners and intelligence so we test glue logic + branching, not LLM calls.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from scripts.briefing_runner import BriefingRunner


@pytest.fixture
def base_config():
    return {
        "arxiv_topics": ["AI"],
        "blog_feeds": [],
        "stocks": [],
        "news_queries": [],
        "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
        "num_paper_picks": 2,
        "max_papers": 5,
        "max_blogs": 5,
        "max_news": 5,
        "arxiv_days_back": 7,
        "output_format": "kindle",
        "file_naming": "Test-{yyyy}.{mm}.{dd}",
        "pdf": {"enabled": False},
        "bedrock": {"enabled": False},
        "gemini": {"enabled": False},
    }


@pytest.fixture
def runner_with_data(base_config, tmp_path, monkeypatch):
    """Builds a runner with no real scanners — methods are patched per test."""
    monkeypatch.chdir(tmp_path)
    return BriefingRunner(base_config, dry_run=True)


class TestRunOrchestration:
    def test_run_no_data_returns_2(self, runner_with_data):
        # All scanners return empty
        with patch.object(runner_with_data, "run_arxiv_scan", return_value=[]), \
             patch.object(runner_with_data, "run_blog_scan", return_value=[]), \
             patch.object(runner_with_data, "run_stock_fetch", return_value=[]), \
             patch.object(runner_with_data, "run_news_aggregation", return_value=[]):
            assert runner_with_data.run() == 2

    def test_run_success_no_intelligence(self, runner_with_data, tmp_path):
        papers = [{"title": "P1", "summary": "abs", "published": "", "arxiv_url": ""}]
        with patch.object(runner_with_data, "run_arxiv_scan", return_value=papers), \
             patch.object(runner_with_data, "run_blog_scan", return_value=[]), \
             patch.object(runner_with_data, "run_stock_fetch", return_value=[]), \
             patch.object(runner_with_data, "run_news_aggregation", return_value=[]):
            rc = runner_with_data.run()
        assert rc in (0, 1)
        # Files generated even without PDF (EPUB always on)
        output_dir = tmp_path / "briefings"
        md_files = list(output_dir.glob("Test-*.md"))
        epub_files = list(output_dir.glob("Test-*.epub"))
        assert len(md_files) == 1
        assert len(epub_files) == 1

    def test_run_with_pdf_enabled(self, base_config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        base_config["pdf"]["enabled"] = True
        runner = BriefingRunner(base_config, dry_run=True)
        papers = [{"title": "P", "summary": "a", "published": "", "arxiv_url": ""}]
        with patch.object(runner, "run_arxiv_scan", return_value=papers), \
             patch.object(runner, "run_blog_scan", return_value=[]), \
             patch.object(runner, "run_stock_fetch", return_value=[]), \
             patch.object(runner, "run_news_aggregation", return_value=[]):
            runner.run()
        assert list((tmp_path / "briefings").glob("Test-*.pdf"))
        assert runner.status["pdf_generated"] is True

    def test_run_records_errors_returns_1(self, runner_with_data):
        papers = [{"title": "P", "summary": "a", "published": "", "arxiv_url": ""}]
        runner_with_data.errors = ["earlier error"]
        with patch.object(runner_with_data, "run_arxiv_scan", return_value=papers), \
             patch.object(runner_with_data, "run_blog_scan", return_value=[]), \
             patch.object(runner_with_data, "run_stock_fetch", return_value=[]), \
             patch.object(runner_with_data, "run_news_aggregation", return_value=[]):
            assert runner_with_data.run() == 1

    def test_run_full_intelligence_path(self, base_config, tmp_path, monkeypatch):
        """Exercise the LLM-enabled branches with a mocked intelligence layer."""
        monkeypatch.chdir(tmp_path)
        base_config["interest_profile"] = [{"topic": "A", "weight": 1.0}]
        base_config["tracked_entities"] = [{"name": "Anthropic", "type": "company"}]
        runner = BriefingRunner(base_config, dry_run=True)

        # Force intelligence to look available, mock all methods
        runner.intelligence = MagicMock()
        runner.intelligence.available = True
        runner.intelligence.expand_topics.side_effect = lambda t: t
        runner.intelligence.generate_dynamic_queries.side_effect = lambda s, q, today_blogs=None: q
        runner.intelligence.filter_papers_by_relevance.side_effect = lambda p, ip: p
        runner.intelligence.rank_and_summarize_news.side_effect = lambda n, t: n
        runner.intelligence.rank_and_summarize_blogs.side_effect = lambda b, t: b
        runner.intelligence.correlate_stocks_and_news.side_effect = lambda s, n: s
        runner.intelligence.detect_emerging_themes.return_value = ["theme"]
        runner.intelligence.track_trending.side_effect = (
            lambda p, b, n, s: (s, p, b, n)
        )
        runner.intelligence.assess_reproduction_feasibility.side_effect = lambda p: p
        runner.intelligence.generate_author_blurbs.side_effect = lambda items, t: items
        runner.intelligence.synthesize_briefing.return_value = {
            "editorial_intro": "Today's summary."
        }
        runner.intelligence.detect_entity_mentions.return_value = []
        runner.intelligence.generate_weekly_deep_dive.return_value = ""

        runner._analyze_market_trend = MagicMock(return_value="market trend")
        runner._enrich_papers = MagicMock(side_effect=lambda p, t: p)
        runner._ensure_paper_summaries = MagicMock(side_effect=lambda p: p)

        papers = [{"title": "P", "summary": "a", "published": "", "arxiv_url": "",
                   "authors": []}]
        with patch.object(runner, "run_arxiv_scan", return_value=papers), \
             patch.object(runner, "run_blog_scan", return_value=[{"title": "B", "source": "s"}]), \
             patch.object(runner, "run_stock_fetch", return_value=[{"symbol": "X", "percent_change": 0}]), \
             patch.object(runner, "run_news_aggregation", return_value=[{"title": "N"}]):
            rc = runner.run()

        assert rc in (0, 1)
        # Verify orchestration touched key intel methods
        runner.intelligence.expand_topics.assert_called_once()
        runner.intelligence.generate_dynamic_queries.assert_called_once()
        runner.intelligence.synthesize_briefing.assert_called_once()
        runner.intelligence.detect_entity_mentions.assert_called_once()

    def test_run_pdf_and_epub_fail_returns_2(self, base_config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        base_config["pdf"]["enabled"] = True
        runner = BriefingRunner(base_config, dry_run=True)
        papers = [{"title": "P", "summary": "a", "published": "", "arxiv_url": ""}]
        with patch.object(runner, "run_arxiv_scan", return_value=papers), \
             patch.object(runner, "run_blog_scan", return_value=[]), \
             patch.object(runner, "run_stock_fetch", return_value=[]), \
             patch.object(runner, "run_news_aggregation", return_value=[]), \
             patch.object(runner, "generate_pdf", return_value=False), \
             patch.object(runner, "generate_epub", return_value=False):
            assert runner.run() == 2

    def test_run_saturday_triggers_deep_dive(self, base_config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = BriefingRunner(base_config, dry_run=True)
        runner.intelligence = MagicMock()
        runner.intelligence.available = True
        runner.intelligence.expand_topics.side_effect = lambda t: t
        runner.intelligence.generate_dynamic_queries.side_effect = lambda s, q, today_blogs=None: q
        runner.intelligence.filter_papers_by_relevance.side_effect = lambda p, ip: p
        runner.intelligence.rank_and_summarize_news.side_effect = lambda n, t: n
        runner.intelligence.rank_and_summarize_blogs.side_effect = lambda b, t: b
        runner.intelligence.correlate_stocks_and_news.side_effect = lambda s, n: s
        runner.intelligence.detect_emerging_themes.return_value = []
        runner.intelligence.track_trending.side_effect = (
            lambda p, b, n, s: (s, p, b, n)
        )
        runner.intelligence.assess_reproduction_feasibility.side_effect = lambda p: p
        runner.intelligence.generate_author_blurbs.side_effect = lambda items, t: items
        runner.intelligence.synthesize_briefing.return_value = {"editorial_intro": "i"}
        runner.intelligence.detect_entity_mentions.return_value = []
        runner.intelligence.generate_weekly_deep_dive.return_value = "Saturday deep dive content"
        runner._analyze_market_trend = MagicMock(return_value="")
        runner._enrich_papers = MagicMock(side_effect=lambda p, t: p)
        runner._ensure_paper_summaries = MagicMock(side_effect=lambda p: p)

        # Force "Saturday" by patching datetime.now
        import scripts.briefing_runner as br
        saturday = datetime(2026, 5, 23, 8, 0, 0)  # Saturday
        with patch.object(br, "datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            # Allow other datetime methods to pass through
            mock_dt.strptime = datetime.strptime

            papers = [{"title": "P", "summary": "a", "published": "", "arxiv_url": "",
                       "authors": []}]
            with patch.object(runner, "run_arxiv_scan", return_value=papers), \
                 patch.object(runner, "run_blog_scan", return_value=[]), \
                 patch.object(runner, "run_stock_fetch", return_value=[]), \
                 patch.object(runner, "run_news_aggregation", return_value=[]):
                runner.run()

        # Saturday path called weekly deep dive
        runner.intelligence.generate_weekly_deep_dive.assert_called_once()

    def test_run_unwritable_markdown_handled(self, base_config, tmp_path, monkeypatch):
        """If markdown save fails, run() should still continue."""
        monkeypatch.chdir(tmp_path)
        runner = BriefingRunner(base_config, dry_run=True)
        papers = [{"title": "P", "summary": "a", "published": "", "arxiv_url": ""}]
        # Mock open to raise IOError for the .md write only
        original_open = open

        def selective_open(path, *args, **kwargs):
            if isinstance(path, str) and path.endswith(".md"):
                raise IOError("write failed")
            return original_open(path, *args, **kwargs)

        with patch.object(runner, "run_arxiv_scan", return_value=papers), \
             patch.object(runner, "run_blog_scan", return_value=[]), \
             patch.object(runner, "run_stock_fetch", return_value=[]), \
             patch.object(runner, "run_news_aggregation", return_value=[]), \
             patch("builtins.open", side_effect=selective_open):
            # Should not crash even though markdown write fails
            rc = runner.run()
        # Run completes (markdown failure is just a warning)
        assert rc in (0, 1, 2)


class TestEnsurePaperSummaries:
    def test_no_papers_returns_empty(self, base_config):
        runner = BriefingRunner(base_config, dry_run=True)
        result = runner._ensure_paper_summaries([])
        assert result == []

    def test_skip_when_unavailable(self, base_config):
        runner = BriefingRunner(base_config, dry_run=True)
        # Intelligence is unavailable (no Gemini)
        papers = [{"title": "P", "summary": "abstract"}]
        result = runner._ensure_paper_summaries(papers)
        # Returns unchanged (no brief_summary added)
        assert result == papers

    def test_already_has_summary_unchanged(self, base_config):
        runner = BriefingRunner(base_config, dry_run=True)
        papers = [{"title": "P", "summary": "abs",
                   "brief_summary": "existing", "score_combined": 4}]
        result = runner._ensure_paper_summaries(papers)
        assert result[0]["brief_summary"] == "existing"

    def test_generates_for_missing(self, base_config):
        runner = BriefingRunner(base_config, dry_run=True)
        runner.intelligence = MagicMock()
        runner.intelligence.available = True
        runner.intelligence.client = MagicMock()
        runner.intelligence.client.invoke.return_value = (
            "[1] SCORE:4/5 Generated summary text."
        )
        runner.intelligence._parse_ranked_response = MagicMock(
            return_value=[(0, "SCORE:4/5 Generated summary text.")]
        )
        runner.intelligence.extract_score = MagicMock(
            return_value=(4, "Generated summary text.")
        )
        papers = [{"title": "P", "summary": "abs"}]
        result = runner._ensure_paper_summaries(papers)
        assert result[0]["brief_summary"] == "Generated summary text."
        assert result[0]["score_combined"] == 4

    def test_no_abstract_skipped(self, base_config):
        runner = BriefingRunner(base_config, dry_run=True)
        runner.intelligence = MagicMock()
        runner.intelligence.available = True
        papers = [{"title": "P", "summary": ""}]  # empty abstract
        result = runner._ensure_paper_summaries(papers)
        assert "brief_summary" not in result[0]


class TestEnrichPapers:
    def test_enrich_papers_chains_calls(self, base_config):
        runner = BriefingRunner(base_config, dry_run=True)
        runner.intelligence = MagicMock()
        runner.intelligence.summarize_papers.side_effect = (
            lambda p: [{**x, "brief_summary": "s"} for x in p]
        )
        runner.intelligence.score_papers_semantically.side_effect = (
            lambda p, t: [{**x, "semantic_score": 7} for x in p]
        )
        papers = [{"title": "P", "summary": "a"}]
        result = runner._enrich_papers(papers, ["topic"])
        assert result[0]["brief_summary"] == "s"
        assert result[0]["semantic_score"] == 7
