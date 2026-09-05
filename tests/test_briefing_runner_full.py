# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Comprehensive tests for briefing_runner orchestration and helpers."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from scripts.briefing_runner import BriefingRunner, load_config, main


@pytest.fixture
def cfg():
    return {
        "arxiv_topics": ["Agent Evaluation"],
        "blog_feeds": [],
        "stocks": [],
        "news_queries": [],
        "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
        "num_paper_picks": 5,
        "max_papers": 5,
        "max_blogs": 5,
        "max_news": 5,
        "arxiv_days_back": 7,
        "output_format": "kindle",
        "file_naming": "Atlas-Briefing-{yyyy}.{mm}.{dd}",
        "pdf": {"font_size": 10, "line_spacing": 1.5, "enabled": False},
        "bedrock": {"enabled": False},
        "gemini": {"enabled": False},
    }


@pytest.fixture
def runner(cfg):
    return BriefingRunner(config=cfg, dry_run=True)


# ---------- format/render helpers ----------


class TestFormatFilename:
    def test_default_pattern(self, runner):
        result = runner._format_filename(datetime(2026, 5, 22))
        assert result == "Atlas-Briefing-2026.05.22"

    def test_custom_pattern(self, cfg):
        cfg["file_naming"] = "MyBriefing-{type}-{yyyy}{mm}{dd}"
        r = BriefingRunner(cfg, dry_run=True)
        assert r._format_filename(datetime(2026, 5, 22)) == "MyBriefing-Daily-20260522"

    def test_unknown_placeholder_raises(self, runner):
        """format_map raises KeyError on unknown placeholders.

        Documented behavior: docstring says 'ignoring unknown keys' but
        format_map() actually raises. Constructor eagerly calls this, so
        an invalid pattern aborts startup — by design, fail fast.
        """
        runner.config["file_naming"] = "X-{unknown}-{yyyy}"
        with pytest.raises(KeyError):
            runner._format_filename(datetime(2026, 5, 22))


class TestRenderStars:
    def test_full_stars(self, runner):
        assert runner._render_stars(5) == "★★★★★"

    def test_zero_stars(self, runner):
        assert runner._render_stars(0) == "☆☆☆☆☆"

    def test_partial(self, runner):
        assert runner._render_stars(3) == "★★★☆☆"

    def test_clamps_above_max(self, runner):
        assert runner._render_stars(10) == "★★★★★"

    def test_clamps_negative(self, runner):
        assert runner._render_stars(-1) == "☆☆☆☆☆"

    def test_none(self, runner):
        assert runner._render_stars(None) == ""


class TestCleanSummary:
    def test_strips_summary_prefix(self, runner):
        result = runner._clean_summary("Summary: Real text here", "Title")
        assert "Summary:" not in result
        assert "Real text" in result

    def test_strips_leading_bold(self, runner):
        assert runner._clean_summary("* Real text", "Title") == "Real text"

    def test_strips_title_echo(self, runner):
        result = runner._clean_summary("AI Breakthrough - core finding", "AI Breakthrough")
        assert "core finding" in result.lower()
        assert "ai breakthrough" not in result.lower()

    def test_strips_paren_source(self, runner):
        result = runner._clean_summary(
            "Title (Reuters) Real content here", "Title"
        )
        assert "Real content" in result

    def test_empty_passthrough(self, runner):
        assert runner._clean_summary("", "Title") == ""

    def test_no_title_no_strip(self, runner):
        assert runner._clean_summary("Some content", "") == "Some content"


# ---------- deduplication ----------


class TestDedupAgainstPrevious:
    def test_no_state_returns_unchanged(self):
        p, b, n, h = BriefingRunner._dedup_against_previous(
            papers=[{"title": "A"}], blogs=[{"title": "B"}], news=[{"title": "C"}],
            previous_state={},
        )
        assert len(p) == 1 and len(b) == 1 and len(n) == 1 and h == []

    def test_filters_seen_titles(self):
        state = {
            "top_paper_titles": ["paper a"],
            "top_blog_titles": ["blog b"],
            "top_news_titles": ["news c"],
            "top_happenings_titles": ["event e"],
        }
        p, b, n, h = BriefingRunner._dedup_against_previous(
            papers=[{"title": "Paper A"}, {"title": "Paper X"}],
            blogs=[{"title": "Blog B"}],
            news=[{"title": "News C"}, {"title": "News Y"}],
            happenings=[{"title": "Event E"}, {"title": "Event Z"}],
            previous_state=state,
        )
        assert [x["title"] for x in p] == ["Paper X"]
        assert b == []
        assert [x["title"] for x in n] == ["News Y"]
        assert [x["title"] for x in h] == ["Event Z"]


# ---------- state ----------


class TestStatePersistence:
    def test_save_and_load_roundtrip(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        papers = [{"title": "P1"}, {"title": "P2"}]
        runner._save_state(
            papers, [{"title": "B"}], [{"title": "N"}],
            [{"symbol": "X", "current_price": 100}],
            emerging_themes=["theme1"],
            trending_topics={"x": {"count": 1}},
            weekly_items=[{"date": "d", "title": "t"}],
        )
        state = runner._load_previous_state()
        assert state["top_paper_titles"] == ["P1", "P2"]
        assert state["emerging_themes"] == ["theme1"]
        assert state["trending_topics"] == {"x": {"count": 1}}
        assert state["weekly_items"][0]["date"] == "d"

    def test_load_missing_returns_empty(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert runner._load_previous_state() == {}

    def test_load_corrupt_returns_empty(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".atlas-state.json").write_text("{not valid json")
        assert runner._load_previous_state() == {}


# ---------- save_status ----------


class TestSaveStatus:
    def test_writes_status(self, runner, tmp_path):
        runner.errors = ["err1", "err2"]
        runner.save_status(str(tmp_path))
        data = json.loads((tmp_path / "status.json").read_text())
        assert data["errors"] == ["err1", "err2"]

    def test_handles_unwritable_dir(self, runner):
        # Should swallow IOError, not crash
        runner.save_status("/nonexistent/path/that/does/not/exist")


# ---------- render sections (more depth) ----------


class TestRenderStocks(object):
    def test_renders_table_with_market_trend(self, runner):
        stocks = [
            {"symbol": "AAPL", "current_price": 150, "percent_change": 1.2,
             "news_correlation": "earnings beat"},
            {"symbol": "BAD", "error": "no data"},
        ]
        md = runner._render_stocks(stocks, market_trend="Trend summary")
        assert "Financial Market Overview" in md
        assert "Trend summary" in md
        assert "AAPL" in md
        assert "earnings beat" in md
        # Error stocks rendered as em-dash
        assert "Error" in md

    def test_truncates_long_driver(self, runner):
        stocks = [{
            "symbol": "X", "current_price": 100, "percent_change": 0,
            "news_correlation": "a" * 50,
        }]
        md = runner._render_stocks(stocks)
        assert "..." in md

    def test_negative_change_no_plus(self, runner):
        stocks = [{"symbol": "X", "current_price": 100, "percent_change": -2.5}]
        md = runner._render_stocks(stocks)
        assert "-2.50%" in md


class TestRenderNews:
    def test_renders_url_and_summary(self, runner):
        news = [{"title": "News T", "url": "http://x.com",
                 "brief_summary": "Summary text"}]
        md = runner._render_news(news)
        assert "AI & Tech News" in md
        assert "[News T](http://x.com)" in md
        assert "Summary text" in md

    def test_no_url_no_link(self, runner):
        news = [{"title": "T", "brief_summary": "summ"}]
        md = runner._render_news(news)
        assert "T" in md
        assert "http" not in md

    def test_includes_author_blurb(self, runner):
        news = [{"title": "T", "url": "u", "author_blurb": "About author"}]
        md = runner._render_news(news)
        assert "Source Information" in md
        assert "About author" in md


class TestRenderBlogs:
    def test_sorted_by_score_when_present(self, runner):
        blogs = [
            {"title": "Low", "source": "S", "score_combined": 1,
             "brief_summary": "low"},
            {"title": "High", "source": "S", "score_combined": 5,
             "brief_summary": "high"},
            {"title": "Mid", "source": "S", "score_combined": 3,
             "brief_summary": "mid"},
        ]
        md = runner._render_blogs(blogs)
        # Low (score 1) filtered out (< 3); High should appear before Mid
        assert md.index("High") < md.index("Mid")
        assert "Low" not in md

    def test_no_scores_uses_first_5(self, runner):
        blogs = [{"title": f"B{i}", "source": "S", "summary": "s" * 400}
                 for i in range(7)]
        md = runner._render_blogs(blogs)
        assert "B0" in md
        assert "B4" in md
        assert "B5" not in md
        # Long summary truncated
        assert "..." in md

    def test_empty_returns_empty(self, runner):
        assert runner._render_blogs([{"score_combined": 1}]) == ""

    def test_low_score_lead_does_not_shrink_section(self, runner):
        """A weak post at the head of the list must not cost the section a slot."""
        blogs = [{"title": "Weak", "source": "S1", "score_combined": 1,
                  "brief_summary": "w"}]
        blogs += [{"title": f"Good{i}", "source": f"S{i}", "score_combined": 4,
                   "brief_summary": "g"} for i in range(5)]
        md = runner._render_blogs(blogs)
        assert "Weak" not in md
        assert md.count("Good") == 5


class TestRenderTopPapers:
    def test_filters_by_score(self, runner):
        papers = [
            {"title": "Hi", "score_combined": 5, "brief_summary": "high",
             "arxiv_url": "u", "authors": ["A"]},
            {"title": "Low", "score_combined": 1, "brief_summary": "low",
             "arxiv_url": "u", "authors": []},
        ]
        md = runner._render_top_papers(papers)
        assert "Hi" in md
        assert "Low" not in md

    def test_renders_repro_badges(self, runner):
        papers = [
            {"title": "Green", "score_combined": 5, "repro_total": 20,
             "repro_verdict": "easy", "reproduction_difficulty": "S",
             "arxiv_url": "u", "authors": []},
            {"title": "Yellow", "score_combined": 5, "repro_total": 15,
             "repro_verdict": "ok", "reproduction_difficulty": "M",
             "arxiv_url": "u", "authors": []},
            {"title": "Red", "score_combined": 5, "repro_total": 8,
             "repro_verdict": "hard", "reproduction_difficulty": "XL",
             "arxiv_url": "u", "authors": []},
        ]
        md = runner._render_top_papers(papers)
        assert "✅" in md
        assert "🟡" in md
        assert "🔴" in md

    def test_no_scored_papers_message(self, runner):
        # Papers with scores but all below 3
        papers = [{"title": "T", "score_combined": 1, "arxiv_url": "u",
                   "authors": []}]
        md = runner._render_top_papers(papers)
        assert "No highly relevant papers" in md

    def test_no_scores_honors_num_paper_picks(self, runner):
        """Unscored papers fall back to config order, capped by config.

        The fallback used to hardcode 3 regardless of num_paper_picks, so a
        briefing configured for 2 picks rendered 3 whenever scoring failed.
        """
        papers = [{"title": f"P{i}", "arxiv_url": "", "authors": []}
                  for i in range(7)]
        md = runner._render_top_papers(papers)  # cfg num_paper_picks == 5
        assert "P0" in md
        assert "P4" in md
        assert "P5" not in md

    def test_low_score_pick_is_replaced_not_dropped(self, runner):
        """A low-scoring leading pick frees a slot for a later good paper.

        Slicing to num_paper_picks before filtering discarded the candidates
        that could have taken the slot, so the section shipped short.
        """
        papers = [
            {"title": f"Good{i}", "score_combined": 5, "brief_summary": "s",
             "arxiv_url": "u", "authors": []}
            for i in range(4)
        ]
        papers.insert(1, {"title": "Weak", "score_combined": 1,
                          "brief_summary": "s", "arxiv_url": "u",
                          "authors": []})
        md = runner._render_top_papers(papers)
        assert "Weak" not in md
        assert md.count("Good") == 4


class TestRenderRecentPapers:
    def test_basic_render(self, runner):
        papers = [{"title": "T", "authors": ["A", "B"], "arxiv_url": "u",
                   "brief_summary": "summ"}]
        # Render is private but useful to test
        md = runner._render_papers(papers)
        assert "Recent Papers" in md
        assert "T" in md
        assert "[arxiv](u)" in md

    def test_includes_blurb(self, runner):
        papers = [{"title": "T", "authors": [], "arxiv_url": "",
                   "brief_summary": "", "author_blurb": "blurb"}]
        md = runner._render_papers(papers)
        assert "Source Information" in md


# ---------- generate_markdown_briefing extras ----------


class TestGenerateMarkdownBriefingExtras:
    def test_fallback_synthesis(self, runner):
        # No synthesis provided → fallback text
        md = runner.generate_markdown_briefing([], [], [], [], [])
        assert "Synthesis unavailable" in md

    def test_strips_llm_headings_from_intro(self, runner):
        synthesis = {
            "editorial_intro": (
                "# Atlas Morning Briefing\n"
                "## Executive Summary\n"
                "2026-05-22\n"
                "Real content sentence here."
            )
        }
        md = runner.generate_markdown_briefing([], [], [], [], [], synthesis)
        assert "Real content sentence" in md
        # LLM heading echo should be stripped
        assert "# Atlas Morning Briefing" not in md.split("Executive Summary")[1]

    def test_user_name_in_subtitle(self, cfg, monkeypatch):
        monkeypatch.setenv("USER_NAME", "Alice")
        r = BriefingRunner(cfg, dry_run=True)
        md = r.generate_markdown_briefing([], [], [], [], [])
        assert "Alice" in md

    def test_weekly_deep_dive_included(self, runner):
        md = runner.generate_markdown_briefing(
            [], [], [], [], [], weekly_deep_dive="Weekly content here"
        )
        assert "This Week in AI" in md
        assert "Weekly content here" in md


# ---------- score_papers / dedup wrappers ----------


class TestScorePapersWrapper:
    def test_empty_papers(self, runner):
        assert runner.score_papers([]) == []

    def test_handles_exception(self, runner, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("scoring failed")
        import scripts.briefing_runner as br
        monkeypatch.setattr(br.PaperScorer, "get_top_picks", boom)
        result = runner.score_papers([{"title": "x"}])
        assert result == []
        assert any("Paper scoring" in e for e in runner.errors)


# ---------- scan wrappers handle exceptions ----------


class TestScanWrappers:
    def test_arxiv_scan_no_topics(self, cfg):
        cfg["arxiv_topics"] = []
        r = BriefingRunner(cfg, dry_run=True)
        assert r.run_arxiv_scan() == []

    def test_blog_scan_no_feeds(self, runner):
        assert runner.run_blog_scan() == []

    def test_stock_fetch_no_api_key(self, runner, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        runner.config["stocks"] = ["AAPL"]
        assert runner.run_stock_fetch() == []

    def test_stock_fetch_no_symbols(self, runner, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "k")
        runner.config["stocks"] = []
        assert runner.run_stock_fetch() == []

    def test_news_no_api_key(self, runner, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        runner.config["news_queries"] = ["q"]
        assert runner.run_news_aggregation() == []

    def test_news_no_queries(self, runner, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "k")
        runner.config["news_queries"] = []
        assert runner.run_news_aggregation() == []

    @patch("scripts.briefing_runner.create_scanner")
    def test_arxiv_scan_exception(self, mock_create, runner):
        instance = MagicMock()
        instance.scan_all_topics.side_effect = RuntimeError("boom")
        mock_create.return_value = instance
        result = runner.run_arxiv_scan(["topic"])
        assert result == []
        assert any("ArXiv" in e for e in runner.errors)

    @patch("scripts.briefing_runner.BlogScanner")
    def test_blog_scan_exception(self, mock_cls, runner):
        runner.config["blog_feeds"] = [{"name": "X", "url": "u"}]
        instance = MagicMock()
        instance.scan_all_feeds.side_effect = RuntimeError("boom")
        mock_cls.return_value = instance
        assert runner.run_blog_scan() == []
        assert any("Blog" in e for e in runner.errors)


# ---------- PDF/EPUB gen ----------


class TestGeneratePdfWrapper:
    def test_disabled_returns_true(self, runner, tmp_path):
        runner.config["pdf"]["enabled"] = False
        # Should be a no-op success
        assert runner.generate_pdf("# x", str(tmp_path / "out.pdf")) is True

    def test_enabled_generates(self, runner, tmp_path):
        runner.config["pdf"]["enabled"] = True
        out = str(tmp_path / "out.pdf")
        assert runner.generate_pdf("# Hi\n\nBody", out) is True
        assert runner.status["pdf_generated"] is True

    def test_generation_error_recorded(self, runner, monkeypatch):
        runner.config["pdf"]["enabled"] = True
        import scripts.briefing_runner as br
        def boom(*a, **kw):
            raise RuntimeError("pdf failed")
        monkeypatch.setattr(br.PDFGenerator, "generate_pdf", boom)
        assert runner.generate_pdf("# x", "out.pdf") is False
        assert any("PDF" in e for e in runner.errors)


class TestGenerateEpubWrapper:
    def test_generates(self, runner, tmp_path):
        out = str(tmp_path / "out.epub")
        assert runner.generate_epub("# Hi", out) is True
        assert runner.status["epub_generated"] is True

    def test_generation_error(self, runner, monkeypatch):
        import scripts.briefing_runner as br
        def boom(*a, **kw):
            raise RuntimeError("epub failed")
        monkeypatch.setattr(br.EPUBGenerator, "generate_epub", boom)
        assert runner.generate_epub("# x", "out.epub") is False
        assert any("EPUB" in e for e in runner.errors)


# ---------- distribute_briefing ----------


class TestDistributeBriefing:
    def test_dry_run_skips(self, runner):
        assert runner.distribute_briefing("md", "x.pdf", "subj") == {}

    def test_no_creds_skips(self, cfg, monkeypatch):
        monkeypatch.delenv("GMAIL_USER", raising=False)
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        r = BriefingRunner(cfg, dry_run=False)
        assert r.distribute_briefing("md", "x.pdf", "subj") == {}

    @patch("scripts.briefing_runner.EmailDistributor")
    def test_calls_distributor(self, mock_cls, cfg, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "u@x.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
        cfg["kindle_email"] = "k@x.com"
        r = BriefingRunner(cfg, dry_run=False)
        inst = MagicMock()
        inst.distribute.return_value = {"kindle:k@x.com": True}
        mock_cls.return_value = inst
        results = r.distribute_briefing("md", "x.pdf", "subj")
        assert results == {"kindle:k@x.com": True}
        assert r.status["email_sent"] is True

    @patch("scripts.briefing_runner.EmailDistributor")
    def test_distribution_exception(self, mock_cls, cfg, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "u@x.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
        r = BriefingRunner(cfg, dry_run=False)
        mock_cls.side_effect = RuntimeError("boom")
        assert r.distribute_briefing("md", "x.pdf", "subj") == {}
        assert any("Distribution" in e for e in r.errors)


# ---------- _analyze_market_trend ----------


class TestAnalyzeMarketTrend:
    def test_unavailable_returns_empty(self, runner):
        # Default config has gemini disabled → intelligence unavailable
        assert runner._analyze_market_trend([{"symbol": "X"}]) == ""

    def test_no_stocks_returns_empty(self, runner):
        assert runner._analyze_market_trend([]) == ""

    def test_invokes_gemini_when_available(self, cfg):
        r = BriefingRunner(cfg, dry_run=True)
        mock_gem = MagicMock()
        mock_gem.available = True
        mock_gem.invoke.return_value = "Markets are mixed today."
        # Replace intelligence with a stub whose `available` returns True
        # without mutating the BriefingIntelligence class itself.
        r.intelligence = MagicMock()
        r.intelligence.available = True
        r.intelligence.client = mock_gem
        result = r._analyze_market_trend([{
            "symbol": "X", "percent_change": 1.5, "news_correlation": "ok"
        }])
        assert "mixed" in result


# ---------- load_config & main ----------


class TestLoadConfigBriefing:
    def test_env_expansion(self, tmp_path, monkeypatch):
        f = tmp_path / "c.yaml"
        f.write_text("key: ${TEST_VAR}\n")
        monkeypatch.setenv("TEST_VAR", "from_env")
        assert load_config(str(f))["key"] == "from_env"

    def test_default_value(self, tmp_path, monkeypatch):
        f = tmp_path / "c.yaml"
        f.write_text("key: ${MISSING_VAR:-fallback}\n")
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert load_config(str(f))["key"] == "fallback"

    def test_no_substitution_when_no_default(self, tmp_path, monkeypatch):
        f = tmp_path / "c.yaml"
        f.write_text("key: \"${UNSET_VAR}\"\n")
        monkeypatch.delenv("UNSET_VAR", raising=False)
        # Original placeholder kept if no default & no env
        assert load_config(str(f))["key"] == "${UNSET_VAR}"

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_config(str(tmp_path / "nope.yaml"))

    def test_malformed_yaml_exits(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text(":not valid:\n  -[\n}")
        with pytest.raises(SystemExit):
            load_config(str(f))


class TestMain:
    def test_invalid_config_returns_2(self, tmp_path, monkeypatch):
        f = tmp_path / "c.yaml"
        # arxiv_topics must be a list
        f.write_text("arxiv_topics: not_a_list\n")
        monkeypatch.setattr("sys.argv", ["b.py", "--config", str(f), "--dry-run"])
        assert main() == 2

    @patch("scripts.briefing_runner.BriefingRunner")
    def test_calls_run_on_valid_config(self, mock_cls, tmp_path, monkeypatch):
        f = tmp_path / "c.yaml"
        f.write_text("arxiv_topics:\n  - X\n")
        instance = MagicMock()
        instance.run.return_value = 0
        mock_cls.return_value = instance
        monkeypatch.setattr("sys.argv", ["b.py", "--config", str(f), "--dry-run"])
        assert main() == 0
        instance.run.assert_called_once()
