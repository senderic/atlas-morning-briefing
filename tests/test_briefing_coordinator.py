"""Tests for BriefingCoordinator (scripts/briefing_runner_v2.py)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.briefing_runner_v2 import BriefingCoordinator, STATE_FILENAME


@pytest.fixture
def stub_llm():
    llm = MagicMock()
    llm.available = True
    llm.invoke.return_value = "stub-response"
    llm.usage_stats = {
        "heavy": {"in_tokens": 0, "out_tokens": 0},
        "medium": {"in_tokens": 0, "out_tokens": 0},
        "light": {"in_tokens": 0, "out_tokens": 0},
    }
    return llm


@pytest.fixture
def base_config():
    return {
        "arxiv_topics": ["agents"],
        "max_workers": 1,
        "pdf": {"enabled": False},
        "gemini": {"enabled": False},  # so coordinator falls into BedrockClient path
        "bedrock": {"enabled": False},
        "output_format": "kindle",
    }


@pytest.fixture
def coordinator(base_config, stub_llm):
    coord = BriefingCoordinator(base_config, dry_run=True)
    coord.llm = stub_llm  # override after construction
    return coord


# --- _extract_items ----------------------------------------------------------

def test_extract_items_separates_workers(coordinator):
    findings = [
        {"worker": "papers_worker", "items": [{"title": "p1"}]},
        {"worker": "blogs_worker", "items": [{"title": "b1"}]},
        {"worker": "news_market_worker", "items": {
            "news": [{"title": "n1"}], "stocks": [{"symbol": "NVDA"}]
        }},
    ]
    papers, blogs, news, stocks = coordinator._extract_items(findings)
    assert papers == [{"title": "p1"}]
    assert blogs == [{"title": "b1"}]
    assert news == [{"title": "n1"}]
    assert stocks == [{"symbol": "NVDA"}]


def test_extract_items_handles_missing_worker(coordinator):
    # Only papers worker reported; others should default to empty.
    findings = [{"worker": "papers_worker", "items": [{"title": "x"}]}]
    papers, blogs, news, stocks = coordinator._extract_items(findings)
    assert papers == [{"title": "x"}]
    assert blogs == []
    assert news == []
    assert stocks == []


def test_extract_items_handles_news_market_with_only_news(coordinator):
    findings = [{"worker": "news_market_worker", "items": {"news": [{"title": "x"}]}}]
    _, _, news, stocks = coordinator._extract_items(findings)
    assert news == [{"title": "x"}]
    assert stocks == []


# --- _detect_emerging_themes ------------------------------------------------

def test_emerging_themes_returns_empty_when_no_input(coordinator):
    result = coordinator._detect_emerging_themes([], [], [])
    assert result == []


def test_emerging_themes_returns_empty_when_llm_unavailable(coordinator):
    coordinator.llm.available = False
    result = coordinator._detect_emerging_themes(
        [{"title": "p"}], [{"title": "b"}], [{"title": "n"}]
    )
    assert result == []
    coordinator.llm.invoke.assert_not_called()


def test_emerging_themes_consumes_papers_blogs_news(coordinator):
    coordinator.llm.invoke.return_value = "agents, evaluation, robotics"
    result = coordinator._detect_emerging_themes(
        [{"title": "Paper A", "score": 5}],
        [{"title": "Blog B"}],
        [{"title": "News C"}],
    )
    assert result == ["agents", "evaluation", "robotics"]
    # Verify all 3 sources were referenced in the prompt.
    prompt = coordinator.llm.invoke.call_args[0][0]
    assert "Paper A" in prompt
    assert "Blog B" in prompt
    assert "News C" in prompt
    assert "[paper]" in prompt
    assert "[blog]" in prompt
    assert "[news]" in prompt


def test_emerging_themes_filters_empty_strings(coordinator):
    coordinator.llm.invoke.return_value = "agents, , robotics, "
    result = coordinator._detect_emerging_themes([{"title": "p", "score": 1}], [], [])
    assert result == ["agents", "robotics"]


def test_emerging_themes_returns_empty_on_falsy_response(coordinator):
    coordinator.llm.invoke.return_value = None
    result = coordinator._detect_emerging_themes([{"title": "p", "score": 1}], [], [])
    assert result == []


# --- _generate_executive_summary --------------------------------------------

def test_executive_summary_returns_offline_marker_without_llm(coordinator):
    coordinator.llm.available = False
    result = coordinator._generate_executive_summary({}, [], [])
    assert result == "LLM offline"


def test_executive_summary_includes_themes_and_worker_summaries(coordinator):
    coordinator.llm.invoke.return_value = "Today the AI scene moved..."
    result = coordinator._generate_executive_summary(
        {"papers_worker": "5 papers", "blogs_worker": "3 blogs"},
        ["agents", "robotics"],
        [],
    )
    assert result == "Today the AI scene moved..."
    prompt = coordinator.llm.invoke.call_args[0][0]
    assert "agents, robotics" in prompt
    assert "5 papers" in prompt
    assert "3 blogs" in prompt


def test_executive_summary_falls_back_when_llm_returns_empty(coordinator):
    coordinator.llm.invoke.return_value = None
    result = coordinator._generate_executive_summary({"papers_worker": "x"}, [], [])
    assert result == "Synthesis failed"


# --- _analyze_market_trend --------------------------------------------------

def test_market_trend_skipped_without_stocks(coordinator):
    assert coordinator._analyze_market_trend([], []) == ""
    coordinator.llm.invoke.assert_not_called()


def test_market_trend_skipped_when_llm_unavailable(coordinator):
    coordinator.llm.available = False
    assert coordinator._analyze_market_trend([{"symbol": "NVDA"}], []) == ""


def test_market_trend_includes_each_symbol_in_prompt(coordinator):
    coordinator.llm.invoke.return_value = "Tech leads"
    result = coordinator._analyze_market_trend(
        [{"symbol": "NVDA", "percent_change": 2.5}, {"symbol": "MSFT", "percent_change": -1.0}],
        [],
    )
    assert result == "Tech leads"
    prompt = coordinator.llm.invoke.call_args[0][0]
    assert "NVDA" in prompt and "MSFT" in prompt


# --- _generate_briefing -----------------------------------------------------

def test_generate_briefing_renders_all_sections(coordinator):
    synthesis = {
        "executive_summary": "Today's exec summary",
        "market_trend": "Tech up",
    }
    papers = [{
        "title": "PaperX",
        "score": 5.0,
        "brief_summary": "novel agent system",
        "arxiv_url": "http://arxiv.org/abs/2401.0001",
    }]
    blogs = [{
        "title": "BlogY",
        "brief_summary": "post about agents",
        "link": "http://blog.example/y",
        "source": "Source A",
    }]
    news = [{
        "title": "NewsZ",
        "brief_summary": "AI funding round",
        "url": "http://news.example/z",
    }]
    stocks = [{"symbol": "NVDA", "percent_change": 2.5}]

    content = coordinator._generate_briefing(synthesis, papers, blogs, news, stocks)
    assert "Today's exec summary" in content
    assert "## Markets" in content
    assert "NVDA" in content
    assert "## News" in content
    assert "NewsZ" in content
    assert "AI funding round" in content
    assert "## Research" in content
    assert "PaperX" in content
    assert "novel agent system" in content
    assert "http://arxiv.org/abs/2401.0001" in content
    assert "## Blogs" in content
    assert "BlogY" in content
    assert "Source A" in content


def test_generate_briefing_skips_empty_sections(coordinator):
    synthesis = {"executive_summary": "x", "market_trend": ""}
    content = coordinator._generate_briefing(synthesis, [], [], [], [])
    assert "## Executive Summary" in content
    assert "## News" not in content
    assert "## Markets" not in content
    assert "## Research" not in content
    assert "## Blogs" not in content


def test_generate_briefing_falls_back_when_brief_summary_missing(coordinator):
    # Confirms the fallback chain: brief_summary -> description -> snippet.
    synthesis = {"executive_summary": "x", "market_trend": ""}
    news = [{"title": "n", "url": "http://x", "description": "raw description"}]
    papers = [{"title": "p", "score": 1.0, "summary": "raw abstract" * 50,
               "arxiv_url": "http://arxiv/abs/1"}]
    blogs = [{"title": "b", "summary": "raw blog body" * 20, "link": "http://b"}]
    content = coordinator._generate_briefing(synthesis, papers, blogs, news, [])
    assert "raw description" in content
    assert "raw abstract" in content
    assert "raw blog body" in content


def test_generate_briefing_handles_missing_paper_link_fields(coordinator):
    synthesis = {"executive_summary": "x", "market_trend": ""}
    papers = [{"title": "p", "score": 1.0, "brief_summary": "s"}]  # no arxiv_url/pdf_link/id
    content = coordinator._generate_briefing(synthesis, papers, [], [], [])
    # The link should render as empty rather than literal "None".
    assert "[ArXiv]()" in content
    assert "None" not in content.replace("\n", " ")  # no leaking None


# --- _generate_pdf / _distribute --------------------------------------------

def test_generate_pdf_skipped_when_disabled(coordinator):
    coordinator.config["pdf"] = {"enabled": False}
    with patch("scripts.briefing_runner_v2.PDFGenerator") as gen_mock:
        result = coordinator._generate_pdf("# content", "Atlas-Test")
    assert result is None
    gen_mock.assert_not_called()


def test_generate_pdf_invoked_when_enabled(coordinator, tmp_path):
    coordinator.config["pdf"] = {
        "enabled": True, "font_size": 11, "line_spacing": 1.4, "include_toc": False
    }
    with patch("scripts.briefing_runner_v2.PDFGenerator") as gen_cls:
        gen_inst = MagicMock()
        gen_cls.return_value = gen_inst
        result = coordinator._generate_pdf("# content", str(tmp_path / "Atlas-Test"))
    gen_cls.assert_called_once_with(
        page_format="kindle", font_size=11, line_spacing=1.4, include_toc=False
    )
    gen_inst.generate_pdf.assert_called_once()
    assert isinstance(result, Path)


def test_distribute_skipped_without_credentials(coordinator, monkeypatch):
    monkeypatch.delenv("GMAIL_USER", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    with patch("scripts.briefing_runner_v2.EmailDistributor") as dist_mock:
        coordinator._distribute("content", "filename", "pdf.pdf", "epub.epub")
    dist_mock.assert_not_called()


def test_distribute_called_with_credentials(coordinator, monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "u@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "secret")
    with patch("scripts.briefing_runner_v2.EmailDistributor") as dist_cls:
        dist_inst = MagicMock()
        dist_cls.return_value = dist_inst
        coordinator._distribute("content", "filename", "pdf.pdf", "epub.epub")
    dist_cls.assert_called_once_with(sender_email="u@example.com", sender_password="secret")
    dist_inst.distribute.assert_called_once_with(
        coordinator.config, "content", "pdf.pdf", "epub.epub", "filename"
    )


# --- _save_state and _load_memory -------------------------------------------

def test_save_state_writes_all_required_keys(coordinator, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    papers = [{"title": "P1"}, {"title": "P2"}]
    blogs = [{"title": "B1"}]
    news = [{"title": "N1"}]
    stocks = [{"symbol": "NVDA", "current_price": 1000.0}]
    synthesis = {"emerging_themes": ["theme1"]}
    coordinator._save_state(papers, blogs, news, stocks=stocks, synthesis=synthesis)
    state = json.loads(Path(STATE_FILENAME).read_text())
    assert state["top_paper_titles"] == ["P1", "P2"]
    assert state["top_blog_titles"] == ["B1"]
    assert state["top_news_titles"] == ["N1"]
    assert state["emerging_themes"] == ["theme1"]
    assert state["stock_closes"] == {"NVDA": 1000.0}
    assert state["trending_topics"] == {}  # no prior state
    assert "date" in state


def test_save_state_carries_forward_trending_topics(coordinator, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Seed an existing state with trending_topics.
    Path(STATE_FILENAME).write_text(json.dumps({
        "trending_topics": {"agents": {"count": 3, "first_seen": "2026-05-01", "last_seen": "2026-05-15"}}
    }))
    coordinator._save_state([], [], [], stocks=[], synthesis={})
    state = json.loads(Path(STATE_FILENAME).read_text())
    assert state["trending_topics"]["agents"]["count"] == 3


def test_load_memory_handles_missing_file(coordinator, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert coordinator._load_memory() == {}


def test_load_memory_handles_corrupt_json(coordinator, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path(STATE_FILENAME).write_text("not valid json{{{")
    assert coordinator._load_memory() == {}


def test_load_memory_returns_parsed_state(coordinator, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = {"date": "2026-05-15", "top_paper_titles": ["X"]}
    Path(STATE_FILENAME).write_text(json.dumps(payload))
    assert coordinator._load_memory() == payload


# --- _spawn_workers honors max_workers ---------------------------------------

def test_spawn_workers_runs_with_configured_max_workers(coordinator):
    coordinator.config["max_workers"] = 1
    # Mock the worker classes so we don't need real scanners.
    with patch("scripts.briefing_runner_v2.PapersWorker") as papers_mock, \
         patch("scripts.briefing_runner_v2.BlogsWorker") as blogs_mock, \
         patch("scripts.briefing_runner_v2.NewsMarketWorker") as news_mock:
        for w in (papers_mock, blogs_mock, news_mock):
            inst = MagicMock()
            inst.execute.return_value = {
                "worker": "x", "status": "success",
                "items": [], "metadata": {"token_count": 0}, "synthesis": "",
            }
            inst.worker_name = "x"
            w.return_value = inst
        findings = coordinator._spawn_workers()
    assert len(findings) == 3


def test_spawn_workers_records_failure_finding(coordinator):
    with patch("scripts.briefing_runner_v2.PapersWorker") as papers_mock, \
         patch("scripts.briefing_runner_v2.BlogsWorker") as blogs_mock, \
         patch("scripts.briefing_runner_v2.NewsMarketWorker") as news_mock:
        # papers raises; others succeed.
        bad = MagicMock()
        bad.execute.side_effect = RuntimeError("boom")
        bad.worker_name = "papers_worker"
        papers_mock.return_value = bad
        for w in (blogs_mock, news_mock):
            inst = MagicMock()
            inst.execute.return_value = {
                "worker": "x", "status": "success", "items": [],
                "metadata": {"token_count": 0}, "synthesis": "",
            }
            inst.worker_name = "x"
            w.return_value = inst
        findings = coordinator._spawn_workers()
    statuses = [f["status"] for f in findings]
    assert "error" in statuses
    assert statuses.count("success") == 2


# --- run() smoke test --------------------------------------------------------

def test_run_aborts_when_all_workers_fail(coordinator, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    failing_finding = {
        "worker": "x", "status": "error", "items": [],
        "metadata": {"token_count": 0}, "synthesis": "", "error": "boom",
    }
    with patch.object(coordinator, "_spawn_workers", return_value=[failing_finding] * 3):
        rc = coordinator.run()
    assert rc == 2  # abort code per the runner


def _success_finding(worker_name, items=None):
    return {
        "worker": worker_name,
        "status": "success",
        "items": items if items is not None else [],
        "metadata": {"token_count": 100},
        "synthesis": f"{worker_name} synthesis",
    }


def test_run_succeeds_end_to_end_writes_md_and_state(
    coordinator, tmp_path, monkeypatch, stub_llm
):
    """Smoke test the full happy path: workers succeed, briefing is written, state saved."""
    monkeypatch.chdir(tmp_path)
    # Make sure pdf/epub/distribute are skipped via dry_run + pdf disabled.
    coordinator.dry_run = True
    coordinator.config["pdf"] = {"enabled": False}

    findings = [
        _success_finding("papers_worker", [{
            "title": "PaperX", "score": 5.0, "brief_summary": "agent system",
            "arxiv_url": "http://arxiv/abs/x", "summary": "long abstract",
        }]),
        _success_finding("blogs_worker", [{
            "title": "BlogY", "brief_summary": "agent post",
            "link": "http://b/y", "source": "S",
        }]),
        _success_finding("news_market_worker", {
            "news": [{"title": "NewsZ", "brief_summary": "AI news", "url": "http://n"}],
            "stocks": [{"symbol": "NVDA", "current_price": 1000.0, "percent_change": 2.5}],
        }),
    ]

    stub_llm.invoke.return_value = "stub-response"  # for synthesis calls

    with patch.object(coordinator, "_spawn_workers", return_value=findings), \
         patch("scripts.briefing_runner_v2.EPUBGenerator") as epub_cls:
        # Stub EPUBGenerator so we don't actually write a real epub.
        epub_inst = MagicMock()
        epub_cls.return_value = epub_inst
        rc = coordinator.run()

    assert rc == 0
    # The .md file should have landed in the cwd.
    md_files = list(tmp_path.glob("Atlas-Briefing-*.md"))
    assert len(md_files) == 1
    md_content = md_files[0].read_text()
    assert "PaperX" in md_content
    assert "BlogY" in md_content
    assert "NewsZ" in md_content
    # State file persisted.
    state_path = tmp_path / STATE_FILENAME
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert state["top_paper_titles"] == ["PaperX"]
    assert state["stock_closes"] == {"NVDA": 1000.0}


def test_run_returns_1_when_some_workers_fail(coordinator, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    coordinator.dry_run = True
    coordinator.config["pdf"] = {"enabled": False}
    findings = [
        _success_finding("papers_worker"),
        _success_finding("blogs_worker"),
        {"worker": "news_market_worker", "status": "error", "items": {"news": [], "stocks": []},
         "metadata": {"token_count": 0}, "synthesis": "", "error": "x"},
    ]
    with patch.object(coordinator, "_spawn_workers", return_value=findings), \
         patch("scripts.briefing_runner_v2.EPUBGenerator"):
        rc = coordinator.run()
    assert rc == 1


# --- main() entry point ------------------------------------------------------

def test_main_aborts_on_invalid_config(tmp_path, monkeypatch):
    """main() should exit with code 2 when validate_config flags errors."""
    from scripts.briefing_runner_v2 import main
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("arxiv_topics: not_a_list\n")
    monkeypatch.setattr("sys.argv", ["briefing_runner_v2.py", "--config", str(bad_config)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_main_runs_dry_run_path(tmp_path, monkeypatch):
    """main() with --dry-run should construct the coordinator and run."""
    from scripts.briefing_runner_v2 import main
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "arxiv_topics: [agents]\n"
        "max_workers: 1\n"
        "pdf: {enabled: false}\n"
        "gemini: {enabled: false}\n"
        "bedrock: {enabled: false}\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "briefing_runner_v2.py", "--config", str(config_path), "--dry-run"
    ])
    # Stub the spawn so we don't actually hit the network or LLMs.
    with patch("scripts.briefing_runner_v2.BriefingCoordinator._spawn_workers",
               return_value=[_success_finding("papers_worker"),
                             _success_finding("blogs_worker"),
                             _success_finding("news_market_worker",
                                              {"news": [], "stocks": []})]), \
         patch("scripts.briefing_runner_v2.EPUBGenerator"):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code in (0, 1)


# --- gemini config branch in BriefingCoordinator.__init__ -------------------

def test_init_uses_gemini_when_enabled(monkeypatch):
    """Verify the coordinator picks GeminiCLIClient when gemini.enabled is set."""
    from scripts.briefing_runner_v2 import BriefingCoordinator
    with patch("scripts.briefing_runner_v2.GeminiCLIClient") as gem_cls:
        gem_cls.return_value = MagicMock()
        coord = BriefingCoordinator({"gemini": {"enabled": True}}, dry_run=True)
    gem_cls.assert_called_once_with({"enabled": True})
    assert coord.llm is gem_cls.return_value