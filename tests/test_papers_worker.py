"""Tests for PapersWorker.

Mocks the ArXiv scanner and intelligence layer so we can exercise the
worker's wiring (config plumbing, fallback path when intelligence is
unavailable, error handling, synthesis generation) deterministically.
"""

from unittest.mock import MagicMock, patch

import pytest

from scripts.workers.papers_worker import PapersWorker


@pytest.fixture
def base_config():
    return {
        "arxiv_topics": ["agents", "evaluation"],
        "arxiv_days_back": 3,
        "max_papers": 30,
        "num_paper_picks": 3,
        "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2},
        "interest_profile": [{"topic": "agents", "weight": 1.0}],
    }


@pytest.fixture
def stub_llm():
    llm = MagicMock()
    llm.usage_stats = {
        "heavy": {"in_tokens": 0, "out_tokens": 0},
        "medium": {"in_tokens": 0, "out_tokens": 0},
        "light": {"in_tokens": 0, "out_tokens": 0},
    }
    return llm


@pytest.fixture
def fake_papers():
    return [
        {
            "title": f"Paper {i}",
            "summary": f"abstract {i} with code at github.com/x/y",
            "published": "2026-05-01T00:00:00Z",
            "categories": ["cs.AI"],
            "arxiv_url": f"http://arxiv.org/abs/240{i}.0001",
            "id": f"http://arxiv.org/abs/240{i}.0001",
        }
        for i in range(1, 6)
    ]


def _patch_scanner(papers):
    """Helper: patch create_scanner so scan_all_topics returns `papers`."""
    scanner = MagicMock()
    scanner.scan_all_topics.return_value = papers
    return patch(
        "scripts.workers.papers_worker.create_scanner",
        return_value=scanner,
    )


def test_empty_scan_returns_no_items(base_config, stub_llm):
    with _patch_scanner([]):
        worker = PapersWorker(base_config, llm_client=stub_llm)
        finding = worker.execute()
    assert finding["status"] == "success"
    assert finding["items"] == []
    assert "No papers found" in finding["synthesis"]


def test_no_intelligence_falls_back_to_tfidf_only(base_config, fake_papers, stub_llm):
    # Make the intelligence layer claim it's unavailable.
    intel = MagicMock()
    intel.available = False

    with _patch_scanner(fake_papers), \
         patch("scripts.workers.papers_worker.BriefingIntelligence", return_value=intel):
        worker = PapersWorker(base_config, llm_client=stub_llm)
        finding = worker.execute()

    assert finding["status"] == "success"
    # Should have run PaperScorer (scores added).
    assert all("score" in p for p in finding["items"])
    assert "LLM enrichment unavailable" in finding["synthesis"]
    # Worker shouldn't have invoked the intelligence layer.
    intel.summarize_papers.assert_not_called()


def test_intelligence_path_invokes_filter_summarize_score(
    base_config, fake_papers, stub_llm
):
    intel = MagicMock()
    intel.available = True
    intel.filter_papers_by_relevance.return_value = fake_papers[:3]
    intel.summarize_papers.return_value = fake_papers[:3]
    intel.score_papers_semantically.return_value = fake_papers[:3]

    with _patch_scanner(fake_papers), \
         patch("scripts.workers.papers_worker.BriefingIntelligence", return_value=intel):
        worker = PapersWorker(base_config, llm_client=stub_llm)
        finding = worker.execute()

    assert finding["status"] == "success"
    intel.filter_papers_by_relevance.assert_called_once()
    intel.summarize_papers.assert_called_once()
    intel.score_papers_semantically.assert_called_once()
    # Token count should be the delta in usage_stats (0 here since stub didn't
    # update them — verifies we read the snapshot rather than fabricating).
    assert finding["metadata"]["token_count"] == 0


def test_skip_relevance_filter_when_no_interest_profile(
    base_config, fake_papers, stub_llm
):
    base_config.pop("interest_profile")
    intel = MagicMock()
    intel.available = True
    intel.summarize_papers.return_value = fake_papers
    intel.score_papers_semantically.return_value = fake_papers

    with _patch_scanner(fake_papers), \
         patch("scripts.workers.papers_worker.BriefingIntelligence", return_value=intel):
        worker = PapersWorker(base_config, llm_client=stub_llm)
        finding = worker.execute()

    assert finding["status"] == "success"
    intel.filter_papers_by_relevance.assert_not_called()


def test_token_count_reflects_actual_usage_delta(
    base_config, fake_papers, stub_llm
):
    intel = MagicMock()
    intel.available = True
    intel.filter_papers_by_relevance.return_value = fake_papers
    # Simulate the LLM consuming tokens during enrichment.

    def bump_tokens_then_return(papers):
        stub_llm.usage_stats["medium"]["in_tokens"] += 800
        stub_llm.usage_stats["medium"]["out_tokens"] += 200
        return papers

    intel.summarize_papers.side_effect = bump_tokens_then_return
    intel.score_papers_semantically.return_value = fake_papers

    with _patch_scanner(fake_papers), \
         patch("scripts.workers.papers_worker.BriefingIntelligence", return_value=intel):
        worker = PapersWorker(base_config, llm_client=stub_llm)
        finding = worker.execute()
    assert finding["metadata"]["token_count"] == 1000


def test_scanner_exception_returns_error_finding(base_config, stub_llm):
    scanner = MagicMock()
    scanner.scan_all_topics.side_effect = RuntimeError("network down")
    with patch("scripts.workers.papers_worker.create_scanner", return_value=scanner):
        worker = PapersWorker(base_config, llm_client=stub_llm)
        finding = worker.execute()
    assert finding["status"] == "error"
    assert "network down" in finding["error"]
    assert finding["items"] == []


def test_synthesis_string_includes_top_paper_title(
    base_config, fake_papers, stub_llm
):
    intel = MagicMock()
    intel.available = True
    intel.filter_papers_by_relevance.return_value = fake_papers
    intel.summarize_papers.return_value = fake_papers

    # Inject a synthesizable score so PaperScorer ranks correctly.
    def add_semantic(papers, topics):
        for p in papers:
            p["semantic_score"] = 5.0
        return papers

    intel.score_papers_semantically.side_effect = add_semantic

    with _patch_scanner(fake_papers), \
         patch("scripts.workers.papers_worker.BriefingIntelligence", return_value=intel):
        worker = PapersWorker(base_config, llm_client=stub_llm)
        finding = worker.execute()
    assert "Top paper" in finding["synthesis"]
    assert "score" in finding["synthesis"]


def test_finding_metadata_records_items_found_vs_kept(
    base_config, fake_papers, stub_llm
):
    intel = MagicMock()
    intel.available = True
    intel.filter_papers_by_relevance.return_value = fake_papers[:2]  # 5 -> 2
    intel.summarize_papers.return_value = fake_papers[:2]
    intel.score_papers_semantically.return_value = fake_papers[:2]

    with _patch_scanner(fake_papers), \
         patch("scripts.workers.papers_worker.BriefingIntelligence", return_value=intel):
        worker = PapersWorker(base_config, llm_client=stub_llm)
        finding = worker.execute()
    assert finding["metadata"]["items_found"] == 5
    assert finding["metadata"]["items_kept"] == 2
