"""Tests for BlogsWorker."""

from unittest.mock import MagicMock, patch

import pytest

from scripts.workers.blogs_worker import BlogsWorker


@pytest.fixture
def base_config():
    return {
        "blog_feeds": [{"name": "Test", "url": "http://example.com/feed"}],
        "max_blogs": 5,
        "arxiv_topics": ["agents"],
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
def fake_blogs():
    return [
        {
            "title": f"Blog {i}",
            "summary": f"raw blog summary {i}",
            "link": f"http://example.com/post-{i}",
            "source": "Source A" if i % 2 == 0 else "Source B",
            "published": "2026-05-01T00:00:00Z",
        }
        for i in range(1, 6)
    ]


def _patch_scanner(blogs):
    scanner = MagicMock()
    scanner.scan_all_feeds.return_value = blogs
    return patch("scripts.workers.blogs_worker.BlogScanner", return_value=scanner)


def test_empty_scan_returns_no_items(base_config, stub_llm):
    with _patch_scanner([]):
        finding = BlogsWorker(base_config, llm_client=stub_llm).execute()
    assert finding["status"] == "success"
    assert finding["items"] == []
    assert "No blog posts found" in finding["synthesis"]


def test_no_intelligence_returns_raw_blogs_capped(base_config, fake_blogs, stub_llm):
    intel = MagicMock()
    intel.available = False
    with _patch_scanner(fake_blogs), \
         patch("scripts.workers.blogs_worker.BriefingIntelligence", return_value=intel):
        finding = BlogsWorker(base_config, llm_client=stub_llm).execute()
    assert finding["status"] == "success"
    # Capped at max_blogs=5; 5 input items so all returned.
    assert len(finding["items"]) == 5
    assert "LLM enrichment unavailable" in finding["synthesis"]


def test_intelligence_path_filters_by_score_combined(
    base_config, fake_blogs, stub_llm
):
    # Intelligence enriches and adds score_combined to some blogs.
    enriched = [
        {**fake_blogs[0], "score_combined": 5, "brief_summary": "high"},
        {**fake_blogs[1], "score_combined": 4, "brief_summary": "good"},
        {**fake_blogs[2], "score_combined": 2, "brief_summary": "low"},  # below 3
    ]
    intel = MagicMock()
    intel.available = True
    intel.rank_and_summarize_blogs.return_value = enriched

    with _patch_scanner(fake_blogs), \
         patch("scripts.workers.blogs_worker.BriefingIntelligence", return_value=intel):
        finding = BlogsWorker(base_config, llm_client=stub_llm).execute()

    titles = [b["title"] for b in finding["items"]]
    # The score-2 blog should be filtered out by the >=3 gate.
    assert "Blog 3" not in titles
    assert finding["items"][0]["score_combined"] >= finding["items"][-1]["score_combined"]


def test_no_score_filter_when_no_scores_present(
    base_config, fake_blogs, stub_llm
):
    # If intelligence returns blogs without score_combined, the gate is skipped.
    enriched = [{**b, "brief_summary": f"summary for {b['title']}"} for b in fake_blogs]
    intel = MagicMock()
    intel.available = True
    intel.rank_and_summarize_blogs.return_value = enriched

    with _patch_scanner(fake_blogs), \
         patch("scripts.workers.blogs_worker.BriefingIntelligence", return_value=intel):
        finding = BlogsWorker(base_config, llm_client=stub_llm).execute()
    # All 5 should survive — no score_combined means no filtering.
    assert len(finding["items"]) == 5


def test_max_blogs_cap_honored(base_config, fake_blogs, stub_llm):
    base_config["max_blogs"] = 2
    enriched = [{**b, "score_combined": 4, "brief_summary": "x"} for b in fake_blogs]
    intel = MagicMock()
    intel.available = True
    intel.rank_and_summarize_blogs.return_value = enriched

    with _patch_scanner(fake_blogs), \
         patch("scripts.workers.blogs_worker.BriefingIntelligence", return_value=intel):
        finding = BlogsWorker(base_config, llm_client=stub_llm).execute()
    assert len(finding["items"]) == 2


def test_token_count_reflects_usage_delta(base_config, fake_blogs, stub_llm):
    intel = MagicMock()
    intel.available = True

    def bump(blogs, topics):
        stub_llm.usage_stats["light"]["in_tokens"] += 300
        stub_llm.usage_stats["light"]["out_tokens"] += 150
        return [{**b, "score_combined": 4} for b in blogs]

    intel.rank_and_summarize_blogs.side_effect = bump
    with _patch_scanner(fake_blogs), \
         patch("scripts.workers.blogs_worker.BriefingIntelligence", return_value=intel):
        finding = BlogsWorker(base_config, llm_client=stub_llm).execute()
    assert finding["metadata"]["token_count"] == 450


def test_scanner_exception_returns_error(base_config, stub_llm):
    scanner = MagicMock()
    scanner.scan_all_feeds.side_effect = RuntimeError("RSS down")
    with patch("scripts.workers.blogs_worker.BlogScanner", return_value=scanner):
        finding = BlogsWorker(base_config, llm_client=stub_llm).execute()
    assert finding["status"] == "error"
    assert "RSS down" in finding["error"]
