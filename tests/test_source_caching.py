# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for source blurb caching and deduplication."""

import pytest
from unittest.mock import MagicMock
from scripts.intelligence import BriefingIntelligence
from scripts.llm_client import BaseLLMClient

@pytest.fixture
def mock_client():
    client = MagicMock(spec=BaseLLMClient)
    client.available = True
    return client

@pytest.fixture
def intelligence(mock_client):
    config = {"arxiv_topics": ["AI"]}
    return BriefingIntelligence(mock_client, config)

def test_caching_across_calls(intelligence, mock_client):
    """Test that blurbs are cached and reused across different calls."""
    items1 = [{"title": "News 1", "source": "TechCrunch"}]
    # 1st call: extract_missing_authors (light), 2nd call: generate_author_blurbs (light)
    mock_client.invoke.side_effect = ["TechCrunch", "[1] TechCrunch is a leading technology news website."]

    # First call: should fetch from LLM
    result1 = intelligence.generate_author_blurbs(items1, "news")
    assert result1[0]["author_blurb"] == "TechCrunch is a leading technology news website."
    assert mock_client.invoke.call_count == 2

    # Second call with same source: should use cache
    # It still calls extract_missing_authors, but generate_author_blurbs should skip the light tier call for cached sources
    items2 = [{"title": "News 2", "source": "TechCrunch"}]
    mock_client.invoke.side_effect = ["TechCrunch"]
    result2 = intelligence.generate_author_blurbs(items2, "news")
    assert result2[0]["author_blurb"] == "TechCrunch is a leading technology news website."
    assert mock_client.invoke.call_count == 3 # 2 from first call + 1 from second

def test_deduplication_within_single_call(intelligence, mock_client):
    """Test that multiple items from the same source in one call only result in one LLM lookup."""
    items = [
        {"title": "News 1", "source": "TechCrunch"},
        {"title": "News 2", "source": "TechCrunch"},
        {"title": "News 3", "source": "The Verge"}
    ]
    # 1st call: extract_missing_authors (light), 2nd call: generate_author_blurbs (light)
    mock_client.invoke.side_effect = [
        "TechCrunch\nTechCrunch\nThe Verge",
        "[1] Blurb for TechCrunch.\n[2] Blurb for The Verge."
    ]

    result = intelligence.generate_author_blurbs(items, "news")

    assert len(result) == 3
    assert result[0]["author_blurb"] == "Blurb for TechCrunch."
    assert result[1]["author_blurb"] == "Blurb for TechCrunch."
    assert result[2]["author_blurb"] == "Blurb for The Verge."

    # Verify LLM light tier was called with only 2 items
    calls = mock_client.invoke.call_args_list
    assert len(calls) == 2
    assert calls[1].kwargs["tier"] == "light"
    prompt = calls[1].args[0]
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "[3]" not in prompt

def test_case_insensitivity_and_normalization(intelligence, mock_client):
    """Test that normalization (lowercase, stripping) works for caching."""
    items1 = [{"title": "News 1", "source": "  TechCrunch  "}]
    mock_client.invoke.side_effect = ["TechCrunch", "[1] Blurb for TechCrunch."]

    intelligence.generate_author_blurbs(items1, "news")
    assert mock_client.invoke.call_count == 2

    items2 = [{"title": "News 2", "source": "techcrunch"}]
    mock_client.invoke.side_effect = ["techcrunch"]
    result2 = intelligence.generate_author_blurbs(items2, "news")
    
    assert result2[0]["author_blurb"] == "Blurb for TechCrunch."
    assert mock_client.invoke.call_count == 3
