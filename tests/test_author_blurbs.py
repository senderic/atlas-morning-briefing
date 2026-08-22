# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for author blurb generation in intelligence module."""

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

def test_generate_author_blurbs_papers(intelligence, mock_client):
    items = [
        {"title": "Paper 1", "authors": ["Author A", "Author B"]},
        {"title": "Paper 2", "authors": ["Author C"]}
    ]
    mock_client.invoke.return_value = "[1] Blurb for Paper 1.\n[2] Blurb for Paper 2."

    result = intelligence.generate_author_blurbs(items, "papers")

    assert len(result) == 2
    assert result[0]["author_blurb"] == "Blurb for Paper 1."
    assert result[1]["author_blurb"] == "Blurb for Paper 2."

    # Check if correct tier was used (light: blurbs are trivial factual lookups)
    mock_client.invoke.assert_called_once()
    args, kwargs = mock_client.invoke.call_args
    assert kwargs["tier"] == "light"
    assert "reputable sources" in args[0]
    assert "PBS, NPR, NYT" in args[0]

def test_generate_author_blurbs_blogs(intelligence, mock_client):
    items = [
        {"title": "Blog 1", "author": "Blogger X", "source": "AI Weekly"}
    ]
    mock_client.invoke.return_value = "[1] Blurb for Blog 1."

    result = intelligence.generate_author_blurbs(items, "blogs")

    assert len(result) == 1
    assert result[0]["author_blurb"] == "Blurb for Blog 1."

    args, kwargs = mock_client.invoke.call_args
    assert "Author: Blogger X, Blog: AI Weekly" in args[0]

def test_generate_author_blurbs_news(intelligence, mock_client):
    items = [
        {"title": "News 1", "source": "Tech News"}
    ]
    mock_client.invoke.return_value = "[1] Blurb for News 1."

    result = intelligence.generate_author_blurbs(items, "news")

    assert len(result) == 1
    assert result[0]["author_blurb"] == "Blurb for News 1."

    args, kwargs = mock_client.invoke.call_args
    assert "Source/Organization: Tech News" in args[0]

def test_generate_author_blurbs_unavailable(intelligence, mock_client):
    mock_client.available = False
    items = [{"title": "Item 1"}]

    result = intelligence.generate_author_blurbs(items)

    assert result == items
    assert "author_blurb" not in result[0]
    mock_client.invoke.assert_not_called()

def test_generate_author_blurbs_empty_list(intelligence, mock_client):
    result = intelligence.generate_author_blurbs([])
    assert result == []
    mock_client.invoke.assert_not_called()

def test_extract_missing_authors_blogs(intelligence, mock_client):
    items = [
        {"title": "Missing Author Blog", "source": "TechCrunch", "summary": "An article by John Doe about AI."}
    ]
    # First call is extract_missing_authors (light), second is generate_author_blurbs (light)
    mock_client.invoke.side_effect = ["[1] John Doe", "[1] Blurb for John Doe."]

    result = intelligence.generate_author_blurbs(items, "blogs")

    assert result[0]["author"] == "John Doe"
    assert result[0]["author_blurb"] == "Blurb for John Doe."

    assert mock_client.invoke.call_count == 2
    # Verify tiers used
    calls = mock_client.invoke.call_args_list
    assert calls[0].kwargs["tier"] == "light"
    assert calls[1].kwargs["tier"] == "light"
