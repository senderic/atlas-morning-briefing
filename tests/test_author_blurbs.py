# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for author blurb generation in intelligence module."""

import pytest
from unittest.mock import MagicMock
from scripts.intelligence import BriefingIntelligence
from scripts.gemini_client import GeminiCLIClient

@pytest.fixture
def mock_gemini():
    client = MagicMock(spec=GeminiCLIClient)
    client.available = True
    return client

@pytest.fixture
def intelligence(mock_gemini):
    config = {"arxiv_topics": ["AI"]}
    return BriefingIntelligence(mock_gemini, config)

def test_generate_author_blurbs_papers(intelligence, mock_gemini):
    items = [
        {"title": "Paper 1", "authors": ["Author A", "Author B"]},
        {"title": "Paper 2", "authors": ["Author C"]}
    ]
    mock_gemini.invoke.return_value = "[1] Blurb for Paper 1.\n[2] Blurb for Paper 2."

    result = intelligence.generate_author_blurbs(items, "papers")

    assert len(result) == 2
    assert result[0]["author_blurb"] == "Blurb for Paper 1."
    assert result[1]["author_blurb"] == "Blurb for Paper 2."

    # Check if correct tier was used
    mock_gemini.invoke.assert_called_once()
    args, kwargs = mock_gemini.invoke.call_args
    assert kwargs["tier"] == "medium"
    assert "reputable sources" in args[0]
    assert "PBS, NPR, NYT" in args[0]

def test_generate_author_blurbs_blogs(intelligence, mock_gemini):
    items = [
        {"title": "Blog 1", "author": "Blogger X", "source": "AI Weekly"}
    ]
    mock_gemini.invoke.return_value = "[1] Blurb for Blog 1."

    result = intelligence.generate_author_blurbs(items, "blogs")

    assert len(result) == 1
    assert result[0]["author_blurb"] == "Blurb for Blog 1."

    args, kwargs = mock_gemini.invoke.call_args
    assert "Author: Blogger X, Blog: AI Weekly" in args[0]

def test_generate_author_blurbs_news(intelligence, mock_gemini):
    items = [
        {"title": "News 1", "source": "Tech News"}
    ]
    mock_gemini.invoke.return_value = "[1] Blurb for News 1."

    result = intelligence.generate_author_blurbs(items, "news")

    assert len(result) == 1
    assert result[0]["author_blurb"] == "Blurb for News 1."

    args, kwargs = mock_gemini.invoke.call_args
    assert "Source/Organization: Tech News" in args[0]

def test_generate_author_blurbs_unavailable(intelligence, mock_gemini):
    mock_gemini.available = False
    items = [{"title": "Item 1"}]

    result = intelligence.generate_author_blurbs(items)

    assert result == items
    assert "author_blurb" not in result[0]
    mock_gemini.invoke.assert_not_called()

def test_generate_author_blurbs_empty_list(intelligence, mock_gemini):
    result = intelligence.generate_author_blurbs([])
    assert result == []
    mock_gemini.invoke.assert_not_called()
