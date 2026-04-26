# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for intelligence module."""

import pytest
from scripts.intelligence import BriefingIntelligence, _parse_numbered_list


class TestExtractScore:
    def test_standard_format(self):
        score, text = BriefingIntelligence.extract_score("SCORE:4/5 Great paper on agents.")
        assert score == 4
        assert text == "Great paper on agents."

    def test_lowercase_variant(self):
        score, text = BriefingIntelligence.extract_score("Score: 3/5 Decent work.")
        assert score == 3
        assert text == "Decent work."

    def test_no_score(self):
        score, text = BriefingIntelligence.extract_score("Just a plain summary.")
        assert score is None
        assert text == "Just a plain summary."

    def test_empty_string(self):
        score, text = BriefingIntelligence.extract_score("")
        assert score is None
        assert text == ""


class TestParseRankedResponse:
    def test_basic_parsing(self):
        text = "[1] First item summary.\n[2] Second item summary."
        result = BriefingIntelligence._parse_ranked_response(text)
        assert len(result) == 2
        assert result[0] == (0, "First item summary.")
        assert result[1] == (1, "Second item summary.")

    def test_bold_markers(self):
        text = "**[1]** Bold first item.\n**[2]** Bold second."
        result = BriefingIntelligence._parse_ranked_response(text)
        assert len(result) == 2
        assert result[0][0] == 0
        assert "Bold first item" in result[0][1]

    def test_multiline_items(self):
        text = "[1] First line of item one.\nContinuation of item one.\n[2] Item two."
        result = BriefingIntelligence._parse_ranked_response(text)
        assert len(result) == 2
        assert "First line" in result[0][1]
        assert "Continuation" in result[0][1]

    def test_empty_input(self):
        assert BriefingIntelligence._parse_ranked_response("") == []

    def test_skips_empty_items(self):
        text = "[1] Real content.\n[2] \n[3] Also real."
        result = BriefingIntelligence._parse_ranked_response(text)
        # [2] has no content so it's skipped
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[1][0] == 2

    def test_numbered_sub_items_stripped(self):
        text = "[1] Summary here.\n1. Sub-point one.\n2. Sub-point two."
        result = BriefingIntelligence._parse_ranked_response(text)
        assert len(result) == 1
        assert "Sub-point one" in result[0][1]


class TestParseNumberedList:
    def test_bracket_format(self):
        text = "[1] First item.\n[2] Second item.\n[3] Third item."
        result = _parse_numbered_list(text, 3)
        assert len(result) == 3
        assert result[0] == "First item."

    def test_dot_format(self):
        text = "1. First.\n2. Second."
        result = _parse_numbered_list(text, 2)
        assert len(result) == 2
        assert result[0] == "First."

    def test_limits_to_expected(self):
        text = "[1] A\n[2] B\n[3] C\n[4] D"
        result = _parse_numbered_list(text, 2)
        assert len(result) == 2

    def test_multiline_item(self):
        text = "[1] Start of item.\nMore of the item.\n[2] Next."
        result = _parse_numbered_list(text, 2)
        assert len(result) == 2
        assert "Start of item. More of the item." == result[0]

    def test_ignores_preamble(self):
        text = "Sure, here are the blurbs:\n1. First blurb.\n2. Second blurb."
        result = _parse_numbered_list(text, 2)
        assert len(result) == 2
        assert result[0] == "First blurb."
        assert result[1] == "Second blurb."

    def test_handles_no_numbers_single_expected(self):
        text = "This is a single block of text."
        result = _parse_numbered_list(text, 1)
        assert len(result) == 1
        assert result[0] == "This is a single block of text."

    def test_handles_no_numbers_multiple_expected(self):
        text = "This is a single block of text, but we expected more."
        result = _parse_numbered_list(text, 2)
        assert len(result) == 0  # Should skip as we can't reliably split it
