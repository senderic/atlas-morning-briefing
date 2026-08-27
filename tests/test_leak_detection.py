# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for CoT / grounding-scaffolding leak detection."""

from scripts.leak_detection import is_cot_leak


class TestIsCotLeak:
    def test_empty_string_false(self):
        assert is_cot_leak("") is False

    def test_none_false(self):
        assert is_cot_leak(None) is False

    def test_clean_prose_false(self):
        text = "Today's briefing highlights a surge in agent evaluation papers."
        assert is_cot_leak(text) is False

    def test_single_weak_marker_alone_false(self):
        text = "The quoted claim is verbatim from the source document."
        assert is_cot_leak(text) is False

    def test_single_strong_marker_true(self):
        text = "Strict Grounding Verification: all facts checked."
        assert is_cot_leak(text) is True

    def test_two_weak_markers_true(self):
        text = "Check whether each claim is verbatim and matches entities/facts."
        assert is_cot_leak(text) is True

    def test_one_strong_one_weak_true(self):
        text = "Strict grounding required; ensure the quote is verbatim."
        assert is_cot_leak(text) is True

    def test_case_insensitive(self):
        text = "STRICT GROUNDING VERIFICATION."
        assert is_cot_leak(text) is True