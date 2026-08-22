#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Detect leaked chain-of-thought / grounding-verification scaffolding in LLM output.

Reasoning models occasionally emit their internal grounding-verification trace
as visible output (e.g. "Strict Grounding Verification**:" or "Check verbatim
entities/facts:" bullet lists) instead of producing the requested editorial.
This module provides the marker vocabulary and a single predicate used by the
intelligence layer and the briefing renderer to recognize such leaks and fall
back to the non-reasoning model or an unavailable placeholder.
"""

STRONG_MARKERS = ("strict grounding", "check verbatim")
WEAK_MARKERS = ("is verbatim", "entities/facts")

_ALL_MARKERS = STRONG_MARKERS + WEAK_MARKERS


def is_cot_leak(text: str) -> bool:
    """
    Return True if ``text`` shows signs of leaked CoT / grounding scaffolding.

    A single strong marker is conclusive; otherwise the leak is only declared
    when at least two of the four total markers (strong + weak) appear together.
    Matching is case-insensitive.

    Args:
        text: Candidate LLM output text (may be None or empty).

    Returns:
        True when the text looks like leaked reasoning/verification scaffolding.
    """
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in STRONG_MARKERS):
        return True
    matched = sum(1 for marker in _ALL_MARKERS if marker in lowered)
    return matched >= 2