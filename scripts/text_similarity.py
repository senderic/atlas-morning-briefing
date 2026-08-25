#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Headline similarity, shared by the pipeline and the quality check.

The pipeline uses this to collapse one story syndicated by many outlets; the
quality check uses it to assert that the collapse actually happened. Those two
have to measure the same thing with the same threshold, or the check quietly
stops testing what it claims to test -- hence one definition, imported by both.

Threshold calibrated on live Brave results: cross-outlet retellings of a single
story scored 0.23-0.47, while genuinely distinct local stories scored 0.00-0.17.
"""

import re
from typing import Iterable, Set

# Words too common to carry identity in a headline.
DEDUP_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "with", "is", "are", "was", "were", "by", "from", "as", "it", "its",
    "this", "that", "new", "says", "after", "over",
}

# Default Jaccard overlap at which two headlines are treated as one story.
DEFAULT_SIMILARITY_THRESHOLD = 0.3


def headline_terms(title: str) -> Set[str]:
    """Content words of a headline, for cross-outlet duplicate detection."""
    return {
        w for w in re.findall(r"[a-z0-9']+", (title or "").lower())
        if len(w) > 2 and w not in DEDUP_STOPWORDS
    }


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Overlap of two term sets, 0.0 when either side is empty."""
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def headlines_match(
    first: str, second: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> bool:
    """True when two headlines look like the same story told twice."""
    return jaccard(headline_terms(first), headline_terms(second)) >= threshold
