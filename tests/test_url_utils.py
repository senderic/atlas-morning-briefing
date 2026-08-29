"""Tests for URL normalization used by deduplication.

Keying dedup on the raw URL string let the same article through twice on
2026-08-29: the local briefing printed one San Diego Magazine roundup under a
trailing-slash variant and one KPBS piece under a ``www.`` variant.
"""

import pytest

from scripts.url_utils import normalize_url


class TestCollapsesCosmeticDifferences:
    @pytest.mark.parametrize("a,b", [
        # The two that actually shipped duplicated.
        ("https://sandiegomagazine.com/things-to-do/x/",
         "https://sandiegomagazine.com/things-to-do/x"),
        ("https://www.kpbs.org/news/y", "https://kpbs.org/news/y"),
        # Same document, different scheme / fragment / tracking params.
        ("http://a.com/p", "https://a.com/p"),
        ("https://a.com/p#section", "https://a.com/p"),
        ("https://a.com/p?utm_source=rss&utm_medium=feed", "https://a.com/p"),
        ("https://a.com/p?id=7&utm_campaign=x", "https://a.com/p?id=7"),
        ("https://A.COM/p", "https://a.com/p"),
        ("  https://a.com/p  ", "https://a.com/p"),
    ])
    def test_variants_normalize_equal(self, a, b):
        assert normalize_url(a) == normalize_url(b)


class TestPreservesMeaningfulDifferences:
    @pytest.mark.parametrize("a,b", [
        ("https://a.com/p?page=1", "https://a.com/p?page=2"),
        ("https://a.com/p", "https://a.com/q"),
        ("https://a.com/p", "https://b.com/p"),
        ("https://a.com/p?id=7", "https://a.com/p?id=8"),
        ("https://a.com:8080/p", "https://a.com/p"),
    ])
    def test_distinct_urls_stay_distinct(self, a, b):
        assert normalize_url(a) != normalize_url(b)


class TestDegradesSafely:
    @pytest.mark.parametrize("value", ["", None])
    def test_empty_is_empty(self, value):
        assert normalize_url(value) == ""

    def test_relative_path_compares_to_itself(self):
        assert normalize_url("/just/a/path") == normalize_url("/just/a/path/")

    def test_garbage_does_not_raise(self):
        assert normalize_url("not a url at all") == normalize_url("not a url at all")
