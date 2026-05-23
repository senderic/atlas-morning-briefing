# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for blog_scanner module."""

import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from scripts.blog_scanner import BlogScanner, load_config, main


def make_feed(entries=None, bozo=False, bozo_exception="parse error"):
    """Build a feedparser-like object."""
    feed = MagicMock()
    feed.bozo = bozo
    feed.bozo_exception = bozo_exception
    feed.entries = entries or []
    return feed


def make_entry(**kwargs):
    """Build a feedparser entry-like dict."""
    e = {
        "title": kwargs.get("title", ""),
        "link": kwargs.get("link", ""),
        "summary": kwargs.get("summary", ""),
        "description": kwargs.get("description", ""),
        "author": kwargs.get("author", ""),
    }
    entry = MagicMock()
    for k, v in e.items():
        entry.get = lambda key, default="", _d=e: _d.get(key, default)
    # feedparser exposes published_parsed as a time.struct_time tuple
    if "published_parsed" in kwargs:
        entry.published_parsed = kwargs["published_parsed"]
        entry.published = "now"
    else:
        entry.published_parsed = None
    if "updated_parsed" in kwargs:
        entry.updated_parsed = kwargs["updated_parsed"]
    else:
        entry.updated_parsed = None

    def get_func(key, default=""):
        return e.get(key, default)
    entry.get = get_func

    # `hasattr(entry, "published_parsed")` needs to be truthy in feedparser code.
    # MagicMock returns truthy by default for missing attrs, so we explicitly
    # set them to None above and the production code's truthiness check works.
    return entry


@pytest.fixture
def scanner():
    feeds = [{"name": "TestFeed", "url": "http://example.com/feed"}]
    return BlogScanner(feeds=feeds, days_back=7, max_items=10)


class TestScanFeed:
    @patch("scripts.blog_scanner.feedparser.parse")
    def test_parses_entries(self, mock_parse, scanner):
        now_struct = datetime.now(timezone.utc).timetuple()
        e1 = make_entry(
            title="Article 1",
            link="http://example.com/1",
            summary="Summary 1",
            author="Author A",
            published_parsed=now_struct,
        )
        mock_parse.return_value = make_feed(entries=[e1])
        articles = scanner.scan_feed("TestFeed", "http://example.com/feed")
        assert len(articles) == 1
        assert articles[0]["title"] == "Article 1"
        assert articles[0]["source"] == "TestFeed"
        assert articles[0]["author"] == "Author A"

    @patch("scripts.blog_scanner.feedparser.parse")
    def test_filters_old_entries(self, mock_parse, scanner):
        old_struct = (datetime.now(timezone.utc) - timedelta(days=30)).timetuple()
        e_old = make_entry(
            title="Old", link="http://x.com/old", published_parsed=old_struct
        )
        mock_parse.return_value = make_feed(entries=[e_old])
        articles = scanner.scan_feed("F", "url")
        assert articles == []

    @patch("scripts.blog_scanner.feedparser.parse")
    def test_falls_back_to_updated_date(self, mock_parse, scanner):
        now_struct = datetime.now(timezone.utc).timetuple()
        e = make_entry(
            title="X", link="http://x.com", updated_parsed=now_struct
        )
        mock_parse.return_value = make_feed(entries=[e])
        articles = scanner.scan_feed("F", "url")
        assert len(articles) == 1

    @patch("scripts.blog_scanner.feedparser.parse")
    def test_includes_entry_without_date(self, mock_parse, scanner):
        # No published date → not filtered (kept defensively)
        e = make_entry(title="NoDate", link="http://x.com/n")
        mock_parse.return_value = make_feed(entries=[e])
        articles = scanner.scan_feed("F", "url")
        assert len(articles) == 1
        assert articles[0]["published"] == ""

    @patch("scripts.blog_scanner.feedparser.parse")
    def test_respects_max_items(self, mock_parse, scanner):
        now_struct = datetime.now(timezone.utc).timetuple()
        entries = [
            make_entry(title=f"E{i}", link=f"http://x.com/{i}", published_parsed=now_struct)
            for i in range(20)
        ]
        scanner.max_items = 5
        mock_parse.return_value = make_feed(entries=entries)
        articles = scanner.scan_feed("F", "url")
        assert len(articles) == 5

    @patch("scripts.blog_scanner.feedparser.parse")
    def test_bozo_feed_logs_but_returns(self, mock_parse, scanner):
        now_struct = datetime.now(timezone.utc).timetuple()
        e = make_entry(title="X", link="http://x.com", published_parsed=now_struct)
        mock_parse.return_value = make_feed(entries=[e], bozo=True)
        articles = scanner.scan_feed("F", "url")
        # bozo is a warning; entries still parsed
        assert len(articles) == 1

    @patch("scripts.blog_scanner.feedparser.parse")
    def test_exception_returns_empty(self, mock_parse, scanner):
        mock_parse.side_effect = RuntimeError("boom")
        assert scanner.scan_feed("F", "url") == []


class TestScanAllFeeds:
    def test_skips_invalid_feed_entries(self):
        feeds = [
            {"name": "Good", "url": "http://x.com"},
            {"name": "", "url": "http://y.com"},  # missing name
            {"url": "http://z.com"},  # missing name
            {"name": "NoUrl"},  # missing url
        ]
        scanner = BlogScanner(feeds=feeds, days_back=7, max_items=5)
        with patch.object(scanner, "scan_feed", return_value=[{"x": 1}]) as m:
            scanner.scan_all_feeds()
            # Only 1 valid feed
            assert m.call_count == 1

    def test_collects_across_feeds(self):
        feeds = [
            {"name": "A", "url": "ua"},
            {"name": "B", "url": "ub"},
        ]
        scanner = BlogScanner(feeds=feeds, days_back=7, max_items=5)
        with patch.object(scanner, "scan_feed", side_effect=[[{"a": 1}], [{"b": 2}]]):
            articles = scanner.scan_all_feeds()
            assert len(articles) == 2

    def test_handles_feed_exception(self):
        feeds = [{"name": "X", "url": "u"}]
        scanner = BlogScanner(feeds=feeds, days_back=7, max_items=5)
        with patch.object(scanner, "scan_feed", side_effect=RuntimeError("boom")):
            assert scanner.scan_all_feeds() == []


class TestLoadConfig:
    def test_loads(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text("blog_feeds:\n  - name: A\n    url: ua\n")
        cfg = load_config(str(f))
        assert cfg["blog_feeds"][0]["name"] == "A"

    def test_missing_file(self, tmp_path):
        with pytest.raises(SystemExit):
            load_config(str(tmp_path / "nope.yaml"))


class TestMain:
    def test_no_feeds(self, tmp_path, monkeypatch):
        f = tmp_path / "c.yaml"
        f.write_text("blog_feeds: []\n")
        monkeypatch.setattr("sys.argv", ["b.py", "--config", str(f)])
        assert main() == 2

    @patch("scripts.blog_scanner.BlogScanner")
    def test_no_articles(self, mock_cls, tmp_path, monkeypatch):
        f = tmp_path / "c.yaml"
        f.write_text("blog_feeds:\n  - name: A\n    url: ua\n")
        instance = MagicMock()
        instance.scan_all_feeds.return_value = []
        mock_cls.return_value = instance
        monkeypatch.setattr("sys.argv", ["b.py", "--config", str(f)])
        assert main() == 1

    @patch("scripts.blog_scanner.BlogScanner")
    def test_writes_output(self, mock_cls, tmp_path, monkeypatch):
        f = tmp_path / "c.yaml"
        f.write_text("blog_feeds:\n  - name: A\n    url: ua\n")
        out = tmp_path / "blogs.json"
        instance = MagicMock()
        instance.scan_all_feeds.return_value = [{"title": "T"}]
        mock_cls.return_value = instance
        monkeypatch.setattr(
            "sys.argv", ["b.py", "--config", str(f), "--output", str(out)]
        )
        assert main() == 0
        assert json.loads(out.read_text())[0]["title"] == "T"
