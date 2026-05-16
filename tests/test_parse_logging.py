"""Tests for parse-failure logging detail in scanner and gemini client.

These tests verify that when something fails to parse, the log message
contains enough context (feed URL, HTTP status, line/column, payload
prefix, model/tier) to diagnose the failure from the journal without
having to reproduce it.
"""

import json
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---- Feed parsing verbose logging ----------------------------------------

def test_blog_scanner_logs_url_status_and_xml_position(caplog, monkeypatch):
    """When feedparser reports a bozo XML error, the log line must include
    the feed URL, HTTP status, and line/col of the parse failure."""
    import scripts.blog_scanner as bs

    # Fake feedparser.parse return: bozo + SAX-style exception.
    class FakeSAXExc(Exception):
        def __init__(self, msg, lineno, colno):
            super().__init__(msg)
            self._line = lineno
            self._col = colno
        def getLineNumber(self):
            return self._line
        def getColumnNumber(self):
            return self._col

    fake_feed = types.SimpleNamespace(
        bozo=1,
        bozo_exception=FakeSAXExc("not well-formed (invalid token)", 17, 42),
        entries=[],
        status=200,
        href="https://www.anthropic.com/rss.xml",
        headers={"content-type": "text/html; charset=utf-8"},
        raw=b"<html>some non-rss content here</html>",
    )
    monkeypatch.setattr(bs.feedparser, "parse", lambda url: fake_feed)

    scanner = bs.BlogScanner(
        feeds=[{"name": "Anthropic", "url": "https://www.anthropic.com/rss.xml"}],
        days_back=7,
        max_items=10,
    )
    with caplog.at_level("WARNING"):
        scanner.scan_feed("Anthropic", "https://www.anthropic.com/rss.xml")

    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Anthropic" in log_text
    assert "https://www.anthropic.com/rss.xml" in log_text
    assert "not well-formed (invalid token)" in log_text
    assert "line=17" in log_text
    assert "col=42" in log_text
    assert "http_status=200" in log_text
    assert "content_type=text/html" in log_text


def test_blog_scanner_logs_resolved_redirect_url(caplog, monkeypatch):
    """If the feed redirected, the resolved URL must appear in the warning."""
    import scripts.blog_scanner as bs

    fake_feed = types.SimpleNamespace(
        bozo=1,
        bozo_exception=Exception("ssl error"),
        entries=[],
        status=301,
        href="https://newhost.example/feed",
        headers={},
        raw=b"",
    )
    monkeypatch.setattr(bs.feedparser, "parse", lambda url: fake_feed)
    scanner = bs.BlogScanner(
        feeds=[{"name": "Old Site", "url": "https://oldhost.example/feed"}],
        days_back=7, max_items=10,
    )
    with caplog.at_level("WARNING"):
        scanner.scan_feed("Old Site", "https://oldhost.example/feed")
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "resolved_url=https://newhost.example/feed" in log_text


def test_blog_scanner_exception_logs_url_and_type(caplog, monkeypatch):
    """A non-bozo top-level exception should still surface the feed URL
    and exception type."""
    import scripts.blog_scanner as bs

    def _raise(_url):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(bs.feedparser, "parse", _raise)
    scanner = bs.BlogScanner(
        feeds=[{"name": "MIT", "url": "https://news.mit.edu/feed.xml"}],
        days_back=7, max_items=10,
    )
    with caplog.at_level("ERROR"):
        scanner.scan_feed("MIT", "https://news.mit.edu/feed.xml")
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "MIT" in log_text
    assert "https://news.mit.edu/feed.xml" in log_text
    assert "RuntimeError" in log_text
    assert "connection refused" in log_text


# ---- ArXiv XML parse verbose logging --------------------------------------

def test_arxiv_scanner_logs_xml_prefix_on_parse_failure(caplog):
    """When ArXiv's XML can't parse, the log must show what it tried to
    parse so we can tell HTML/error-page-instead-of-XML at a glance."""
    from datetime import datetime, timezone
    from scripts.arxiv_scanner import LegacyArxivScanner

    scanner = LegacyArxivScanner(topics=["x"], days_back=7, max_results=10)
    # Malformed XML triggers ET.ParseError. Include something memorable
    # in the payload so we can assert it appears in the log prefix.
    bad = "503 Service Unavailable - check arxiv status, this is not XML <unclosed"
    with caplog.at_level("ERROR"):
        result = scanner._parse_arxiv_response(bad, datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert result == []
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Failed to parse ArXiv XML" in log_text
    assert "503 Service Unavailable" in log_text


# ---- Gemini JSON parse failure verbose logging ----------------------------

def test_gemini_client_logs_stdout_and_stderr_on_json_parse_failure(caplog):
    """When gemini-cli emits something that isn't valid JSON (e.g. the auth
    flow printed help text), the warning must include the model/tier plus
    a prefix of stdout AND stderr so we know what went wrong."""
    from scripts.gemini_client import GeminiCLIClient

    client = GeminiCLIClient({"enabled": True})
    client._available = True
    client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
    bad_stdout = "this is not JSON\nbut you got it anyway"
    bad_stderr = "WARN: 256-color support not detected"
    with patch("subprocess.run", return_value=MagicMock(
        returncode=0, stdout=bad_stdout, stderr=bad_stderr,
    )):
        with caplog.at_level("WARNING"):
            result = client.invoke("p", tier="medium")
    # Even on parse failure, we return the stripped raw stdout — not None.
    assert result == bad_stdout.strip()
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "tier=medium" in log_text
    assert "this is not JSON" in log_text
    assert "256-color support" in log_text
