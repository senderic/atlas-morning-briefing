# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for arxiv_scanner module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.arxiv_scanner import (
    ArxivScanner,
    _load_deepxiv_token,
    create_scanner,
    load_config,
    main,
)
import scripts.arxiv_scanner as arxiv_mod


def _build_arxiv_xml(recent_iso: str) -> str:
    """Build a 3-entry feed with one recent paper, one old, one undated."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Evaluating Multi-Agent Systems</title>
    <summary>We propose a benchmark for agent evaluation.</summary>
    <published>{recent_iso}</published>
    <updated>{recent_iso}</updated>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.AI"/>
    <category term="cs.MA"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Old Paper on Something</title>
    <summary>This is an old paper.</summary>
    <published>2020-01-01T00:00:00Z</published>
    <updated>2020-01-01T00:00:00Z</updated>
    <author><name>Charlie</name></author>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00003v1</id>
    <title>No Date Paper</title>
    <summary>Missing published date.</summary>
    <author><name>Diana</name></author>
  </entry>
</feed>"""


@pytest.fixture
def scanner():
    return ArxivScanner(topics=["Agent Evaluation"], days_back=7, max_results=10)


@pytest.fixture
def recent_xml():
    """XML where the first entry is published 'now' so it's always recent."""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return _build_arxiv_xml(now_iso)


class TestParseArxivResponse:
    def test_parses_valid_entry(self, scanner, recent_xml):
        start_date = datetime.now(timezone.utc) - timedelta(days=30)
        papers = scanner._parse_arxiv_response(recent_xml, start_date)
        # Should include the recent paper at minimum
        assert len(papers) >= 1
        paper = next(p for p in papers if p["title"] == "Evaluating Multi-Agent Systems")
        assert paper["authors"] == ["Alice Smith", "Bob Jones"]
        assert "cs.AI" in paper["categories"]
        assert "cs.MA" in paper["categories"]
        assert paper["pdf_link"] == "http://arxiv.org/pdf/2401.00001v1"
        assert paper["arxiv_url"] == "http://arxiv.org/abs/2401.00001v1"

    def test_filters_by_date(self, scanner, recent_xml):
        # Start date excludes the 2020 paper
        start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        papers = scanner._parse_arxiv_response(recent_xml, start_date)
        titles = [p["title"] for p in papers]
        assert "Evaluating Multi-Agent Systems" in titles
        assert "Old Paper on Something" not in titles

    def test_skips_entries_without_date(self, scanner, recent_xml):
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        papers = scanner._parse_arxiv_response(recent_xml, start_date)
        titles = [p["title"] for p in papers]
        assert "No Date Paper" not in titles

    def test_handles_malformed_xml(self, scanner):
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        papers = scanner._parse_arxiv_response("not xml", start_date)
        assert papers == []

    def test_fallback_pdf_link_from_id(self, scanner):
        """When no <link title='pdf'> is present, scanner derives URL from id."""
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.99999v1</id>
    <title>No Link Paper</title>
    <summary>Missing pdf link.</summary>
    <published>{now_iso}</published>
    <author><name>Eve</name></author>
  </entry>
</feed>"""
        papers = scanner._parse_arxiv_response(
            xml, datetime.now(timezone.utc) - timedelta(days=30)
        )
        assert len(papers) == 1
        assert papers[0]["pdf_link"] == "http://arxiv.org/pdf/2401.99999v1.pdf"

    def test_empty_feed(self, scanner):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Empty</title></feed>"""
        papers = scanner._parse_arxiv_response(
            xml, datetime.now(timezone.utc) - timedelta(days=30)
        )
        assert papers == []


class TestSearchTopic:
    @patch("scripts.arxiv_scanner.requests.get")
    def test_search_topic_success(self, mock_get, scanner, recent_xml):
        mock_resp = MagicMock(text=recent_xml)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        papers = scanner.search_topic("Agent Evaluation")

        assert len(papers) >= 1
        # Verify query was built correctly
        kwargs = mock_get.call_args.kwargs
        assert kwargs["params"]["search_query"] == "all:Agent Evaluation"
        assert kwargs["params"]["sortBy"] == "submittedDate"

    @patch("scripts.arxiv_scanner.requests.get")
    def test_search_topic_network_error(self, mock_get, scanner):
        mock_get.side_effect = requests.RequestException("network down")
        papers = scanner.search_topic("Anything")
        assert papers == []

    @patch("scripts.arxiv_scanner.requests.get")
    def test_search_topic_http_error(self, mock_get, scanner):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_get.return_value = mock_resp
        papers = scanner.search_topic("Anything")
        assert papers == []


class TestScanAllTopics:
    def test_deduplicates_papers(self, scanner, monkeypatch):
        paper = {
            "id": "http://arxiv.org/abs/2401.00001v1",
            "title": "Test",
            "summary": "",
            "authors": [],
            "published": "",
            "updated": "",
            "categories": [],
            "pdf_link": "",
            "arxiv_url": "",
        }
        monkeypatch.setattr(scanner, "search_topic", lambda topic: [paper, paper])
        scanner.topics = ["topic1", "topic2"]
        papers = scanner.scan_all_topics()
        assert len(papers) == 1

    def test_handles_search_exception(self, scanner, monkeypatch):
        def boom(topic):
            raise RuntimeError("simulated failure")
        monkeypatch.setattr(scanner, "search_topic", boom)
        scanner.topics = ["t1", "t2"]
        # Exceptions in futures should be swallowed; result is empty
        papers = scanner.scan_all_topics()
        assert papers == []

    def test_empty_topics(self):
        scanner = ArxivScanner(topics=[], days_back=7, max_results=10)
        papers = scanner.scan_all_topics()
        assert papers == []


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("arxiv_topics:\n  - Topic A\n  - Topic B\n")
        cfg = load_config(str(config_path))
        assert cfg["arxiv_topics"] == ["Topic A", "Topic B"]

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            load_config(str(tmp_path / "nope.yaml"))
        assert exc.value.code == 2

    def test_malformed_yaml_exits(self, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(": this is not valid : yaml :\n  - [")
        with pytest.raises(SystemExit) as exc:
            load_config(str(config_path))
        assert exc.value.code == 2


class TestLoadDeepXivToken:
    def test_returns_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("DEEPXIV_TOKEN", "from-env-123")
        assert _load_deepxiv_token() == "from-env-123"

    def test_reads_from_home_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DEEPXIV_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".env").write_text(
            "OTHER_VAR=x\nDEEPXIV_TOKEN=from-file-abc\nNEXT=y\n"
        )
        assert _load_deepxiv_token() == "from-file-abc"

    def test_strips_quotes_around_value(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DEEPXIV_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".env").write_text('DEEPXIV_TOKEN="quoted-token"\n')
        assert _load_deepxiv_token() == "quoted-token"

    def test_returns_none_when_missing_everywhere(self, monkeypatch, tmp_path):
        """The 'None' result is what tells Reader() to auto-register."""
        monkeypatch.delenv("DEEPXIV_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _load_deepxiv_token() is None

    def test_env_var_wins_over_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEEPXIV_TOKEN", "env-wins")
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".env").write_text("DEEPXIV_TOKEN=file-value\n")
        assert _load_deepxiv_token() == "env-wins"


class TestCreateScanner:
    def test_picks_deepxiv_when_sdk_available(self, monkeypatch):
        """DeepXivScanner is preferred whenever HAS_DEEPXIV is True —
        no token needed (SDK auto-registers on first use)."""
        if not arxiv_mod.HAS_DEEPXIV:
            pytest.skip("deepxiv-sdk not installed")
        monkeypatch.delenv("DEEPXIV_TOKEN", raising=False)
        scanner = create_scanner(topics=["AI"], days_back=1, max_results=2)
        assert scanner.__class__.__name__ == "DeepXivScanner"

    def test_falls_back_to_legacy_when_sdk_missing(self, monkeypatch):
        monkeypatch.setattr(arxiv_mod, "HAS_DEEPXIV", False)
        scanner = create_scanner(topics=["AI"], days_back=1, max_results=2)
        assert scanner.__class__.__name__ == "ArxivScanner"

    def test_deepxiv_scanner_init_when_no_token(self, monkeypatch):
        """Reader() is called with no token args, allowing auto-register."""
        if not arxiv_mod.HAS_DEEPXIV:
            pytest.skip("deepxiv-sdk not installed")
        monkeypatch.delenv("DEEPXIV_TOKEN", raising=False)
        # Patch Reader to a mock so we can inspect what was passed
        with patch.object(arxiv_mod, "DeepXivReader") as mock_reader:
            create_scanner(topics=["X"], days_back=1, max_results=1)
            mock_reader.assert_called_once_with()  # no kwargs == auto-register

    def test_deepxiv_scanner_init_with_token(self, monkeypatch):
        if not arxiv_mod.HAS_DEEPXIV:
            pytest.skip("deepxiv-sdk not installed")
        monkeypatch.setenv("DEEPXIV_TOKEN", "explicit-token")
        with patch.object(arxiv_mod, "DeepXivReader") as mock_reader:
            create_scanner(topics=["X"], days_back=1, max_results=1)
            mock_reader.assert_called_once_with(token="explicit-token")


class TestMain:
    @patch("scripts.arxiv_scanner.ArxivScanner")
    def test_main_writes_output(self, mock_scanner_cls, tmp_path, monkeypatch):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("arxiv_topics:\n  - X\n")
        out_path = tmp_path / "papers.json"

        instance = MagicMock()
        instance.scan_all_topics.return_value = [{"id": "1", "title": "T"}]
        mock_scanner_cls.return_value = instance

        monkeypatch.setattr(
            "sys.argv",
            ["arxiv_scanner.py", "--config", str(cfg_path), "--output", str(out_path)],
        )
        code = main()
        assert code == 0
        assert out_path.exists()
        import json
        data = json.loads(out_path.read_text())
        assert data[0]["title"] == "T"

    def test_main_no_topics_returns_2(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("arxiv_topics: []\n")
        monkeypatch.setattr("sys.argv", ["arxiv_scanner.py", "--config", str(cfg_path)])
        assert main() == 2

    @patch("scripts.arxiv_scanner.ArxivScanner")
    def test_main_no_papers_returns_1(self, mock_cls, tmp_path, monkeypatch):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("arxiv_topics:\n  - X\n")
        instance = MagicMock()
        instance.scan_all_topics.return_value = []
        mock_cls.return_value = instance
        monkeypatch.setattr("sys.argv", ["arxiv_scanner.py", "--config", str(cfg_path)])
        assert main() == 1
