# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for news_aggregator module."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.news_aggregator import NewsAggregator, load_config, main


@pytest.fixture
def aggregator():
    return NewsAggregator(api_key="fake-key", queries=["AI"], max_results=10, request_delay=0)


SAMPLE_RESPONSE = {
    "results": [
        {
            "title": "AI Breakthrough",
            "url": "https://news.example.com/1",
            "description": "Some description",
            "age": "1 hour ago",
            "meta_url": {"hostname": "news.example.com"},
            "thumbnail": {"src": "https://img.example.com/x.jpg"},
        },
        {
            "title": "Another Story",
            "url": "https://other.example.com/2",
            "description": "More details",
            "meta_url": {"hostname": "other.example.com"},
        },
    ]
}


class TestSearchNews:
    @patch("scripts.news_aggregator.requests.get")
    def test_search_returns_articles(self, mock_get, aggregator):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        articles = aggregator.search_news("AI safety")

        assert len(articles) == 2
        assert articles[0]["title"] == "AI Breakthrough"
        assert articles[0]["source"] == "news.example.com"
        assert articles[0]["thumbnail"] == "https://img.example.com/x.jpg"
        assert articles[0]["query"] == "AI safety"

        # Verify auth and params sent correctly
        call = mock_get.call_args
        assert call.kwargs["headers"]["X-Subscription-Token"] == "fake-key"
        assert call.kwargs["params"]["q"] == "AI safety"
        assert call.kwargs["params"]["freshness"] == "pd"

    @patch("scripts.news_aggregator.requests.get")
    def test_search_handles_missing_meta(self, mock_get, aggregator):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"title": "T", "url": "u"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        articles = aggregator.search_news("q")
        assert articles[0]["source"] == ""
        assert articles[0]["thumbnail"] == ""

    @patch("scripts.news_aggregator.requests.get")
    def test_search_request_exception(self, mock_get, aggregator):
        mock_get.side_effect = requests.RequestException("boom")
        articles = aggregator.search_news("q")
        assert articles == []

    @patch("scripts.news_aggregator.requests.get")
    def test_search_http_error(self, mock_get, aggregator):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("429")
        mock_get.return_value = mock_resp
        assert aggregator.search_news("q") == []

    @patch("scripts.news_aggregator.requests.get")
    def test_search_empty_results(self, mock_get, aggregator):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        assert aggregator.search_news("q") == []


class TestAggregateAllQueries:
    def test_deduplicates_by_url(self, aggregator, monkeypatch):
        same_article = {
            "title": "Dup", "url": "https://x.com/1", "description": "",
            "age": "", "source": "x.com", "thumbnail": "", "query": "",
        }
        monkeypatch.setattr(aggregator, "search_news", lambda q: [same_article, same_article])
        aggregator.queries = ["q1", "q2"]
        articles = aggregator.aggregate_all_queries()
        assert len(articles) == 1

    def test_skips_empty_urls(self, aggregator, monkeypatch):
        articles_per_query = [
            {"title": "no url", "url": "", "description": "", "source": "", "query": ""},
        ]
        monkeypatch.setattr(aggregator, "search_news", lambda q: articles_per_query)
        aggregator.queries = ["q1"]
        result = aggregator.aggregate_all_queries()
        assert result == []

    def test_handles_search_exception(self, aggregator, monkeypatch):
        def boom(q):
            raise RuntimeError("fail")
        monkeypatch.setattr(aggregator, "search_news", boom)
        aggregator.queries = ["q1", "q2"]
        # Exception in worker future is swallowed; result is empty
        assert aggregator.aggregate_all_queries() == []


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text("news_queries:\n  - x\n  - y\n")
        cfg = load_config(str(f))
        assert cfg["news_queries"] == ["x", "y"]

    def test_missing_file(self, tmp_path):
        with pytest.raises(SystemExit):
            load_config(str(tmp_path / "nope.yaml"))

    def test_bad_yaml(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text(": invalid :\n  - [")
        with pytest.raises(SystemExit):
            load_config(str(f))


class TestMain:
    def test_main_no_api_key(self, tmp_path, monkeypatch):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("news_queries:\n  - x\n")
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.setattr("sys.argv", ["news_aggregator.py", "--config", str(cfg)])
        assert main() == 2

    def test_main_no_queries(self, tmp_path, monkeypatch):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("news_queries: []\n")
        monkeypatch.setenv("BRAVE_API_KEY", "fake")
        monkeypatch.setattr("sys.argv", ["news_aggregator.py", "--config", str(cfg)])
        assert main() == 2

    @patch("scripts.news_aggregator.NewsAggregator")
    def test_main_no_articles_returns_1(self, mock_cls, tmp_path, monkeypatch):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("news_queries:\n  - x\n")
        monkeypatch.setenv("BRAVE_API_KEY", "fake")
        instance = MagicMock()
        instance.aggregate_all_queries.return_value = []
        mock_cls.return_value = instance
        monkeypatch.setattr("sys.argv", ["news_aggregator.py", "--config", str(cfg)])
        assert main() == 1

    @patch("scripts.news_aggregator.NewsAggregator")
    def test_main_writes_output(self, mock_cls, tmp_path, monkeypatch):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("news_queries:\n  - x\n")
        out = tmp_path / "news.json"
        monkeypatch.setenv("BRAVE_API_KEY", "fake")
        instance = MagicMock()
        instance.aggregate_all_queries.return_value = [{"title": "T", "url": "u"}]
        mock_cls.return_value = instance
        monkeypatch.setattr(
            "sys.argv",
            ["news_aggregator.py", "--config", str(cfg), "--output", str(out)],
        )
        assert main() == 0
        assert json.loads(out.read_text())[0]["title"] == "T"
