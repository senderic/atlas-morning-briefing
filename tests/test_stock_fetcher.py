# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for stock_fetcher module."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.stock_fetcher import StockFetcher, load_config, main


@pytest.fixture
def fetcher():
    return StockFetcher(api_key="k", symbols=["AAPL"], request_delay=0)


QUOTE_JSON = {"c": 150.5, "d": 1.5, "dp": 1.0, "h": 152.0, "l": 149.0, "o": 150.0, "pc": 149.0, "t": 1700000000}
PROFILE_JSON = {
    "name": "Apple Inc.",
    "ticker": "AAPL",
    "exchange": "NASDAQ",
    "finnhubIndustry": "Tech",
    "marketCapitalization": 3000000,
    "currency": "USD",
}


class TestFetchQuote:
    @patch("scripts.stock_fetcher.requests.get")
    def test_quote_success(self, mock_get, fetcher):
        mock_resp = MagicMock()
        mock_resp.json.return_value = QUOTE_JSON
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        q = fetcher.fetch_quote("AAPL")
        assert q["symbol"] == "AAPL"
        assert q["current_price"] == 150.5
        assert q["percent_change"] == 1.0
        # Auth header passed
        assert mock_get.call_args.kwargs["headers"]["X-Finnhub-Token"] == "k"

    @patch("scripts.stock_fetcher.requests.get")
    def test_quote_network_error(self, mock_get, fetcher):
        mock_get.side_effect = requests.RequestException("boom")
        q = fetcher.fetch_quote("AAPL")
        assert "error" in q
        assert q["symbol"] == "AAPL"

    @patch("scripts.stock_fetcher.requests.get")
    def test_quote_missing_fields(self, mock_get, fetcher):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}  # Empty response
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        q = fetcher.fetch_quote("AAPL")
        # All fields default to 0
        assert q["current_price"] == 0
        assert q["percent_change"] == 0


class TestFetchCompanyProfile:
    @patch("scripts.stock_fetcher.requests.get")
    def test_profile_success(self, mock_get, fetcher):
        mock_resp = MagicMock()
        mock_resp.json.return_value = PROFILE_JSON
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        p = fetcher.fetch_company_profile("AAPL")
        assert p["name"] == "Apple Inc."
        assert p["industry"] == "Tech"
        assert p["market_cap"] == 3000000

    @patch("scripts.stock_fetcher.requests.get")
    def test_profile_network_error(self, mock_get, fetcher):
        mock_get.side_effect = requests.RequestException("boom")
        p = fetcher.fetch_company_profile("AAPL")
        # Falls back to symbol-based profile
        assert p == {"name": "AAPL", "ticker": "AAPL"}

    @patch("scripts.stock_fetcher.requests.get")
    def test_profile_missing_name(self, mock_get, fetcher):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        p = fetcher.fetch_company_profile("AAPL")
        assert p["name"] == "AAPL"


class TestFetchAllStocks:
    @patch("scripts.stock_fetcher.requests.get")
    def test_combines_quote_and_profile(self, mock_get, fetcher):
        # First call is quote, second is profile
        mock_get.side_effect = [
            MagicMock(json=lambda: QUOTE_JSON, raise_for_status=MagicMock()),
            MagicMock(json=lambda: PROFILE_JSON, raise_for_status=MagicMock()),
        ]
        result = fetcher.fetch_all_stocks()
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["current_price"] == 150.5
        assert result[0]["name"] == "Apple Inc."

    @patch("scripts.stock_fetcher.requests.get")
    def test_skips_profile_on_quote_error(self, mock_get, fetcher):
        # Quote fails → profile not fetched
        mock_get.side_effect = requests.RequestException("boom")
        result = fetcher.fetch_all_stocks()
        assert len(result) == 1
        assert "error" in result[0]
        assert "name" not in result[0]  # profile not merged

    @patch("scripts.stock_fetcher.time.sleep")
    @patch("scripts.stock_fetcher.requests.get")
    def test_respects_request_delay(self, mock_get, mock_sleep):
        fetcher = StockFetcher(api_key="k", symbols=["A", "B"], request_delay=0.5)
        mock_get.side_effect = [
            MagicMock(json=lambda: QUOTE_JSON, raise_for_status=MagicMock()),
            MagicMock(json=lambda: PROFILE_JSON, raise_for_status=MagicMock()),
            MagicMock(json=lambda: QUOTE_JSON, raise_for_status=MagicMock()),
            MagicMock(json=lambda: PROFILE_JSON, raise_for_status=MagicMock()),
        ]
        fetcher.fetch_all_stocks()
        # sleep called between symbols and between quote/profile pairs
        assert mock_sleep.called


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text("stocks:\n  - AAPL\n")
        cfg = load_config(str(f))
        assert cfg["stocks"] == ["AAPL"]

    def test_missing_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_config(str(tmp_path / "nope.yaml"))


class TestMain:
    def test_main_no_api_key(self, tmp_path, monkeypatch):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("stocks:\n  - AAPL\n")
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.setattr("sys.argv", ["s.py", "--config", str(cfg)])
        assert main() == 2

    def test_main_no_symbols(self, tmp_path, monkeypatch):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("stocks: []\n")
        monkeypatch.setenv("FINNHUB_API_KEY", "k")
        monkeypatch.setattr("sys.argv", ["s.py", "--config", str(cfg)])
        assert main() == 2

    @patch("scripts.stock_fetcher.StockFetcher")
    def test_main_all_errors_returns_2(self, mock_cls, tmp_path, monkeypatch):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("stocks:\n  - X\n")
        monkeypatch.setenv("FINNHUB_API_KEY", "k")
        instance = MagicMock()
        instance.fetch_all_stocks.return_value = [{"symbol": "X", "error": "fail"}]
        mock_cls.return_value = instance
        monkeypatch.setattr("sys.argv", ["s.py", "--config", str(cfg)])
        assert main() == 2

    @patch("scripts.stock_fetcher.StockFetcher")
    def test_main_partial_failure_returns_1(self, mock_cls, tmp_path, monkeypatch):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("stocks:\n  - X\n  - Y\n")
        out = tmp_path / "stocks.json"
        monkeypatch.setenv("FINNHUB_API_KEY", "k")
        instance = MagicMock()
        instance.fetch_all_stocks.return_value = [
            {"symbol": "X", "current_price": 100},
            {"symbol": "Y", "error": "boom"},
        ]
        mock_cls.return_value = instance
        monkeypatch.setattr(
            "sys.argv", ["s.py", "--config", str(cfg), "--output", str(out)]
        )
        assert main() == 1
        data = json.loads(out.read_text())
        assert len(data) == 2

    @patch("scripts.stock_fetcher.StockFetcher")
    def test_main_writes_output(self, mock_cls, tmp_path, monkeypatch):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("stocks:\n  - X\n")
        out = tmp_path / "stocks.json"
        monkeypatch.setenv("FINNHUB_API_KEY", "k")
        instance = MagicMock()
        instance.fetch_all_stocks.return_value = [{"symbol": "X", "current_price": 100}]
        mock_cls.return_value = instance
        monkeypatch.setattr(
            "sys.argv", ["s.py", "--config", str(cfg), "--output", str(out)]
        )
        assert main() == 0
