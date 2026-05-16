"""Tests for NewsMarketWorker."""

from unittest.mock import MagicMock, patch

import pytest

from scripts.workers.news_market_worker import NewsMarketWorker


@pytest.fixture
def base_config():
    return {
        "news_queries": ["AI defense"],
        "stocks": ["NVDA", "MSFT"],
        "max_news": 5,
        "arxiv_topics": ["agents"],
    }


@pytest.fixture
def stub_llm():
    llm = MagicMock()
    llm.usage_stats = {
        "heavy": {"in_tokens": 0, "out_tokens": 0},
        "medium": {"in_tokens": 0, "out_tokens": 0},
        "light": {"in_tokens": 0, "out_tokens": 0},
    }
    return llm


@pytest.fixture
def fake_news():
    return [
        {"title": f"News {i}", "url": f"http://n.example/{i}", "description": f"snippet {i}"}
        for i in range(1, 6)
    ]


@pytest.fixture
def fake_stocks():
    return [
        {"symbol": "NVDA", "current_price": 1000.0, "percent_change": 2.5},
        {"symbol": "MSFT", "current_price": 400.0, "percent_change": -1.2},
    ]


def _patches(news, stocks):
    news_agg = MagicMock()
    news_agg.aggregate_all_queries.return_value = news
    stock_fetch = MagicMock()
    stock_fetch.fetch_all_stocks.return_value = stocks
    return [
        patch("scripts.workers.news_market_worker.NewsAggregator", return_value=news_agg),
        patch("scripts.workers.news_market_worker.StockFetcher", return_value=stock_fetch),
    ]


def test_no_intelligence_returns_raw_data(base_config, fake_news, fake_stocks, stub_llm):
    intel = MagicMock()
    intel.available = False
    patches = _patches(fake_news, fake_stocks) + [
        patch("scripts.workers.news_market_worker.BriefingIntelligence", return_value=intel)
    ]
    for p in patches:
        p.start()
    try:
        finding = NewsMarketWorker(base_config, llm_client=stub_llm).execute()
    finally:
        for p in patches:
            p.stop()
    assert finding["status"] == "success"
    assert finding["items"]["news"] == fake_news
    assert finding["items"]["stocks"] == fake_stocks


def test_intelligence_path_ranks_news_and_correlates_stocks(
    base_config, fake_news, fake_stocks, stub_llm
):
    intel = MagicMock()
    intel.available = True
    # rank_and_summarize_news returns annotated news
    intel.rank_and_summarize_news.return_value = [
        {**n, "brief_summary": f"summary for {n['title']}"} for n in fake_news[:3]
    ]
    intel.correlate_stocks_and_news.return_value = [
        {**s, "news_correlation": "demand"} for s in fake_stocks
    ]

    patches = _patches(fake_news, fake_stocks) + [
        patch("scripts.workers.news_market_worker.BriefingIntelligence", return_value=intel)
    ]
    for p in patches:
        p.start()
    try:
        finding = NewsMarketWorker(base_config, llm_client=stub_llm).execute()
    finally:
        for p in patches:
            p.stop()

    assert finding["status"] == "success"
    intel.rank_and_summarize_news.assert_called_once()
    intel.correlate_stocks_and_news.assert_called_once()
    # All news items should have brief_summary attached.
    assert all("brief_summary" in n for n in finding["items"]["news"])
    # Stocks should have news_correlation attached.
    assert all(s.get("news_correlation") == "demand" for s in finding["items"]["stocks"])


def test_news_capped_at_max_news(base_config, fake_news, fake_stocks, stub_llm):
    base_config["max_news"] = 2
    intel = MagicMock()
    intel.available = True
    intel.rank_and_summarize_news.return_value = [
        {**n, "brief_summary": "x"} for n in fake_news  # returns 5
    ]
    intel.correlate_stocks_and_news.return_value = fake_stocks

    patches = _patches(fake_news, fake_stocks) + [
        patch("scripts.workers.news_market_worker.BriefingIntelligence", return_value=intel)
    ]
    for p in patches:
        p.start()
    try:
        finding = NewsMarketWorker(base_config, llm_client=stub_llm).execute()
    finally:
        for p in patches:
            p.stop()
    assert len(finding["items"]["news"]) == 2


def test_token_count_reflects_actual_usage_delta(
    base_config, fake_news, fake_stocks, stub_llm
):
    intel = MagicMock()
    intel.available = True

    def news_fn(news, topics):
        stub_llm.usage_stats["medium"]["in_tokens"] += 500
        stub_llm.usage_stats["medium"]["out_tokens"] += 250
        return [{**n, "brief_summary": "x"} for n in news]

    def corr_fn(stocks, news):
        stub_llm.usage_stats["heavy"]["in_tokens"] += 200
        stub_llm.usage_stats["heavy"]["out_tokens"] += 50
        return stocks

    intel.rank_and_summarize_news.side_effect = news_fn
    intel.correlate_stocks_and_news.side_effect = corr_fn

    patches = _patches(fake_news, fake_stocks) + [
        patch("scripts.workers.news_market_worker.BriefingIntelligence", return_value=intel)
    ]
    for p in patches:
        p.start()
    try:
        finding = NewsMarketWorker(base_config, llm_client=stub_llm).execute()
    finally:
        for p in patches:
            p.stop()
    assert finding["metadata"]["token_count"] == 1000  # 500+250+200+50


def test_news_aggregator_failure_returns_error(base_config, stub_llm):
    news_agg = MagicMock()
    news_agg.aggregate_all_queries.side_effect = RuntimeError("brave down")
    with patch("scripts.workers.news_market_worker.NewsAggregator", return_value=news_agg), \
         patch("scripts.workers.news_market_worker.StockFetcher"):
        finding = NewsMarketWorker(base_config, llm_client=stub_llm).execute()
    assert finding["status"] == "error"
    assert "brave down" in finding["error"]
    assert finding["items"] == {"news": [], "stocks": []}


def test_synthesis_describes_market_state(
    base_config, fake_news, fake_stocks, stub_llm
):
    intel = MagicMock()
    intel.available = True
    intel.rank_and_summarize_news.return_value = [
        {**n, "brief_summary": "x"} for n in fake_news
    ]
    intel.correlate_stocks_and_news.return_value = fake_stocks  # 1 up, 1 down

    patches = _patches(fake_news, fake_stocks) + [
        patch("scripts.workers.news_market_worker.BriefingIntelligence", return_value=intel)
    ]
    for p in patches:
        p.start()
    try:
        finding = NewsMarketWorker(base_config, llm_client=stub_llm).execute()
    finally:
        for p in patches:
            p.stop()
    syn = finding["synthesis"]
    assert "1 up, 1 down" in syn or "bullish" in syn or "bearish" in syn or "flat" in syn
