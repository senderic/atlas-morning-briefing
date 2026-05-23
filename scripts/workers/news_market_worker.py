#!/usr/bin/env python3
"""
News and market worker for v0.2 multi-agent architecture.

PURPOSE: Fetch news articles and stock data, enrich with LLM ranking and
correlation analysis, and return structured findings.

Self-contained worker that does NOT delegate back to coordinator.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.news_aggregator import NewsAggregator
from scripts.stock_fetcher import StockFetcher
from scripts.intelligence import BriefingIntelligence
from scripts.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class NewsMarketWorker(BaseWorker):
    """Fetches and enriches news + stock data independently."""

    def __init__(self, config: Dict[str, Any], llm_client: Any = None):
        """
        Initialize NewsMarketWorker.

        Args:
            config: Full configuration dictionary
            llm_client: Optional shared LLM client; see BaseWorker.
        """
        super().__init__(config, "news_market_worker", llm_client=llm_client)
        self.news_queries = config.get("news_queries", [])
        self.stocks = config.get("stocks", [])
        self.max_news = config.get("max_news", 15)

    def execute(self) -> Dict[str, Any]:
        """
        Execute news+market workflow: fetch + enrich + correlate.

        Returns:
            Finding dict with enriched news and stocks
        """
        self._start_timing()
        token_count = 0

        try:
            logger.info(f"[{self.worker_name}] Starting news and market data fetch")

            # Get API keys from environment.
            brave_api_key = os.environ.get("BRAVE_API_KEY", "")
            finnhub_api_key = os.environ.get("FINNHUB_API_KEY", "")

            # Step 1: Fetch news articles. Skip the call entirely if no key —
            # NewsAggregator/Brave does not gate on an empty key and would
            # produce one 401 per query.
            if brave_api_key and self.news_queries:
                news_aggregator = NewsAggregator(brave_api_key, self.news_queries, self.max_news)
                news = news_aggregator.aggregate_all_queries()
            else:
                if not brave_api_key:
                    logger.warning(f"[{self.worker_name}] BRAVE_API_KEY not set; skipping news fetch")
                news = []
            news_found = len(news)
            logger.info(f"[{self.worker_name}] Fetched {news_found} news articles")

            # Step 2: Fetch stock data. Same guard — skip rather than 401.
            if finnhub_api_key and self.stocks:
                stock_fetcher = StockFetcher(finnhub_api_key, self.stocks)
                stocks = stock_fetcher.fetch_all_stocks()
            else:
                if not finnhub_api_key:
                    logger.warning(f"[{self.worker_name}] FINNHUB_API_KEY not set; skipping stock fetch")
                stocks = []
            stocks_found = len(stocks)
            logger.info(f"[{self.worker_name}] Fetched {stocks_found} stock prices")

            # Step 3: Initialize intelligence layer for enrichment.
            # Reuse the coordinator's shared client when available so call
            # budgets and Gemini key-rotation state are honored once per run.
            llm = self._get_llm_client()
            intelligence = BriefingIntelligence(llm, self.config)

            if not intelligence.available:
                logger.warning(f"[{self.worker_name}] Intelligence layer unavailable, returning raw data")
                return self._create_finding(
                    status="success",
                    items={"news": news[:self.max_news], "stocks": stocks},
                    synthesis=f"Found {news_found} news articles and {stocks_found} stocks. LLM enrichment unavailable.",
                    items_found=news_found + stocks_found
                )

            # Step 4: Rank and summarize news with LLM
            logger.info(f"[{self.worker_name}] Enriching news with LLM")
            topics = self.config.get("arxiv_topics", [])
            tokens_before = self._count_client_tokens(llm)
            news = intelligence.rank_and_summarize_news(news, topics)

            # Step 5: Correlate stocks and news
            logger.info(f"[{self.worker_name}] Correlating stocks with news")
            stocks = intelligence.correlate_stocks_and_news(stocks, news)
            token_count = max(0, self._count_client_tokens(llm) - tokens_before)

            # Step 6: Trim to top news. rank_and_summarize_news already returns the
            # LLM's top picks in ranked order and does not assign a numeric score,
            # so we trust that ordering and just cap the count.
            news = news[:self.max_news]

            # Step 7: Generate synthesis
            synthesis = self._generate_synthesis(news, stocks)

            logger.info(f"[{self.worker_name}] Completed. {len(news)} news + {len(stocks)} stocks enriched.")

            return self._create_finding(
                status="success",
                items={"news": news, "stocks": stocks},
                synthesis=synthesis,
                token_count=token_count,
                items_found=news_found + stocks_found
            )

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error: {e}")
            return self._create_finding(
                status="error",
                items={"news": [], "stocks": []},
                error=str(e)
            )

    def _generate_synthesis(self, news: list, stocks: list) -> str:
        """
        Generate synthesis summary from news and stocks.

        Args:
            news: Enriched news articles
            stocks: Enriched stock data

        Returns:
            Summary string
        """
        news_summary = f"{len(news)} high-relevance news articles"
        stocks_summary = f"{len(stocks)} stocks tracked"

        # Calculate market trend
        gainers = [s for s in stocks if s.get("percent_change", 0) > 0]
        losers = [s for s in stocks if s.get("percent_change", 0) < 0]

        if gainers and losers:
            trend = f"{len(gainers)} up, {len(losers)} down"
        elif gainers:
            trend = "bullish (all up)"
        elif losers:
            trend = "bearish (all down)"
        else:
            trend = "flat"

        synthesis = f"{news_summary}. {stocks_summary} ({trend}). "

        if news:
            synthesis += f"Top story: '{news[0].get('title', 'Unknown')}'"

        return synthesis
