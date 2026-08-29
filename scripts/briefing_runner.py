#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Morning briefing runner.

Main orchestrator that runs all scanners, applies the intelligence layer,
and generates the briefing. Uses an LLM client (GeminiCLIClient or
OpencodeClient) for LLM-powered synthesis and summarization (with a
deterministic fallback when unavailable).
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

# Load environment variables from .env if it exists, override existing shell env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Ensure scripts directory is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.arxiv_scanner import ArxivScanner, create_scanner
from scripts.snapshot_manager import SnapshotManager
from scripts.blog_scanner import BlogScanner
from scripts.stock_fetcher import StockFetcher
from scripts.text_similarity import DEDUP_STOPWORDS, headline_terms
from scripts.alerts_scanner import create_scanner as create_alerts_scanner
from scripts.geo_filter import apply_config_filter
from scripts.interest_graph import query_freshness
from scripts.news_aggregator import NewsAggregator
from scripts.paper_scorer import PaperScorer
from scripts.pdf_generator import PDFGenerator
from scripts.epub_generator import EPUBGenerator
from scripts.email_distributor import EmailDistributor
from scripts.event_dates import has_only_past_dates
from scripts.url_utils import normalize_url
from scripts.config_validator import validate_config, check_environment
from scripts.gemini_client import GeminiCLIClient
from scripts.intelligence import BriefingIntelligence
from scripts.leak_detection import is_cot_leak
from scripts.llm_client import BaseLLMClient


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYNTHESIS_UNAVAILABLE_TEXT = (
    "*Synthesis unavailable for today's briefing. Please see the individual "
    "sections below for key updates in tech, defense, and research.*\n\n"
)

DEFAULT_FILE_NAMING = "Atlas-Briefing-{yyyy}.{mm}.{dd}"


def format_briefing_filename(file_naming: str, now: datetime) -> str:
    """
    Render a briefing filename from its config pattern.

    Module-level so anything that needs to *find* today's briefing -- the
    quality check, for one -- derives the name from the same rule that wrote
    it, instead of keeping a second copy that drifts.
    """
    known_vars = {
        "yyyy": now.strftime("%Y"),
        "mm": now.strftime("%m"),
        "dd": now.strftime("%d"),
        "type": "Daily",
    }
    return (file_naming or DEFAULT_FILE_NAMING).format_map(known_vars)


class BriefingRunner:
    """Main orchestrator for morning briefing generation."""

    def __init__(
        self,
        config: Dict[str, Any],
        dry_run: bool = False,
        use_snapshots: Optional[str] = None,
    ):
        """
        Initialize BriefingRunner.

        Args:
            config: Configuration dictionary.
            dry_run: If True, don't send email.
            use_snapshots: If set to a date string (e.g. "2026-07-27"),
                           load raw data from snapshots/{date}/ instead of
                           making live API calls.
        """
        self.config = config
        self.dry_run = dry_run
        self.use_snapshots = use_snapshots
        self.user_name = os.getenv("USER_NAME", "")
        self.errors = []
        self.state_file_path = config.get("state_file_path", ".atlas-state.json")
        self.section_order = config.get("section_order", ["stocks", "news", "top_papers", "blogs"])
        features = config.get("features", {})
        self.feature_solo_founder_angle = features.get("solo_founder_angle", True)
        self.feature_agent_cost_optimization = features.get("agent_cost_optimization", True)
        self.feature_weekly_deep_dive = features.get("weekly_deep_dive", True)
        self._headings = config.get("section_headings", {})
        self._happenings_cache: List[Dict[str, Any]] = []
        self._happenings_cache_date: Optional[str] = None
        self._briefing_title = self._format_filename(datetime.now())
        self.status = {
            "timestamp": datetime.now().isoformat(),
            "papers_found": 0,
            "blogs_found": 0,
            "stocks_fetched": 0,
            "news_found": 0,
            "happenings_found": 0,
            "alerts_found": 0,
            "synthesis_degraded": False,
            "geo_filtered_out": 0,
            "intelligence_enabled": False,
            "errors": [],
            "pdf_generated": False,
            "epub_generated": False,
            "email_sent": False,
            "elapsed_seconds": 0,
        }

        # Initialize LLM client and intelligence layer.
        preflight_data = self._load_preflight_models()
        gemini_config = config.get("gemini", config.get("bedrock", {}))

        # Chain construction lives in scripts.llm_chain so the briefing runner
        # and the quality checker cannot drift on backend ordering. Order is
        # cost: free backends first, any paid one last.
        from scripts.composite_client import CompositeClient
        from scripts.llm_chain import build_llm_chain, chain_timeout

        chain = build_llm_chain(config, preflight_models=preflight_data)

        if not chain:
            # No LLM backend enabled at all — deterministic mode.
            self.llm_client = GeminiCLIClient(gemini_config)
        elif len(chain) == 1:
            self.llm_client = chain[0]
        else:
            self.llm_client = CompositeClient(chain, timeout=chain_timeout(config))
        self.intelligence = BriefingIntelligence(self.llm_client, config)
        self.status["intelligence_enabled"] = self.intelligence.available

        snapshot_cfg = config.get("snapshot", {})
        self.snapshot_manager = SnapshotManager(
            snapshot_dir=snapshot_cfg.get("dir", "snapshots"),
            enabled=snapshot_cfg.get("enabled", True),
        )
        logger.debug(
            "Initialized BriefingRunner: state_file=%s section_order=%s "
            "features(solo=%s agent=%s weekly=%s) dry_run=%s use_snapshots=%s",
            self.state_file_path, self.section_order,
            self.feature_solo_founder_angle, self.feature_agent_cost_optimization,
            self.feature_weekly_deep_dive, self.dry_run, self.use_snapshots,
        )

    @staticmethod
    def _load_snapshot(path: str) -> List[Dict[str, Any]]:
        """Load a JSON snapshot file, returning an empty list on any failure."""
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Snapshot load failed: {path} — {e}")
            return []

    def run_arxiv_scan(self, topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run arxiv paper scan."""
        try:
            logger.info("=== Scanning ArXiv Papers ===")
            if topics is None:
                topics = self.config.get("arxiv_topics", [])
            days_back = self.config.get("arxiv_days_back", 7)
            max_papers = self.config.get("max_papers", 20)

            if not topics:
                logger.warning("No arxiv_topics configured, skipping")
                return []

            # create_scanner picks DeepXivScanner when the SDK is installed
            # (auto-registers a free token on first use) and falls back to
            # our sequential defusedxml ArxivScanner otherwise.
            scanner = create_scanner(
                topics=topics,
                days_back=days_back,
                max_results=max_papers,
                request_delay=self.config.get("arxiv_request_delay"),
                exact_phrase=self.config.get("arxiv_exact_phrase", False),
            )
            papers = scanner.scan_all_topics()
            self.status["papers_found"] = len(papers)
            logger.info(f"Found {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"ArXiv scan failed: {e}")
            self.errors.append(f"ArXiv scan: {e}")
            return []

    def run_blog_scan(self) -> List[Dict[str, Any]]:
        """Run blog feed scan."""
        try:
            logger.info("=== Scanning Blog Feeds ===")
            feeds = self.config.get("blog_feeds", [])
            days_back = self.config.get("arxiv_days_back", 7)
            max_blogs = self.config.get("max_blogs", 10)

            if not feeds:
                logger.warning("No blog_feeds configured, skipping")
                return []

            scanner = BlogScanner(
                feeds=feeds,
                days_back=days_back,
                max_items=max_blogs,
            )
            articles = scanner.scan_all_feeds()
            self.status["blogs_found"] = len(articles)
            logger.info(f"Found {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"Blog scan failed: {e}")
            self.errors.append(f"Blog scan: {e}")
            return []

    def run_stock_fetch(self) -> List[Dict[str, Any]]:
        """Run stock data fetch."""
        try:
            logger.info("=== Fetching Stock Data ===")
            api_key = os.environ.get("FINNHUB_API_KEY")
            symbols = self.config.get("stocks", [])

            if not api_key:
                logger.warning("FINNHUB_API_KEY not set, skipping stocks")
                return []

            if not symbols:
                logger.warning("No stocks configured, skipping")
                return []

            fetcher = StockFetcher(api_key=api_key, symbols=symbols)
            stocks = fetcher.fetch_all_stocks()
            self.status["stocks_fetched"] = len(stocks)
            logger.info(f"Fetched data for {len(stocks)} stocks")
            return stocks

        except Exception as e:
            logger.error(f"Stock fetch failed: {e}")
            self.errors.append(f"Stock fetch: {e}")
            return []

    def run_news_aggregation(
        self, queries: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run news aggregation, honoring each query's freshness binding.

        Interest-graph queries carry the freshness window their branch asked
        for (hyperlocal branches need a wider window than market branches), so
        queries are grouped by window and each group is fetched separately.

        Args:
            queries: Optional list of queries. If None, uses config.

        Returns:
            List of news articles.
        """
        try:
            logger.info("=== Aggregating News ===")
            api_key = os.environ.get("BRAVE_API_KEY")
            if queries is None:
                queries = self.config.get("news_queries", [])
            max_news = self.config.get("max_news", 15)

            if not api_key:
                logger.warning("BRAVE_API_KEY not set, skipping news")
                return []

            if not queries:
                logger.warning("No news_queries configured, skipping")
                return []

            default_freshness = self.config.get("news_freshness", "pd")
            groups: Dict[str, List[str]] = {}
            for query in queries:
                groups.setdefault(
                    query_freshness(query, default_freshness), []
                ).append(str(query))

            articles: List[Dict[str, Any]] = []
            seen_urls = set()
            for freshness, group in groups.items():
                logger.info(
                    "News group freshness=%s queries=%d", freshness, len(group)
                )
                aggregator = NewsAggregator(
                    api_key=api_key,
                    queries=group,
                    max_results=max_news,
                    freshness=freshness,
                )
                for article in aggregator.aggregate_all_queries():
                    # Normalized so the same article fetched under two
                    # freshness windows is not printed twice.
                    key = normalize_url(article.get("url", ""))
                    if key and key in seen_urls:
                        continue
                    if key:
                        seen_urls.add(key)
                    articles.append(article)

            self.status["news_found"] = len(articles)
            logger.info(f"Found {len(articles)} news articles")
            return articles

        except Exception as e:
            logger.error(f"News aggregation failed: {e}")
            self.errors.append(f"News aggregation: {e}")
            return []

    def run_alerts_scan(self) -> List[Dict[str, Any]]:
        """
        Fetch active public-safety alerts (NWS) for the configured zones.

        Deterministic and LLM-free: alerts are rendered as fetched, so this
        section survives an LLM outage intact.

        Returns:
            List of active alerts, most severe first.
        """
        try:
            scanner = create_alerts_scanner(self.config)
            if scanner is None:
                return []
            logger.info("=== Fetching Public-Safety Alerts ===")
            alerts = scanner.fetch()
            self.status["alerts_found"] = len(alerts)
            logger.info(f"Found {len(alerts)} active alerts")
            return alerts
        except Exception as e:
            logger.error(f"Alerts scan failed: {e}")
            self.errors.append(f"Alerts scan: {e}")
            return []

    def _apply_geo_filter(
        self, items: List[Dict[str, Any]], label: str
    ) -> List[Dict[str, Any]]:
        """Drop items with no reference to the configured area (config-driven)."""
        before = len(items)
        kept = apply_config_filter(items, self.config, label=label)
        dropped = before - len(kept)
        if dropped:
            self.status["geo_filtered_out"] += dropped
        return kept

    def run_happenings_aggregation(self) -> List[Dict[str, Any]]:
        """
        Run neighborhood happenings aggregation (Brave Search).

        Uses a dedicated query list and a wider freshness window than the main
        news so event announcements published several days ahead are caught.

        Returns:
            List of happening articles.
        """
        try:
            logger.info("=== Aggregating Neighborhood Happenings ===")
            api_key = os.environ.get("BRAVE_API_KEY")
            queries = self.config.get("happenings_queries", [])
            max_happenings = self.config.get("max_happenings", 6)
            freshness = self.config.get("happenings_freshness", "pw")

            if not api_key:
                logger.warning("BRAVE_API_KEY not set, skipping happenings")
                return []

            if not queries:
                logger.warning("No happenings_queries configured, skipping")
                return []

            aggregator = NewsAggregator(
                api_key=api_key,
                queries=queries,
                max_results=max_happenings,
                freshness=freshness,
            )
            articles = aggregator.aggregate_all_queries()
            self.status["happenings_found"] = len(articles)
            logger.info(f"Found {len(articles)} happening articles")
            return articles

        except Exception as e:
            logger.error(f"Happenings aggregation failed: {e}")
            self.errors.append(f"Happenings aggregation: {e}")
            return []

    def _happenings_fetch_days(self) -> Optional[set]:
        """
        Weekdays on which happenings are re-fetched (0=Mon ... 6=Sun).

        Accepts a single weekday or a list of them; returns None when the key
        is absent, meaning "fetch every run". A list lets a cache refresh land
        both before the weekend and again midweek, so the section does not
        spend the back half of its cycle advertising events that have passed.
        """
        raw = self.config.get("happenings_fetch_weekday")
        if raw is None:
            return None
        if isinstance(raw, (list, tuple, set)):
            days = {int(d) for d in raw}
            return days or None
        return {int(raw)}

    def _load_or_fetch_happenings(
        self, previous_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        today = datetime.now()
        fetch_days = self._happenings_fetch_days()
        fetch_weekday = self.config.get("happenings_fetch_weekday")
        should_fetch = fetch_days is None or today.weekday() in fetch_days
        if should_fetch:
            logger.info(
                "Fetching fresh happenings (fetch weekday=%s, today=%s)",
                fetch_weekday, today.weekday(),
            )
            fetched = self._drop_past_happenings(
                self._dedupe_happenings_by_url(self.run_happenings_aggregation())
            )
            if fetched:
                self._happenings_cache = list(fetched)
                self._happenings_cache_date = today.strftime("%Y-%m-%d")
                logger.info(
                    "Cached %d raw happenings for the coming week",
                    len(fetched),
                )
                return fetched
            logger.warning("Happenings fetch returned empty, falling back to cache")
        cached = self._drop_past_happenings(
            self._dedupe_happenings_by_url(
                previous_state.get("cached_happenings", [])
            )
        )
        if cached:
            cache_date = previous_state.get("cached_happenings_date", "unknown")
            logger.info(
                "Reusing %d cached happenings from %s (fetch weekday=%s, today=%s)",
                len(cached), cache_date,
                fetch_weekday, today.weekday(),
            )
            self._happenings_cache = list(cached)
            self._happenings_cache_date = cache_date
            self.status["happenings_found"] = len(cached)
            return list(cached)
        logger.info("No happenings cache available and today is not fetch day")
        return []

    @staticmethod
    def _dedupe_happenings_by_url(
        happenings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collapse happenings that point at the same document.

        Applied to the cached path as well as the fresh one: a cache written
        before URL normalization existed still holds raw-string duplicates,
        and on 2026-08-29 two of them reached the reader — the same San Diego
        Magazine roundup under a trailing-slash variant, and the same KPBS
        piece under a ``www.`` variant.
        """
        seen = set()
        deduped = []
        for item in happenings or []:
            key = normalize_url(item.get("url", ""))
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(item)
        removed = len(happenings or []) - len(deduped)
        if removed:
            logger.info("Dedup: removed %d happening(s) with duplicate URLs", removed)
        return deduped

    def _drop_past_happenings(
        self, happenings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove happenings whose every calendar date has already passed.

        Happenings are cached between fetch days, so by the end of the window
        the cache can still be advertising last weekend. On 2026-08-28 the
        local briefing led with three "Aug. 21-23 this weekend" items — a week
        past — because the cache was only refreshed on Saturdays.

        A shorter cache window narrows this but cannot close it: a Thursday
        fetch still serves Saturday, and Brave's ``pw`` freshness returns
        results from the past week by design. So the dates are checked at use
        time rather than trusted to be fresh.

        Undated items are kept: a venue's standing events calendar or a rule
        change ("volleyball can now begin at 6 a.m.") is not stale for lacking
        a day. Ranges are judged by when they end.
        """
        if not happenings:
            return happenings
        today = datetime.now().date()
        kept, dropped = [], []
        for item in happenings:
            text = " ".join(
                str(item.get(field, ""))
                for field in ("title", "description", "age")
            )
            if has_only_past_dates(text, today):
                dropped.append(item.get("title", "")[:80])
            else:
                kept.append(item)
        if dropped:
            logger.info(
                "Dropped %d happening(s) whose dates have passed: %s",
                len(dropped), "; ".join(dropped),
            )
        return kept

    def score_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank papers."""
        try:
            if not papers:
                return []

            logger.info("=== Scoring Papers ===")
            topics = self.config.get("arxiv_topics", [])
            weights = self.config.get("paper_scoring", {})
            num_picks = self.config.get("num_paper_picks", 3)

            scorer = PaperScorer(topics=topics, weights=weights, num_picks=num_picks)
            top_papers = scorer.get_top_picks(papers)
            logger.info(f"Selected top {len(top_papers)} papers")
            return top_papers

        except Exception as e:
            logger.error(f"Paper scoring failed: {e}")
            self.errors.append(f"Paper scoring: {e}")
            return []

    def deduplicate_news_and_blogs(
        self,
        news: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
    ) -> tuple:
        """
        Remove duplicate content between news and blog sections.

        Args:
            news: News articles.
            blogs: Blog articles.

        Returns:
            Tuple of (deduplicated_news, deduplicated_blogs).
        """
        blog_domains = set()
        blog_titles_lower = set()

        for blog in blogs:
            link = blog.get("link", "")
            if link:
                try:
                    domain = urlparse(link).netloc.lower()
                    blog_domains.add(domain)
                except Exception:
                    pass
            title = blog.get("title", "").lower().strip()
            if title:
                blog_titles_lower.add(title)

        deduped_news = []
        for article in news:
            url = article.get("url", "")
            title = article.get("title", "").lower().strip()

            # Skip if same title appears in blogs
            if title and title in blog_titles_lower:
                logger.debug(f"Dedup: removing news '{title}' (duplicate of blog)")
                continue

            # Skip if URL points to same domain as a blog feed
            if url:
                try:
                    domain = urlparse(url).netloc.lower()
                    if domain in blog_domains:
                        logger.debug(f"Dedup: removing news from {domain} (covered by blog feed)")
                        continue
                except Exception:
                    pass

            deduped_news.append(article)

        removed = len(news) - len(deduped_news)
        if removed > 0:
            logger.info(f"Dedup: removed {removed} news articles duplicated in blogs")

        return deduped_news, blogs

    @staticmethod
    def deduplicate_happenings(
        happenings: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Drop happenings whose title already appears in the news section."""
        news_titles = set(n.get("title", "").lower().strip() for n in news if n.get("title"))
        deduped = [
            h for h in happenings
            if h.get("title", "").lower().strip() not in news_titles
        ]
        removed = len(happenings) - len(deduped)
        if removed:
            logger.info(f"Dedup: removed {removed} happenings duplicated in news")
        return deduped

    @staticmethod
    def _dedup_against_previous(
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        previous_state: Dict[str, Any],
        happenings: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple:
        """Remove papers, blogs, news, and happenings that appeared in yesterday's briefing."""
        if not previous_state:
            return papers, blogs, news, (happenings or [])

        prev_papers = set(t.lower() for t in previous_state.get("top_paper_titles", []))
        prev_blogs = set(t.lower() for t in previous_state.get("top_blog_titles", []))
        prev_news = set(t.lower() for t in previous_state.get("top_news_titles", []))
        prev_happenings = set(t.lower() for t in previous_state.get("top_happenings_titles", []))

        def _filter(items, prev_titles):
            before = len(items)
            filtered = [i for i in items if i.get("title", "").lower() not in prev_titles]
            removed = before - len(filtered)
            if removed:
                logger.info(f"Cross-day dedup: removed {removed} items seen yesterday")
            return filtered

        return (
            _filter(papers, prev_papers),
            _filter(blogs, prev_blogs),
            _filter(news, prev_news),
            _filter(happenings or [], prev_happenings),
        )

    # Defined in scripts/text_similarity.py so the quality check measures
    # duplicates exactly the way this collapse does -- two copies of the rule
    # would silently drift apart.
    _DEDUP_STOPWORDS = DEDUP_STOPWORDS

    @classmethod
    def _headline_terms(cls, title: str) -> set:
        """Content words of a headline, for cross-outlet duplicate detection."""
        return headline_terms(title)

    def deduplicate_similar_news(
        self, news: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Collapse the same story reported by many outlets.

        A viral local story ("Darth Vader speaks at city council") arrives from
        a dozen syndicating outlets with different headlines, crowding out
        everything else. URL dedup does not catch it and title SequenceMatcher
        is unreliable across rewrites, so this uses Jaccard overlap of headline
        content words.

        Threshold calibrated on real Brave results: cross-outlet retellings of
        one story scored 0.23-0.47, while genuinely distinct local stories
        scored 0.00-0.17. The default 0.3 sits in that gap.

        When a duplicate is found, the copy from a local outlet
        (``geo_filter.trusted_sources``) wins over a national aggregator.
        """
        cfg = self.config.get("news_similarity_dedup") or {}
        if not cfg.get("enabled") or len(news) <= 1:
            return news

        threshold = float(cfg.get("threshold", 0.3))
        trusted = {
            t.lower() for t in
            (self.config.get("geo_filter", {}) or {}).get("trusted_sources", [])
        }

        def is_trusted(article: Dict[str, Any]) -> bool:
            host = (article.get("source") or "").lower()
            return any(host == t or host.endswith("." + t) for t in trusted)

        kept: List[Dict[str, Any]] = []
        kept_terms: List[set] = []
        for article in news:
            terms = self._headline_terms(article.get("title", ""))
            if not terms:
                kept.append(article)
                kept_terms.append(terms)
                continue

            dup_index = None
            for i, other in enumerate(kept_terms):
                if not other:
                    continue
                overlap = len(terms & other) / len(terms | other)
                if overlap >= threshold:
                    dup_index = i
                    break

            if dup_index is None:
                kept.append(article)
                kept_terms.append(terms)
            elif is_trusted(article) and not is_trusted(kept[dup_index]):
                # Prefer the local outlet's version of the same story.
                kept[dup_index] = article
                kept_terms[dup_index] = terms

        removed = len(news) - len(kept)
        if removed:
            logger.info(
                "Dedup: collapsed %d syndicated copies of stories already kept",
                removed,
            )
        return kept

    def deduplicate_similar_papers(
        self, papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove papers with very similar titles (>85% match).

        Catches near-duplicates found via different topic queries.

        Args:
            papers: List of paper dictionaries.

        Returns:
            Deduplicated paper list.
        """
        if len(papers) <= 1:
            return papers

        deduped = []
        for paper in papers:
            title = paper.get("title", "").lower()
            is_dup = False
            for kept in deduped:
                kept_title = kept.get("title", "").lower()
                if SequenceMatcher(None, title, kept_title).ratio() > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(paper)

        removed = len(papers) - len(deduped)
        if removed:
            logger.info(f"Dedup: removed {removed} near-duplicate papers by title similarity")
        return deduped

    def generate_markdown_briefing(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        stocks: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        top_papers: List[Dict[str, Any]],
        synthesis: Optional[Dict[str, str]] = None,
        market_trend: str = "",
        weekly_deep_dive: str = "",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        happenings: Optional[List[Dict[str, Any]]] = None,
        alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate markdown briefing from all data.

        Args:
            papers: ArXiv papers.
            blogs: Blog articles.
            stocks: Stock data.
            news: News articles.
            top_papers: Top-scored papers.
            synthesis: Optional intelligence synthesis output.
            market_trend: Pre-generated market trend summary.
            weekly_deep_dive: Optional weekly deep dive section (Saturday only).
            start_time: Optional briefing start unix timestamp.
            end_time: Optional briefing end unix timestamp.
            happenings: Optional neighborhood happenings (Pacific Beach area).
            alerts: Optional active public-safety alerts (NWS).

        Returns:
            Markdown string.
        """
        logger.info("=== Generating Briefing ===")

        md = []

        # Add title and timestamp (localized with timezone)
        now = datetime.now().astimezone()
        timestamp_str = now.strftime("%A, %B %d, %Y | %I:%M %p %Z")
        md.append(f"# {self.config.get('briefing_title', 'Atlas Morning Briefing')}\n")
        
        user_suffix = f" [RIGHT]for {self.user_name}[/RIGHT]" if self.user_name else ""
        md.append(f"*{timestamp_str}{user_suffix}*\n\n")
        md.append("---\n\n")

        # Editorial intro (from synthesis)
        if synthesis and synthesis.get("editorial_intro"):
            intro = synthesis["editorial_intro"].strip()
            # Aggressively strip LLM preamble: headings, titles, dates
            lines = intro.split("\n")
            cleaned = []
            for line in lines:
                stripped = line.strip()
                # Skip markdown headings
                if stripped.startswith("#"):
                    continue
                # Skip lines containing "Executive Summary" (LLM echo)
                if "executive summary" in stripped.lower():
                    continue
                # Skip lines containing "Morning Briefing" or "AI Briefing" (LLM title echo)
                if "morning briefing" in stripped.lower() or "ai briefing" in stripped.lower():
                    continue
                # Skip stray date lines (e.g. "– 2026-03-08", "2026-03-07")
                date_stripped = stripped.lstrip("–—-*# ").strip()
                if re.match(r"^\d{4}-\d{2}-\d{2}$", date_stripped):
                    continue
                cleaned.append(line)
            intro = "\n".join(cleaned).strip()

            # Guard: detect leaked reasoning / verification scaffolding.
            # Reasoning models can emit their grounding-verification trace
            # as visible output (e.g. "Strict Grounding Verification**:" or
            # "Check verbatim entities/facts:" bullet lists).  If the cleaned
            # intro contains those telltales — drop it and fall back to the
            # unavailable placeholder.
            _looks_like_cot = is_cot_leak(intro)
            if _looks_like_cot:
                logger.warning(
                    "Editorial intro looks like leaked CoT scaffolding; "
                    "falling back to placeholder."
                )
                intro = ""

            if intro:
                md.append(f"## {self._headings.get('executive_summary', 'Executive Summary')}\n\n")
                md.append(f"{intro}\n\n")
            else:
                md.append(f"## {self._headings.get('executive_summary', 'Executive Summary')}\n\n")
                md.append(SYNTHESIS_UNAVAILABLE_TEXT)
                self._record_degraded_synthesis()
        else:
            md.append(f"## {self._headings.get('executive_summary', 'Executive Summary')}\n\n")
            md.append(SYNTHESIS_UNAVAILABLE_TEXT)
            self._record_degraded_synthesis()

        if self.feature_solo_founder_angle:
            solo_angle = synthesis.get("solo_startup", "") if synthesis else ""
            if solo_angle:
                md.append(f"## {self._headings.get('solo_founder_angle', 'Solo Founder Angle')}\n\n")
                md.append(f"{solo_angle}\n\n")

        if self.feature_agent_cost_optimization:
            cost_play = synthesis.get("agent_cost_play", "") if synthesis else ""
            if cost_play:
                md.append(f"## {self._headings.get('agent_cost_optimization', 'Agent Cost-Optimization Play')}\n\n")
                md.append(f"{cost_play}\n\n")

            # Feature 3: Entity Watch — DISABLED per user request (2026-03-08)
            # Only show if an entity has a spike (e.g., 5+ mentions).
            # entity_mentions = synthesis.get("entity_mentions", [])
            # if entity_mentions: ...

        section_order = self.section_order

        # Section renderers
        section_data = {
            "stocks": stocks,
            "news": news,
            "blogs": blogs,
            "top_papers": top_papers,
            "papers": papers,
            "happenings": happenings or [],
            "alerts": alerts or [],
        }
        logger.debug(
            "Generating markdown: section_order=%s headings_keys=%s "
            "briefing_title=%s sections_with_data=%s",
            section_order, list(self._headings.keys()),
            self.config.get('briefing_title', 'Atlas Morning Briefing'),
            [k for k, v in section_data.items() if v],
        )

        for section in section_order:
            data = section_data.get(section, [])
            if not data:
                continue

            if section == "stocks":
                md.append(self._render_stocks(data, market_trend=market_trend))
            elif section == "news":
                md.append(self._render_news(data))
            elif section == "blogs":
                md.append(self._render_blogs(data))
            elif section == "top_papers":
                md.append(self._render_top_papers(data))
            elif section == "papers":
                md.append(self._render_papers(data))
            elif section == "happenings":
                md.append(self._render_happenings(data))
            elif section == "alerts":
                md.append(self._render_alerts(data))

        # Feature 2: Weekly Deep Dive section (Saturday only)
        if self.feature_weekly_deep_dive and weekly_deep_dive:
            # Guard: drop leaked reasoning / verification scaffolding.
            if weekly_deep_dive and is_cot_leak(weekly_deep_dive):
                logger.warning(
                    "Weekly deep dive looks like leaked CoT scaffolding; "
                    "omitting section."
                )
                weekly_deep_dive = ""
            if weekly_deep_dive:
                md.append(f"## {self._headings.get('weekly_deep_dive', 'This Week in AI')}\n\n")
                md.append(f"{weekly_deep_dive}\n\n")

        # Errors section
        if self.errors:
            md.append(f"## {self._headings.get('errors', 'Errors')}\n\n")
            for error in self.errors:
                md.append(f"- {error}\n")
            md.append("\n")

        # Gemini Usage Summary
        if self.llm_client:
            md.append(self.llm_client.get_usage_summary(start_time=start_time, end_time=end_time))

        return "".join(md)

    # Preflight results older than this are ignored: a stale file would pin a
    # model chosen for yesterday's conditions, and the whole point of the check
    # is that free-model availability changes hour to hour.
    PREFLIGHT_MAX_AGE_SECONDS = 6 * 3600

    def _load_preflight_models(self) -> Dict[str, Any]:
        """Load .model-availability.json written by scripts/preflight_model_check.py.

        Returns an empty dict (meaning "use the configured models") when the
        file is missing, unreadable, or stale.
        """
        preflight_path = Path(
            self.config.get("preflight_file_path", ".model-availability.json")
        )
        if not preflight_path.exists():
            logger.info(
                "No preflight model availability at %s; using configured models",
                preflight_path,
            )
            return {}
        try:
            with open(preflight_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load preflight data from %s: %s", preflight_path, e)
            return {}
        if not isinstance(data, dict):
            logger.warning("Preflight data in %s is not an object; ignoring", preflight_path)
            return {}

        age = time.time() - preflight_path.stat().st_mtime
        if age > self.PREFLIGHT_MAX_AGE_SECONDS:
            logger.warning(
                "Preflight data in %s is %.1f hours old (max %.1f); using configured models",
                preflight_path, age / 3600, self.PREFLIGHT_MAX_AGE_SECONDS / 3600,
            )
            return {}

        available = {
            provider: [
                tier for tier, entry in tiers.items()
                if isinstance(entry, dict) and entry.get("available")
            ]
            for provider, tiers in data.items()
            if isinstance(tiers, dict)
        }
        logger.info(
            "Loaded preflight model availability from %s (%.0f min old): %s",
            preflight_path, age / 60,
            {k: v for k, v in available.items() if v} or "nothing available",
        )
        return data

    def _format_filename(self, now: datetime) -> str:
        """Format the output filename from config pattern, ignoring unknown keys."""
        return format_briefing_filename(
            self.config.get("file_naming", DEFAULT_FILE_NAMING), now
        )

    def _enrich_papers(self, papers: list, topics: list) -> list:
        """Run paper summarization + semantic scoring sequentially (used in parallel batch)."""
        papers = self.intelligence.summarize_papers(papers)
        papers = self.intelligence.score_papers_semantically(papers, topics)
        return papers

    def _analyze_market_trend(self, stocks: List[Dict[str, Any]]) -> str:
        """Generate a 2-line market trend summary from stock data."""
        if not stocks or not self.intelligence.available:
            return ""
        stock_lines = []
        for s in stocks:
            if "error" not in s:
                pct = s.get("percent_change", 0)
                sign = "+" if pct >= 0 else ""
                corr = s.get("news_correlation", "")
                line = f"{s.get('symbol', '')}: {sign}{pct:.2f}%"
                if corr:
                    line += f" ({corr})"
                stock_lines.append(line)
        if not stock_lines:
            return ""
        data_block = "\n".join(stock_lines)
        prompt = (
            "You are a financial analyst. Given today's stock movements, "
            "write exactly 2 sentences summarizing the market trend and key drivers. "
            "Be specific about which sectors/stocks moved and why.\n\n"
            f"<stock_data>\n{data_block}\n</stock_data>"
        )
        result = self.intelligence.client.invoke(
            prompt, tier="light"
        )
        return result.strip() if result else ""

    def _render_stocks(self, stocks: List[Dict[str, Any]], market_trend: str = "") -> str:
        """Render stock watchlist as compact overview table with trend analysis."""
        md = [f"## {self._headings.get('stocks', 'Financial Market Overview')}\n\n"]

        if market_trend:
            md.append(f"{market_trend}\n\n")

        md.append("| Ticker | Price | Change | Driver |\n")
        md.append("|--------|-------|--------|--------|\n")
        for stock in stocks:
            if "error" in stock:
                md.append(f"| {stock['symbol']} | — | Error | — |\n")
                continue

            symbol = stock.get("symbol", "")
            price = stock.get("current_price", 0)
            pct = stock.get("percent_change", 0)
            sign = "+" if pct >= 0 else ""
            driver = stock.get("news_correlation", "")
            if len(driver) > 30:
                driver = driver[:27] + "..."

            md.append(f"| **{symbol}** | ${price:.2f} | {sign}{pct:.2f}% | {driver} |\n")
        md.append("\n")
        return "".join(md)

    @staticmethod
    def _render_stars(score: int) -> str:
        """Render score as Amazon-style stars. 5 filled = best, 5 empty = worst."""
        if score is None:
            return ""
        score = max(0, min(score, 5))
        return "★" * score + "☆" * (5 - score)

    @staticmethod
    def _clean_summary(summary: str, title: str, source: str = "") -> str:
        """Remove title/source echo from LLM-generated summary."""
        if not summary:
            return summary
        # Strip leading * / ** markdown bold and "Summary:" prefix
        s = summary.lstrip("* ").strip()
        if s.lower().startswith("summary:"):
            s = s[8:].lstrip("* ").strip()
        if not title:
            return s
        # Check if summary starts with title text
        title_lower = title.lower()[:40]
        if s.lower().startswith(title_lower):
            rest = s[len(title):].strip()
            if rest.startswith("(") and ")" in rest:
                rest = rest[rest.index(")") + 1:].strip()
            if rest.startswith(("-", ":", "\u2013")):
                rest = rest[1:].strip()
            return rest if rest else summary
        return s

    def _render_news(self, news: List[Dict[str, Any]]) -> str:
        """Render news section (top 5, with summaries)."""
        md = [f"## {self._headings.get('news', 'AI & Tech News')}\n\n"]
        for article in news[:5]:
            article_title = article.get("title", "")
            url = article.get("url", "")
            summary = self._clean_summary(
                article.get("brief_summary", ""), article_title
            )

            if url:
                md.append(f"**[{article_title}]({url})**\n")
            else:
                md.append(f"**{article_title}**\n")
            if summary:
                md.append(f"{summary}\n")

            author_blurb = article.get("author_blurb")
            if author_blurb:
                md.append(f"\n#### Source Information\n{author_blurb}\n")

            md.append("\n")
        return "".join(md)

    @staticmethod
    def _format_alert_window(onset: str, expires: str) -> str:
        """Render an alert's active window as a compact local-time range."""
        def fmt(value: str) -> str:
            if not value:
                return ""
            try:
                return datetime.fromisoformat(value).strftime("%a %b %-d, %-I:%M %p")
            except (ValueError, TypeError):
                return value

        start, end = fmt(onset), fmt(expires)
        if start and end:
            return f"{start} → {end}"
        return start or end

    def _render_alerts(self, alerts: List[Dict[str, Any]]) -> str:
        """Render active public-safety alerts (deterministic, no LLM needed)."""
        max_alerts = self.config.get("alerts", {}).get("max_alerts", 5)
        items = alerts[:max_alerts]
        if not items:
            return ""

        md = [f"## {self._headings.get('alerts', 'Active Alerts')}\n\n"]
        for alert in items:
            event = alert.get("event", "Alert")
            severity = alert.get("severity", "")
            window = self._format_alert_window(
                alert.get("onset", ""), alert.get("expires", "")
            )
            md.append(f"**{event}**" + (f" — {severity}" if severity else "") + "\n")
            if window:
                md.append(f"{window}\n")
            area = alert.get("area", "")
            if area:
                md.append(f"{area}\n")
            instruction = (alert.get("instruction") or "").replace("\n", " ").strip()
            if instruction:
                if len(instruction) > 300:
                    instruction = instruction[:297].rstrip() + "..."
                md.append(f"\n{instruction}\n")
            md.append("\n")
        md.append(f"_Source: {items[0].get('source', 'National Weather Service')}_\n\n")
        return "".join(md)

    def _render_happenings(self, happenings: List[Dict[str, Any]]) -> str:
        """Render neighborhood happenings section (upcoming events near Pacific Beach)."""
        max_happenings = self.config.get("max_happenings", 6)
        items = happenings[:max_happenings]
        if not items:
            return ""
        md = [f"## {self._headings.get('happenings', 'Pacific Beach Area — Upcoming Happenings')}\n\n"]
        for article in items:
            article_title = article.get("title", "")
            url = article.get("url", "")
            summary = self._clean_summary(
                article.get("brief_summary", ""), article_title
            )

            if url:
                md.append(f"**[{article_title}]({url})**\n")
            else:
                md.append(f"**{article_title}**\n")
            if summary:
                md.append(f"{summary}\n")
            md.append("\n")
        return "".join(md)

    def _render_blogs(self, blogs: List[Dict[str, Any]]) -> str:
        """Render blog updates section (top 5, with summaries, sorted by score)."""
        md = [f"## {self._headings.get('blogs', 'Blog Updates')}\n\n"]
        
        # Sort by score if available, otherwise use original order
        if any(b.get("score_combined") for b in blogs):
            sorted_blogs = sorted(blogs[:8], key=lambda x: x.get("score_combined", 0), reverse=True)
            # Only filter low scores if we actually have scores
            sorted_blogs = [b for b in sorted_blogs if b.get("score_combined", 0) >= 3]
        else:
            sorted_blogs = blogs[:5]

        if not sorted_blogs:
            return ""

        for article in sorted_blogs:
            article_title = article.get("title", "")
            source = article.get("source", "")
            link = article.get("link", "")
            score = article.get("score_combined")
            summary = self._clean_summary(
                article.get("brief_summary", article.get("summary", "")), article_title, source
            )

            score_tag = f" {self._render_stars(score)}" if score else ""
            if link:
                md.append(f"**[{article_title}]({link})** *({source})*{score_tag}\n")
            else:
                md.append(f"**{article_title}** *({source})*{score_tag}\n")
            if summary:
                # Limit summary length if it wasn't processed by LLM
                if not article.get("brief_summary") and len(summary) > 300:
                    summary = summary[:297] + "..."
                md.append(f"{summary}\n")

            author_blurb = article.get("author_blurb")
            if author_blurb:
                md.append(f"\n#### Source Information\n{author_blurb}\n")

            md.append("\n")
        return "".join(md)

    def _ensure_paper_summaries(
        self, papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ensure each paper has a brief_summary and score. Batch-generate missing ones."""
        missing = [
            (i, p) for i, p in enumerate(papers)
            if not (p.get("brief_summary") and p.get("score_combined"))
        ]
        if not missing or not self.intelligence.available:
            return papers

        paper_texts = []
        indices = []
        for i, paper in missing:
            title = paper.get("title", "")
            abstract = paper.get("summary", "")[:600]
            if not abstract:
                continue
            paper_texts.append(f"[{len(paper_texts)+1}] {title}\n{abstract}")
            indices.append(i)

        if not paper_texts:
            return papers

        papers_block = "\n\n".join(paper_texts)
        prompt = (
            "For each paper, write a 2-3 sentence summary of its key contribution "
            "and rate it.\n\n"
            f"<papers>\n{papers_block}\n</papers>\n\n"
            "Respond in this exact format for each paper:\n"
            "[number] SCORE:X/5 Your 2-3 sentence summary here.\n\n"
            "SCORE is a combined rating (1-5) of impact, complexity, and innovation. "
            "5 = groundbreaking, 1 = routine.\n"
            "Be factual. Do not add information not in the abstract."
        )
        result = self.intelligence.client.invoke(
            prompt, tier="light"
        )
        if not result:
            return papers

        parsed = self.intelligence._parse_ranked_response(result)
        for rank_idx, text in parsed:
            if 0 <= rank_idx < len(indices):
                paper_idx = indices[rank_idx]
                score, summary = self.intelligence.extract_score(text)
                papers[paper_idx]["brief_summary"] = summary
                if score:
                    papers[paper_idx]["score_combined"] = score
                logger.info(
                    f"Generated summary+score for: {papers[paper_idx].get('title', '')[:50]}"
                )
        return papers

    def _render_top_papers(self, top_papers: List[Dict[str, Any]]) -> str:
        """Render top papers section (top N per config.num_paper_picks, with summaries, scores, and repro assessment)."""
        md = [f"## {self._headings.get('top_papers', 'Top Papers')}\n\n"]
        num_picks = self.config.get("num_paper_picks", 5)
        sorted_papers = sorted(top_papers[:num_picks], key=lambda x: x.get("score_combined", 0), reverse=True)
        if any(p.get("score_combined") for p in sorted_papers):
            sorted_papers = [p for p in sorted_papers if p.get("score_combined", 0) >= 3]
        else:
            sorted_papers = top_papers[:3]

        if not sorted_papers:
            md.append("*No highly relevant papers found today based on scoring.*\n\n")
            return "".join(md)

        for i, paper in enumerate(sorted_papers, 1):
            paper_title = paper.get("title", "")
            authors = paper.get("authors", [])
            arxiv_url = paper.get("arxiv_url", "")
            brief_summary = paper.get("brief_summary", paper.get("summary", ""))
            relevance_reason = paper.get("relevance_reason", "")
            score = paper.get("score_combined")
            repro_total = paper.get("repro_total")
            repro_verdict = paper.get("repro_verdict", "")
            difficulty = paper.get("reproduction_difficulty", "")

            score_tag = f" {self._render_stars(score)}" if score else ""
            pdf_link = paper.get("pdf_link", "")
            if arxiv_url:
                md.append(f"### {i}. [{paper_title}]({arxiv_url}){score_tag}\n")
            else:
                md.append(f"### {i}. {paper_title}{score_tag}\n")
            if authors:
                md.append(f"*{', '.join(authors[:3])}*\n\n")
            # Link row: abstract + PDF
            link_parts = []
            if arxiv_url:
                link_parts.append(f"[abs]({arxiv_url})")
            if pdf_link:
                link_parts.append(f"[📄 PDF]({pdf_link})")
            if link_parts:
                md.append("🔗 " + " · ".join(link_parts) + "\n\n")

            if brief_summary:
                # Limit length if it's the raw abstract
                if not paper.get("brief_summary") and len(brief_summary) > 600:
                    brief_summary = brief_summary[:597] + "..."
                md.append(f"{brief_summary}\n\n")
            elif relevance_reason:
                md.append(f"{relevance_reason}\n\n")

            author_blurb = paper.get("author_blurb")
            if author_blurb:
                md.append(f"#### Source Information\n{author_blurb}\n\n")

            # Show reproduction feasibility badge
            if repro_total is not None:
                badge = "✅" if repro_total >= 18 else "🟡" if repro_total >= 12 else "🔴"
                md.append(f"**Repro: {badge} {repro_total}/25** ({difficulty})")
                if repro_verdict:
                    md.append(f" — {repro_verdict}")
                md.append("\n")

            md.append("\n\n")
        return "".join(md)

    def _render_papers(self, papers: List[Dict[str, Any]]) -> str:
        """Render recent papers section (top 5, compact)."""
        md = [f"## {self._headings.get('recent_papers', 'Recent Papers')}\n\n"]
        for paper in papers[:5]:
            paper_title = paper.get("title", "")
            authors = paper.get("authors", [])
            arxiv_url = paper.get("arxiv_url", "")
            brief_summary = paper.get("brief_summary", "")

            md.append(f"**{paper_title}**")
            if authors:
                md.append(f" *{', '.join(authors[:2])}*")
            if arxiv_url:
                md.append(f" [arxiv]({arxiv_url})")
            md.append("\n")
            if brief_summary:
                md.append(f"{brief_summary}\n")

            author_blurb = paper.get("author_blurb")
            if author_blurb:
                md.append(f"\n#### Source Information\n{author_blurb}\n")

            md.append("\n")
        return "".join(md)

    def generate_pdf(self, markdown_content: str, output_path: str) -> bool:
        """Generate PDF from markdown."""
        try:
            pdf_config = self.config.get("pdf", {})
            if not pdf_config.get("enabled", True):
                logger.info("PDF generation disabled in config, skipping")
                return True

            logger.info("=== Generating PDF ===")
            page_format = self.config.get("output_format", "kindle")
            font_size = pdf_config.get("font_size", 10)
            line_spacing = pdf_config.get("line_spacing", 1.5)

            generator = PDFGenerator(
                page_format=page_format,
                font_size=font_size,
                line_spacing=line_spacing,
            )
            generator.generate_pdf(markdown_content, output_path)
            self.status["pdf_generated"] = True
            return True

        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            self.errors.append(f"PDF generation: {e}")
            return False

    def generate_epub(self, markdown_content: str, output_path: str) -> bool:
        """Generate EPUB from markdown."""
        try:
            logger.info("=== Generating EPUB ===")
            epub_cfg = self.config.get("epub", {})
            epub_title_fmt = epub_cfg.get("title_format", "Morning Briefing - {date}")
            epub_author = epub_cfg.get("author", "Atlas")
            generator = EPUBGenerator(
                title=epub_title_fmt.format(date=datetime.now().strftime("%Y-%m-%d")),
                author=epub_author
            )
            generator.generate_epub(markdown_content, output_path)
            self.status["epub_generated"] = True
            return True

        except Exception as e:
            logger.error(f"EPUB generation failed: {e}")
            self.errors.append(f"EPUB generation: {e}")
            return False

    def distribute_briefing(
        self, markdown_content: str, pdf_path: str, subject: str, epub_path: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Distribute briefing to all configured channels.

        Sends PDF/EPUB to Kindle + rich HTML to email recipients.

        Args:
            markdown_content: Markdown briefing content.
            pdf_path: Path to generated PDF.
            subject: Email subject / filename.
            epub_path: Optional path to generated EPUB.

        Returns:
            Dictionary mapping channel -> success boolean.
        """
        if self.dry_run:
            logger.info("Dry run: Skipping all distribution")
            return {}

        sender_email = os.environ.get("GMAIL_USER")
        sender_password = os.environ.get("GMAIL_APP_PASSWORD")

        if not sender_email or not sender_password:
            logger.warning("Gmail credentials not set, skipping distribution")
            return {}

        try:
            distributor = EmailDistributor(
                sender_email=sender_email,
                sender_password=sender_password,
            )

            results = distributor.distribute(
                config=self.config,
                markdown_content=markdown_content,
                pdf_path=pdf_path,
                epub_path=epub_path,
                subject=subject,
                dry_run=self.dry_run,
            )

            # Update status
            sent_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            self.status["email_sent"] = sent_count > 0
            self.status["distribution"] = {
                "sent": sent_count,
                "total": total_count,
                "details": results,
            }

            logger.info(f"Distribution: {sent_count}/{total_count} channels delivered")
            return results

        except Exception as e:
            logger.error(f"Distribution failed: {e}")
            self.errors.append(f"Distribution: {e}")
            return {}

    def _record_degraded_synthesis(self) -> None:
        """
        Record that the briefing shipped without its lead section.

        Falling back to the placeholder is correct behavior when every LLM
        backend is down, but a run that delivers no Executive Summary is not a
        clean run. Without this the status file reported errors: [] on a
        briefing whose most-read section was a stub -- which is exactly the
        kind of silent degradation the quality check exists to surface.
        """
        message = "Executive summary unavailable (LLM synthesis failed)"
        if message not in self.errors:
            self.errors.append(message)
        self.status["synthesis_degraded"] = True
        logger.warning(message)

    def save_status(self, output_dir: str = ".") -> None:
        """
        Save run status to JSON file for monitoring.

        The filename is config-driven because a machine can run more than one
        pipeline: run_briefing.sh runs the main briefing and then the local one
        ~15 minutes later, and with a shared filename the second run silently
        overwrites the first run's counters. Monitoring that reads the file
        would then report one pipeline's numbers as if they were both — which
        is exactly what happened on 2026-08-28, when status.json showed
        papers_found: 0 from the local pipeline (which scans no papers) while
        the main run had in fact collected 172.

        Args:
            output_dir: Directory to save status file.
        """
        self.status["errors"] = self.errors
        self.status["pipeline"] = self.config.get("pipeline_name", "")
        status_filename = self.config.get("status_file_path", "status.json")
        status_path = Path(output_dir) / status_filename
        try:
            with open(status_path, "w") as f:
                json.dump(self.status, f, indent=2)
            logger.info(f"Status saved: {status_path}")
        except IOError as e:
            logger.warning(f"Failed to save status: {e}")

    def _load_previous_state(self) -> Dict[str, Any]:
        """Load previous briefing state for cross-day trend tracking."""
        state_path = Path(self.state_file_path)
        if state_path.exists():
            try:
                with open(state_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_state(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        stocks: List[Dict[str, Any]],
        emerging_themes: List[str],
        trending_topics: Optional[Dict[str, Any]] = None,
        weekly_items: Optional[List[Dict[str, Any]]] = None,
        happenings: Optional[List[Dict[str, Any]]] = None,
        cached_happenings: Optional[List[Dict[str, Any]]] = None,
        cached_happenings_date: Optional[str] = None,
    ) -> None:
        """Save current briefing state for next run's trend tracking and dedup."""
        state = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "top_paper_titles": [p.get("title", "") for p in papers[:10]],
            "top_blog_titles": [b.get("title", "") for b in blogs[:10]],
            "top_news_titles": [n.get("title", "") for n in news[:10]],
            "top_happenings_titles": [h.get("title", "") for h in (happenings or [])[:10]],
            "stock_closes": {
                s.get("symbol", ""): s.get("current_price", 0)
                for s in stocks if "error" not in s
            },
            "emerging_themes": emerging_themes,
        }
        if cached_happenings:
            state["cached_happenings"] = cached_happenings
            state["cached_happenings_date"] = cached_happenings_date or datetime.now().strftime("%Y-%m-%d")
        # Feature 1: Save trending topics
        if trending_topics is not None:
            state["trending_topics"] = trending_topics
        # Feature 2: Save weekly items for Saturday deep dive
        if weekly_items is not None:
            state["weekly_items"] = weekly_items
        try:
            with open(self.state_file_path, "w") as f:
                json.dump(state, f, indent=2)
        except IOError:
            pass

    def run(self) -> int:
        """
        Run the complete briefing pipeline.

        Returns:
            Exit code (0=success, 1=partial failure, 2=total failure).
        """
        start_time = time.time()
        logger.info("=== Starting Morning Briefing ===")

        # --- Load previous state for cross-day tracking ---
        previous_state = self._load_previous_state()

        if self.use_snapshots:
            # --- Load from saved snapshots (skip all live API calls) ---
            logger.info(
                "=== Loading from snapshots/%s (skipping live fetches) ===",
                self.use_snapshots,
            )
            papers = self._load_snapshot(
                f"snapshots/{self.use_snapshots}/arxiv_papers.json"
            )
            blogs = self._load_snapshot(
                f"snapshots/{self.use_snapshots}/rss_feeds.json"
            )
            stocks = self._load_snapshot(
                f"snapshots/{self.use_snapshots}/finnhub_data.json"
            )
            news = self._load_snapshot(
                f"snapshots/{self.use_snapshots}/brave_news.json"
            )
            happenings = self._load_snapshot(
                f"snapshots/{self.use_snapshots}/happenings.json"
            )
            alerts = self._load_snapshot(
                f"snapshots/{self.use_snapshots}/alerts.json"
            )
            topics = self.config.get("arxiv_topics", [])
            news_queries = self.config.get("news_queries", [])
            self.status["papers_found"] = len(papers)
            self.status["blogs_found"] = len(blogs)
            self.status["stocks_fetched"] = len(stocks)
            self.status["news_found"] = len(news)
            self.status["happenings_found"] = len(happenings)
            self.status["alerts_found"] = len(alerts)
        else:
            # --- Topic expansion (intelligence layer) ---
            topics = self.config.get("arxiv_topics", [])
            if self.intelligence.available:
                logger.info("=== Intelligence Layer: Expanding Topics ===")
                topics = self.intelligence.expand_topics(topics)

            # --- Run scanners in parallel (papers + blogs + stocks are independent) ---
            from concurrent.futures import ThreadPoolExecutor

            logger.info("=== Parallel data fetch (papers/blogs/stocks) ===")
            with ThreadPoolExecutor(max_workers=self.config.get("max_workers", 1)) as pool:
                fut_papers = pool.submit(self.run_arxiv_scan, topics)
                fut_blogs = pool.submit(self.run_blog_scan)
                fut_stocks = pool.submit(self.run_stock_fetch)
                fut_alerts = pool.submit(self.run_alerts_scan)

                papers = fut_papers.result()
                blogs = fut_blogs.result()
                stocks = fut_stocks.result()
                alerts = fut_alerts.result()

            # --- Generate news queries (interest graph, or static + dynamic) ---
            news_queries = self.config.get("news_queries", [])
            if self.intelligence.available or self.config.get("interest_graph"):
                logger.info("=== Generating News Queries ===")
                news_queries = self.intelligence.generate_dynamic_queries(
                    previous_state, news_queries, today_blogs=blogs
                )

            news = self.run_news_aggregation(queries=news_queries)
            happenings = self._load_or_fetch_happenings(previous_state)

            # Geographic relevance gate: news search treats a place name as a
            # ranking hint, not a constraint, so drop out-of-area results
            # before they reach the LLM ranking layer.
            news = self._apply_geo_filter(news, "news")
            happenings = self._apply_geo_filter(happenings, "happenings")

            # --- Save raw data snapshots ---
            logger.info("=== Saving raw data snapshots ===")
            self.snapshot_manager.save_stocks(stocks)
            self.snapshot_manager.save_news(news)
            self.snapshot_manager.save_happenings(happenings)
            self.snapshot_manager.save_alerts(alerts)
            self.snapshot_manager.save_blogs(blogs)
            self.snapshot_manager.save_papers(papers)
            self.snapshot_manager.save_manifest()

        # --- Cross-section deduplication ---
        news, blogs = self.deduplicate_news_and_blogs(news, blogs)
        news = self.deduplicate_similar_news(news)
        happenings = self.deduplicate_happenings(happenings, news)

        # --- Deduplicate similar papers by title ---
        papers = self.deduplicate_similar_papers(papers)

        # --- Cross-day deduplication (skip items from yesterday) ---
        # Note: happenings cross-day dedup is skipped because they are
        # cached weekly (freshness="pw" already covers the week); we
        # intentionally keep them across days so the same upcoming-events
        # list is referenced Monday through Saturday.
        papers, blogs, news, _ = self._dedup_against_previous(
            papers, blogs, news, previous_state
        )

        # --- Intelligence layer: enrich data ---
        synthesis = {}
        emerging_themes = []
        if self.intelligence.available:
            logger.info("=== Intelligence Layer: Enriching Data ===")
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Two-stage relevance filtering (NEW) — must run first (reduces paper count)
            interest_profile = self.config.get("interest_profile")
            if interest_profile:
                logger.info("=== Intelligence Layer: Stage 1 Relevance Filtering ===")
                papers = self.intelligence.filter_papers_by_relevance(papers, interest_profile)

            # --- Parallel batch 1: papers, news, blogs, happenings are independent ---
            logger.info("=== Intelligence Layer: Parallel enrichment (papers/news/blogs/happenings) ===")
            with ThreadPoolExecutor(max_workers=self.config.get("max_workers", 1)) as pool:
                fut_papers = pool.submit(self._enrich_papers, papers, topics)
                fut_news = pool.submit(self.intelligence.rank_and_summarize_news, news, topics)
                fut_blogs = pool.submit(self.intelligence.rank_and_summarize_blogs, blogs, topics)
                if happenings:
                    fut_happenings = pool.submit(
                        self.intelligence.rank_and_summarize_happenings,
                        happenings,
                        self.config.get("max_happenings", 6),
                    )

                papers = fut_papers.result()
                news = fut_news.result()
                blogs = fut_blogs.result()
                if happenings:
                    happenings = fut_happenings.result()

            # --- Parallel batch 2: stocks + themes (both depend on news) ---
            with ThreadPoolExecutor(max_workers=self.config.get("max_workers", 1)) as pool:
                fut_stocks = pool.submit(self.intelligence.correlate_stocks_and_news, stocks, news)
                fut_themes = pool.submit(self.intelligence.detect_emerging_themes, papers, blogs, news)

                stocks = fut_stocks.result()
                emerging_themes = fut_themes.result()

            # Track trending topics across days (Feature 1)
            previous_state, papers, blogs, news = self.intelligence.track_trending(
                papers, blogs, news, previous_state
            )

        # --- Market trend analysis (must happen after correlation) ---
        market_trend = ""
        if self.intelligence.available and stocks:
            market_trend = self._analyze_market_trend(stocks)

        # --- Score papers (combines TF-IDF + semantic if available) ---
        top_papers = self.score_papers(papers)

        # --- Intelligence layer: assess top papers & synthesize ---
        if self.intelligence.available:
            top_papers = self.intelligence.assess_reproduction_feasibility(top_papers)

            # Ensure all display papers have summaries (batched)
            display_n = self.config.get("num_paper_picks", 5)
            top_papers = self._ensure_paper_summaries(top_papers[:display_n]) + top_papers[display_n:]

            # --- Generate author blurbs for all sections ---
            logger.info("=== Intelligence Layer: Generating Author Blurbs ===")
            with ThreadPoolExecutor(max_workers=self.config.get("max_workers", 1)) as pool:
                fut_news_blurbs = pool.submit(self.intelligence.generate_author_blurbs, news, "news")
                fut_blogs_blurbs = pool.submit(self.intelligence.generate_author_blurbs, blogs, "blogs")
                fut_top_papers_blurbs = pool.submit(self.intelligence.generate_author_blurbs, top_papers[:5], "papers")
                fut_recent_papers_blurbs = pool.submit(self.intelligence.generate_author_blurbs, papers[:5], "papers")

                # Lists are mutated in-place by generate_author_blurbs
                fut_news_blurbs.result()
                fut_blogs_blurbs.result()
                fut_top_papers_blurbs.result()
                fut_recent_papers_blurbs.result()

            synthesis = self.intelligence.synthesize_briefing(
                papers, blogs[:5], stocks, news[:5], top_papers[:3],
                emerging_themes=emerging_themes,
                previous_state=previous_state,
            )

            if self.feature_solo_founder_angle:
                try:
                    solo_angle = self.intelligence.generate_solo_startup_angle(
                        papers, blogs[:6], news[:6], top_papers[:3],
                        emerging_themes=emerging_themes,
                    )
                    if solo_angle:
                        synthesis["solo_startup"] = solo_angle
                except Exception as e:
                    logger.warning(f"Solo-startup angle generation failed: {e}")

            if self.feature_agent_cost_optimization:
                try:
                    cost_play = self.intelligence.generate_agent_cost_optimization(
                        papers, blogs[:6], news[:6], top_papers[:3],
                        emerging_themes=emerging_themes,
                    )
                    if cost_play:
                        synthesis["agent_cost_play"] = cost_play
                except Exception as e:
                    logger.warning(f"Agent cost-optimization generation failed: {e}")

            # Feature 3: Competitive Intelligence (entity tracking)
            tracked_entities = self.config.get("tracked_entities", [])
            entity_mentions = []
            if tracked_entities:
                logger.info("=== Intelligence Layer: Entity Tracking ===")
                entity_mentions = self.intelligence.detect_entity_mentions(
                    papers, blogs, news, tracked_entities
                )
                # Add to synthesis for rendering in Executive Summary
                synthesis["entity_mentions"] = entity_mentions

        now = datetime.now()
        weekly_deep_dive = ""
        weekly_items = []
        if self.feature_weekly_deep_dive:
            is_saturday = now.weekday() == 5
            weekly_items = previous_state.get("weekly_items", [])

            # Accumulate today's top items for the week
            today_str = now.strftime("%Y-%m-%d")
            for paper in top_papers[:3]:
                weekly_items.append({
                    "date": today_str,
                    "type": "paper",
                    "title": paper.get("title", ""),
                })
            for article in news[:3]:
                weekly_items.append({
                    "date": today_str,
                    "type": "news",
                    "title": article.get("title", ""),
                })

            # On Saturday, generate the deep dive and clear weekly_items
            if is_saturday and self.intelligence.available and weekly_items:
                logger.info("=== Intelligence Layer: Weekly Deep Dive (Saturday) ===")
                weekly_deep_dive = self.intelligence.generate_weekly_deep_dive(weekly_items)
                # Clear weekly items after generation
                weekly_items = []

        # --- Check if we have any data ---
        has_data = any([papers, blogs, stocks, news, happenings, alerts])
        if not has_data:
            logger.error("No data collected from any source")
            self.status["elapsed_seconds"] = round(time.time() - start_time, 1)
            self.save_status()
            return 2

        # --- Ensure output directory ---
        output_dir = self.config.get("output_dir", "briefings")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # --- Generate markdown briefing ---
        filename = self._format_filename(now)
        self._briefing_title = filename
        markdown_content = self.generate_markdown_briefing(
            papers, blogs, stocks, news, top_papers, synthesis,
            market_trend=market_trend,
            weekly_deep_dive=weekly_deep_dive,
            start_time=start_time,
            end_time=time.time(),
            happenings=happenings,
            alerts=alerts,
        )

        # --- Save markdown ---
        md_path = f"{output_dir}/{filename}.md"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.info(f"Saved markdown: {md_path}")
        except IOError as e:
            logger.warning(f"Failed to save markdown: {e}")

        # --- Generate PDF ---
        pdf_config = self.config.get("pdf", {})
        pdf_enabled = pdf_config.get("enabled", True)
        pdf_path = f"{output_dir}/{filename}.pdf" if pdf_enabled else None
        pdf_success = self.generate_pdf(markdown_content, pdf_path) if pdf_enabled else True

        # --- Generate EPUB (Reflowable for Kindle) ---
        epub_path = f"{output_dir}/{filename}.epub"
        epub_success = self.generate_epub(markdown_content, epub_path)

        if pdf_enabled and not pdf_success and not epub_success:
            logger.error("Failed to generate both PDF and EPUB")
            self.status["elapsed_seconds"] = round(time.time() - start_time, 1)
            self.save_status()
            return 2

        # --- Distribute to all channels (Kindle EPUB/PDF + HTML email) ---
        self.distribute_briefing(markdown_content, pdf_path, filename, epub_path=epub_path)

        # --- Save state for cross-day tracking ---
        # Save updated trending_topics and weekly_items from current run
        cached_happenings = self._happenings_cache
        cached_happenings_date = self._happenings_cache_date
        if not cached_happenings:
            cached_happenings = previous_state.get("cached_happenings", [])
            cached_happenings_date = previous_state.get("cached_happenings_date")
        self._save_state(
            top_papers, blogs, news, stocks, emerging_themes,
            trending_topics=previous_state.get("trending_topics", {}),
            weekly_items=weekly_items,  # Use updated weekly_items from this run
            happenings=happenings,
            cached_happenings=cached_happenings,
            cached_happenings_date=cached_happenings_date,
        )

        # --- Finalize ---
        elapsed = time.time() - start_time
        self.status["elapsed_seconds"] = round(elapsed, 1)
        self.save_status()

        logger.info(f"=== Briefing Complete in {elapsed:.1f}s ===")

        if self.errors:
            logger.warning(f"Completed with {len(self.errors)} errors")
            return 1
        else:
            logger.info("Completed successfully")
            return 0


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable expansion.
    Supports ${VAR} and ${VAR:-default} syntax.
    """
    pattern = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")

    def env_var_constructor(loader, node):
        value = loader.construct_scalar(node)
        match = pattern.match(value)
        if match:
            var_name, default = match.groups()
            return os.environ.get(var_name, default if default is not None else value)
        return value

    # Add constructor for any string that matches the pattern
    yaml.SafeLoader.add_implicit_resolver("!env", pattern, None)
    yaml.SafeLoader.add_constructor("!env", env_var_constructor)

    try:
        with open(config_path, "r") as f:
            # First read as raw string to expand environment variables
            content = f.read()
            
            # Helper to replace ${VAR} with os.environ.get(VAR)
            def replace_env_var(match):
                var_name = match.group(1)
                default = match.group(2)
                val = os.environ.get(var_name)
                if val is not None:
                    return val
                return default if default is not None else match.group(0)
            
            expanded_content = pattern.sub(replace_env_var, content)
            return yaml.safe_load(expanded_content)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(2)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        sys.exit(2)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate morning briefing")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate briefing but don't send email",
    )
    parser.add_argument(
        "--use-snapshots",
        type=str,
        metavar="DATE",
        default=None,
        help="Load raw data from snapshots/DATE/ instead of making live API calls",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(getattr(logging, args.log_level))

    # Load config
    config = load_config(args.config)

    # Validate config
    is_valid, messages = validate_config(config)
    if not is_valid:
        logger.error("Configuration is invalid. Fix errors above and retry.")
        return 2

    # Check environment
    check_environment(config, dry_run=args.dry_run)

    # Run briefing
    runner = BriefingRunner(
        config=config,
        dry_run=args.dry_run,
        use_snapshots=args.use_snapshots,
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
