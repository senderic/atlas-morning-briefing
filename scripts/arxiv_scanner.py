#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
ArXiv paper scanner.

Provides two scanners:
- ArxivScanner: legacy XML/Atom client using defusedxml + parallel topic
  scanning (the default; what tests import and what briefing_runner uses).
- DeepXivScanner: optional DeepXiv SDK wrapper with semantic search,
  TLDRs, and GitHub URL extraction. Opt-in via `create_scanner()` when
  DeepXiv is installed AND a DEEPXIV_TOKEN is available.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

try:
    from defusedxml.ElementTree import fromstring as _xml_fromstring
except ImportError:
    raise ImportError(
        "defusedxml is required for safe XML parsing. "
        "Install it with: pip install defusedxml"
    )


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# DeepXiv is an optional upgrade — if not installed, ArxivScanner is used.
# 2026-07-03: DeepXiv index is stale (returns 0 papers on all queries since ~mid-June 2026).
# Force-disable DeepXiv and use legacy arxiv.org API directly — reliable in cron path.
# Re-enable when DeepXiv confirms index refresh.
FORCE_DISABLE_DEEPXIV = True

try:
    if FORCE_DISABLE_DEEPXIV:
        raise ImportError("DeepXiv force-disabled (stale index, see 2026-07-03 fix)")
    from deepxiv_sdk import Reader as DeepXivReader  # type: ignore
    HAS_DEEPXIV = True
except ImportError:
    HAS_DEEPXIV = False
    DeepXivReader = None  # type: ignore


# Known envelope keys the DeepXiv SDK has used across versions. Ordered by
# observed frequency / specificity. The "unified retrieve endpoint"
# (introduced May 2026) is what triggered this: the legacy `results` key
# stopped showing up and the daily briefing started rendering with zero
# Top Papers. The fallback walker (in _extract_papers_list below)
# recovered automatically by finding the list-of-dicts value under
# `result` (singular) and logged a breadcrumb telling us to add the key
# here — which is why `result` is now in this list. If the SDK ever
# changes shape again, the same fallback path will recover the data and
# log the next key to add.
_KNOWN_DEEPXIV_LIST_KEYS = (
    "results", "result", "data", "docs", "items", "hits", "papers",
    "top_k", "retrieve",
)


def _extract_papers_list(response: Any) -> List[Dict[str, Any]]:
    """Pull the list of paper records out of a DeepXiv search response.

    The SDK has gone through several response shapes:
      v1: {"total": N, "results": [...], "took": ms}
      v2 (unified retrieve endpoint, May 2026): different envelope key
      Direct list (some endpoints): [...]

    Strategy: list → use as-is; dict → try known keys, then fall back to
    any list-of-dicts value we can find. Logs which path was taken so
    that quiet shape changes leave breadcrumbs in the briefing log.
    """
    if isinstance(response, list):
        return response
    if not isinstance(response, dict):
        logger.warning(
            "Unexpected DeepXiv response type: %s (expected list or dict)",
            type(response).__name__,
        )
        return []

    for key in _KNOWN_DEEPXIV_LIST_KEYS:
        value = response.get(key)
        if isinstance(value, list):
            if key != "results":
                logger.info(
                    "DeepXiv response used key '%s' (legacy key 'results' "
                    "no longer present)", key,
                )
            return value

    # Unknown shape: walk the dict looking for a list-of-dicts value
    for key, value in response.items():
        if (isinstance(value, list) and value
                and isinstance(value[0], dict)):
            logger.warning(
                "DeepXiv response shape unrecognized; falling back to key "
                "'%s' (a list of %d dicts). Add '%s' to "
                "_KNOWN_DEEPXIV_LIST_KEYS to silence this warning.",
                key, len(value), key,
            )
            return value

    logger.warning(
        "DeepXiv response had no list-of-dicts value. Top-level keys: %s",
        list(response.keys()),
    )
    return []


def _load_deepxiv_token() -> Optional[str]:
    """
    Read an existing DeepXiv token from environment or ~/.env (where the
    deepxiv CLI saves auto-registered tokens). Returns None if none found —
    in that case Reader() will auto-register a free 1,000-req/day token on
    first use and persist it to ~/.env.
    """
    token = os.environ.get("DEEPXIV_TOKEN")
    if token:
        return token
    env_path = Path.home() / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text().splitlines():
                if line.startswith("DEEPXIV_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return None


class ArxivScanner:
    """Scans ArXiv for papers matching configured topics."""

    ARXIV_API_URL = "https://export.arxiv.org/api/query"

    def __init__(self, topics: List[str], days_back: int = 7, max_results: int = 20):
        """
        Initialize ArxivScanner.

        Args:
            topics: List of topics to search for
            days_back: Number of days to look back
            max_results: Maximum number of results per topic
        """
        self.topics = topics
        self.days_back = days_back
        self.max_results = max_results

    def search_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Search ArXiv for papers on a specific topic.

        Args:
            topic: Topic to search for

        Returns:
            List of paper dictionaries
        """
        try:
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.days_back)

            # Build query
            query = f"all:{topic}"
            params = {
                "search_query": query,
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            logger.info(f"Searching ArXiv for topic: {topic}")
            response = requests.get(self.ARXIV_API_URL, params=params, timeout=30)
            response.raise_for_status()

            papers = self._parse_arxiv_response(response.text, start_date)
            logger.info(f"Found {len(papers)} papers for topic: {topic}")
            return papers

        except requests.RequestException as e:
            logger.error(f"Failed to search ArXiv for topic '{topic}': {e}")
            return []

    def _parse_arxiv_response(
        self, xml_content: str, start_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Parse ArXiv API XML response.

        Args:
            xml_content: XML response from ArXiv API
            start_date: Filter papers published after this date

        Returns:
            List of paper dictionaries
        """
        papers = []
        try:
            root = _xml_fromstring(xml_content)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", namespace):
                # Extract paper details
                paper_id = entry.find("atom:id", namespace)
                title = entry.find("atom:title", namespace)
                summary = entry.find("atom:summary", namespace)
                published = entry.find("atom:published", namespace)
                updated = entry.find("atom:updated", namespace)

                # Extract authors
                authors = []
                for author in entry.findall("atom:author", namespace):
                    name = author.find("atom:name", namespace)
                    if name is not None and name.text:
                        authors.append(name.text.strip())

                # Extract categories
                categories = []
                for category in entry.findall("atom:category", namespace):
                    term = category.get("term")
                    if term:
                        categories.append(term)

                # Extract links
                pdf_link = None
                for link in entry.findall("atom:link", namespace):
                    if link.get("title") == "pdf":
                        pdf_link = link.get("href")
                        break

                if not pdf_link:
                    # Fallback to constructing PDF link from ID
                    paper_url = paper_id.text if paper_id is not None else ""
                    if paper_url:
                        pdf_link = paper_url.replace("/abs/", "/pdf/") + ".pdf"

                # Parse published date
                if published is not None and published.text:
                    pub_date = datetime.fromisoformat(
                        published.text.replace("Z", "+00:00")
                    )
                    # Filter by date range
                    if pub_date < start_date:
                        continue
                else:
                    continue

                paper = {
                    "id": paper_id.text.strip() if paper_id is not None else "",
                    "title": title.text.strip() if title is not None else "",
                    "summary": summary.text.strip() if summary is not None else "",
                    "authors": authors,
                    "published": published.text if published is not None else "",
                    "updated": updated.text if updated is not None else "",
                    "categories": categories,
                    "pdf_link": pdf_link,
                    "arxiv_url": paper_id.text if paper_id is not None else "",
                }

                papers.append(paper)

        except Exception as e:
            logger.error(f"Failed to parse ArXiv XML response: {e}")

        return papers

    def scan_all_topics(self) -> List[Dict[str, Any]]:
        """
        Scan all configured topics in parallel.

        Returns:
            List of all papers found across topics
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_papers = []
        seen_ids = set()

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(self.search_topic, topic): topic
                for topic in self.topics
            }
            for future in as_completed(futures):
                try:
                    papers = future.result()
                    for paper in papers:
                        paper_id = paper.get("id", "")
                        if paper_id and paper_id not in seen_ids:
                            all_papers.append(paper)
                            seen_ids.add(paper_id)
                except Exception as e:
                    topic = futures[future]
                    logger.warning(f"ArXiv scan failed for topic '{topic}': {e}")

        logger.info(f"Total unique papers found: {len(all_papers)}")
        return all_papers


# ── Optional DeepXiv SDK scanner (upstream v0.2 addition) ──


class DeepXivScanner:
    """
    DeepXiv SDK scanner with semantic search and progressive reading.

    Advantages over the legacy ArXiv API:
    - Semantic / hybrid search vs keyword-only
    - TLDR briefs without loading the full paper
    - Citation counts and GitHub URLs
    - 200M+ papers indexed, T+1 daily sync

    Requires `deepxiv_sdk` installed AND a DEEPXIV_TOKEN.
    Falls back to ArxivScanner if either is missing — see create_scanner().
    """

    def __init__(self, topics: List[str], days_back: int = 7, max_results: int = 20):
        if not HAS_DEEPXIV:
            raise RuntimeError(
                "DeepXiv SDK not installed. "
                "Use create_scanner() or fall back to ArxivScanner."
            )
        self.topics = topics
        self.days_back = days_back
        self.max_results = max_results
        # If a token is already cached we pass it explicitly; otherwise let
        # Reader() auto-register a free anonymous token (1,000 req/day) on
        # first use and persist it to ~/.env.
        token = _load_deepxiv_token()
        self.reader = DeepXivReader(token=token) if token else DeepXivReader()
        if not token:
            logger.info(
                "No DEEPXIV_TOKEN found; DeepXiv SDK will auto-register a "
                "free anonymous token (1,000 req/day) on first request."
            )

    def search_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Search DeepXiv for papers on a topic."""
        try:
            logger.info(f"DeepXiv search: {topic}")
            start_date = datetime.now(timezone.utc) - timedelta(days=self.days_back)
            date_from = start_date.strftime("%Y-%m-%d")

            response = self.reader.search(
                topic,
                size=self.max_results,
                search_mode="hybrid",
                date_from=date_from,
            )

            raw_papers = _extract_papers_list(response)

            papers = []
            for r in raw_papers:
                paper = self._normalize_result(r)
                if paper:
                    papers.append(paper)

            logger.info(f"Found {len(papers)} papers for: {topic}")
            return papers

        except Exception as e:
            logger.error(f"DeepXiv search failed for '{topic}': {e}")
            return []

    def _normalize_result(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a DeepXiv search result to our standard paper format."""
        try:
            arxiv_id = data.get("arxiv_id", data.get("id", ""))
            if isinstance(arxiv_id, str) and "arxiv.org" in arxiv_id:
                arxiv_id = arxiv_id.split("/")[-1]

            title = data.get("title", "").strip()
            summary = data.get("abstract", data.get("summary", "")).strip()

            raw_authors = data.get("authors", [])
            if isinstance(raw_authors, list) and raw_authors:
                authors = [
                    a.get("name", str(a)) if isinstance(a, dict) else str(a)
                    for a in raw_authors
                ]
            elif isinstance(raw_authors, str) and "," in raw_authors:
                authors = [a.strip() for a in raw_authors.split(",")]
            else:
                an = data.get("author_names", "")
                authors = [a.strip() for a in an.split(",")] if "," in an else []

            published = data.get(
                "publish_at",
                data.get("published", data.get("created_at", "")),
            )
            categories = data.get("categories", [])
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split(",") if c.strip()]

            citations = data.get("citation", data.get("citations", 0)) or 0
            score = data.get("score", 0)

            return {
                "id": f"http://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "arxiv_id": str(arxiv_id),
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": str(published) if published else "",
                "updated": str(
                    data.get("updated_at", data.get("modified_at", published) or "")
                ),
                "categories": categories,
                "pdf_link": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "citations": citations,
                "deepxiv_score": score,
                "source": "deepxiv",
            }
        except Exception as e:
            logger.warning(f"Failed to normalize DeepXiv result: {e}")
            return None

    def enrich_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """Get a brief summary for a paper (saves tokens vs full read)."""
        try:
            clean_id = arxiv_id.split("/")[-1] if "/" in arxiv_id else arxiv_id
            brief = self.reader.brief(clean_id)
            if isinstance(brief, dict):
                return brief
            if isinstance(brief, str):
                return {"brief": brief}
            if hasattr(brief, "__dict__"):
                return brief.__dict__
            return {}
        except Exception as e:
            logger.debug(f"Brief failed for {arxiv_id}: {e}")
            return {}

    def scan_all_topics(self) -> List[Dict[str, Any]]:
        """Scan all configured topics, dedupe by arxiv_id, enrich top 10."""
        all_papers: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for topic in self.topics:
            for paper in self.search_topic(topic):
                pid = paper.get("arxiv_id", paper.get("id", ""))
                if pid and pid not in seen_ids:
                    all_papers.append(paper)
                    seen_ids.add(pid)

        logger.info(f"Total unique papers: {len(all_papers)}")

        # 2026-06-27: DeepXiv index appears to be stale (no papers indexed past ~early June 2026).
        # If we got nothing, fall back to direct arxiv.org API.
        if len(all_papers) == 0:
            logger.warning("DeepXiv returned 0 papers across all topics — falling back to legacy arxiv.org API")
            try:
                legacy = ArxivScanner(
                    topics=self.topics, days_back=self.days_back, max_results=self.max_results
                )
                fallback_papers = legacy.scan_all_topics()
                logger.info(f"Legacy ArXiv fallback returned {len(fallback_papers)} papers")
                return fallback_papers
            except Exception as e:
                logger.error(f"Legacy ArXiv fallback failed: {e}")
                return []

        # Enrich top 10 papers with briefs (saves DeepXiv API budget)
        top_papers = sorted(
            all_papers, key=lambda p: p.get("deepxiv_score", 0), reverse=True
        )[:10]
        for paper in top_papers:
            aid = paper.get("arxiv_id", "")
            if not aid:
                continue
            brief_data = self.enrich_paper(aid)
            if not brief_data:
                continue
            if "tldr" in brief_data:
                paper["tldr"] = brief_data["tldr"]
            if brief_data.get("github_url"):
                paper["github_url"] = brief_data["github_url"]
            if "keywords" in brief_data:
                paper["keywords"] = brief_data["keywords"]

        return all_papers


def create_scanner(
    topics: List[str], days_back: int = 7, max_results: int = 20
):
    """
    Return the best available scanner.

    Picks DeepXivScanner whenever the SDK is installed — the SDK
    auto-registers a free anonymous token on first use, so no explicit
    DEEPXIV_TOKEN is required. Falls back to the parallel defusedxml
    ArxivScanner when the SDK is not installed.
    """
    if HAS_DEEPXIV:
        logger.info("Using DeepXiv SDK for paper search")
        return DeepXivScanner(
            topics=topics, days_back=days_back, max_results=max_results
        )
    logger.info("Using legacy ArXiv API scanner")
    return ArxivScanner(
        topics=topics, days_back=days_back, max_results=max_results
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        SystemExit: If config file cannot be loaded
    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(2)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        sys.exit(2)


def main() -> int:
    """
    Main entry point for arxiv_scanner.

    Returns:
        Exit code (0 for success, 1 for partial failure, 2 for total failure)
    """
    parser = argparse.ArgumentParser(description="Scan ArXiv for papers on topics")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="papers.json",
        help="Output JSON file path",
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

    # Extract settings
    topics = config.get("arxiv_topics", [])
    days_back = config.get("arxiv_days_back", 7)
    max_papers = config.get("max_papers", 20)

    if not topics:
        logger.error("No arxiv_topics configured")
        return 2

    # Scan papers
    scanner = ArxivScanner(topics=topics, days_back=days_back, max_results=max_papers)
    papers = scanner.scan_all_topics()

    if not papers:
        logger.warning("No papers found")
        return 1

    # Save results
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(papers, f, indent=2)
        logger.info(f"Saved {len(papers)} papers to {args.output}")
        return 0
    except IOError as e:
        logger.error(f"Failed to write output file: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
