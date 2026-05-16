#!/usr/bin/env python3
"""
Morning briefing runner v0.2 - Coordinator + Parallel Workers Architecture.

KEY CHANGES FROM V0.1:
- Coordinator pattern: coordinator READS findings and synthesizes (no lazy delegation)
- Workers are fully self-contained (fetch + enrich independently)
- All workers run in parallel
- Memory system for cross-day learning
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.workers.papers_worker import PapersWorker
from scripts.workers.blogs_worker import BlogsWorker
from scripts.workers.news_market_worker import NewsMarketWorker
from scripts.bedrock_client import BedrockClient
from scripts.gemini_client import GeminiCLIClient
from scripts.pdf_generator import PDFGenerator
from scripts.epub_generator import EPUBGenerator
from scripts.email_distributor import EmailDistributor
from scripts.config_validator import validate_config, check_environment, expand_env_vars

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STATE_FILENAME = ".atlas-state.json"
MEMORY_DIR = Path("briefing-memory")


class BriefingCoordinator:
    """Coordinator for v0.2 multi-agent briefing generation."""

    def __init__(self, config: Dict[str, Any], dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        
        gemini_config = config.get("gemini", {})
        if gemini_config.get("enabled", False):
            logger.info("Using Gemini CLI for intelligence")
            self.llm = GeminiCLIClient(gemini_config)
        else:
            logger.info("Using Amazon Bedrock for intelligence")
            self.llm = BedrockClient(config.get("bedrock", {}))

        self.memory_dir = MEMORY_DIR
        self.memory_dir.mkdir(exist_ok=True)

    def run(self) -> int:
        start_time = time.time()
        logger.info("=== Morning Briefing v0.2 - Coordinator + Parallel Workers ===")

        memory = self._load_memory()
        logger.info("=== Spawning parallel workers ===")
        findings = self._spawn_workers()

        failed_workers = [f for f in findings if f["status"] == "error"]
        if len(failed_workers) == len(findings):
            logger.error("All workers failed. Aborting.")
            return 2

        papers, blogs, news, stocks = self._extract_items(findings)
        logger.info("=== Coordinator Synthesis ===")
        synthesis = self._synthesize_findings(findings, papers, blogs, news, stocks, memory)

        logger.info("=== Generating briefing document ===")
        briefing_content = self._generate_briefing(synthesis, papers, blogs, news, stocks)

        output_filename = self._get_output_filename()
        md_path = f"{output_filename}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(briefing_content)

        pdf_path = self._generate_pdf(briefing_content, output_filename)
        epub_path = self._generate_epub(briefing_content, output_filename)

        if not self.dry_run:
            self._distribute(
                briefing_content,
                output_filename,
                str(pdf_path) if pdf_path else None,
                str(epub_path) if epub_path else None,
            )

        self._update_memory(synthesis, papers, blogs, news, stocks)
        self._save_state(papers, blogs, news, stocks=stocks, synthesis=synthesis)

        total_tokens = sum(
            f.get("metadata", {}).get("token_count", 0) for f in findings
        )
        logger.info(f"Total LLM tokens used: {total_tokens}")

        elapsed = time.time() - start_time
        logger.info(f"=== Briefing completed in {elapsed:.1f}s ===")
        return 0 if not failed_workers else 1

    def _spawn_workers(self) -> List[Dict[str, Any]]:
        workers = [
            PapersWorker(self.config, llm_client=self.llm),
            BlogsWorker(self.config, llm_client=self.llm),
            NewsMarketWorker(self.config, llm_client=self.llm),
        ]
        # Honor config.max_workers (default 1). Serial execution avoids stacking
        # concurrent gemini subprocess calls against the same per-key RPM cap;
        # set higher only if you have plenty of API quota headroom.
        max_workers = max(1, int(self.config.get("max_workers", 1)))
        if max_workers == 1:
            logger.info("Running workers serially (max_workers=1)")
        else:
            logger.info(f"Running workers with max_workers={max_workers}")
        findings = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker.execute): worker for worker in workers}
            for future in as_completed(futures):
                worker = futures[future]
                try:
                    finding = future.result()
                    findings.append(finding)
                except Exception as e:
                    logger.error(f"Worker {worker.worker_name} failed: {e}")
                    findings.append({"worker": worker.worker_name, "status": "error", "items": [], "metadata": {"token_count": 0}, "synthesis": ""})
        return findings

    def _extract_items(self, findings: List[Dict[str, Any]]) -> tuple:
        papers, blogs, news, stocks = [], [], [], []
        for f in findings:
            if f["worker"] == "papers_worker":
                papers = f.get("items", [])
            elif f["worker"] == "blogs_worker":
                blogs = f.get("items", [])
            elif f["worker"] == "news_market_worker":
                items = f.get("items", {})
                news = items.get("news", [])
                stocks = items.get("stocks", [])
        return papers, blogs, news, stocks

    def _synthesize_findings(self, findings, papers, blogs, news, stocks, memory) -> Dict[str, Any]:
        worker_syntheses = {f["worker"]: f["synthesis"] for f in findings}
        emerging_themes = self._detect_emerging_themes(papers, blogs, news)
        executive_summary = self._generate_executive_summary(worker_syntheses, emerging_themes, stocks)
        market_trend = self._analyze_market_trend(stocks, news)
        return {"executive_summary": executive_summary, "emerging_themes": emerging_themes, "market_trend": market_trend}

    def _detect_emerging_themes(self, papers, blogs, news) -> List[str]:
        if not self.llm.available:
            return []
        top_papers = sorted(papers, key=lambda p: p.get("score", 0), reverse=True)[:3]
        top_blogs = blogs[:3]
        top_news = news[:3]
        if not (top_papers or top_blogs or top_news):
            return []

        lines = ["Detect 2-3 emerging themes across today's papers, blogs, and news.",
                 "Return ONLY a comma-separated list of short theme phrases."]
        for p in top_papers:
            lines.append(f"- [paper] {p.get('title', '')}")
        for b in top_blogs:
            lines.append(f"- [blog] {b.get('title', '')}")
        for n in top_news:
            lines.append(f"- [news] {n.get('title', '')}")
        res = self.llm.invoke("\n".join(lines), tier="light")
        if not res:
            return []
        return [t.strip() for t in res.split(",") if t.strip()]

    def _generate_executive_summary(self, syntheses, themes, stocks) -> str:
        if not self.llm.available:
            return "LLM offline"
        prompt = f"Summarize today's findings:\nThemes: {', '.join(themes)}\n"
        for worker, summary in syntheses.items():
            prompt += f"{worker}: {summary}\n"
        return self.llm.invoke(prompt, tier="medium") or "Synthesis failed"

    def _analyze_market_trend(self, stocks, news) -> str:
        if not stocks or not self.llm.available:
            return ""
        prompt = "Analyze market trend from these stocks:\n"
        for s in stocks:
            prompt += f"- {s.get('symbol')}: {s.get('percent_change', 0.0):.2f}%\n"
        return self.llm.invoke(prompt, tier="light") or ""

    def _generate_briefing(self, synthesis, papers, blogs, news, stocks) -> str:
        content = f"# Morning Briefing - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"
        content += f"## Executive Summary\n\n{synthesis['executive_summary']}\n\n"
        if stocks:
            content += f"## Markets\n\n{synthesis['market_trend']}\n\n"
            for s in stocks:
                content += f"- **{s.get('symbol')}**: {s.get('percent_change', 0.0):.2f}%\n"
        if news:
            content += "## News\n\n"
            for n in news[:5]:
                summary = n.get("brief_summary", "") or n.get("description", "") or n.get("snippet", "")
                url = n.get("url", "")
                content += f"### {n.get('title', '')}\n\n{summary}\n\n[Link]({url})\n\n"
        if papers:
            content += "## Research\n\n"
            for p in sorted(papers, key=lambda p: p.get("score", 0), reverse=True)[:5]:
                summary = p.get("brief_summary", "") or p.get("summary", "")[:400]
                link = p.get("arxiv_url") or p.get("pdf_link") or p.get("id", "")
                content += f"### {p.get('title', '')}\n\n{summary}\n\n[ArXiv]({link})\n\n"
        if blogs:
            content += "## Blogs\n\n"
            for b in blogs[:5]:
                summary = b.get("brief_summary", "") or b.get("summary", "")[:400]
                link = b.get("link", "")
                source = b.get("source", "")
                source_tag = f" *({source})*" if source else ""
                content += f"### {b.get('title', '')}{source_tag}\n\n{summary}\n\n[Read more]({link})\n\n"
        return content

    def _generate_pdf(self, content, filename):
        pdf_cfg = self.config.get("pdf", {})
        if not pdf_cfg.get("enabled", True):
            logger.info("PDF generation disabled via config")
            return None
        path = f"{filename}.pdf"
        logger.info(f"Generating PDF: {path}")
        generator = PDFGenerator(
            page_format=self.config.get("output_format", "kindle"),
            font_size=pdf_cfg.get("font_size", 10),
            line_spacing=pdf_cfg.get("line_spacing", 1.5),
            include_toc=pdf_cfg.get("include_toc", True),
        )
        generator.generate_pdf(content, path)
        return Path(path)

    def _generate_epub(self, content, filename):
        path = f"{filename}.epub"
        logger.info(f"Generating EPUB: {path}")
        EPUBGenerator(title=filename).generate_epub(content, path)
        return Path(path)

    def _distribute(self, content, filename, pdf_path, epub_path):
        gmail_user = os.environ.get("GMAIL_USER", "")
        gmail_password = os.environ.get("GMAIL_APP_PASSWORD", "")
        if not gmail_user or not gmail_password:
            logger.warning(
                "Skipping distribution: GMAIL_USER or GMAIL_APP_PASSWORD not set"
            )
            return
        logger.info("Distributing briefing")
        distributor = EmailDistributor(
            sender_email=gmail_user,
            sender_password=gmail_password,
        )
        distributor.distribute(self.config, content, pdf_path, epub_path, filename)

    def _get_output_filename(self) -> str:
        now = datetime.now(timezone.utc)
        return f"Atlas-Briefing-{now.year}.{now.month:02d}.{now.day:02d}"

    def _load_memory(self) -> Dict[str, Any]:
        """Load previous run state. Falls back to empty dict on first run."""
        state_path = Path(STATE_FILENAME)
        if not state_path.exists():
            return {}
        try:
            with state_path.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load previous state from {STATE_FILENAME}: {e}")
            return {}

    def _update_memory(self, synthesis, papers, blogs, news, stocks):
        """Hook for future cross-day memory updates; intentionally minimal."""
        return

    def _save_state(self, papers, blogs, news, stocks=None, synthesis=None):
        """Persist run state for cross-day trend tracking.

        Preserves the keys that BriefingIntelligence.track_trending and
        synthesize_briefing read on subsequent days: top_paper_titles,
        top_blog_titles, top_news_titles, emerging_themes, stock_closes,
        and trending_topics (carried over verbatim from the previous run
        because trending bookkeeping happens inside the intelligence
        layer, which v0.2 doesn't yet invoke).
        """
        previous = self._load_memory()
        stocks = stocks or []
        synthesis = synthesis or {}

        stock_closes = {}
        for s in stocks:
            symbol = s.get("symbol")
            price = s.get("current_price")
            if symbol and price is not None:
                stock_closes[symbol] = price

        state = {
            "date": datetime.now(timezone.utc).isoformat(),
            "top_paper_titles": [p.get("title", "") for p in papers[:10]],
            "top_blog_titles": [b.get("title", "") for b in blogs[:10]],
            "top_news_titles": [n.get("title", "") for n in news[:10]],
            "emerging_themes": synthesis.get("emerging_themes", []),
            "stock_closes": stock_closes,
            "trending_topics": previous.get("trending_topics", {}),
        }
        with open(STATE_FILENAME, "w") as f:
            json.dump(state, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    # Expand ${VAR} / ${VAR:-default} placeholders BEFORE validation so
    # downstream code (email distribution, file paths, LLM prompts) never
    # sees literal bash-style strings.
    config = expand_env_vars(config)

    is_valid, _ = validate_config(config)
    if not is_valid:
        logger.error("Configuration is invalid. Fix errors above and retry.")
        sys.exit(2)

    check_environment(config, dry_run=args.dry_run)
    sys.exit(BriefingCoordinator(config, args.dry_run).run())

if __name__ == "__main__": main()
