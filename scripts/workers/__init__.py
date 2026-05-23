"""Worker modules for v0.2 multi-agent architecture.

Imports are intentionally lazy: each concrete worker has its own scanner
dependency (feedparser, requests, sklearn), and importing them all eagerly
forces every dependency to load even when callers only need one worker.
"""

from .base_worker import BaseWorker

__all__ = ["BaseWorker", "PapersWorker", "BlogsWorker", "NewsMarketWorker"]


def __getattr__(name):
    if name == "PapersWorker":
        from .papers_worker import PapersWorker
        return PapersWorker
    if name == "BlogsWorker":
        from .blogs_worker import BlogsWorker
        return BlogsWorker
    if name == "NewsMarketWorker":
        from .news_market_worker import NewsMarketWorker
        return NewsMarketWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
