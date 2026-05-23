#!/usr/bin/env python3
"""
Base worker class for v0.2 multi-agent architecture.

Each worker is self-contained and reports findings in a structured format.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseWorker(ABC):
    """Base class for all workers in the multi-agent architecture."""

    def __init__(
        self,
        config: Dict[str, Any],
        worker_name: str,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize worker.

        Args:
            config: Full configuration dictionary
            worker_name: Name of this worker (for logging/reporting)
            llm_client: Optional pre-built LLM client (BedrockClient or
                GeminiCLIClient). When provided, workers share a single
                client so the per-run call budget and Gemini key-rotation
                state are honored across all workers. When None, the
                worker constructs its own client (useful for standalone
                invocation).
        """
        self.config = config
        self.worker_name = worker_name
        self.llm_client = llm_client
        self.start_time = None
        self.end_time = None

    def _get_llm_client(self) -> Any:
        """Return the shared LLM client, or build one on demand."""
        if self.llm_client is not None:
            return self.llm_client

        # Lazy fallback for standalone use. Imported here to avoid making
        # base_worker depend on concrete client modules at import time.
        gemini_config = self.config.get("gemini", {})
        if gemini_config.get("enabled", False):
            from scripts.gemini_client import GeminiCLIClient
            return GeminiCLIClient(gemini_config)
        from scripts.bedrock_client import BedrockClient
        return BedrockClient(self.config.get("bedrock", {}))

    @staticmethod
    def _count_client_tokens(llm: Any) -> int:
        """Return the total in+out tokens recorded by the LLM client, if any.

        GeminiCLIClient exposes a usage_stats dict per tier; BedrockClient
        doesn't track tokens at all. Returns 0 when unavailable so callers
        don't need to special-case backends.
        """
        stats = getattr(llm, "usage_stats", None)
        if not isinstance(stats, dict):
            return 0
        total = 0
        for tier_stats in stats.values():
            if isinstance(tier_stats, dict):
                total += tier_stats.get("in_tokens", 0) + tier_stats.get("out_tokens", 0)
        return total

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the worker's task.

        Returns:
            Dictionary with findings in this format:
            {
                "worker": str,           # Worker name
                "status": str,           # "success" or "error"
                "items": List[Dict],     # Found items (papers/blogs/news/stocks)
                "metadata": {
                    "processing_time": float,  # Seconds
                    "token_count": int,        # LLM tokens used
                    "items_found": int,        # Raw items before filtering
                    "items_kept": int          # Items after enrichment/filtering
                },
                "synthesis": str,        # Worker's own summary of findings
                "error": str             # Error message if status=="error"
            }
        """
        pass

    def _start_timing(self):
        """Start timing the worker execution."""
        self.start_time = time.time()

    def _end_timing(self) -> float:
        """End timing and return elapsed seconds."""
        self.end_time = time.time()
        return self.end_time - self.start_time if self.start_time else 0.0

    def _create_finding(
        self,
        status: str,
        items: list,
        synthesis: str = "",
        token_count: int = 0,
        items_found: int = 0,
        error: str = ""
    ) -> Dict[str, Any]:
        """
        Create a standardized finding report.

        Args:
            status: "success" or "error"
            items: List of items (papers/blogs/news/stocks)
            synthesis: Worker's summary of findings
            token_count: LLM tokens used
            items_found: Raw items before filtering
            error: Error message if status=="error"

        Returns:
            Standardized finding dictionary
        """
        processing_time = self._end_timing()
        return {
            "worker": self.worker_name,
            "status": status,
            "items": items,
            "metadata": {
                "processing_time": processing_time,
                "token_count": token_count,
                "items_found": items_found or len(items),
                "items_kept": len(items)
            },
            "synthesis": synthesis,
            "error": error
        }
