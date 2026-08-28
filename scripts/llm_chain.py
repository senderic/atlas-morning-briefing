#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Single source of truth for building the LLM backend chain.

Both the briefing runner and the quality checker need the same chain, and when
each built its own the two drifted: the checker kept an `opencode`-first order
after the runner had moved to cost-ordered `openrouter`-first, so every daily
quality run billed the paid backstop while the briefing itself ran free.

Order is cost. The chain is tried front to back, so free backends come first
and any paid one goes last, reached only when everything ahead of it failed.
"""

import logging
from typing import Any, Dict, List, Optional

from scripts.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

DEFAULT_BACKEND_PRIORITY = ["openrouter", "gemini", "opencode"]


def build_llm_chain(
    config: Dict[str, Any],
    preflight_models: Optional[Dict[str, Any]] = None,
) -> List[BaseLLMClient]:
    """Construct the enabled backends in cost order.

    Args:
        config: Full pipeline config.
        preflight_models: Optional ``.model-availability.json`` payload, keyed
            by provider, used to pin a per-tier model.

    Returns:
        Backend clients, cheapest first. Empty when none are enabled.
    """
    preflight_models = preflight_models or {}
    gemini_config = config.get("gemini", config.get("bedrock", {})) or {}
    openrouter_config = config.get("openrouter", {}) or {}
    opencode_config = config.get("opencode", {}) or {}

    def _openrouter():
        from scripts.openrouter_client import OpenRouterClient
        return OpenRouterClient(
            openrouter_config, preflight_models=preflight_models.get("openrouter", {})
        )

    def _gemini():
        from scripts.gemini_client import GeminiCLIClient
        return GeminiCLIClient(gemini_config)

    def _opencode():
        from scripts.opencode_client import OpencodeClient
        return OpencodeClient(
            opencode_config, preflight_models=preflight_models.get("opencode", {})
        )

    builders = {"openrouter": _openrouter, "gemini": _gemini, "opencode": _opencode}
    enabled = {
        "openrouter": bool(openrouter_config.get("enabled")),
        "gemini": bool(gemini_config.get("enabled")),
        "opencode": bool(opencode_config.get("enabled")),
    }

    priority = config.get("llm", {}).get("backend_priority", DEFAULT_BACKEND_PRIORITY)
    unknown = [name for name in priority if name not in builders]
    if unknown:
        logger.warning("Ignoring unknown llm.backend_priority entries: %s", unknown)
    # A backend omitted from the priority list still runs, appended last, so a
    # typo in config cannot silently drop a configured backend.
    ordered = [n for n in priority if n in builders]
    ordered += [n for n in builders if n not in ordered]

    chain: List[BaseLLMClient] = []
    for name in ordered:
        if enabled[name]:
            chain.append(builders[name]())
    if chain:
        logger.info(
            "LLM backend chain (first tried to last): %s",
            " -> ".join(type(c).__name__ for c in chain),
        )
    return chain


def chain_timeout(config: Dict[str, Any]) -> float:
    """Per-backend window for CompositeClient, from config."""
    return config.get("llm", {}).get(
        "fallback_timeout_seconds",
        (config.get("composite", {}) or {}).get("timeout_seconds", 240),
    )
