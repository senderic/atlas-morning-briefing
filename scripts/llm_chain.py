#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Single source of truth for the LLM fallback chain.

The chain is a list of MODELS, not of backends. A backend is only the transport
that can reach a given model, inferred from the model's routing prefix, so a
tier's order is free to interleave them:

    heavy:
      - opencode/muse-spark-1.3-contributor-free          # free, via the CLI
      - openrouter/nvidia/nemotron-3-ultra-550b-a55b:free # free, via HTTP
      - opencode-go/deepseek-v4-pro                       # paid, last

This replaces the older backend-ordered scheme (`llm.backend_priority` plus a
per-backend `models:`/`fallback_models:` roster). That shape made a model's rung
a consequence of who hosted it: the loop visited each backend once, so a free
model on the paid backend's transport could only ever be reached after every
model on the free backend's transport had failed. `opencode/muse-spark-*` was
configured as a free heavy primary and served zero calls for exactly that
reason.

Order is still cost — free rungs first, paid ones last — but that is now a
property of how the chain is written, not of which backend a model lives on.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from scripts.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

TIERS = ("heavy", "medium", "light")

# Routing prefix -> backend name. Longest prefix wins, so `opencode-go/` is
# matched before `opencode/`. The prefix is part of the slug the backend
# receives (`opencode run -m opencode-go/deepseek-v4-pro`); only OpenRouter
# strips its own, in OpenRouterClient._to_api_model.
BACKEND_PREFIXES = {
    "openrouter/": "openrouter",
    "opencode-go/": "opencode",
    "opencode/": "opencode",
    "gemini/": "gemini",
}

DEFAULT_CHAINS: Dict[str, List[str]] = {
    "heavy": [
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/dots-studio/dots-3-note-preview:free",
    ],
    "medium": [
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    ],
    "light": [
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    ],
}


@dataclass(frozen=True)
class Rung:
    """One model in a tier's fallback chain, plus the backend that reaches it."""

    model: str
    backend: str

    def __str__(self) -> str:  # pragma: no cover - logging sugar
        return f"{self.model} (via {self.backend})"


def resolve_backend(model: str) -> Optional[str]:
    """Map a model slug to the backend that can serve it.

    Returns None for a slug with no known routing prefix — the caller reports
    it rather than guessing, because guessing wrong on a paid prefix spends
    money.
    """
    for prefix in sorted(BACKEND_PREFIXES, key=len, reverse=True):
        if model.startswith(prefix):
            return BACKEND_PREFIXES[prefix]
    return None


def build_model_chains(config: Dict[str, Any]) -> Dict[str, List[Rung]]:
    """Read `llm.chains` into per-tier rungs, dropping unroutable entries."""
    configured = (config.get("llm", {}) or {}).get("chains") or {}
    chains: Dict[str, List[Rung]] = {}

    for tier in TIERS:
        models = configured.get(tier) or DEFAULT_CHAINS[tier]
        rungs: List[Rung] = []
        seen = set()
        for model in models:
            if model in seen:
                logger.warning(
                    "llm.chains.%s lists %s twice; keeping the first rung", tier, model
                )
                continue
            backend = resolve_backend(model)
            if backend is None:
                logger.warning(
                    "llm.chains.%s: no backend prefix on %r (expected one of %s); "
                    "skipping this rung",
                    tier, model, ", ".join(sorted(BACKEND_PREFIXES)),
                )
                continue
            seen.add(model)
            rungs.append(Rung(model=model, backend=backend))
        if not rungs:
            logger.warning("llm.chains.%s resolved to no usable rungs", tier)
        chains[tier] = rungs

    return chains


def build_clients(
    config: Dict[str, Any],
    preflight_models: Optional[Dict[str, Any]] = None,
) -> Dict[str, BaseLLMClient]:
    """Instantiate every enabled backend, keyed by backend name.

    Backends are transports here — they carry credentials, timeouts, call
    budgets and pricing, but no longer decide which model serves a tier.
    """
    preflight_models = preflight_models or {}
    gemini_config = config.get("gemini", config.get("bedrock", {})) or {}
    openrouter_config = config.get("openrouter", {}) or {}
    opencode_config = config.get("opencode", {}) or {}

    def _openrouter():
        from scripts.openrouter_client import OpenRouterClient
        return OpenRouterClient(openrouter_config)

    def _gemini():
        from scripts.gemini_client import GeminiCLIClient
        return GeminiCLIClient(gemini_config)

    def _opencode():
        from scripts.opencode_client import OpencodeClient
        return OpencodeClient(opencode_config)

    builders = {"openrouter": _openrouter, "gemini": _gemini, "opencode": _opencode}
    enabled = {
        "openrouter": bool(openrouter_config.get("enabled")),
        "gemini": bool(gemini_config.get("enabled")),
        "opencode": bool(opencode_config.get("enabled")),
    }

    clients: Dict[str, BaseLLMClient] = {}
    for name, builder in builders.items():
        if enabled[name]:
            clients[name] = builder()
    return clients


def preflight_pins(preflight_models: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Per-tier model that preflight found healthy, if any.

    The runner starts a tier at its pinned rung instead of the top, so a model
    that was already unreachable at 05:45 does not cost every call a timeout.
    """
    pins: Dict[str, str] = {}
    for tier, entry in ((preflight_models or {}).get("chains") or {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("available") and entry.get("model"):
            pins[tier] = entry["model"]
    return pins


def apply_pins(
    chains: Dict[str, List[Rung]], pins: Dict[str, str]
) -> Dict[str, List[Rung]]:
    """Rotate each tier's chain to start at its pinned model.

    Rungs above the pin are moved to the back rather than dropped: preflight is
    a 05:45 snapshot, and a model that was briefly 429 at probe time is often
    fine by the time the run needs it.
    """
    pinned: Dict[str, List[Rung]] = {}
    for tier, rungs in chains.items():
        model = pins.get(tier)
        idx = next((i for i, r in enumerate(rungs) if r.model == model), None)
        if idx is None or idx == 0:
            pinned[tier] = rungs
            continue
        logger.info("Preflight pin for %s: starting at %s", tier, model)
        pinned[tier] = rungs[idx:] + rungs[:idx]
    return pinned


def chain_timeout(config: Dict[str, Any]) -> float:
    """Per-rung window for CompositeClient, from config.

    This is a ceiling on ONE model's attempt, not on a backend's whole chain.
    The older per-backend window had to cover `timeout x (1 + retries) x
    models-in-chain`, and a backend whose worst case overran it was cut off
    mid-chain on every call — how the paid backstop was once found to be dead
    on arrival. With one model per rung there is no inner chain to outrun.
    """
    llm_config = config.get("llm", {}) or {}
    return llm_config.get(
        "rung_timeout_seconds",
        llm_config.get(
            "fallback_timeout_seconds",
            (config.get("composite", {}) or {}).get("timeout_seconds", 240),
        ),
    )
