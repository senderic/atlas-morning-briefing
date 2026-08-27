#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
OpenRouter LLM client.

Calls the OpenRouter chat completions API (OpenAI-compatible) for tiered model
access. Used as a fallback backend when other clients (opencode CLI, Gemini CLI)
are unavailable. Requires an OPENROUTER_API_KEY in the environment.
"""

import json
import logging
import os
import random
import time
from typing import Any, Dict, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from scripts.llm_client import BaseLLMClient
from scripts.llm_errors import classify_error

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Non-reasoning model to swap in when reasoning_enabled=False.
_NON_REASONING_SWAPS = {
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free": "openrouter/nvidia/nemotron-3.5-lightning:free",
}

# Default model IDs per tier. These are common OpenRouter model slugs.
DEFAULT_MODELS = {
    "heavy": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "medium": "openrouter/minimax/minimax-m3:free",
    "light": "openrouter/google/gemma-4-31b-it:free",
}

# Fallback model slugs tried per tier when the primary fails.
DEFAULT_FALLBACK_MODELS = {
    "heavy": ["openrouter/z-ai/glm-5.2:free", "openrouter/openrouter/free"],
    "medium": ["openrouter/z-ai/glm-5.2:free", "openrouter/openrouter/free"],
    "light": ["openrouter/nvidia/nemotron-3.5-lightning:free", "openrouter/openrouter/free"],
}

DEFAULT_PRICING = {
    "input_per_million": 0.25,
    "output_per_million": 0.75,
}

MAX_RETRIES_PER_MODEL = 2
RETRY_BACKOFF_BASE = 5
RETRY_BACKOFF_MAX = 15


class OpenRouterClient(BaseLLMClient):
    """LLM client that calls OpenRouter's OpenAI-compatible completions API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, preflight_models: Optional[Dict[str, Dict]] = None):
        config = config or {}
        preflight_models = preflight_models or {}
        self.enabled = config.get("enabled", True)
        self.provider = config.get("provider", "openrouter")
        self.render_key_rotation = True  # CompositeClient sets False to unify
        self.api_key = (
            config.get("api_key")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.api_base = config.get("api_base", API_BASE_URL)

        models_config = config.get("models", {})
        fallback_config = config.get("fallback_models", {})
        self.models = {}
        self.fallback_models = {}

        for tier in ("heavy", "medium", "light"):
            # Check preflight first
            pf = preflight_models.get(tier, {})
            if pf.get("available") and pf.get("model"):
                self.models[tier] = pf["model"]
                logger.info(f"OpenRouter preflight override {tier}: using {pf['model']}")
            else:
                self.models[tier] = models_config.get(tier, DEFAULT_MODELS[tier])

            # Build fallback chain: config fallbacks minus the selected primary
            primary = self.models[tier]
            config_fallbacks = list(fallback_config.get(tier, DEFAULT_FALLBACK_MODELS[tier]))
            self.fallback_models[tier] = [m for m in config_fallbacks if m != primary]

        self.max_calls = config.get("max_calls_per_run", 50)
        self._timeout = config.get("timeout", 120)
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.3)
        self._call_count = 0
        self._available: Optional[bool] = None

        pricing_config = config.get("pricing", {})
        self._pricing = {
            "input_per_million": pricing_config.get(
                "input_per_million", DEFAULT_PRICING["input_per_million"]
            ),
            "output_per_million": pricing_config.get(
                "output_per_million", DEFAULT_PRICING["output_per_million"]
            ),
        }

        self._tier_calls = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_failures = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_input_tokens = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_output_tokens = {"heavy": 0, "medium": 0, "light": 0}

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
            return False
        if not HAS_REQUESTS:
            logger.warning("requests not installed. OpenRouter features disabled.")
            self._available = False
            return False
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set. OpenRouter features disabled.")
            self._available = False
            return False
        self._available = True
        return True

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        system_prompt: Optional[str] = None,
        reasoning_enabled: bool = True,
        **kwargs: Any,
    ) -> Optional[str]:
        if kwargs:
            logger.debug(
                "OpenRouter ignoring unexpected kwargs to invoke(): %s",
                ", ".join(kwargs),
            )
        if not self.available:
            return None
        if self._call_count >= self.max_calls:
            logger.warning(
                "OpenRouter call budget exhausted (%d / %d calls)",
                self._call_count,
                self.max_calls,
            )
            return None

        primary = self.models.get(tier, self.models["medium"])
        chain = [primary] + [
            m for m in self.fallback_models.get(tier, []) if m != primary
        ]

        # When reasoning is disabled, swap reasoning models for their
        # non-reasoning equivalents so chain-of-thought cannot leak into
        # the visible answer.
        if not reasoning_enabled:
            chain = [_NON_REASONING_SWAPS.get(m, m) for m in chain]
            if any(m != orig for m, orig in zip(chain, [primary] + self.fallback_models.get(tier, []))):
                logger.info(
                    "OpenRouter reasoning disabled — swapped tier=%s models: "
                    "%s -> %s", tier, primary, chain[0],
                )

        # Enforce the per-run budget across the whole chain.
        budget_remaining = self.max_calls - self._call_count
        chain = chain[:budget_remaining]
        if not chain:
            logger.warning("OpenRouter budget exhausted before any model tried")
            return None

        for idx, model in enumerate(chain):
            is_fallback = idx > 0
            # Bound per-model retries for transient errors only. Out-of-usage
            # errors ("fallback") skip straight to the next model.
            attempts = 0
            while True:
                try:
                    result, action = self._single_call(
                        model=model,
                        prompt=prompt,
                        tier=tier,
                        system_prompt=system_prompt,
                    )
                except Exception as e:
                    logger.error("OpenRouter call exception (model=%s): %s", model, e)
                    result, action = None, "retry"
                if result:
                    return result
                if action == "fallback":
                    logger.warning(
                        "OpenRouter out-of-usage for %s (tier=%s); skipping to next",
                        model, tier,
                    )
                    break  # move to next model/providester
                # Transient error: retry a bounded number of times with backoff.
                attempts += 1
                if attempts >= MAX_RETRIES_PER_MODEL:
                    logger.warning(
                        "OpenRouter exhausted retries for %s (tier=%s); trying next",
                        model, tier,
                    )
                    break
                backoff = min(
                    RETRY_BACKOFF_BASE * (2 ** (attempts - 1))
                    + random.uniform(0, 2),
                    RETRY_BACKOFF_MAX,
                )
                logger.info(
                    "OpenRouter retrying %s (tier=%s, attempt=%d/%d, backoff %.1fs)",
                    model, tier, attempts, MAX_RETRIES_PER_MODEL, backoff,
                )
                time.sleep(backoff)

        logger.warning(
            "All OpenRouter models failed for tier=%s (primary=%s)",
            tier, primary,
        )
        return None

    def _single_call(
        self,
        model: str,
        prompt: str,
        tier: str,
        system_prompt: Optional[str],
    ):
        """Make one OpenRouter call. Returns (content_or_None, action).

        action is ``"fallback"`` (out of usage / non-recoverable -> try next
        model) or ``"retry"`` (transient -> callers may retry with backoff).
        """
        self._call_count += 1
        if tier not in self._tier_calls:
            tier = "medium"
        self._tier_calls[tier] += 1

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.time()
        logger.info(
            "Invoking OpenRouter (tier=%s, model=%s, call=%d/%d)",
            tier, model, self._call_count, self.max_calls,
        )
        try:
            resp = requests.post(
                self.api_base,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            logger.error("OpenRouter request failed (model=%s): %s", model, e)
            self._tier_failures[tier] += 1
            # Network errors are usually transient.
            return None, "retry"

        elapsed = time.time() - start
        if resp.status_code != 200:
            action = classify_error(status_code=resp.status_code, text=resp.text)
            logger.error(
                "OpenRouter HTTP %s (model=%s, %.1fs, action=%s): %s",
                resp.status_code, model, elapsed, action, resp.text[:300],
            )
            self._tier_failures[tier] += 1
            return None, action

        try:
            data = resp.json()
        except ValueError as e:
            logger.error("OpenRouter invalid JSON response (model=%s): %s", model, e)
            self._tier_failures[tier] += 1
            return None, "retry"

        content = self._extract_content(data)
        if not content:
            logger.warning("OpenRouter empty content (model=%s)", model)
            self._tier_failures[tier] += 1
            return None, "retry"

        # Track usage tokens for the cost summary.
        usage = data.get("usage", {}) or {}
        self._tier_input_tokens[tier] += int(usage.get("prompt_tokens", 0))
        self._tier_output_tokens[tier] += int(usage.get("completion_tokens", 0))

        logger.info(
            "OpenRouter response received (model=%s, %.1fs, %d chars)",
            model, elapsed, len(content),
        )
        return content, "retry"

    def _extract_content(self, data: Dict[str, Any]) -> Optional[str]:
        try:
            choices = data.get("choices", [])
            if not choices:
                return None
            message = choices[0].get("message", {}) or {}
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
            return content if content else None
        except Exception:
            return None

    def get_key_rotation_rows(self):
        """Return per-tier model rows for the unified provider summary.

        OpenRouter authenticates with a single API key, but models differ per
        tier. Each row: (provider, key_index, preview, success, failures).
        """
        rows = []
        for tier in ("heavy", "medium", "light"):
            calls = self._tier_calls.get(tier, 0)
            failures = self._tier_failures.get(tier, 0)
            if calls == 0 and failures == 0:
                continue
            model = self.models.get(tier, "?")
            rows.append((self.provider, tier, model, calls, failures))
        return rows

    def get_usage_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        total_calls = sum(self._tier_calls.values())
        total_failures = sum(self._tier_failures.values())
        if total_calls == 0 and total_failures == 0:
            return ""

        in_rate = self._pricing["input_per_million"]
        out_rate = self._pricing["output_per_million"]

        lines = ["\n---\n\n## OpenRouter Usage Summary\n\n"]
        lines.append("| Tier | Calls | Failures | Input (tok) | Output (tok) | Est. Cost |\n")
        lines.append("| :--- | :---: | :---: | :--- | :--- | :--- |\n")

        total_cost = 0.0
        total_in = 0
        total_out = 0
        for tier in ("heavy", "medium", "light"):
            calls = self._tier_calls[tier]
            failures = self._tier_failures[tier]
            if calls == 0 and failures == 0:
                continue
            in_tok = self._tier_input_tokens[tier]
            out_tok = self._tier_output_tokens[tier]
            cost = (in_tok * in_rate + out_tok * out_rate) / 1_000_000
            total_cost += cost
            total_in += in_tok
            total_out += out_tok
            lines.append(
                f"| {tier.capitalize()} | {calls} | {failures} | "
                f"{in_tok:,} | {out_tok:,} | ${cost:.4f} |\n"
            )

        lines.append(
            f"| **Total** | **{total_calls}** | **{total_failures}** | "
            f"**{total_in:,}** | **{total_out:,}** | **${total_cost:.4f}** |\n\n"
        )
        lines.append(
            f"*Costs estimated at ${in_rate:.2f}/1M input and ${out_rate:.2f}/1M output "
            "(OpenRouter). Token counts from the API usage field.*\n\n"
        )
        return "".join(lines)
