#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
OpenRouter LLM client.

Calls the OpenRouter chat completions API (OpenAI-compatible) for tiered model
access. Requires an OPENROUTER_API_KEY in the environment.

Reasoning control is handled via the capability registry
(config/model_capabilities.yaml), which defines per-model how to disable
reasoning. On OpenRouter the only parameter that actually suppresses reasoning
tokens is ``reasoning: {"enabled": false}``; ``reasoning_effort`` is accepted
but is a no-op on the free models. Endpoints that refuse suppression return
HTTP 400 ("Reasoning is mandatory for this endpoint"), which is handled by
retrying the same model without the parameter.

Free-tier models routinely burn the whole ``max_tokens`` budget on reasoning
tokens and return HTTP 200 with an EMPTY content field and
``finish_reason == "length"``. That is a recoverable condition, not a dead
model: the same call is retried once with reasoning disabled before the chain
advances to the next model.

The client is safe to call from multiple threads (the runner enriches sections
in parallel via ``max_workers``). All counters are mutated under a lock and a
semaphore caps how many requests are in flight against the free tier at once.
"""

import logging
import os
import random
import threading
import time
from typing import Any, Dict, Optional, Tuple

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

# Default model IDs per tier. Verified free and responsive on 2026-08-27.
# The ladder is deliberately monotonic: heavy > medium > light in both
# parameter count and context window. Fallbacks may only degrade DOWN a tier,
# never up, so a heavy prompt never silently lands on a light model.
DEFAULT_MODELS = {
    # 550B / 1M ctx
    "heavy": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    # 1M ctx, emits no reasoning tokens -> consistently the fastest responder
    "medium": "openrouter/minimax/minimax-m3:free",
    # 120B-A12B / 262k ctx
    "light": "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
}

# Fallback model slugs tried per tier when the primary fails.
# Every entry is free, verified reachable, and supports reasoning suppression.
DEFAULT_FALLBACK_MODELS = {
    "heavy": [
        "openrouter/dots-studio/dots-3-note-preview:free",   # 512k ctx
        "openrouter/minimax/minimax-m3:free",                # degrade to medium
    ],
    "medium": [
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",  # degrade to light
        "openrouter/dots-studio/dots-3-note-preview:free",
    ],
    "light": [
        "openrouter/cohere/north-mini-code:free",            # 256k ctx
        "openrouter/minimax/minimax-m3:free",
    ],
}

DEFAULT_PRICING = {
    "input_per_million": 0.0,
    "output_per_million": 0.0,
}

# Retries BEYOND the first attempt, matching OpencodeClient's convention:
# max_retries_per_model = N means N+1 total attempts against that model.
DEFAULT_MAX_RETRIES_PER_MODEL = 1
RETRY_BACKOFF_BASE = 5
RETRY_BACKOFF_MAX = 15

# Cap on simultaneous in-flight requests. At 6 the free tier answered 6/6;
# at 12 it silently returned empty bodies for 4 of them, so stay well under.
DEFAULT_MAX_CONCURRENT = 4

# Substring identifying an endpoint that refuses reasoning suppression.
_REASONING_MANDATORY = "reasoning is mandatory"


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
            configured = models_config.get(tier, DEFAULT_MODELS[tier])
            # A preflight override may only replace the model WITHIN its own
            # tier, and only when preflight actually reached it. Anything else
            # keeps the configured model so tiers can never be crossed.
            pf = preflight_models.get(tier, {})
            pf_model = pf.get("model")
            if pf.get("available") and pf_model:
                if pf.get("tier", tier) != tier:
                    logger.warning(
                        "OpenRouter ignoring preflight entry for %s: it names tier %s",
                        tier, pf.get("tier"),
                    )
                    self.models[tier] = configured
                else:
                    self.models[tier] = pf_model
                    if pf_model != configured:
                        logger.info(
                            "OpenRouter preflight override %s: %s -> %s",
                            tier, configured, pf_model,
                        )
            else:
                self.models[tier] = configured

            # Build fallback chain: config fallbacks minus the selected primary
            primary = self.models[tier]
            config_fallbacks = list(fallback_config.get(tier, DEFAULT_FALLBACK_MODELS[tier]))
            # When preflight promoted a fallback to primary, keep the configured
            # primary in the chain so a transient preflight blip is recoverable.
            if primary != configured and configured not in config_fallbacks:
                config_fallbacks.insert(0, configured)
            self.fallback_models[tier] = [m for m in config_fallbacks if m != primary]

        self.max_calls = config.get("max_calls_per_run", 50)
        self._timeout = config.get("timeout", 120)
        # Exposed as `max_retries` so CompositeClient can compute this
        # backend's worst-case wall time and warn if it cannot finish inside
        # its per-backend window.
        self.max_retries = config.get(
            "max_retries_per_model", DEFAULT_MAX_RETRIES_PER_MODEL
        )
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.3)
        self._call_count = 0
        self._available: Optional[bool] = None

        # --- Concurrency control -------------------------------------------
        # Counters are mutated from the runner's enrichment thread pools.
        self._lock = threading.Lock()
        max_concurrent = int(config.get("max_concurrent_requests", DEFAULT_MAX_CONCURRENT))
        self._slots = threading.BoundedSemaphore(max(1, max_concurrent))
        self._max_concurrent = max(1, max_concurrent)

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
        self._tier_served_by: Dict[str, Optional[str]] = {
            "heavy": None, "medium": None, "light": None,
        }
        # Reported by the API when a non-free model is reached; should stay 0.
        self._billed_cost = 0.0

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

    def _reserve_call(self) -> bool:
        """Atomically claim one unit of the per-run call budget."""
        with self._lock:
            if self._call_count >= self.max_calls:
                return False
            self._call_count += 1
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
        if tier not in self.models:
            logger.warning("OpenRouter unknown tier %r; using medium", tier)
            tier = "medium"

        primary = self.models[tier]
        chain = [primary] + [
            m for m in self.fallback_models.get(tier, []) if m != primary
        ]

        for model in chain:
            # Each model gets its own reasoning state so a forced downgrade on
            # one model does not silently carry over to the next.
            model_reasoning = reasoning_enabled
            attempts = 0
            while True:
                if not self._reserve_call():
                    logger.warning(
                        "OpenRouter call budget exhausted (%d / %d calls)",
                        self._call_count, self.max_calls,
                    )
                    return None
                try:
                    result, action = self._single_call(
                        model=model,
                        prompt=prompt,
                        tier=tier,
                        system_prompt=system_prompt,
                        reasoning_enabled=model_reasoning,
                    )
                except Exception as e:
                    logger.error("OpenRouter call exception (model=%s): %s", model, e)
                    result, action = None, "retry"

                if result:
                    with self._lock:
                        self._tier_served_by[tier] = model
                    return result

                if action == "reasoning_overflow":
                    # HTTP 200 but the model spent the entire max_tokens budget
                    # on reasoning tokens. Retrying the same model with
                    # reasoning off recovers this; it is not a dead endpoint.
                    if model_reasoning:
                        logger.info(
                            "OpenRouter %s (tier=%s) exhausted max_tokens on reasoning; "
                            "retrying with reasoning disabled",
                            model, tier,
                        )
                        model_reasoning = False
                        continue
                    logger.warning(
                        "OpenRouter %s (tier=%s) returned empty content even with "
                        "reasoning disabled; trying next model",
                        model, tier,
                    )
                    break

                if action == "fallback":
                    logger.warning(
                        "OpenRouter non-recoverable error for %s (tier=%s); skipping to next",
                        model, tier,
                    )
                    break  # move to next model/provider

                # Transient error: retry a bounded number of times with backoff.
                attempts += 1
                if attempts > self.max_retries:
                    logger.warning(
                        "OpenRouter exhausted retries for %s (tier=%s); trying next",
                        model, tier,
                    )
                    break
                backoff = min(
                    RETRY_BACKOFF_BASE * (2 ** (attempts - 1)) + random.uniform(0, 2),
                    RETRY_BACKOFF_MAX,
                )
                logger.info(
                    "OpenRouter retrying %s (tier=%s, attempt=%d/%d, backoff %.1fs)",
                    model, tier, attempts, self.max_retries, backoff,
                )
                time.sleep(backoff)

        logger.warning(
            "All OpenRouter models failed for tier=%s (primary=%s, tried %d)",
            tier, primary, len(chain),
        )
        return None

    @staticmethod
    def _to_api_model(model: str) -> str:
        """Map an internal slug to the id OpenRouter expects.

        Internal slugs carry a single ``openrouter/`` routing prefix, e.g.
        ``openrouter/nvidia/nemotron-...``. Only that first segment is removed
        — ``openrouter/openrouter/free`` must resolve to the real free model
        ``openrouter/free``, never to the *paid* ``openrouter/auto`` router.
        """
        if model.startswith("openrouter/"):
            return model[len("openrouter/"):]
        return model

    def _build_payload(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        reasoning_enabled: bool,
    ) -> Tuple[Dict[str, Any], str, bool]:
        """Build the request body. Returns (payload, effective_model, reasoning_param_sent)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        before = set(payload)
        adjusted = self.apply_reasoning_control(model, payload, reasoning_enabled)
        if isinstance(adjusted, str):
            # Capability registry asked for a model swap, not a payload edit.
            model = adjusted
            payload["model"] = model
        else:
            payload = adjusted
        reasoning_param_sent = bool(set(payload) - before)

        payload["model"] = self._to_api_model(payload["model"])
        return payload, model, reasoning_param_sent

    def _single_call(
        self,
        model: str,
        prompt: str,
        tier: str,
        system_prompt: Optional[str],
        reasoning_enabled: bool = True,
    ) -> Tuple[Optional[str], str]:
        """Make one OpenRouter call. Returns (content_or_None, action).

        action is one of:
          ``"fallback"``            non-recoverable -> try the next model
          ``"retry"``               transient -> retry the same model
          ``"reasoning_overflow"``  200 + empty content, reasoning ate max_tokens
        """
        with self._lock:
            self._tier_calls[tier] += 1

        payload, model, reasoning_param_sent = self._build_payload(
            model, prompt, system_prompt, reasoning_enabled
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.time()
        logger.info(
            "Invoking OpenRouter (tier=%s, model=%s, call=%d/%d, reasoning_enabled=%s)",
            tier, model, self._call_count, self.max_calls, reasoning_enabled,
        )

        # Cap in-flight requests so parallel enrichment cannot stampede the
        # free tier into silent empty responses.
        with self._slots:
            try:
                resp = requests.post(
                    self.api_base, headers=headers, json=payload, timeout=self._timeout,
                )
            except requests.RequestException as e:
                logger.error("OpenRouter request failed (model=%s): %s", model, e)
                with self._lock:
                    self._tier_failures[tier] += 1
                return None, "retry"  # network errors are usually transient

            # An endpoint that refuses reasoning suppression answers 400. Drop
            # the parameter and retry once rather than discarding the model.
            if (
                resp.status_code == 400
                and reasoning_param_sent
                and _REASONING_MANDATORY in resp.text.lower()
            ):
                logger.info(
                    "OpenRouter %s requires reasoning; resending without the "
                    "suppression parameter (update model_capabilities.yaml)",
                    model,
                )
                retry_payload, _, _ = self._build_payload(
                    model, prompt, system_prompt, reasoning_enabled=True
                )
                try:
                    resp = requests.post(
                        self.api_base, headers=headers, json=retry_payload,
                        timeout=self._timeout,
                    )
                except requests.RequestException as e:
                    logger.error("OpenRouter retry failed (model=%s): %s", model, e)
                    with self._lock:
                        self._tier_failures[tier] += 1
                    return None, "retry"

        elapsed = time.time() - start
        if resp.status_code != 200:
            action = classify_error(status_code=resp.status_code, text=resp.text)
            logger.error(
                "OpenRouter HTTP %s (model=%s, %.1fs, action=%s): %s",
                resp.status_code, model, elapsed, action, resp.text[:300],
            )
            with self._lock:
                self._tier_failures[tier] += 1
            return None, action

        try:
            data = resp.json()
        except ValueError as e:
            logger.error("OpenRouter invalid JSON response (model=%s): %s", model, e)
            with self._lock:
                self._tier_failures[tier] += 1
            return None, "retry"

        # An error can arrive inside a 200 body.
        if isinstance(data.get("error"), dict):
            err = data["error"]
            action = classify_error(
                status_code=err.get("code") if isinstance(err.get("code"), int) else None,
                text=str(err.get("message", "")),
            )
            logger.error(
                "OpenRouter error in 200 body (model=%s, action=%s): %s",
                model, action, str(err)[:300],
            )
            with self._lock:
                self._tier_failures[tier] += 1
            return None, action

        content, finish_reason, reasoning_len = self._extract_content(data)

        # Track usage regardless of whether content came back — reasoning
        # tokens are billed/counted even when the content field is empty.
        usage = data.get("usage", {}) or {}
        with self._lock:
            self._tier_input_tokens[tier] += int(usage.get("prompt_tokens", 0) or 0)
            self._tier_output_tokens[tier] += int(usage.get("completion_tokens", 0) or 0)
            try:
                self._billed_cost += float(usage.get("cost", 0) or 0)
            except (TypeError, ValueError):
                pass

        if not content:
            with self._lock:
                self._tier_failures[tier] += 1
            if finish_reason == "length" or reasoning_len:
                logger.warning(
                    "OpenRouter empty content (model=%s, finish=%s, reasoning=%d chars) "
                    "— reasoning consumed the token budget",
                    model, finish_reason, reasoning_len,
                )
                return None, "reasoning_overflow"
            logger.warning("OpenRouter empty content (model=%s)", model)
            return None, "retry"

        # Check for CoT leakage when reasoning was disabled
        if not reasoning_enabled and self.detect_cot_leakage(content):
            logger.warning(
                "OpenRouter CoT leakage detected for %s (tier=%s) with reasoning "
                "disabled; treating as failure and falling back",
                model, tier,
            )
            with self._lock:
                self._tier_failures[tier] += 1
            return None, "fallback"

        logger.info(
            "OpenRouter response received (model=%s, %.1fs, %d chars)",
            model, elapsed, len(content),
        )
        return content, "ok"

    def _extract_content(self, data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], int]:
        """Return (content, finish_reason, reasoning_char_count)."""
        try:
            choices = data.get("choices", [])
            if not choices:
                return None, None, 0
            choice = choices[0] or {}
            finish_reason = choice.get("finish_reason")
            message = choice.get("message", {}) or {}
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
            reasoning = message.get("reasoning") or ""
            return (content or None), finish_reason, len(reasoning)
        except Exception:
            return None, None, 0

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
            model = self._tier_served_by.get(tier) or self.models.get(tier, "?")
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
        lines.append("| Tier | Model | Calls | Failures | Input (tok) | Output (tok) |\n")
        lines.append("| :--- | :--- | :---: | :---: | :--- | :--- |\n")

        total_in = 0
        total_out = 0
        for tier in ("heavy", "medium", "light"):
            calls = self._tier_calls[tier]
            failures = self._tier_failures[tier]
            if calls == 0 and failures == 0:
                continue
            in_tok = self._tier_input_tokens[tier]
            out_tok = self._tier_output_tokens[tier]
            total_in += in_tok
            total_out += out_tok
            served = self._tier_served_by.get(tier) or self.models.get(tier, "?")
            lines.append(
                f"| {tier.capitalize()} | `{served}` | {calls} | {failures} | "
                f"{in_tok:,} | {out_tok:,} |\n"
            )

        lines.append(
            f"| **Total** | | **{total_calls}** | **{total_failures}** | "
            f"**{total_in:,}** | **{total_out:,}** |\n\n"
        )

        # The roster is all `:free` models, so the API-reported cost should be
        # exactly zero. Surface it rather than an estimate, so any accidental
        # routing to a paid model is impossible to miss.
        if self._billed_cost > 0:
            lines.append(
                f"*⚠️ **OpenRouter billed ${self._billed_cost:.6f}** this run — a "
                "non-free model was reached. Check the model roster in "
                "`config.yaml`.*\n\n"
            )
        else:
            lines.append(
                "*OpenRouter billed **$0.00** this run (all tiers served by "
                "`:free` models).*\n\n"
            )
        if in_rate or out_rate:
            est = (total_in * in_rate + total_out * out_rate) / 1_000_000
            lines.append(
                f"*Equivalent paid-tier cost at ${in_rate:.2f}/1M input and "
                f"${out_rate:.2f}/1M output would have been ${est:.4f}.*\n\n"
            )
        return "".join(lines)
