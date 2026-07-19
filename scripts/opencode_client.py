#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
OpenCode CLI client.

Calls `opencode run --format json` as a subprocess and parses the NDJSON
event stream to extract response text. Uses free-tier OpenCode Zen models
(opencode/deepseek-v4-flash-free) by default.
"""

import json
import logging
import random
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from scripts.llm_client import BaseLLMClient

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "heavy": "opencode/deepseek-v4-flash-free",
    "medium": "opencode/deepseek-v4-flash-free",
    "light": "opencode/deepseek-v4-flash-free",
}

# Free-tier backup models tried in order if the primary model for a tier
# fails (non-zero exit, empty response, or timeout). Keep glm-5.2 first
# since it has the largest free quota on the opencode-go Zen provider.
DEFAULT_FALLBACK_MODELS = {
    "heavy": ["opencode-go/glm-5.2", "opencode/deepseek-v4-flash-free"],
    "medium": ["opencode-go/glm-5.2", "opencode/deepseek-v4-flash-free"],
    "light": ["opencode-go/glm-5.2", "opencode/deepseek-v4-flash-free"],
}

DEFAULT_PRICING = {
    "input_per_million": 0.14,
    "output_per_million": 0.28,
}

# Retry policy for retryable errors (rate limits, server overload).
# Non-retryable errors (model not found, insufficient balance) skip
# straight to the next model in the fallback chain.
MAX_RETRIES_PER_MODEL = 2
RETRY_BACKOFF_BASE = 5
RETRY_BACKOFF_MAX = 15


class OpencodeClient(BaseLLMClient):
    """LLM client that calls the `opencode` CLI in headless mode."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpencodeClient.

        Args:
            config: Optional opencode configuration from config.yaml.
                    Keys: models (dict of tier->model_id),
                    max_calls_per_run, pricing.
        """
        config = config or {}
        self.enabled = config.get("enabled", True)
        models_config = config.get("models", {})
        self.models = {
            "heavy": models_config.get("heavy", DEFAULT_MODELS["heavy"]),
            "medium": models_config.get("medium", DEFAULT_MODELS["medium"]),
            "light": models_config.get("light", DEFAULT_MODELS["light"]),
        }
        # Per-tier fallback chain. The primary model is always tried first;
        # if it fails (rc != 0, empty NDJSON, or timeout) we walk this list.
        # Set to an empty list to disable fallback for a tier.
        fallback_config = config.get("fallback_models", {})
        self.fallback_models: Dict[str, list] = {
            tier: list(fallback_config.get(tier, DEFAULT_FALLBACK_MODELS[tier]))
            for tier in ("heavy", "medium", "light")
        }
        self.max_calls = config.get("max_calls_per_run", 50)
        self._timeout = config.get("timeout", 600)
        self.max_retries = config.get("max_retries_per_model", MAX_RETRIES_PER_MODEL)
        self._call_count = 0
        self._available: Optional[bool] = None

        # Per-tier usage tracking
        self._tier_calls: Dict[str, int] = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_failures: Dict[str, int] = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_input_chars: Dict[str, int] = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_output_chars: Dict[str, int] = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_time: Dict[str, float] = {"heavy": 0.0, "medium": 0.0, "light": 0.0}
        # Track which model actually served each tier's last successful call,
        # surfaced in the usage summary so we can see fallbacks in action.
        self._tier_served_by: Dict[str, Optional[str]] = {
            "heavy": None, "medium": None, "light": None,
        }
        self._tier_fallback_hits: Dict[str, int] = {
            "heavy": 0, "medium": 0, "light": 0,
        }

        pricing_config = config.get("pricing", {})
        self._pricing = {
            "input_per_million": pricing_config.get(
                "input_per_million", DEFAULT_PRICING["input_per_million"]
            ),
            "output_per_million": pricing_config.get(
                "output_per_million", DEFAULT_PRICING["output_per_million"]
            ),
        }

    @property
    def available(self) -> bool:
        """Check whether the `opencode` binary is on PATH."""
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
            return False
        self._available = shutil.which("opencode") is not None
        if self._available:
            logger.info("Opencode binary found on PATH")
        else:
            logger.warning("Opencode binary not found on PATH")
        return self._available

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Send a prompt via `opencode run --format json` and parse the response.

        The primary model for the tier is tried first. If it fails, each
        model in `self.fallback_models[tier]` is tried in order until one
        succeeds or the chain is exhausted.

        Failure detection is immediate for structured errors the CLI returns
        in the NDJSON stream (model not found, insufficient balance, rate
        limit), avoiding a full timeout wait. Retryable errors (rate limits,
        server overload) get up to `self.max_retries` retries with
        exponential backoff before advancing to the next model.

        Args:
            prompt: The user prompt.
            tier: Model tier ("light", "medium", "heavy") — maps to model ID.
            system_prompt: Optional system-level instructions (prepended).

        Returns:
            Response text, or None on failure.
        """
        if not self.available:
            return None

        if self._call_count >= self.max_calls:
            logger.warning(
                "Opencode call budget exhausted (%d / %d calls)",
                self._call_count,
                self.max_calls,
            )
            return None

        primary = self.models.get(tier, self.models["medium"])
        chain = [primary] + [
            m for m in self.fallback_models.get(tier, []) if m != primary
        ]

        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser Request: {prompt}"
        else:
            full_prompt = prompt

        full_prompt_len = len(full_prompt.encode("utf-8"))
        last_error_snippet = ""

        for idx, model in enumerate(chain):
            is_fallback = idx > 0
            if is_fallback:
                logger.info(
                    "Opencode falling back to %s for tier=%s (primary %s failed)",
                    model, tier, primary,
                )

            if self._call_count >= self.max_calls:
                logger.warning(
                    "Opencode call budget exhausted during fallback (%d / %d)",
                    self._call_count, self.max_calls,
                )
                return None

            cmd = [
                "opencode", "run",
                "-m", model,
                "--format", "json",
                "--auto",
                "--dir", "/tmp",
                "--pure",
                full_prompt,
            ]

            # Track whether we should try the next model in the chain.
            model_failed = True

            for attempt in range(1 + self.max_retries):
                if self._call_count >= self.max_calls:
                    break

                logger.debug(
                    "Invoking opencode (tier=%s, model=%s, attempt=%d/%d, call=%d/%d%s)",
                    tier, model, attempt + 1, 1 + self.max_retries,
                    self._call_count + 1, self.max_calls,
                    ", fallback" if is_fallback else "",
                )

                t0 = time.monotonic()

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self._timeout,
                    )

                    text, error = self._parse_ndjson_result(result.stdout)
                    elapsed = time.monotonic() - t0
                    self._tier_time[tier] += elapsed

                    if error:
                        action = self._classify_error(error)
                        err_name = error.get("name", "UnknownError")
                        err_msg = error.get("message", "")[:200]

                        if action == "fallback":
                            logger.info(
                                "Opencode fast-fallback for %s (tier=%s): "
                                "%s — %s (%.1fs)",
                                model, tier, err_name, err_msg, elapsed,
                            )
                            model_failed = True
                            break  # skip retries, try next model

                        # Retryable error (rate limit, server overload)
                        if attempt < self.max_retries:
                            backoff = min(
                                RETRY_BACKOFF_BASE * (2 ** attempt)
                                + random.uniform(0, 2),
                                RETRY_BACKOFF_MAX,
                            )
                            logger.info(
                                "Opencode retrying %s (tier=%s, attempt=%d/%d): "
                                "%s — %s (%.1fs, backoff %.1fs)",
                                model, tier, attempt + 1, 1 + self.max_retries,
                                err_name, err_msg, elapsed, backoff,
                            )
                            time.sleep(backoff)
                            continue  # retry same model
                        else:
                            logger.info(
                                "Opencode exhausted retries for %s (tier=%s): "
                                "%s — %s (%.1fs)",
                                model, tier, err_name, err_msg, elapsed,
                            )
                            model_failed = True
                            break  # all retries used, try next model

                    if result.returncode != 0 and not text:
                        last_error_snippet = (result.stderr or "")[:300]
                        if attempt < self.max_retries:
                            backoff = min(
                                RETRY_BACKOFF_BASE * (2 ** attempt)
                                + random.uniform(0, 2),
                                RETRY_BACKOFF_MAX,
                            )
                            logger.info(
                                "Opencode retrying %s (tier=%s, attempt=%d/%d): "
                                "rc=%d (%.1fs, backoff %.1fs)",
                                model, tier, attempt + 1, 1 + self.max_retries,
                                result.returncode, elapsed, backoff,
                            )
                            time.sleep(backoff)
                            continue
                        else:
                            model_failed = True
                            break

                    if text:
                        # Success
                        self._call_count += 1
                        self._tier_calls[tier] += 1
                        self._tier_input_chars[tier] += full_prompt_len
                        self._tier_output_chars[tier] += len(text.encode("utf-8"))
                        self._tier_served_by[tier] = model
                        if is_fallback:
                            self._tier_fallback_hits[tier] += 1
                        return text

                    # Empty response (no text, no error, rc=0)
                    logger.debug(
                        "opencode returned empty NDJSON response (model=%s, attempt=%d)",
                        model, attempt,
                    )
                    model_failed = True
                    break  # no point retrying empty responses

                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - t0
                    self._tier_time[tier] += elapsed
                    if attempt < self.max_retries:
                        backoff = min(
                            RETRY_BACKOFF_BASE * (2 ** attempt)
                            + random.uniform(0, 2),
                            RETRY_BACKOFF_MAX,
                        )
                        logger.info(
                            "Opencode retrying %s (tier=%s, attempt=%d/%d): "
                            "timeout after %ds (backoff %.1fs)",
                            model, tier, attempt + 1, 1 + self.max_retries,
                            self._timeout, backoff,
                        )
                        time.sleep(backoff)
                        continue
                    logger.warning(
                        "Opencode exhausted retries for %s (tier=%s): "
                        "timeout after %ds",
                        model, tier, self._timeout,
                    )
                    model_failed = True
                    break

                except Exception as e:
                    logger.debug("opencode run exception (model=%s): %s", model, e)
                    model_failed = True
                    break

            if model_failed:
                continue  # try next model in the chain

        # All models in the chain failed
        self._tier_failures[tier] += 1
        if last_error_snippet:
            logger.warning(
                "All opencode models failed for tier=%s (primary=%s, tried %d models). "
                "Last error: %s",
                tier, primary, len(chain), last_error_snippet,
            )
        else:
            logger.warning(
                "All opencode models failed for tier=%s (primary=%s, tried %d models)",
                tier, primary, len(chain),
            )
        return None

    @staticmethod
    def _parse_ndjson_result(stdout: str) -> Tuple[str, Optional[Dict]]:
        """
        Parse opencode's --format json NDJSON output.

        Extracts both text events and structured error events. The caller
        should check the error dict first: when it is not None, the model
        returned an immediate error (model not found, insufficient balance,
        rate limit) and the text string is empty.

        Args:
            stdout: Raw stdout from the opencode process.

        Returns:
            Tuple of (concatenated_text, error_dict_or_None).
            error_dict has keys: name, message, isRetryable, statusCode
            (where available).
        """
        parts = []
        last_error = None
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON line in NDJSON output: %.80s", line)
                continue

            event_type = event.get("type")

            if event_type == "text":
                text = event.get("part", {}).get("text", "")
                if text:
                    parts.append(text)

            elif event_type == "error":
                err = event.get("error", {})
                err_name = err.get("name", "UnknownError")
                err_data = err.get("data", {})
                last_error = {
                    "name": err_name,
                    "message": err_data.get("message", ""),
                    "isRetryable": err_data.get("isRetryable"),
                    "statusCode": err_data.get("statusCode"),
                }

        return "".join(parts), last_error

    @staticmethod
    def _classify_error(error: Dict) -> str:
        """
        Classify an NDJSON error event into an action.

        Returns:
            "fallback" — skip straight to the next model in the chain.
            "retry"   — retry the same model with exponential backoff.
        """
        name = error.get("name", "")
        is_retryable = error.get("isRetryable")
        status_code = error.get("statusCode")

        # Non-retryable API errors: insufficient balance, auth failure, etc.
        if name == "APIError" and is_retryable is False:
            return "fallback"

        # Server-side rate limits / overload — worth a retry.
        if is_retryable is True:
            return "retry"

        if status_code and status_code in (429, 502, 503, 504):
            return "retry"

        # UnknownError (model not found, provider not found) or anything
        # else we haven't seen — fail safe and move to the next model.
        return "fallback"

    def get_usage_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        """
        Generate a formatted markdown summary of opencode usage and estimated costs.

        The opencode CLI does not expose token counts, so input/output tokens
        are estimated at ~4 bytes per token. Costs use configurable DeepSeek
        V4 Flash pricing (default: $0.14/1M input, $0.28/1M output).
        """
        total_calls = sum(self._tier_calls.values())
        total_failures = sum(self._tier_failures.values())
        if total_calls == 0 and total_failures == 0:
            return ""

        in_rate = self._pricing["input_per_million"]
        out_rate = self._pricing["output_per_million"]

        lines = ["\n---\n\n## Opencode Usage Summary\n\n"]
        lines.append("| Tier | Success | Failures | Input (est.) | Output (est.) | Est. Cost |\n")
        lines.append("| :--- | :---: | :---: | :--- | :--- | :--- |\n")

        total_cost = 0.0
        total_in_tok = 0
        total_out_tok = 0

        for tier in ["heavy", "medium", "light"]:
            calls = self._tier_calls[tier]
            failures = self._tier_failures[tier]
            if calls == 0 and failures == 0:
                continue

            in_chars = self._tier_input_chars[tier]
            out_chars = self._tier_output_chars[tier]
            in_tok = int(in_chars / 4)
            out_tok = int(out_chars / 4)
            cost = (in_tok * in_rate + out_tok * out_rate) / 1_000_000
            total_cost += cost
            total_in_tok += in_tok
            total_out_tok += out_tok

            lines.append(
                f"| {tier.capitalize()} | {calls} | {failures} | "
                f"{in_tok:,} | "
                f"{out_tok:,} | "
                f"${cost:.4f} |\n"
            )

        total_label = f"**{total_calls}**"
        total_fail_label = f"**{total_failures}**"
        total_cost_label = f"**${total_cost:.4f}**"
        lines.append(
            f"| **Total** | {total_label} | {total_fail_label} | "
            f"**{total_in_tok:,}** | **{total_out_tok:,}** | {total_cost_label} |\n\n"
        )

        lines.append(
            f"*Costs estimated at ${in_rate:.2f}/1M input and ${out_rate:.2f}/1M output "
            f"(DeepSeek V4 Flash paid-tier rates). "
            f"Tokens estimated at ~4 bytes per token. "
            f"This run used the free `opencode/deepseek-v4-flash-free` model "
            f"via the opencode CLI — actual cost was $0.00.*\n\n"
        )

        # Surface which models actually served each tier so fallback activity
        # is visible at a glance.
        served_lines = []
        for tier in ["heavy", "medium", "light"]:
            served = self._tier_served_by[tier]
            hits = self._tier_fallback_hits[tier]
            if self._tier_calls[tier] == 0 and hits == 0:
                continue
            if served and served != self.models[tier]:
                served_lines.append(
                    f"- **{tier}**: served by `{served}` (fallback, {hits} fallback hit(s))"
                )
            elif hits > 0:
                served_lines.append(
                    f"- **{tier}**: {hits} fallback hit(s), last served by `{served or 'n/a'}`"
                )
        if served_lines:
            lines.append("**Model fallback activity:**\n\n")
            lines.extend(f"{l}\n" for l in served_lines)
            lines.append("\n")

        if start_time and end_time:
            duration = end_time - start_time
            start_str = datetime.fromtimestamp(start_time).strftime("%I:%M:%S %p")
            end_str = datetime.fromtimestamp(end_time).strftime("%I:%M:%S %p")
            if duration >= 3600:
                h = int(duration // 3600)
                m = int((duration % 3600) // 60)
                s = int(duration % 60)
                duration_str = f"{h}h {m}m {s}s"
            else:
                m = int(duration // 60)
                s = int(duration % 60)
                duration_str = f"{m}m {s}s"
            lines.append(
                f"**Briefing generation took {duration_str}** "
                f"(Started: {start_str}, Finished: {end_str})\n"
            )

        return "".join(lines)
