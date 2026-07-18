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
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, Optional

from scripts.llm_client import BaseLLMClient

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "heavy": "opencode/deepseek-v4-flash-free",
    "medium": "opencode/deepseek-v4-flash-free",
    "light": "opencode/deepseek-v4-flash-free",
}

DEFAULT_PRICING = {
    "input_per_million": 0.14,
    "output_per_million": 0.28,
}


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
        self.max_calls = config.get("max_calls_per_run", 50)
        self._timeout = config.get("timeout", 600)
        self._call_count = 0
        self._available: Optional[bool] = None

        # Per-tier usage tracking
        self._tier_calls: Dict[str, int] = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_failures: Dict[str, int] = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_input_chars: Dict[str, int] = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_output_chars: Dict[str, int] = {"heavy": 0, "medium": 0, "light": 0}
        self._tier_time: Dict[str, float] = {"heavy": 0.0, "medium": 0.0, "light": 0.0}

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

        model = self.models.get(tier, self.models["medium"])
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser Request: {prompt}"
        else:
            full_prompt = prompt

        cmd = [
            "opencode", "run",
            "-m", model,
            "--format", "json",
            "--auto",
            "--dir", "/tmp",
            "--pure",
            full_prompt,
        ]

        try:
            logger.debug(
                "Invoking opencode (tier=%s, model=%s, call=%d/%d)",
                tier, model, self._call_count + 1, self.max_calls,
            )
            t0 = time.monotonic()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )

            if result.returncode != 0:
                self._tier_failures[tier] += 1
                logger.debug(
                    "opencode run failed (rc=%d): %s",
                    result.returncode,
                    result.stderr[:300],
                )
                return None

            self._call_count += 1
            self._tier_calls[tier] += 1
            response = self._parse_ndjson_response(result.stdout)
            elapsed = time.monotonic() - t0
            self._tier_time[tier] += elapsed

            # Track input/output character counts for cost estimation
            full_prompt_len = len(full_prompt.encode("utf-8"))
            self._tier_input_chars[tier] += full_prompt_len

            if response:
                self._tier_output_chars[tier] += len(response.encode("utf-8"))
                return response

            logger.debug("opencode returned empty NDJSON response")
            self._tier_failures[tier] += 1
            return None

        except subprocess.TimeoutExpired:
            self._tier_failures[tier] += 1
            logger.warning("opencode run timed out after %ds (tier=%s)", self._timeout, tier)
            return None
        except Exception as e:
            self._tier_failures[tier] += 1
            logger.debug("opencode run exception: %s", e)
            return None

    @staticmethod
    def _parse_ndjson_response(stdout: str) -> str:
        """
        Parse opencode's --format json NDJSON output and extract text.

        The CLI emits one JSON object per line (JSONL). Only events with
        type="text" contribute to the response.

        Args:
            stdout: Raw stdout from the opencode process.

        Returns:
            Concatenated text from all text-type events.
        """
        parts = []
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON line in NDJSON output: %.80s", line)
                continue
            if event.get("type") == "text":
                text = event.get("part", {}).get("text", "")
                if text:
                    parts.append(text)
        return "".join(parts)

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
