#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Gemini CLI model client.

Provides tiered model access to Gemini CLI for intelligence features.
Uses subprocess to call the 'gemini' command.
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_random_exponential, 
    retry_if_exception, 
    before_sleep_log
)

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class GeminiCLIClient:
    """Client for Gemini CLI model inference with tiered model support."""

    # Default model IDs for each tier based on gemini-cli help
    DEFAULT_MODELS = {
        "heavy": "pro",
        "medium": "flash",
        "light": "flash-lite",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GeminiCLIClient.

        Args:
            config: Optional Gemini configuration from config.yaml.
                    Keys: models (dict of tier->model_id),
                    max_calls_per_run, retry_wait_seconds.
        """
        config = config or {}
        self.enabled = config.get("enabled", True)
        
        # Model IDs per tier
        models_config = config.get("models", {})
        self.models = {
            "heavy": models_config.get("heavy", self.DEFAULT_MODELS["heavy"]),
            "medium": models_config.get("medium", self.DEFAULT_MODELS["medium"]),
            "light": models_config.get("light", self.DEFAULT_MODELS["light"]),
        }

        self.max_calls = config.get("max_calls_per_run", 50)
        self.retry_wait_seconds = config.get("retry_wait_seconds", 61)
        self._call_count = 0
        self._available = None

        # Usage tracking
        self.usage_stats = {
            "heavy": {"calls": 0, "in_tokens": 0, "out_tokens": 0, "in_chars": 0, "out_chars": 0},
            "medium": {"calls": 0, "in_tokens": 0, "out_tokens": 0, "in_chars": 0, "out_chars": 0},
            "light": {"calls": 0, "in_tokens": 0, "out_tokens": 0, "in_chars": 0, "out_chars": 0},
        }

    @property
    def available(self) -> bool:
        """Check if Gemini CLI is available and enabled."""
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
            return False
        
        try:
            # Check if 'gemini' command exists
            subprocess.run(["which", "gemini"], capture_output=True, check=True)
            self._available = True
        except subprocess.CalledProcessError:
            logger.warning("gemini-cli not found in PATH. Gemini features disabled.")
            self._available = False
        
        return self._available

    def _execute_command(self, model_id: str, prompt: str, tier: str) -> str:
        """Execute the gemini command. Internal method for tenacity retries."""
        process_env = os.environ.copy()
        if "GEMINI_API_KEY" in process_env:
            logger.info("Using GEMINI_API_KEY from environment for gemini-cli call (passed in thru env)")

        # Add a small delay to avoid burst rate limits on Gemini Free Tier
        time.sleep(2)
        
        logger.info(f"Invoking Gemini model: {model_id} (tier: {tier})")
        self._call_count += 1

        cmd = [
            "gemini", "--model", model_id, "--prompt", prompt,
            "--approval-mode", "yolo", "--raw-output", "--accept-raw-output-risk",
            "--output-format", "json"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=900, env=process_env)
        
        try:
            # Parse JSON output from gemini-cli
            data = json.loads(result.stdout)
            output = data.get("response", "").strip()
            
            # Extract stats if available
            stats = data.get("stats", {}).get("models", {})
            model_stats = {}
            for k, v in stats.items():
                if model_id in k or k in model_id:
                    model_stats = v.get("tokens", {})
                    break
            if not model_stats and stats:
                model_stats = next(iter(stats.values())).get("tokens", {})

            # Update usage metrics
            self.usage_stats[tier]["calls"] += 1
            in_tokens = model_stats.get("input", 0) or model_stats.get("prompt", 0)
            out_tokens = model_stats.get("candidates", 0)
            
            self.usage_stats[tier]["in_tokens"] += in_tokens
            self.usage_stats[tier]["out_tokens"] += out_tokens
            
            self.usage_stats[tier]["in_chars"] += len(prompt)
            self.usage_stats[tier]["out_chars"] += len(output)

            if not output:
                raise ValueError(f"Empty response from {tier}")
                
            logger.info(f"Gemini response received ({len(output)} chars, {out_tokens} tokens) from {tier}")
            return output

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse JSON response from gemini-cli: {e}")
            output = result.stdout.strip()
            self.usage_stats[tier]["calls"] += 1
            self.usage_stats[tier]["in_chars"] += len(prompt)
            self.usage_stats[tier]["out_chars"] += len(output)
            return output

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> Optional[str]:
        """
        Invoke a Gemini model via CLI with recursive tier fallback.

        Args:
            prompt: User prompt text.
            tier: Model tier - "heavy", "medium", or "light".
            max_tokens: Override default max tokens.
            temperature: Override default temperature.
            system_prompt: Optional system prompt.
            allow_fallback: If True, recurse to lower tiers on failure.

        Returns:
            Model response text, or None if all attempts fail.
        """
        if not self.available:
            return None

        if self._call_count >= self.max_calls:
            logger.warning(f"LLM call budget exhausted. Skipping {tier}.")
            return None

        model_id = self.models.get(tier, self.models["medium"])
        full_prompt = f"{system_prompt}\n\nUser Request: {prompt}" if system_prompt else prompt

        # Ultra-persistence for the Heavy (Pro) model to avoid falling back early
        # 15 attempts gives it ~45-60 minutes of total retry time if needed
        max_attempts = 15 if tier == "heavy" else 4

        def is_transient_error(exception):
            """Check if the error is worth retrying."""
            if isinstance(exception, (subprocess.TimeoutExpired, ValueError)):
                return True
            if isinstance(exception, subprocess.CalledProcessError):
                error_msg = (exception.stderr or str(exception)).lower()
                # Stop retrying if we hit the daily hard quota
                hard_quota_keywords = ["daily", "rpd", "limit reached"]
                if any(kw in error_msg for kw in hard_quota_keywords):
                    logger.warning(f"Hard quota reached for {tier}. Skipping retries.")
                    return False
                # Retry on typical transient errors
                quota_keywords = ["resource_exhausted", "capacity", "rate limit", "429", "503", "500", "exhausted your capacity"]
                return any(kw in error_msg for kw in quota_keywords)
            return False

        try:
            # Define retry logic with tenacity: exponential backoff with jitter
            # Starts ~60s, scales exponentially with random jitter up to 300s
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_random_exponential(multiplier=45 if tier == "heavy" else 30, min=60, max=300),
                retry=retry_if_exception(is_transient_error),
                before_sleep=before_sleep_log(logger, logging.INFO),
                reraise=True
            )
            def retry_call():
                return self._execute_command(model_id, full_prompt, tier)

            return retry_call()

        except Exception as e:
            # Capture more detail in the error log
            error_detail = str(e)
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                error_detail += f" | stderr: {e.stderr.strip()}"
            
            logger.error(f"All attempts for tier {tier} failed after {max_attempts-1} retries: {error_detail}")
            
            # Recursive fallback for both timeout and other failures
            if allow_fallback:
                next_tier = {"heavy": "medium", "medium": "light"}.get(tier)
                if next_tier:
                    logger.info(f"--- Falling back from {tier} to {next_tier} ---")
                    return self.invoke(
                        prompt, tier=next_tier, max_tokens=max_tokens,
                        temperature=temperature, system_prompt=system_prompt,
                        allow_fallback=True
                    )

        return None

    def get_usage_summary(self) -> str:
        """Generate a formatted markdown summary of Gemini API usage and estimated costs."""
        # Cost constants (Gemini 1.5 Pay-as-you-go pricing)
        # Pro: $1.25/1M in, $3.75/1M out
        # Flash: $0.075/1M in, $0.30/1M out
        PRICING = {
            "heavy": {"in": 1.25 / 1_000_000, "out": 3.75 / 1_000_000},
            "medium": {"in": 0.075 / 1_000_000, "out": 0.30 / 1_000_000},
            "light": {"in": 0.075 / 1_000_000, "out": 0.30 / 1_000_000}, # Same as flash
        }

        total_cost = 0.0
        lines = ["\n---\n\n## Gemini Usage Summary\n\n"]
        lines.append("| Tier | Calls | Input (Tok/Char) | Output (Tok/Char) | Est. Cost |\n")
        lines.append("| :--- | :---: | :--- | :--- | :--- |\n")

        for tier in ["heavy", "medium", "light"]:
            stats = self.usage_stats[tier]
            if stats["calls"] == 0:
                continue
            
            # Estimate tokens if they were missing (4 chars per token)
            in_tok = stats["in_tokens"] or (stats["in_chars"] // 4)
            out_tok = stats["out_tokens"] or (stats["out_chars"] // 4)
            
            cost = (in_tok * PRICING[tier]["in"]) + (out_tok * PRICING[tier]["out"])
            total_cost += cost
            
            lines.append(
                f"| {tier.capitalize()} | {stats['calls']} | "
                f"{in_tok:,} / {stats['in_chars']:,} | "
                f"{out_tok:,} / {stats['out_chars']:,} | "
                f"${cost:.4f} |\n"
            )

        if total_cost == 0 and sum(s["calls"] for s in self.usage_stats.values()) == 0:
            return ""

        lines.append(f"| **Total** | **{sum(s['calls'] for s in self.usage_stats.values())}** | | | **${total_cost:.4f}** |\n\n")
        lines.append(f"*Note: Costs are estimated based on Gemini 1.5 Pay-as-you-go pricing.*\n")
        
        return "".join(lines)
