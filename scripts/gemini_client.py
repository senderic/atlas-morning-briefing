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
            "--approval-mode", "yolo", "--raw-output", "--accept-raw-output-risk"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=900, env=process_env)
        output = result.stdout.strip()
        if not output:
            raise ValueError(f"Empty response from {tier}")
            
        logger.info(f"Gemini response received ({len(output)} chars) from {tier}")
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
        # 12 attempts gives it ~30-45 minutes of total retry time if needed
        max_attempts = 12 if tier == "heavy" else 4

        def is_transient_error(exception):
            """Check if the error is worth retrying."""
            if isinstance(exception, (subprocess.TimeoutExpired, ValueError)):
                return True
            if isinstance(exception, subprocess.CalledProcessError):
                error_msg = (exception.stderr or str(exception)).lower()
                # Stop retrying if we hit the daily hard quota
                if "daily" in error_msg or "rpd" in error_msg:
                    logger.warning(f"Hard daily quota reached for {tier}. Skipping retries.")
                    return False
                # Retry on typical transient errors
                quota_keywords = ["resource_exhausted", "capacity", "rate limit", "429", "503", "500"]
                return any(kw in error_msg for kw in quota_keywords)
            return False

        try:
            # Define retry logic with tenacity: exponential backoff with jitter
            # Starts ~60s, scales exponentially with random jitter up to 240s
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_random_exponential(multiplier=30, min=60, max=240),
                retry=retry_if_exception(is_transient_error),
                before_sleep=before_sleep_log(logger, logging.INFO),
                reraise=True
            )
            def retry_call():
                return self._execute_command(model_id, full_prompt, tier)

            return retry_call()

        except Exception as e:
            logger.error(f"All attempts for tier {tier} failed after {max_attempts-1} retries: {str(e)}")
            
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
