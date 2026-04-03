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
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class GeminiCLIClient:
    """Client for Gemini CLI model inference with tiered model support."""

    # Default model IDs for each tier based on gemini-cli help
    DEFAULT_MODELS = {
        "heavy": "pro",
        "medium": "flash",
        "light": "flash",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GeminiCLIClient.

        Args:
            config: Optional Gemini configuration from config.yaml.
                    Keys: models (dict of tier->model_id),
                    max_calls_per_run.
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

        try:
            logger.info(f"Invoking Gemini model: {model_id} (tier: {tier})")
            self._call_count += 1

            cmd = [
                "gemini", "--model", model_id, "--prompt", full_prompt,
                "--approval-mode", "yolo", "--raw-output", "--accept-raw-output-risk"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            
            if output:
                logger.info(f"Gemini response received ({len(output)} chars) from {tier}")
                return output
            
            raise ValueError(f"Empty response from {tier}")

        except (subprocess.CalledProcessError, Exception) as e:
            logger.error(f"Tier {tier} failed: {str(e)[:100]}")
            
            # Recursive fallback
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
