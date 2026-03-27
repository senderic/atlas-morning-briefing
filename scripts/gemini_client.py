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
        "heavy": "gemini-3-pro-preview",
        "medium": "gemini-3-flash-preview",
        "light": "gemini-2.5-flash-lite",
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
    ) -> Optional[str]:
        """
        Invoke a Gemini model via CLI.

        Args:
            prompt: User prompt text.
            tier: Model tier - "heavy", "medium", or "light".
            max_tokens: Override default max tokens (ignored by CLI usually, but passed if supported).
            temperature: Override default temperature.
            system_prompt: Optional system prompt.

        Returns:
            Model response text, or None if invocation fails.
        """
        if not self.available:
            logger.debug("Gemini CLI not available, skipping invocation")
            return None

        if self._call_count >= self.max_calls:
            logger.warning(
                f"LLM call budget exhausted ({self.max_calls} calls). "
                "Skipping invocation."
            )
            return None
        self._call_count += 1

        model_id = self.models.get(tier, self.models["medium"])
        
        # Build the full prompt including system prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser Request: {prompt}"

        try:
            logger.info(f"Invoking Gemini model: {model_id} (tier: {tier})")

            # Using --accept-raw-output-risk to suppress the warning and get clean output
            # We pass the prompt via stdin to avoid "Argument list too long" errors
            # and use --prompt "" to ensure headless mode.
            cmd = [
                "gemini", 
                "--model", model_id, 
                "--approval-mode", "yolo", 
                "--raw-output", 
                "--accept-raw-output-risk",
                "--prompt", ""
            ]
            
            # Execute headless with prompt from stdin
            # Added timeout to prevent hanging indefinitely
            try:
                result = subprocess.run(
                    cmd, 
                    input=full_prompt, 
                    capture_output=True, 
                    text=True, 
                    timeout=120  # 2 minute timeout per call
                )
            except subprocess.TimeoutExpired:
                logger.error(f"Gemini CLI call timed out after 120s (model: {model_id})")
                return None

            if result.returncode != 0:
                logger.error(f"Gemini CLI failed with exit code {result.returncode}")
                if result.stderr:
                    logger.error(f"Gemini CLI stderr: {result.stderr.strip()}")
                return None
            
            output = result.stdout.strip()
            
            # Remove any trailing "Loaded cached credentials." or similar if they leaked into stdout
            if "Loaded cached credentials." in output:
                output = output.split("Loaded cached credentials.")[-1].strip()
            
            logger.info(f"Gemini response received ({len(output)} chars)")
            return output

        except Exception as e:
            logger.error(f"Unexpected error during Gemini invocation: {e}")
            return None
