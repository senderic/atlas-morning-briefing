#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Gemini model client.

Provides tiered model access to Google Gemini for intelligence features.
Supports multiple models with automatic fallback.
"""

import logging
import os
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for Google Gemini model inference with tiered model support."""

    # Default model IDs for each tier
    DEFAULT_MODELS = {
        "heavy": "gemini-2.5-pro",
        "medium": "gemini-2.5-flash",
        "light": "gemini-2.5-flash-8b",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GeminiClient.

        Args:
            config: Optional Gemini configuration from config.yaml.
        """
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.3)
        self.api_key = config.get("api_key", os.environ.get("GEMINI_API_KEY"))

        # Model IDs per tier
        models_config = config.get("models", {})
        self.models = {
            "heavy": models_config.get("heavy", self.DEFAULT_MODELS["heavy"]),
            "medium": models_config.get("medium", self.DEFAULT_MODELS["medium"]),
            "light": models_config.get("light", self.DEFAULT_MODELS["light"]),
        }

        self.max_calls = config.get("max_calls_per_run", 30)
        self._call_count = 0
        self._available = None

        if HAS_GENAI and self.api_key:
            genai.configure(api_key=self.api_key)

    @property
    def available(self) -> bool:
        """Check if Gemini is available and enabled."""
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
            return False
        if not HAS_GENAI:
            logger.warning("google-generativeai not installed. Gemini features disabled.")
            self._available = False
            return False
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. Gemini features disabled.")
            self._available = False
            return False

        self._available = True
        return True

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Invoke a Gemini model.

        Args:
            prompt: User prompt text.
            tier: Model tier - "heavy", "medium", or "light".
            max_tokens: Override default max tokens.
            temperature: Override default temperature.
            system_prompt: Optional system prompt.

        Returns:
            Model response text, or None if invocation fails.
        """
        if not self.available:
            logger.debug("Gemini not available, skipping invocation")
            return None

        if self._call_count >= self.max_calls:
            logger.warning(
                f"LLM call budget exhausted ({self.max_calls} calls). "
                "Skipping invocation. Increase max_calls_per_run to allow more."
            )
            return None
        self._call_count += 1

        model_id = self.models.get(tier, self.models["medium"])
        tokens = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        try:
            logger.info(f"Invoking Gemini model: {model_id} (tier: {tier})")

            generation_config = genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=tokens,
            )

            model = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=system_prompt,
                generation_config=generation_config
            )

            response = model.generate_content(prompt)

            if response.text:
                result = response.text
                logger.info(f"Gemini response received ({len(result)} chars)")
                return result
            else:
                logger.error("Gemini response was empty or blocked")
                return None

        except Exception as e:
            logger.error(f"Gemini invocation failed: {e}")
            return None
