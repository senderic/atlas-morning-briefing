#!/usr/bin/env python3
"""Direct NVIDIA NIM transport using its OpenAI-compatible endpoint."""

import logging
import os
from typing import Any, Dict, Optional, Tuple

from scripts.openrouter_client import HAS_REQUESTS, OpenRouterClient

logger = logging.getLogger(__name__)

NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

DEFAULT_MODELS = {
    "heavy": "nvidia-direct/nvidia/nemotron-3-ultra-550b-a55b",
    "medium": "nvidia-direct/nvidia/nemotron-3-super-120b-a12b",
    "light": "nvidia-direct/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
}


class NvidiaClient(OpenRouterClient):
    """Serve one explicitly selected model through NVIDIA's hosted NIM API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        values = dict(config or {})
        values.setdefault("provider", "nvidia")
        values.setdefault("display_name", "NVIDIA NIM")
        values.setdefault("free_model_description", "free hosted NIM models")
        values.setdefault("api_base", NVIDIA_API_BASE_URL)
        values.setdefault("api_key", os.environ.get("NVIDIA_API_KEY", ""))
        configured_models = dict(values.get("models", {}) or {})
        for tier, model in DEFAULT_MODELS.items():
            configured_models.setdefault(tier, model)
        values["models"] = configured_models
        super().__init__(values)
        # Never fall through to OPENROUTER_API_KEY / OPENAI_API_KEY: those
        # credentials target a different account and endpoint.
        self.api_key = os.environ.get("NVIDIA_API_KEY", "") or str(
            (config or {}).get("api_key", "")
        )
        self.reasoning_budget = max(0, int(values.get("reasoning_budget", 512)))

    def _build_payload(
        self, model: str, prompt: str, system_prompt: Optional[str],
        reasoning_enabled: bool,
    ) -> Tuple[Dict[str, Any], str, bool]:
        payload, model, sent = super()._build_payload(
            model, prompt, system_prompt, reasoning_enabled,
        )
        # NIM defaults to a reasoning budget larger than our entire output
        # limit. Reserve most tokens for the ranked stories / JSON answer.
        # https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-super-120b-a12b-infer
        if reasoning_enabled and payload["model"] == "nvidia/nemotron-3-super-120b-a12b":
            payload["chat_template_kwargs"] = {
                "enable_thinking": True,
                "low_effort": True,
                "reasoning_budget": min(self.reasoning_budget, max(0, self.max_tokens - 1)),
            }
            sent = True
        elif reasoning_enabled and payload["model"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning":
            budget = min(self.reasoning_budget, max(0, self.max_tokens - 1))
            payload["chat_template_kwargs"] = {
                "enable_thinking": True, "reasoning_budget": budget,
            }
            # Hosted NIM rejects the self-hosted thinking_token_budget field;
            # send only the supported chat-template settings.
            sent = True
        return payload, model, sent

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
        elif not HAS_REQUESTS:
            logger.warning("requests not installed. NVIDIA NIM features disabled.")
            self._available = False
        elif not self.api_key:
            logger.warning("NVIDIA_API_KEY not set. NVIDIA NIM features disabled.")
            self._available = False
        else:
            self._available = True
        return self._available

    @staticmethod
    def _to_api_model(model: str) -> str:
        """Remove the internal routing prefix, preserving NVIDIA's model id."""
        prefix = "nvidia-direct/"
        return model[len(prefix):] if model.startswith(prefix) else model
