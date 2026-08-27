#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Abstract base class for LLM clients.

Defines the common interface that all LLM client implementations must satisfy:
GeminiCLIClient, OpencodeClient, and any future backends.

Also provides a ReasoningControlMixin for model-agnostic reasoning control
using the capability registry in config/model_capabilities.yaml.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Global capability cache
_capability_cache: Optional[Dict[str, Any]] = None


def _load_capabilities() -> Dict[str, Any]:
    """Load model capabilities from config/model_capabilities.yaml."""
    global _capability_cache
    if _capability_cache is not None:
        return _capability_cache

    config_path = Path(__file__).resolve().parent.parent / "config" / "model_capabilities.yaml"
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
            _capability_cache = data or {}
            logger.debug(f"Loaded model capabilities from {config_path}")
            return _capability_cache
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load model capabilities: {e}")
        _capability_cache = {"model_capabilities": {}, "default_capabilities": {}}
        return _capability_cache


def get_model_capabilities(model: str) -> Dict[str, Any]:
    """
    Get capabilities for a specific model.

    Args:
        model: Model identifier (e.g., "opencode/nemotron-3-ultra-free")

    Returns:
        Dict with keys: supports_reasoning_control, reasoning_control_method,
        cli_flag_name, cli_flag_value, api_param_name, api_param_value,
        known_cot_leakage, non_reasoning_variant
    """
    caps = _load_capabilities()
    model_caps = caps.get("model_capabilities", {}).get(model, {})
    default_caps = caps.get("default_capabilities", {})

    # Merge with defaults
    result = default_caps.copy()
    result.update(model_caps)
    return result


class ReasoningControlMixin:
    """Mixin providing model-agnostic reasoning control via capability registry."""

    COT_LEAKAGE_MARKERS = (
        "strict grounding",
        "check verbatim",
        "is verbatim",
        "entities/facts",
        "grounding verification",
        "verification scaffolding",
        "chain of thought",
        "reasoning trace",
        "thinking process",
        "internal monologue",
    )

    def get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """Get capabilities for a model. Override in subclass if needed."""
        return get_model_capabilities(model)

    def apply_reasoning_control(self, model: str, cmd_or_payload: Dict[str, Any], reasoning_enabled: bool) -> Dict[str, Any]:
        """
        Apply reasoning control to a command or payload based on model capabilities.

        Args:
            model: Model identifier
            cmd_or_payload: Command list (opencode) or payload dict (OpenRouter)
            reasoning_enabled: Whether reasoning should be enabled

        Returns:
            Modified command/payload, or string model ID if model swap required
        """
        if reasoning_enabled:
            return cmd_or_payload

        caps = self.get_model_capabilities(model)

        if not caps.get("supports_reasoning_control", False):
            # No control available - proceed anyway, will detect CoT leakage at runtime
            logger.debug(f"Model {model} has no reasoning control support")
            return cmd_or_payload

        method = caps.get("reasoning_control_method", "none")

        if method == "cli_flag":
            # opencode CLI flag approach
            flag_name = caps.get("cli_flag_name", "variant")
            flag_value = caps.get("cli_flag_value", "minimal")
            if isinstance(cmd_or_payload, list):
                # Add flag if not present
                if flag_name not in cmd_or_payload:
                    cmd_or_payload.extend([f"--{flag_name}", flag_value])
            return cmd_or_payload

        elif method == "api_param":
            # OpenRouter API parameter approach
            param_name = caps.get("api_param_name", "reasoning_effort")
            param_value = caps.get("api_param_value", "minimal")
            if isinstance(cmd_or_payload, dict):
                cmd_or_payload[param_name] = param_value
            return cmd_or_payload

        elif method == "model_swap":
            # Must swap to non-reasoning variant
            swap_model = caps.get("non_reasoning_variant")
            if swap_model:
                logger.info(f"Reasoning disabled: swapping {model} -> {swap_model}")
                return swap_model
            return cmd_or_payload

        return cmd_or_payload

    def detect_cot_leakage(self, text: str) -> bool:
        """
        Detect if response contains leaked chain-of-thought reasoning.

        Args:
            text: Response text to check

        Returns:
            True if CoT leakage detected
        """
        if not text:
            return False
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.COT_LEAKAGE_MARKERS)


class BaseLLMClient(ABC, ReasoningControlMixin):
    """Abstract interface for LLM client implementations."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether the client is available for use (binary on PATH, enabled)."""

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        system_prompt: Optional[str] = None,
        reasoning_enabled: bool = True,
    ) -> Optional[str]:
        """
        Send a prompt to the LLM and return the response text.

        Args:
            prompt: The user prompt.
            tier: Model tier ("light", "medium", "heavy").
            system_prompt: Optional system-level instructions.
            reasoning_enabled: When False, disables chain-of-thought / reasoning
                tokens for models that support it (e.g. DeepSeek V4 Pro).
                Non-reasoning backends ignore this flag.

        Returns:
            Response string, or None on failure.
        """

    @abstractmethod
    def get_usage_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        """
        Return a markdown summary of API usage and costs.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            Markdown string (may be empty if no usage data).
        """