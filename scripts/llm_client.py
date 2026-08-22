#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Abstract base class for LLM clients.

Defines the common interface that all LLM client implementations must satisfy:
GeminiCLIClient, OpencodeClient, and any future backends.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
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
