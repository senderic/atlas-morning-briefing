"""A narrow routing boundary for reader-facing report prose."""

import logging
from typing import Any, List, Optional

from scripts.leak_detection import is_cot_leak
from scripts.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class ReportWriter(BaseLLMClient):
    """Prefer Codex for report prose, with the existing chain as a backstop."""

    def __init__(self, codex: BaseLLMClient, fallback: BaseLLMClient):
        self.codex = codex
        self.fallback = fallback
        self.last_backend = "unavailable"
        self.backends: List[str] = []
        self.calls = 0
        self.fallback_count = 0
        self.model = getattr(codex, "model", "")

    @staticmethod
    def _is_available(client: BaseLLMClient) -> bool:
        try:
            return bool(client.available)
        except Exception:  # availability checks must not stop report delivery
            return False

    @staticmethod
    def _acceptable(result: Optional[str]) -> Optional[str]:
        if not isinstance(result, str):
            return None
        result = result.strip()
        if not result or is_cot_leak(result):
            return None
        return result

    @property
    def available(self) -> bool:
        """Report writing remains available while either route can serve it."""
        return self._is_available(self.codex) or self._is_available(self.fallback)

    def _record(self, backend: str) -> None:
        self.last_backend = backend
        self.backends.append(backend)

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Try Codex once, then the established heavy-tier client if needed."""
        self.calls += 1
        if self._is_available(self.codex):
            try:
                result = self._acceptable(
                    self.codex.invoke(
                        prompt, tier=tier, system_prompt=system_prompt, **kwargs
                    )
                )
            except Exception as exc:  # a writer failure must retain the old fallback
                logger.warning("Codex report writer failed (%s)", type(exc).__name__)
                result = None
            if result:
                self._record("codex")
                return result

        if self._is_available(self.fallback):
            self.fallback_count += 1
            fallback_kwargs = dict(kwargs)
            try:
                result = self._acceptable(
                    self.fallback.invoke(
                        prompt,
                        tier="heavy",
                        system_prompt=system_prompt,
                        **fallback_kwargs,
                    )
                )
            except Exception as exc:  # preserve deterministic report fallback
                logger.warning("Fallback report writer failed (%s)", type(exc).__name__)
                result = None
            if result:
                self._record("fallback")
                return result

        self._record("unavailable")
        return None

    def get_usage_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        """Render Codex token/cost usage plus routing, without fallback duplication."""
        if not self.calls:
            return ""
        try:
            codex_usage = self.codex.get_usage_summary(
                start_time=start_time, end_time=end_time
            )
        except Exception:
            codex_usage = ""
        fallback_word = "fallback" if self.fallback_count == 1 else "fallbacks"
        routing = (
            f"**Report writer routing ({self.model or 'Codex'}):** {self.calls} calls, "
            f"{self.fallback_count} {fallback_word}; last backend: {self.last_backend}"
        )
        return f"{codex_usage}\n\n{routing}" if codex_usage else routing
