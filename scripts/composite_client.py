#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Composite LLM client that tries a sequence of backend clients in order.

Used to build a multi-tier fallback chain across different LLM backends
(e.g. opencode/DeepSeek first, then the Gemini CLI). The first client that
returns a non-None result wins; if every client fails, invoke() returns None.
"""

import logging
import threading
from typing import List, Optional, Tuple

from scripts.llm_client import BaseLLMClient

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CompositeClient(BaseLLMClient):
    """Try each backend client in order until one returns a result.

    Each back-end call is bounded by a per-client timeout so a single hanging
    back-end (e.g. Gemini stuck retrying quota-exhausted keys) cannot block the
    rest of the chain. If a client exceeds the timeout it is skipped for this
    call and marked slow so later calls in the run avoid it too.
    """

    def __init__(self, clients: List[BaseLLMClient], timeout: Optional[float] = None):
        if not clients:
            raise ValueError("CompositeClient requires at least one client")
        self.clients: List[BaseLLMClient] = clients
        # Default: 240s per back-end (enough for real LLM calls, small enough
        # that a quota-exhausted / hanging client can't stall the run).
        self._timeout = timeout if timeout is not None else 240.0
        self._served_by: List[str] = []  # which client last handled each call
        # Clients that timed out previously; skipped for the rest of the run.
        self._slow_clients: set = set()

    @property
    def available(self) -> bool:
        return any(c.available for c in self.clients)

    def _invoke_with_timeout(
        self,
        client: BaseLLMClient,
        prompt: str,
        tier: str,
        system_prompt: Optional[str],
        kwargs: dict,
    ) -> Tuple[Optional[str], bool]:
        """Run a client invoke in a thread. Returns (result, timed_out)."""
        result_box: List[Optional[str]] = [None]
        exc_box: List[BaseException] = []

        def runner():
            try:
                result_box[0] = client.invoke(
                    prompt, tier=tier, system_prompt=system_prompt, **kwargs
                )
            except BaseException as e:  # noqa: BLE001 - capture any backend failure
                exc_box.append(e)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=self._timeout)
        if t.is_alive():
            return None, True  # timed out
        if exc_box:
            logger.warning(
                "Composite: client %s raised %r (tier=%s); trying next",
                type(client).__name__, exc_box[0], tier,
            )
            return None, False
        return result_box[0], False

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        any_available = False
        for client in self.clients:
            name = type(client).__name__
            try:
                if not client.available:
                    logger.debug(
                        "Composite: skipping unavailable client %s (tier=%s)",
                        name, tier,
                    )
                    continue
            except Exception:
                continue
            if name in self._slow_clients:
                logger.debug(
                    "Composite: skipping slow client %s (timed out earlier) (tier=%s)",
                    name, tier,
                )
                continue
            any_available = True
            result, timed_out = self._invoke_with_timeout(
                client, prompt, tier, system_prompt, kwargs
            )
            if timed_out:
                logger.warning(
                    "Composite: client %s timed out after %.0fs for tier=%s; "
                    "marking slow and trying next",
                    name, self._timeout, tier,
                )
                self._slow_clients.add(name)
                continue
            if result:
                self._served_by.append(name)
                logger.info(
                    "Composite: served tier=%s by %s",
                    tier, name,
                )
                return result
            logger.warning(
                "Composite: client %s returned None for tier=%s; trying next",
                name, tier,
            )

        if not any_available:
            logger.warning("Composite: no backend client available for invoke (tier=%s)", tier)
        else:
            logger.warning("Composite: all clients failed for tier=%s", tier)
        return None

    def _counts(self) -> Tuple[int, int]:
        """Return (successful_served, total_calls)."""
        return len(self._served_by), len(self._served_by)

    def _collect_key_rows(self) -> list:
        """Aggregate per-key rotation rows across all backend clients.

        Each row: (provider, key_index, preview, success, failures).
        """
        rows = []
        for client in self.clients:
            # Ask each client to suppress its own inline key table; the
            # composite renders one unified table with a Provider column.
            if hasattr(client, "render_key_rotation"):
                client.render_key_rotation = False
            try:
                getter = client.get_key_rotation_rows
            except Exception:
                continue
            try:
                for row in getter() or []:
                    rows.append(row)
            except Exception as e:
                logger.debug(
                    "Composite: key rotation rows failed for %s: %s",
                    type(client).__name__, e,
                )
        return rows

    def get_usage_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        """Merge usage summaries from all backend clients (non-empty only)."""
        # Gather key rows first so clients suppress their inline key tables.
        key_rows = self._collect_key_rows()

        parts = []
        for client in self.clients:
            try:
                s = client.get_usage_summary(start_time=start_time, end_time=end_time)
            except Exception as e:
                logger.debug("Composite: usage summary failed for %s: %s", type(client).__name__, e)
                continue
            if s and s.strip():
                parts.append(s)
        if not parts:
            return ""

        rotation = self._render_key_rotation(key_rows)

        joined = "\n\n".join(parts)
        if rotation:
            joined += "\n\n" + rotation
        return joined

    def _render_key_rotation(self, rows: list) -> str:
        """Render the unified API Key Rotation Summary with a Provider column."""
        if not rows:
            return ""
        # De-duplicate: keep later (more complete) rows per (provider, key_index).
        dedup = {}
        for provider, key_idx, preview, success, failures in rows:
            dedup[(provider, key_idx)] = (preview, success, failures)

        lines = ["---\n\n## API Key Rotation Summary\n\n"]
        lines.append("| Provider | Key | Preview / Model | Success | Failures |\n")
        lines.append("| :--- | :--- | :--- | :---: | :---: |\n")
        for (provider, key_idx), (preview, success, failures) in sorted(
            dedup.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
        ):
            lines.append(
                f"| {provider} | {key_idx} | `{preview}` | {success} | {failures} |\n"
            )
        lines.append("\n")
        return "".join(lines)
