#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Composite LLM client that walks a tier's chain of MODELS, in order.

The unit of fallback is a model, not a backend. Each rung names a model and
the backend that can reach it, so a chain is free to interleave transports:

    heavy: opencode/muse-spark-1.3  ->  openrouter/nemotron-3-ultra  ->  ...

The older version looped over backends and let each pick its own model, which
meant a model's position was decided by who hosted it. A free model reachable
only through the paid backend's transport could never be tried before the free
backend's entire roster had failed, and in practice was never tried at all.

The first rung that returns a non-None result wins; if every rung fails,
invoke() returns None.
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

from scripts.llm_chain import Rung
from scripts.llm_client import BaseLLMClient

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CompositeClient(BaseLLMClient):
    """Walk a tier's model chain until one rung returns a result.

    Each rung is bounded by its own timeout, so a hanging model (e.g. a CLI
    stuck on a quota-exhausted key) costs one window rather than the run. A
    rung that exceeds its window is skipped for THIS call only; nothing is
    permanently marked slow.

    One window per model is also what retired the old budget guard. When a
    backend served a whole chain behind a single window, its worst case was
    `timeout x (1 + retries) x models-in-chain`, and any backend that overran
    it was cut off mid-chain on every call — how the paid backstop was once
    found to be dead on arrival. A rung has no inner chain to outrun.
    """

    def __init__(
        self,
        clients: Dict[str, BaseLLMClient],
        chains: Dict[str, List[Rung]],
        timeout: Optional[float] = None,
    ):
        if not clients:
            raise ValueError("CompositeClient requires at least one client")
        self.clients: Dict[str, BaseLLMClient] = clients
        self.chains: Dict[str, List[Rung]] = chains or {}
        # Default: 240s per rung (enough for a real LLM call, small enough that
        # a hanging model can't stall the run).
        self._timeout = timeout if timeout is not None else 240.0
        self._served_by: List[str] = []  # which model handled each call
        # tier -> (model, rung index). Only this layer knows a rung's position,
        # so "served by the second choice" is only reportable from here.
        self._tier_rung: Dict[str, Tuple[str, int]] = {}
        self._warn_if_rungs_unreachable()

    def _warn_if_rungs_unreachable(self) -> None:
        """Warn about rungs whose backend is not enabled, or that overrun the window.

        A rung naming a disabled backend is dead weight in the chain; saying so
        at startup beats discovering it as a silent skip at 06:00.
        """
        for tier, rungs in self.chains.items():
            for rung in rungs:
                if rung.backend not in self.clients:
                    logger.warning(
                        "Composite: %s rung %s needs backend %r, which is not "
                        "enabled — it will always be skipped",
                        tier, rung.model, rung.backend,
                    )
                    continue
                worst = self._worst_case_seconds(self.clients[rung.backend])
                if worst is not None and worst > self._timeout:
                    logger.warning(
                        "Composite: rung %s needs up to %.0fs (timeout x retries) "
                        "but the per-rung window is %.0fs — it will be cut off. "
                        "Lower that backend's timeout/max_retries or raise "
                        "llm.rung_timeout_seconds.",
                        rung.model, worst, self._timeout,
                    )

    @staticmethod
    def _worst_case_seconds(client: BaseLLMClient) -> Optional[float]:
        """Worst-case wall time for ONE model on this backend."""
        per_call = getattr(client, "_timeout", None)
        if not per_call:
            return None
        retries = getattr(client, "max_retries", 0) or 0
        return per_call * (1 + retries)

    @property
    def available(self) -> bool:
        return any(c.available for c in self.clients.values())

    def _invoke_with_timeout(
        self,
        client: BaseLLMClient,
        model: str,
        prompt: str,
        tier: str,
        system_prompt: Optional[str],
        kwargs: dict,
    ) -> Tuple[Optional[str], bool]:
        """Run one rung in a thread. Returns (result, timed_out)."""
        result_box: List[Optional[str]] = [None]
        exc_box: List[BaseException] = []

        def runner():
            try:
                result_box[0] = client.invoke(
                    prompt,
                    tier=tier,
                    system_prompt=system_prompt,
                    model=model,
                    **kwargs,
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
                "Composite: rung %s raised %r (tier=%s); trying next",
                model, exc_box[0], tier,
            )
            return None, False
        return result_box[0], False

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """Walk this tier's chain, rung by rung, until one serves the prompt.

        `model`, if given, pins the call to that one rung — used by diagnostics
        that need to exercise a specific model rather than the chain.
        """
        rungs = self.chains.get(tier) or []
        if model is not None:
            rungs = [r for r in rungs if r.model == model] or [
                Rung(model=model, backend=self._backend_for(model))
            ]
        if not rungs:
            logger.warning("Composite: no chain configured for tier=%s", tier)
            return None

        any_available = False
        for idx, rung in enumerate(rungs):
            client = self.clients.get(rung.backend)
            if client is None:
                logger.debug(
                    "Composite: skipping %s (tier=%s); backend %r not enabled",
                    rung.model, tier, rung.backend,
                )
                continue
            try:
                if not client.available:
                    logger.debug(
                        "Composite: skipping %s (tier=%s); backend %r unavailable",
                        rung.model, tier, rung.backend,
                    )
                    continue
            except Exception:
                continue
            any_available = True
            result, timed_out = self._invoke_with_timeout(
                client, rung.model, prompt, tier, system_prompt, kwargs
            )
            if timed_out:
                logger.warning(
                    "Composite: rung %s timed out after %.0fs for tier=%s; trying next",
                    rung.model, self._timeout, tier,
                )
                continue
            if result:
                self._served_by.append(rung.model)
                self._tier_rung[tier] = (rung.model, idx)
                logger.info(
                    "Composite: served tier=%s by %s (rung %d)", tier, rung.model, idx
                )
                return result
            logger.warning(
                "Composite: rung %s returned None for tier=%s; trying next",
                rung.model, tier,
            )

        if not any_available:
            logger.warning("Composite: no rung reachable for tier=%s", tier)
        else:
            logger.warning("Composite: every rung failed for tier=%s", tier)
        return None

    @staticmethod
    def _backend_for(model: str) -> str:
        from scripts.llm_chain import resolve_backend
        return resolve_backend(model) or ""

    def _counts(self) -> Tuple[int, int]:
        """Return (successful_served, total_calls)."""
        return len(self._served_by), len(self._served_by)

    def _collect_key_rows(self) -> list:
        """Aggregate per-key rotation rows across all backend clients.

        Each row: (provider, key_index, preview, success, failures).
        """
        rows = []
        for client in self.clients.values():
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
        for client in self.clients.values():
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
        fallback_note = self._render_fallback_note()
        if fallback_note:
            joined += "\n\n" + fallback_note
        if rotation:
            joined += "\n\n" + rotation
        return joined

    def _render_fallback_note(self) -> str:
        """Name any tier that had to fall past its first-choice rung.

        A quiet fallback is how a chain rots unnoticed: the run still succeeds,
        bills nothing, and reads normally, while the model at the top of the
        chain has been dead for weeks.
        """
        fell_back = {
            tier: (model, idx)
            for tier, (model, idx) in self._tier_rung.items()
            if idx > 0
        }
        if not fell_back:
            return ""
        lines = ["**Chain fallbacks this run:**\n"]
        for tier in ("heavy", "medium", "light"):
            if tier not in fell_back:
                continue
            model, idx = fell_back[tier]
            first = self.chains.get(tier, [])
            first_model = first[0].model if first else "n/a"
            lines.append(
                f"- **{tier}**: served by rung {idx} `{model}` "
                f"— `{first_model}` did not answer\n"
            )
        return "".join(lines)

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
