#!/usr/bin/env python3
"""Small, noninteractive client for the Codex CLI JSONL interface."""

import json
import logging
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from scripts.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class CodexClient(BaseLLMClient):
    """Invoke one configured Codex CLI process per report-writing request."""

    DEFAULT_MODEL = "gpt-5.6-sol"
    DEFAULT_REASONING = "high"
    DEFAULT_TIMEOUT = 300
    DEFAULT_MAX_CALLS = 5

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        # Accepting the complete config as well as config["codex"] is useful
        # for callers constructing clients from a loaded YAML document.
        if isinstance(config.get("codex"), dict):
            config = config["codex"]

        self.enabled = bool(config.get("enabled", True))
        self.executable = str(
            config.get("executable", config.get("binary", config.get("cli_binary", "codex")))
        )
        self.model = str(config.get("model", self.DEFAULT_MODEL))
        self.reasoning = str(config.get("reasoning", self.DEFAULT_REASONING))
        self.timeout = float(config.get("timeout", self.DEFAULT_TIMEOUT))
        self.max_calls = int(config.get("max_calls_per_run", config.get("max_calls", self.DEFAULT_MAX_CALLS)))
        self._available: Optional[bool] = None
        self._call_count = 0

        self.usage_stats: Dict[str, Any] = {
            "calls": 0,
            "failures": 0,
            "latency_s": 0.0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "model": self.model,
        }
        # These aliases make the state convenient to inspect and mirror the
        # terminology used by the other CLI clients.
        self.calls = 0
        self.failures = 0

        log_path = config.get("call_log_path")
        if log_path:
            path = Path(log_path).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parent.parent / path
            self.call_log_path: Optional[Path] = path
        else:
            self.call_log_path = None

    @property
    def available(self) -> bool:
        """Whether the configured executable is available on PATH/filesystem."""
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
            return False
        self._available = shutil.which(self.executable) is not None
        if not self._available:
            logger.warning("Codex executable not found: %s", self.executable)
        return self._available

    def _command(self, model: str) -> list[str]:
        return [
            self.executable,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "-C",
            "/tmp",
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{self.reasoning}"',
            "--json",
            "-",
        ]

    @staticmethod
    def _prompt_payload(prompt: str, system_prompt: Optional[str]) -> str:
        system = system_prompt or ""
        return (
            "--- SYSTEM PROMPT ---\n"
            f"{system}\n"
            "--- END SYSTEM PROMPT ---\n\n"
            "--- USER PROMPT ---\n"
            f"{prompt}\n"
            "--- END USER PROMPT ---\n"
        )

    @staticmethod
    def _parse_jsonl(stdout: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, bool]:
        """Return message, usage, completed, failed from a Codex JSONL stream."""
        message: Optional[str] = None
        usage: Optional[Dict[str, Any]] = None
        saw_completion = False
        saw_failure = False

        for line in (stdout or "").splitlines():
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                # Codex may write a non-JSON diagnostic line. It is harmless
                # if the stream still contains a complete valid turn.
                continue
            if not isinstance(event, dict):
                continue

            event_type = str(event.get("type", "")).lower()
            if (
                event_type in {"error", "turn.failed", "turn.error", "item.failed", "item.error"}
                or event_type.endswith(".failed")
                or event_type.endswith(".error")
                or event.get("status") in {"failed", "error"}
                or event.get("error") is not None
            ):
                saw_failure = True

            item = event.get("item")
            if event_type in {"item.completed", "item.complete"} and isinstance(item, dict):
                if item.get("status") in {"failed", "error"}:
                    saw_failure = True
                if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    message = item["text"]
            elif event_type in {"agent_message", "agent_message.completed"}:
                if isinstance(event.get("text"), str):
                    message = event["text"]

            if event_type == "turn.completed":
                saw_completion = True
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]

        return message, usage, saw_completion, saw_failure

    # A descriptive alias is useful to callers and keeps the parsing seam
    # independently testable without exposing subprocess details.
    _parse_response = _parse_jsonl

    def _log_call(self, record: Dict[str, Any]) -> None:
        if not self.call_log_path:
            return
        try:
            self.call_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.call_log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:  # logging must never affect inference
            logger.debug("Could not write Codex call log: %s", exc)

    def _finish_attempt(
        self,
        *,
        started: float,
        model: str,
        status: str,
        error: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        latency = time.monotonic() - started
        self.usage_stats["latency_s"] += latency
        if status != "success":
            self.failures += 1
            self.usage_stats["failures"] += 1
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "model": model,
            "latency_s": round(latency, 3),
        }
        if error:
            record["error"] = error[:500]
        if usage:
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
                if key in usage:
                    record[key] = usage[key]
        self._log_call(record)

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        system_prompt: Optional[str] = None,
        reasoning_enabled: bool = True,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        del tier, reasoning_enabled, kwargs
        if not self.available:
            return None
        if self._call_count >= self.max_calls:
            logger.warning("Codex call budget exhausted (%d / %d)", self._call_count, self.max_calls)
            return None

        effective_model = model or self.model
        self._call_count += 1
        self.calls += 1
        self.usage_stats["calls"] += 1
        started = time.monotonic()
        usage: Optional[Dict[str, Any]] = None
        try:
            result = subprocess.run(
                self._command(effective_model),
                input=self._prompt_payload(prompt, system_prompt),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            self._finish_attempt(started=started, model=effective_model, status="error", error=str(exc))
            return None

        if result.returncode != 0:
            self._finish_attempt(
                started=started,
                model=effective_model,
                status="error",
                error=(result.stderr or f"Codex exited with status {result.returncode}"),
            )
            return None

        message, usage, completed, failed = self._parse_jsonl(result.stdout)
        if failed or not completed or not usage or not message or not message.strip():
            self._finish_attempt(
                started=started,
                model=effective_model,
                status="error",
                error="incomplete, failed, or empty Codex turn",
                usage=usage,
            )
            return None

        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
            value = usage.get(key, 0)
            if isinstance(value, (int, float)):
                self.usage_stats[key] += value
        self.usage_stats["model"] = effective_model
        self._finish_attempt(started=started, model=effective_model, status="success", usage=usage)
        return message.strip()

    def get_usage_summary(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> str:
        del start_time, end_time
        stats = self.usage_stats
        if not stats["calls"]:
            return "**Codex:** no calls"
        return (
            f"**Codex ({stats['model']}):** {stats['calls']} calls, "
            f"{stats['failures']} failed; "
            f"{stats['input_tokens']} input ({stats['cached_input_tokens']} cached), "
            f"{stats['output_tokens']} output tokens; "
            f"{stats['latency_s']:.1f}s"
        )


# Keep the explicit CLI spelling available to callers that distinguish this
# backend from API clients; CodexClient remains the canonical short name.
CodexCLIClient = CodexClient
