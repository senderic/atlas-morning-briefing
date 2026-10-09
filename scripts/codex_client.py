#!/usr/bin/env python3
"""Small, noninteractive client for the Codex CLI JSONL interface.

One class, two roles, selected at construction:

* ``writer`` (the default) — the report writer's client. One configured model
  (`codex.model`), one reasoning effort, no retries, a small call budget.
* ``chain`` — the transport behind `codex/<model>` rungs in `llm.chains`. It
  serves exactly the model CompositeClient hands it, scales reasoning effort
  with the tier, and takes its budget, timeout, retries and concurrency cap
  from `codex.chain`, so analysis traffic cannot starve the writer.

The roles are separate instances with separate counters. They share the
executable, the pricing table and the per-pipeline call log, where a `caller`
field tells their records apart.
"""

import json
import logging
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from scripts.llm_client import BaseLLMClient
from scripts.llm_errors import classify_error

logger = logging.getLogger(__name__)

# Writer and chain instances append to the same JSONL file from the runner's
# enrichment threads; one process-wide lock keeps their lines whole.
_LOG_LOCK = threading.Lock()

_TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
_RATE_KEYS = ("input_per_million", "cached_input_per_million", "output_per_million")

RETRY_BACKOFF_SECONDS = 2.0


class CodexClient(BaseLLMClient):
    """Invoke one Codex CLI process per request, as the writer or a chain rung."""

    # Routing prefix of a chain rung (`codex/gpt-5.6-terra`). It is ours, not
    # the CLI's: `codex exec -m` wants the bare model id.
    ROUTING_PREFIX = "codex/"

    DEFAULT_MODEL = "gpt-5.6-sol"
    DEFAULT_REASONING = "high"
    DEFAULT_TIMEOUT = 300
    DEFAULT_MAX_CALLS = 5

    # --- chain role defaults (`codex.chain`) ---
    # A main run makes ~14 chain calls and each attempt spends one unit, so
    # 40 covers a run in which every call needed its retry.
    DEFAULT_CHAIN_MAX_CALLS = 40
    # Trivial prompts return in 12-25s and report prose in 13-54s; 180s is
    # generous without letting one hung process hold a rung for long.
    DEFAULT_CHAIN_TIMEOUT = 180
    DEFAULT_CHAIN_MAX_RETRIES = 1
    DEFAULT_CHAIN_MAX_CONCURRENT = 3
    # Longest a call waits for a free slot. With the two attempts above this
    # keeps a rung inside llm.rung_timeout_seconds: 120 + 2 x 180 + backoff.
    DEFAULT_CHAIN_QUEUE_TIMEOUT = 120
    # Consecutive failed calls after which the chain stops trying Codex for
    # the rest of the run, so an outage costs a few timeouts, not one per call.
    DEFAULT_CHAIN_MAX_CONSECUTIVE_FAILURES = 3
    DEFAULT_CHAIN_REASONING = {"heavy": "high", "medium": "medium", "light": "low"}
    # API-equivalent rates for GPT-5.6 Sol. Codex CLI remains authenticated
    # through the user's subscription; these rates estimate comparable API
    # value and are not a claim that the CLI call incurred a token charge.
    DEFAULT_PRICING = {
        "input_per_million": 2.0,
        "cached_input_per_million": 0.2,
        "output_per_million": 10.0,
    }
    # Rates are per model. A model with no entry is reported as "unknown"
    # rather than costed at another model's rates.
    DEFAULT_MODEL_PRICING = {"gpt-5.6-sol": DEFAULT_PRICING}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        role: str = "writer",
        caller: Optional[str] = None,
    ):
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
        self.reasoning = str(
            config.get("reasoning_effort", config.get("reasoning", self.DEFAULT_REASONING))
        )
        self.timeout = float(
            config.get("timeout_seconds", config.get("timeout", self.DEFAULT_TIMEOUT))
        )
        self.max_calls = int(config.get("max_calls_per_run", config.get("max_calls", self.DEFAULT_MAX_CALLS)))
        self.max_retries = 0
        self.max_concurrent: Optional[int] = None
        self.queue_timeout = 0.0
        self.max_consecutive_failures = 0
        self._tier_reasoning: Dict[str, str] = {}

        self.role = "chain" if role == "chain" else "writer"
        if self.role == "chain":
            chain = config.get("chain")
            chain = chain if isinstance(chain, dict) else {}
            self.timeout = float(
                chain.get("timeout_seconds", chain.get("timeout", self.DEFAULT_CHAIN_TIMEOUT))
            )
            self.max_calls = int(
                chain.get("max_calls_per_run", self.DEFAULT_CHAIN_MAX_CALLS)
            )
            self.max_retries = max(
                0, int(chain.get("max_retries", self.DEFAULT_CHAIN_MAX_RETRIES))
            )
            self.max_concurrent = max(
                1,
                int(chain.get("max_concurrent_requests", self.DEFAULT_CHAIN_MAX_CONCURRENT)),
            )
            self.queue_timeout = float(
                chain.get("queue_timeout_seconds", self.DEFAULT_CHAIN_QUEUE_TIMEOUT)
            )
            self.max_consecutive_failures = max(
                0,
                int(
                    chain.get(
                        "max_consecutive_failures",
                        self.DEFAULT_CHAIN_MAX_CONSECUTIVE_FAILURES,
                    )
                ),
            )
            self._tier_reasoning = dict(self.DEFAULT_CHAIN_REASONING)
            efforts = chain.get("reasoning_effort")
            if isinstance(efforts, dict):
                self._tier_reasoning.update(
                    {str(tier): str(effort) for tier, effort in efforts.items()}
                )
        self.caller = str(caller or self.role)
        # Read by CompositeClient's startup check: one rung's worst case is
        # `_timeout x (1 + max_retries)`.
        self._timeout = self.timeout

        self._pricing = self._load_pricing(config.get("pricing"), self.model)
        self._available: Optional[bool] = None
        self._call_count = 0
        self._consecutive_failures = 0
        self._tripped = False
        # Category of the most recent failed attempt (never the CLI's text).
        self.last_error: Optional[str] = None
        # Counters are mutated from the runner's enrichment thread pools.
        self._lock = threading.Lock()
        self._slots = (
            threading.BoundedSemaphore(self.max_concurrent)
            if self.max_concurrent
            else None
        )

        self.usage_stats: Dict[str, Any] = {
            "calls": 0,
            "failures": 0,
            "latency_s": 0.0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "model": self.model,
        }
        # Same counters, split by the model that was actually invoked.
        self._model_stats: Dict[str, Dict[str, Any]] = {}
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

    @classmethod
    def _to_cli_model(cls, model: str) -> str:
        """Remove the internal routing prefix, preserving the CLI's model id."""
        if model.startswith(cls.ROUTING_PREFIX):
            return model[len(cls.ROUTING_PREFIX):]
        return model

    @classmethod
    def _load_pricing(cls, pricing: Any, model: str) -> Dict[str, Optional[Dict[str, float]]]:
        """Build the per-model rate table from `codex.pricing`.

        Two shapes are accepted: `{<model>: {<rates>}}`, and the older flat
        `{<rates>}`, which described the one model the writer used and is
        applied to `codex.model` only. `<model>: null` marks a model as
        deliberately unpriced.
        """
        table: Dict[str, Optional[Dict[str, float]]] = {
            name: dict(rates) for name, rates in cls.DEFAULT_MODEL_PRICING.items()
        }
        pricing = pricing if isinstance(pricing, dict) else {}
        flat = {key: pricing[key] for key in _RATE_KEYS if key in pricing}
        if flat:
            table[cls._to_cli_model(model)] = {
                key: float(flat.get(key, default))
                for key, default in cls.DEFAULT_PRICING.items()
            }
        for name, rates in pricing.items():
            if name in _RATE_KEYS:
                continue
            name = cls._to_cli_model(str(name))
            if not isinstance(rates, dict):
                table[name] = None
                continue
            try:
                entry = {
                    "input_per_million": float(rates["input_per_million"]),
                    "output_per_million": float(rates["output_per_million"]),
                }
                # Without a cached rate, cached input is costed as fresh.
                entry["cached_input_per_million"] = float(
                    rates.get("cached_input_per_million", entry["input_per_million"])
                )
            except (KeyError, TypeError, ValueError):
                table[name] = None
                continue
            table[name] = entry
        return table

    @property
    def available(self) -> bool:
        """Whether the configured executable is available on PATH/filesystem."""
        if self._tripped:
            return False
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
            return False
        self._available = shutil.which(self.executable) is not None
        if not self._available:
            logger.warning("Codex executable not found: %s", self.executable)
        return self._available

    def _effort_for(self, tier: str, model: str, reasoning_enabled: bool) -> str:
        """Reasoning effort for one call.

        The writer has a single configured effort. A chain call scales with
        its tier, and `reasoning_enabled=False` drops it to the floor the
        capability registry records for the model — Codex has no switch that
        turns reasoning off, only the effort setting.
        """
        if self.role != "chain":
            return self.reasoning
        if not reasoning_enabled:
            caps = self.get_model_capabilities(model)
            floor = caps.get("reasoning_disabled_effort")
            if caps.get("reasoning_control_method") == "effort_flag" and floor:
                return str(floor)
        return self._tier_reasoning.get(tier, self.reasoning)

    def _command(self, model: str, effort: Optional[str] = None) -> list[str]:
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
            f'model_reasoning_effort="{effort or self.reasoning}"',
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
            # A completed turn is terminal.  A later semantic event belongs
            # to another/incomplete turn (or is an invalid stream), never to
            # the prose and usage already selected for this invocation.
            if saw_completion and event_type:
                saw_failure = True
                continue
            if (
                event_type in {"error", "turn.failed", "turn.error", "item.failed", "item.error"}
                or event_type.endswith(".failed")
                or event_type.endswith(".error")
                or event.get("status") in {"failed", "error"}
                or event.get("error") is not None
            ):
                saw_failure = True

            item = event.get("item")
            if event_type == "item.completed" and isinstance(item, dict):
                if item.get("status") in {"failed", "error"}:
                    saw_failure = True
                if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    message = item["text"]

            if event_type == "turn.completed":
                saw_completion = True
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]

        return message, usage, saw_completion, saw_failure

    # A descriptive alias is useful to callers and keeps the parsing seam
    # independently testable without exposing subprocess details.
    _parse_response = _parse_jsonl

    @staticmethod
    def _failure_detail(stdout: str, stderr: str = "") -> Tuple[Optional[int], str]:
        """Return (HTTP status, text) describing why a CLI turn failed.

        `codex exec --json` reports an API refusal on stdout, as an `error` /
        `turn.failed` event whose message is itself a JSON document carrying
        the HTTP status; stderr is usually empty. The text is used only to
        classify the failure and is never logged.
        """
        status: Optional[int] = None
        messages = []
        for line in (stdout or "").splitlines():
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", "")).lower()
            if event_type != "error" and not event_type.endswith(".failed"):
                continue
            message = event.get("message")
            error = event.get("error")
            if isinstance(error, dict):
                message = error.get("message", message)
            if not isinstance(message, str):
                continue
            messages.append(message)
            try:
                inner = json.loads(message)
            except (TypeError, ValueError):
                continue
            if isinstance(inner, dict) and isinstance(inner.get("status"), int):
                status = inner["status"]
        return status, " ".join(messages + [stderr or ""])

    def _log_call(self, record: Dict[str, Any]) -> None:
        if not self.call_log_path:
            return
        try:
            line = json.dumps(record, sort_keys=True) + "\n"
            with _LOG_LOCK:
                self.call_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.call_log_path, "a", encoding="utf-8") as stream:
                    stream.write(line)
        except Exception as exc:  # logging must never affect inference
            logger.debug("Could not write Codex call log: %s", exc)

    def _stats_for(self, model: str) -> Dict[str, Any]:
        """Per-model counters. Call with `self._lock` held."""
        return self._model_stats.setdefault(
            model,
            {"calls": 0, "failures": 0, "latency_s": 0.0, **{key: 0 for key in _TOKEN_KEYS}},
        )

    def _finish_attempt(
        self,
        *,
        started: float,
        model: str,
        status: str,
        error_category: Optional[str] = None,
        exit_status: Optional[int] = None,
        usage: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        effort: Optional[str] = None,
        http_status: Optional[int] = None,
    ) -> None:
        latency = time.monotonic() - started
        with self._lock:
            model_stats = self._stats_for(model)
            self.usage_stats["latency_s"] += latency
            model_stats["latency_s"] += latency
            if status != "success":
                self.failures += 1
                self.usage_stats["failures"] += 1
                model_stats["failures"] += 1
                self.last_error = error_category
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "model": model,
            "latency_s": round(latency, 3),
            # Writer and chain calls share this file; `caller` separates them.
            "caller": self.caller,
        }
        if tier:
            record["tier"] = tier
        if effort:
            record["reasoning_effort"] = effort
        if error_category:
            record["error_category"] = error_category
        if exit_status is not None:
            record["exit_status"] = exit_status
        if http_status is not None:
            record["http_status"] = http_status
        if usage:
            for key in _TOKEN_KEYS:
                if key in usage:
                    record[key] = usage[key]
        self._log_call(record)

    def _accumulate_usage(
        self, usage: Optional[Dict[str, Any]], model: Optional[str] = None
    ) -> None:
        """Account for tokens reported by any completed CLI turn."""
        if not usage:
            return
        with self._lock:
            model_stats = self._stats_for(model) if model else None
            for key in _TOKEN_KEYS:
                value = usage.get(key, 0)
                if isinstance(value, (int, float)):
                    self.usage_stats[key] += value
                    if model_stats is not None:
                        model_stats[key] += value

    def _estimated_cost(
        self,
        model: Optional[str] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """API-equivalent cost of `stats`, or None when the model is unpriced."""
        stats = self.usage_stats if stats is None else stats
        rates = self._pricing.get(self._to_cli_model(model or stats.get("model") or self.model))
        if not rates:
            return None
        cached = max(0, stats["cached_input_tokens"])
        fresh = max(0, stats["input_tokens"] - cached)
        return (
            fresh * rates["input_per_million"]
            + cached * rates["cached_input_per_million"]
            + stats["output_tokens"] * rates["output_per_million"]
        ) / 1_000_000

    def _reserve_call(self) -> bool:
        """Atomically claim one unit of the per-run call budget."""
        with self._lock:
            if self._call_count >= self.max_calls:
                return False
            self._call_count += 1
            return True

    def _note_outcome(self, succeeded: bool) -> None:
        """Track the failure streak that takes a dead transport out of the run."""
        with self._lock:
            if succeeded:
                self._consecutive_failures = 0
                return
            self._consecutive_failures += 1
            tripped = (
                self.max_consecutive_failures > 0
                and self._consecutive_failures >= self.max_consecutive_failures
                and not self._tripped
            )
            if tripped:
                self._tripped = True
        if tripped:
            logger.warning(
                "Codex failed %d calls in a row; skipping it for the rest of this run",
                self.max_consecutive_failures,
            )

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        system_prompt: Optional[str] = None,
        reasoning_enabled: bool = True,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Serve one model. Fallback across models belongs to the chain.

        `model` names the rung CompositeClient picked and is served as given,
        minus the routing prefix. Without it the configured `codex.model` is
        used, which is how the report writer drives this client.
        """
        del kwargs
        if not self.available:
            return None

        effective_model = self._to_cli_model(model or self.model)
        effort = self._effort_for(tier, model or self.model, reasoning_enabled)

        if self._slots is not None and not self._slots.acquire(timeout=self.queue_timeout):
            # Nothing was sent, so this spends no budget and is not a failure
            # of Codex itself; the chain moves on to the next rung.
            logger.warning(
                "Codex %s (tier=%s) waited %.0fs for one of %d slots; failing this rung",
                effective_model, tier, self.queue_timeout, self.max_concurrent,
            )
            return None
        try:
            return self._invoke_with_retries(
                prompt, system_prompt, effective_model, tier, effort
            )
        finally:
            if self._slots is not None:
                self._slots.release()

    def _invoke_with_retries(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        tier: str,
        effort: str,
    ) -> Optional[str]:
        attempts = 0
        while True:
            if not self._reserve_call():
                logger.warning(
                    "Codex %s call budget exhausted (%d / %d)",
                    self.caller, self._call_count, self.max_calls,
                )
                return None
            result, action = self._single_call(prompt, system_prompt, model, tier, effort)
            if result:
                self._note_outcome(True)
                return result
            if action == "fallback":
                break
            # Transient error: retry a bounded number of times.
            attempts += 1
            if attempts > self.max_retries:
                break
            logger.info(
                "Codex retrying %s (tier=%s, attempt=%d/%d)",
                model, tier, attempts, self.max_retries,
            )
            time.sleep(RETRY_BACKOFF_SECONDS)

        self._note_outcome(False)
        return None

    def _single_call(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        tier: str,
        effort: str,
    ) -> Tuple[Optional[str], str]:
        """Run one CLI process. Returns (text, "ok" | "retry" | "fallback")."""
        with self._lock:
            self.calls += 1
            self.usage_stats["calls"] += 1
            self._stats_for(model)["calls"] += 1
        started = time.monotonic()
        attempt = {"started": started, "model": model, "tier": tier, "effort": effort}
        usage: Optional[Dict[str, Any]] = None
        try:
            result = subprocess.run(
                self._command(model, effort),
                input=self._prompt_payload(prompt, system_prompt),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            self._finish_attempt(status="error", error_category="timeout", **attempt)
            return None, classify_error(text="timeout")
        except OSError:
            self._finish_attempt(status="error", error_category="os_error", **attempt)
            return None, "fallback"

        if result.returncode != 0:
            http_status, detail = self._failure_detail(result.stdout, result.stderr)
            self._finish_attempt(
                status="error",
                error_category="nonzero_exit",
                exit_status=result.returncode,
                http_status=http_status,
                **attempt,
            )
            return None, classify_error(status_code=http_status, text=detail)

        message, usage, completed, failed = self._parse_jsonl(result.stdout)
        self._accumulate_usage(usage, model)
        if failed or not completed or not usage or not message or not message.strip():
            self._finish_attempt(
                status="error",
                error_category="invalid_response",
                usage=usage,
                **attempt,
            )
            if failed:
                http_status, detail = self._failure_detail(result.stdout, result.stderr)
                return None, classify_error(status_code=http_status, text=detail)
            # A turn that ended without a usable message is worth one more try.
            return None, "retry"

        with self._lock:
            self.usage_stats["model"] = model
        self._finish_attempt(status="success", usage=usage, **attempt)
        return message.strip(), "ok"

    def get_usage_summary(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> str:
        del start_time, end_time
        stats = self.usage_stats
        chain = self.role == "chain"
        if chain and not stats["calls"]:
            return ""
        title = "Codex Chain Usage Summary" if chain else "Codex Usage Summary"
        lines = [f"\n---\n\n## {title}\n\n"]
        if not stats["calls"]:
            lines.append(
                f"**Codex (`{stats['model']}`):** no calls; estimated API-equivalent "
                "cost **$0.000000**.\n"
            )
            return "".join(lines)

        call_word = "call" if stats["calls"] == 1 else "calls"
        lines.append(
            f"**{stats['calls']} {call_word}**, {stats['failures']} failed; "
            f"{stats['latency_s']:.1f}s total latency.\n\n"
        )
        lines.append(
            "| Model | Calls | Failed | Input (total / cached) | "
            "Output (total / reasoning) | Est. API Cost |\n"
        )
        lines.append("| :--- | :---: | :---: | :--- | :--- | :--- |\n")
        with self._lock:
            rows = {model: dict(row) for model, row in self._model_stats.items()}
        # Each model is costed at its own rates; one with no configured rate
        # is reported as unknown rather than borrowing another model's.
        for model, row in (rows or {stats["model"]: stats}).items():
            cost = self._estimated_cost(model, row)
            cost_text = "unknown" if cost is None else f"${cost:.6f}"
            lines.append(
                f"| `{model}` | {row['calls']} | {row['failures']} | "
                f"{row['input_tokens']:,} / {row['cached_input_tokens']:,} | "
                f"{row['output_tokens']:,} / {row['reasoning_output_tokens']:,} | "
                f"{cost_text} |\n"
            )
        lines.append(
            "\n*Codex CLI uses subscription authentication, not an API key: these "
            "calls were not billed per token. Est. API Cost is an API-equivalent "
            "estimate at the configured per-model rates (`unknown` where none is "
            "configured). Cached input uses its discounted rate; reasoning tokens "
            "are included in total output and are not counted twice.*\n"
        )
        return "".join(lines)


# Keep the explicit CLI spelling available to callers that distinguish this
# backend from API clients; CodexClient remains the canonical short name.
CodexCLIClient = CodexClient
