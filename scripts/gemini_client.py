#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Google Gemini CLI client (supports both `agy` (Antigravity) and legacy `gemini`).

The Antigravity CLI (`agy`) is the Go-based successor to `gemini-cli` and
replaces it before the June 18, 2026 deadline. This module auto-detects
which binary is available on PATH (preferring `agy`) and dispatches the
right argv layout for each via the BINARY_PROFILES table.

Per-binary differences captured by the profiles:
- `agy`: prompt is positional, uses `--output=json`, `--quiet`, and
  `--dangerously-skip-permissions`. Config dir is `.agy/`.
- `gemini`: prompt via `--prompt`, uses `--output-format json`,
  `--raw-output --accept-raw-output-risk`, and `--approval-mode yolo`.
  Config dir is `.gemini/`.

Override the auto-detection by setting `gemini.cli_binary` in config.yaml
to "agy" or "gemini" — useful when both binaries exist or for testing.
"""

import datetime
import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from scripts.llm_client import BaseLLMClient
from scripts.llm_errors import classify_error

RETRY_BACKOFF_BASE = 5
RETRY_BACKOFF_MAX = 15
MAX_ATTEMPTS_DEFAULT = 6

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Local-timezone ISO 8601 timestamp for per-call log records."""
    return datetime.datetime.now().astimezone().isoformat()


def _build_agy_cmd(model_id: str, prompt: str) -> List[str]:
    """
    Build argv for the Antigravity CLI (`agy`).

    Flag layout follows MIGRATION_PLAN_ANTIGRAVITY.md §4. Verify against
    `agy --help` on your installation; adjust profile if upstream renames.
    """
    return [
        "agy", prompt,
        "--model", model_id,
        "--output=json",
        "--quiet",
        "--dangerously-skip-permissions",
    ]


def _build_gemini_cmd(model_id: str, prompt: str) -> List[str]:
    """Build argv for the legacy gemini-cli (Node.js implementation)."""
    return [
        "gemini", "--model", model_id, "--prompt", prompt,
        "--approval-mode", "yolo",
        "--raw-output", "--accept-raw-output-risk",
        "--output-format", "json",
    ]


# Per-binary configuration. Order of `_DETECTION_ORDER` controls preference
# when multiple binaries are available. `gemini` wins because gemini-cli is
# the actively maintained headless tool with API-key auth, while `agy`
# (Antigravity 1.0.1) is OAuth-only and not viable for cron use — see
# MIGRATION_PLAN_ANTIGRAVITY.md §Findings. Flip the order or set
# `gemini.cli_binary: "agy"` in config.yaml once agy gains headless auth.
BINARY_PROFILES: Dict[str, Dict[str, Any]] = {
    "agy": {
        "build_cmd": _build_agy_cmd,
        "config_dirname": ".agy",
        "config_dir_env": "AGY_CONFIG_DIR",
        "trust_workspace_env": "AGY_TRUST_WORKSPACE",
        "display_name": "Antigravity CLI (agy)",
    },
    "gemini": {
        "build_cmd": _build_gemini_cmd,
        "config_dirname": ".gemini",
        "config_dir_env": "GEMINI_CONFIG_DIR",
        "trust_workspace_env": "GEMINI_CLI_TRUST_WORKSPACE",
        "display_name": "Gemini CLI (gemini)",
    },
}
_DETECTION_ORDER: List[str] = ["gemini", "agy"]


class GeminiCLIClient(BaseLLMClient):
    """Tiered LLM CLI client (auto-detects between `agy` and `gemini`)."""

    # Default model IDs for each tier — used by both agy and gemini.
    DEFAULT_MODELS = {
        "heavy": "pro",
        "medium": "flash",
        "light": "flash-lite",
    }

    # Cost constants — Gemini paid-tier rates, audited 2026-05-22 against
    # https://ai.google.dev/gemini-api/docs/pricing. CLI aliases resolve as
    # verified by scripts/audit_gemini.py (re-run it to refresh after CLI
    # updates shift the alias pointers):
    #   heavy  -> "pro"        -> gemini-3.x-pro-preview ($2/$12 for prompts
    #             <=200k; $4/$18 above). In practice this row stays $0: Pro has
    #             no free tier, so free-tier keys can't serve it at all, and the
    #             paid key 429'd ("exhausted capacity") in the audit too, with
    #             its monthly budget cap nearly used up. The exact alias (3-pro
    #             vs 3.1-pro) is therefore still unverified, but both share the
    #             <=200k rate and briefing prompts run well under 200k, so the
    #             figure holds either way.
    #   medium -> "flash"      -> gemini-3-flash-preview  ($0.50 in / $3.00 out)
    #   light  -> "flash-lite" -> gemini-3.1-flash-lite   ($0.25 in / $1.50 out)
    PRICING = {
        "heavy": {"in": 2.00 / 1_000_000, "out": 12.00 / 1_000_000},
        "medium": {"in": 0.50 / 1_000_000, "out": 3.00 / 1_000_000},
        "light": {"in": 0.25 / 1_000_000, "out": 1.50 / 1_000_000},
    }

    # Cached input tokens bill at a fraction of the normal input rate. Gemini
    # implicit context caching is ~75% off → 25% of the input price. Verify on
    # the pricing page if Google changes the cache discount.
    CACHED_INPUT_DISCOUNT = 0.25

    @classmethod
    def _cost_for(
        cls, tier: str, fresh_in: int, cached_in: int, out_text: int, thoughts: int
    ) -> float:
        """Estimated USD for one call's token usage.

        Cached input bills at CACHED_INPUT_DISCOUNT x the input rate; thinking
        ("thoughts") tokens bill at the OUTPUT rate alongside the visible answer
        (candidates). Tool tokens are tracked for visibility but not charged
        here — the CLI's `total` excludes them, so they're either zero or already
        subsumed in prompt/candidates."""
        p = cls.PRICING.get(tier)
        if not p:
            return 0.0
        return (
            fresh_in * p["in"]
            + cached_in * p["in"] * cls.CACHED_INPUT_DISCOUNT
            + (out_text + thoughts) * p["out"]
        )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GeminiCLIClient.

        Args:
            config: Optional Gemini configuration from config.yaml.
                    Keys: models (dict of tier->model_id),
                    max_calls_per_run, retry_wait_seconds.
        """
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.provider = config.get("provider", "gemini")
        self.render_key_rotation = True  # CompositeClient sets False to unify
        self.ignore_hard_quota = config.get("ignore_hard_quota", False)
        self.track_hard_quotas = config.get("track_hard_quotas", False) # Flag to permanently skip keys
        self.internal_max_attempts = config.get("internal_max_attempts", 1)
        # Bounded number of CLI attempts per logical invocation across all keys
        # (replaces the old heavy-tier "12 attempts @ up to 450s" persistence).
        self.config_retries = config.get("retries", config.get("max_attempts", MAX_ATTEMPTS_DEFAULT))

        # Explicit binary override (e.g. "agy" or "gemini"); None = auto-detect.
        # Validate so a typo doesn't silently fall through to auto-detect.
        cli_binary = config.get("cli_binary")
        if cli_binary is not None and cli_binary not in BINARY_PROFILES:
            raise ValueError(
                f"cli_binary={cli_binary!r} is not one of "
                f"{sorted(BINARY_PROFILES.keys())}"
            )
        self.cli_binary_override = cli_binary
        self._binary: Optional[str] = None  # populated on first .available

        # Model IDs per tier
        models_config = config.get("models", {})
        self.models = {
            "heavy": models_config.get("heavy", self.DEFAULT_MODELS["heavy"]),
            "medium": models_config.get("medium", self.DEFAULT_MODELS["medium"]),
            "light": models_config.get("light", self.DEFAULT_MODELS["light"]),
        }

        self.max_calls = config.get("max_calls_per_run", 50)
        self.retry_wait_seconds = config.get("retry_wait_seconds", 61)
        self.key_swap_delay = config.get("key_swap_delay", 5)
        self._call_count = 0
        self._available = None
        self._api_keys = self._load_api_keys()
        self._current_key_index = 0
        self._exhausted_keys = set()  # indices of keys that hit hard quotas (RPD)

        # Key usage tracking
        self.key_usage_stats = {
            i: {"success": 0, "failure": 0, "preview": (k[:6] + "..." + k[-4:]) if k else "None"}
            for i, k in enumerate(self._api_keys)
        }

        # Usage tracking. Token fields mirror the CLI's stats.models breakdown:
        #   in_tokens      = fresh (uncached) input  — full input rate
        #   cached_tokens  = cached input            — discounted input rate
        #   out_tokens     = output text (candidates)
        #   thought_tokens = thinking tokens         — billed at the output rate
        #   tool_tokens    = tool tokens             — tracked, not charged
        def _empty_tier_stats() -> Dict[str, int]:
            return {
                "calls": 0, "failed_attempts": 0,
                "in_tokens": 0, "cached_tokens": 0,
                "out_tokens": 0, "thought_tokens": 0, "tool_tokens": 0,
                "in_chars": 0, "out_chars": 0,
            }
        self.usage_stats = {
            "heavy": _empty_tier_stats(),
            "medium": _empty_tier_stats(),
            "light": _empty_tier_stats(),
        }

        # Optional per-call JSONL log (one record per LLM call). Disabled unless
        # `call_log_path` is configured; relative paths resolve against the repo
        # root so cron jobs running from any CWD land the log in the same place.
        log_path = config.get("call_log_path")
        if log_path:
            p = Path(log_path).expanduser()
            if not p.is_absolute():
                p = Path(__file__).resolve().parent.parent / p
            self.call_log_path: Optional[Path] = p
        else:
            self.call_log_path = None

    def _log_call(self, record: Dict[str, Any]) -> None:
        """Append one JSON line to the per-call log, if configured.

        Best-effort: a logging failure must never break the LLM call, so all
        errors are swallowed (logged at debug)."""
        if not self.call_log_path:
            return
        try:
            self.call_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.call_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug(f"Could not write call-log record: {e}")

    def _load_api_keys(self) -> List[str]:
        """
        Load API keys from environment.
        Supports:
        - GEMINI_API_KEY (can be comma-separated)
        - GEMINI_API_KEY_* (alphanumeric suffix, sorted)
        """
        keys = []
        seen_values = set()
        
        def add_key(val):
            if val and val not in seen_values:
                keys.append(val)
                seen_values.add(val)

        # 1. Check for primary GEMINI_API_KEY first (highest priority)
        raw = os.environ.get("GEMINI_API_KEY", "")
        if raw:
            for k in raw.split(","):
                add_key(k.strip())
            
        # 2. Check for suffixed variants (GEMINI_API_KEY_*), sorted by name
        suffixed_vars = sorted(
            v for v in os.environ if v.startswith("GEMINI_API_KEY_")
        )
        for var_name in suffixed_vars:
            add_key(os.environ[var_name].strip())

        if not keys:
            logger.warning("No Gemini API key found in environment!")
        elif len(keys) > 1:
            logger.info(f"Loaded {len(keys)} API keys for rotation.")
            
        return keys

    def _get_current_key(self) -> Optional[str]:
        """Get the current API key from the rotation."""
        if not self._api_keys:
            return None
        return self._api_keys[self._current_key_index]

    def _rotate_key(self) -> bool:
        """Rotate to the next API key. Returns True if a new key is available."""
        if len(self._api_keys) <= 1:
            return False
        
        # Best practice: Use a base delay + random jitter to avoid synchronized retry storms
        jitter = random.uniform(0, 10)
        total_delay = self.key_swap_delay + jitter
        
        # Determine the next available (non-exhausted) key
        next_index = (self._current_key_index + 1) % len(self._api_keys)
        attempts = 0
        if self.track_hard_quotas:
            while next_index in self._exhausted_keys and attempts < len(self._api_keys):
                next_index = (next_index + 1) % len(self._api_keys)
                attempts += 1
        
        if attempts >= len(self._api_keys) and self.track_hard_quotas:
            logger.error("All available API keys are exhausted for this run.")
            return False

        # User requirement: Key 3 (Index 2) is last resort
        if next_index == 2 and self._current_key_index < 2:
            logger.warning("⚠️ LAST RESORT: Free keys (Index 0, 1) exhausted. Rotating to PAID key (Index 2).")
        
        logger.info(f"🔄 ROTATING API KEY: Moving from index {self._current_key_index} to {next_index}")
        
        self._current_key_index = next_index
        new_key = self._get_current_key()
        key_preview = new_key[:6] + "..." + new_key[-4:] if new_key else "None"
        
        logger.info(f"✅ Now using key at index {self._current_key_index} (preview: {key_preview})")
        logger.debug(f"⏳ Waiting {total_delay:.2f}s for key swap cooldown (base: {self.key_swap_delay}s, jitter: {jitter:.2f}s)...")
        time.sleep(total_delay)
            
        return True

    @property
    def available(self) -> bool:
        """
        Resolve which CLI binary to use and cache the result.

        If `cli_binary_override` is set, only that binary is checked.
        Otherwise we walk _DETECTION_ORDER and pick the first that's on PATH.
        """
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
            return False

        candidates = (
            [self.cli_binary_override]
            if self.cli_binary_override
            else _DETECTION_ORDER
        )
        for binary in candidates:
            try:
                subprocess.run(
                    ["which", binary], capture_output=True, check=True
                )
            except subprocess.CalledProcessError:
                continue
            self._binary = binary
            self._available = True
            logger.info(
                f"Using {BINARY_PROFILES[binary]['display_name']} for LLM calls"
            )
            return True

        attempted = ", ".join(candidates)
        logger.warning(
            f"No LLM CLI binary found in PATH (looked for: {attempted}). "
            "Intelligence features disabled."
        )
        self._available = False
        return False

    @property
    def binary(self) -> Optional[str]:
        """
        Return the active binary name.

        Resolution order:
          1. The binary picked by .available (cached on first read).
          2. The explicit cli_binary override (if no detection was run).
          3. The first entry in _DETECTION_ORDER as a last-resort default.

        The last two fallbacks let callers that bypass .available (e.g. unit
        tests that pre-set ._available = True) still build a coherent argv.
        """
        if self._binary is not None:
            return self._binary
        if self._available is None and self.enabled:
            # Try real detection first
            _ = self.available
            if self._binary is not None:
                return self._binary
        # Fall back to the override or the default preference
        return self.cli_binary_override or _DETECTION_ORDER[0]

    def _execute_command(self, model_id: str, prompt: str, tier: str) -> str:
        """Execute the active CLI binary with a sandboxed config dir."""
        import tempfile
        import shutil

        # `binary` falls back to override / preferred default when detection
        # hasn't run, so this is always defined for an enabled client.
        binary = self.binary
        profile = BINARY_PROFILES[binary]

        # Create a temporary directory for the CLI config (sandboxed per call)
        tmp_config_dir = tempfile.mkdtemp(prefix=f"atlas_{binary}_config_")

        try:
            # Drop a settings.json into <tmp>/<config_dirname>/ so the CLI
            # picks up our retry override. Gemini honors `general.maxAttempts`
            # and `general.requestTimeout`; agy is expected to use the same
            # shape per the migration plan — verify against `agy --help` and
            # tweak BINARY_PROFILES if upstream changes the key names.
            cli_cfg_dir = Path(tmp_config_dir) / profile["config_dirname"]
            cli_cfg_dir.mkdir(parents=True, exist_ok=True)
            settings_path = cli_cfg_dir / "settings.json"
            with open(settings_path, "w") as f:
                json.dump({
                    "general": {
                        "maxAttempts": self.internal_max_attempts,
                        "requestTimeout": 120000,
                    },
                    "tools": {"autoAccept": True},
                }, f)

            # Use standard environment but override the config directory and HOME
            process_env = os.environ.copy()
            process_env[profile["config_dir_env"]] = tmp_config_dir

            # API key plumbing — set both GEMINI_API_KEY (legacy) and the
            # agy-named alias so either binary picks it up.
            api_key = self._get_current_key()
            key_index = self._current_key_index

            if api_key:
                process_env["GEMINI_API_KEY"] = api_key
                process_env["AGY_API_KEY"] = api_key
                key_preview = api_key[:6] + "..." + api_key[-4:]
                logger.debug(
                    f"Using API Key index {key_index} for tier {tier}: {key_preview}"
                )
            else:
                logger.warning(f"No API key available for tier {tier}!")

            # CRITICAL: Force the CLI to use the API key by masking system-wide
            # auth. This prevents fallback to gcloud/ADC credentials or OAuth.
            process_env["GOOGLE_API_KEY"] = ""
            process_env["GOOGLE_APPLICATION_CREDENTIALS"] = ""
            process_env["CLOUDSDK_AUTH_ACCESS_TOKEN"] = ""
            process_env["HOME"] = tmp_config_dir  # block ~/.config / ~/.gemini lookups
            process_env[profile["trust_workspace_env"]] = "true"  # headless mode
            logger.debug(
                f"Strict Auth: Masked system-wide Google Cloud auth, "
                f"redirected HOME, cleared GOOGLE_API_KEY (binary={binary})."
            )

            # Add a small initial delay for the heavy tier to avoid hitting RPM limits
            if tier == "heavy":
                time.sleep(1)

            logger.info(f"Invoking {binary} model: {model_id} (tier: {tier})")

            cmd = profile["build_cmd"](model_id, prompt)

            t0 = time.monotonic()
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True,
                    timeout=900, env=process_env,
                )
                if key_index in self.key_usage_stats:
                    self.key_usage_stats[key_index]["success"] += 1
            except Exception as e:
                self.usage_stats[tier]["failed_attempts"] += 1
                if key_index in self.key_usage_stats:
                    self.key_usage_stats[key_index]["failure"] += 1
                # Capture stderr too — it carries the actionable message
                # (e.g. "exhausted capacity"), which str(e) alone omits.
                err = str(e)
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    err += f" | stderr: {e.stderr.strip()}"
                self._log_call({
                    "ts": _now_iso(), "tier": tier, "model_id": model_id,
                    "key_index": key_index,
                    "latency_s": round(time.monotonic() - t0, 3),
                    "status": "error", "error": err[:500],
                })
                raise e

            latency = round(time.monotonic() - t0, 3)

            try:
                # Parse JSON output from gemini-cli
                data = json.loads(result.stdout)
                output = data.get("response", "").strip()

                # Locate this model's token stats. The CLI keys stats.models by
                # the resolved model ID (e.g. "gemini-3-flash-preview"), not the
                # alias we passed in ("flash"), so match loosely then fall back
                # to the first entry.
                stats = data.get("stats", {}).get("models", {})
                resolved_model, model_stats = model_id, {}
                for k, v in stats.items():
                    if model_id in k or k in model_id:
                        resolved_model, model_stats = k, v.get("tokens", {})
                        break
                if not model_stats and stats:
                    resolved_model, first = next(iter(stats.items()))
                    model_stats = first.get("tokens", {})

                # Token breakdown. The CLI reports:
                #   prompt = total input = input(fresh) + cached
                #   total  = prompt + candidates + thoughts
                # `input` is the full-price (uncached) portion; prefer it, else
                # derive it as prompt - cached for older CLIs that omit `input`.
                cached_tokens = model_stats.get("cached", 0)
                prompt_tokens = model_stats.get("prompt", 0)
                fresh_in = model_stats.get("input")
                if fresh_in is None:
                    fresh_in = max(prompt_tokens - cached_tokens, 0)
                out_tokens = model_stats.get("candidates", 0)
                thought_tokens = model_stats.get("thoughts", 0)
                tool_tokens = model_stats.get("tool", 0)

                # Update usage metrics
                self.usage_stats[tier]["calls"] += 1
                self.usage_stats[tier]["in_tokens"] += fresh_in
                self.usage_stats[tier]["cached_tokens"] += cached_tokens
                self.usage_stats[tier]["out_tokens"] += out_tokens
                self.usage_stats[tier]["thought_tokens"] += thought_tokens
                self.usage_stats[tier]["tool_tokens"] += tool_tokens
                self.usage_stats[tier]["in_chars"] += len(prompt)
                self.usage_stats[tier]["out_chars"] += len(output)

                self._log_call({
                    "ts": _now_iso(), "tier": tier, "model_id": model_id,
                    "resolved_model": resolved_model, "key_index": key_index,
                    "latency_s": latency,
                    "status": "ok" if output else "empty",
                    "tokens": {
                        "input": fresh_in, "cached": cached_tokens,
                        "candidates": out_tokens, "thoughts": thought_tokens,
                        "tool": tool_tokens, "prompt": prompt_tokens,
                        "total": model_stats.get("total", 0),
                    },
                    "cost_usd": round(
                        self._cost_for(tier, fresh_in, cached_tokens, out_tokens, thought_tokens), 6
                    ),
                    "response_chars": len(output),
                })

                if not output:
                    raise ValueError(f"Empty response from {tier}")

                logger.info(
                    f"Gemini response received ({len(output)} chars, "
                    f"{out_tokens}+{thought_tokens} out tok, {latency}s) from {tier}"
                )
                return output

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse JSON response from gemini-cli: {e}")
                output = result.stdout.strip()
                self.usage_stats[tier]["calls"] += 1
                self.usage_stats[tier]["in_chars"] += len(prompt)
                self.usage_stats[tier]["out_chars"] += len(output)
                self._log_call({
                    "ts": _now_iso(), "tier": tier, "model_id": model_id,
                    "key_index": key_index, "latency_s": latency,
                    "status": "ok_unparsed", "response_chars": len(output),
                })
                return output
        
        finally:
            # Clean up the temporary config directory
            shutil.rmtree(tmp_config_dir, ignore_errors=True)

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        system_prompt: Optional[str] = None,
        allow_fallback: bool = True,
        reasoning_enabled: bool = True,
    ) -> Optional[str]:
        """
        Invoke a Gemini model via CLI with recursive tier fallback.

        Note: The Gemini CLI does not expose max_tokens or temperature flags
        on the command line, so those knobs are intentionally absent here.
        Control output length via prompt wording.
        """
        if not self.available:
            return None

        # Check budget BEFORE incrementing (so retries don't burn it)
        if self._call_count >= self.max_calls:
            logger.warning(f"LLM call budget exhausted ({self._call_count}/{self.max_calls}). Skipping {tier}.")
            return None

        # Always try to start from the first non-exhausted key to prefer free tiers.
        # This ensures we don't 'stick' to a paid key (like Index 2) if a free one
        # might have reset its RPM (Requests Per Minute) quota.
        start_index = 0
        if self.track_hard_quotas:
            while start_index in self._exhausted_keys and start_index < len(self._api_keys) - 1:
                start_index += 1
        
        if self._current_key_index != start_index:
            logger.debug(f"Resetting key index from {self._current_key_index} to {start_index} to prefer free tiers.")
            self._current_key_index = start_index

        # Track attempts on the CURRENT key within this invoke() call.
        # This allows us to retry the same key multiple times before rotating.
        self._attempts_on_current_key = 0
        max_attempts_per_key = 3 

        model_id = self.models.get(tier, self.models["medium"])
        full_prompt = f"{system_prompt}\n\nUser Request: {prompt}" if system_prompt else prompt

        # Consistent, bounded retry policy across ALL tiers (no heavy 12-attempt
        # ultra-persistence). max_attempts_total caps the number of CLI calls
        # for this logical invocation across all keys.
        max_attempts_total = self.config_retries
        # Per-key soft-quota retries before rotating to the next key.
        max_soft_per_key = 2

        last_action = "retry"
        attempt = 0
        while attempt < max_attempts_total:
            attempt += 1
            try:
                result = self._execute_command(model_id, full_prompt, tier)
                if result:
                    self._call_count += 1
                    return result
                # Empty output is treated as a transient problem.
                logger.warning(
                    f"Gemini returned empty output (tier={tier}, attempt={attempt}/{max_attempts_total}); "
                    "treating as retryable."
                )
                last_action = "retry"
            except (subprocess.TimeoutExpired, ValueError) as e:
                logger.warning(
                    f"Gemini transient error (tier={tier}, attempt={attempt}/{max_attempts_total}): {e}"
                )
                last_action = "retry"
                if attempt < max_attempts_total:
                    time.sleep(self._retry_sleep(attempt))
                continue
            except subprocess.CalledProcessError as e:
                err = str(e)
                if e.stderr:
                    err += f" | stderr: {e.stderr.strip()}"
                action = classify_error(text=err)

                if action == "fallback":
                    # Out of usage (quota/balance/auth exhausted). Try rotating
                    # to a different key once (another key may still work), but
                    # if we've already rotated through everything, return None
                    # so the composite falls through to the next provider.
                    logger.warning(
                        f"Gemini out-of-usage (tier={tier}, attempt={attempt}/{max_attempts_total}): {err[:300]}"
                    )
                    last_action = "fallback"
                    if not self._try_rotate_for_new_key(err):
                        return None
                    if attempt < max_attempts_total:
                        continue
                    return None

                # Retryable (transient) error: rotate key after soft-quota
                # retries on the current key, else backoff.
                logger.warning(
                    f"Gemini retryable error (tier={tier}, attempt={attempt}/{max_attempts_total}): {err[:300]}"
                )
                last_action = "retry"
                self._attempts_on_current_key += 1
                if self._attempts_on_current_key >= max_soft_per_key:
                    self._attempts_on_current_key = 0
                    if self._rotate_key() and attempt < max_attempts_total:
                        continue
                if attempt < max_attempts_total:
                    time.sleep(self._retry_sleep(attempt))
                continue

        logger.error(f"All Gemini attempts for tier {tier} failed after {max_attempts_total} attempts (last action={last_action}).")

        # Recursive fallback for both timeout and other failures
        if allow_fallback:
            next_tier = {"heavy": "medium", "medium": "light"}.get(tier)
            if next_tier:
                logger.info(f"--- Falling back from {tier} to {next_tier} ---")
                return self.invoke(
                    prompt, tier=next_tier,
                    system_prompt=system_prompt,
                    allow_fallback=True,
                )

        return None

    def _retry_sleep(self, attempt: int) -> float:
        """Short bounded backoff (matches other providers), capped at RETRY_BACKOFF_MAX."""
        backoff = min(
            RETRY_BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 2),
            RETRY_BACKOFF_MAX,
        )
        return backoff

    def _try_rotate_for_new_key(self, err: str) -> bool:
        """Rotate to a fresh, non-exhausted key. Returns True if one is available.

        If the provider-wide error is a hard quota / payment failure on the
        account, rotating is pointless — return False so the caller falls back.
        """
        hard_look = (err or "").lower()
        if any(k in hard_look for k in ("insufficient balance", "billing", "payment", "forbidden")):
            logger.warning("Gemini provider-wide block (billing/auth) — not rotating keys.")
            return False
        return self._rotate_key()

    def get_key_rotation_rows(self):
        """Return per-key rotation rows for the unified provider summary.

        Each row: (provider, key_index, preview, success, failures).
        """
        rows = []
        for idx in sorted(self.key_usage_stats.keys()):
            stats = self.key_usage_stats[idx]
            rows.append((self.provider, idx, stats["preview"], stats["success"], stats["failure"]))
        return rows

    def get_usage_summary(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> str:
        """Generate a formatted markdown summary of Gemini API usage and estimated costs.

        Pricing constants and the cost formula live on the class (PRICING,
        CACHED_INPUT_DISCOUNT, _cost_for) so this summary and the per-call log
        agree to the cent. Cached input is billed at a discount and thinking
        ("thoughts") tokens at the output rate — see _cost_for."""
        total_cost = 0.0
        lines = ["\n---\n\n## Gemini Usage Summary\n\n"]
        lines.append("| Tier | Success | Failures | Input (fresh/cached) | Output (text/think) | Est. Cost |\n")
        lines.append("| :--- | :---: | :---: | :--- | :--- | :--- |\n")

        for tier in ["heavy", "medium", "light"]:
            stats = self.usage_stats[tier]
            if stats["calls"] == 0 and stats["failed_attempts"] == 0:
                continue

            # Estimate tokens from chars if the CLI didn't report any
            # (~3.5 chars/token for Gemini). Cached/thinking have no char proxy.
            in_tok = stats["in_tokens"] or int(stats["in_chars"] / 3.5)
            out_tok = stats["out_tokens"] or int(stats["out_chars"] / 3.5)
            cached_tok = stats.get("cached_tokens", 0)
            think_tok = stats.get("thought_tokens", 0)

            cost = self._cost_for(tier, in_tok, cached_tok, out_tok, think_tok)
            total_cost += cost

            lines.append(
                f"| {tier.capitalize()} | {stats['calls']} | {stats['failed_attempts']} | "
                f"{in_tok:,} / {cached_tok:,} | "
                f"{out_tok:,} / {think_tok:,} | "
                f"${cost:.4f} |\n"
            )

        if total_cost == 0 and sum(s["calls"] + s["failed_attempts"] for s in self.usage_stats.values()) == 0:
            return ""

        lines.append(
            f"| **Total** | **{sum(s['calls'] for s in self.usage_stats.values())}** | "
            f"**{sum(s['failed_attempts'] for s in self.usage_stats.values())}** | | | **${total_cost:.4f}** |\n\n"
        )

        # Add API Key Rotation Summary (suppressed when CompositeClient renders
        # a unified provider-aware table).
        if self.render_key_rotation and self.key_usage_stats:
            lines.append("### API Key Rotation Summary\n\n")
            lines.append("| Provider | Key Index | Preview | Success | Quota/Failures |\n")
            lines.append("| :--- | :---: | :--- | :---: | :---: |\n")
            for idx in sorted(self.key_usage_stats.keys()):
                stats = self.key_usage_stats[idx]
                lines.append(f"| {self.provider} | {idx} | `{stats['preview']}` | {stats['success']} | {stats['failure']} |\n")
            lines.append("\n")

        lines.append(
            f"*Note: Costs are estimated from Gemini paid-tier rates (audited May 2026): "
            f"Pro $2/$12, Flash $0.50/$3, Flash-Lite $0.25/$1.50 per 1M input/output tokens. "
            f"Thinking tokens bill at the output rate; cached input at "
            f"{int(self.CACHED_INPUT_DISCOUNT * 100)}% of the input rate. "
            f"Failed calls are not charged but represent retries/rotations.*\n\n"
        )
        
        # Add timing information if provided
        if start_time and end_time:
            from datetime import datetime
            duration = end_time - start_time
            start_str = datetime.fromtimestamp(start_time).strftime("%I:%M:%S %p")
            end_str = datetime.fromtimestamp(end_time).strftime("%I:%M:%S %p")
            
            # Format duration as H:M:S if over an hour, else M:S
            if duration >= 3600:
                h = int(duration // 3600)
                m = int((duration % 3600) // 60)
                s = int(duration % 60)
                duration_str = f"{h}h {m}m {s}s"
            else:
                m = int(duration // 60)
                s = int(duration % 60)
                duration_str = f"{m}m {s}s"
                
            lines.append(f"**Briefing generation took {duration_str}** (Started: {start_str}, Finished: {end_str})\n")

        return "".join(lines)
