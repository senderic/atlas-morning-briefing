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

import json
import logging
import os
import random
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
    before_sleep_log
)

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


class GeminiCLIClient:
    """Tiered LLM CLI client (auto-detects between `agy` and `gemini`)."""

    # Default model IDs for each tier — used by both agy and gemini.
    DEFAULT_MODELS = {
        "heavy": "pro",
        "medium": "flash",
        "light": "flash-lite",
    }

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
        self.ignore_hard_quota = config.get("ignore_hard_quota", False)
        self.internal_max_attempts = config.get("internal_max_attempts", 1)

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

        # Key usage tracking
        self.key_usage_stats = {
            i: {"success": 0, "failure": 0, "preview": (k[:6] + "..." + k[-4:]) if k else "None"}
            for i, k in enumerate(self._api_keys)
        }

        # Usage tracking
        self.usage_stats = {
            "heavy": {"calls": 0, "failed_attempts": 0, "in_tokens": 0, "out_tokens": 0, "in_chars": 0, "out_chars": 0},
            "medium": {"calls": 0, "failed_attempts": 0, "in_tokens": 0, "out_tokens": 0, "in_chars": 0, "out_chars": 0},
            "light": {"calls": 0, "failed_attempts": 0, "in_tokens": 0, "out_tokens": 0, "in_chars": 0, "out_chars": 0},
        }

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
        
        logger.info(f"🔄 ROTATING API KEY: Moving from index {self._current_key_index} to {(self._current_key_index + 1) % len(self._api_keys)}")
        
        self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
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
        from pathlib import Path

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
                raise e
            
            try:
                # Parse JSON output from gemini-cli
                data = json.loads(result.stdout)
                output = data.get("response", "").strip()
                
                # Extract stats if available
                stats = data.get("stats", {}).get("models", {})
                model_stats = {}
                for k, v in stats.items():
                    if model_id in k or k in model_id:
                        model_stats = v.get("tokens", {})
                        break
                if not model_stats and stats:
                    model_stats = next(iter(stats.values())).get("tokens", {})

                # Update usage metrics
                self.usage_stats[tier]["calls"] += 1
                in_tokens = model_stats.get("input", 0) or model_stats.get("prompt", 0)
                out_tokens = model_stats.get("candidates", 0)
                
                self.usage_stats[tier]["in_tokens"] += in_tokens
                self.usage_stats[tier]["out_tokens"] += out_tokens
                
                self.usage_stats[tier]["in_chars"] += len(prompt)
                self.usage_stats[tier]["out_chars"] += len(output)

                if not output:
                    raise ValueError(f"Empty response from {tier}")
                    
                logger.info(f"Gemini response received ({len(output)} chars, {out_tokens} tokens) from {tier}")
                return output

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse JSON response from gemini-cli: {e}")
                output = result.stdout.strip()
                self.usage_stats[tier]["calls"] += 1
                self.usage_stats[tier]["in_chars"] += len(prompt)
                self.usage_stats[tier]["out_chars"] += len(output)
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

        model_id = self.models.get(tier, self.models["medium"])
        full_prompt = f"{system_prompt}\n\nUser Request: {prompt}" if system_prompt else prompt

        # Ultra-persistence for the Heavy (Pro) model
        # 12 attempts with longer wait can span ~1 hour
        max_attempts = 12 if tier == "heavy" else 4

        def is_transient_error(exception):
            """Check if the error is worth retrying, and rotate key if quota hit."""
            if isinstance(exception, (subprocess.TimeoutExpired, ValueError)):
                return True
            if isinstance(exception, subprocess.CalledProcessError):
                # Check both stderr and stdout for error messages
                error_msg = ""
                if exception.stderr:
                    error_msg += exception.stderr
                if exception.stdout:
                    error_msg += exception.stdout
                
                if not error_msg:
                    error_msg = str(exception)
                
                error_msg = error_msg.lower()
                
                # Keywords for typical transient errors
                quota_keywords = ["resource_exhausted", "capacity", "rate limit", "429", "503", "500", "exhausted", "quota"]
                # Keywords that usually mean a hard daily stop
                hard_quota_keywords = ["daily", "rpd", "limit reached", "quota exceeded"]
                
                is_quota = any(kw in error_msg for kw in quota_keywords) or any(kw in error_msg for kw in hard_quota_keywords)
                
                if is_quota:
                    if self._rotate_key():
                        logger.info(f"Quota error detected for tier {tier}. Rotated API key and retrying...")
                        return True
                    
                    if any(kw in error_msg for kw in hard_quota_keywords):
                        if self.ignore_hard_quota:
                            logger.info(f"Potential hard quota hit for {tier}, but ignore_hard_quota is enabled. Retrying with current key...")
                            return True
                        logger.warning(f"Hard quota reached for {tier} and no more keys to rotate. Skipping retries.")
                        return False
                
                return any(kw in error_msg for kw in quota_keywords)
            return False

        try:
            # Smarter Wait Strategy:
            # For Heavy tier, we start with a 90s floor to clear 60s RPM windows.
            # Max wait of 450s (7.5m) to clear larger sliding windows.
            wait_strategy = wait_random_exponential(
                multiplier=60 if tier == "heavy" else 30, 
                min=90 if tier == "heavy" else 60, 
                max=450 if tier == "heavy" else 300
            )

            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_strategy,
                retry=retry_if_exception(is_transient_error),
                before_sleep=before_sleep_log(logger, logging.INFO),
                reraise=True
            )
            def retry_call():
                return self._execute_command(model_id, full_prompt, tier)

            # Successful logical call - increment budget ONCE
            result = retry_call()
            if result:
                self._call_count += 1
            return result

        except Exception as e:
            # Capture more detail in the error log
            error_detail = str(e)
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                error_detail += f" | stderr: {e.stderr.strip()}"
            
            logger.error(f"All attempts for tier {tier} failed after {max_attempts-1} retries: {error_detail}")
            
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

    def get_usage_summary(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> str:
        """Generate a formatted markdown summary of Gemini API usage and estimated costs."""
        # Cost constants — Gemini paid-tier rates, audited 2026-05-22 against
        # https://ai.google.dev/gemini-api/docs/pricing. CLI aliases resolve as
        # verified by scripts/audit_gemini.py (logs/gemini-audit-20260522-225038.txt):
        #   heavy  -> "pro"        -> gemini-3.x-pro-preview ($2/$12 for prompts
        #             <=200k; $4/$18 above). PAID-TIER ONLY — free-tier keys 429
        #             with "exhausted capacity", so in practice this row stays $0.
        #             The exact alias (3-pro vs 3.1-pro) can't be probed on a free
        #             key, but both share the <=200k rate and briefing prompts are
        #             well under 200k, so the figure holds either way.
        #   medium -> "flash"      -> gemini-3-flash-preview  ($0.50 in / $3.00 out)
        #   light  -> "flash-lite" -> gemini-3.1-flash-lite   ($0.25 in / $1.50 out)
        PRICING = {
            "heavy": {"in": 2.00 / 1_000_000, "out": 12.00 / 1_000_000},
            "medium": {"in": 0.50 / 1_000_000, "out": 3.00 / 1_000_000},
            "light": {"in": 0.25 / 1_000_000, "out": 1.50 / 1_000_000},
        }

        total_cost = 0.0
        lines = ["\n---\n\n## Gemini Usage Summary\n\n"]
        lines.append("| Tier | Success | Failures | Input (Tok/Char) | Output (Tok/Char) | Est. Cost |\n")
        lines.append("| :--- | :---: | :---: | :--- | :--- | :--- |\n")

        for tier in ["heavy", "medium", "light"]:
            stats = self.usage_stats[tier]
            if stats["calls"] == 0 and stats["failed_attempts"] == 0:
                continue
            
            # Estimate tokens if they were missing (3.5 chars per token for Gemini 1.5+)
            in_tok = stats["in_tokens"] or int(stats["in_chars"] / 3.5)
            out_tok = stats["out_tokens"] or int(stats["out_chars"] / 3.5)
            
            cost = (in_tok * PRICING[tier]["in"]) + (out_tok * PRICING[tier]["out"])
            total_cost += cost
            
            lines.append(
                f"| {tier.capitalize()} | {stats['calls']} | {stats['failed_attempts']} | "
                f"{in_tok:,} / {stats['in_chars']:,} | "
                f"{out_tok:,} / {stats['out_chars']:,} | "
                f"${cost:.4f} |\n"
            )

        if total_cost == 0 and sum(s["calls"] + s["failed_attempts"] for s in self.usage_stats.values()) == 0:
            return ""

        lines.append(
            f"| **Total** | **{sum(s['calls'] for s in self.usage_stats.values())}** | "
            f"**{sum(s['failed_attempts'] for s in self.usage_stats.values())}** | | | **${total_cost:.4f}** |\n\n"
        )

        # Add API Key Rotation Summary
        if self.key_usage_stats:
            lines.append("### API Key Rotation Summary\n\n")
            lines.append("| Key Index | Preview | Success | Quota/Failures |\n")
            lines.append("| :--- | :--- | :---: | :---: |\n")
            for idx in sorted(self.key_usage_stats.keys()):
                stats = self.key_usage_stats[idx]
                lines.append(f"| {idx} | `{stats['preview']}` | {stats['success']} | {stats['failure']} |\n")
            lines.append("\n")

        lines.append(f"*Note: Costs are estimated from Gemini paid-tier rates (audited May 2026): Pro $2/$12, Flash $0.50/$3, Flash-Lite $0.25/$1.50 per 1M input/output tokens. Failed calls are not charged but represent retries/rotations.*\n\n")
        
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
