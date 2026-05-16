#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Gemini CLI model client.

Provides tiered model access to Gemini CLI for intelligence features.
Uses subprocess to call the 'gemini' command.
"""

import json
import logging
import os
import random
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_random_exponential, 
    retry_if_exception, 
    before_sleep_log
)

logger = logging.getLogger(__name__)


class GeminiCLIClient:
    """Client for Gemini CLI model inference with tiered model support."""

    # Default model IDs for each tier based on gemini-cli help
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
        
        # Model IDs per tier
        models_config = config.get("models", {})
        self.models = {
            "heavy": models_config.get("heavy", self.DEFAULT_MODELS["heavy"]),
            "medium": models_config.get("medium", self.DEFAULT_MODELS["medium"]),
            "light": models_config.get("light", self.DEFAULT_MODELS["light"]),
        }
        # Per-tier alternate models to try BEFORE falling to the next tier
        # when the primary model exhausts its retries. Each entry on this
        # list has its own per-day RPD quota, so when gemini-2.5-pro is
        # exhausted for the day, gemini-2.0-pro and gemini-1.5-pro may
        # still have headroom. List order = try order.
        fallbacks_config = config.get("fallback_models", {}) or {}
        self.fallback_models = {
            "heavy": list(fallbacks_config.get("heavy", [])),
            "medium": list(fallbacks_config.get("medium", [])),
            "light": list(fallbacks_config.get("light", [])),
        }

        self.max_calls = config.get("max_calls_per_run", 50)
        self.retry_wait_seconds = config.get("retry_wait_seconds", 61)
        self.key_swap_delay = config.get("key_swap_delay", 5)
        self._call_count = 0
        self._available = None
        self._api_keys = self._load_api_keys()
        self._current_key_index = 0
        # Cursor for per-call round-robin key selection on the heavy tier when
        # multiple keys are loaded. Distinct from _current_key_index (which
        # tracks the most recent rotation in response to an observed 429).
        self._next_heavy_key = 0
        # Cached per-instance config dir (written once, reused across calls).
        self._config_dir: Optional[str] = None

        # Per-tier minimum interval (seconds) between successive calls. Sized
        # for free-tier RPM caps: heavy=Pro is the tightest at ~2 RPM, medium
        # and light are Flash variants with looser caps. Override via config
        # under gemini.tier_min_interval_seconds.
        intervals = config.get("tier_min_interval_seconds", {}) or {}
        self.tier_min_interval = {
            "heavy": float(intervals.get("heavy", 30.0)),
            "medium": float(intervals.get("medium", 5.0)),
            "light": float(intervals.get("light", 2.0)),
        }
        self._tier_last_call: Dict[str, float] = {"heavy": 0.0, "medium": 0.0, "light": 0.0}
        self._call_lock = threading.Lock()
        # How many consecutive quota strikes we'll absorb on the SAME heavy
        # key before rotating to the next one in the pool. Tenacity's
        # random_exponential wait already spaces these attempts out
        # (90-450s on heavy), so 3 strikes ≈ 5-22 min of patient retries on
        # a single key before we move on. Override via gemini.max_strikes_per_key.
        self.max_strikes_per_key = int(config.get("max_strikes_per_key", 3))

        # Heavy-tier retry budget. 20 attempts × ~90-450s backoff ≈ 2-3 hours
        # of patient retrying — appropriate for an unattended cron run.
        self.heavy_max_attempts = int(config.get("heavy_max_attempts", 20))

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
            
        # 2. Check for suffixed variants, sorted by environment variable name
        suffixed_vars = []
        for var_name in os.environ:
            if var_name.startswith("GEMINI_API_KEY_"):
                # Skip the primary one we already handled
                if var_name != "GEMINI_API_KEY":
                    suffixed_vars.append(var_name)
        
        for var_name in sorted(suffixed_vars):
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
        """Check if Gemini CLI is available and enabled."""
        if self._available is not None:
            return self._available
        if not self.enabled:
            self._available = False
            return False
        
        try:
            # Check if 'gemini' command exists
            subprocess.run(["which", "gemini"], capture_output=True, check=True)
            self._available = True
        except subprocess.CalledProcessError:
            logger.warning("gemini-cli not found in PATH. Gemini features disabled.")
            self._available = False
        
        return self._available

    def _pace_tier(self, tier: str) -> None:
        """Sleep so successive calls to `tier` honor its minimum interval.

        Computes the wait under a short lock (and reserves the next slot by
        writing _tier_last_call to the future intent time), then sleeps
        outside the lock so concurrent callers on other tiers aren't blocked
        by this tier's wait. Set tier_min_interval_seconds.<tier>=0 in
        config to disable.
        """
        min_interval = self.tier_min_interval.get(tier, 0.0)
        if min_interval <= 0:
            return
        with self._call_lock:
            now = time.time()
            elapsed = now - self._tier_last_call[tier]
            wait = max(0.0, min_interval - elapsed)
            # Reserve the slot at intent time before releasing the lock so
            # any other caller racing into this tier sees us as the most
            # recent and waits behind us.
            self._tier_last_call[tier] = now + wait
        if wait > 0:
            logger.info(
                f"Pacing {tier} tier: sleeping {wait:.1f}s to honor {min_interval:.0f}s min interval"
            )
            time.sleep(wait)

    def _ensure_config_dir(self) -> str:
        """Build (once) and return a config dir holding .gemini/settings.json."""
        import tempfile
        from pathlib import Path

        if self._config_dir is not None and os.path.isdir(self._config_dir):
            return self._config_dir

        tmp_config_dir = tempfile.mkdtemp(prefix="atlas_gemini_config_")
        gemini_dir = Path(tmp_config_dir) / ".gemini"
        gemini_dir.mkdir(parents=True, exist_ok=True)
        settings_path = gemini_dir / "settings.json"
        # Force the CLI's internal retry count so Tenacity (Python-side) owns
        # retries instead. `general.maxAttempts` is the only documented and
        # accepted key here in current gemini-cli (cap is 10).
        # `tools.autoAccept` and `general.requestTimeout` are NOT recognized
        # keys — `--approval-mode yolo` (CLI flag) handles auto-approval, and
        # the Python subprocess timeout enforces the call ceiling.
        with open(settings_path, "w") as f:
            json.dump({
                "general": {"maxAttempts": self.internal_max_attempts},
            }, f)
        self._config_dir = tmp_config_dir
        return tmp_config_dir

    def _execute_command(self, model_id: str, prompt: str, tier: str) -> str:
        """Execute the gemini command with a cached local config."""
        tmp_config_dir = self._ensure_config_dir()

        # Use standard environment but override the config directory
        process_env = os.environ.copy()
        process_env["GEMINI_CONFIG_DIR"] = tmp_config_dir

        # Pick the API key. For heavy with multiple keys, _next_heavy_key is
        # the "currently sticky" key — it does NOT auto-advance on every
        # call. invoke()'s tenacity loop is responsible for deciding when to
        # rotate (after max_strikes_per_key consecutive quota errors), which
        # gives the same key several chances to recover through backoff
        # before we move on. Non-heavy tiers stay on key 0 (Flash RPD is huge).
        if tier == "heavy" and len(self._api_keys) > 1:
            with self._call_lock:
                key_index = self._next_heavy_key
                api_key = self._api_keys[key_index]
        elif tier == "heavy":
            api_key = self._get_current_key()
            key_index = self._current_key_index
        else:
            api_key = self._api_keys[0] if self._api_keys else None
            key_index = 0
        # Remember which key this attempt used (for diagnostic logging in
        # is_transient_error).
        self._last_used_key_idx = key_index

        if api_key:
            process_env["GEMINI_API_KEY"] = api_key
            key_preview = api_key[:6] + "..." + api_key[-4:]
            logger.debug(f"Using API Key index {key_index} for tier {tier}: {key_preview}")
        else:
            logger.warning(f"No Gemini API key available for tier {tier}!")

        # CRITICAL: Force the CLI to use GEMINI_API_KEY by removing higher-
        # precedence auth env vars. GOOGLE_API_KEY normally wins over
        # GEMINI_API_KEY in google-genai's auth resolution, so its presence
        # would shadow our key. Pop instead of setting "" — empty-string env
        # vars are still "set" and some auth backends treat them differently
        # from missing.
        for var in (
            "GOOGLE_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "CLOUDSDK_AUTH_ACCESS_TOKEN",
        ):
            process_env.pop(var, None)
        # Redirect HOME so gemini-cli doesn't pick up ~/.gemini settings or
        # ~/.config OAuth caches we don't want.
        process_env["HOME"] = tmp_config_dir
        logger.debug("Strict Auth: removed Google Cloud auth env vars, redirected HOME.")

        logger.info(f"Invoking Gemini model: {model_id} (tier: {tier})")

        # `--skip-trust` is the documented replacement for the workspace-trust
        # prompt that would otherwise block headless mode. (The
        # GEMINI_CLI_TRUST_WORKSPACE env var is not recognized.)
        cmd = [
            "gemini",
            "--model", model_id,
            "--prompt", prompt,
            "--approval-mode", "yolo",
            "--raw-output", "--accept-raw-output-risk",
            "--output-format", "json",
            "--skip-trust",
        ]

        # Proactively honor per-tier RPM cap before spending the call.
        self._pace_tier(tier)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=900, env=process_env)
        except Exception as e:
            self.usage_stats[tier]["failed_attempts"] += 1
            raise e

        try:
            # Parse JSON output from gemini-cli. The current schema is:
            # {
            #   "response": str,
            #   "stats": {"models": {"<model-id>": {
            #     "api": {...},
            #     "tokens": {"prompt": int, "candidates": int, "total": int,
            #                "cached": int, "thoughts": int, "tool": int}
            #   }}}
            # }
            data = json.loads(result.stdout)
            output = data.get("response", "").strip()

            # Find the per-model stats block. The model_id we passed in
            # ("pro", "flash", "flash-lite") is normalized by gemini-cli to
            # the full id like "gemini-2.5-pro" — so substring-match either
            # direction.
            stats = data.get("stats", {}).get("models", {})
            model_stats = {}
            for k, v in stats.items():
                if model_id in k or k in model_id:
                    model_stats = v.get("tokens", {})
                    break
            if not model_stats and stats:
                model_stats = next(iter(stats.values())).get("tokens", {})

            # Update usage metrics. Use the documented "prompt" / "candidates"
            # fields. Fall back to "input" only as a defensive measure for
            # older gemini-cli versions.
            self.usage_stats[tier]["calls"] += 1
            in_tokens = model_stats.get("prompt", 0) or model_stats.get("input", 0)
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
            raw_stdout = (result.stdout or "").strip()
            stdout_preview = raw_stdout[:400].replace("\n", " ")
            stderr_preview = (result.stderr or "").strip()[:400].replace("\n", " ")
            logger.warning(
                f"Failed to parse JSON response from gemini-cli (tier={tier}, "
                f"model={model_id}): {type(e).__name__}: {e}. "
                f"stdout[0:400]={stdout_preview!r}; stderr[0:400]={stderr_preview!r}"
            )
            output = raw_stdout
            self.usage_stats[tier]["calls"] += 1
            self.usage_stats[tier]["in_chars"] += len(prompt)
            self.usage_stats[tier]["out_chars"] += len(output)
            return output

    def cleanup(self) -> None:
        """Remove the cached config directory. Safe to call multiple times."""
        import shutil
        if self._config_dir and os.path.isdir(self._config_dir):
            shutil.rmtree(self._config_dir, ignore_errors=True)
        self._config_dir = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def invoke(
        self,
        prompt: str,
        tier: str = "medium",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> Optional[str]:
        """
        Invoke a Gemini model via CLI with recursive tier fallback.

        Note: `max_tokens` and `temperature` are accepted for interface
        compatibility with BedrockClient but are NOT currently forwarded
        to gemini-cli. The CLI's per-call generation parameters live
        under modelConfig.generateContentConfig in settings.json, which
        we cache once per client instance, so per-call overrides would
        require rewriting the settings file before each subprocess.run.
        gemini-cli falls back to the model's default output cap
        (8,192 tokens for Pro/Flash) — fine for our prompt sizes.
        """
        if not self.available:
            return None

        # Cheap fast-path budget check (no lock). The cap is for cost control,
        # so a small slop on the boundary under heavy concurrency is fine —
        # the precise increment under the lock is on the success path below.
        if self._call_count >= self.max_calls:
            logger.warning(
                f"LLM call budget exhausted ({self._call_count}/{self.max_calls}). Skipping {tier}."
            )
            return None

        # Build the candidate-model list for this tier: primary first, then any
        # configured fallback_models[tier] (de-duplicated, primary not repeated).
        # Each entry is its own model id with its own per-day RPD on Gemini
        # free tier, so when gemini-2.5-pro is RPD'd we still have a shot at
        # gemini-2.0-pro or gemini-1.5-pro before falling to medium.
        primary_model = self.models.get(tier, self.models["medium"])
        candidates = [primary_model]
        for alt in self.fallback_models.get(tier, []):
            if alt and alt not in candidates:
                candidates.append(alt)

        full_prompt = f"{system_prompt}\n\nUser Request: {prompt}" if system_prompt else prompt

        # Ultra-persistence for the Heavy (Pro) model. With min=90s and
        # max=450s exponential backoff, 20 attempts can span ~2-3 hours of
        # patient retries — well-suited for an unattended early-morning run
        # that prefers eventual success over fast failure. Override via
        # gemini.heavy_max_attempts in config.
        max_attempts = (
            int(self.heavy_max_attempts) if tier == "heavy" else 4
        )

        # Cap retries on subprocess hangs separately. Each TimeoutExpired
        # already cost the full 900s subprocess timeout — letting it consume
        # the full heavy_max_attempts budget would mean up to 20*15min=5h on
        # network black-holing alone, with no chance of recovery.
        max_timeout_retries = 3
        timeout_retry_state = {"count": 0}
        # Sticky-key persistence: we stay on the same heavy key for up to
        # max_strikes_per_key consecutive quota errors before rotating to
        # the next one. After we've rotated through every loaded key
        # (key_rotations >= len(api_keys)), the model is considered done
        # for this call and we fall through to alternate models / next
        # tier. No permanent blacklist — same key is freely tried again
        # by future invoke() calls.
        strike_state = {"strikes_on_current_key": 0, "key_rotations": 0}
        max_key_rotations = (
            len(self._api_keys) if (tier == "heavy" and len(self._api_keys) > 1) else 1
        )

        def _trim_error(msg: str, head: int = 80, tail: int = 280) -> str:
            """Capture both ends of the error. Real reason is usually at the
            tail; head helps identify the source (which model/key)."""
            if len(msg) <= head + tail:
                return msg
            return f"{msg[:head]} ... {msg[-tail:]}"

        def is_transient_error(exception):
            """Decide whether to retry; rotate key on quota when appropriate."""
            if isinstance(exception, RuntimeError) and "exhausted" in str(exception).lower():
                # _execute_command's "all keys exhausted" signal: don't retry,
                # just propagate so the model/tier fallback takes over.
                return False
            if isinstance(exception, subprocess.TimeoutExpired):
                if timeout_retry_state["count"] >= max_timeout_retries:
                    logger.warning(
                        f"TimeoutExpired retry budget ({max_timeout_retries}) exhausted; aborting."
                    )
                    return False
                timeout_retry_state["count"] += 1
                logger.info(
                    f"Subprocess timeout {timeout_retry_state['count']}/{max_timeout_retries}; will retry."
                )
                return True
            if isinstance(exception, ValueError):
                # Only the explicit "Empty response from <tier>" ValueError is
                # treated as transient (model occasionally returns empty under
                # safety filtering or spurious cutoffs). Any other ValueError
                # is a programming bug and should surface, not loop.
                msg = str(exception).lower()
                return msg.startswith("empty response from")
            if isinstance(exception, subprocess.CalledProcessError):
                # Capture both stderr and stdout. CLI startup warnings often
                # land in stdout/early stderr while the real error message
                # arrives at the tail, so retain both ends when logging.
                error_msg = ""
                if exception.stderr:
                    error_msg += exception.stderr
                if exception.stdout:
                    error_msg += exception.stdout

                if not error_msg:
                    error_msg = str(exception)

                lower_msg = error_msg.lower()

                # Keywords for typical transient errors (network + quota)
                network_keywords = ["fetch failed", "connection", "econnrefused", "econnreset", "etimedout", "enetunreach", "socket hang up"]
                quota_keywords = ["resource_exhausted", "capacity", "rate limit", "429", "503", "500", "exhausted", "quota"]
                # Keywords that usually mean a hard daily stop. "terminalquotaerror"
                # is the explicit class name gemini-cli emits when the API
                # returns code=8 RESOURCE_EXHAUSTED with isTerminal=true.
                hard_quota_keywords = ["daily", "rpd", "limit reached", "quota exceeded", "terminalquotaerror"]

                if any(kw in lower_msg for kw in network_keywords):
                    logger.info(f"Network error detected for tier {tier}. Retrying...")
                    return True

                is_hard_quota = any(kw in lower_msg for kw in hard_quota_keywords)
                is_soft_quota = any(kw in lower_msg for kw in quota_keywords)
                is_quota = is_hard_quota or is_soft_quota

                if not is_quota:
                    return False

                # Quota response (soft 429/capacity or hard daily). Same
                # logic for both: stay on this key for up to
                # max_strikes_per_key consecutive strikes (tenacity's
                # exponential backoff between them gives the key a chance
                # to recover — could be a transient RPM burst, RPM window
                # rolling over, or even a Google-side service blip
                # masquerading as 'quota'). Once we've struck out enough
                # times on the same key, rotate to the next one.
                #
                # `ignore_hard_quota=true` would also fall into this path
                # since hard-quota responses are treated as a strike, not
                # a hard abort. The only way to give up is to exhaust all
                # max_key_rotations (= number of keys for multi-key heavy,
                # else 1) — meaning we tried every key max_strikes_per_key
                # times and still nothing worked.
                last_idx = getattr(self, "_last_used_key_idx", None)
                strike_state["strikes_on_current_key"] += 1
                quota_label = "hard quota" if is_hard_quota else "soft quota"

                if strike_state["strikes_on_current_key"] < self.max_strikes_per_key:
                    logger.info(
                        f"{quota_label.capitalize()} on tier {tier} key idx={last_idx} "
                        f"(strike {strike_state['strikes_on_current_key']}/"
                        f"{self.max_strikes_per_key} on this key; rotation "
                        f"{strike_state['key_rotations'] + 1}/"
                        f"{max_key_rotations}); will retry after backoff. msg: "
                        f"{_trim_error(error_msg)}"
                    )
                    return True

                # Strikes-on-current-key exhausted. Rotate to next key (if
                # we have one) and reset the per-key strike counter.
                if tier == "heavy" and len(self._api_keys) > 1:
                    with self._call_lock:
                        self._next_heavy_key = (self._next_heavy_key + 1) % len(self._api_keys)
                    strike_state["key_rotations"] += 1
                    strike_state["strikes_on_current_key"] = 0
                    if strike_state["key_rotations"] >= max_key_rotations:
                        logger.warning(
                            f"Tier {tier}: rotated through all "
                            f"{max_key_rotations} keys, each took "
                            f"{self.max_strikes_per_key} strikes. "
                            f"Giving up on this model. msg: {_trim_error(error_msg)}"
                        )
                        return False
                    logger.info(
                        f"Tier {tier}: {quota_label} strikes on key idx={last_idx} "
                        f"reached {self.max_strikes_per_key}; rotating to next key "
                        f"(rotation {strike_state['key_rotations']}/{max_key_rotations})."
                    )
                    return True

                # Single-key heavy or non-heavy: no other key to rotate to.
                # We've already given the one key max_strikes_per_key chances
                # spread across tenacity backoff. Give up on this model.
                logger.warning(
                    f"Tier {tier}: {quota_label} strikes on single key "
                    f"reached {self.max_strikes_per_key}. Giving up on this "
                    f"model. msg: {_trim_error(error_msg)}"
                )
                return False
            return False

        # Walk the same-tier candidate models. Each model gets its own retry
        # budget (timeout + strike state). Reset the per-call counters
        # between candidates so a quota-exhausted gemini-2.5-pro doesn't
        # poison the budget for gemini-2.0-pro.
        last_exception: Optional[Exception] = None
        for model_id in candidates:
            timeout_retry_state["count"] = 0
            strike_state["strikes_on_current_key"] = 0
            strike_state["key_rotations"] = 0
            if model_id != primary_model:
                logger.info(
                    f"--- Trying alternate {tier}-tier model: {model_id} ---"
                )
            try:
                wait_strategy = wait_random_exponential(
                    multiplier=60 if tier == "heavy" else 30,
                    min=90 if tier == "heavy" else 60,
                    max=450 if tier == "heavy" else 300,
                )

                @retry(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_strategy,
                    retry=retry_if_exception(is_transient_error),
                    before_sleep=before_sleep_log(logger, logging.INFO),
                    reraise=True,
                )
                def retry_call():
                    return self._execute_command(model_id, full_prompt, tier)

                result = retry_call()
                if result:
                    with self._call_lock:
                        self._call_count += 1
                    return result
            except Exception as e:
                last_exception = e
                error_detail = str(e)
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    error_detail += f" | stderr: {e.stderr.strip()}"
                logger.error(
                    f"Model {model_id} (tier {tier}) failed after retries: "
                    f"{error_detail}"
                )
                # Continue to next candidate model in this tier.
                continue

        # Every same-tier model exhausted. Recurse into the next tier.
        if allow_fallback:
            next_tier = {"heavy": "medium", "medium": "light"}.get(tier)
            if next_tier:
                logger.info(f"--- Falling back from {tier} to {next_tier} ---")
                return self.invoke(
                    prompt, tier=next_tier, max_tokens=max_tokens,
                    temperature=temperature, system_prompt=system_prompt,
                    allow_fallback=True,
                )

        return None

    def get_usage_summary(self) -> str:
        """Generate a formatted markdown summary of Gemini API usage and estimated costs."""
        # Cost constants (Gemini 1.5 Pay-as-you-go pricing)
        # Pro: $1.25/1M in, $3.75/1M out
        # Flash: $0.075/1M in, $0.30/1M out
        PRICING = {
            "heavy": {"in": 1.25 / 1_000_000, "out": 3.75 / 1_000_000},
            "medium": {"in": 0.075 / 1_000_000, "out": 0.30 / 1_000_000},
            "light": {"in": 0.075 / 1_000_000, "out": 0.30 / 1_000_000}, # Same as flash
        }

        total_cost = 0.0
        lines = ["\n---\n\n## Gemini Usage Summary\n\n"]
        lines.append("| Tier | Success | Failures | Input (Tok/Char) | Output (Tok/Char) | Est. Cost |\n")
        lines.append("| :--- | :---: | :---: | :--- | :--- | :--- |\n")

        for tier in ["heavy", "medium", "light"]:
            stats = self.usage_stats[tier]
            if stats["calls"] == 0 and stats["failed_attempts"] == 0:
                continue
            
            # Estimate tokens if they were missing (4 chars per token)
            in_tok = stats["in_tokens"] or (stats["in_chars"] // 4)
            out_tok = stats["out_tokens"] or (stats["out_chars"] // 4)
            
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
        lines.append(f"*Note: Costs are estimated based on Gemini 1.5 Pay-as-you-go pricing. Failed calls are not charged but represent retries/rotations.*\n")
        
        return "".join(lines)
