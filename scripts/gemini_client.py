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
import time
from typing import Any, Dict, List, Optional
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_random_exponential, 
    retry_if_exception, 
    before_sleep_log
)

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
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

        self.max_calls = config.get("max_calls_per_run", 50)
        self.retry_wait_seconds = config.get("retry_wait_seconds", 61)
        self.key_swap_delay = config.get("key_swap_delay", 5)
        self._call_count = 0
        self._available = None
        self._api_keys = self._load_api_keys()
        self._current_key_index = 0

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

    def _execute_command(self, model_id: str, prompt: str, tier: str) -> str:
        """Execute the gemini command with a local config to force maxAttempts=1."""
        import tempfile
        import shutil
        from pathlib import Path

        # Create a temporary directory for the gemini config
        tmp_config_dir = tempfile.mkdtemp(prefix="atlas_gemini_config_")
        
        try:
            # Create .gemini/settings.json in the temp dir
            gemini_dir = Path(tmp_config_dir) / ".gemini"
            gemini_dir.mkdir(parents=True, exist_ok=True)
            settings_path = gemini_dir / "settings.json"
            
            # CRITICAL: Force maxAttempts to the configured value so Python Tenacity controls the retries
            # Also set a shorter connection timeout to fail fast and retry at our level
            with open(settings_path, "w") as f:
                json.dump({
                    "general": {"maxAttempts": self.internal_max_attempts, "requestTimeout": 120000},
                    "tools": {"autoAccept": True}
                }, f)

            # Use standard environment but override the config directory
            process_env = os.environ.copy()
            process_env["GEMINI_CONFIG_DIR"] = tmp_config_dir
            
            # Ensure compatibility by setting both common API key environment variables
            if tier == "heavy":
                api_key = self._get_current_key()
                key_index = self._current_key_index
            else:
                # For non-heavy tiers, always stick to the first API key
                api_key = self._api_keys[0] if self._api_keys else None
                key_index = 0

            if api_key:
                process_env["GEMINI_API_KEY"] = api_key
                key_preview = api_key[:6] + "..." + api_key[-4:]
                logger.debug(f"Using API Key index {key_index} for tier {tier}: {key_preview}")
            else:
                logger.warning(f"No Gemini API key available for tier {tier}!")

            # CRITICAL: Force the CLI to use the API key by masking system-wide auth
            # This prevents fallback to local gcloud/ADC credentials or OAuth
            process_env["GOOGLE_API_KEY"] = ""  # Explicitly clear any inherited Google key
            process_env["GOOGLE_APPLICATION_CREDENTIALS"] = ""
            process_env["CLOUDSDK_AUTH_ACCESS_TOKEN"] = ""
            process_env["HOME"] = tmp_config_dir  # Prevent looking up ~/.config or ~/.gemini
            process_env["GEMINI_CLI_TRUST_WORKSPACE"] = "true" # Ensure it runs in headless mode
            logger.debug("Strict Auth: Masked system-wide Google Cloud auth (ADC/OAuth), redirected HOME, and cleared GOOGLE_API_KEY.")

            # Add a small initial delay for the heavy tier to avoid hitting RPM limits
            if tier == "heavy":
                time.sleep(1)
            
            logger.info(f"Invoking Gemini model: {model_id} (tier: {tier})")

            cmd = [
                "gemini", "--model", model_id, "--prompt", prompt,
                "--approval-mode", "yolo", "--raw-output", "--accept-raw-output-risk",
                "--output-format", "json"
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=900, env=process_env)
            except Exception as e:
                self.usage_stats[tier]["failed_attempts"] += 1
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
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> Optional[str]:
        """
        Invoke a Gemini model via CLI with recursive tier fallback.
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
                        prompt, tier=next_tier, max_tokens=max_tokens,
                        temperature=temperature, system_prompt=system_prompt,
                        allow_fallback=True
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
