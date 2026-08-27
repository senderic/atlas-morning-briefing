#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
5:45 AM pre-flight model availability check.

Runs concurrently (thread pool) against all free models to determine
which are responsive before the 6:00 AM briefing run. Writes results to
.model-availability.json for the briefing runner to consume.

Also tests reasoning control (reasoning_enabled=False) for each model
and detects CoT leakage, which is treated as a failure triggering fallback.
"""

import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.llm_client import get_model_capabilities, ReasoningControlMixin

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Test prompt - minimal tokens
TEST_PROMPT = "Reply with OK if you can respond."
TEST_TIMEOUT = 15  # seconds per model
MAX_WORKERS = 8

# Free model definitions per provider and tier
FREE_MODELS = {
    "opencode": {
        "heavy": {
            "primary": "opencode/nemotron-3-ultra-free",
            "fallbacks": ["opencode/mimo-v2.5-free", "opencode/nemotron-3.5-lightning-free"],
        },
        "medium": {
            "primary": "opencode/deepseek-v4-flash-free",
            "fallbacks": ["opencode/mimo-v2.5-free", "opencode/nemotron-3.5-lightning-free"],
        },
        "light": {
            "primary": "opencode/deepseek-v4-flash-free",
            "fallbacks": ["opencode/mimo-v2.5-free", "opencode/nemotron-3.5-lightning-free"],
        },
    },
    "openrouter": {
        "heavy": {
            "primary": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
            "fallbacks": ["openrouter/z-ai/glm-5.2:free", "openrouter/openrouter/free"],
        },
        "medium": {
            "primary": "openrouter/minimax/minimax-m3:free",
            "fallbacks": ["openrouter/z-ai/glm-5.2:free", "openrouter/openrouter/free"],
        },
        "light": {
            "primary": "openrouter/google/gemma-4-31b-it:free",
            "fallbacks": ["openrouter/nvidia/nemotron-3.5-lightning:free", "openrouter/openrouter/free"],
        },
    },
}

# CoT leakage markers (from ReasoningControlMixin)
COT_LEAKAGE_MARKERS = (
    "strict grounding",
    "check verbatim",
    "is verbatim",
    "entities/facts",
    "grounding verification",
    "verification scaffolding",
    "chain of thought",
    "reasoning trace",
    "thinking process",
    "internal monologue",
)


def detect_cot_leakage(text: str) -> bool:
    """Detect if response contains leaked chain-of-thought reasoning."""
    if not text:
        return False
    text_lower = text.lower()
    return any(marker in text_lower for marker in COT_LEAKAGE_MARKERS)


def test_opencode_model(model: str, timeout: int = TEST_TIMEOUT, reasoning_enabled: bool = True) -> Dict[str, Any]:
    """Test a single opencode model with optional reasoning control."""
    start = time.monotonic()
    
    # Build base command
    base_cmd = [
        "opencode", "run",
        "-m", model,
        "--format", "json",
        "--auto",
        "--dir", "/tmp",
        "--pure",
        TEST_PROMPT,
    ]
    
    # Apply reasoning control via capability registry
    caps = get_model_capabilities(model)
    cmd = base_cmd
    if not reasoning_enabled:
        # Apply reasoning control
        if caps.get("supports_reasoning_control", False):
            method = caps.get("reasoning_control_method", "none")
            if method == "cli_flag":
                flag_name = caps.get("cli_flag_name", "variant")
                flag_value = caps.get("cli_flag_value", "minimal")
                if flag_name not in cmd:
                    cmd = cmd + [f"--{flag_name}", flag_value]
            elif method == "model_swap":
                swap_model = caps.get("non_reasoning_variant")
                if swap_model:
                    logger.info(f"Preflight: reasoning disabled for {model}, swapping to {swap_model}")
                    model = swap_model
                    cmd = [
                        "opencode", "run",
                        "-m", model,
                        "--format", "json",
                        "--auto",
                        "--dir", "/tmp",
                        "--pure",
                        TEST_PROMPT,
                    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = (time.monotonic() - start) * 1000
        if result.returncode == 0 and result.stdout.strip():
            # Parse NDJSON for text response
            text = ""
            for line in result.stdout.strip().split("\n"):
                try:
                    event = json.loads(line)
                    if event.get("type") == "text":
                        text += event.get("part", {}).get("text", "")
                except json.JSONDecodeError:
                    continue
            
            # Check for CoT leakage when reasoning was disabled
            cot_leaked = False
            if not reasoning_enabled and detect_cot_leakage(text):
                logger.warning(f"Preflight: CoT leakage detected for {model} with reasoning disabled")
                cot_leaked = True
            
            if "OK" in text.upper() or text.strip():
                if cot_leaked:
                    return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": "CoT leakage detected", "cot_leaked": True}
                return {"available": True, "latency_ms": round(elapsed), "fallback_used": False, "error": None, "cot_leaked": False}
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": result.stderr[:200] if result.stderr else "Empty response", "cot_leaked": False}
    except subprocess.TimeoutExpired:
        elapsed = (time.monotonic() - start) * 1000
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": f"Timeout after {timeout}s", "cot_leaked": False}
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": str(e)[:200], "cot_leaked": False}


def test_openrouter_model(model: str, timeout: int = TEST_TIMEOUT, reasoning_enabled: bool = True) -> Dict[str, Any]:
    """Test a single OpenRouter model via API with optional reasoning control."""
    import requests
    start = time.monotonic()
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"available": False, "latency_ms": 0, "fallback_used": False, "error": "No API key", "cot_leaked": False}
    
    # Strip 'openrouter/' prefix if present for API call
    api_model = model.replace("openrouter/", "")
    if api_model == "openrouter/free":
        api_model = "openrouter/auto"  # OpenRouter's free auto-router
    
    # Build payload with reasoning control
    caps = get_model_capabilities(model)
    payload = {
        "model": api_model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": 10,
        "temperature": 0,
    }
    
    # Apply reasoning control via capability registry
    if not reasoning_enabled and caps.get("supports_reasoning_control", False):
        method = caps.get("reasoning_control_method", "none")
        if method == "api_param":
            param_name = caps.get("api_param_name", "reasoning_effort")
            param_value = caps.get("api_param_value", "minimal")
            payload[param_name] = param_value
        elif method == "model_swap":
            swap_model = caps.get("non_reasoning_variant")
            if swap_model:
                logger.info(f"Preflight: reasoning disabled for {model}, swapping to {swap_model}")
                swap_api_model = swap_model.replace("openrouter/", "")
                if swap_api_model == "openrouter/free":
                    swap_api_model = "openrouter/auto"
                payload["model"] = swap_api_model
                model = swap_model
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        elapsed = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Check for CoT leakage when reasoning was disabled
            cot_leaked = False
            if not reasoning_enabled and detect_cot_leakage(content):
                logger.warning(f"Preflight: CoT leakage detected for {model} with reasoning disabled")
                cot_leaked = True
            
            if "OK" in content.upper() or content.strip():
                if cot_leaked:
                    return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": "CoT leakage detected", "cot_leaked": True}
                return {"available": True, "latency_ms": round(elapsed), "fallback_used": False, "error": None, "cot_leaked": False}
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}", "cot_leaked": False}
    except requests.Timeout:
        elapsed = (time.monotonic() - start) * 1000
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": f"Timeout after {timeout}s", "cot_leaked": False}
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": str(e)[:200], "cot_leaked": False}


def test_model_chain(provider: str, tier: str, models: Dict) -> Dict[str, Any]:
    """Test primary model, then fallbacks if needed.
    
    Tests both with reasoning_enabled=True and reasoning_enabled=False.
    """
    primary = models["primary"]
    fallbacks = models["fallbacks"]
    
    test_func = test_opencode_model if provider == "opencode" else test_openrouter_model
    
    # Test primary with reasoning enabled
    result = test_func(primary, reasoning_enabled=True)
    result["model"] = primary
    result["tier"] = tier
    result["provider"] = provider
    result["reasoning_enabled"] = True
    
    # Test primary with reasoning disabled
    result_reasoning_off = test_func(primary, reasoning_enabled=False)
    result_reasoning_off["model"] = primary
    result_reasoning_off["tier"] = tier
    result_reasoning_off["provider"] = provider
    result_reasoning_off["reasoning_enabled"] = False
    
    # Determine overall availability: both reasoning modes must work
    reasoning_control_works = result_reasoning_off.get("available", False) and not result_reasoning_off.get("cot_leaked", False)
    
    if result["available"] and reasoning_control_works:
        result["reasoning_control_works"] = True
        result["reasoning_disabled_latency_ms"] = result_reasoning_off.get("latency_ms")
        return result
    
    # Try fallbacks
    for fb in fallbacks:
        fb_result = test_func(fb, reasoning_enabled=True)
        fb_result_reasoning_off = test_func(fb, reasoning_enabled=False)
        
        fb_reasoning_control_works = fb_result_reasoning_off.get("available", False) and not fb_result_reasoning_off.get("cot_leaked", False)
        
        if fb_result.get("available", False) and fb_reasoning_control_works:
            fb_result["model"] = fb
            fb_result["tier"] = tier
            fb_result["provider"] = provider
            fb_result["fallback_used"] = True
            fb_result["reasoning_control_works"] = True
            fb_result["reasoning_disabled_latency_ms"] = fb_result_reasoning_off.get("latency_ms")
            logger.info(f"Preflight: {provider}/{tier} primary failed, fallback {fb} succeeded")
            return fb_result
    
    logger.warning(f"Preflight: {provider}/{tier} ALL models failed: primary={primary}, fallbacks={fallbacks}")
    result["reasoning_control_works"] = False
    result["reasoning_disabled_latency_ms"] = result_reasoning_off.get("latency_ms")
    return result


def load_configs() -> List[Dict]:
    """Load model configs from config.yaml and config_local.yaml."""
    configs = []
    for config_path in ["config.yaml", "config_local.yaml"]:
        if Path(config_path).exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    configs.append(cfg)
    return configs


def main():
    """Run pre-flight checks and write results."""
    logger.info("=== Starting 5:45 AM Pre-Flight Model Check ===")
    
    # Load configs to see which providers are enabled
    configs = load_configs()
    enabled_providers = set()
    for cfg in configs:
        if cfg.get("opencode", {}).get("enabled"):
            enabled_providers.add("opencode")
        if cfg.get("openrouter", {}).get("enabled"):
            enabled_providers.add("openrouter")
    
    if not enabled_providers:
        logger.warning("No LLM providers enabled in config, skipping preflight")
        return 0
    
    # Build test matrix
    test_tasks = []
    for provider in enabled_providers:
        for tier in ("heavy", "medium", "light"):
            if tier in FREE_MODELS[provider]:
                test_tasks.append((provider, tier, FREE_MODELS[provider][tier]))
    
    logger.info(f"Testing {len(test_tasks)} model tiers across {len(enabled_providers)} providers")
    
    # Run concurrently
    results = {"timestamp": datetime.now().isoformat(), "opencode": {}, "openrouter": {}}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(test_model_chain, provider, tier, models): (provider, tier)
            for provider, tier, models in test_tasks
        }
        
        for future in as_completed(future_to_task):
            provider, tier = future_to_task[future]
            try:
                result = future.result()
                results[provider][tier] = result
                status = "✓" if result.get("available") else "✗"
                fb = " (fallback)" if result.get("fallback_used") else ""
                rc = " reasoning_control_ok" if result.get("reasoning_control_works") else " reasoning_control_fail"
                logger.info(f"  {status} {provider}/{tier}: {result['model']}{fb}{rc} ({result['latency_ms']}ms)")
            except Exception as e:
                logger.error(f"  ✗ {provider}/{tier}: Exception: {e}")
                results[provider][tier] = {
                    "available": False,
                    "latency_ms": 0,
                    "fallback_used": False,
                    "error": str(e)[:200],
                    "model": "unknown",
                    "tier": tier,
                    "provider": provider,
                    "reasoning_control_works": False,
                }
    
    # Write results
    output_path = Path(".model-availability.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Pre-flight results written to {output_path}")
    
    # Summary
    for provider in enabled_providers:
        available_tiers = [t for t, r in results[provider].items() if r.get("available")]
        reasoning_control_tiers = [t for t, r in results[provider].items() if r.get("reasoning_control_works")]
        logger.info(f"{provider}: {len(available_tiers)}/3 tiers available: {available_tiers}")
        logger.info(f"{provider}: {len(reasoning_control_tiers)}/3 tiers reasoning_control_ok: {reasoning_control_tiers}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())