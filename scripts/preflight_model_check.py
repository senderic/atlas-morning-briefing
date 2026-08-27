#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
5:45 AM pre-flight model availability check.

Runs concurrently (thread pool) against all free models to determine
which are responsive before the 6:00 AM briefing run. Writes results to
.model-availability.json for the briefing runner to consume.
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


def test_opencode_model(model: str, timeout: int = TEST_TIMEOUT) -> Dict[str, Any]:
    """Test a single opencode model."""
    start = time.monotonic()
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
            if "OK" in text.upper() or text.strip():
                return {"available": True, "latency_ms": round(elapsed), "fallback_used": False, "error": None}
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": result.stderr[:200] if result.stderr else "Empty response"}
    except subprocess.TimeoutExpired:
        elapsed = (time.monotonic() - start) * 1000
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": str(e)[:200]}


def test_openrouter_model(model: str, timeout: int = TEST_TIMEOUT) -> Dict[str, Any]:
    """Test a single OpenRouter model via API."""
    import requests
    start = time.monotonic()
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"available": False, "latency_ms": 0, "fallback_used": False, "error": "No API key"}
    
    # Strip 'openrouter/' prefix if present for API call
    api_model = model.replace("openrouter/", "")
    if api_model == "openrouter/free":
        api_model = "openrouter/auto"  # OpenRouter's free auto-router
    
    payload = {
        "model": api_model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": 10,
        "temperature": 0,
    }
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
            if "OK" in content.upper() or content.strip():
                return {"available": True, "latency_ms": round(elapsed), "fallback_used": False, "error": None}
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except requests.Timeout:
        elapsed = (time.monotonic() - start) * 1000
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return {"available": False, "latency_ms": round(elapsed), "fallback_used": False, "error": str(e)[:200]}


def test_model_chain(provider: str, tier: str, models: Dict) -> Dict[str, Any]:
    """Test primary model, then fallbacks if needed."""
    primary = models["primary"]
    fallbacks = models["fallbacks"]
    
    test_func = test_opencode_model if provider == "opencode" else test_openrouter_model
    
    # Test primary
    result = test_func(primary)
    result["model"] = primary
    result["tier"] = tier
    result["provider"] = provider
    
    if result["available"]:
        return result
    
    # Try fallbacks
    for fb in fallbacks:
        fb_result = test_func(fb)
        fb_result["model"] = fb
        fb_result["tier"] = tier
        fb_result["provider"] = provider
        fb_result["fallback_used"] = True
        if fb_result["available"]:
            logger.info(f"Preflight: {provider}/{tier} primary failed, fallback {fb} succeeded")
            return fb_result
    
    logger.warning(f"Preflight: {provider}/{tier} ALL models failed: primary={primary}, fallbacks={fallbacks}")
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
                status = "✓" if result["available"] else "✗"
                fb = " (fallback)" if result.get("fallback_used") else ""
                logger.info(f"  {status} {provider}/{tier}: {result['model']}{fb} ({result['latency_ms']}ms)")
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
                }
    
    # Write results
    output_path = Path(".model-availability.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Pre-flight results written to {output_path}")
    
    # Summary
    for provider in enabled_providers:
        available_tiers = [t for t, r in results[provider].items() if r.get("available")]
        logger.info(f"{provider}: {len(available_tiers)}/3 tiers available: {available_tiers}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())