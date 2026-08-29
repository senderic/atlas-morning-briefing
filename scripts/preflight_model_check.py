#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Pre-flight model availability check (runs ~15 min before the briefing).

Probes the tiered model roster concurrently and writes .model-availability.json
next to the config, which the briefing runner consumes to pin a per-tier model
for the run.

Design rules learned the hard way:

* **The roster comes from config, never from a table in this file.** A
  hardcoded copy drifts from config.yaml, and because the runner lets preflight
  override the configured model, that drift silently swaps tiers.
* **A probe must budget enough tokens for reasoning.** Free models emit
  reasoning tokens before any content; with max_tokens=10 the content field
  comes back empty and a perfectly healthy model is marked dead.
* **A tier's result may only ever name a model from that tier's own chain**, so
  preflight can never promote a light model into the heavy slot.
* **Report what actually happened.** If the primary works, say so; only claim
  "all models failed" when every model in the chain failed.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from scripts.llm_client import get_model_capabilities  # noqa: E402
from scripts.openrouter_client import (  # noqa: E402
    API_BASE_URL,
    DEFAULT_FALLBACK_MODELS as OR_DEFAULT_FALLBACKS,
    DEFAULT_MODELS as OR_DEFAULT_MODELS,
    OpenRouterClient,
)
from scripts.opencode_client import (  # noqa: E402
    DEFAULT_FALLBACK_MODELS as OC_DEFAULT_FALLBACKS,
    DEFAULT_MODELS as OC_DEFAULT_MODELS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# A probe that asks for real prose, so a model that only answers trivially
# (or that stalls on anything non-trivial) is caught here rather than at 6 AM.
TEST_PROMPT = (
    "In one sentence, say what a morning intelligence briefing is for. "
    "Reply with the sentence only."
)
TEST_TIMEOUT = 45          # seconds per probe
# Must comfortably exceed the reasoning trace these models emit before content.
TEST_MAX_TOKENS = 1024
MAX_WORKERS = 6            # matches the runtime concurrency cap
TIERS = ("heavy", "medium", "light")
OUTPUT_FILENAME = ".model-availability.json"

# Substring an endpoint uses when it genuinely will not disable reasoning,
# as opposed to a probe that failed for an unrelated transient reason.
_REASONING_REFUSED = "reasoning is mandatory"


def _probe_result(
    available: bool,
    latency_ms: float,
    error: Optional[str] = None,
    cot_leaked: bool = False,
    reasoning_disabled: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "available": available,
        "latency_ms": round(latency_ms),
        "error": error,
        "cot_leaked": cot_leaked,
        "reasoning_disabled_ok": reasoning_disabled,
    }


def test_opencode_model(model: str, timeout: int = TEST_TIMEOUT) -> Dict[str, Any]:
    """Probe one opencode model through the CLI."""
    start = time.monotonic()
    cmd = [
        "opencode", "run", "-m", model,
        "--format", "json", "--auto", "--dir", "/tmp", "--pure",
        TEST_PROMPT,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _probe_result(False, (time.monotonic() - start) * 1000,
                             f"Timeout after {timeout}s")
    except Exception as e:
        return _probe_result(False, (time.monotonic() - start) * 1000, str(e)[:200])

    elapsed = (time.monotonic() - start) * 1000
    text = ""
    for line in (result.stdout or "").strip().split("\n"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "text":
            text += event.get("part", {}).get("text", "")

    if not text.strip():
        err = (result.stderr or "")[:200] or "Empty response"
        return _probe_result(False, elapsed, err)
    return _probe_result(True, elapsed)


def _openrouter_post(model: str, payload_extra: Optional[Dict[str, Any]], timeout: int):
    """POST one probe. Returns (content, reasoning_len, error_string_or_None)."""
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, 0, "No API key"

    payload: Dict[str, Any] = {
        # Strip only the single routing prefix. `openrouter/openrouter/free`
        # must resolve to `openrouter/free`, NOT to the paid `openrouter/auto`.
        "model": OpenRouterClient._to_api_model(model),
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": TEST_MAX_TOKENS,
        "temperature": 0,
    }
    if payload_extra:
        payload.update(payload_extra)

    try:
        resp = requests.post(
            API_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except Exception as e:
        return None, 0, f"{type(e).__name__}: {str(e)[:150]}"

    if resp.status_code != 200:
        return None, 0, f"HTTP {resp.status_code}: {resp.text[:200]}"
    try:
        data = resp.json()
    except ValueError:
        return None, 0, "Invalid JSON response"
    if isinstance(data.get("error"), dict):
        return None, 0, f"error in body: {str(data['error'])[:200]}"

    choice = (data.get("choices") or [{}])[0] or {}
    message = choice.get("message", {}) or {}
    content = message.get("content") or ""
    reasoning_len = len(message.get("reasoning") or "")
    if not content.strip():
        return None, reasoning_len, (
            f"Empty content (finish={choice.get('finish_reason')}, "
            f"reasoning={reasoning_len} chars)"
        )
    return content, reasoning_len, None


def test_openrouter_model(model: str, timeout: int = TEST_TIMEOUT) -> Dict[str, Any]:
    """Probe one OpenRouter model, and separately probe reasoning suppression.

    Reasoning suppression is reported but is NOT a precondition for
    availability: it is used on exactly one prompt (the editorial intro), so a
    model that answers well with reasoning on is still usable.
    """
    start = time.monotonic()
    content, _, err = _openrouter_post(model, None, timeout)
    elapsed = (time.monotonic() - start) * 1000
    if err:
        return _probe_result(False, elapsed, err)

    # Second probe: does the documented reasoning-suppression param work?
    caps = get_model_capabilities(model)
    reasoning_ok: Optional[bool] = None
    inconclusive = False
    cot_leaked = False
    if caps.get("supports_reasoning_control") and caps.get("reasoning_control_method") == "api_param":
        extra = {caps.get("api_param_name", "reasoning"): caps.get("api_param_value")}
        off_content, off_reasoning_len, off_err = _openrouter_post(model, extra, timeout)
        if off_err:
            # Distinguish "the endpoint refuses to disable reasoning" from
            # "this one probe happened to fail". A transient upstream 502 is
            # not evidence about the model's capabilities, and reporting it as
            # UNSUPPORTED sends the reader chasing a config problem that does
            # not exist (observed 2026-08-28 on nemotron-3-ultra).
            if _REASONING_REFUSED in off_err.lower():
                reasoning_ok = False
                logger.info(
                    "Preflight: %s refuses reasoning suppression: %s", model, off_err
                )
            else:
                reasoning_ok = None  # unknown, not a failure
                inconclusive = True
                logger.info(
                    "Preflight: %s reasoning-suppression probe did not complete "
                    "(transient, capability unknown): %s",
                    model, off_err,
                )
        else:
            # Suppression "works" only if reasoning tokens actually stopped.
            reasoning_ok = off_reasoning_len == 0
            if not reasoning_ok:
                logger.info(
                    "Preflight: %s accepted the reasoning parameter but still "
                    "emitted %d chars of reasoning",
                    model, off_reasoning_len,
                )

    result = _probe_result(True, elapsed, cot_leaked=cot_leaked,
                           reasoning_disabled=reasoning_ok)
    result["reasoning_probe_inconclusive"] = inconclusive
    return result


def test_model_chain(provider: str, tier: str, primary: str, fallbacks: List[str]) -> Dict[str, Any]:
    """Probe a tier's chain in order, returning the first model that answers.

    The returned record always names a model drawn from THIS tier's chain, so a
    preflight result can never move a model between tiers.
    """
    test_func = test_opencode_model if provider == "opencode" else test_openrouter_model
    chain = [primary] + [m for m in fallbacks if m != primary]
    attempts = []

    for idx, model in enumerate(chain):
        result = test_func(model)
        attempts.append({"model": model, "available": result["available"],
                         "error": result["error"]})
        if result["available"]:
            result.update({
                "model": model, "tier": tier, "provider": provider,
                "fallback_used": idx > 0, "attempts": attempts,
            })
            if idx > 0:
                logger.info(
                    "Preflight: %s/%s primary %s failed, using fallback %s",
                    provider, tier, primary, model,
                )
            return result

    logger.warning(
        "Preflight: %s/%s ALL %d models failed (primary=%s)",
        provider, tier, len(chain), primary,
    )
    return {
        "available": False, "latency_ms": 0, "error": attempts[-1]["error"],
        "cot_leaked": False, "reasoning_disabled_ok": None,
        "model": primary, "tier": tier, "provider": provider,
        "fallback_used": False, "attempts": attempts,
    }


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        logger.warning("Config not found: %s", config_path)
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def build_test_matrix(config: Dict[str, Any]) -> List[Tuple[str, str, str, List[str]]]:
    """Derive (provider, tier, primary, fallbacks) tuples from the config.

    Reading the roster from config is what keeps preflight and runtime in
    agreement; a local copy of the model table is how tiers get crossed.
    """
    defaults = {
        "opencode": (OC_DEFAULT_MODELS, OC_DEFAULT_FALLBACKS),
        "openrouter": (OR_DEFAULT_MODELS, OR_DEFAULT_FALLBACKS),
    }
    matrix = []
    for provider, (default_models, default_fallbacks) in defaults.items():
        section = config.get(provider, {}) or {}
        if not section.get("enabled"):
            continue
        # A paid last-resort backstop should not be probed every morning —
        # the probe itself would be the only thing that ever bills it.
        if not section.get("preflight_check", True):
            logger.info("Skipping preflight for %s (preflight_check: false)", provider)
            continue
        models = section.get("models", {}) or {}
        fallbacks = section.get("fallback_models", {}) or {}
        for tier in TIERS:
            primary = models.get(tier, default_models[tier])
            chain = list(fallbacks.get(tier, default_fallbacks[tier]))
            matrix.append((provider, tier, primary, chain))
    return matrix


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-flight LLM model availability check")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"),
                        help="Config file whose model roster is probed")
    parser.add_argument("--output", default=None,
                        help=f"Where to write results (default: <config dir>/{OUTPUT_FILENAME})")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    output_path = Path(args.output) if args.output else config_path.parent / OUTPUT_FILENAME

    logger.info("=== Pre-flight model check (%s) ===", config_path)
    config = load_config(config_path)
    matrix = build_test_matrix(config)
    if not matrix:
        logger.warning("No LLM providers enabled in %s; skipping preflight", config_path)
        return 0

    providers = sorted({p for p, _, _, _ in matrix})
    logger.info("Probing %d tiers across %s", len(matrix), ", ".join(providers))

    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "prompt_tokens_budget": TEST_MAX_TOKENS,
    }
    for provider in providers:
        results[provider] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(test_model_chain, provider, tier, primary, chain): (provider, tier)
            for provider, tier, primary, chain in matrix
        }
        for future in as_completed(futures):
            provider, tier = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error("  ✗ %s/%s: %s", provider, tier, e)
                results[provider][tier] = {
                    "available": False, "latency_ms": 0, "error": str(e)[:200],
                    "cot_leaked": False, "reasoning_disabled_ok": None,
                    "model": "unknown", "tier": tier, "provider": provider,
                    "fallback_used": False, "attempts": [],
                }
                continue
            results[provider][tier] = result
            mark = "✓" if result["available"] else "✗"
            note = " (fallback)" if result.get("fallback_used") else ""
            rd = result.get("reasoning_disabled_ok")
            # Tri-state: True = verified working, False = endpoint refuses,
            # None = not determined (no control configured, or the probe
            # failed transiently). Never report None as UNSUPPORTED.
            rd_note = {
                True: " reasoning-off:ok",
                False: " reasoning-off:REFUSED",
            }.get(rd, "" if rd is None else "")
            if rd is None and result.get("reasoning_probe_inconclusive"):
                rd_note = " reasoning-off:unknown"
            logger.info("  %s %s/%s: %s%s%s (%dms)", mark, provider, tier,
                        result["model"], note, rd_note, result["latency_ms"])

    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Pre-flight results written to %s", output_path)

    exit_code = 0
    for provider in providers:
        ok = [t for t in TIERS if results[provider].get(t, {}).get("available")]
        logger.info("%s: %d/%d tiers available: %s", provider, len(ok), len(TIERS), ok)
        if not ok:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
