#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Verify Antigravity CLI (`agy`) flag layout against MIGRATION_PLAN_ANTIGRAVITY.md.

Run this on a machine where `agy` is installed and `GEMINI_API_KEY` is set.
Output is plain text with no secrets — paste the whole output back to your
collaborator (or open a PR with the suggested BINARY_PROFILES diff).

Usage:
    python3 scripts/verify_agy.py
    python3 scripts/verify_agy.py --binary agy --model flash-lite
    python3 scripts/verify_agy.py --skip-real-call   # help-only, zero quota use

Cost: one tiny `flash-lite` call (~1¢ at most). Add --skip-real-call to spend none.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Flag combos we want to confirm. These match what's currently in
# scripts/gemini_client.py:BINARY_PROFILES["agy"] — anything that fails
# here is something to fix in that table.
PLAN_ARGS = {
    "prompt_via": "positional",          # not "--prompt"
    "output_flag": "--output=json",      # not "--output-format json"
    "quiet_flag": "--quiet",             # not "--raw-output"
    "skip_perm_flag": "--dangerously-skip-permissions",  # not "--yolo"
}

FALLBACK_VARIANTS = {
    "output_flag": ["--output=json", "--output", "--output-format"],
    "quiet_flag": ["--quiet", "--silent", "--raw-output", "--no-spinner"],
    "skip_perm_flag": [
        "--dangerously-skip-permissions",
        "--yolo",
        "--approval-mode",
        "--skip-permissions",
        "--no-confirm",
    ],
}


def mask(s: Optional[str]) -> str:
    """Mask anything that looks like a credential before printing."""
    if not s:
        return "<unset>"
    if len(s) <= 8:
        return "***"
    return f"{s[:4]}...{s[-4:]}"


def run(cmd: List[str], env: Optional[Dict[str, str]] = None,
        timeout: int = 30) -> Tuple[int, str, str]:
    """Run a command, return (exit_code, stdout, stderr). Errors don't raise."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", f"binary not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_binary(binary: str) -> Optional[str]:
    section(f"[1] Binary check: {binary}")
    path = shutil.which(binary)
    if not path:
        print(f"  FAIL: '{binary}' not on PATH")
        print(f"  PATH={os.environ.get('PATH', '')[:200]}...")
        return None
    print(f"  PASS: which {binary} → {path}")

    for flag in ("--version", "-v", "version"):
        rc, out, err = run([binary, flag])
        if rc == 0 and out:
            first_line = out.splitlines()[0][:80]
            print(f"  PASS: {binary} {flag} → {first_line}")
            return path
    print(f"  WARN: {binary} has no --version/--v/version subcommand")
    return path


def grep_help(binary: str) -> Optional[str]:
    section("[2] --help analysis")
    rc, out, err = run([binary, "--help"], timeout=10)
    if rc != 0:
        # Some CLIs put help on stderr
        out = out or err
    if not out:
        print("  FAIL: no --help output")
        return None

    print(f"  Lines: {len(out.splitlines())} ({len(out)} chars)")
    print()

    # Find the lines that mention each interesting flag pattern
    patterns = {
        "model": r"--model\b",
        "prompt": r"--prompt\b",
        "output": r"--output(\b|[=-])",
        "quiet": r"--quiet\b",
        "raw": r"--raw[\w-]*\b",
        "json": r"\bjson\b",
        "yolo": r"--yolo\b|approval[\w-]*",
        "skip-perm": r"--(?:dangerously-)?skip[\w-]*permission",
        "api_key_env": r"GEMINI_API_KEY|AGY_API_KEY|ANTIGRAVITY_API_KEY",
        "config_dir_env": r"GEMINI_CONFIG_DIR|AGY_CONFIG_DIR|ANTIGRAVITY_CONFIG_DIR",
    }
    for name, pat in patterns.items():
        hits = [
            line.strip() for line in out.splitlines()
            if re.search(pat, line, flags=re.IGNORECASE)
        ]
        if hits:
            print(f"  [{name:14}] " + (hits[0][:100]))
            for extra in hits[1:3]:
                print(f"  {'':17} " + extra[:100])
        else:
            print(f"  [{name:14}] (not found in --help)")

    return out


def attempt_call(binary: str, model: str, prompt: str,
                 argv_after_prompt: List[str],
                 env_overrides: Dict[str, str],
                 label: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """One verification call. Returns (success, parsed_json_or_None)."""
    process_env = os.environ.copy()
    process_env.update(env_overrides)

    cmd = [binary, prompt, "--model", model] + argv_after_prompt
    pretty_cmd = " ".join(
        ["agy", repr(prompt), "--model", model] + argv_after_prompt
    )
    print(f"\n  Trying: {pretty_cmd}")

    rc, stdout, stderr = run(cmd, env=process_env, timeout=60)
    if rc != 0:
        first_err = (stderr or stdout).splitlines()[:3]
        print(f"  FAIL ({label}): exit {rc}")
        for line in first_err:
            print(f"    | {line[:120]}")
        return False, None

    print(f"  PASS ({label}): exit 0, stdout {len(stdout)} chars")
    try:
        data = json.loads(stdout)
        print(f"    JSON keys: {sorted(data.keys())[:8]}")
        return True, data
    except json.JSONDecodeError:
        first_line = stdout.splitlines()[0][:120] if stdout else "(empty)"
        print(f"    NOT JSON, first line: {first_line}")
        return True, None  # call succeeded but response wasn't JSON


def try_plan_argv(binary: str, model: str, prompt: str) -> Optional[Dict[str, Any]]:
    section("[3] Real call with migration-plan argv")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("AGY_API_KEY")
    if not api_key:
        print("  SKIP: no GEMINI_API_KEY or AGY_API_KEY in env")
        return None

    print(f"  Using key: {mask(api_key)}")
    # Set both envs as the production client does
    env = {"GEMINI_API_KEY": api_key, "AGY_API_KEY": api_key}
    argv = [
        PLAN_ARGS["output_flag"],
        PLAN_ARGS["quiet_flag"],
        PLAN_ARGS["skip_perm_flag"],
    ]
    success, data = attempt_call(binary, model, prompt, argv, env, "plan argv")
    return data if success else None


def try_fallback_combos(binary: str, model: str, prompt: str) -> None:
    section("[4] Flag variant probing (only if plan argv failed)")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("AGY_API_KEY")
    if not api_key:
        print("  SKIP: no key, cannot probe variants")
        return

    env = {"GEMINI_API_KEY": api_key, "AGY_API_KEY": api_key}

    # Walk fallbacks for the skip-perm flag first (most likely to be renamed)
    for variant in FALLBACK_VARIANTS["skip_perm_flag"]:
        # Some flags take a value (e.g. --approval-mode yolo)
        argv = [PLAN_ARGS["output_flag"], PLAN_ARGS["quiet_flag"]]
        if variant == "--approval-mode":
            argv += [variant, "yolo"]
        else:
            argv += [variant]
        attempt_call(binary, model, prompt, argv, env, f"skip-perm={variant}")


def env_var_probe(binary: str, model: str, prompt: str) -> None:
    section("[5] API key env var detection")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("AGY_API_KEY")
    if not api_key:
        print("  SKIP: no key in env")
        return

    plan_argv = [
        PLAN_ARGS["output_flag"],
        PLAN_ARGS["quiet_flag"],
        PLAN_ARGS["skip_perm_flag"],
    ]
    base_env = {k: "" for k in (
        "GEMINI_API_KEY", "AGY_API_KEY", "ANTIGRAVITY_API_KEY",
        "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
    )}

    for var_name in ("GEMINI_API_KEY", "AGY_API_KEY", "ANTIGRAVITY_API_KEY"):
        env = {**base_env, var_name: api_key}
        attempt_call(binary, model, prompt, plan_argv, env, f"only {var_name}")


def config_dir_probe(binary: str, model: str, prompt: str) -> None:
    section("[6] Config dir env var detection")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("AGY_API_KEY")
    if not api_key:
        print("  SKIP: no key in env")
        return

    plan_argv = [
        PLAN_ARGS["output_flag"],
        PLAN_ARGS["quiet_flag"],
        PLAN_ARGS["skip_perm_flag"],
    ]

    for var_name in ("AGY_CONFIG_DIR", "ANTIGRAVITY_CONFIG_DIR", "GEMINI_CONFIG_DIR"):
        with tempfile.TemporaryDirectory(prefix=f"verify_{var_name}_") as tmp:
            # Drop a settings.json into both .agy/ and .gemini/ for safety
            for subdir in (".agy", ".gemini"):
                d = Path(tmp) / subdir
                d.mkdir()
                (d / "settings.json").write_text(json.dumps({
                    "general": {"maxAttempts": 1, "requestTimeout": 120000},
                    "tools": {"autoAccept": True},
                }))
            env = {
                "GEMINI_API_KEY": api_key,
                "AGY_API_KEY": api_key,
                var_name: tmp,
                "HOME": tmp,
            }
            attempt_call(binary, model, prompt, plan_argv, env, f"only {var_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--binary", default="agy",
                        help="CLI binary to test (default: agy)")
    parser.add_argument("--model", default="flash-lite",
                        help="Model tier name (default: flash-lite, cheapest)")
    parser.add_argument("--prompt", default="Reply with just '4' and nothing else: what is 2+2?",
                        help="Short test prompt (default: math question)")
    parser.add_argument("--skip-real-call", action="store_true",
                        help="Help-only mode — zero quota usage")
    parser.add_argument("--skip-variant-probe", action="store_true",
                        help="Skip [4-6] which make multiple LLM calls")
    args = parser.parse_args()

    print(f"verify_agy.py — checking '{args.binary}' against migration plan")
    print(f"Plan argv (from BINARY_PROFILES[\"agy\"]):")
    for k, v in PLAN_ARGS.items():
        print(f"  {k:18} = {v}")

    path = check_binary(args.binary)
    if path is None:
        print("\n→ Binary missing — install agy and retry. Nothing else to check.")
        return 1

    help_text = grep_help(args.binary)

    if args.skip_real_call:
        print("\n→ --skip-real-call set; stopping before any LLM call.")
        return 0

    data = try_plan_argv(args.binary, args.model, args.prompt)
    if data is None and not args.skip_variant_probe:
        try_fallback_combos(args.binary, args.model, args.prompt)

    if not args.skip_variant_probe:
        env_var_probe(args.binary, args.model, args.prompt)
        config_dir_probe(args.binary, args.model, args.prompt)

    section("[7] Summary")
    if data is not None:
        print("  Migration-plan argv WORKS as-is. No changes needed to")
        print("  BINARY_PROFILES['agy'] in scripts/gemini_client.py.")
        # Inspect the JSON shape so we know whether usage_stats keys match
        print(f"\n  JSON top-level keys: {sorted(data.keys())}")
        stats = data.get("stats", {}).get("models")
        if stats:
            sample_key = next(iter(stats.keys()))
            sample_tok = stats[sample_key].get("tokens", {})
            print(f"  stats.models.{sample_key}.tokens = {sample_tok}")
    else:
        print("  Migration-plan argv FAILED. Review section [4] above to")
        print("  see which flag variant worked, then patch BINARY_PROFILES")
        print("  ['agy'] in scripts/gemini_client.py.")
    return 0 if data is not None else 1


if __name__ == "__main__":
    sys.exit(main())
