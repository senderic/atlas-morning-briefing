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

# Flag combos that ACTUALLY exist in agy 1.0.1 (per `agy --help`):
#   --print / --prompt / -p   Run a single prompt non-interactively
#   --print-timeout           Timeout for print mode (default 5m)
#   --dangerously-skip-permissions  Auto-approve tool permissions
#   --sandbox                 Run with terminal sandbox
#   --add-dir                 Add workspace dir (repeatable)
#
# Notably absent: --model, --output, --output-format, --quiet, --raw-output,
# --approval-mode. So model selection must happen via env var / config /
# default. Output is plain text, not JSON.
PLAN_ARGS = {
    "prompt_mode_flag": "--print",
    "prompt_via": "positional",
    "skip_perm_flag": "--dangerously-skip-permissions",
}

# Variant probes — if the above primary args fail, we walk these to find what
# does work. Plain values where flag takes a value, list of (flag, value)
# tuples where the flag is a switch on its own.
FALLBACK_VARIANTS = {
    "prompt_mode_flag": ["--print", "--prompt", "-p"],
    "model_selection_env": [
        "AGY_MODEL", "GEMINI_MODEL", "ANTIGRAVITY_MODEL", "MODEL",
    ],
    "api_key_env": [
        "GEMINI_API_KEY", "AGY_API_KEY", "ANTIGRAVITY_API_KEY",
        "GOOGLE_API_KEY",
    ],
    "config_dir_env": [
        "AGY_CONFIG_DIR", "ANTIGRAVITY_CONFIG_DIR", "GEMINI_CONFIG_DIR",
    ],
}


def mask(s: Optional[str]) -> str:
    """Mask anything that looks like a credential before printing."""
    if not s:
        return "<unset>"
    if len(s) <= 8:
        return "***"
    return f"{s[:4]}...{s[-4:]}"


def load_env_files() -> List[str]:
    """
    Load .env from common locations into os.environ so the verifier picks
    up credentials the same way briefing_runner.py does.

    Returns the list of paths that were loaded (for reporting).
    """
    loaded = []
    # Look at the project root (parent of scripts/) and the user's home
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",
        Path.home() / ".env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            # Strip optional `export ` prefix
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        loaded.append(str(path))
    return loaded


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
    # Try several conventions: --help, -h, help subcommand, no args.
    # Capture both stdout and stderr separately — many Go CLIs print
    # help on stderr by default.
    attempts = [
        [binary, "--help"],
        [binary, "-h"],
        [binary, "help"],
        [binary],
    ]
    out_text = ""
    for cmd in attempts:
        rc, out, err = run(cmd, timeout=10)
        combined = (out or "") + ("\n" + err if err else "")
        combined = combined.strip()
        print(f"  Tried: {' '.join(cmd):20} exit={rc} "
              f"stdout={len(out)}c stderr={len(err)}c")
        if combined and len(combined) > out_text.__len__():
            out_text = combined

    if not out_text:
        print("\n  FAIL: no help output from any invocation")
        print("  Try manually:  agy --help; agy -h; agy help; agy")
        return None

    print(f"\n  Best output: {len(out_text.splitlines())} lines, "
          f"{len(out_text)} chars")
    print("  --- first 30 help lines (truncated) ---")
    for line in out_text.splitlines()[:30]:
        print(f"  | {line[:110]}")
    print("  --- end ---")
    print()

    # Find the lines that mention each interesting flag pattern
    patterns = {
        "model": r"--model\b|-m\b",
        "prompt": r"--prompt\b|-p\b",
        "output": r"--output(\b|[=-])|-o\b",
        "quiet": r"--quiet\b|-q\b",
        "raw": r"--raw[\w-]*\b",
        "json": r"\bjson\b",
        "yolo": r"--yolo\b|approval[\w-]*",
        "skip-perm": r"--(?:dangerously-)?skip[\w-]*permission|--unsafe",
        "api_key_env": r"GEMINI_API_KEY|AGY_API_KEY|ANTIGRAVITY_API_KEY|API[\s_-]?KEY",
        "config_dir_env": r"GEMINI_CONFIG_DIR|AGY_CONFIG_DIR|ANTIGRAVITY_CONFIG_DIR|CONFIG[\s_-]?DIR",
    }
    print("  --- flag/env greps ---")
    for name, pat in patterns.items():
        hits = [
            line.strip() for line in out_text.splitlines()
            if re.search(pat, line, flags=re.IGNORECASE)
        ]
        if hits:
            print(f"  [{name:14}] " + (hits[0][:110]))
            for extra in hits[1:3]:
                print(f"  {'':17} " + extra[:110])
        else:
            print(f"  [{name:14}] (not found)")

    return out_text


def first_api_key() -> Optional[str]:
    """Match scripts/gemini_client.py:_load_api_keys precedence."""
    raw = os.environ.get("GEMINI_API_KEY", "")
    if raw:
        return raw.split(",")[0].strip()
    for var in sorted(k for k in os.environ if k.startswith("GEMINI_API_KEY_")):
        val = os.environ[var].strip()
        if val:
            return val
    for var in ("AGY_API_KEY", "ANTIGRAVITY_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return None


def attempt_call(binary: str, prompt: str,
                 argv: List[str],
                 env_overrides: Dict[str, str],
                 label: str,
                 timeout: int = 90) -> Tuple[bool, str, str]:
    """
    One verification call. Returns (success, stdout, stderr).

    Unlike the previous version this passes the full argv as-is so each
    test can supply its own ordering / flag combination.
    """
    process_env = os.environ.copy()
    process_env.update(env_overrides)

    pretty = " ".join("'" + a + "'" if " " in a else a for a in argv)
    print(f"\n  Trying ({label}): {pretty}")

    rc, stdout, stderr = run(argv, env=process_env, timeout=timeout)
    if rc != 0:
        print(f"  FAIL: exit {rc}")
        for line in (stderr or stdout).splitlines()[:5]:
            print(f"    err| {line[:120]}")
        return False, stdout, stderr

    print(f"  PASS: exit 0, stdout={len(stdout)}c stderr={len(stderr)}c")
    if stdout:
        head = "\n".join(stdout.splitlines()[:5])
        for line in head.splitlines():
            print(f"    out| {line[:120]}")
        if len(stdout.splitlines()) > 5:
            print(f"    out| ... ({len(stdout.splitlines()) - 5} more lines)")
    if stderr:
        head = "\n".join(stderr.splitlines()[:3])
        for line in head.splitlines():
            print(f"    err| {line[:120]}")
    return True, stdout, stderr


def try_plan_argv(binary: str, prompt: str) -> Optional[str]:
    section("[3] Real call with agy 1.0.1 best-guess argv")
    api_key = first_api_key()
    if not api_key:
        print("  SKIP: no API key found in any known env var")
        return None
    print(f"  Using key: {mask(api_key)}")

    # Set every env var agy *might* read; the production client does the same.
    env = {
        "GEMINI_API_KEY": api_key,
        "AGY_API_KEY": api_key,
        "ANTIGRAVITY_API_KEY": api_key,
    }
    # Per `agy --help`: --print is the non-interactive mode flag, prompt is
    # positional. We also try with --dangerously-skip-permissions to avoid
    # any interactive permission grant.
    argv = [
        binary,
        "--print",
        "--dangerously-skip-permissions",
        prompt,
    ]
    ok, stdout, stderr = attempt_call(binary, prompt, argv, env, "primary")
    return stdout if ok else None


def try_alternate_argv(binary: str, prompt: str) -> None:
    section("[4] Alternate argv variants")
    api_key = first_api_key()
    if not api_key:
        print("  SKIP: no API key")
        return
    env = {
        "GEMINI_API_KEY": api_key,
        "AGY_API_KEY": api_key,
        "ANTIGRAVITY_API_KEY": api_key,
    }
    # Try several orderings & flag aliases to see what agy accepts
    variants = [
        # No skip-permissions
        [binary, "--print", prompt],
        # --prompt alias
        [binary, "--prompt", prompt],
        # -p short alias
        [binary, "-p", prompt],
        # Flag-then-prompt with skip-permissions BEFORE --print
        [binary, "--dangerously-skip-permissions", "--print", prompt],
        # Prompt BEFORE the flags (some CLIs are positional-first)
        [binary, prompt, "--print", "--dangerously-skip-permissions"],
        # Just positional, no mode flag (does agy print-mode by default in non-tty?)
        [binary, prompt],
    ]
    for argv in variants:
        attempt_call(binary, prompt, argv, env, label=f"{argv[1]} ...", timeout=45)


def env_var_probe(binary: str, prompt: str) -> None:
    section("[5] API key env var detection (which one does agy read?)")
    api_key = first_api_key()
    if not api_key:
        print("  SKIP: no key in env")
        return

    base_argv = [binary, "--print", "--dangerously-skip-permissions", prompt]
    # Clear EVERY known auth env so only the one under test is set.
    base_env = {k: "" for k in (
        "GEMINI_API_KEY", "AGY_API_KEY", "ANTIGRAVITY_API_KEY",
        "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
    )}
    # Also clear any GEMINI_API_KEY_* suffix vars from the inherited env.
    for k in list(os.environ):
        if k.startswith("GEMINI_API_KEY_"):
            base_env[k] = ""

    for var_name in FALLBACK_VARIANTS["api_key_env"]:
        env = {**base_env, var_name: api_key}
        attempt_call(binary, prompt, base_argv, env, label=f"only {var_name}")


def config_dir_probe(binary: str, prompt: str) -> None:
    section("[6] Config dir env var detection")
    api_key = first_api_key()
    if not api_key:
        print("  SKIP: no key in env")
        return

    base_argv = [binary, "--print", "--dangerously-skip-permissions", prompt]

    for var_name in FALLBACK_VARIANTS["config_dir_env"]:
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
            attempt_call(binary, prompt, base_argv, env, label=f"{var_name}={mask(tmp)}")


def model_env_probe(binary: str, prompt: str) -> None:
    section("[5b] Model selection env var probe")
    api_key = first_api_key()
    if not api_key:
        print("  SKIP: no key in env")
        return

    base_argv = [binary, "--print", "--dangerously-skip-permissions", prompt]
    base_env = {
        "GEMINI_API_KEY": api_key,
        "AGY_API_KEY": api_key,
    }
    # agy 1.0.1 has no --model flag, so model selection must be env-driven.
    # Try each candidate env var with a known model name. If the call still
    # succeeds we can't distinguish; we just record the result.
    for var_name in FALLBACK_VARIANTS["model_selection_env"]:
        env = {**base_env, var_name: "flash-lite"}
        attempt_call(binary, prompt, base_argv, env, label=f"{var_name}=flash-lite")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--binary", default="agy",
                        help="CLI binary to test (default: agy)")
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

    loaded = load_env_files()
    print(f"\n.env files loaded: {loaded if loaded else '(none found)'}")
    print("Key env vars now visible to subprocess:")
    for name in (
        "GEMINI_API_KEY", "AGY_API_KEY", "ANTIGRAVITY_API_KEY",
        "GEMINI_API_KEY_1", "GEMINI_API_KEY_PRIMARY",
    ):
        print(f"  {name:24} = {mask(os.environ.get(name))}")
    extra_gemini = [k for k in os.environ if k.startswith("GEMINI_API_KEY_")
                    and k not in ("GEMINI_API_KEY_1", "GEMINI_API_KEY_PRIMARY")]
    if extra_gemini:
        print(f"  (+{len(extra_gemini)} more GEMINI_API_KEY_* keys)")

    path = check_binary(args.binary)
    if path is None:
        print("\n→ Binary missing — install agy and retry. Nothing else to check.")
        return 1

    help_text = grep_help(args.binary)

    if args.skip_real_call:
        print("\n→ --skip-real-call set; stopping before any LLM call.")
        return 0

    primary_stdout = try_plan_argv(args.binary, args.prompt)
    if primary_stdout is None and not args.skip_variant_probe:
        try_alternate_argv(args.binary, args.prompt)

    if not args.skip_variant_probe:
        env_var_probe(args.binary, args.prompt)
        model_env_probe(args.binary, args.prompt)
        config_dir_probe(args.binary, args.prompt)

    section("[7] Summary")
    if primary_stdout is not None:
        print("  Primary argv WORKS:")
        print("    agy --print --dangerously-skip-permissions <prompt>")
        print()
        print("  Response is plain text (not JSON). First 200 chars of stdout:")
        print(f"    {primary_stdout[:200]!r}")
        print()
        print("  Next steps:")
        print("  1. Update scripts/gemini_client.py:_build_agy_cmd to match.")
        print("  2. Add a plain-text response parser to BINARY_PROFILES['agy']")
        print("     (the existing JSON parser falls back to raw stdout already,")
        print("     which works but loses token stats).")
        print("  3. Review sections [5], [5b], [6] above for which env vars")
        print("     agy actually reads.")
    else:
        print("  Primary argv FAILED. Look at section [4] to see which")
        print("  argv variant returned exit 0 with a coherent response.")
    return 0 if primary_stdout is not None else 1


if __name__ == "__main__":
    sys.exit(main())
