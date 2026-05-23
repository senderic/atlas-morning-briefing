#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Audit which Gemini models the CLI invokes for each tier, so the
PRICING table in scripts/gemini_client.py can be calibrated against
real Google rates instead of guesses.

Background
----------

Atlas's tier names ("heavy"/"medium"/"light") map to gemini-cli model
aliases ("pro"/"flash"/"flash-lite"). Those aliases resolve to actual
model IDs (e.g. "gemini-3-flash-preview") that Google bills at
specific per-token rates published on
https://ai.google.dev/gemini-api/docs/pricing.

If the alias resolution shifts (which it does — gemini-cli updates
move the "latest" pointers), the PRICING table goes out of sync and
the briefing's cost line silently drifts. This script tells you:

  1. gemini-cli version (for the audit log)
  2. Every Gemini model your API key has access to (via REST, no CLI)
  3. The actual model ID each alias resolves to (one tiny CLI call
     per tier, asks the model to reply with the literal token "OK")
  4. Token counts from that probe — useful as a "what does the CLI
     report as input tokens for an empty-ish prompt" baseline,
     because gemini-cli typically inflates `input` with built-in
     agent harness context

Output goes to logs/gemini-audit-<timestamp>.txt by default. Commit
that file back so it can be analyzed against current Google pricing.

Usage
-----

    uv run scripts/audit_gemini.py
    # or:
    python3 scripts/audit_gemini.py

    # custom output path:
    uv run scripts/audit_gemini.py --output logs/my-audit.txt

    # longer timeout for the slow "pro" tier (default 60s, raise if
    # you saw a parse error on pro in the previous run):
    uv run scripts/audit_gemini.py --timeout 180

    # pick a specific key when earlier ones in rotation are quota'd:
    uv run scripts/audit_gemini.py --key-var GEMINI_API_KEY_3

Cost
----

Three CLI probes x 1-token responses is roughly $0.001 total. The
REST list is free. The whole thing finishes in 30-180s depending on
Pro latency.

Security
--------

Reads `GEMINI_API_KEY` (or `GEMINI_API_KEY_0`) from env, then falls
back to `.env` in the project root. The API key is NEVER written to
the output file — only a redacted preview (`AIzaSy...xxxx`).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional, TextIO

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_api_key(name: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Find a Gemini API key via env vars or .env.

    If `name` is given (e.g. "GEMINI_API_KEY_3"), only that specific
    variable is consulted — first in os.environ, then in .env. Otherwise
    the legacy fallback order is used: GEMINI_API_KEY, GEMINI_API_KEY_0,
    then the first GEMINI_API_KEY[_N] match in .env.

    Returns (key, source_var) so the caller can log which variable won —
    important when the user is debugging quota/key-rotation issues."""
    candidates = [name] if name else ["GEMINI_API_KEY", "GEMINI_API_KEY_0"]

    for var in candidates:
        v = os.environ.get(var)
        if v:
            return v.strip().strip('"').strip("'"), var

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("#"):
                continue
            if name:
                pattern = rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$"
                m = re.match(pattern, line)
                if m:
                    return m.group(1).strip().strip('"').strip("'"), name
            else:
                m = re.match(r"^\s*(GEMINI_API_KEY(?:_\d+)?)\s*=\s*(.+?)\s*$", line)
                if m:
                    return m.group(2).strip().strip('"').strip("'"), m.group(1)
    return None, None


def redact_key(key: str) -> str:
    """AIzaSyABCDEFG12345 -> AIzaSy...2345 (safe to log)."""
    if not key or len(key) < 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def list_available_models(api_key: str, out: TextIO) -> None:
    """GET /v1beta/models — lists every model the key can access."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models"
        f"?key={api_key}&pageSize=200"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        out.write(f"  (REST call failed: {e})\n")
        return

    names = sorted(
        m["name"].replace("models/", "")
        for m in data.get("models", [])
        if "gemini" in m.get("name", "").lower()
    )
    for name in names:
        out.write(f"  {name}\n")


def probe_alias(alias: str, timeout: int, api_key: Optional[str], out: TextIO) -> None:
    """Run one gemini-cli call for the alias and dump what the CLI
    resolved it to + the token accounting.

    `api_key` is forced into the subprocess env as GEMINI_API_KEY, with
    competing Google auth (GOOGLE_API_KEY, ADC, gcloud token) blanked
    out to prevent silent fallback — same hardening as
    gemini_client._execute_command. Without this, the CLI would pick
    whichever key happens to be in the caller's shell, defeating the
    `--key-var` selection."""
    out.write(f"--- gemini --model {alias} ---\n")
    cmd = [
        "gemini", "--model", alias,
        "--prompt", "Reply with the literal token: OK",
        "--approval-mode", "yolo",
        "--raw-output", "--accept-raw-output-risk",
        "--output-format", "json",
    ]
    probe_env = os.environ.copy()
    if api_key:
        probe_env["GEMINI_API_KEY"] = api_key
    probe_env["GOOGLE_API_KEY"] = ""
    probe_env["GOOGLE_APPLICATION_CREDENTIALS"] = ""
    probe_env["CLOUDSDK_AUTH_ACCESS_TOKEN"] = ""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=probe_env
        )
    except subprocess.TimeoutExpired:
        out.write(f"  TIMEOUT after {timeout}s — try --timeout {timeout*2}\n")
        return
    except FileNotFoundError:
        out.write("  gemini-cli not found on PATH\n")
        return

    if result.returncode != 0:
        out.write(f"  CLI exited {result.returncode}\n")
        if result.stderr:
            out.write(f"  stderr (first 200 chars): {result.stderr[:200]!r}\n")

    raw = result.stdout.strip()
    if not raw:
        out.write("  empty stdout (CLI returned nothing — likely auth or quota issue)\n")
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        out.write(f"  parse error: {e}\n")
        out.write(f"  raw stdout (first 200 chars): {raw[:200]!r}\n")
        return

    models = data.get("stats", {}).get("models", {})
    if not models:
        out.write("  (no stats.models in response — unusual)\n")
        return

    for model_key, model_stats in models.items():
        out.write(f"  resolved_model: {model_key}\n")
        out.write(f"  tokens:         {model_stats.get('tokens', {})}\n")
    out.write(f"  response_chars: {len(data.get('response', ''))}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs" / f"gemini-audit-{ts}.txt",
        help="Where to write the audit (default: logs/gemini-audit-<ts>.txt)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-CLI-probe timeout in seconds (raise if Pro times out)",
    )
    parser.add_argument(
        "--key-var",
        default=None,
        help="Specific env / .env variable to read the API key from "
             "(e.g. GEMINI_API_KEY_3). Useful when earlier keys in "
             "rotation are quota-exhausted. Default: GEMINI_API_KEY, "
             "then GEMINI_API_KEY_0, then first .env match.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    api_key, key_source = load_api_key(args.key_var)
    if args.key_var and not api_key:
        print(
            f"error: --key-var {args.key_var} requested but not found "
            f"in os.environ or .env",
            file=sys.stderr,
        )
        return 1

    with args.output.open("w") as out:
        out.write("=== ENV ===\n")
        try:
            ver = subprocess.run(
                ["gemini", "--version"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            ver = "MISSING"
        out.write(f"gemini version: {ver or 'MISSING'}\n")
        out.write(f"Date:           {datetime.datetime.now().astimezone().isoformat()}\n")
        out.write(f"API key:        {redact_key(api_key) if api_key else 'NOT FOUND'}\n")
        out.write(f"Key source:     {key_source or 'n/a'}\n")
        out.write(f"Probe timeout:  {args.timeout}s\n")

        out.write("\n=== Available models via Gemini REST API ===\n")
        if api_key:
            list_available_models(api_key, out)
        else:
            out.write("  (no GEMINI_API_KEY found in env or .env)\n")

        out.write("\n=== CLI alias resolution (one invocation per tier) ===\n")
        for alias in ("pro", "flash", "flash-lite"):
            probe_alias(alias, args.timeout, api_key, out)

    print(f"Audit written to: {args.output}")
    print()
    print(args.output.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
