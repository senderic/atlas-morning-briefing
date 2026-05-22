# Migration: Gemini CLI → Antigravity CLI (`agy`)

> **Status:** Code migration **landed**. Flag names follow the plan below
> and **still need verification against `agy --help`** on a machine with
> the binary installed. The implementation is dual-backend so the legacy
> `gemini` path stays available as a fallback until the June 18, 2026
> deadline.

## What was implemented

`scripts/gemini_client.py` now auto-detects which CLI binary is available
and dispatches the correct argv layout for each. The defaults make the
common case zero-touch:

| Scenario | Result |
| --- | --- |
| Only `gemini` on PATH | Uses `gemini` (existing flow) |
| Only `agy` on PATH | Uses `agy` (new flow) |
| Both on PATH | Picks `agy` (deadline is approaching) |
| `gemini.cli_binary: "agy"` in config.yaml | Forces `agy` even if `gemini` exists |
| `gemini.cli_binary: "gemini"` in config.yaml | Pins the legacy binary |
| Neither | Disables intelligence features cleanly |

The detection order, per-binary argv layout, config-dir name, and trust
environment variables all live in the `BINARY_PROFILES` table at the top
of `gemini_client.py`. Add a profile there if a new CLI variant ships.

### Flag mapping (already wired up)

| Concept | `gemini` argv | `agy` argv |
|---|---|---|
| Binary name | `gemini` | `agy` |
| Prompt | `--prompt <text>` | positional `<text>` |
| Model | `--model pro` | `--model pro` |
| JSON output | `--output-format json` | `--output=json` |
| Quiet/raw | `--raw-output --accept-raw-output-risk` | `--quiet` |
| Skip prompts | `--approval-mode yolo` | `--dangerously-skip-permissions` |
| Config dir env | `GEMINI_CONFIG_DIR` | `AGY_CONFIG_DIR` |
| Trust workspace env | `GEMINI_CLI_TRUST_WORKSPACE` | `AGY_TRUST_WORKSPACE` |
| API key env (set by us) | `GEMINI_API_KEY` | `GEMINI_API_KEY` **and** `AGY_API_KEY` (both set defensively) |

The settings.json poked into the temp config dir uses the same shape for
both (`general.maxAttempts`, `general.requestTimeout`,
`tools.autoAccept`). If `agy` renames any of those keys upstream, edit
`BINARY_PROFILES["agy"]` and ship a follow-up; nothing else in the
codebase needs to change.

## What still needs verification

The migration plan called these out, and they remain `# VERIFY` markers
in the code/tests until someone confirms them against `agy --help`:

- [ ] `agy` accepts the prompt as a positional arg (no `--prompt`).
- [ ] `agy --output=json` is correct (uses `=`, not a space).
- [ ] `agy --quiet` suppresses banners/spinner without dropping the JSON body.
- [ ] `agy --dangerously-skip-permissions` is the right flag (vs e.g. `--yolo`).
- [ ] `agy --model <id>` works with the same tier names (`pro` / `flash` / `flash-lite`).
- [ ] `agy` reads `GEMINI_API_KEY` (the migration plan suspects so; we
      also set `AGY_API_KEY` as a belt-and-suspenders fallback).
- [ ] `agy` honors `AGY_CONFIG_DIR` and looks for `.agy/settings.json`.
- [ ] Settings keys (`general.maxAttempts`, `general.requestTimeout`,
      `tools.autoAccept`) match agy's schema.

To smoke-test once the binary is on your PATH:

```bash
# Verify the binary itself
which agy
agy --version
agy --help | grep -iE "model|prompt|api.?key|json|quiet"

# Verify with a dummy key and short prompt
GEMINI_API_KEY="dummy" agy "what is 2+2?" --model flash-lite \
    --output=json --quiet --dangerously-skip-permissions

# Then run the briefing pipeline with cli_binary pinned to agy
python3 scripts/briefing_runner.py --config config.yaml --dry-run
```

If any flag is wrong, the fix is local: edit the relevant function
(`_build_agy_cmd` or the agy entry in `BINARY_PROFILES`) — every caller
already routes through that table.

## Installation (still manual / out-of-band)

The Antigravity CLI is a Go binary not yet on PyPI. Until that changes,
install it out-of-band on each machine that runs the briefing. Pick a
release from the official source (verify the domain — should be a Google
property), then:

```bash
# 1) Pull the latest release artifact (replace URL once verified)
mkdir -p /mnt/fast_scratch/bin
wget -O /tmp/agy.tar.gz "<official release URL>"

# 2) Inspect before extracting — never pipe to bash
file /tmp/agy.tar.gz
tar -tzf /tmp/agy.tar.gz | head -20
# (compare sha256sum against the release page)

# 3) Extract and make executable
tar -xzf /tmp/agy.tar.gz -C /mnt/fast_scratch/bin/
chmod +x /mnt/fast_scratch/bin/agy

# 4) Make sure run_briefing.sh's PATH includes /mnt/fast_scratch/bin
which agy && agy --version
```

The existing `run_briefing.sh` already exports PATH so cron sessions see
it; just make sure the install directory is on that PATH.

## Key rotation

The multi-key rotation logic in `_load_api_keys` / `_rotate_key` was not
touched. Both `GEMINI_API_KEY` (comma-separated) and `GEMINI_API_KEY_*`
suffix variants continue to feed the rotation pool, and rotation still
only kicks in on the heavy (Pro) tier where free quotas bind tightest.
Whichever binary is active receives the rotated key through the same
`GEMINI_API_KEY` / `AGY_API_KEY` plumbing in `_execute_command`.

## Tests

The suite now has **535 tests passing**, including:

- 16 new tests in `tests/test_gemini_client_agy.py` covering the agy
  profile, auto-detection order (agy preferred over gemini), explicit
  override, invalid `cli_binary` rejection, and the full agy argv layout
  + env scrubbing.
- The existing 32 Gemini tests in `test_gemini_client.py`,
  `test_gemini_client_full.py`, and `test_gemini_rotation.py` were
  updated to pin `cli_binary: "gemini"` so subprocess mocks don't
  accidentally pick agy via auto-detection.

## Timeline

- **Done (May 22, 2026):** Dual-backend client landed, agy code path
  unit-tested, config schema accepts `cli_binary` override, docs
  updated.
- **Once `agy` binary is available locally:** run the verification
  checklist above, adjust any wrong flag in `BINARY_PROFILES`, push a
  follow-up if needed.
- **By June 18, 2026:** Once verified, optionally remove the legacy
  gemini profile + `_build_gemini_cmd` and simplify to agy-only.

---

*Updated: May 22, 2026 — migration landed in dual-backend mode.*
