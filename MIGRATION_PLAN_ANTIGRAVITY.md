# Migration Plan: Gemini CLI → Antigravity CLI (`agy`)

> **Status: PAUSED — premise was wrong.** Verification on agy 1.0.1
> revealed two showstoppers, and a closer read shows gemini-cli is not
> being deprecated on any announced timeline. See §Findings below.
>
> **Current state of the code:** dual-backend support shipped
> (BINARY_PROFILES in `scripts/gemini_client.py`), but the auto-detect
> preference is now **`gemini` first, `agy` second** — opposite of what
> the original plan envisioned. To opt into `agy` explicitly once it
> becomes headless-viable, set `gemini.cli_binary: "agy"` in
> `config.yaml`.

## Findings (May 22, 2026, real-machine verification)

A real run of `scripts/verify_agy.py` against `agy 1.0.1` showed that
**every assumption in the original plan was wrong**:

### 1. `agy 1.0.1` is OAuth-only — no API key authentication

Every call returns:

```
Authentication required. Please visit the URL to log in:
  https://accounts.google.com/o/oauth2/auth?access_type=offline&client_id=…
Waiting for authentication (timeout 30s)…
Or, paste the authorization code here and press Enter:
```

This happened with `GEMINI_API_KEY`, `AGY_API_KEY`,
`ANTIGRAVITY_API_KEY`, and `GOOGLE_API_KEY` each set in isolation. The
binary ignores all four and demands interactive browser OAuth.

**For unattended cron use (our 6:50 AM briefing), agy is a non-starter
as of 1.0.1.** There is no human at the terminal at 6:50 AM to click an
OAuth URL.

### 2. `agy` is shaped like an interactive coding agent, not a CLI wrapper

`agy --help` exposes only these flags:

```
--add-dir       Add a directory to the workspace (repeatable)
-c / --continue Continue the most recent conversation
--conversation  Resume a previous conversation by ID
--dangerously-skip-permissions   Auto-approve tool requests
-i / --prompt-interactive        Interactive session
-p / --print / --prompt          Non-interactive single prompt
--print-timeout                  Print mode wait (default 5m)
--sandbox       Terminal-restricted sandbox
--log-file      Log path override

Subcommands: changelog, help, install, plugin, plugins, update
```

What's **missing**: `--model`, `--output` (any form), `--quiet`,
`--raw-output`, `--approval-mode`, anything `--api-key`/`--token`. The
shape (`--add-dir`, `--sandbox`, conversation management, plugin
subcommand) suggests `agy` is Google's answer to Claude Code / Cursor,
not a successor to `gemini-cli`. They're two different products that
happen to use Google's models.

### 3. The "June 18, 2026 deadline" appears to be unfounded

The original plan asserted gemini-cli would be deprecated on June 18,
2026. A check of `github.com/google-gemini/gemini-cli` shows no
deprecation notice, no sunset date, and continued active releases
through May 2026. Paid-tier Gemini API keys aren't on any deprecation
timer either — they talk to the Gemini API directly, so as long as
that API exists, any CLI wrapping it (gemini-cli included) keeps
working.

The deadline claim may have come from confusion with `agy` (Antigravity)
being released, which is a separate product launch — not a successor.

## What this means for our codebase

- **Keep gemini-cli.** Your paid-tier API key + the existing flow keeps
  working for the foreseeable future. No migration urgency.
- **Keep the dual-backend code** in `scripts/gemini_client.py`. It cost
  little, lets us test agy in dev easily, and gives us a clean
  opt-in path if (a) agy adds headless auth, or (b) we ever want to
  swap to a third CLI.
- **Auto-detection now prefers gemini.** If `agy` ends up on PATH (e.g.
  installed for interactive use elsewhere), it does NOT silently take
  over and break cron. To explicitly use agy, set
  `gemini.cli_binary: "agy"` in `config.yaml`.

## To revisit when…

The migration plan should be reopened when one of these becomes true:

- **agy gains API key auth** — a flag like `--api-key` or an env var
  like `AGY_API_KEY` that actually works. Run `scripts/verify_agy.py`
  to confirm; section [5] will show which env var is honored.
- **agy supports persistent OAuth that survives cron** — i.e. you can
  bootstrap auth once interactively, and `agy` reuses the refresh token
  forever from `~/.config/agy/` (or wherever). Verify by completing
  OAuth interactively once, then running cron a few times to confirm
  the saved token holds.
- **gemini-cli announces a deprecation date** — at which point we
  reread the migration plan and check whether the alternative is `agy`,
  some other tool, or direct API calls from Python (which we could do
  via `google-generativeai` SDK without any CLI at all).

## Verification helper

`scripts/verify_agy.py` is kept in the repo. Run it whenever `agy`
updates to re-check whether the auth situation has changed:

```bash
uv run scripts/verify_agy.py > claude.out 2>&1
```

It walks the flag surface, attempts a real call with the migration-plan
argv, then probes auth env vars in isolation to see if any one finally
lets agy run headless. If section [5] ever shows a non-OAuth response,
agy is unblocked and we can reopen the migration.

## Code reference

The dual-backend client lives in `scripts/gemini_client.py`. To
re-activate the migration:

1. Add the agy headless-auth flag/env var to `_build_agy_cmd` and/or
   to the subprocess env setup in `_execute_command`.
2. Flip `_DETECTION_ORDER` back to `["agy", "gemini"]` (or just set
   `cli_binary: "agy"` in `config.yaml` for one machine at a time).
3. Run `pytest tests/test_gemini_client_agy.py` — the agy-side test
   coverage is already there; just adjust the preference-order tests
   if you flip detection back.
4. Re-run `scripts/verify_agy.py` end-to-end on the production host.
5. Run one cron-style invocation to confirm headless auth actually
   holds: `python3 scripts/briefing_runner.py --config config.yaml
   --dry-run`.

---

*Original plan: written before agy was released, based on speculation
about flag names and a misread of gemini-cli's roadmap.*

*Verified update: May 22, 2026 — real run against agy 1.0.1 + check of
gemini-cli's actual release activity.*
