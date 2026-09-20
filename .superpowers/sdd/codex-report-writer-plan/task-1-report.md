# Task 1 report: Codex CLI client

## Result

Implemented the standalone `CodexClient` in `scripts/codex_client.py` and its
focused test suite in `tests/test_codex_client.py`. The client is a
`BaseLLMClient` implementation and does not inspect authentication files or
include prompt/authentication data in call logs.

## TDD evidence

1. Wrote the focused behavior tests before the production module.
2. RED command:

   ```text
   pytest -q tests/test_codex_client.py --tb=short
   ```

   Expected failure:

   ```text
   ModuleNotFoundError: No module named 'scripts.codex_client'
   1 error during collection
   ```

3. Added the minimal client implementation.
4. GREEN command:

   ```text
   pytest -q tests/test_codex_client.py --tb=short
   ..............                                                           [100%]
   14 passed in 0.15s
   ```

## Implemented behavior

- Checks only `enabled` and the configured executable with `shutil.which`.
- Builds the required `codex exec --ephemeral --ignore-user-config
  --ignore-rules --sandbox read-only --skip-git-repo-check -C /tmp -m ...
  -c model_reasoning_effort=... --json -` command.
- Sends system and user prompts through clearly delimited stdin sections.
- Parses Codex JSONL, retaining the last completed agent message and turn
  usage while tolerating diagnostic/malformed lines when the turn is valid.
- Rejects timeout, OS error, nonzero exit, explicit failure/error events,
  incomplete turns, empty output, and exhausted budgets.
- Tracks calls, failures, latency, model, and token usage; appends best-effort
  JSONL call records when configured.
- Provides a concise Markdown usage summary.

## Verification

The full repository suite was run:

```text
pytest -q --tb=short
1268 passed, 3 skipped in 15.38s
```

The final focused suite was rerun after the last logging-boundary adjustment:

```text
pytest -q tests/test_codex_client.py --tb=short
..............                                                           [100%]
14 passed in 0.25s
```

## Files

- `scripts/codex_client.py`
- `tests/test_codex_client.py`
- This report

## Self-review

- Scope is limited to the standalone client, focused tests, and this report;
  no runner routing, intelligence prompts, YAML configuration, or validation
  was changed.
- The subprocess boundary is the only inference dependency; no retries are
  performed, so each allowed call has one bounded timeout.
- Call-log writes are isolated in a broad best-effort exception boundary and
  never affect inference.
- Prompt text is sent to the subprocess but is not written to call logs.
- `git diff --check` passed before final verification.

## Review fixes (follow-up)

The review identified four compatibility/safety gaps. I added focused tests
before changing production code. The RED command was:

```text
pytest -q tests/test_codex_client.py --tb=short
...F.F..F.....F.F                                                        [100%]
5 failed, 12 passed in 0.22s
```

The failures covered approved `reasoning_effort`/`timeout_seconds` keys,
`reasoning_output_tokens`, rejection of a top-level agent message, sanitized
nonzero-stderr logging, and usage-summary reporting.

The implementation now gives the approved keys precedence over compatibility
aliases, accounts/logs/summarizes `reasoning_output_tokens`, accepts final
text only from `item.completed` agent-message items, and records only a
sanitized error category plus exit status. Raw subprocess stderr is neither
logged nor emitted by runtime error handling.

Fix GREEN evidence:

```text
pytest -q tests/test_codex_client.py --tb=short
.................                                                        [100%]
17 passed in 0.09s

pytest -q --tb=short
1271 passed, 3 skipped in 14.28s
```

Follow-up self-review found no routing, configuration, prompt, or unrelated
file changes. `git diff --check` passed.
