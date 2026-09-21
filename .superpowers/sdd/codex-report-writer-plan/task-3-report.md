# Task 3 — Editorial policy, configuration, and regressions

## Result

Added the evidence-calibrated Executive Summary policy, enabled the bounded
Codex report writer in both briefing configurations, and validated enabled
Codex configuration before a cron run starts.

The prompt now treats completed incidents as context, says that duplicate
coverage corroborates an event rather than proving a trend, and permits clear
reader action only for an active closure/advisory, continuing threat, repeated
pattern, deadline, scheduled decision, or measured risk.

## Configuration and validation

`config.yaml` and `config_local.yaml` now contain identical Codex settings:

- enabled: `true`
- binary: `/home/eric/.local/bin/codex`
- model: `gpt-5.6-sol`
- reasoning effort: `high`
- timeout: `300` seconds
- call budget: `5`

Their only Codex difference is the required call-log path:
`logs/codex-calls.jsonl` for the main briefing and
`logs/local-codex-calls.jsonl` for the local briefing. Both logs are ignored.

`validate_config` rejects an enabled Codex mapping with no usable binary or
model, a nonpositive/non-numeric timeout, a nonpositive/non-integer call
budget, or a reasoning effort outside `low`, `medium`, `high`, and `xhigh`.

## TDD evidence

The following tests were written before the corresponding production/config
changes.

| Behavior | RED command and observed result | GREEN command and observed result |
| --- | --- | --- |
| Enabled Codex validation and the four editorial evidence cases | `uv run pytest -q tests/test_config_validator.py tests/test_intelligence_full.py -k 'CodexConfig or ExecutiveSummaryEditorialPolicy' --tb=short` | Same command: `11 passed, 172 deselected in 0.34s` |
| Initial RED result | `9 failed, 1 passed, 172 deselected in 0.74s`: no Codex validation existed and the delivered prompt lacked each policy condition | — |
| Checked-in YAML parity and exact defaults | `uv run pytest -q tests/test_config_validator.py::TestCodexConfig::test_checked_in_configs_enable_the_same_bounded_codex_writer --tb=short` | Included in the 11-pass GREEN run above |
| YAML parity RED result | `1 failed in 0.10s`: `KeyError: 'codex'` from `config.yaml` | — |

The four editorial regressions use an in-memory recording report writer. Each
asserts the returned executive result, proves the analysis client was not
routed for editorial writing, and inspects the actual delivered prompt for:

1. a completed, reopened boardwalk incident;
2. duplicate headlines covering that same incident;
3. a live water-contact advisory; and
4. a documented recurring shoreline closure.

No test invokes a real Codex process: the existing autouse discovery guard
keeps the executable unavailable except where the Codex-client unit tests
explicitly mock the process boundary.

## Verification

- `uv run pytest -q tests/test_config_validator.py tests/test_intelligence_full.py --tb=short` — `183 passed in 0.60s`.
- `uv run pytest -q tests/test_briefing_runner.py tests/test_briefing_runner_orchestration.py tests/test_report_writer.py tests/test_codex_client.py --tb=short` — `98 passed in 1.55s`.
- `git diff --check` — no whitespace errors.
- `git check-ignore -v logs/codex-calls.jsonl logs/local-codex-calls.jsonl` — both paths matched the new ignore entries.
- `pgrep -af '[c]odex exec'` — no process remained or was launched.
- Required full run, exactly once: `uv run pytest -q --tb=short` — `1295 passed, 3 skipped in 13.92s`.

## Files changed

- `.gitignore`
- `config.yaml`
- `config_local.yaml`
- `scripts/config_validator.py`
- `scripts/intelligence.py`
- `tests/test_config_validator.py`
- `tests/test_intelligence_full.py`
- this report

## Self-review

Reviewed the full Task 3 diff from `60af637` before commit. The YAML test and
manual diff check confirm both run configurations match except for the local
log path. Validation is confined to enabled Codex mappings and accepts the
existing client aliases while requiring the configured binary/model. The
editorial policy is delivered through the report-writer prompt; it does not
change `intelligence_enabled`, report-only routing, status, usage, JSONL
parsing, or logging. The report-writer and client regression suites confirm
there is no analysis reroute, no duplicated usage-summary path, and no
unmocked Codex subprocess.

## Concerns

None.
