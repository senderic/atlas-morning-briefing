# Task 2 — Report-writer fallback and routing

## Result

Implemented a dedicated `ReportWriter` boundary in `scripts/report_writer.py`.
It prefers the configured Codex client for reader-facing prose, rejects empty
or chain-of-thought-leaking output, and makes one heavy-tier call through the
existing analysis chain only when Codex cannot provide acceptable prose.

`BriefingRunner` now creates the Codex client from the `codex` mapping (or its
defaults when the mapping is absent), retains `CompositeClient` as the analysis
client, and passes the writer separately into `BriefingIntelligence`.

## Routing

| Work | Client |
| --- | --- |
| Topic expansion, ranking, enrichment, paper scoring, entity analysis | Existing analysis client |
| Executive synthesis | Report writer |
| Enabled extension sections | Report writer |
| Saturday weekly deep dive | Report writer |

The runner executes the three report-writing operations when the writer is
available even when the analysis client is unavailable. Their prompts therefore
operate on raw collected data in that degraded mode.

## Status and footer

`status.json` now retains `intelligence_enabled` and separately publishes:

- `writer_enabled`
- `writer_model`
- `writer_backend`
- `writer_fallback_count`

The rendered footer appends the writer's routing-only summary after the normal
analysis usage summary, separated by a blank line. `ReportWriter` never asks
the existing fallback client for its usage summary, so composite usage is not
duplicated.

## Tests and TDD evidence

Every behavior was introduced with a focused test and a recorded failing run:

| Behavior | RED evidence | GREEN evidence |
| --- | --- | --- |
| Codex preference, leakage rejection, independent fallback, routing-only summary | `tests/test_report_writer.py`: import failed because `scripts.report_writer` did not exist | 4 passed |
| Writer-only executive synthesis and weekly deep dive | `TestReportWriterRouting`: `BriefingIntelligence.__init__` rejected `report_writer` | 2 passed |
| Runner writer construction/status/footer and raw-input extension routing | focused runner run: missing `CodexClient`/`report_writer`; raw output fell back to placeholder | 11 focused tests passed |
| Footer formatting | expected blank line between analysis and writer summary, observed concatenation | 1 passed |
| Analysis remains on existing client | mutation temporarily routed `detect_emerging_themes` to the writer; the test failed with no themes | restored route: 3 routing tests passed |

Focused changed-module regression run after integration: **319 passed**.

Final suite run (after the final preservation adjustment): **1281 passed, 3
skipped** (`uv run pytest -q`). `git diff --check` also completed with no
whitespace errors.

## Scope checks

- No YAML configuration, config validation, or editorial prompt wording was changed.
- Existing direct `BriefingIntelligence(client, config)` callers preserve their
  original client and retry behavior; the writer route is opt-in at the
  constructor and enabled by the active runner.
- Entity-analysis data is retained across the extracted report-writing phase.

## Concern

No implementation blocker. The Codex mapping remains intentionally absent from
the checked-in YAML files; Task 3 owns explicit YAML enablement/configuration.
Without that mapping, `CodexClient` uses its own defaults and the established
analysis chain remains the per-call fallback.

## Recovery audit (2026-09-20)

Recovered the staged Task 2 work from base `d739132` and audited its diff
line-by-line against the task brief. The audit confirmed that only executive
synthesis, enabled extension sections, and the weekly deep dive move to the
writer; all analysis invocations remain on `BriefingIntelligence.client`.
`ReportWriter` records one backend result per call, re-attempts Codex on every
call, and its footer text never asks the fallback client for usage.

### Additional RED/GREEN evidence

The staged suite did not directly prove independent per-call fallback routing.
Added `test_each_call_retries_codex_after_an_earlier_fallback` to
`tests/test_report_writer.py`. Its first normal run passed against the staged
implementation. A controlled mutation changing the Codex eligibility check to
skip Codex once `fallback_count` was nonzero produced the expected RED:
`AssertionError: 'First call from fallback.' != 'Second call from Codex.'`.
Restoring the eligibility check produced GREEN: `5 passed in 0.06s`.

### Verification commands and results

- `uv run pytest -q --tb=short tests/test_report_writer.py tests/test_intelligence_full.py tests/test_briefing_runner.py tests/test_briefing_runner_orchestration.py tests/test_local_briefing.py tests/test_briefing_extensions.py` — `299 passed in 3.39s` (before the recovery regression was added).
- `uv run pytest -q --tb=short tests/test_report_writer.py` — `5 passed in 0.06s` (after recovery regression and restored routing).
- `uv run pytest -q --tb=short` — `1282 passed, 3 skipped in 16.19s`.

### Files and self-review

Task files: `.gitignore`, `.superpowers/sdd/.gitignore`, this report,
`scripts/report_writer.py`, `scripts/briefing_runner.py`,
`scripts/intelligence.py`, `tests/test_report_writer.py`,
`tests/test_briefing_runner.py`, `tests/test_briefing_runner_orchestration.py`,
`tests/test_intelligence_full.py`, and `tests/test_local_briefing.py`.

Self-review found no scope expansion into YAML or editorial policy. The only
recovery change is the focused independent-call regression test. Remaining
concern: explicit Codex YAML configuration remains Task 3's concern; fallback
delivery remains intentional meanwhile.

## Review fix round 1 (2026-09-20)

Review found that an absent `codex` mapping was converted to
`{"enabled": False}`, overriding `CodexClient` defaults. Added
`test_absent_codex_mapping_preserves_codex_client_defaults` first. RED command:
`uv run pytest -q --tb=short tests/test_briefing_runner.py::TestStatus::test_absent_codex_mapping_preserves_codex_client_defaults`;
result: `Expected: CodexClient({})`, `Actual: CodexClient({'enabled': False})`
and `1 failed in 1.41s`.

GREEN minimal fix: `BriefingRunner` now passes `{}` when `config["codex"]` is
absent or not a mapping, retaining `CodexClient` defaults. Focused verification
with Codex unavailable on PATH (to keep test execution offline and
deterministic): `PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/eric/.local/bin/uv run pytest -q --tb=short tests/test_briefing_runner.py tests/test_report_writer.py tests/test_intelligence_full.py tests/test_briefing_runner_orchestration.py tests/test_local_briefing.py tests/test_briefing_extensions.py` — `301 passed in 2.45s`.
Full verification under the same offline test environment: `1283 passed, 3
skipped in 13.56s`.

Files in this fix: `scripts/briefing_runner.py`,
`tests/test_briefing_runner.py`, and this report. Self-review: no changes to
`CodexClient`, YAML, editorial policy, fallback-count semantics, or the
controller ledger. The default-enabled production behavior can invoke the
local Codex CLI when present; the test PATH removes that external dependency
while preserving unit coverage of the constructor contract.

### Test isolation follow-up

The unmocked paths were all `BriefingRunner` constructions with no `codex`
mapping that subsequently called `run()` (the orchestration and local-briefing
test fixtures, plus their derived runner configurations). They can reach
executive synthesis and therefore executable discovery. No live `pytest`,
`uv`, or `codex exec` process remained when cleanup was checked; the prior
full-suite process had already exited before the stop instruction arrived.

Added `test_default_codex_writer_never_starts_a_real_cli_process` first.
Its RED command was
`uv run pytest -q --tb=short tests/test_briefing_runner_orchestration.py::TestRunOrchestration::test_default_codex_writer_never_starts_a_real_cli_process`;
it observed a mocked `subprocess.run` call beginning `['codex', 'exec', ...]`
and failed with `Expected 'run' to not have been called` (`1 failed in
1.38s`). The minimal fixture fix is an autouse test guard that makes Codex
executable discovery unavailable. `tests/test_codex_client.py` continues to
explicitly patch discovery and `subprocess.run` for its controlled client
tests.

GREEN focused command:
`uv run pytest -q --tb=short tests/test_briefing_runner_orchestration.py::TestRunOrchestration::test_default_codex_writer_never_starts_a_real_cli_process tests/test_briefing_runner.py::TestStatus::test_absent_codex_mapping_preserves_codex_client_defaults tests/test_codex_client.py` — `19 passed in 1.24s`.
Required exact full command: `uv run pytest -q --tb=short` — `1284 passed, 3
skipped in 13.71s`; an immediate process audit found no `pytest`, `uv`, or
`codex exec` process. Files added to this follow-up: `tests/conftest.py` and
`tests/test_briefing_runner_orchestration.py`. Self-review: the guard is test
only, affects executable discovery at the external boundary, and leaves
production defaults untouched.
