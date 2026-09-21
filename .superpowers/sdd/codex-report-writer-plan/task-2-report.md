# Task 2 — Report-writer fallback and routing

## Result

Implemented a dedicated `ReportWriter` boundary in `scripts/report_writer.py`.
It prefers the configured Codex client for reader-facing prose, rejects empty
or chain-of-thought-leaking output, and makes one heavy-tier call through the
existing analysis chain only when Codex cannot provide acceptable prose.

`BriefingRunner` now creates the Codex client from the `codex` mapping (or a
disabled mapping when it is absent), retains `CompositeClient` as the analysis
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
the checked-in YAML files; Task 3 owns enabling/configuring it. Without that
mapping, the writer is disabled as primary and the existing analysis chain
continues serving the three prose calls.

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
concern: Codex stays disabled in checked-in configurations until Task 3 owns
the configuration change; fallback delivery remains intentional meanwhile.
