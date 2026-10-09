# Atlas Morning Briefing — Agent Context

## What This Is

A single-machine, cron-driven pipeline that fetches ArXiv papers, RSS blogs, stock quotes, and news headlines, runs them through an LLM intelligence layer for scoring/summarization/synthesis, and delivers a Kindle-optimized PDF + HTML email each morning.

## Schedule

Cron (America/Los_Angeles): main Atlas at `0 6 * * 1-6`; San Diego local at `0 7 * * 1-6` so the ~6:30 AM Axios newsletter is available.
Downstream consumer: `~/sender-trades/` runs at 6:28 AM Mon-Fri, after Atlas and before the local briefing.
Quality check: chained after the 7:00 AM local run; Saturday adds the deep feed probe.

## Active Binary Paths

| Binary | Location |
|--------|----------|
| Python venv | `~/.venv/bin/python3` (3.12.3) |
| Gemini CLI | `~/.nvm/versions/node/v20.19.5/bin/gemini` |
| opencode | `/home/linuxbrew/.linuxbrew/bin/opencode` (v1.18.3) |

## Key Commands

```bash
# Full run
./run_briefing.sh

# Dry run (no email; redirect output/status/state paths for isolated previews)
python3 scripts/briefing_runner.py --config config.yaml --dry-run

# Tests
uv run pytest -v --tb=short
uv run pytest tests/test_briefing_runner.py -v --tb=short
```

## Critical Context

- **Active analysis uses the model chains in `llm.chains`.** Every tier starts with a Codex CLI rung (heavy `codex/gpt-5.6-sol`, medium `codex/gpt-5.6-terra`, light `codex/gpt-5.6-luna`), with free NVIDIA NIM and OpenRouter rungs behind it as fallback. OpenCode, Gemini and Bedrock are disabled. Codex CLI also writes the reader-facing report prose, through a separate client with its own call budget (`codex.max_calls_per_run`; the chain's is `codex.chain.max_calls_per_run`). NVIDIA reasoning budgets are bounded so reasoning cannot consume the entire answer allowance.
- **v0.1 runner is the active one** (`scripts/briefing_runner.py`). `briefing_runner_v2.py` is experimental.
- **Two config files exist:** `config.yaml` (main config, 10KB) and `config.json` (small model override for opencode, 100B). The shell script references `config.yaml`.
- **THREE configs for runs — blanket model/LLM changes must touch all of them:** `config.yaml` (Atlas), `config_local.yaml` (San Diego), and `config_finance.yaml` (finance). Also keep code defaults and `config/model_capabilities.yaml` aligned. `config.json` only overrides the OpenCode editor model.
- **`.gitignore` ignores all `*.md`** except a whitelist. Add new doc files to `.gitignore` whitelist.
- **Scripts in `scripts/` that are NOT pytest tests:** `test_briefing_alignment.py`, `verify_agy.py`, `audit_gemini.py`, `benchmark_*.py`, `check_weekly_state.py` — these are live diagnostics needing API keys. Test paths are pinned to `tests/`.
- **Config topics are intentional** — defense/space/AI theming is user-specific, not placeholder.
- **Notifications go out by EMAIL, never Telegram.** The user does not use Telegram. Anything that needs to reach him — quality-check alerts, failure notices — reuses `scripts/email_distributor.py` (`EmailDistributor.send_html_email`, credentials in `GMAIL_USER` / `GMAIL_APP_PASSWORD`), the same path that delivers the briefing itself. One delivery mechanism to keep working, not two. `scripts/send_briefing_telegram.py` is dead weight from an earlier machine (it hardcodes `/home/ubuntu/...` paths) — do not build on it.

## Pipeline

```
load .atlas-state.json → [LLM] expand arxiv topics
  → parallel fetch: papers ‖ blogs ‖ stocks
  → [LLM] generate news queries → fetch news
  → dedup: cross-section × similar papers × cross-day
  → [LLM] enrich: summaries, scoring, correlation, themes, synthesis
  → reproduction gate → generate markdown → PDF/EPUB → email
  → write status.json + .atlas-state.json
```

Analysis has deterministic fallbacks (TF-IDF ranking and actual source excerpts) when all rungs fail. `status.json` records completed `intelligence_calls` and `intelligence_degraded`; credential presence alone does not prove analysis succeeded.

`--dry-run` skips distribution but still writes report artifacts, status and state. For reviews, use a copied config with separate `output_dir`, `status_file_path`, `state_file_path`, and call-log paths, and disable snapshot writes.

Email newsletters use an explicit sender allowlist, read-only IMAP and BODY.PEEK. Local digests are split into stories; newsletter ranking complements selected news. Relative dates are anchored to the issue date, and official NWS alert windows take precedence in the executive summary. Union-Tribune newsletter text is available; its e-Edition requires an authenticated subscriber session.

## Pipeline Outputs (Consumed by sender-trades)

| Output | Path | Purpose |
|--------|------|---------|
| Briefing markdown | `briefings/Atlas-Briefing-YYYY.MM.DD.md` | Parsed for tickers, news, blogs, sentiment |
| Status JSON | `status.json` | `intelligence_enabled` flag, counts, errors |
| Snapshots | `snapshots/YYYY-MM-DD/{finnhub,brave,rss}.json` | Reused to avoid duplicate API calls |
| .env | `.env` | API keys sourced by sender-trades shell script |

## Incident History

See `AI_LOG.md` for full details. Key incident:

- **2026-07-18:** Cron PATH missing linuxbrew → opencode not found → empty briefing. Fixed by adding linuxbrew to `run_briefing.sh` PATH and adding fallback model chain.

## Quality Check

`scripts/quality_check.py` (wrapper: `run_quality_check.sh`) reviews what the
briefings actually produced, because the pipeline reports `errors: []` for a
briefing that is well-formed and wrong. Three layers:

1. **Source health** (`scripts/source_health.py`) — harvests per-feed and
   per-query yield from journald into `logs/source-health.jsonl` and separates
   the four things "zero items" can mean: dead URL, feed frozen at HTTP 200,
   yield collapse, and a blogger who posts twice a year.
2. **Report invariants** (`scripts/report_invariants.py`) — deterministic scan of
   the rendered markdown: past-dated events, leaked model rationale, blocked
   press-release portals, out-of-area items, near-duplicates.
3. **LLM judge** — scores a per-pipeline rubric from `quality_check.judge.dimensions`.

Read a morning's result with `journalctl -t quality-check --since today -o cat`.
Exit codes: 0 clean, 1 CRITICAL findings, 2 the checker itself failed.

**The governing rule when adding a check: a healthy day must be silent.**
Thresholds compare a source against its *own* history, never an absolute. A
check that cries wolf on a working system teaches the reader to ignore it, and
the next real alarm goes unread. See `references/quality_monitoring_design.md`.

## Config

`config.yaml` uses `${VAR:-default}` interpolation. `.env` is loaded via python-dotenv with `override=True`. Required API keys: FINNHUB_API_KEY, BRAVE_API_KEY. Email delivery: GMAIL_USER, GMAIL_APP_PASSWORD.
