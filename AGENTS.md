# Atlas Morning Briefing — Agent Context

## What This Is

A single-machine, cron-driven pipeline that fetches ArXiv papers, RSS blogs, stock quotes, and news headlines, runs them through an LLM intelligence layer for scoring/summarization/synthesis, and delivers a Kindle-optimized PDF + HTML email each morning.

## Schedule

Cron (America/Los_Angeles): `0 6 * * 1-6` — 6:00 AM Mon-Sat.
Downstream consumer: `~/sender-trades/` runs at 6:30 AM Mon-Fri.
Quality check: `40 6 * * 1-6` — reviews what both briefings produced; `15 7 * * 0` adds a deep feed probe.

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

# Dry run (no email, no state writes)
python3 scripts/briefing_runner.py --config config.yaml --dry-run

# Tests
uv run pytest -v --tb=short
uv run pytest tests/test_briefing_runner.py -v --tb=short
```

## Critical Context

- **Active LLM backend is Gemini CLI** (not Amazon Bedrock). `config.yaml` has `gemini.enabled: true`, `bedrock.enabled: false`. The README's Bedrock framing is aspirational/legacy.
- **v0.1 runner is the active one** (`scripts/briefing_runner.py`). `briefing_runner_v2.py` is experimental.
- **Two config files exist:** `config.yaml` (main config, 10KB) and `config.json` (small model override for opencode, 100B). The shell script references `config.yaml`.
- **THREE configs for runs — blanket changes must touch all of them:** `config.yaml` (main Atlas briefing) AND `config_local.yaml` (San Diego local briefing, run via `run_briefing.sh` second invocation) both need the same model/LLM edits — e.g. 2026-08-08 heavy-tier OpenRouter model was updated in `config.yaml` but not `config_local.yaml`, so the local run kept using `deepseek/deepseek-chat`. `config.json` only overrides the opencode editor model. `scripts/openrouter_client.py` etc. hold code defaults that must match too.
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

Every LLM step has a deterministic fallback (TF-IDF scoring, flat headlines) for when Gemini CLI is unavailable.

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
