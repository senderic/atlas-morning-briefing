# Atlas Morning Briefing — Agent Context

## What This Is

A single-machine, cron-driven pipeline that fetches ArXiv papers, RSS blogs, stock quotes, and news headlines, runs them through an LLM intelligence layer for scoring/summarization/synthesis, and delivers a Kindle-optimized PDF + HTML email each morning.

## Schedule

Cron (America/Los_Angeles): `30 5 * * 1-6` — 5:30 AM Mon-Sat.
Downstream consumer: `~/sender-trades/` runs at 6:30 AM Mon-Fri.

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
- **`.gitignore` ignores all `*.md`** except a whitelist. Add new doc files to `.gitignore` whitelist.
- **Scripts in `scripts/` that are NOT pytest tests:** `test_briefing_alignment.py`, `verify_agy.py`, `audit_gemini.py`, `benchmark_*.py`, `check_weekly_state.py` — these are live diagnostics needing API keys. Test paths are pinned to `tests/`.
- **Config topics are intentional** — defense/space/AI theming is user-specific, not placeholder.

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

## Config

`config.yaml` uses `${VAR:-default}` interpolation. `.env` is loaded via python-dotenv with `override=True`. Required API keys: FINNHUB_API_KEY, BRAVE_API_KEY. Email delivery: GMAIL_USER, GMAIL_APP_PASSWORD.
