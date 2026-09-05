# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-machine, cron-driven pipeline that fetches ArXiv papers, RSS blogs, stock quotes, and news headlines, runs them through an LLM "intelligence" layer for scoring/summarization/synthesis, and delivers a Kindle-optimized PDF + HTML email each morning. Prototype/testing project — not production-hardened.

## Commands

```bash
# Setup (CI uses uv; local dev can use either)
uv sync --all-extras --dev          # what CI runs
pip install -e ".[dev]"             # equivalent for a venv workflow

# Tests — testpaths is pinned to tests/ in pyproject.toml
uv run pytest -v --tb=short          # full suite (matches CI)
uv run pytest tests/test_paper_scorer.py -v               # one module
uv run pytest tests/test_paper_scorer.py::TestScore::test_has_code -v  # one test

# Run the pipeline (active runner is briefing_runner.py)
python3 scripts/briefing_runner.py --config config.yaml --dry-run     # no email, no state writes
python3 scripts/briefing_runner.py --config config.yaml               # full run + delivery
python3 scripts/briefing_runner.py --config config.yaml --log-level DEBUG
# Console entry point (same thing): morning-briefing --config config.yaml --dry-run
# Cron wrapper: run_briefing.sh (resolves paths, uses .venv/bin/python3, pipes to logger)
```

There is no separate lint step; the project follows PEP 8 + type hints by convention, not enforced by a linter in CI.

## Critical context (read before editing)

These are the things that mislead a fresh reader of this repo:

1. **The LLM backend is OpenRouter free models, not Amazon Bedrock and no longer the Gemini CLI.** The README and CHANGELOG are written around Amazon Bedrock; treat that framing as aspirational/legacy (`scripts/bedrock_client.py` is only wired into the *experimental* v2 runner, below). The *active* chain is built in `briefing_runner.py` from `llm.backend_priority`, default `["openrouter", "gemini", "opencode"]`, and wrapped in a `CompositeClient` that tries each backend in order:
   - **`openrouter` (primary, free).** `scripts/openrouter_client.py` calls OpenRouter's HTTP API with a per-tier roster of `:free` models. A healthy run bills **$0.00**, and the usage summary flags it loudly if it doesn't.
   - **`gemini` (disabled).** `config.yaml` has `gemini.enabled: false`.
   - **`opencode` (last-resort, PAID).** `scripts/opencode_client.py` shells out to the `opencode` CLI using `opencode-go/*` models. Reached only when every free model for the tier has failed.

   **Order is cost — free first, paid last.** If you add a backend, put it in `llm.backend_priority` deliberately.

   Two rules the model roster must keep: tiers are **distinct and monotonic** (heavy > medium > light in size and context window), and a tier's fallbacks may only **degrade down** a tier, never up. `config/model_capabilities.yaml` records per-model reasoning control; on OpenRouter the only parameter that actually suppresses reasoning tokens is `reasoning: {enabled: false}` — `reasoning_effort` is accepted but is a no-op.

   **Every backend's internal retry+fallback chain must fit inside `composite.timeout_seconds`.** The composite gives each backend ONE window for its whole chain, so a backend whose worst case (`timeout x (1 + max_retries_per_model) x models-in-chain`) exceeds that window is killed mid-chain on every call and can silently never serve a request. `CompositeClient` logs a warning at startup when this is violated — do not ignore it.

2. **Two orchestrators exist.** `scripts/briefing_runner.py` is the **v0.1 single-pass runner — this is the one in use** (referenced by `run_briefing.sh`, the `morning-briefing` entry point, and all the `tests/test_briefing_runner*.py`). `scripts/briefing_runner_v2.py` is an **experimental** coordinator + parallel-workers redesign (`scripts/workers/`) still wired to `BedrockClient`. Don't assume changes to one apply to the other.

3. **`scripts/` contains non-unit scripts that need live API keys.** `test_briefing_alignment.py`, `verify_agy.py`, `audit_gemini.py`, `benchmark_*.py`, and `check_weekly_state.py` are live diagnostics, **not** pytest tests. `pyproject.toml` pins `testpaths = ["tests"]` specifically so these are never collected, and CI has an explicit guard that fails if `test_briefing_alignment` gets discovered. **Keep `testpaths = ["tests"]`** — the real unit tests live only in `tests/`.

4. **`.gitignore` ignores all `*.md` except a whitelist.** New top-level markdown docs won't be committed unless added to the `!<name>.md` whitelist in `.gitignore` (this CLAUDE.md was added to it). Generated artifacts (`*.pdf`, `*.epub`, `*.log`, `status.json`, `.atlas-state.json`, `*.json` state files, `logs/*.jsonl`) are all gitignored — do not commit run outputs.

5. **Notifications go out by email, never Telegram.** The user does not use Telegram. Alerts (quality check, failures) reuse `scripts/email_distributor.py` — `EmailDistributor.send_html_email()` with `GMAIL_USER`/`GMAIL_APP_PASSWORD` — the same path that delivers the briefing. `scripts/send_briefing_telegram.py` is dead weight from an earlier machine; don't build on it.

6. **`config.yaml` is a live, checked-in config with a defense/military topic profile.** That theming (autonomous weapons, ISR, defense contractors, defense-tilted stock watchlist) is intentional and user-specific, not placeholder text. `config.yaml.example` is the generic template. Don't "fix" config.yaml's topics.

## Architecture

**Pipeline (in `BriefingRunner.run()`, `briefing_runner.py:952`):**

```
load .atlas-state.json (cross-day memory)
  → [LLM] expand arxiv topics
  → parallel fetch: papers ‖ blogs ‖ stocks  (ThreadPoolExecutor, max_workers from config)
  → [LLM] generate dynamic news queries → fetch news
  → dedup: cross-section (news×blogs by domain/title) → similar papers (>~85% SequenceMatcher) → cross-day (vs yesterday's state)
  → [LLM] enrich: paper summaries+scoring, news/blog ranking, stock-news correlation, emerging themes, synthesis, (Sat) weekly deep dive
  → reproduction-feasibility gate (repro_min_score/25, drops weak papers)
  → generate markdown → PDF (ReportLab) / EPUB → deliver (Gmail SMTP: PDF to Kindle, HTML to recipients)
  → write status.json + .atlas-state.json (incl. accumulated weekly_items)
```

Every `[LLM]` step has a **deterministic fallback** — when no LLM backend is available (`intelligence.available is False`), the pipeline still fetches, scores papers via TF-IDF (`paper_scorer.py`), and delivers. Preserving this graceful degradation is a hard requirement (see `GEMINI.md`).

**ArXiv is scanned sequentially and paced** (`arxiv_request_delay`, default 3s), matching upstream. It briefly ran 8 topics concurrently, which exceeds what `export.arxiv.org` tolerates — 20/20 topic searches failed with 429s and the briefing shipped with no papers. Don't reintroduce a thread pool there.

**Module map (`scripts/`):**
- `briefing_runner.py` — orchestrator, dedup logic, markdown generation, state I/O, CLI (`--config` required, `--dry-run`, `--log-level`).
- `intelligence.py` — `BriefingIntelligence`: all prompts and LLM-powered features; `_sanitize_prompt_input()` strips injection markers before embedding external text in prompts.
- `gemini_client.py` — `GeminiCLIClient`: tiered model dispatch (`heavy=pro`, `medium=flash`, `light=flash-lite`), retry/key-rotation, per-call cost logging to `logs/gemini-calls.jsonl`, and **dual-binary support** via `BINARY_PROFILES` (`gemini` and `agy`/Antigravity). Auto-detect prefers `gemini`; `agy` is opt-in via `gemini.cli_binary: "agy"`. See `MIGRATION_PLAN_ANTIGRAVITY.md` for why `agy` is not cron-viable (OAuth-only in 1.0.1).
- `arxiv_scanner.py` — `create_scanner()` factory returns DeepXiv SDK scanner (semantic search) with automatic fallback to the legacy `ArxivScanner` (ArXiv API) when `deepxiv-sdk` is absent.
- `llm_client.py` — `BaseLLMClient` interface + `ReasoningControlMixin` (capability lookup, CoT-leakage detection). `composite_client.py` — ordered multi-backend fallback with a per-backend timeout and the startup budget guard. `openrouter_client.py` / `opencode_client.py` — the two live backends. `llm_errors.py` — shared `classify_error()` (`"fallback"` vs `"retry"`).
- `preflight_model_check.py` — probes the tiered roster before the run and writes `.model-availability.json`, which the runner uses to pin a per-tier model (ignored if older than 6h). It reads the roster **from config**, never a local table, and a tier's result may only name a model from that tier's own chain. Backends with `preflight_check: false` (the paid one) are skipped.
- `blog_scanner.py`, `stock_fetcher.py` (Finnhub), `news_aggregator.py` (Brave) — data collectors.
- `paper_scorer.py` — deterministic TF-IDF scoring: `has_code×7 + topic_match×3 + recency×2 + citation×1`, minus infra/theory penalties.
- `pdf_generator.py` (ReportLab, Kindle 6×8"), `epub_generator.py`, `email_distributor.py` (SMTP, nh3-sanitized HTML, masks addresses in logs).
- `config_validator.py` — `validate_config()` + `check_environment()`, run at startup before any API calls.

**Config & env:** `config.yaml` supports `${VAR:-default}` interpolation. The runner loads `.env` via `python-dotenv` with `override=True` (`.env` wins over shell env). Live runs need `FINNHUB_API_KEY` and `BRAVE_API_KEY`; delivery needs `GMAIL_USER`/`GMAIL_APP_PASSWORD` and the `*_EMAIL` addresses (see `.env.example`). The runner reads the LLM config from `config["gemini"]`, falling back to `config["bedrock"]`.

## Conventions

- **Commit messages:** prefix-based — `feat:`, `fix:`, `refactor:` (existing history convention).
- **Adding a scanner:** new `scripts/<x>_scanner.py` with a class + standalone `argparse` CLI, wire into `briefing_runner.py`, add `config.yaml` keys, add `tests/test_<x>_scanner.py`, update `references/config_guide.md`.
- **Adding an intelligence feature:** add the method to `intelligence.py`, pick a tier (light/medium/heavy), **handle the `not self.available` fallback**, wire into the runner's intelligence section and into `generate_markdown_briefing`, and test with a mocked `GeminiCLIClient`.
- **Externalize prompt wording into config — never hardcode topics/domain in prompts.** The briefing's domain framing (audience, topic area, "landscape") is user-specific and lives in `config.yaml` under `briefing_profile` (`domain`, `audience`, `landscape`), read once in `BriefingIntelligence.__init__` into `self.briefing_*` with generic defaults. Prompts interpolate those attributes (e.g. `f"a daily {self.briefing_domain} briefing"`) instead of baking in words like "defense" or "AI/tech". When you add a prompt that names a topic, audience, or field, pull it from config (extend `briefing_profile` if needed), keep a sensible generic default, and mirror the key in `config.yaml.example`. Keep `SYSTEM_PROMPT` domain-neutral — it ships to every call.
- **Fork-only sections go in `extension_sections`, not in `intelligence.py`.** Upstream deleted "Solo Founder Angle" and "Agent Cost-Optimization Play"; carrying their prompt text inside the shared modules put the fork's most-edited prose in the merge path, and it silently went stale (the cost section was still writing Amazon Bedrock advice long after the pipeline moved to OpenRouter). Those sections are now declared as data in `config.yaml` under `extension_sections` and rendered by `scripts/briefing_extensions.py`; `briefing_runner.py` keeps only a generic loop. Retargeting one — different reader, stack, fields, or rules — is a config edit. Grounding rules (cite only `<signals>`, today's date, no leaked reasoning) are appended automatically, so don't repeat them per section. `features.<key>: false` still disables a section.
- **Observability contract (`GEMINI.md`):** every run updates `status.json`; scanners report found/processed counts and append failures to the `errors` list rather than crashing the pipeline.
- `--dry-run` must never send email or write state files.
