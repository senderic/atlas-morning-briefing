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

1. **The fallback chain orders MODELS, not backends.** The README and CHANGELOG are written around Amazon Bedrock; treat that framing as aspirational/legacy (`scripts/bedrock_client.py` is only wired into the *experimental* v2 runner, below). The *active* chain is `llm.chains` in `config.yaml` — one ordered list of model slugs per tier, tried front to back:

   ```yaml
   llm:
     chains:
       heavy:
         - "opencode/muse-spark-1.3-contributor-free"          # free, via the CLI
         - "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free" # free, via HTTP
         - "opencode-go/deepseek-v4-pro"                       # PAID, last
   ```

   A backend is a **transport**, not a tier. Which one serves a rung is inferred from its routing prefix (`openrouter/`, `opencode/`, `opencode-go/`, `gemini/`) in `scripts/llm_chain.py`; `CompositeClient` walks the rungs and calls `client.invoke(..., model=<slug>)`. A client serves exactly the model it is handed and **never substitutes another** — a backend that picked its own would jump the chain's queue, possibly onto a paid rung the chain deliberately placed last.
   - **`openrouter`** — `scripts/openrouter_client.py`, HTTP API, `:free` slugs only. A healthy run bills **$0.00** and the usage summary flags it loudly if it doesn't.
   - **`opencode`** — `scripts/opencode_client.py`, shells out to `opencode run -m <model>`. Carries **both** free (`opencode/*` Zen contributor tier) and paid (`opencode-go/*`) models, so it is neither "the free backend" nor "the paid one".
   - **`gemini` (disabled).** `config.yaml` has `gemini.enabled: false`.

   **Order is cost — free rungs first, paid last — and that is a property of how the chain is written.** This replaced `llm.backend_priority` plus per-backend `models:`/`fallback_models:` rosters, which made a model's rung a consequence of who hosted it: the loop visited each backend once, so `opencode/muse-spark-*` sat configured as a free heavy primary and served **zero** calls, because reaching it meant getting past OpenRouter's entire roster first. If you add a model, put it in `llm.chains` at the rung its cost earns.

   Two rules the chain must keep: tiers are **distinct and monotonic** (heavy > medium > light in size and context window), and within a tier a rung may only **degrade down** a weight class, never up. `config_validator.py` warns when two tiers lead with the same model — that collapse is real, not hypothetical: `minimax/minimax-m3:free` stopped being free, medium fell through to light's model, and both tiers quietly became one. `config/model_capabilities.yaml` records per-model reasoning control and must list **every** rung; on OpenRouter the only parameter that actually suppresses reasoning tokens is `reasoning: {enabled: false}` — `reasoning_effort` is accepted but is a no-op.

   **Each rung gets its own window (`llm.rung_timeout_seconds`), sized `timeout x (1 + max_retries_per_model)` for its backend.** This is much safer than what it replaced: a per-*backend* window had to cover that backend's entire internal chain, and a backend whose worst case overran it was cut off mid-chain on every call — how the paid backstop was once found dead on arrival. `CompositeClient` still warns at startup when a rung cannot finish in its window, or names a backend that isn't enabled. Don't ignore either.

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
- `llm_client.py` — `BaseLLMClient` interface + `ReasoningControlMixin` (capability lookup, CoT-leakage detection). `composite_client.py` — walks a tier's model chain rung by rung, each under its own timeout, and reports in the usage summary when a tier had to fall past its first choice (a quiet fallback is how a chain rots unnoticed). `openrouter_client.py` / `opencode_client.py` — the two live backends. `llm_errors.py` — shared `classify_error()` (`"fallback"` vs `"retry"`).
- `llm_chain.py` — the chain itself: `resolve_backend()` (prefix → transport), `build_model_chains()` (config → per-tier `Rung`s), `build_clients()` (enabled transports), and `apply_pins()`. Shared by the runner and the quality checker so the two cannot drift on order, which they once did — the checker billed the paid backstop on every daily run long after the briefing had gone free-first.
- `preflight_model_check.py` — probes each tier's chain before the run and writes `.model-availability.json`, which the runner uses to pin the rung a tier starts at (ignored if older than 6h; skipped rungs rotate to the back rather than being dropped, since a 05:45 probe is a snapshot, not a verdict). It reads the chain **from config**, never a local table, and a tier's result may only name a rung from that tier's own chain. Skipping is **per rung**, via `llm.preflight_skip` (the paid prefixes) — not per backend, which is how a free model sharing a mostly-paid transport went unprobed and unused.
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
