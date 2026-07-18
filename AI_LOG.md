# AI Operational Log

A running log of incidents, root causes, and fixes involving the LLM
backend (Gemini CLI, OpenCode CLI, Bedrock) and the intelligence layer.
Each entry should be terse enough to skim but specific enough to act on
(file:line refs, exact symptom, exact remediation). Newest entries on
top.

---

## 2026-07-18 — Empty briefing: opencode not on cron PATH, no fallback model

**Symptom:** The 06:00 cron run delivered an empty briefing (65 lines,
no executive summary, no synthesis, "AI & Tech News" section just
flattened headlines). `status.json` reported `"intelligence_enabled":
false` despite `opencode.enabled: true` in `config.yaml`. Total runtime
67s instead of the usual 5–15 min.

**Root causes (two compounding):**

1. **Opencode binary not on cron PATH.** `run_briefing.sh` line 8
   hardcoded `PATH` to nvm + system bins, but the `opencode` binary
   lives at `/home/linuxbrew/.linuxbrew/bin/opencode` (linuxbrew).
   `OpencodeClient.available` (verified via `shutil.which`) returned
   `False`, so `BriefingIntelligence.available` cascaded to `False`
   and the entire LLM layer was skipped with deterministic fallbacks.
   The cron journal confirmed it directly:
   `WARNING: Opencode binary not found on PATH`.

2. **No model fallback.** Even when opencode was reachable, a primary
   model failure (non-zero exit, empty NDJSON, or timeout) returned
   `None` immediately. There was no retry on a different model, so a
   single quota-exhaustion or transient hang on the free DeepSeek model
   would silently strip the LLM layer out of the run.

**Secondary (unrelated):** DeepXiv (`data.rag.ac.cn`) returned HTTP 200
with empty `result` arrays for every arxiv topic. Not an LLM issue; not
addressed here. Tracked separately if it persists.

**Fixes applied (committed):**

1. **Cron PATH** (`run_briefing.sh:8`) — added
   `$HOME/.linuxbrew/bin` and `/home/linuxbrew/.linuxbrew/bin` to
   `PATH`. Verified `which opencode` resolves under the new PATH.

2. **Per-tier fallback chain** (`scripts/opencode_client.py`):
   - New `DEFAULT_FALLBACK_MODELS` dict and `OpencodeClient.fallback_models`
     attribute, one list per tier (`heavy`/`medium`/`light`).
   - `invoke()` rewritten to walk the chain: primary first, then each
     fallback in order. Tries the next model on non-zero exit / empty
     NDJSON / `subprocess.TimeoutExpired`. Chain is de-duplicated so a
     primary that also appears in the fallback list is only tried once.
   - Bookkeeping: `_tier_served_by` records which model actually served
     each tier's last successful call; `_tier_fallback_hits` counts
     fallback successes. Both surface in `get_usage_summary()` so
     fallback activity is visible in the briefing footer.

3. **Config plumbing:**
   - `config.yaml` and `config.yaml.example` got `opencode.fallback_models`
     with `opencode-go/glm-5.2` as the first (and default) fallback.
     Choice rationale: glm-5.2 on OpenCode Zen has the largest free
     quota today and consistently returns in seconds when DeepSeek free
     tier is hanging.
   - `config.yaml` also got a temporary `opencode.timeout: 60` to drop
     the DeepSeek primary timeout from the default 600s to 60s so
     every failed call falls back to glm-5.2 within a minute instead
     of after 10 minutes. **TODO: revert this when DeepSeek free
     tier recovers (delete the `timeout:` line or set it back to 600).**

4. **Tests** (`tests/test_opencode_client.py`):
   - New `TestFallback` suite (11 cases): default chain populated,
     custom override, fallback on non-zero exit / empty NDJSON /
     timeout, all-models-fail returns `None`, empty-list disables
     fallback, chain de-dups primary, budget exhaustion during
     fallback stops the chain, success on primary skips fallback.
   - Suite: 43 opencode tests pass, 607 total pass.

**Validation (live rerun):** Manual rerun via
`run_briefing.sh` after fixes:
- `Opencode binary found on PATH` ✓
- DeepSeek primary timed out at 60s on every call → glm-5.2 fallback
  served all 13 LLM calls → briefing finished in 18 min with
  `intelligence_enabled: true`, `email_sent: true`, 141 lines of
  markdown vs 65 in the failed run.

**Lessons:**

1. **Cron PATH must match interactive PATH.** Any binary the pipeline
   depends on (`opencode`, `gemini`, `agy`) must be on the PATH that
   `run_briefing.sh` exports, not just on the user's interactive
   shell. When adding a new external CLI dependency, update
   `run_briefing.sh:8` in the same commit.

2. **One binary missing should not silently disable the LLM layer.**
   `OpencodeClient` already logged a `WARNING`; the runner also
   needs to surface this in `status.json`. The
   `intelligence_enabled` flag does this implicitly (it's `False`
   when `available` is `False`) but a distinct
   `llm_backend_reason` field would help downstream consumers (e.g.
   sender-trades) distinguish "disabled by config" vs "binary missing"
   vs "all models failing". **TODO: add `llm_backend_reason` to
   `status.json` (deferred).**

3. **Always have a fallback model, even for free tiers.** Tonight's
   13/13 fallback rate could have been 13/13 silent failures without
   the new chain. The cost is one extra timeout per failed call; with
   a 60s timeout that's tolerable, with a 600s timeout it's not.

4. **Long timeouts compound across multi-call pipelines.** The
   briefing runs ~10–13 LLM calls per morning. A 600s timeout on a
   dead primary model would mean 2+ hours of wasted waits before a
   single fallback fires. Strongly prefer short timeouts + immediate
   fallback over long timeouts, especially for free-tier models
   where you have no SLA.

5. **Make fallback activity observable.** The `_tier_served_by` /
   `_tier_fallback_hits` fields and the new "Model fallback activity"
   block in the usage-summary footer let you see at a glance when a
   primary is sick — without scraping journal logs. Anytime you add a
   retry/fallback mechanism, also add an observability surface for it.

---