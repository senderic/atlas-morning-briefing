# Gemini Manual Run Performance Report (May 16, 2026 - 06:18 to 06:51 UTC)

## Summary of Findings

The manually triggered run successfully completed in **1987.3s (~33 minutes)**. Despite severe quota exhaustion across the Pro and Flash models, the system's multi-tier fallback architecture ensured the briefing was generated and delivered.

### 1. Model Quota & Fallback Execution
This run experienced a "quota massacre," with multiple models hitting their daily limits.

| Phase | Tier | Model | Outcome | Action |
| :--- | :--- | :--- | :--- | :--- |
| Blog Enrichment | Light | `flash-lite` | **SUCCESS** | Recovered from temporary demand spike via retry. |
| Stock Correlation| Heavy | `pro` | **FAILED** | Both keys hit Hard Quota; fell back to Medium. |
| Stock Correlation| Medium| `flash` | **SUCCESS** | Completed task as fallback for Heavy tier. |
| Synthesis | Light | `flash-lite` | **SUCCESS** | Completed initial synthesis part. |
| Synthesis | Medium| `flash` | **FAILED** | Hit Hard Quota after 3 strikes; fell back to Light. |
| Synthesis | Light | `flash-lite` | **SUCCESS** | Completed final synthesis as fallback. |

### 2. API Key Rotation Evidence
The system successfully attempted to rotate keys when hitting "Hard Quota" errors on the `heavy` tier.

**Key 0 Failure:**
```log
May 16 06:27:21 eric-NUC7i7BNHX atlas-briefing[3585198]: INFO: Tier heavy: hard quota strikes on key idx=0 reached 3; rotating to next key (rotation 1/2).
```

**Key 1 Failure (Final Exhaustion):**
```log
May 16 06:43:19 eric-NUC7i7BNHX atlas-briefing[3585198]: WARNING: Tier heavy: rotated through all 2 keys, each took 3 strikes. Giving up on this model.
May 16 06:43:19 eric-NUC7i7BNHX atlas-briefing[3585198]: INFO: --- Falling back from heavy to medium ---
```

### 3. Distribution Success
Unlike the previous run, distribution to Kindle was **SUCCESSFUL**. The logs indicate that the Kindle email address was correctly expanded to a valid recipient.

**Evidence:**
```log
May 16 06:51:21 eric-NUC7i7BNHX atlas-briefing[3585198]: INFO: Sending EPUB to Kindle: sen***@kindle.com
May 16 06:51:23 eric-NUC7i7BNHX atlas-briefing[3585198]: INFO: EPUB sent to Kindle: sen***@kindle.com
```
*Note: The successful expansion suggests the manual run was executed in a shell environment where `$KINDLE_EMAIL` was properly defined.*

### 4. Feed Parsing Issues
Persistent warnings remain for several feeds, which appear to be served as HTML/404 instead of valid RSS:
- **Anthropic**, **Meta AI**, and **MIT News AI** consistently fail with `SAXParseException: not well-formed`.

---

## Technical Observations
1. **Flash Quota:** The `flash` model is now also hitting daily limits, indicating that the current volume of requests (possibly due to fallback load) is exhausting the free tier for both Pro and Flash.
2. **Resilience:** The tiered fallback (`heavy` -> `medium` -> `light`) proved critical today, as `flash-lite` was the only model with remaining quota to finish the job.
3. **Timing:** The run took significantly longer (~33m vs ~4m) due to the extensive backoffs (up to 392s) between quota strikes.
