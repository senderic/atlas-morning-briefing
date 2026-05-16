# Gemini API & Atlas Service Log Report (May 16, 2026)

## Summary of Findings

Analysis of the `journalctl` logs for the `atlas-briefing` service from **04:00 to 04:04 UTC** reveals a mix of successful model execution, quota-based fallbacks, and delivery failures due to configuration issues.

### 1. Model Usage & Fallback Performance
The system successfully utilized the tiered model architecture, but encountered daily quota limits on the Pro model.

| Tier | Model | Result | Evidence |
| :--- | :--- | :--- | :--- |
| **Heavy** | `pro` | **FAILED** | `TerminalQuotaError: You have exhausted your daily quota on this model.` |
| **Medium** | `flash` | **SUCCESS** | `Gemini response received ... from medium` |
| **Light** | `flash-lite`| **SUCCESS** | `Gemini response received ... from light` (after retry) |

**Evidence (Fallback):**
```log
May 16 04:01:34 eric-NUC7i7BNHX atlas-briefing[3322528]: TerminalQuotaError: You have exhausted your daily quota on this model.
May 16 04:01:34 eric-NUC7i7BNHX atlas-briefing[3322528]: INFO: --- Falling back from heavy to medium ---
May 16 04:01:34 eric-NUC7i7BNHX atlas-briefing[3322528]: INFO: Invoking Gemini model: flash (tier: medium)
May 16 04:01:47 eric-NUC7i7BNHX atlas-briefing[3322528]: INFO: Gemini response received (483 chars, 227 tokens) from medium
```

### 2. API Key Rotation
The system loaded 2 API keys but did not successfully rotate to the second key when the first hit the "Hard Quota" limit. The client aborted the `heavy` tier instead of trying the sibling key.

**Evidence:**
```log
May 16 04:00:03 eric-NUC7i7BNHX atlas-briefing[3322528]: INFO: Loaded 2 API keys for rotation.
...
May 16 04:01:34 eric-NUC7i7BNHX atlas-briefing[3322528]: WARNING: Hard quota reached for heavy (msg: warning: 256-color support not detected...). Aborting retries.
```

### 3. Critical Delivery Failures
All briefing distributions failed. The `config.yaml` contains environment variable placeholders (e.g., `${KINDLE_EMAIL}`) which are not being expanded by the Python application, resulting in invalid recipient addresses.

**Evidence:**
```log
May 16 04:04:02 eric-NUC7i7BNHX atlas-briefing[3322528]: ERROR: Kindle send failed: {'-YOUR_NAME@kindle.com}': (553, b'5.1.3 The recipient address <-YOUR_NAME@kindle.com}> is not a valid RFC 5321')}
May 16 04:04:03 eric-NUC7i7BNHX atlas-briefing[3322528]: ERROR: Failed to send to ${RECIPIENT_EMAIL:-your-email@example.com}: {'-your-email@example.com}': (553, b'5.1.3 The recipient address <-your-email@example.com}> is not a valid RFC 5321 address.')}
```

### 4. Feed Parsing Warnings
Several high-profile feeds failed to parse correctly during the scan:
- **Anthropic**, **Meta AI**, and **MIT News AI** reported "not well-formed (invalid token)" errors.

---

## Technical Recommendations
1. **Fix Config Expansion:** Update `scripts/config_validator.py` or the coordinator to expand `${VAR}` syntax in `config.yaml`.
2. **Improve Key Rotation:** Modify `GeminiCLIClient` in `scripts/gemini_client.py` to attempt a key rotation even on "Hard Quota" errors before falling back to the next tier.
3. **Log Truncation Fix:** Increase the character limit when logging quota error messages to avoid capturing CLI startup warnings instead of the actual API error.
