# Skill: API Resilience & Reliability

## Goal
Ensure the briefing pipeline is robust against API failures, rate limits, and transient network issues.

## Implementation Patterns

### 1. Multi-Tier Fallback
Always use tiered model access to handle outages or quota limits.
- **Pattern:** `Heavy` (Pro) -> `Medium` (Flash) -> `Light` (Flash-Lite) -> `Deterministic` (No AI).
- **Rule:** Never fail the entire run if a high-tier model is unavailable. Fall back to a lower tier or a non-AI summary.

### 2. Tenacity Retry Strategy
Use the `tenacity` library for all API calls.
- **Wait Strategy:** Use `wait_random_exponential` with jitter.
  - *Multiplier:* 30-60 (higher for Pro models).
  - *Min/Max:* Floor of 60-90s to clear 1-minute RPM windows.
- **Stop Strategy:** Use `stop_after_attempt`. Pro models get "Ultra-Persistence" (12-15 attempts).
- **Transient vs. Hard Errors:** 
  - *Transient (Retry):* `429`, `RESOURCE_EXHAUSTED`, `503`, `500`, `capacity`.
  - *Hard (Stop):* `403` (Unauthorized), `daily quota reached`, `rpd limit reached`.

### 3. API Key Rotation
Support multiple keys to bypass individual quota limits.
- **Loading:** Keys can be comma-separated in `GEMINI_API_KEY` or provided as `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, etc.
- **Rotation Trigger:** Rotate on `429` or `RESOURCE_EXHAUSTED` errors *within* the tenacity retry loop if possible, or before falling back to a lower tier.
- **Cooldown:** Implement a `key_swap_delay` (default 5-10s) with jitter when switching keys.

### 4. Budget & Safety
- **Call Budget:** Respect `max_calls_per_run` to prevent unexpected billing or accidental infinite loops.
- **Sanitization:** Always use `_sanitize_prompt_input` before sending data to an LLM to prevent prompt injection.

## Validation
- Verify that `usage_stats` in the report footer correctly tracks failed attempts and cost.
- Run tests in `tests/test_gemini_rotation.py` when modifying rotation logic.
