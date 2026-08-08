#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Shared LLM error classification for the multi-provider fallback chain.

All backends (opencode CLI, OpenRouter HTTP, Gemini CLI) funnel their failures
through :func:`classify_error` so the composite behaves consistently:

* ``"fallback"`` — the provider is OUT OF USAGE or otherwise non-recoverable
  (insufficient balance, quota/resource exhausted, auth/payment failure, model
  not found). Advance immediately to the next model / next provider. Retrying
  an exhausted provider only wastes time and blocks delivery.
* ``"retry"`` — a transient condition (HTTP 429 rate limit, 5xx, timeout).
  Retry the same provider a bounded number of times with backoff.

Return codes are only trusted when they are unambiguous (401/402/403 always
mean "not usable right now"). For a shared 429 the body is inspected: most 429s
are transient rate limits, but "quota/resource exhausted" wording means out of
usage.
"""

from typing import Optional

# HTTP codes that always mean "out of usage" / not usable right now.
_HTTP_OUT_OF_USAGE = (401, 402, 403)

# HTTP codes that are usually transient and worth a bounded retry.
_HTTP_TRANSIENT = (408, 425, 429, 500, 502, 503, 504)

# Body/substring signals that mean out of usage (not worth retrying).
_OUT_OF_USAGE_PATTERNS = (
    "insufficient balance",
    "insufficient_balance",
    "no credit",
    "billing required",
    "payment required",
    "resource_exhausted",
    "resource exhausted",
    "quota exceeded",
    "quota reached",
    "daily limit",
    "rpd limit",
    "limit reached",
    "out of usage",
    "exceeded your",
    "does not have access",
)


def _is_out_of_usage_text(text: str) -> bool:
    lowered = (text or "").lower()
    return any(p in lowered for p in _OUT_OF_USAGE_PATTERNS)


def classify_error(
    status_code: Optional[int] = None,
    text: str = "",
) -> str:
    """Classify a provider error into ``"fallback"`` or ``"retry"``.

    Args:
        status_code: Optional HTTP status code (None for CLI/NDJSON errors).
        text: Optional error message / body / stderr text.

    Returns:
        ``"fallback"`` to skip straight to the next model/provider, or
        ``"retry"`` to retry the same provider with backoff.
    """
    if status_code is not None:
        if status_code in _HTTP_OUT_OF_USAGE:
            return "fallback"
        if status_code in _HTTP_TRANSIENT and not _is_out_of_usage_text(text):
            return "retry"
        # Any other explicit status (incl. transient-with-quota wording): if it
        # smells like out-of-usage, fall back; otherwise retry if transient.
        if status_code in _HTTP_TRANSIENT:
            return "fallback"
        return "fallback"

    # No HTTP status — rely on the message text.
    if _is_out_of_usage_text(text):
        return "fallback"
    # Free-form message without a status: transient if it looks like a
    # rate limit / overload / timeout; otherwise fall back to be safe.
    lowered = (text or "").lower()
    transient_hints = (
        "rate limit",
        "overloaded",
        "temporarily",
        "try again",
        "timeout",
        "timed out",
        "server error",
        "streaming response failed",
        "queue is full",
    )
    if any(h in lowered for h in transient_hints):
        return "retry"
    return "fallback"
