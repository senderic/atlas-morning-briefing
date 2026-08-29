#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
URL normalization for deduplication.

Sources hand back the same article under cosmetically different URLs. Keying a
dedup set on the raw string therefore lets duplicates straight through: on
2026-08-29 the local briefing printed "The Best Things to Do in San Diego This
Weekend" twice, the two URLs differing only by a trailing slash.

``normalize_url`` collapses the differences that never denote a different
document — scheme, ``www.``, a trailing slash, a fragment, and the usual
tracking parameters — while leaving meaningful query strings intact, because
``?page=2`` and ``?id=17`` genuinely are different pages.
"""

from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Parameters that identify a referral, not a document.
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid",
    "igshid", "ref", "ref_src", "s", "src", "source", "cmpid",
    "campaign_id", "spm", "at_medium", "at_campaign",
}


def _is_tracking(key: str) -> bool:
    lowered = key.lower()
    return lowered in _TRACKING_PARAMS or lowered.startswith(_TRACKING_PREFIXES)


def normalize_url(url: Optional[str]) -> str:
    """Return a canonical form of ``url`` for equality comparison.

    Falls back to a stripped, lower-cased copy of the input when the URL
    cannot be parsed, so a malformed value still dedupes against itself
    rather than raising.
    """
    if not url:
        return ""
    raw = url.strip()
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw.lower()
    if not parts.netloc:
        # Not an absolute URL (a bare path or junk); compare it literally.
        return raw.lower().rstrip("/")

    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parts.port:
        host = f"{host}:{parts.port}"

    path = parts.path.rstrip("/")

    query = urlencode(
        [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
         if not _is_tracking(k)]
    )

    # Scheme and fragment are dropped: http/https and #section do not make it
    # a different document.
    return urlunsplit(("", host, path, query, ""))
