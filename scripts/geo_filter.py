#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Geographic relevance gate for fetched items.

News search APIs treat a place name as just another ranking term, so a query
like "San Diego road closure detour" happily returns Bay Area, Texas, and
Pennsylvania stories. This module drops candidates that never mention a
configured place term, before they reach the (paid) LLM ranking layer.

A place-term match alone is not enough to prove an item is trustworthy: a
pay-to-publish press-release portal will happily run marketing copy that
mentions the target place by name. ``blocked_sources`` lets such sources be
excluded outright -- a blocked source is dropped even if it mentions a place
term or is also listed as trusted; blocked always wins.

The term and source lists are config-driven -- no place names or source
names are hardcoded here.

Usage:
    python3 scripts/geo_filter.py --config config_local.yaml --input news.json
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FIELDS = ("title", "description", "snippet", "summary", "url", "source")


def _haystack(item: Dict[str, Any], fields: Sequence[str]) -> str:
    parts = []
    for field in fields:
        value = item.get(field)
        if isinstance(value, str) and value:
            parts.append(value)
    return " ".join(parts).lower()


def _hostname(item: Dict[str, Any]) -> str:
    source = (item.get("source") or "").lower()
    if source:
        return source
    url = item.get("url") or ""
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _host_matches(host: str, sources: Sequence[str]) -> bool:
    """Exact host match, or host is a subdomain of a configured entry."""
    for entry in sources:
        entry = entry.lower().strip()
        if entry and (host == entry or host.endswith("." + entry)):
            return True
    return False


def is_local(
    item: Dict[str, Any],
    place_terms: Sequence[str],
    trusted_sources: Sequence[str] = (),
    fields: Sequence[str] = DEFAULT_FIELDS,
    blocked_sources: Sequence[str] = (),
) -> bool:
    """
    Decide whether an item is about the configured area.

    An item passes if it mentions any place term, or if it came from a source
    that only covers the area (a local outlet needs no place name in the
    headline to be local). A source in ``blocked_sources`` is rejected
    outright, even if it mentions a place term or is also listed as trusted
    -- blocked beats both place-term matches and trusted sources.

    An empty (or blank-only) ``place_terms`` means "no geographic constraint
    applies" -- everything passes -- rather than "nothing is local"; this
    lets ``blocked_sources`` be used on its own, with no place terms
    configured, to filter untrustworthy sources globally.
    """
    host = _hostname(item)
    if _host_matches(host, blocked_sources):
        return False
    terms = [t for t in place_terms or [] if str(t).strip()]
    if not terms:
        return True
    if _host_matches(host, trusted_sources):
        return True
    text = _haystack(item, fields)
    return any(term.lower().strip() in text for term in terms)


def filter_by_place(
    items: Iterable[Dict[str, Any]],
    place_terms: Sequence[str],
    trusted_sources: Sequence[str] = (),
    fields: Sequence[str] = DEFAULT_FIELDS,
    blocked_sources: Sequence[str] = (),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split items into (kept, dropped) by geographic relevance.

    With no place terms and no blocked sources configured the filter is a
    complete no-op: everything is kept, so an unconfigured briefing behaves
    exactly as before. With ``blocked_sources`` configured but no
    ``place_terms``, blocked items are dropped and everything else is kept
    (no place terms means no geographic constraint, not "nothing is local").
    ``blocked_sources`` always takes precedence over both a place-term match
    and ``trusted_sources``.
    """
    items = list(items)
    terms = [t for t in place_terms or [] if str(t).strip()]
    blocked = [b for b in blocked_sources or [] if str(b).strip()]
    if not terms and not blocked:
        return items, []

    kept, dropped = [], []
    for item in items:
        is_kept = is_local(item, terms, trusted_sources, fields, blocked_sources)
        (kept if is_kept else dropped).append(item)
    return kept, dropped


def _classify_dropped(
    dropped: Sequence[Dict[str, Any]], blocked_sources: Sequence[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a ``filter_by_place`` drop list into (blocked, no_local_reference).

    An item lands in ``blocked`` if its resolved host matches a configured
    blocked source; everything else was dropped for lacking a place-term
    match, so it lands in ``no_local_reference``. This re-derives the reason
    ``filter_by_place`` already used internally, purely for logging -- it
    does not change what was dropped or kept.
    """
    blocked_items, no_local_items = [], []
    for item in dropped:
        target = blocked_items if _host_matches(_hostname(item), blocked_sources) else no_local_items
        target.append(item)
    return blocked_items, no_local_items


def apply_config_filter(
    items: Iterable[Dict[str, Any]], config: Dict[str, Any], label: str = "items"
) -> List[Dict[str, Any]]:
    """
    Apply the ``geo_filter`` config block, logging what was dropped.

    Returns the input unchanged when the block is absent or disabled. Reads
    ``place_terms``, ``trusted_sources``, and ``blocked_sources`` from the
    block; an item from a blocked source is dropped even if it mentions a
    place term or is also listed as trusted. Drops are logged with the
    correct reason -- "no local reference" vs. "from blocked sources" --
    reported as separate lines, each only when that reason produced at
    least one drop.
    """
    cfg = config.get("geo_filter") or {}
    if not cfg.get("enabled"):
        return list(items)

    blocked_sources = cfg.get("blocked_sources", [])
    kept, dropped = filter_by_place(
        items,
        cfg.get("place_terms", []),
        cfg.get("trusted_sources", []),
        blocked_sources=blocked_sources,
    )
    if dropped:
        total = len(dropped) + len(kept)
        blocked_dropped, no_local_dropped = _classify_dropped(dropped, blocked_sources)
        if no_local_dropped:
            logger.info(
                "Geo filter: dropped %d/%d %s with no local reference (e.g. %s)",
                len(no_local_dropped), total, label,
                (no_local_dropped[0].get("title", "")[:70] or "?"),
            )
        if blocked_dropped:
            example = blocked_dropped[0]
            logger.info(
                "Geo filter: dropped %d/%d %s from blocked sources (e.g. %s [%s])",
                len(blocked_dropped), total, label,
                (example.get("title", "")[:70] or "?"),
                _hostname(example) or "?",
            )
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter items by place terms")
    parser.add_argument("--config", required=True, help="Config file with geo_filter block")
    parser.add_argument("--input", required=True, help="JSON file with a list of items")
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    with open(args.input, encoding="utf-8") as f:
        items = json.load(f)

    kept = apply_config_filter(items, config, label="items")
    print(json.dumps(kept, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
