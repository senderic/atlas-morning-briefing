#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Geographic relevance gate for fetched items.

News search APIs treat a place name as just another ranking term, so a query
like "San Diego road closure detour" happily returns Bay Area, Texas, and
Pennsylvania stories. This module drops candidates that never mention a
configured place term, before they reach the (paid) LLM ranking layer.

The term list is config-driven -- no place names are hardcoded here.

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


def is_local(
    item: Dict[str, Any],
    place_terms: Sequence[str],
    trusted_sources: Sequence[str] = (),
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> bool:
    """
    Decide whether an item is about the configured area.

    An item passes if it mentions any place term, or if it came from a source
    that only covers the area (a local outlet needs no place name in the
    headline to be local).
    """
    host = _hostname(item)
    for trusted in trusted_sources:
        trusted = trusted.lower().strip()
        if trusted and (host == trusted or host.endswith("." + trusted)):
            return True
    text = _haystack(item, fields)
    return any(term.lower().strip() in text for term in place_terms if term.strip())


def filter_by_place(
    items: Iterable[Dict[str, Any]],
    place_terms: Sequence[str],
    trusted_sources: Sequence[str] = (),
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split items into (kept, dropped) by geographic relevance.

    With no place terms configured the filter is a no-op: everything is kept,
    so an unconfigured briefing behaves exactly as before.
    """
    items = list(items)
    terms = [t for t in place_terms or [] if str(t).strip()]
    if not terms:
        return items, []

    kept, dropped = [], []
    for item in items:
        (kept if is_local(item, terms, trusted_sources, fields) else dropped).append(item)
    return kept, dropped


def apply_config_filter(
    items: Iterable[Dict[str, Any]], config: Dict[str, Any], label: str = "items"
) -> List[Dict[str, Any]]:
    """
    Apply the ``geo_filter`` config block, logging what was dropped.

    Returns the input unchanged when the block is absent or disabled.
    """
    cfg = config.get("geo_filter") or {}
    if not cfg.get("enabled"):
        return list(items)

    kept, dropped = filter_by_place(
        items,
        cfg.get("place_terms", []),
        cfg.get("trusted_sources", []),
    )
    if dropped:
        logger.info(
            "Geo filter: dropped %d/%d %s with no local reference (e.g. %s)",
            len(dropped), len(dropped) + len(kept), label,
            (dropped[0].get("title", "")[:70] or "?"),
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
