#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Configuration validator.

Validates config.yaml values at startup to catch errors early.
"""

import logging
import os
import shutil
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _validate_interest_graph(
    graph: Dict[str, Any], errors: List[str], warnings: List[str]
) -> None:
    """Validate the interest_graph taxonomy DAG structure."""
    max_dynamic = graph.get("max_dynamic_queries")
    if max_dynamic is not None and (
        not isinstance(max_dynamic, int) or max_dynamic < 1 or max_dynamic > 50
    ):
        errors.append(
            f"'interest_graph.max_dynamic_queries' must be an integer (1-50), "
            f"got {max_dynamic!r}"
        )

    roots = graph.get("roots")
    if not isinstance(roots, list) or not roots:
        errors.append("'interest_graph.roots' must be a non-empty list")
        return

    seen_ids = set()

    def _validate_node(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            errors.append(f"{path} must be a dict")
            return
        node_id = node.get("id")
        query = node.get("query")
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"{path}.id must be a non-empty string")
        elif node_id in seen_ids:
            errors.append(f"duplicate interest_graph node id: '{node_id}'")
        else:
            seen_ids.add(node_id)
        if not isinstance(query, str) or not query:
            errors.append(f"{path}.query must be a non-empty string")
        priority = node.get("priority")
        if priority is not None and not isinstance(priority, (int, float)):
            errors.append(f"{path}.priority must be a number")
        children = node.get("children")
        if children is not None:
            if not isinstance(children, list):
                errors.append(f"{path}.children must be a list")
            else:
                for i, child in enumerate(children):
                    _validate_node(child, f"{path}.children[{i}]")

    for i, root in enumerate(roots):
        _validate_node(root, f"interest_graph.roots[{i}]")


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration loaded from YAML.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
        is_valid is True if no critical errors found.
    """
    errors = []
    warnings = []

    # --- Required fields ---
    if not isinstance(config.get("arxiv_topics"), list):
        errors.append("'arxiv_topics' must be a list of strings")
    elif not config["arxiv_topics"]:
        warnings.append("'arxiv_topics' is empty -- no papers will be scanned")

    # --- Type checks ---
    int_fields = {
        "arxiv_days_back": (1, 365),
        "max_papers": (1, 200),
        "max_blogs": (1, 100),
        "max_news": (1, 100),
        "num_paper_picks": (1, 20),
    }
    for field, (min_val, max_val) in int_fields.items():
        value = config.get(field)
        if value is not None:
            if not isinstance(value, int):
                errors.append(f"'{field}' must be an integer, got {type(value).__name__}")
            elif value < min_val or value > max_val:
                warnings.append(f"'{field}' value {value} is outside recommended range ({min_val}-{max_val})")

    # --- Blog feeds ---
    feeds = config.get("blog_feeds")
    if feeds is not None:
        if not isinstance(feeds, list):
            errors.append("'blog_feeds' must be a list")
        else:
            for i, feed in enumerate(feeds):
                if not isinstance(feed, dict):
                    errors.append(f"blog_feeds[{i}] must be a dict with 'name' and 'url'")
                elif not feed.get("name") or not feed.get("url"):
                    errors.append(f"blog_feeds[{i}] missing 'name' or 'url'")

    # --- Stocks ---
    stocks = config.get("stocks")
    if stocks is not None:
        if not isinstance(stocks, list):
            errors.append("'stocks' must be a list of ticker symbols")
        elif len(stocks) > 30:
            warnings.append(
                f"'stocks' has {len(stocks)} tickers. "
                "Finnhub free tier allows 60 calls/min; consider reducing."
            )

    # --- News queries ---
    queries = config.get("news_queries")
    if queries is not None:
        if not isinstance(queries, list):
            errors.append("'news_queries' must be a list of strings")

    # --- Interest graph (taxonomy DAG) ---
    interest_graph = config.get("interest_graph")
    if interest_graph is not None:
        if not isinstance(interest_graph, dict):
            errors.append("'interest_graph' must be a mapping")
        else:
            _validate_interest_graph(interest_graph, errors, warnings)

    # --- Happenings queries ---
    happenings_queries = config.get("happenings_queries")
    if happenings_queries is not None:
        if not isinstance(happenings_queries, list):
            errors.append("'happenings_queries' must be a list of strings")
        elif not all(isinstance(q, str) for q in happenings_queries):
            errors.append("'happenings_queries' must contain only strings")

    happenings_freshness = config.get("happenings_freshness")
    if happenings_freshness is not None:
        allowed_freshness = {"pd", "pw", "pm", "py"}
        if not isinstance(happenings_freshness, str) or happenings_freshness not in allowed_freshness:
            errors.append(
                f"'happenings_freshness' must be one of {sorted(allowed_freshness)}, "
                f"got '{happenings_freshness}'"
            )

    max_happenings = config.get("max_happenings")
    if max_happenings is not None:
        if not isinstance(max_happenings, int):
            errors.append(f"'max_happenings' must be an integer, got {type(max_happenings).__name__}")
        elif max_happenings < 1 or max_happenings > 50:
            warnings.append(f"'max_happenings' value {max_happenings} is outside recommended range (1-50)")

    # --- Kindle/email ---
    kindle_email = config.get("kindle_email", "")
    if kindle_email and "kindle" not in kindle_email.lower() and kindle_email != "YOUR_NAME@kindle.com":
        warnings.append(
            f"'kindle_email' ({kindle_email}) does not contain 'kindle' -- "
            "verify this is correct"
        )

    # --- Paper scoring ---
    scoring = config.get("paper_scoring")
    if scoring is not None:
        if not isinstance(scoring, dict):
            errors.append("'paper_scoring' must be a dictionary")
        else:
            for key in ["has_code", "topic_match", "recency", "citation_count"]:
                val = scoring.get(key)
                if val is not None and not isinstance(val, (int, float)):
                    errors.append(f"paper_scoring.{key} must be a number")

    # --- PDF settings ---
    pdf = config.get("pdf")
    if pdf is not None:
        if not isinstance(pdf, dict):
            errors.append("'pdf' must be a dictionary")
        else:
            font_size = pdf.get("font_size")
            if font_size is not None and not isinstance(font_size, (int, float)):
                errors.append("pdf.font_size must be a number")
            line_spacing = pdf.get("line_spacing")
            if line_spacing is not None and not isinstance(line_spacing, (int, float)):
                errors.append("pdf.line_spacing must be a number")

    # --- Output format ---
    output_format = config.get("output_format")
    if output_format and output_format not in ("kindle", "a4", "letter"):
        errors.append(f"'output_format' must be 'kindle', 'a4', or 'letter', got '{output_format}'")

    # --- Opencode config ---
    opencode = config.get("opencode")
    if opencode is not None:
        if not isinstance(opencode, dict):
            errors.append("'opencode' must be a dictionary")
        elif opencode.get("enabled"):
            if not shutil.which("opencode"):
                warnings.append(
                    "'opencode.enabled' is true but 'opencode' binary not found "
                    "on PATH — install opencode or set 'opencode.enabled: false'"
                )

    # --- Codex report-writer config ---
    codex = config.get("codex")
    if codex is not None:
        if not isinstance(codex, dict):
            errors.append("'codex' must be a dictionary")
        elif codex.get("enabled"):
            binary = codex.get(
                "binary", codex.get("executable", codex.get("cli_binary"))
            )
            if not isinstance(binary, str) or not binary.strip():
                errors.append("'codex.binary' must be a non-empty string when enabled")

            model = codex.get("model")
            if not isinstance(model, str) or not model.strip():
                errors.append("'codex.model' must be a non-empty string when enabled")

            timeout = codex.get("timeout_seconds", codex.get("timeout", 300))
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or timeout <= 0
            ):
                errors.append("'codex.timeout_seconds' must be a positive number")

            max_calls = codex.get(
                "max_calls_per_run", codex.get("max_calls", 5)
            )
            if (
                isinstance(max_calls, bool)
                or not isinstance(max_calls, int)
                or max_calls <= 0
            ):
                errors.append("'codex.max_calls_per_run' must be a positive integer")

            reasoning = codex.get(
                "reasoning_effort", codex.get("reasoning", "high")
            )
            supported_reasoning = {"low", "medium", "high", "xhigh"}
            if reasoning not in supported_reasoning:
                errors.append(
                    "'codex.reasoning_effort' must be one of "
                    f"{sorted(supported_reasoning)}"
                )
    # --- LLM model chains ---
    # The chain is the roster now: a typo here is not a tier that falls back,
    # it is a rung that silently never runs.
    llm = config.get("llm")
    if llm is not None:
        if not isinstance(llm, dict):
            errors.append("'llm' must be a dictionary")
        else:
            chains = llm.get("chains")
            if chains is not None and not isinstance(chains, dict):
                errors.append("llm.chains must be a dictionary of tier -> model list")
            elif isinstance(chains, dict):
                from scripts.llm_chain import TIERS, resolve_backend

                enabled_backends = {
                    name
                    for name in ("openrouter", "opencode", "gemini")
                    if (config.get(name, {}) or {}).get("enabled")
                }
                for tier, models in chains.items():
                    if tier not in TIERS:
                        warnings.append(
                            f"llm.chains.{tier} is not a recognized tier "
                            "(expected: heavy, medium, light)"
                        )
                        continue
                    if not isinstance(models, list) or not models:
                        errors.append(f"llm.chains.{tier} must be a non-empty list")
                        continue
                    for model in models:
                        backend = resolve_backend(model) if isinstance(model, str) else None
                        if backend is None:
                            errors.append(
                                f"llm.chains.{tier}: {model!r} has no known routing "
                                "prefix (expected openrouter/, opencode/, "
                                "opencode-go/ or gemini/)"
                            )
                        elif backend not in enabled_backends:
                            warnings.append(
                                f"llm.chains.{tier}: {model} needs the '{backend}' "
                                "backend, which is not enabled — it will be skipped"
                            )
                missing = [t for t in TIERS if t not in chains]
                if missing:
                    warnings.append(
                        f"llm.chains has no entry for {', '.join(missing)}; "
                        "built-in defaults will be used"
                    )
                leads = [m[0] for t, m in chains.items()
                         if t in TIERS and isinstance(m, list) and m]
                if len(leads) == len(TIERS) and len(set(leads)) < len(TIERS):
                    warnings.append(
                        "llm.chains: two tiers lead with the same model, so they "
                        "are no longer distinct tiers"
                    )

    # --- Bedrock config ---
    bedrock = config.get("bedrock")
    if bedrock is not None:
        if not isinstance(bedrock, dict):
            errors.append("'bedrock' must be a dictionary")
        else:
            models = bedrock.get("models")
            if models is not None:
                if not isinstance(models, dict):
                    errors.append("bedrock.models must be a dictionary")
                else:
                    for tier in models:
                        if tier not in ("heavy", "medium", "light"):
                            warnings.append(
                                f"bedrock.models.{tier} is not a recognized tier "
                                "(expected: heavy, medium, light)"
                            )

    # --- Log results ---
    for w in warnings:
        logger.warning(f"Config warning: {w}")
    for e in errors:
        logger.error(f"Config error: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors + warnings


def check_environment(config: Dict[str, Any], dry_run: bool = False) -> List[str]:
    """
    Check required environment variables based on config.

    Args:
        config: Configuration dictionary.
        dry_run: If True, skip email credential checks.

    Returns:
        List of warning messages for missing variables.
    """
    warnings = []

    # Stocks require Finnhub key
    if config.get("stocks"):
        if not os.environ.get("FINNHUB_API_KEY"):
            warnings.append(
                "FINNHUB_API_KEY not set -- stock data will be skipped"
            )

    # News requires Brave key
    if config.get("news_queries") or config.get("interest_graph"):
        if not os.environ.get("BRAVE_API_KEY"):
            warnings.append(
                "BRAVE_API_KEY not set -- news aggregation will be skipped"
            )

    # Email requires Gmail credentials (unless dry run)
    if not dry_run:
        if not os.environ.get("GMAIL_USER"):
            warnings.append("GMAIL_USER not set -- Kindle delivery will be skipped")
        if not os.environ.get("GMAIL_APP_PASSWORD"):
            warnings.append(
                "GMAIL_APP_PASSWORD not set -- Kindle delivery will be skipped"
            )

    for w in warnings:
        logger.warning(w)

    return warnings
