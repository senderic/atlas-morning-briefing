#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Configuration validator.

Validates config.yaml values at startup to catch errors early.
"""

import logging
import os
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ${VAR} or ${VAR:-default}. Inner default is captured non-greedily, stopping
# at the closing brace. We deliberately don't support nested expansion since
# config values shouldn't need it.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} in a config value tree.

    Strings get substituted; lists and dicts are walked element-wise; other
    types pass through. ${VAR} resolves to os.environ[VAR] when present,
    otherwise the literal placeholder is preserved (so config validation
    can still flag missing required vars). ${VAR:-default} resolves to
    os.environ[VAR] when truthy, otherwise to `default`.

    This is run BEFORE validate_config so downstream code never sees raw
    bash-style placeholders that would render as literal strings in emails,
    file paths, or LLM prompts.
    """
    if isinstance(value, str):
        def _sub(match):
            var = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var)
            if env_value:
                return env_value
            if default is not None:
                return default
            return match.group(0)  # leave unresolved placeholder in place
        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    return value


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

    # --- Gemini config ---
    gemini = config.get("gemini")
    if gemini is not None:
        if not isinstance(gemini, dict):
            errors.append("'gemini' must be a dictionary")
        else:
            models = gemini.get("models")
            if models is not None:
                if not isinstance(models, dict):
                    errors.append("gemini.models must be a dictionary")
                else:
                    for tier in models:
                        if tier not in ("heavy", "medium", "light"):
                            warnings.append(
                                f"gemini.models.{tier} is not a recognized tier "
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
    if config.get("news_queries"):
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

    # Gemini requires API key and CLI
    gemini_config = config.get("gemini", {})
    if gemini_config.get("enabled", False):
        if not os.environ.get("GEMINI_API_KEY") and not any(k.startswith("GEMINI_API_KEY_") for k in os.environ):
            warnings.append("Gemini is enabled but no GEMINI_API_KEY found in environment")
        
        # Check for 'gemini' command
        import subprocess
        try:
            subprocess.run(["which", "gemini"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            warnings.append("Gemini is enabled but 'gemini' CLI command not found in PATH")

    for w in warnings:
        logger.warning(w)

    return warnings
