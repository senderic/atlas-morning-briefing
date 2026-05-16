# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for config_validator module."""

import os
from unittest.mock import patch

import pytest
from scripts.config_validator import (
    validate_config,
    check_environment,
    expand_env_vars,
)


class TestExpandEnvVars:
    """Verify bash-style ${VAR} / ${VAR:-default} expansion in config trees."""

    def test_expands_simple_var(self, monkeypatch):
        monkeypatch.setenv("FOO", "hello")
        assert expand_env_vars("${FOO}") == "hello"

    def test_default_used_when_var_unset(self, monkeypatch):
        monkeypatch.delenv("MISSING", raising=False)
        assert expand_env_vars("${MISSING:-fallback}") == "fallback"

    def test_default_used_when_var_empty(self, monkeypatch):
        monkeypatch.setenv("EMPTY", "")
        # Empty string is falsy in Bash ${VAR:-default} semantics.
        assert expand_env_vars("${EMPTY:-fallback}") == "fallback"

    def test_set_var_wins_over_default(self, monkeypatch):
        monkeypatch.setenv("X", "real")
        assert expand_env_vars("${X:-fallback}") == "real"

    def test_unresolved_placeholder_kept_when_no_default(self, monkeypatch):
        monkeypatch.delenv("NOPE", raising=False)
        # No default and var unset: keep literal so validate_config can warn.
        assert expand_env_vars("${NOPE}") == "${NOPE}"

    def test_multiple_placeholders_in_one_string(self, monkeypatch):
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        assert expand_env_vars("a=${A} b=${B}") == "a=1 b=2"

    def test_default_with_special_chars(self, monkeypatch):
        monkeypatch.delenv("X", raising=False)
        assert expand_env_vars("${X:-user@example.com}") == "user@example.com"

    def test_walks_lists_recursively(self, monkeypatch):
        monkeypatch.setenv("R", "you@example.com")
        result = expand_env_vars(["${R}", "static@example.com"])
        assert result == ["you@example.com", "static@example.com"]

    def test_walks_dicts_recursively(self, monkeypatch):
        monkeypatch.setenv("K", "secret")
        result = expand_env_vars({"key": "${K}", "nested": {"inner": "${K:-x}"}})
        assert result == {"key": "secret", "nested": {"inner": "secret"}}

    def test_non_string_passthrough(self):
        # Numbers, bools, None must pass through untouched.
        assert expand_env_vars(42) == 42
        assert expand_env_vars(3.14) == 3.14
        assert expand_env_vars(True) is True
        assert expand_env_vars(None) is None

    def test_repro_report_case_kindle_email(self, monkeypatch):
        """The exact pattern in config.yaml that failed in production
        on 2026-05-16: literal '${KINDLE_EMAIL:-YOUR_NAME@kindle.com}'
        was being passed straight through to SMTP, which parsed it as
        the invalid recipient '-YOUR_NAME@kindle.com}'."""
        monkeypatch.delenv("KINDLE_EMAIL", raising=False)
        result = expand_env_vars("${KINDLE_EMAIL:-YOUR_NAME@kindle.com}")
        assert result == "YOUR_NAME@kindle.com"
        monkeypatch.setenv("KINDLE_EMAIL", "real@kindle.com")
        result = expand_env_vars("${KINDLE_EMAIL:-YOUR_NAME@kindle.com}")
        assert result == "real@kindle.com"

    def test_full_config_tree_like_production(self, monkeypatch):
        monkeypatch.setenv("KINDLE_EMAIL", "me@kindle.com")
        monkeypatch.delenv("SENDER_EMAIL", raising=False)
        monkeypatch.delenv("RECIPIENT_EMAIL", raising=False)
        config = {
            "kindle_email": "${KINDLE_EMAIL:-YOUR_NAME@kindle.com}",
            "sender_email": "${SENDER_EMAIL:-sender@gmail.com}",
            "email_recipients": ["${RECIPIENT_EMAIL:-you@example.com}"],
            "stocks": ["NVDA", "MSFT"],
            "max_workers": 1,
        }
        out = expand_env_vars(config)
        assert out["kindle_email"] == "me@kindle.com"
        assert out["sender_email"] == "sender@gmail.com"
        assert out["email_recipients"] == ["you@example.com"]
        assert out["stocks"] == ["NVDA", "MSFT"]
        assert out["max_workers"] == 1


class TestValidateConfig:
    def test_valid_config(self):
        config = {
            "arxiv_topics": ["Agent Evaluation"],
            "blog_feeds": [{"name": "Test", "url": "http://example.com/rss"}],
            "stocks": ["AMZN"],
            "news_queries": ["AI"],
            "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
            "output_format": "kindle",
            "arxiv_days_back": 7,
            "max_papers": 20,
            "max_blogs": 10,
            "max_news": 15,
            "num_paper_picks": 3,
        }
        is_valid, messages = validate_config(config)
        assert is_valid is True

    def test_invalid_arxiv_topics_type(self):
        config = {"arxiv_topics": "not a list"}
        is_valid, messages = validate_config(config)
        assert is_valid is False
        assert any("arxiv_topics" in m for m in messages)

    def test_invalid_int_field(self):
        config = {"arxiv_topics": ["test"], "arxiv_days_back": "seven"}
        is_valid, messages = validate_config(config)
        assert is_valid is False
        assert any("arxiv_days_back" in m for m in messages)

    def test_invalid_blog_feed_missing_url(self):
        config = {
            "arxiv_topics": ["test"],
            "blog_feeds": [{"name": "Test"}],
        }
        is_valid, messages = validate_config(config)
        assert is_valid is False
        assert any("blog_feeds" in m for m in messages)

    def test_invalid_blog_feed_not_dict(self):
        config = {
            "arxiv_topics": ["test"],
            "blog_feeds": ["not a dict"],
        }
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_invalid_output_format(self):
        config = {"arxiv_topics": ["test"], "output_format": "tabloid"}
        is_valid, messages = validate_config(config)
        assert is_valid is False
        assert any("output_format" in m for m in messages)

    def test_invalid_paper_scoring_type(self):
        config = {"arxiv_topics": ["test"], "paper_scoring": "bad"}
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_invalid_paper_scoring_value(self):
        config = {"arxiv_topics": ["test"], "paper_scoring": {"has_code": "five"}}
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_invalid_pdf_config(self):
        config = {"arxiv_topics": ["test"], "pdf": "bad"}
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_invalid_bedrock_config(self):
        config = {"arxiv_topics": ["test"], "bedrock": "bad"}
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_warning_for_many_stocks(self):
        config = {
            "arxiv_topics": ["test"],
            "stocks": [f"TICK{i}" for i in range(35)],
        }
        is_valid, messages = validate_config(config)
        assert is_valid is True  # Warning, not error
        assert any("tickers" in m for m in messages)

    def test_warning_for_empty_topics(self):
        config = {"arxiv_topics": []}
        is_valid, messages = validate_config(config)
        assert is_valid is True  # Warning, not error
        assert any("empty" in m for m in messages)

    def test_empty_config(self):
        config = {}
        is_valid, messages = validate_config(config)
        assert is_valid is False

    def test_valid_bedrock_config(self):
        config = {
            "arxiv_topics": ["test"],
            "bedrock": {
                "enabled": True,
                "region": "us-east-1",
                "models": {"heavy": "some-model", "medium": "some-model", "light": "some-model"},
            },
        }
        is_valid, messages = validate_config(config)
        assert is_valid is True


class TestCheckEnvironment:
    def test_warns_missing_finnhub(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        config = {"stocks": ["AMZN"]}
        warnings = check_environment(config)
        assert any("FINNHUB_API_KEY" in w for w in warnings)

    def test_warns_missing_brave(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        config = {"news_queries": ["AI"]}
        warnings = check_environment(config)
        assert any("BRAVE_API_KEY" in w for w in warnings)

    def test_warns_missing_gmail_not_dry_run(self, monkeypatch):
        monkeypatch.delenv("GMAIL_USER", raising=False)
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        config = {}
        warnings = check_environment(config, dry_run=False)
        assert any("GMAIL_USER" in w for w in warnings)

    def test_no_gmail_warning_on_dry_run(self, monkeypatch):
        monkeypatch.delenv("GMAIL_USER", raising=False)
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        config = {}
        warnings = check_environment(config, dry_run=True)
        assert not any("GMAIL_USER" in w for w in warnings)

    def test_no_warnings_when_no_features(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        config = {}  # No stocks or news configured
        warnings = check_environment(config, dry_run=True)
        assert len(warnings) == 0
