# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for config_validator module."""

from pathlib import Path

import pytest
import yaml
from scripts.config_validator import validate_config, check_environment


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

    def test_enabled_direct_nvidia_chain_has_no_backend_warning(self):
        config = {
            "arxiv_topics": ["test"],
            "nvidia": {"enabled": True},
            "llm": {
                "chains": {
                    "heavy": ["nvidia-direct/nvidia/nemotron-3-ultra-550b-a55b"],
                    "medium": ["nvidia-direct/nvidia/nemotron-3-super-120b-a12b"],
                    "light": [
                        "nvidia-direct/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
                    ],
                }
            },
        }

        is_valid, messages = validate_config(config)

        assert is_valid is True
        assert not any("not enabled" in message for message in messages)
        assert not any("no known routing prefix" in message for message in messages)

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


class TestCodexConfig:
    def test_implicit_enabled_codex_rejects_invalid_timeout(self):
        """Catches validation allowing a mapping the runtime enables by default."""
        is_valid, messages = validate_config(
            {"arxiv_topics": ["test"], "codex": {"timeout_seconds": -1}}
        )

        assert is_valid is False
        assert any("timeout_seconds" in message for message in messages)

    def test_explicitly_disabled_codex_mapping_remains_exempt(self):
        """Catches disabled writer settings being treated as runnable config."""
        is_valid, messages = validate_config(
            {
                "arxiv_topics": ["test"],
                "codex": {"enabled": False, "timeout_seconds": -1},
            }
        )

        assert is_valid is True
        assert not any("codex" in message.lower() for message in messages)

    def test_validator_uses_runtime_executable_alias_precedence(self):
        """Catches validating binary while runtime would select executable."""
        is_valid, messages = validate_config(
            {
                "arxiv_topics": ["test"],
                "codex": {
                    "executable": "",
                    "binary": "/opt/codex",
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "high",
                    "timeout_seconds": 300,
                    "max_calls_per_run": 5,
                },
            }
        )

        assert is_valid is False
        assert any("codex.binary" in message for message in messages)

    @pytest.mark.parametrize("timeout_literal", [".inf", ".nan"])
    def test_enabled_codex_rejects_nonfinite_yaml_timeout(self, timeout_literal):
        """Catches an unbounded or invalid subprocess timeout reaching cron."""
        config = yaml.safe_load(
            f"""
            arxiv_topics: [test]
            codex:
              enabled: true
              binary: /opt/codex
              model: gpt-5.6-sol
              reasoning_effort: high
              timeout_seconds: {timeout_literal}
              max_calls_per_run: 5
            """
        )

        is_valid, messages = validate_config(config)

        assert is_valid is False
        assert any("timeout_seconds" in message and "finite" in message for message in messages)

    @pytest.mark.parametrize(
        ("codex", "missing"),
        [
            (
                {
                    "enabled": True,
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "high",
                    "timeout_seconds": 300,
                    "max_calls_per_run": 5,
                },
                "binary",
            ),
            (
                {
                    "enabled": True,
                    "binary": "/opt/codex",
                    "reasoning_effort": "high",
                    "timeout_seconds": 300,
                    "max_calls_per_run": 5,
                },
                "model",
            ),
            (
                {
                    "enabled": True,
                    "binary": "/opt/codex",
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "high",
                    "timeout_seconds": 0,
                    "max_calls_per_run": 5,
                },
                "timeout_seconds",
            ),
            (
                {
                    "enabled": True,
                    "binary": "/opt/codex",
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "high",
                    "timeout_seconds": 300,
                    "max_calls_per_run": 0,
                },
                "max_calls_per_run",
            ),
            (
                {
                    "enabled": True,
                    "binary": "/opt/codex",
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "unsupported",
                    "timeout_seconds": 300,
                    "max_calls_per_run": 5,
                },
                "reasoning_effort",
            ),
        ],
    )
    def test_enabled_codex_rejects_invalid_required_setting(self, codex, missing):
        """Catches a malformed enabled writer config reaching cron unchecked."""
        is_valid, messages = validate_config({"arxiv_topics": ["test"], "codex": codex})

        assert is_valid is False
        assert any(missing in message for message in messages)

    def test_enabled_codex_with_supported_settings_is_valid(self):
        """Catches rejecting the bounded writer configuration used by the run."""
        config = {
            "arxiv_topics": ["test"],
            "codex": {
                "enabled": True,
                "binary": "/home/eric/.local/bin/codex",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "high",
                "timeout_seconds": 300,
                "max_calls_per_run": 5,
            },
        }

        is_valid, messages = validate_config(config)

        assert is_valid is True
        assert not any("codex" in message.lower() for message in messages)

    def test_checked_in_configs_enable_the_same_bounded_codex_writer(self):
        """Catches the local run silently drifting from the main writer setup."""
        root = Path(__file__).resolve().parent.parent
        with (root / "config.yaml").open() as stream:
            main = yaml.safe_load(stream)["codex"]
        with (root / "config_local.yaml").open() as stream:
            local = yaml.safe_load(stream)["codex"]

        expected = {
            "enabled": True,
            "binary": "/home/eric/.local/bin/codex",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "timeout_seconds": 300,
            "max_calls_per_run": 5,
        }
        assert {key: main[key] for key in expected} == expected
        assert {key: local[key] for key in expected} == expected
        assert main["call_log_path"] == "logs/codex-calls.jsonl"
        assert local["call_log_path"] == "logs/local-codex-calls.jsonl"


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
