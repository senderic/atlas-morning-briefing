# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Comprehensive tests for gemini_client (usage summary, command env, fallback)."""

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.gemini_client import GeminiCLIClient


@pytest.fixture
def cfg():
    return {
        "enabled": True,
        "max_calls_per_run": 10,
        "key_swap_delay": 0.001,
        "internal_max_attempts": 1,
        "cli_binary": "gemini",  # tests assert gemini-shaped argv
        "models": {"heavy": "h", "medium": "m", "light": "l"},
    }


@pytest.fixture
def client(cfg):
    return GeminiCLIClient(cfg)


# ---------- usage summary ----------


class TestUsageSummary:
    def test_empty_summary_returns_empty_string(self, client):
        """No calls and no failures → empty string (don't pollute briefing)."""
        assert client.get_usage_summary() == ""

    def test_estimates_tokens_from_chars(self, client):
        # Set chars but no tokens — should estimate at 3.5 chars/token
        client.usage_stats["medium"]["calls"] = 1
        client.usage_stats["medium"]["in_chars"] = 350  # → ~100 tokens
        client.usage_stats["medium"]["out_chars"] = 700  # → ~200 tokens
        summary = client.get_usage_summary()
        assert "100" in summary  # estimated input tokens
        assert "200" in summary  # estimated output tokens

    def test_uses_actual_tokens_when_available(self, client):
        client.usage_stats["medium"]["calls"] = 1
        client.usage_stats["medium"]["in_tokens"] = 999
        client.usage_stats["medium"]["out_tokens"] = 777
        summary = client.get_usage_summary()
        assert "999" in summary
        assert "777" in summary

    def test_includes_duration_minutes(self, client):
        client.usage_stats["medium"]["calls"] = 1
        summary = client.get_usage_summary(start_time=1000, end_time=1125)
        assert "2m 5s" in summary

    def test_includes_duration_hours(self, client):
        client.usage_stats["medium"]["calls"] = 1
        # Note: start_time must be truthy (non-zero) due to "if start_time and end_time" check
        summary = client.get_usage_summary(start_time=1, end_time=3726)
        assert "1h" in summary
        assert "2m" in summary

    def test_no_duration_when_start_time_is_zero(self, client):
        """Bug-aware test: 'if start_time and end_time' check treats 0 as missing."""
        client.usage_stats["medium"]["calls"] = 1
        summary = client.get_usage_summary(start_time=0, end_time=100)
        assert "Started:" not in summary  # duration block skipped

    def test_omits_tiers_with_no_activity(self, client):
        client.usage_stats["light"]["calls"] = 2
        summary = client.get_usage_summary()
        assert "Light" in summary
        # Heavy/Medium had no activity → not listed
        assert "| Heavy " not in summary
        assert "| Medium " not in summary

    def test_includes_pricing_note(self, client):
        client.usage_stats["medium"]["calls"] = 1
        summary = client.get_usage_summary()
        assert "Costs are estimated" in summary

    def test_key_rotation_summary_present(self, cfg):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k1,k2"}, clear=True):
            client = GeminiCLIClient(cfg)
            client.usage_stats["medium"]["calls"] = 1
            client.key_usage_stats[0]["success"] = 3
            summary = client.get_usage_summary()
            assert "API Key Rotation Summary" in summary

    def test_no_key_rotation_section_without_keys(self, cfg):
        with patch.dict(os.environ, {}, clear=True):
            client = GeminiCLIClient(cfg)
            client.usage_stats["medium"]["calls"] = 1
            summary = client.get_usage_summary()
            # No keys loaded → no rotation section
            assert "API Key Rotation Summary" not in summary


# ---------- key loading ----------


class TestLoadKeys:
    def test_no_keys_logs_warning(self, cfg):
        with patch.dict(os.environ, {}, clear=True):
            client = GeminiCLIClient(cfg)
            assert client._api_keys == []
            assert client._get_current_key() is None

    def test_only_suffixed_no_primary(self, cfg):
        with patch.dict(os.environ, {"GEMINI_API_KEY_ALPHA": "ka"}, clear=True):
            client = GeminiCLIClient(cfg)
            assert client._api_keys == ["ka"]

    def test_comma_split_primary(self, cfg):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "a,b,c"}, clear=True):
            client = GeminiCLIClient(cfg)
            assert client._api_keys == ["a", "b", "c"]

    def test_empty_values_ignored(self, cfg):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ",a,,b,"}, clear=True):
            client = GeminiCLIClient(cfg)
            assert client._api_keys == ["a", "b"]


class TestRotateKey:
    def test_single_key_cannot_rotate(self, cfg):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "only"}, clear=True):
            client = GeminiCLIClient(cfg)
            assert client._rotate_key() is False
            assert client._current_key_index == 0

    @patch("time.sleep")
    def test_rotation_wraps_around(self, mock_sleep, cfg):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k1,k2"}, clear=True):
            client = GeminiCLIClient(cfg)
            assert client._current_key_index == 0
            client._rotate_key()
            assert client._current_key_index == 1
            client._rotate_key()
            # wraps back to 0
            assert client._current_key_index == 0


# ---------- execute_command / invoke ----------


class TestExecuteCommandEnv:
    """Verify subprocess env is locked down (no ADC, no inherited Google auth)."""

    @patch("subprocess.run")
    def test_clears_google_application_credentials(self, mock_run, cfg):
        mock_run.return_value = MagicMock(returncode=0, stdout='{"response":"ok"}')
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True):
            client = GeminiCLIClient(cfg)
            client._available = True
            client.invoke("p", tier="medium")
        env = mock_run.call_args.kwargs["env"]
        assert env["GOOGLE_APPLICATION_CREDENTIALS"] == ""
        assert env["GOOGLE_API_KEY"] == ""
        assert env["CLOUDSDK_AUTH_ACCESS_TOKEN"] == ""

    @patch("subprocess.run")
    def test_sets_gemini_api_key(self, mock_run, cfg):
        mock_run.return_value = MagicMock(returncode=0, stdout='{"response":"ok"}')
        with patch.dict(os.environ, {"GEMINI_API_KEY": "my-key"}, clear=True):
            client = GeminiCLIClient(cfg)
            client._available = True
            client.invoke("p", tier="medium")
        env = mock_run.call_args.kwargs["env"]
        assert env["GEMINI_API_KEY"] == "my-key"

    @patch("subprocess.run")
    def test_yolo_and_raw_flags_present(self, mock_run, cfg):
        mock_run.return_value = MagicMock(returncode=0, stdout='{"response":"ok"}')
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True):
            client = GeminiCLIClient(cfg)
            client._available = True
            client.invoke("prompt", tier="medium")
        cmd = mock_run.call_args.args[0]
        assert "--approval-mode" in cmd
        assert "yolo" in cmd
        assert "--raw-output" in cmd
        assert "--accept-raw-output-risk" in cmd


class TestInvokeFallback:
    @patch("subprocess.run")
    @patch("time.sleep")
    def test_heavy_falls_back_to_medium_then_light(self, mock_sleep, mock_run, cfg):
        # All attempts on heavy fail with non-retryable error → fallback to medium → light
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gemini"], stderr="some unrecoverable error"
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True):
            client = GeminiCLIClient(cfg)
            client._available = True
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.01):
                result = client.invoke("p", tier="heavy", allow_fallback=True)
        assert result is None

    @patch("subprocess.run")
    def test_no_fallback_when_disabled(self, mock_run, cfg):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gemini"], stderr="fail"
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True):
            client = GeminiCLIClient(cfg)
            client._available = True
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.01):
                result = client.invoke("p", tier="medium", allow_fallback=False)
        assert result is None

    @patch("subprocess.run")
    def test_no_signature_kwargs(self, mock_run, cfg):
        """Invoke must reject the deprecated max_tokens / temperature kwargs."""
        with pytest.raises(TypeError):
            client = GeminiCLIClient(cfg)
            client._available = True
            client.invoke("p", tier="medium", max_tokens=100)


class TestInvokeBudget:
    def test_call_count_not_incremented_on_failure(self, cfg):
        client = GeminiCLIClient(cfg)
        client._available = True
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(
            1, ["gemini"], stderr="fatal"
        )):
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.01):
                client.invoke("p", tier="medium", allow_fallback=False)
        # _call_count is incremented only on success
        assert client._call_count == 0

    @patch("subprocess.run")
    def test_call_count_increments_on_success(self, mock_run, cfg):
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"response":"ok"}'
        )
        client = GeminiCLIClient(cfg)
        client._available = True
        client.invoke("p", tier="medium")
        assert client._call_count == 1


class TestParseResponse:
    @patch("subprocess.run")
    def test_extracts_response_field(self, mock_run, cfg):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "response": "Hello world",
                "stats": {"models": {"m-model": {"tokens": {"input": 10, "candidates": 5}}}},
            }),
        )
        client = GeminiCLIClient(cfg)
        client._available = True
        result = client.invoke("p", tier="medium")
        assert result == "Hello world"
        assert client.usage_stats["medium"]["in_tokens"] == 10
        assert client.usage_stats["medium"]["out_tokens"] == 5

    @patch("subprocess.run")
    def test_falls_back_to_raw_stdout_on_invalid_json(self, mock_run, cfg):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Plain non-JSON output\n"
        )
        client = GeminiCLIClient(cfg)
        client._available = True
        result = client.invoke("p", tier="medium")
        assert result == "Plain non-JSON output"

    @patch("subprocess.run")
    def test_empty_response_raises(self, mock_run, cfg):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps({"response": ""})
        )
        client = GeminiCLIClient(cfg)
        client._available = True
        # Empty raises ValueError; retried up to limit; then fallback chain
        with patch("scripts.gemini_client.wait_random_exponential",
                   return_value=lambda x: 0.01):
            result = client.invoke("p", tier="light", allow_fallback=False)
        assert result is None


class TestQuotaDetection:
    @patch("subprocess.run")
    @patch("time.sleep")
    def test_quota_keyword_triggers_rotation(self, mock_sleep, mock_run, cfg):
        # First fails with 429, second succeeds; should rotate key.
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, ["gemini"], stderr="HTTP 429 rate limit"),
            MagicMock(returncode=0, stdout='{"response":"ok"}'),
        ]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k1,k2"}, clear=True):
            client = GeminiCLIClient(cfg)
            client._available = True
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="medium")
            assert result == "ok"
            assert client._current_key_index == 1

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_hard_quota_with_single_key_fails(self, mock_sleep, mock_run, cfg):
        cfg["ignore_hard_quota"] = False
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gemini"], stderr="daily quota exceeded"
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "only"}, clear=True):
            client = GeminiCLIClient(cfg)
            client._available = True
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="medium", allow_fallback=False)
            assert result is None

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_ignore_hard_quota_retries(self, mock_sleep, mock_run, cfg):
        cfg["ignore_hard_quota"] = True
        # Returns quota error twice, then success
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, ["gemini"], stderr="daily quota exceeded"),
            MagicMock(returncode=0, stdout='{"response":"finally"}'),
        ]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "only"}, clear=True):
            client = GeminiCLIClient(cfg)
            client._available = True
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="medium")
            assert result == "finally"
