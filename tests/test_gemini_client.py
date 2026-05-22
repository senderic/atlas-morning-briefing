#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for GeminiCLIClient."""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from scripts.gemini_client import GeminiCLIClient


@pytest.fixture
def mock_config():
    return {
        "enabled": True,
        "max_calls_per_run": 10,
        "models": {
            "heavy": "test-heavy",
            "medium": "test-medium",
            "light": "test-light",
        }
    }


class TestGeminiCLIClient:
    def test_init(self, mock_config):
        client = GeminiCLIClient(mock_config)
        assert client.enabled is True
        assert client.max_calls == 10
        assert client.models["heavy"] == "test-heavy"
        assert client.models["medium"] == "test-medium"
        assert client.models["light"] == "test-light"

    def test_default_models(self):
        client = GeminiCLIClient({})
        assert client.models["heavy"] == "pro"
        assert client.models["medium"] == "flash"
        assert client.models["light"] == "flash-lite"

    @patch("subprocess.run")
    def test_available_true(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        client = GeminiCLIClient({"enabled": True})
        assert client.available is True
        mock_run.assert_called_once_with(["which", "gemini"], capture_output=True, check=True)

    @patch("subprocess.run")
    def test_available_false_not_in_path(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["which", "gemini"])
        client = GeminiCLIClient({"enabled": True})
        assert client.available is False

    def test_available_false_disabled(self):
        client = GeminiCLIClient({"enabled": False})
        assert client.available is False

    @patch("subprocess.run")
    def test_invoke_success(self, mock_run, mock_config):
        mock_run.return_value = MagicMock(returncode=0, stdout="Test response\n")
        client = GeminiCLIClient(mock_config)
        client._available = True  # bypass check
        
        response = client.invoke("Test prompt", tier="medium")
        
        assert response == "Test response"
        assert client._call_count == 1
        
        # Verify command construction
        cmd = mock_run.call_args[0][0]
        assert "gemini" in cmd
        assert "test-medium" in cmd
        assert "Test prompt" in cmd
        assert "--raw-output" in cmd
        assert "--accept-raw-output-risk" in cmd

    @patch("subprocess.run")
    def test_invoke_with_system_prompt(self, mock_run, mock_config):
        mock_run.return_value = MagicMock(returncode=0, stdout="Response")
        client = GeminiCLIClient(mock_config)
        client._available = True
        
        client.invoke("User prompt", tier="heavy", system_prompt="System instructions")
        
        cmd = mock_run.call_args[0][0]
        # Check that both prompts are in the final prompt string
        prompt_arg_idx = cmd.index("--prompt") + 1
        full_prompt = cmd[prompt_arg_idx]
        assert "System instructions" in full_prompt
        assert "User prompt" in full_prompt

    @patch("subprocess.run")
    def test_invoke_exhausted_budget(self, mock_run, mock_config):
        client = GeminiCLIClient(mock_config)
        client._available = True
        client._call_count = 10  # max_calls is 10
        
        response = client.invoke("Prompt")
        assert response is None
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_invoke_failure_tracking(self, mock_run, mock_config):
        # Mock a non-retryable failure to keep test fast
        mock_run.side_effect = subprocess.CalledProcessError(1, ["gemini"], stderr="Fatal error")
        client = GeminiCLIClient(mock_config)
        client._available = True
        
        # Disable fallback for this test to focus on single tier failure
        response = client.invoke("Prompt", tier="medium", allow_fallback=False)
        
        assert response is None
        # Should have 1 failed attempt (no retry because "Fatal error" isn't a quota keyword)
        assert client.usage_stats["medium"]["failed_attempts"] == 1
        assert client.usage_stats["medium"]["calls"] == 0

    @patch("subprocess.run")
    def test_invoke_retry_and_rotation_tracking(self, mock_run, mock_config):
        # Mock 2 quota failures then 1 success
        # "429" triggers retry/rotation
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, ["gemini"], stderr="Error: 429 quota exceeded"),
            subprocess.CalledProcessError(1, ["gemini"], stderr="Error: 429 quota exceeded"),
            MagicMock(returncode=0, stdout='{"response": "Success after retries"}')
        ]
        
        # Add multiple keys to allow rotation
        with patch.dict(os.environ, {"GEMINI_API_KEY": "key1,key2,key3"}):
            client = GeminiCLIClient(mock_config)
            client._available = True
            # Shorten delays for test
            client.key_swap_delay = 0.01
            
            # Using patch to avoid long exponential waits in test
            with patch("scripts.gemini_client.wait_random_exponential", return_value=lambda x: 0.01):
                response = client.invoke("Prompt", tier="heavy")
        
        assert response == "Success after retries"
        # 2 failures + 1 success
        assert client.usage_stats["heavy"]["failed_attempts"] == 2
        assert client.usage_stats["heavy"]["calls"] == 1

    def test_get_usage_summary_with_failures(self, mock_config):
        client = GeminiCLIClient(mock_config)
        client.usage_stats["heavy"]["failed_attempts"] = 5
        client.usage_stats["medium"]["calls"] = 2
        client.usage_stats["medium"]["failed_attempts"] = 1
        client.usage_stats["medium"]["in_tokens"] = 1000
        client.usage_stats["medium"]["out_tokens"] = 500
        
        summary = client.get_usage_summary()
        
        assert "## Gemini Usage Summary" in summary
        assert "| Tier | Success | Failures |" in summary
        assert "| Heavy | 0 | 5 |" in summary
        assert "| Medium | 2 | 1 |" in summary
        assert "Failed calls are not charged" in summary

    def test_get_usage_summary_with_key_rotation(self, mock_config):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "key1,key2"}):
            client = GeminiCLIClient(mock_config)
            # Simulate some usage in stats so the summary isn't empty
            client.usage_stats["medium"]["calls"] = 1
            # Simulate key usage
            client.key_usage_stats[0]["success"] = 10
            client.key_usage_stats[0]["failure"] = 2
            client.key_usage_stats[1]["success"] = 5
            client.key_usage_stats[1]["failure"] = 1
            
            summary = client.get_usage_summary()
            
            assert "### API Key Rotation Summary" in summary
            assert "| Key Index | Preview | Success | Quota/Failures |" in summary
            assert "| 0 | `key1...key1` | 10 | 2 |" in summary
            assert "| 1 | `key2...key2` | 5 | 1 |" in summary
