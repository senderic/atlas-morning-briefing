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
        assert client.models["heavy"] == "gemini-3-pro-preview"
        assert client.models["medium"] == "gemini-3-flash-preview"
        assert client.models["light"] == "gemini-2.5-flash-lite"

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
    def test_invoke_failure(self, mock_run, mock_config):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["gemini"], stderr="Error message")
        client = GeminiCLIClient(mock_config)
        client._available = True
        
        response = client.invoke("Prompt")
        assert response is None
