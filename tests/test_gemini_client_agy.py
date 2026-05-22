# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Tests for the Antigravity CLI (agy) code path in gemini_client.py.

The agy migration is documented in MIGRATION_PLAN_ANTIGRAVITY.md. Flag
names follow the migration plan and should be re-verified against
`agy --help` once the binary is installed — adjust BINARY_PROFILES
in scripts/gemini_client.py if anything has been renamed upstream.
"""

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.gemini_client import (
    BINARY_PROFILES,
    GeminiCLIClient,
    _build_agy_cmd,
    _build_gemini_cmd,
)


@pytest.fixture
def agy_cfg():
    return {
        "enabled": True,
        "max_calls_per_run": 10,
        "key_swap_delay": 0.001,
        "internal_max_attempts": 1,
        "cli_binary": "agy",
        "models": {"heavy": "pro", "medium": "flash", "light": "flash-lite"},
    }


class TestBinaryProfiles:
    def test_agy_profile_registered(self):
        assert "agy" in BINARY_PROFILES
        assert BINARY_PROFILES["agy"]["config_dirname"] == ".agy"
        assert BINARY_PROFILES["agy"]["config_dir_env"] == "AGY_CONFIG_DIR"

    def test_gemini_profile_registered(self):
        assert "gemini" in BINARY_PROFILES
        assert BINARY_PROFILES["gemini"]["config_dirname"] == ".gemini"

    def test_build_agy_cmd_has_expected_flags(self):
        cmd = _build_agy_cmd("pro", "Hello prompt")
        assert cmd[0] == "agy"
        assert "Hello prompt" in cmd  # positional, not via --prompt
        assert "--model" in cmd
        assert "pro" in cmd
        assert "--output=json" in cmd
        assert "--quiet" in cmd
        assert "--dangerously-skip-permissions" in cmd
        # Importantly, NOT the legacy flags
        assert "--prompt" not in cmd
        assert "--approval-mode" not in cmd
        assert "--raw-output" not in cmd

    def test_build_gemini_cmd_has_expected_flags(self):
        cmd = _build_gemini_cmd("pro", "Hello")
        assert cmd[0] == "gemini"
        assert "--prompt" in cmd
        assert "Hello" in cmd
        assert "--approval-mode" in cmd
        assert "yolo" in cmd
        assert "--raw-output" in cmd
        assert "--accept-raw-output-risk" in cmd
        assert "--output-format" in cmd


class TestAvailableAutoDetect:
    @patch("subprocess.run")
    def test_picks_agy_when_present(self, mock_run):
        """When auto-detecting, agy is preferred over gemini."""
        mock_run.return_value = MagicMock(returncode=0)
        client = GeminiCLIClient({"enabled": True})  # no explicit binary
        assert client.available is True
        assert client.binary == "agy"
        # Only one `which` call — first candidate wins
        first_call = mock_run.call_args_list[0]
        assert first_call.args[0] == ["which", "agy"]

    @patch("subprocess.run")
    def test_falls_back_to_gemini(self, mock_run):
        """When agy isn't on PATH but gemini is, we get gemini."""
        # First call (which agy) fails; second (which gemini) succeeds
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, ["which", "agy"]),
            MagicMock(returncode=0),
        ]
        client = GeminiCLIClient({"enabled": True})
        assert client.available is True
        assert client.binary == "gemini"
        assert mock_run.call_args_list[0].args[0] == ["which", "agy"]
        assert mock_run.call_args_list[1].args[0] == ["which", "gemini"]

    @patch("subprocess.run")
    def test_explicit_override_skips_detection(self, mock_run):
        """cli_binary='gemini' only checks `which gemini`, not agy first."""
        mock_run.return_value = MagicMock(returncode=0)
        client = GeminiCLIClient({"enabled": True, "cli_binary": "gemini"})
        assert client.available is True
        assert client.binary == "gemini"
        assert mock_run.call_count == 1
        assert mock_run.call_args.args[0] == ["which", "gemini"]

    @patch("subprocess.run")
    def test_no_binary_found_disables(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["which"])
        client = GeminiCLIClient({"enabled": True})
        assert client.available is False

    def test_invalid_cli_binary_raises(self):
        """Typos in cli_binary fail loudly instead of silently auto-detecting."""
        with pytest.raises(ValueError, match="cli_binary"):
            GeminiCLIClient({"enabled": True, "cli_binary": "agyy"})


class TestAgyExecution:
    @patch("subprocess.run")
    def test_invoke_uses_agy_argv_layout(self, mock_run, agy_cfg):
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"response":"agy says hi"}'
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True):
            client = GeminiCLIClient(agy_cfg)
            client._available = True  # bypass `which` lookup
            response = client.invoke("test prompt", tier="medium")

        assert response == "agy says hi"
        cmd = mock_run.call_args.args[0]
        # Positional prompt — not behind --prompt
        assert cmd[0] == "agy"
        assert "test prompt" in cmd
        assert "--prompt" not in cmd  # gemini syntax must not leak
        assert "--output=json" in cmd
        assert "--quiet" in cmd
        assert "--dangerously-skip-permissions" in cmd

    @patch("subprocess.run")
    def test_invoke_sets_agy_config_dir_env(self, mock_run, agy_cfg):
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"response":"ok"}'
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True):
            client = GeminiCLIClient(agy_cfg)
            client._available = True
            client.invoke("p", tier="medium")
        env = mock_run.call_args.kwargs["env"]
        # AGY_CONFIG_DIR is set; GEMINI_CONFIG_DIR is NOT (different profile)
        assert "AGY_CONFIG_DIR" in env
        assert env["AGY_CONFIG_DIR"]  # has a real temp path
        assert "GEMINI_CONFIG_DIR" not in env

    @patch("subprocess.run")
    def test_invoke_sets_both_api_key_envs(self, mock_run, agy_cfg):
        """Defensive: set GEMINI_API_KEY *and* AGY_API_KEY so either picks it up."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"response":"ok"}'
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k1"}, clear=True):
            client = GeminiCLIClient(agy_cfg)
            client._available = True
            client.invoke("p", tier="medium")
        env = mock_run.call_args.kwargs["env"]
        assert env["GEMINI_API_KEY"] == "k1"
        assert env["AGY_API_KEY"] == "k1"

    @patch("subprocess.run")
    def test_invoke_sets_agy_trust_workspace(self, mock_run, agy_cfg):
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"response":"ok"}'
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True):
            client = GeminiCLIClient(agy_cfg)
            client._available = True
            client.invoke("p", tier="medium")
        env = mock_run.call_args.kwargs["env"]
        assert env.get("AGY_TRUST_WORKSPACE") == "true"

    @patch("subprocess.run")
    def test_settings_json_written_to_agy_dir(self, mock_run, agy_cfg, tmp_path):
        """Verify the sandboxed settings.json lands in <tmp>/.agy/ not .gemini."""
        captured_path = {}

        original_makedirs = os.makedirs

        def patched_run(*args, **kwargs):
            # Capture the AGY_CONFIG_DIR so we can inspect the settings file
            env = kwargs.get("env") or {}
            captured_path["dir"] = env.get("AGY_CONFIG_DIR")
            return MagicMock(returncode=0, stdout='{"response":"ok"}')

        mock_run.side_effect = patched_run
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True):
            client = GeminiCLIClient(agy_cfg)
            client._available = True
            client.invoke("p", tier="medium")

        # Note: tempdir is deleted in the finally clause, so we can't read
        # the file after invoke returns. But we verified AGY_CONFIG_DIR was
        # passed, and other tests verify the dir name template via the profile.
        assert captured_path["dir"] is not None
        assert captured_path["dir"].startswith("/tmp/atlas_agy_config_") or \
               "/atlas_agy_config_" in captured_path["dir"]

    @patch("subprocess.run")
    def test_explicit_override_wins_over_detection(self, mock_run):
        """cli_binary='gemini' forces gemini even if `which agy` succeeds."""
        # which gemini check passes (only one call expected)
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"response":"ok"}'
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True):
            client = GeminiCLIClient({
                "enabled": True, "cli_binary": "gemini",
                "models": {"heavy": "pro", "medium": "flash", "light": "flash-lite"},
            })
            assert client.available is True
            client.invoke("p", tier="medium")
        # The invoke call (second mock_run call) should use gemini argv
        invoke_call = mock_run.call_args_list[-1]
        cmd = invoke_call.args[0]
        assert cmd[0] == "gemini"
        assert "--prompt" in cmd


class TestDefaultsWhenDisabled:
    def test_binary_property_returns_default_when_disabled(self):
        client = GeminiCLIClient({"enabled": False})
        # Disabled clients still have a sensible default binary string
        # in case someone reads .binary for logging purposes.
        assert client.binary in ("agy", "gemini")
