#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for GeminiCLIClient Key Rotation."""

import os
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest
from scripts.gemini_client import GeminiCLIClient

@pytest.fixture
def rotation_config():
    return {
        "enabled": True,
        "max_calls_per_run": 10,
        "key_swap_delay": 0.001, # Set to 1ms for fast testing as requested
        "cli_binary": "gemini",  # pin to legacy binary so test mocks are stable
        "models": {
            "heavy": "test-heavy",
            "medium": "test-medium",
            "light": "test-light",
        }
    }

class TestGeminiRotation:
    def test_load_keys_alphanumeric(self):
        """Test loading keys from descriptive alphanumeric env vars."""
        env = {
            "GEMINI_API_KEY_WORK": "work_key",
            "GEMINI_API_KEY_BACKUP": "backup_key",
            "GEMINI_API_KEY_PERSONAL": "personal_key",
            "GEMINI_API_KEY": "primary_key"
        }
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient({})
            keys = client._api_keys
            # Verify primary is first
            assert keys[0] == "primary_key"
            # Verify others are sorted alphanumerically: BACKUP, PERSONAL, WORK
            assert keys[1] == "backup_key"
            assert keys[2] == "personal_key"
            assert keys[3] == "work_key"
            assert len(keys) == 4

    def test_load_keys_deduplication(self):
        """Test that duplicate keys are not added twice."""
        env = {
            "GEMINI_API_KEY": "same_key",
            "GEMINI_API_KEY_1": "same_key",
            "GEMINI_API_KEY_2": "other_key"
        }
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient({})
            assert client._api_keys == ["same_key", "other_key"]

    @patch("time.sleep")
    def test_rotate_key_logic(self, mock_sleep, rotation_config):
        """Test manual rotation logic and delay."""
        env = {"GEMINI_API_KEY": "k1,k2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            assert client._get_current_key() == "k1"
            
            rotated = client._rotate_key()
            assert rotated is True
            assert client._get_current_key() == "k2"
            
            # Verify sleep was called with base + jitter
            assert mock_sleep.called
            delay = mock_sleep.call_args[0][0]
            assert 0.001 <= delay <= 10.001 # 1ms base + up to 10s jitter

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_invoke_with_rotation_on_quota(self, mock_sleep, mock_run, rotation_config):
        """Test that invoke rotates keys on out-of-usage quota errors."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True

            # "Resource exhausted" is out-of-usage → rotate to next key each time.
            quota_error = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Resource exhausted (429)."
            )
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')
            mock_run.side_effect = [quota_error, quota_error, success_result]

            response = client.invoke("Prompt", tier="heavy")
            assert response == "Success"
            # key1 → key2 (attempt 2) → key1 (attempt 3 success)
            assert client._get_current_key() == "key1"
            assert mock_run.call_args_list[0][1]['env']["GEMINI_API_KEY"] == "key1"
            assert mock_run.call_args_list[1][1]['env']["GEMINI_API_KEY"] == "key2"
            assert mock_run.call_args_list[2][1]['env']["GEMINI_API_KEY"] == "key1"

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_first_free_preference(self, mock_sleep, mock_run, rotation_config):
        """Test that every invoke resets to prefer the first available key."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True
            
            # 1. Manually set index to 1 (simulating a previous rotation)
            client._current_key_index = 1
            
            # 2. Call medium tier
            mock_run.return_value = MagicMock(returncode=0, stdout='{"response": "Success"}')
            client.invoke("Prompt", tier="medium")
            
            # 3. Verify it reset to key1 (index 0)
            last_env = mock_run.call_args[1]['env']
            assert last_env["GEMINI_API_KEY"] == "key1"
            assert client._current_key_index == 0

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_last_resort_warning(self, mock_sleep, mock_run, rotation_config):
        """Test that a warning is logged when rotating to Key 3 (Index 2)."""
        env = {"GEMINI_API_KEY": "key1,key2,key3"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True

            # Start at key1
            client._current_key_index = 0

            # "429" text is classified fallback → rotate each attempt.
            quota_error = subprocess.CalledProcessError(1, ["gemini"], stderr="429")
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')

            # 3 attempts: key1 → key2 → key3, then success on key1 next cycle
            mock_run.side_effect = [quota_error, quota_error, quota_error, success_result]

            with patch("scripts.gemini_client.logger.warning") as mock_warn:
                client.invoke("Prompt", tier="heavy")
                # Verify 'LAST RESORT' warning was called when reaching index 2
                warning_msgs = [call.args[0] for call in mock_warn.call_args_list]
                assert any("LAST RESORT" in msg for msg in warning_msgs)

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_exhausted_key_skipping(self, mock_sleep, mock_run, rotation_config):
        """Test that rotation moves forward even when keys hit out-of-usage."""
        cfg = rotation_config.copy()
        cfg["track_hard_quotas"] = True
        env = {"GEMINI_API_KEY": "key1,key2,key3"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(cfg)
            client._available = True

            # A hard-quota style error → classified fallback → rotate to next.
            hard_quota_error = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Daily limit reached."
            )
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')
            mock_run.side_effect = [hard_quota_error, success_result]

            client.invoke("Task 1")
            # After out-of-usage on key1 we rotate to key2, which succeeds.
            assert client._current_key_index == 1

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_key_persistence_before_rotation(self, mock_sleep, mock_run, rotation_config):
        """Test soft-quota (transient) errors retry on the same key before rotating."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True

            # Transient error text → classified "retry" (not out-of-usage).
            retryable_error = subprocess.CalledProcessError(
                1, ["gemini"], stderr="HTTP 503 temporarily unavailable"
            )
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')

            # max_soft_per_key=2: retry key1 twice, then rotate to key2 and succeed.
            mock_run.side_effect = [retryable_error, retryable_error, success_result]

            response = client.invoke("Prompt")
            assert response == "Success"
            assert mock_run.call_count == 3
            # First 2 transient retries on key1, then rotate to key2 for success.
            assert mock_run.call_args_list[0][1]['env']["GEMINI_API_KEY"] == "key1"
            assert mock_run.call_args_list[1][1]['env']["GEMINI_API_KEY"] == "key1"
            assert mock_run.call_args_list[2][1]['env']["GEMINI_API_KEY"] == "key2"

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_rotation_exhausted(self, mock_sleep, mock_run, rotation_config):
        """Test behavior when ONLY one key exists and hits quota for heavy tier."""
        env = {"GEMINI_API_KEY": "key1"} # Only one key
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True
            
            # Fail with quota error
            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Quota exceeded."
            )
            
            # Capture the rotation attempt
            with patch.object(client, '_rotate_key', side_effect=client._rotate_key) as spy_rotate:
                client.invoke("Prompt", tier="heavy", allow_fallback=False)
                
                # Should have tried to rotate
                assert spy_rotate.called
                # Final check: index stayed at 0 because there were no other keys
                assert client._current_key_index == 0
