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
        """Test that invoke rotates after 3 failed attempts on one key."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True
    
            # Fail 3 times on key1, then succeed on key2
            quota_error = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Resource exhausted (429)."
            )
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')
            mock_run.side_effect = [quota_error, quota_error, quota_error, success_result]
    
            response = client.invoke("Prompt", tier="heavy")
            assert response == "Success"
            assert client._current_key_index == 1
            # First 3 calls should use key1
            for i in range(3):
                assert mock_run.call_args_list[i][1]['env']["GEMINI_API_KEY"] == "key1"
            # 4th call should use key2
            assert mock_run.call_args_list[3][1]['env']["GEMINI_API_KEY"] == "key2"

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

            # To reach index 2, we need to fail 3 times on key1 AND 3 times on key2
            quota_error = subprocess.CalledProcessError(1, ["gemini"], stderr="429")
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')

            # Need 7 calls total: 3x key1, 3x key2, 1x key3
            mock_run.side_effect = [quota_error] * 6 + [success_result]

            # Use heavy tier max_attempts (12) to allow enough retries
            with patch("scripts.gemini_client.logger.warning") as mock_warn:
                client.invoke("Prompt", tier="heavy")
                # Verify 'LAST RESORT' warning was called
                warning_msgs = [call.args[0] for call in mock_warn.call_args_list]
                assert any("LAST RESORT" in msg for msg in warning_msgs)
                assert client._current_key_index == 2

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_exhausted_key_skipping(self, mock_sleep, mock_run, rotation_config):
        """Test that keys hitting hard quotas are skipped in subsequent calls."""
        env = {"GEMINI_API_KEY": "key1,key2,key3"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True
            
            # First call hits a HARD quota on key1
            hard_quota_error = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Daily limit reached."
            )
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')
            mock_run.side_effect = [hard_quota_error, success_result]
            
            client.invoke("Task 1")
            assert 0 in client._exhausted_keys
            assert client._current_key_index == 1
            
            # Second call should SKIP key1 and start at key2
            mock_run.reset_mock()
            mock_run.side_effect = [success_result]
            
            client.invoke("Task 2")
            # Should have used key2 (Index 1) immediately
            last_env = mock_run.call_args[1]['env']
            assert last_env["GEMINI_API_KEY"] == "key2"
            assert client._current_key_index == 1

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_key_persistence_before_rotation(self, mock_sleep, mock_run, rotation_config):
        """Test that we retry the same key multiple times before rotating."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True
            
            # Fail with soft quota error
            quota_error = subprocess.CalledProcessError(1, ["gemini"], stderr="429 Resource Exhausted")
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')
            
            # 1. Fail 2 times, then succeed on 3rd (same key)
            mock_run.side_effect = [quota_error, quota_error, success_result]
            
            response = client.invoke("Prompt")
            assert response == "Success"
            assert client._current_key_index == 0 # Should NOT have rotated
            assert mock_run.call_count == 3
            for call in mock_run.call_args_list:
                assert call[1]['env']["GEMINI_API_KEY"] == "key1"
            
            # 2. Fail 3 times, should rotate on 4th
            mock_run.reset_mock()
            # We need enough total attempts in the decorator to reach the rotation
            # max_attempts is 4 for medium tier in the client
            mock_run.side_effect = [quota_error, quota_error, quota_error, success_result]
            
            response = client.invoke("Prompt")
            assert response == "Success"
            assert client._current_key_index == 1 # Should HAVE rotated
            assert mock_run.call_count == 4
            # First 3 calls use key1
            assert mock_run.call_args_list[0][1]['env']["GEMINI_API_KEY"] == "key1"
            assert mock_run.call_args_list[1][1]['env']["GEMINI_API_KEY"] == "key1"
            assert mock_run.call_args_list[2][1]['env']["GEMINI_API_KEY"] == "key1"
            # 4th call uses key2
            assert mock_run.call_args_list[3][1]['env']["GEMINI_API_KEY"] == "key2"

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
