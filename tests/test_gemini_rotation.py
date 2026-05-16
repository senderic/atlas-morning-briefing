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
        "models": {
            "heavy": "test-heavy",
            "medium": "test-medium",
            "light": "test-light",
        },
        # Disable per-tier pacing so tests don't sleep between calls.
        "tier_min_interval_seconds": {"heavy": 0, "medium": 0, "light": 0},
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
    def test_invoke_heavy_tier_round_robins_keys(self, mock_sleep, mock_run, rotation_config):
        """With multiple keys, heavy-tier calls round-robin per call (no rotation)."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True

            # Simulate first heavy call hitting quota on key1, succeeding on key2.
            quota_error = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Error 429: capacity exhausted, please retry"
            )
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')
            mock_run.side_effect = [quota_error, success_result]

            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                response = client.invoke("Prompt", tier="heavy")
            assert response == "Success"
            # Round-robin advances per call: attempt 1 used key1, retry used key2.
            assert mock_run.call_args_list[0][1]['env']["GEMINI_API_KEY"] == "key1"
            assert mock_run.call_args_list[1][1]['env']["GEMINI_API_KEY"] == "key2"
            # _current_key_index should be unchanged: round-robin uses _next_heavy_key
            # instead of mutating the rotation cursor.
            assert client._current_key_index == 0
            # _next_heavy_key advanced once per attempt: 0 -> 1 -> 0 (mod 2).
            assert client._next_heavy_key == 0

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_invoke_medium_tier_stays_on_key0(self, mock_sleep, mock_run, rotation_config):
        """Non-heavy tiers always use key 0; quota retries reuse the same key."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True

            quota_error = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Error 429: capacity exhausted, please retry"
            )
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')
            mock_run.side_effect = [quota_error, success_result]

            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                response = client.invoke("Prompt", tier="medium")
            assert response == "Success"
            assert mock_run.call_args_list[0][1]['env']["GEMINI_API_KEY"] == "key1"
            assert mock_run.call_args_list[1][1]['env']["GEMINI_API_KEY"] == "key1"

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_tier_isolation_round_robin(self, mock_sleep, mock_run, rotation_config):
        """Medium tier always uses key 0 regardless of heavy round-robin state."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True

            mock_run.return_value = MagicMock(returncode=0, stdout='{"response": "Success"}')

            # Heavy call advances _next_heavy_key from 0 -> 1.
            client.invoke("Prompt", tier="heavy")
            assert mock_run.call_args[1]['env']["GEMINI_API_KEY"] == "key1"
            assert client._next_heavy_key == 1

            # Medium call still uses key 0 (key1) — independent of heavy state.
            client.invoke("Prompt", tier="medium")
            assert mock_run.call_args[1]['env']["GEMINI_API_KEY"] == "key1"

            # Next heavy call uses the round-robin'd key (index 1 = key2).
            client.invoke("Prompt", tier="heavy")
            assert mock_run.call_args[1]['env']["GEMINI_API_KEY"] == "key2"
            assert client._next_heavy_key == 0

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_rotation_exhausted(self, mock_sleep, mock_run, rotation_config):
        """Test behavior when ONLY one key exists and hits quota for heavy tier."""
        env = {"GEMINI_API_KEY": "key1"} # Only one key
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True
            
            # Fail with soft quota / 429 capacity error so retries actually
            # happen (a hard "quota exceeded" message would now correctly
            # abort instead of rotating).
            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Error 429: capacity exhausted"
            )
            
            # Capture the rotation attempt
            with patch.object(client, '_rotate_key', side_effect=client._rotate_key) as spy_rotate:
                client.invoke("Prompt", tier="heavy", allow_fallback=False)
                
                # Should have tried to rotate
                assert spy_rotate.called
                # Final check: index stayed at 0 because there were no other keys
                assert client._current_key_index == 0
