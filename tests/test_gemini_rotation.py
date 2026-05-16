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
    def test_heavy_tier_sticky_until_max_strikes(self, mock_sleep, mock_run, rotation_config):
        """Sticky key behavior: a single quota strike retries on the SAME key.
        Only after max_strikes_per_key (3 by default) does rotation kick in."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True
            # 1 strike on key1, then success on the same key.
            quota_error = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Error 429: capacity exhausted, please retry"
            )
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')
            mock_run.side_effect = [quota_error, success_result]

            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                response = client.invoke("Prompt", tier="heavy")
            assert response == "Success"
            # Both attempts use key1 (sticky) — strike count of 1 < max=3.
            assert mock_run.call_args_list[0][1]['env']["GEMINI_API_KEY"] == "key1"
            assert mock_run.call_args_list[1][1]['env']["GEMINI_API_KEY"] == "key1"
            # No rotation happened.
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
    def test_tier_isolation_sticky_keys(self, mock_sleep, mock_run, rotation_config):
        """Medium tier always uses key 0; heavy tier sticks to _next_heavy_key.
        Successful calls don't rotate (sticky), so all of these stay put."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True

            mock_run.return_value = MagicMock(returncode=0, stdout='{"response": "Success"}')

            # Heavy call: sticky, doesn't advance _next_heavy_key.
            client.invoke("Prompt", tier="heavy")
            assert mock_run.call_args[1]['env']["GEMINI_API_KEY"] == "key1"
            assert client._next_heavy_key == 0

            # Medium call: always key 0 — independent of heavy state.
            client.invoke("Prompt", tier="medium")
            assert mock_run.call_args[1]['env']["GEMINI_API_KEY"] == "key1"

            # Next heavy call: still on key1 (sticky, no failure).
            client.invoke("Prompt", tier="heavy")
            assert mock_run.call_args[1]['env']["GEMINI_API_KEY"] == "key1"
            assert client._next_heavy_key == 0

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_single_key_burns_max_strikes_then_aborts(self, mock_sleep, mock_run, rotation_config):
        """Single key + persistent quota: strikes_per_key attempts on the same
        key, then give up on the model. _rotate_key isn't called any more
        (the new sticky-key path uses _next_heavy_key directly when rotating
        across multiple keys, and a single key has nothing to rotate to)."""
        env = {"GEMINI_API_KEY": "key1"}  # Only one key
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True
            client.max_strikes_per_key = 3

            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Error 429: capacity exhausted"
            )

            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("Prompt", tier="heavy", allow_fallback=False)
            assert result is None
            # Exactly max_strikes_per_key attempts on the single key.
            assert mock_run.call_count == 3
