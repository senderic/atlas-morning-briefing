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
        """Test that invoke automatically rotates keys ONLY for heavy tier."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True
            
            # 1. Test heavy tier rotates
            quota_error = subprocess.CalledProcessError(
                1, ["gemini"], stderr="Quota exceeded for this model."
            )
            success_result = MagicMock(returncode=0, stdout='{"response": "Success"}')
            mock_run.side_effect = [quota_error, success_result]
            
            response = client.invoke("Prompt", tier="heavy")
            assert response == "Success"
            assert client._current_key_index == 1
            assert mock_run.call_args_list[0][1]['env']["GEMINI_API_KEY"] == "key1"
            assert mock_run.call_args_list[1][1]['env']["GEMINI_API_KEY"] == "key2"
            
            # 2. Test medium tier does NOT rotate and stays on key1
            mock_run.reset_mock()
            mock_run.side_effect = [quota_error, success_result]
            
            # Reset index for clean test
            client._current_key_index = 0
            
            # Medium tier call - should hit quota error but NOT rotate
            # It will retry with the SAME key (index 0) because tier != "heavy"
            # Since mock_run.side_effect has 2 items, the retry will succeed on 2nd attempt
            response = client.invoke("Prompt", tier="medium")
            
            assert response == "Success"
            assert client._current_key_index == 0 # Should still be 0
            assert mock_run.call_args_list[0][1]['env']["GEMINI_API_KEY"] == "key1"
            assert mock_run.call_args_list[1][1]['env']["GEMINI_API_KEY"] == "key1"

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_tier_isolation_after_rotation(self, mock_sleep, mock_run, rotation_config):
        """Test that non-heavy tiers use index 0 even after heavy tier has rotated the index."""
        env = {"GEMINI_API_KEY": "key1,key2"}
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient(rotation_config)
            client._available = True
            
            # 1. Manually set index to 1 (simulating a previous heavy rotation)
            client._current_key_index = 1
            
            # 2. Call medium tier
            mock_run.return_value = MagicMock(returncode=0, stdout='{"response": "Success"}')
            client.invoke("Prompt", tier="medium")
            
            # 3. Verify it used key1 (index 0) despite client._current_key_index being 1
            last_env = mock_run.call_args[1]['env']
            assert last_env["GEMINI_API_KEY"] == "key1"
            
            # 4. Call heavy tier
            client.invoke("Prompt", tier="heavy")
            
            # 5. Verify heavy tier DID use key2 (index 1)
            last_env = mock_run.call_args[1]['env']
            assert last_env["GEMINI_API_KEY"] == "key2"

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
