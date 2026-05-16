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
        # Mock 2 transient (soft-quota) failures then 1 success.
        # Use "429 capacity" — that's a SOFT quota / RPM-burst keyword that
        # triggers retry; "quota exceeded" would now correctly classify as
        # hard daily quota and abort.
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, ["gemini"], stderr="Error: 429 capacity"),
            subprocess.CalledProcessError(1, ["gemini"], stderr="Error: 429 capacity"),
            MagicMock(returncode=0, stdout='{"response": "Success after retries"}')
        ]
        
        # Add multiple keys to allow rotation
        with patch.dict(os.environ, {"GEMINI_API_KEY": "key1,key2,key3"}):
            client = GeminiCLIClient(mock_config)
            client._available = True
            # Shorten delays for test
            client.key_swap_delay = 0.01
            # Disable per-tier pacing so the test doesn't wait 30s between retries
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}

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


class TestPerTierPacing:
    """Verify _pace_tier sleeps the right amount between calls."""

    def test_pace_first_call_no_wait(self, mock_config):
        # First call to a tier shouldn't sleep — _tier_last_call starts at 0.
        client = GeminiCLIClient(mock_config)
        client.tier_min_interval = {"heavy": 30, "medium": 5, "light": 2}
        with patch("time.sleep") as sleep_mock:
            client._pace_tier("heavy")
        sleep_mock.assert_not_called()

    def test_pace_disabled_when_interval_zero(self, mock_config):
        client = GeminiCLIClient(mock_config)
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        client._tier_last_call["heavy"] = 1.0  # would normally trigger wait
        with patch("time.sleep") as sleep_mock, \
             patch("time.time", return_value=2.0):
            client._pace_tier("heavy")
        sleep_mock.assert_not_called()

    def test_pace_waits_for_min_interval(self, mock_config):
        client = GeminiCLIClient(mock_config)
        client.tier_min_interval = {"heavy": 30, "medium": 5, "light": 2}
        # Last call was 5s ago; need to wait 25 more.
        with patch("time.time", side_effect=[1005.0, 1005.0, 1030.0]):
            client._tier_last_call["heavy"] = 1000.0
            with patch("time.sleep") as sleep_mock:
                client._pace_tier("heavy")
        sleep_mock.assert_called_once()
        wait_arg = sleep_mock.call_args[0][0]
        assert 24.9 <= wait_arg <= 25.1

    def test_pace_no_wait_when_interval_already_elapsed(self, mock_config):
        client = GeminiCLIClient(mock_config)
        client.tier_min_interval = {"heavy": 30, "medium": 5, "light": 2}
        # Last call was 60s ago; already past min interval.
        with patch("time.time", side_effect=[1060.0, 1060.0]):
            client._tier_last_call["heavy"] = 1000.0
            with patch("time.sleep") as sleep_mock:
                client._pace_tier("heavy")
        sleep_mock.assert_not_called()

    def test_pace_records_intent_time_not_completion(self, mock_config):
        # _tier_last_call should be set to "now" before the call, so the next
        # caller paces relative to when this attempt started, not when it ended.
        client = GeminiCLIClient(mock_config)
        client.tier_min_interval = {"heavy": 30, "medium": 5, "light": 2}
        with patch("time.time", side_effect=[100.0, 100.0]):
            with patch("time.sleep"):
                client._pace_tier("heavy")
        assert client._tier_last_call["heavy"] == 100.0

    def test_pace_per_tier_independent(self, mock_config):
        # Setting heavy doesn't affect medium pacing.
        client = GeminiCLIClient(mock_config)
        client.tier_min_interval = {"heavy": 30, "medium": 5, "light": 2}
        with patch("time.time", return_value=1000.0):
            with patch("time.sleep"):
                client._pace_tier("heavy")
        # Reservation stamp = now + wait. First call wait=0, so stamp=now.
        assert client._tier_last_call["heavy"] == 1000.0
        assert client._tier_last_call["medium"] == 0.0
        assert client._tier_last_call["light"] == 0.0

    def test_pace_releases_lock_before_sleeping(self, mock_config):
        """The pace sleep must NOT hold _call_lock or other tiers stall.

        Verifies by checking that during a heavy-tier sleep, a separate
        thread can still acquire _call_lock for any other operation.
        """
        import threading
        import time as _time
        real_sleep = _time.sleep  # capture before patching
        client = GeminiCLIClient(mock_config)
        client.tier_min_interval = {"heavy": 30, "medium": 5, "light": 2}
        client._tier_last_call["heavy"] = _time.time()  # force a wait

        lock_acquired = threading.Event()
        sleep_started = threading.Event()

        def secondary_grabs_lock():
            sleep_started.wait(timeout=1.0)
            with client._call_lock:
                lock_acquired.set()

        t = threading.Thread(target=secondary_grabs_lock)
        t.start()

        def fake_sleep(seconds):
            sleep_started.set()
            real_sleep(0.05)  # let secondary thread acquire

        with patch("scripts.gemini_client.time.sleep", side_effect=fake_sleep):
            client._pace_tier("heavy")
        t.join(timeout=1.0)
        assert lock_acquired.is_set(), \
            "secondary thread should have acquired _call_lock during pace sleep"


class TestRoundRobinHeavy:
    """Verify per-call round-robin key selection on heavy tier."""

    def test_round_robin_advances_per_call(self, mock_config):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k1,k2,k3"}, clear=True):
            client = GeminiCLIClient(mock_config)
            client._available = True
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}

            envs_used = []

            def capture_env(*args, **kwargs):
                envs_used.append(kwargs["env"]["GEMINI_API_KEY"])
                return MagicMock(returncode=0, stdout='{"response": "ok"}')

            with patch("subprocess.run", side_effect=capture_env):
                client.invoke("p", tier="heavy", allow_fallback=False)
                client.invoke("p", tier="heavy", allow_fallback=False)
                client.invoke("p", tier="heavy", allow_fallback=False)
                client.invoke("p", tier="heavy", allow_fallback=False)
            # 3 keys, 4 calls -> k1, k2, k3, k1
            assert envs_used == ["k1", "k2", "k3", "k1"]

    def test_round_robin_inactive_with_single_key(self, mock_config):
        # With one key, every heavy call uses it; _next_heavy_key shouldn't
        # silently advance and confuse later state inspection.
        with patch.dict(os.environ, {"GEMINI_API_KEY": "only_key"}, clear=True):
            client = GeminiCLIClient(mock_config)
            client._available = True
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}

            envs_used = []

            def capture_env(*args, **kwargs):
                envs_used.append(kwargs["env"]["GEMINI_API_KEY"])
                return MagicMock(returncode=0, stdout='{"response": "ok"}')

            with patch("subprocess.run", side_effect=capture_env):
                client.invoke("p", tier="heavy", allow_fallback=False)
                client.invoke("p", tier="heavy", allow_fallback=False)
            assert envs_used == ["only_key", "only_key"]
            assert client._next_heavy_key == 0  # untouched


class TestConfigDirCaching:
    """Verify the gemini config dir is built once and reused."""

    def test_config_dir_built_lazily(self, mock_config):
        client = GeminiCLIClient(mock_config)
        assert client._config_dir is None
        path = client._ensure_config_dir()
        assert client._config_dir == path
        assert os.path.isdir(path)
        assert os.path.exists(os.path.join(path, ".gemini", "settings.json"))
        client.cleanup()

    def test_config_dir_reused_on_second_call(self, mock_config):
        client = GeminiCLIClient(mock_config)
        first = client._ensure_config_dir()
        second = client._ensure_config_dir()
        assert first == second
        client.cleanup()

    def test_cleanup_removes_dir_and_resets_state(self, mock_config):
        client = GeminiCLIClient(mock_config)
        path = client._ensure_config_dir()
        assert os.path.isdir(path)
        client.cleanup()
        assert client._config_dir is None
        assert not os.path.exists(path)

    def test_cleanup_idempotent(self, mock_config):
        client = GeminiCLIClient(mock_config)
        client._ensure_config_dir()
        client.cleanup()
        client.cleanup()  # should not raise
        assert client._config_dir is None

    def test_settings_file_honors_internal_max_attempts(self):
        import json as _json
        client = GeminiCLIClient({"enabled": True, "internal_max_attempts": 7})
        path = client._ensure_config_dir()
        with open(os.path.join(path, ".gemini", "settings.json")) as f:
            data = _json.load(f)
        assert data["general"]["maxAttempts"] == 7
        # Only documented keys are written. autoAccept and requestTimeout
        # are not recognized by gemini-cli — auto-approval is handled via
        # the --approval-mode yolo CLI flag, and the subprocess timeout
        # caps the call duration in Python.
        assert "autoAccept" not in data.get("tools", {})
        assert "requestTimeout" not in data["general"]
        client.cleanup()


class TestBudgetEnforcement:
    """Verify max_calls_per_run is honored and only counts successful calls."""

    @patch("subprocess.run")
    def test_budget_blocks_after_exhaustion(self, mock_run, mock_config):
        # mock_config sets max_calls_per_run=10
        client = GeminiCLIClient(mock_config)
        client._available = True
        client._call_count = 10  # at the cap
        result = client.invoke("p", tier="medium")
        assert result is None
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_budget_increments_only_on_success(self, mock_run, mock_config):
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}

        # Failure that won't retry (no quota/network keyword).
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gemini"], stderr="some non-transient error"
        )
        result = client.invoke("p", tier="medium", allow_fallback=False)
        assert result is None
        assert client._call_count == 0

        # Now a success.
        mock_run.side_effect = None
        mock_run.return_value = MagicMock(returncode=0, stdout='{"response": "ok"}')
        result = client.invoke("p", tier="medium")
        assert result == "ok"
        assert client._call_count == 1


class TestInvokeFallbackChain:
    """Heavy → medium → light fallback after retries exhaust."""

    @patch("subprocess.run")
    def test_falls_back_to_next_tier_on_persistent_failure(self, mock_run, mock_config):
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}

        # Heavy persistently 429s; medium succeeds.
        quota = subprocess.CalledProcessError(1, ["gemini"], stderr="429 quota")
        success = MagicMock(returncode=0, stdout='{"response": "from-medium"}')

        # ~Use small max_attempts via patching to keep test fast. We can't easily
        # change max_attempts without a config knob, so patch wait + cap attempts via
        # exhausting the side_effect list.
        # Heavy max_attempts default in mock_config = 4 (since heavy_max_attempts not set
        # and tier=='heavy' would normally use heavy_max_attempts, but mock_config doesn't
        # override). Actually heavy_max_attempts default is 20. Cap by failing all 20.
        client.heavy_max_attempts = 3  # test override
        responses = [quota] * 3 + [success]
        mock_run.side_effect = responses

        with patch("scripts.gemini_client.wait_random_exponential",
                   return_value=lambda x: 0.001):
            result = client.invoke("p", tier="heavy")
        assert result == "from-medium"

    @patch("subprocess.run")
    def test_disable_fallback_returns_none(self, mock_run, mock_config):
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        client.heavy_max_attempts = 2

        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gemini"], stderr="429 quota exceeded"
        )

        with patch("scripts.gemini_client.wait_random_exponential",
                   return_value=lambda x: 0.001):
            result = client.invoke("p", tier="heavy", allow_fallback=False)
        assert result is None


class TestUsageStatsParsing:
    """Verify that parsed JSON output updates token counters correctly."""

    @patch("subprocess.run")
    def test_parses_prompt_and_candidates_tokens(self, mock_run, mock_config):
        """Current gemini-cli schema uses 'prompt' (not 'input') for input tokens."""
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                '{"response": "hello", "stats": {"models": {"test-medium": '
                '{"tokens": {"prompt": 100, "candidates": 50, "total": 150}}}}}'
            ),
        )
        client.invoke("prompt", tier="medium")
        assert client.usage_stats["medium"]["in_tokens"] == 100
        assert client.usage_stats["medium"]["out_tokens"] == 50
        assert client.usage_stats["medium"]["calls"] == 1

    @patch("subprocess.run")
    def test_falls_back_to_input_field_for_older_cli(self, mock_run, mock_config):
        """Older gemini-cli versions used 'input' instead of 'prompt'."""
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                '{"response": "hello", "stats": {"models": {"test-medium": '
                '{"tokens": {"input": 75, "candidates": 25}}}}}'
            ),
        )
        client.invoke("prompt", tier="medium")
        assert client.usage_stats["medium"]["in_tokens"] == 75

    @patch("subprocess.run")
    def test_handles_unparseable_json_gracefully(self, mock_run, mock_config):
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        mock_run.return_value = MagicMock(returncode=0, stdout="raw text not JSON")
        result = client.invoke("prompt", tier="medium")
        assert result == "raw text not JSON"
        # Even on JSON failure, should still count the call.
        assert client.usage_stats["medium"]["calls"] == 1
        # No token counts since we couldn't parse them.
        assert client.usage_stats["medium"]["in_tokens"] == 0


class TestHardQuotaGate:
    """Verify hard-daily-quota errors abort retries on every code path."""

    @patch("subprocess.run")
    def test_hard_quota_tries_all_heavy_keys_before_aborting(self, mock_run, mock_config):
        """Each key has its own RPD on free tier, so we try every loaded key
        once before declaring the tier done. After N strikes (where N is the
        number of configured keys), abort."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k1,k2,k3"}, clear=True):
            client = GeminiCLIClient(mock_config)
            client._available = True
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
            client.heavy_max_attempts = 20  # plenty of headroom
            client.ignore_hard_quota = False

            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["gemini"], stderr="429: daily quota exceeded"
            )
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="heavy", allow_fallback=False)
            assert result is None
            # 3 keys → 3 attempts → abort.
            assert mock_run.call_count == 3

    @patch("subprocess.run")
    def test_hard_quota_aborts_single_key_heavy_immediately(self, mock_run, mock_config):
        """Only one key configured: no sibling to try, so one strike = abort."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "only_key"}, clear=True):
            client = GeminiCLIClient(mock_config)
            client._available = True
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
            client.ignore_hard_quota = False

            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["gemini"], stderr="429: daily quota exceeded"
            )
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="heavy", allow_fallback=False)
            assert result is None
            assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_hard_quota_succeeds_when_second_key_has_headroom(self, mock_run, mock_config):
        """If key 1 is RPD'd but key 2 still has quota, we succeed on retry."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "exhausted,fresh"}, clear=True):
            client = GeminiCLIClient(mock_config)
            client._available = True
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
            client.ignore_hard_quota = False

            mock_run.side_effect = [
                subprocess.CalledProcessError(1, ["gemini"], stderr="terminalquotaerror: daily limit"),
                MagicMock(returncode=0, stdout='{"response": "from fresh key"}'),
            ]
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="heavy", allow_fallback=False)
            assert result == "from fresh key"
            assert mock_run.call_count == 2
            # First attempt used k1, retry used k2 (round-robin advanced).
            assert mock_run.call_args_list[0][1]["env"]["GEMINI_API_KEY"] == "exhausted"
            assert mock_run.call_args_list[1][1]["env"]["GEMINI_API_KEY"] == "fresh"

    @patch("subprocess.run")
    def test_hard_quota_terminalquotaerror_keyword_matched(self, mock_run, mock_config):
        """The exact production log keyword 'TerminalQuotaError' is recognized."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "only_key"}, clear=True):
            client = GeminiCLIClient(mock_config)
            client._available = True
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
            client.ignore_hard_quota = False

            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["gemini"],
                # Real wording observed in production logs.
                stderr="warning: 256-color support not detected\nTerminalQuotaError: You have exhausted your daily quota on this model.",
            )
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="heavy", allow_fallback=False)
            assert result is None
            assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_hard_quota_aborts_non_heavy(self, mock_run, mock_config):
        """Non-heavy (medium/light) tiers also need to honor the hard-quota gate."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k1"}, clear=True):
            client = GeminiCLIClient(mock_config)
            client._available = True
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
            client.ignore_hard_quota = False

            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["gemini"], stderr="rpd limit reached"
            )
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="medium", allow_fallback=False)
            assert result is None
            assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_hard_quota_overridden_by_ignore_flag(self, mock_run, mock_config):
        """ignore_hard_quota=True keeps retrying even on hard quota messages."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k1,k2"}, clear=True):
            client = GeminiCLIClient(mock_config)
            client._available = True
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
            client.heavy_max_attempts = 3
            client.ignore_hard_quota = True

            mock_run.side_effect = [
                subprocess.CalledProcessError(1, ["gemini"], stderr="429 daily quota exceeded"),
                subprocess.CalledProcessError(1, ["gemini"], stderr="429 daily quota exceeded"),
                MagicMock(returncode=0, stdout='{"response": "ok"}'),
            ]
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="heavy", allow_fallback=False)
            assert result == "ok"
            assert mock_run.call_count == 3

    @patch("subprocess.run")
    def test_soft_quota_still_retries(self, mock_run, mock_config):
        """Pure 429/capacity (no hard-quota keyword) should retry as before."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k1,k2"}, clear=True):
            client = GeminiCLIClient(mock_config)
            client._available = True
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
            client.heavy_max_attempts = 3
            client.ignore_hard_quota = False

            mock_run.side_effect = [
                subprocess.CalledProcessError(1, ["gemini"], stderr="429 capacity"),
                MagicMock(returncode=0, stdout='{"response": "ok"}'),
            ]
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="heavy", allow_fallback=False)
            assert result == "ok"
            assert mock_run.call_count == 2

    @patch("subprocess.run")
    def test_non_quota_error_fails_fast(self, mock_run, mock_config):
        """Errors that don't match any keyword (e.g., unknown flag) fail fast."""
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gemini"], stderr="Unknown command flag --raw-output"
        )
        result = client.invoke("p", tier="medium", allow_fallback=False)
        assert result is None
        # No retry — error message didn't match any retry keyword.
        assert mock_run.call_count == 1


class TestTimeoutAndValueErrorRetries:
    """Verify timeout + ValueError retry caps don't burn the heavy budget."""

    @patch("subprocess.run")
    def test_timeout_capped_separately_from_quota_retries(self, mock_run, mock_config):
        """Persistent subprocess hangs should NOT consume the full 20-attempt budget."""
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        client.heavy_max_attempts = 20
        # Every call hangs.
        mock_run.side_effect = subprocess.TimeoutExpired(["gemini"], 900)
        with patch("scripts.gemini_client.wait_random_exponential",
                   return_value=lambda x: 0.001):
            result = client.invoke("p", tier="heavy", allow_fallback=False)
        assert result is None
        # 4 attempts total: initial + 3 retries (max_timeout_retries=3).
        assert mock_run.call_count == 4

    @patch("subprocess.run")
    def test_empty_response_value_error_is_transient(self, mock_run, mock_config):
        """Empty model response retries (often safety-filter false-trip)."""
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        client.heavy_max_attempts = 3
        # First call returns empty response, second returns content.
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout='{"response": ""}'),
            MagicMock(returncode=0, stdout='{"response": "real answer"}'),
        ]
        with patch("scripts.gemini_client.wait_random_exponential",
                   return_value=lambda x: 0.001):
            result = client.invoke("p", tier="heavy", allow_fallback=False)
        assert result == "real answer"
        assert mock_run.call_count == 2

    def test_unrelated_value_error_is_not_transient(self, mock_config):
        """Non-empty-response ValueError shouldn't loop forever."""
        client = GeminiCLIClient(mock_config)
        # Build the is_transient_error closure by entering invoke just enough
        # to access it, but easier: directly check that a generic ValueError
        # raised inside _execute_command would NOT trigger a retry. We simulate
        # via mocking _execute_command to raise a programming-bug ValueError.
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        with patch.object(client, "_execute_command",
                          side_effect=ValueError("bad config: missing key")):
            with patch("scripts.gemini_client.wait_random_exponential",
                       return_value=lambda x: 0.001):
                result = client.invoke("p", tier="medium", allow_fallback=False)
        # The non-transient ValueError should propagate, then invoke catches
        # at the outer try/except and returns None without retrying.
        assert result is None


class TestEnvVarHandling:
    """Verify GOOGLE_API_KEY etc. are unset (popped), not blanked, and that
    --skip-trust is in the cmd args, not the env."""

    @patch("subprocess.run")
    def test_google_api_key_is_popped_not_blanked(self, mock_run, mock_config):
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "k1",
            "GOOGLE_API_KEY": "should_be_removed",
            "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds",
            "CLOUDSDK_AUTH_ACCESS_TOKEN": "stale_token",
        }, clear=True):
            client = GeminiCLIClient(mock_config)
            client._available = True
            client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
            mock_run.return_value = MagicMock(returncode=0, stdout='{"response": "ok"}')
            client.invoke("p", tier="medium")
        env = mock_run.call_args[1]["env"]
        assert "GOOGLE_API_KEY" not in env
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in env
        assert "CLOUDSDK_AUTH_ACCESS_TOKEN" not in env
        # GEMINI_API_KEY should still be set.
        assert env["GEMINI_API_KEY"] == "k1"

    @patch("subprocess.run")
    def test_skip_trust_in_cmd_not_env(self, mock_run, mock_config):
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        mock_run.return_value = MagicMock(returncode=0, stdout='{"response": "ok"}')
        client.invoke("p", tier="medium")
        cmd = mock_run.call_args[0][0]
        assert "--skip-trust" in cmd
        env = mock_run.call_args[1]["env"]
        assert "GEMINI_CLI_TRUST_WORKSPACE" not in env

    @patch("subprocess.run")
    def test_required_cli_flags_present(self, mock_run, mock_config):
        client = GeminiCLIClient(mock_config)
        client._available = True
        client.tier_min_interval = {"heavy": 0, "medium": 0, "light": 0}
        mock_run.return_value = MagicMock(returncode=0, stdout='{"response": "ok"}')
        client.invoke("p", tier="medium")
        cmd = mock_run.call_args[0][0]
        for required in [
            "--model", "--prompt", "--approval-mode", "yolo",
            "--raw-output", "--accept-raw-output-risk",
            "--output-format", "json", "--skip-trust",
        ]:
            assert required in cmd, f"missing flag/value: {required}"


class TestApiKeyLoading:
    """Cover the env-var key loader paths."""

    def test_no_keys_logs_warning(self, caplog):
        with patch.dict(os.environ, {}, clear=True):
            client = GeminiCLIClient({})
        assert client._api_keys == []

    def test_single_key_no_rotation(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "one"}, clear=True):
            client = GeminiCLIClient({})
        assert client._api_keys == ["one"]
        # _rotate_key returns False when nothing to rotate to.
        assert client._rotate_key() is False
        assert client._current_key_index == 0

    def test_comma_separated_keys_split(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "a, b, c "}, clear=True):
            client = GeminiCLIClient({})
        assert client._api_keys == ["a", "b", "c"]

    def test_suffix_keys_sorted_after_primary(self):
        env = {
            "GEMINI_API_KEY": "primary",
            "GEMINI_API_KEY_Z": "zkey",
            "GEMINI_API_KEY_A": "akey",
        }
        with patch.dict(os.environ, env, clear=True):
            client = GeminiCLIClient({})
        assert client._api_keys == ["primary", "akey", "zkey"]
