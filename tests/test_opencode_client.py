"""Tests for OpencodeClient."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.opencode_client import OpencodeClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_NDJSON = (
    '{"type":"step_start","timestamp":1,"sessionID":"ses_1","part":{"type":"step-start"}}\n'
    '{"type":"reasoning","timestamp":2,"sessionID":"ses_1","part":{"type":"reasoning","text":"thinking"}}\n'
    '{"type":"text","timestamp":3,"sessionID":"ses_1","part":{"type":"text","text":"Hello there"}}\n'
    '{"type":"step_finish","timestamp":4,"sessionID":"ses_1","part":{"type":"step-finish"}}\n'
)

SAMPLE_NDJSON_MULTI_TEXT = (
    '{"type":"text","timestamp":1,"sessionID":"ses_1","part":{"type":"text","text":"Part one"}}\n'
    '{"type":"text","timestamp":2,"sessionID":"ses_1","part":{"type":"text","text":" part two"}}\n'
)


def make_mock_run(rc=0, stdout="", stderr=""):
    """Return a MagicMock that simulates subprocess.run."""
    m = MagicMock()
    m.returncode = rc
    m.stdout = stdout
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# TestAvailable
# ---------------------------------------------------------------------------

class TestAvailable:
    def test_binary_on_path(self):
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            client = OpencodeClient({})
            assert client.available is True

    def test_binary_not_on_path(self):
        with patch("shutil.which", return_value=None):
            client = OpencodeClient({})
            assert client.available is False

    def test_result_cached(self):
        mock_which = MagicMock(return_value="/usr/bin/opencode")
        with patch("shutil.which", mock_which):
            client = OpencodeClient({})
            _ = client.available
            _ = client.available
            assert mock_which.call_count == 1

    def test_disabled(self):
        client = OpencodeClient({"enabled": False})
        assert client.available is False

    def test_disabled_does_not_check_binary(self):
        mock_which = MagicMock()
        with patch("shutil.which", mock_which):
            client = OpencodeClient({"enabled": False})
            _ = client.available
            mock_which.assert_not_called()


# ---------------------------------------------------------------------------
# TestInvoke
# ---------------------------------------------------------------------------

class TestInvoke:
    def test_success(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, SAMPLE_NDJSON)),
        ):
            client = OpencodeClient({})
            result = client.invoke("say hello")
            assert result == "Hello there"
            assert client._call_count == 1

    def test_multiple_text_events(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, SAMPLE_NDJSON_MULTI_TEXT)),
        ):
            client = OpencodeClient({})
            result = client.invoke("say something")
            assert result == "Part one part two"

    def test_ignores_non_text_events(self):
        ndjson = (
            '{"type":"step_start","part":{"type":"step-start"}}\n'
            '{"type":"reasoning","part":{"type":"reasoning","text":"thinking"}}\n'
            '{"type":"tool_use","part":{"type":"tool","tool":"bash","state":{"status":"completed"}}}\n'
            '{"type":"text","part":{"type":"text","text":"Only text"}}\n'
        )
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, ndjson)),
        ):
            client = OpencodeClient({})
            result = client.invoke("test")
            assert result == "Only text"

    def test_empty_response_no_text_events(self):
        ndjson = (
            '{"type":"step_start","part":{"type":"step-start"}}\n'
            '{"type":"step_finish","part":{"type":"step-finish"}}\n'
        )
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, ndjson)),
        ):
            client = OpencodeClient({})
            result = client.invoke("test")
            assert result is None

    def test_empty_stdout(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, "")),
        ):
            client = OpencodeClient({})
            result = client.invoke("test")
            assert result is None

    def test_nonzero_exit(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(1, "", "error")),
        ):
            client = OpencodeClient({"max_retries_per_model": 0})
            result = client.invoke("test")
            assert result is None
            assert client._call_count == 0

    def test_timeout(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["opencode"], timeout=300)),
        ):
            client = OpencodeClient({"max_retries_per_model": 0})
            result = client.invoke("test")
            assert result is None
            assert client._call_count == 0

    def test_budget_exhausted(self):
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            client = OpencodeClient({"max_calls_per_run": 2})
            client._call_count = 2
            result = client.invoke("test")
            assert result is None

    def test_tier_model_selection_heavy(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, SAMPLE_NDJSON)) as mock_run,
        ):
            client = OpencodeClient({})
            client.invoke("test", tier="heavy")
            cmd = mock_run.call_args[0][0]
            model_idx = cmd.index("-m") + 1
            assert cmd[model_idx] == "opencode-go/deepseek-v4-pro"

    def test_tier_model_selection_light(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, SAMPLE_NDJSON)) as mock_run,
        ):
            client = OpencodeClient({"models": {"light": "opencode/big-pickle"}})
            client.invoke("test", tier="light")
            cmd = mock_run.call_args[0][0]
            model_idx = cmd.index("-m") + 1
            assert cmd[model_idx] == "opencode/big-pickle"

    def test_system_prompt_prepended(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, SAMPLE_NDJSON)) as mock_run,
        ):
            client = OpencodeClient({})
            client.invoke("user text", system_prompt="System instructions")
            args = mock_run.call_args[0][0]
            # The prompt is the last positional argument
            prompt = args[-1]
            assert "System instructions" in prompt
            assert "User Request: user text" in prompt

    def test_no_system_prompt(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, SAMPLE_NDJSON)) as mock_run,
        ):
            client = OpencodeClient({})
            client.invoke("just this")
            args = mock_run.call_args[0][0]
            prompt = args[-1]
            assert prompt == "just this"

    def test_unavailable_returns_none(self):
        with patch("shutil.which", return_value=None):
            client = OpencodeClient({})
            result = client.invoke("test")
            assert result is None
            assert client._call_count == 0

    def test_error_event_in_ndjson(self):
        ndjson = (
            '{"type":"error","timestamp":1,"sessionID":"ses_1","part":{"type":"error","text":"API error"}}\n'
        )
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, ndjson)),
        ):
            client = OpencodeClient({})
            result = client.invoke("test")
            assert result is None


# ---------------------------------------------------------------------------
# TestParseNDJSON
# ---------------------------------------------------------------------------

class TestParseNDJSON:
    def test_empty_string(self):
        text, error = OpencodeClient._parse_ndjson_result("")
        assert text == ""
        assert error is None

    def test_non_json_lines_skipped(self):
        raw = "not json\n{\"type\":\"text\",\"part\":{\"text\":\"hi\"}}\n"
        text, error = OpencodeClient._parse_ndjson_result(raw)
        assert text == "hi"
        assert error is None

    def test_no_text_type(self):
        raw = '{"type":"step_start"}\n{"type":"step_finish"}\n'
        text, error = OpencodeClient._parse_ndjson_result(raw)
        assert text == ""
        assert error is None

    def test_missing_part_field(self):
        raw = '{"type":"text"}'
        text, error = OpencodeClient._parse_ndjson_result(raw)
        assert text == ""
        assert error is None

    def test_missing_text_in_part(self):
        raw = '{"type":"text","part":{"type":"text"}}'
        text, error = OpencodeClient._parse_ndjson_result(raw)
        assert text == ""
        assert error is None

    def test_multiple_mixed_types(self):
        raw = (
            '{"type":"step_start"}\n'
            '{"type":"text","part":{"type":"text","text":"A"}}\n'
            'invalid json line\n'
            '{"type":"text","part":{"type":"text","text":"B"}}\n'
        )
        text, error = OpencodeClient._parse_ndjson_result(raw)
        assert text == "AB"
        assert error is None

    def test_error_event_in_ndjson_result(self):
        raw = '{"type":"error","error":{"name":"APIError","data":{"message":"Insufficient balance","isRetryable":false,"statusCode":401}}}\n'
        text, error = OpencodeClient._parse_ndjson_result(raw)
        assert text == ""
        assert error == {
            "name": "APIError",
            "message": "Insufficient balance",
            "isRetryable": False,
            "statusCode": 401,
        }

    def test_text_and_error_together(self):
        raw = (
            '{"type":"text","part":{"type":"text","text":"partial"}}\n'
            '{"type":"error","error":{"name":"APIError","data":{"message":"rate limit","isRetryable":true}}}\n'
        )
        text, error = OpencodeClient._parse_ndjson_result(raw)
        assert text == "partial"
        assert error == {
            "name": "APIError",
            "message": "rate limit",
            "isRetryable": True,
            "statusCode": None,
        }


# ---------------------------------------------------------------------------
# TestClassifyError
# ---------------------------------------------------------------------------

class TestClassifyError:
    def test_non_retryable_api_error(self):
        error = {"name": "APIError", "message": "Insufficient balance", "isRetryable": False}
        assert OpencodeClient._classify_error(error) == "fallback"

    def test_retryable_api_error(self):
        error = {"name": "APIError", "message": "Rate limit", "isRetryable": True}
        assert OpencodeClient._classify_error(error) == "retry"

    def test_unknown_error_falls_back(self):
        error = {"name": "UnknownError", "message": "Model not found"}
        assert OpencodeClient._classify_error(error) == "fallback"

    def test_status_code_429_is_retryable(self):
        error = {"name": "APIError", "message": "Too Many Requests", "statusCode": 429}
        assert OpencodeClient._classify_error(error) == "retry"

    def test_status_code_503_is_retryable(self):
        error = {"name": "APIError", "message": "Service Unavailable", "statusCode": 503}
        assert OpencodeClient._classify_error(error) == "retry"

    def test_empty_error_falls_back(self):
        error = {}
        assert OpencodeClient._classify_error(error) == "fallback"


# ---------------------------------------------------------------------------
# TestUsageSummary
# ---------------------------------------------------------------------------

class TestUsageSummary:
    def test_returns_empty_string(self):
        client = OpencodeClient({})
        assert client.get_usage_summary() == ""
        assert client.get_usage_summary(100.0, 200.0) == ""


# ---------------------------------------------------------------------------
# TestDefaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_models(self):
        client = OpencodeClient({})
        assert client.models["heavy"] == "opencode-go/deepseek-v4-pro"
        assert client.models["medium"] == "opencode/deepseek-v4-flash-free"
        assert client.models["light"] == "opencode/deepseek-v4-flash-free"

    def test_custom_model_override(self):
        client = OpencodeClient({"models": {"heavy": "other/model"}})
        assert client.models["heavy"] == "other/model"
        assert client.models["medium"] == "opencode/deepseek-v4-flash-free"

    def test_default_max_calls(self):
        client = OpencodeClient({})
        assert client.max_calls == 50

    def test_custom_max_calls(self):
        client = OpencodeClient({"max_calls_per_run": 10})
        assert client.max_calls == 10

    def test_disabled_by_default(self):
        client = OpencodeClient({})
        assert client.enabled is True  # defaults to enabled (binary check gates it)

    def test_explicitly_disabled(self):
        client = OpencodeClient({"enabled": False})
        assert client.enabled is False


# ---------------------------------------------------------------------------
# TestFallback
# ---------------------------------------------------------------------------

class TestFallback:
    """Validate the per-tier fallback-model chain."""

    def test_default_fallback_models_set(self):
        client = OpencodeClient({})
        for tier in ("heavy", "medium", "light"):
            assert client.fallback_models[tier] == ["opencode-go/deepseek-v4-flash"]

    def test_custom_fallback_models(self):
        client = OpencodeClient({
            "fallback_models": {
                "heavy": ["opencode-go/deepseek-v4-flash"],
                "medium": [],
                "light": ["opencode-go/deepseek-v4-flash", "opencode/x"],
            },
        })
        assert client.fallback_models["heavy"] == ["opencode-go/deepseek-v4-flash"]
        assert client.fallback_models["medium"] == []
        assert client.fallback_models["light"] == ["opencode-go/deepseek-v4-flash", "opencode/x"]

    def test_fallback_after_nonzero_exit(self):
        # Primary fails (rc=1), first fallback succeeds.
        side_effects = [
            make_mock_run(1, "", "quota exceeded"),
            make_mock_run(0, SAMPLE_NDJSON),
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects) as mock_run,
        ):
            client = OpencodeClient({"max_retries_per_model": 0})
            result = client.invoke("test", tier="heavy")
        assert result == "Hello there"
        assert client._call_count == 1
        assert client._tier_served_by["heavy"] == "opencode-go/deepseek-v4-flash"
        assert client._tier_fallback_hits["heavy"] == 1
        # Primary then first fallback were tried
        assert mock_run.call_count == 2
        first_cmd = mock_run.call_args_list[0][0][0]
        second_cmd = mock_run.call_args_list[1][0][0]
        assert first_cmd[first_cmd.index("-m") + 1] == "opencode-go/deepseek-v4-pro"
        assert second_cmd[second_cmd.index("-m") + 1] == "opencode-go/deepseek-v4-flash"

    def test_fallback_after_empty_ndjson(self):
        # Primary returns rc=0 but empty NDJSON; fallback succeeds.
        side_effects = [
            make_mock_run(0, ""),
            make_mock_run(0, SAMPLE_NDJSON),
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects),
        ):
            client = OpencodeClient({"max_retries_per_model": 0, "fallback_models": {"heavy": ["opencode-go/deepseek-v4-flash"]}})
            result = client.invoke("test", tier="heavy")
        assert result == "Hello there"
        assert client._tier_fallback_hits["heavy"] == 1
        assert client._tier_served_by["heavy"] == "opencode-go/deepseek-v4-flash"

    def test_fallback_after_timeout(self):
        side_effects = [
            subprocess.TimeoutExpired(cmd=["opencode"], timeout=300),
            make_mock_run(0, SAMPLE_NDJSON),
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects),
        ):
            client = OpencodeClient({"max_retries_per_model": 0, "fallback_models": {"heavy": ["opencode-go/deepseek-v4-flash"]}})
            result = client.invoke("test", tier="heavy")
        assert result == "Hello there"
        assert client._tier_fallback_hits["heavy"] == 1

    def test_all_models_fail_returns_none(self):
        # Default chain after dedup: [deepseek-v4-flash-free, glm-5.2]
        # (the second default fallback duplicates the primary, so only 2
        # distinct models are tried).
        side_effects = [
            make_mock_run(1, "", "fail1"),
            make_mock_run(1, "", "fail2"),
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects) as mock_run,
        ):
            client = OpencodeClient({"max_retries_per_model": 0})
            result = client.invoke("test", tier="heavy")
        assert result is None
        assert client._call_count == 0
        assert client._tier_failures["heavy"] == 1
        # primary + 1 unique fallback = 2 invocations
        assert mock_run.call_count == 2

    def test_all_models_fail_with_distinct_fallbacks(self):
        # Three distinct models: primary + two unique fallbacks.
        side_effects = [
            make_mock_run(1, "", "fail1"),
            make_mock_run(1, "", "fail2"),
            make_mock_run(1, "", "fail3"),
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects) as mock_run,
        ):
            client = OpencodeClient({
                "max_retries_per_model": 0,
                "models": {"heavy": "opencode-zen/deepseek-v4-flash-free"},
                "fallback_models": {"heavy": ["opencode-go/deepseek-v4-flash", "opencode/mimo-v2.5-free"]},
            })
            result = client.invoke("test", tier="heavy")
        assert result is None
        assert mock_run.call_count == 3

    def test_fallback_disabled_when_empty_list(self):
        # fallback_models: [] disables fallback; primary failure → None.
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(1, "", "err")) as mock_run,
        ):
            client = OpencodeClient({"max_retries_per_model": 0, "fallback_models": {"heavy": []}})
            result = client.invoke("test", tier="heavy")
        assert result is None
        assert mock_run.call_count == 1

    def test_chain_dedupes_primary(self):
        # If primary appears in fallback list, it's only tried once.
        side_effects = [
            make_mock_run(1, "", "fail"),
            make_mock_run(0, SAMPLE_NDJSON),
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects) as mock_run,
        ):
            client = OpencodeClient({
                "max_retries_per_model": 0,
                "models": {"heavy": "opencode-zen/deepseek-v4-flash-free"},
                "fallback_models": {"heavy": [
                    "opencode-zen/deepseek-v4-flash-free",  # duplicates primary
                    "opencode-go/deepseek-v4-flash",
                ]},
            })
            client.invoke("test", tier="heavy")
        # Primary dedup'd, so only 2 calls (primary + deepseek-v4-flash)
        assert mock_run.call_count == 2
        second_cmd = mock_run.call_args_list[1][0][0]
        assert second_cmd[second_cmd.index("-m") + 1] == "opencode-go/deepseek-v4-flash"

    def test_no_fallback_invoked_when_primary_succeeds(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, SAMPLE_NDJSON)) as mock_run,
        ):
            client = OpencodeClient({})
            result = client.invoke("test", tier="medium")
        assert result == "Hello there"
        assert mock_run.call_count == 1
        assert client._tier_fallback_hits["medium"] == 0
        assert client._tier_served_by["medium"] == "opencode/deepseek-v4-flash-free"

    def test_budget_check_applies_during_fallback(self):
        # Burn budget on the primary attempt; even if fallback would succeed,
        # budget exhaustion prevents it.
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(1, "", "quota")) as mock_run,
        ):
            client = OpencodeClient({"max_retries_per_model": 0, "max_calls_per_run": 1})
            client._call_count = 1  # already at budget
            result = client.invoke("test", tier="heavy")
        assert result is None
        assert mock_run.call_count == 0


# ---------------------------------------------------------------------------
# TestFastFallback
# ---------------------------------------------------------------------------

class TestFastFallback:
    """Structured error events in NDJSON trigger immediate fallback."""

    def test_fast_fallback_on_non_retryable_error(self):
        # Primary returns rc=0 with NDJSON error event (isRetryable=False).
        # Should skip straight to fallback without retry.
        error_ndjson = (
            '{"type":"error","error":{"name":"APIError","data":{"message":"Insufficient balance","isRetryable":false}}}\n'
        )
        side_effects = [
            make_mock_run(0, error_ndjson),
            make_mock_run(0, SAMPLE_NDJSON),
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects) as mock_run,
        ):
            client = OpencodeClient({"max_retries_per_model": 2, "fallback_models": {"heavy": ["opencode-go/deepseek-v4-flash"]}})
            result = client.invoke("test", tier="heavy")
        assert result == "Hello there"
        assert client._tier_fallback_hits["heavy"] == 1
        assert client._tier_served_by["heavy"] == "opencode-go/deepseek-v4-flash"
        # Only 2 calls: primary (fast-fail) + fallback (success) — no retries
        assert mock_run.call_count == 2

    def test_fast_fallback_on_unknown_error(self):
        # UnknownError (model not found) also triggers immediate fallback.
        error_ndjson = (
            '{"type":"error","error":{"name":"UnknownError","data":{"message":"Model not found: test"}}}\n'
        )
        side_effects = [
            make_mock_run(0, error_ndjson),
            make_mock_run(0, SAMPLE_NDJSON),
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects),
        ):
            client = OpencodeClient({"max_retries_per_model": 2, "fallback_models": {"heavy": ["opencode-go/deepseek-v4-flash"]}})
            result = client.invoke("test", tier="heavy")
        assert result == "Hello there"
        assert client._tier_fallback_hits["heavy"] == 1


# ---------------------------------------------------------------------------
# TestRetryThenFallback
# ---------------------------------------------------------------------------

class TestRetryThenFallback:
    """Retryable errors get MAX_RETRIES retries before falling back."""

    def test_retryable_error_retries_then_fallback(self):
        # Primary returns retryable error 3 times (initial + 2 retries),
        # then fallback succeeds.
        error_ndjson = (
            '{"type":"error","error":{"name":"APIError","data":{"message":"Rate limit","isRetryable":true}}}\n'
        )
        side_effects = [
            make_mock_run(0, error_ndjson),  # initial
            make_mock_run(0, error_ndjson),  # retry 1
            make_mock_run(0, error_ndjson),  # retry 2 (exhausted)
            make_mock_run(0, SAMPLE_NDJSON),  # fallback success
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects) as mock_run,
            patch("time.sleep"),  # don't actually sleep
        ):
            client = OpencodeClient({"max_retries_per_model": 2, "fallback_models": {"heavy": ["opencode-go/deepseek-v4-flash"]}})
            result = client.invoke("test", tier="heavy")
        assert result == "Hello there"
        assert client._tier_fallback_hits["heavy"] == 1
        assert client._tier_served_by["heavy"] == "opencode-go/deepseek-v4-flash"
        # 3 primary attempts + 1 fallback = 4 total
        assert mock_run.call_count == 4

    def test_retryable_then_success_on_primary(self):
        # First attempt gets rate limit, retry succeeds on the same model.
        error_ndjson = (
            '{"type":"error","error":{"name":"APIError","data":{"message":"Rate limit","isRetryable":true}}}\n'
        )
        side_effects = [
            make_mock_run(0, error_ndjson),  # initial — retryable
            make_mock_run(0, SAMPLE_NDJSON),  # retry — success
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects) as mock_run,
            patch("time.sleep"),
        ):
            client = OpencodeClient({"max_retries_per_model": 2})
            result = client.invoke("test", tier="heavy")
        assert result == "Hello there"
        assert client._tier_fallback_hits["heavy"] == 0  # no fallback
        assert client._tier_served_by["heavy"] == "opencode-go/deepseek-v4-pro"
        assert mock_run.call_count == 2  # initial + retry

    def test_nonzero_exit_retries_then_fallback(self):
        # Non-zero exit without NDJSON error text is retried, then falls back.
        side_effects = [
            make_mock_run(1, "", "overloaded"),
            make_mock_run(1, "", "overloaded"),
            make_mock_run(1, "", "overloaded"),
            make_mock_run(0, SAMPLE_NDJSON),
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects) as mock_run,
            patch("time.sleep"),
        ):
            client = OpencodeClient({"max_retries_per_model": 2, "fallback_models": {"heavy": ["opencode-go/deepseek-v4-flash"]}})
            result = client.invoke("test", tier="heavy")
        assert result == "Hello there"
        assert client._tier_fallback_hits["heavy"] == 1
        assert mock_run.call_count == 4

    def test_timeout_retries_then_fallback(self):
        side_effects = [
            subprocess.TimeoutExpired(cmd=["opencode"], timeout=120),
            subprocess.TimeoutExpired(cmd=["opencode"], timeout=120),
            subprocess.TimeoutExpired(cmd=["opencode"], timeout=120),
            make_mock_run(0, SAMPLE_NDJSON),
        ]
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=side_effects) as mock_run,
            patch("time.sleep"),
        ):
            client = OpencodeClient({"max_retries_per_model": 2, "fallback_models": {"heavy": ["opencode-go/deepseek-v4-flash"]}})
            result = client.invoke("test", tier="heavy")
        assert result == "Hello there"
        assert client._tier_fallback_hits["heavy"] == 1
        assert mock_run.call_count == 4
