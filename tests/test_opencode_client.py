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
            assert cmd[model_idx] == "opencode/muse-spark-1.3-contributor-free"

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

    def test_names_the_model_and_uses_configured_free_pricing(self):
        client = OpencodeClient(
            {"pricing": {"input_per_million": 0.0, "output_per_million": 0.0}}
        )
        client._tier_calls["heavy"] = 1
        client._tier_input_chars["heavy"] = 400
        client._tier_output_chars["heavy"] = 80
        client._tier_served_by["heavy"] = "opencode/muse-spark-1.3-contributor-free"

        summary = client.get_usage_summary()

        assert "| Tier | Model |" in summary
        assert "`opencode/muse-spark-1.3-contributor-free`" in summary
        assert "100" in summary and "20" in summary
        assert "$0.0000" in summary
        assert "DeepSeek V4 Flash paid-tier rates" not in summary


# ---------------------------------------------------------------------------
# TestDefaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_models(self):
        client = OpencodeClient({})
        assert client.models["heavy"] == "opencode/muse-spark-1.3-contributor-free"
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
# One model per call
# ---------------------------------------------------------------------------

class TestServesExactlyOneModel:
    """This client runs the model it is handed and nothing else.

    Fallback across models is CompositeClient's job (see
    tests/test_composite_client.py). A backend that substituted its own model
    here would jump the chain's queue — possibly onto a paid rung the chain
    had deliberately placed last.
    """

    def test_runs_the_requested_model(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, SAMPLE_NDJSON)) as mock_run,
        ):
            client = OpencodeClient({})
            result = client.invoke("test", tier="heavy", model="opencode/muse-spark-1.3-contributor-free")
        assert result == "Hello there"
        cmd = mock_run.call_args_list[0][0][0]
        assert cmd[:2] == ["opencode", "run"]
        assert cmd[cmd.index("-m") + 1] == "opencode/muse-spark-1.3-contributor-free"
        assert client._tier_served_by["heavy"] == "opencode/muse-spark-1.3-contributor-free"

    def test_falls_back_to_the_tier_default_without_an_explicit_model(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, SAMPLE_NDJSON)) as mock_run,
        ):
            client = OpencodeClient({})
            client.invoke("test", tier="heavy")
        cmd = mock_run.call_args_list[0][0][0]
        assert cmd[cmd.index("-m") + 1] == "opencode/muse-spark-1.3-contributor-free"

    def test_never_substitutes_another_model_on_failure(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(1, "", "quota exceeded")) as mock_run,
        ):
            client = OpencodeClient({"max_retries_per_model": 0})
            assert client.invoke("test", tier="heavy", model="opencode/x") is None
        models = [
            c[0][0][c[0][0].index("-m") + 1] for c in mock_run.call_args_list
        ]
        assert set(models) == {"opencode/x"}

    def test_non_retryable_error_fails_the_rung_without_retrying(self):
        error_ndjson = (
            '{"type":"error","error":{"name":"APIError","data":'
            '{"message":"Insufficient balance","isRetryable":false}}}\n'
        )
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, error_ndjson)) as mock_run,
        ):
            client = OpencodeClient({"max_retries_per_model": 2})
            assert client.invoke("test", tier="heavy", model="opencode/x") is None
        assert mock_run.call_count == 1, "a dead endpoint must not burn retries"

    def test_unknown_error_fails_the_rung_without_retrying(self):
        error_ndjson = (
            '{"type":"error","error":{"name":"UnknownError","data":'
            '{"message":"Model not found: test"}}}\n'
        )
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, error_ndjson)) as mock_run,
        ):
            client = OpencodeClient({"max_retries_per_model": 2})
            assert client.invoke("test", tier="heavy", model="opencode/x") is None
        assert mock_run.call_count == 1

    def test_retryable_error_retries_the_same_model(self):
        error_ndjson = (
            '{"type":"error","error":{"name":"APIError","data":'
            '{"message":"rate limited","isRetryable":true}}}\n'
        )
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, error_ndjson)) as mock_run,
            patch("time.sleep"),
        ):
            client = OpencodeClient({"max_retries_per_model": 2})
            assert client.invoke("test", tier="heavy", model="opencode/x") is None
        assert mock_run.call_count == 3, "max_retries_per_model=2 means 3 attempts"
        models = [c[0][0][c[0][0].index("-m") + 1] for c in mock_run.call_args_list]
        assert set(models) == {"opencode/x"}, "retries must stay on the same model"

    def test_retryable_then_success_on_the_same_model(self):
        error_ndjson = (
            '{"type":"error","error":{"name":"APIError","data":'
            '{"message":"rate limited","isRetryable":true}}}\n'
        )
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=[
                make_mock_run(0, error_ndjson),
                make_mock_run(0, SAMPLE_NDJSON),
            ]) as mock_run,
            patch("time.sleep"),
        ):
            client = OpencodeClient({"max_retries_per_model": 2})
            assert client.invoke("test", tier="heavy", model="opencode/x") == "Hello there"
        assert mock_run.call_count == 2

    def test_timeout_retries_then_fails_the_rung(self):
        import subprocess as _sp

        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=_sp.TimeoutExpired("opencode", 1)) as mock_run,
            patch("time.sleep"),
        ):
            client = OpencodeClient({"max_retries_per_model": 1})
            assert client.invoke("test", tier="heavy", model="opencode/x") is None
        assert mock_run.call_count == 2

    def test_call_budget_exhaustion_returns_none(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", return_value=make_mock_run(0, SAMPLE_NDJSON)) as mock_run,
        ):
            client = OpencodeClient({"max_calls_per_run": 0})
            assert client.invoke("test", tier="heavy", model="opencode/x") is None
        assert mock_run.call_count == 0
