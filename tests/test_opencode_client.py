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
            client = OpencodeClient({})
            result = client.invoke("test")
            assert result is None
            assert client._call_count == 0

    def test_timeout(self):
        with (
            patch("shutil.which", return_value="/usr/bin/opencode"),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["opencode"], timeout=300)),
        ):
            client = OpencodeClient({})
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
            assert cmd[model_idx] == "opencode/deepseek-v4-flash-free"

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
        assert OpencodeClient._parse_ndjson_response("") == ""

    def test_non_json_lines_skipped(self):
        raw = "not json\n{\"type\":\"text\",\"part\":{\"text\":\"hi\"}}\n"
        assert OpencodeClient._parse_ndjson_response(raw) == "hi"

    def test_no_text_type(self):
        raw = '{"type":"step_start"}\n{"type":"step_finish"}\n'
        assert OpencodeClient._parse_ndjson_response(raw) == ""

    def test_missing_part_field(self):
        raw = '{"type":"text"}'
        assert OpencodeClient._parse_ndjson_response(raw) == ""

    def test_missing_text_in_part(self):
        raw = '{"type":"text","part":{"type":"text"}}'
        assert OpencodeClient._parse_ndjson_response(raw) == ""

    def test_multiple_mixed_types(self):
        raw = (
            '{"type":"step_start"}\n'
            '{"type":"text","part":{"type":"text","text":"A"}}\n'
            'invalid json line\n'
            '{"type":"text","part":{"type":"text","text":"B"}}\n'
        )
        assert OpencodeClient._parse_ndjson_response(raw) == "AB"


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
        assert client.models["heavy"] == "opencode/deepseek-v4-flash-free"
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
