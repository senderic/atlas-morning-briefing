"""Focused tests for the standalone Codex CLI client."""

import json
import subprocess
from unittest.mock import MagicMock, patch

from scripts.codex_client import CodexClient


CODEX = "/opt/codex/bin/codex"


def completed(stdout, returncode=0, stderr=""):
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


def success_jsonl(message="A useful report.", **usage):
    usage = usage or {
        "input_tokens": 101,
        "cached_input_tokens": 7,
        "output_tokens": 23,
    }
    return "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "t1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": message},
                }
            ),
            json.dumps({"type": "turn.completed", "usage": usage}),
        ]
    ) + "\n"


def make_client(**config):
    values = {"enabled": True, "executable": CODEX}
    values.update(config)
    return CodexClient(values)


class TestAvailability:
    def test_enabled_and_configured_executable_on_path(self):
        with patch("scripts.codex_client.shutil.which", return_value=CODEX):
            client = make_client()
            assert client.available is True

    def test_disabled_does_not_check_path(self):
        with patch("scripts.codex_client.shutil.which") as which:
            client = make_client(enabled=False)
            assert client.available is False
            which.assert_not_called()

    def test_missing_executable_is_unavailable_without_auth_inspection(self):
        with patch("scripts.codex_client.shutil.which", return_value=None) as which:
            client = make_client()
            assert client.available is False
            which.assert_called_once_with(CODEX)


class TestInvocation:
    def test_approved_config_keys_control_reasoning_and_timeout(self):
        response = completed(success_jsonl())
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=response) as run,
        ):
            client = make_client(
                reasoning_effort="low",
                timeout_seconds=17,
                reasoning="high",
                timeout=99,
            )
            assert client.invoke("request") == "A useful report."

        assert 'model_reasoning_effort="low"' in run.call_args.args[0]
        assert 0 < run.call_args.kwargs["timeout"] <= 17

    def test_builds_exact_command_and_delimited_stdin(self):
        response = completed(success_jsonl())
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=response) as run,
        ):
            client = make_client(model="my-model", reasoning="medium", timeout=42)
            assert client.invoke("user request", system_prompt="system rules") == "A useful report."

        assert run.call_args.args[0] == [
            CODEX,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "-C",
            "/tmp",
            "-m",
            "my-model",
            "-c",
            'model_reasoning_effort="medium"',
            "--json",
            "-",
        ]
        kwargs = run.call_args.kwargs
        assert kwargs["input"].startswith("--- SYSTEM PROMPT ---")
        assert "system rules" in kwargs["input"]
        assert "--- USER PROMPT ---" in kwargs["input"]
        assert "user request" in kwargs["input"]
        assert 0 < kwargs["timeout"] <= 42
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True

    def test_success_parses_last_message_and_usage(self):
        raw = "\n".join(
            [
                json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "first"}}),
                "not json",
                json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "last"}}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 4, "reasoning_output_tokens": 6}}),
            ]
        )
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(raw)),
        ):
            client = make_client()
            assert client.invoke("request") == "last"
            assert client.usage_stats["calls"] == 1
            assert client.usage_stats["failures"] == 0
            assert client.usage_stats["input_tokens"] == 10
            assert client.usage_stats["cached_input_tokens"] == 2
            assert client.usage_stats["output_tokens"] == 4
            assert client.usage_stats["reasoning_output_tokens"] == 6
            assert client.usage_stats["model"] == "gpt-5.6-sol"

    def test_malformed_lines_are_ignored_when_completion_is_valid(self):
        raw = "bad\n" + success_jsonl()
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(raw)),
        ):
            assert make_client().invoke("request") == "A useful report."

    def test_post_terminal_agent_message_is_rejected_not_selected(self):
        """Catches an incomplete later turn replacing the completed prose."""
        raw = success_jsonl("completed prose") + json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "later partial prose"},
            }
        )
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(raw)),
        ):
            assert make_client().invoke("request") is None

    def test_incomplete_turn_is_rejected(self):
        raw = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "partial"}})
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(raw)),
        ):
            client = make_client()
            assert client.invoke("request") is None
            assert client.usage_stats["failures"] == 1

    def test_top_level_agent_message_is_not_a_completed_agent_message(self):
        raw = "\n".join(
            [
                json.dumps({"type": "agent_message", "text": "partial"}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}),
            ]
        )
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(raw)),
        ):
            assert make_client().invoke("request") is None

    def test_explicit_error_event_is_rejected_even_with_message(self):
        raw = success_jsonl() + json.dumps({"type": "turn.failed", "error": {"message": "failed"}}) + "\n"
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(raw)),
        ):
            assert make_client().invoke("request") is None

    def test_nonzero_exit_is_rejected(self):
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl(), returncode=1, stderr="bad")),
        ):
            assert make_client().invoke("request") is None

    def test_timeout_and_os_error_are_rejected(self):
        for error in (subprocess.TimeoutExpired([CODEX], 3), OSError("not executable")):
            with (
                patch("scripts.codex_client.shutil.which", return_value=CODEX),
                patch("scripts.codex_client.subprocess.run", side_effect=error),
            ):
                client = make_client()
                assert client.invoke("request") is None
                assert client.usage_stats["failures"] == 1

    def test_call_budget_blocks_calls_after_limit(self):
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())) as run,
        ):
            client = make_client(max_calls_per_run=1)
            assert client.invoke("one") == "A useful report."
            assert client.invoke("two") is None
            run.assert_called_once()

    def test_each_call_gets_its_timeout_even_with_a_stale_wrapper_deadline(self, monkeypatch):
        """Catches pipeline wall time preventing otherwise healthy Codex calls."""
        monkeypatch.setenv("ATLAS_CODEX_DEADLINE_EPOCH", "1")
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch(
                "scripts.codex_client.subprocess.run",
                return_value=completed(success_jsonl()),
            ) as run,
        ):
            client = make_client(timeout_seconds=10)
            assert client.invoke("first") == "A useful report."
            assert client.invoke("second") == "A useful report."

        assert [call.kwargs["timeout"] for call in run.call_args_list] == [10, 10]


class TestCallLog:
    def test_logs_one_record_per_attempt_without_prompt_contents(self, tmp_path):
        log_path = tmp_path / "calls.jsonl"
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())),
        ):
            client = make_client(call_log_path=str(log_path))
            assert client.invoke("secret user prompt", system_prompt="secret system") == "A useful report."

        record = json.loads(log_path.read_text().strip())
        assert record["status"] == "success"
        assert record["model"] == "gpt-5.6-sol"
        assert record["input_tokens"] == 101
        assert record["output_tokens"] == 23
        assert "secret" not in log_path.read_text()

    def test_nonzero_stderr_is_sanitized_in_call_log(self, tmp_path):
        log_path = tmp_path / "calls.jsonl"
        hostile = "secret user prompt / auth-token=do-not-log"
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch(
                "scripts.codex_client.subprocess.run",
                return_value=completed("", returncode=7, stderr=hostile),
            ),
        ):
            assert make_client(call_log_path=str(log_path)).invoke("request") is None

        record = json.loads(log_path.read_text().strip())
        assert record["status"] == "error"
        assert record["error_category"] == "nonzero_exit"
        assert record["exit_status"] == 7
        assert hostile not in log_path.read_text()

    def test_logging_failure_does_not_fail_inference(self, tmp_path):
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())),
            patch("scripts.codex_client.Path.mkdir", side_effect=OSError("read-only")),
        ):
            assert make_client(call_log_path=str(tmp_path / "calls.jsonl")).invoke("request") == "A useful report."


def test_usage_summary_is_concise_markdown():
    client = make_client(
        pricing={
            "input_per_million": 2.0,
            "cached_input_per_million": 0.2,
            "output_per_million": 10.0,
        }
    )
    client.usage_stats.update(
        calls=2,
        failures=1,
        input_tokens=100,
        cached_input_tokens=10,
        output_tokens=25,
        reasoning_output_tokens=12,
        latency_s=3.5,
    )
    summary = client.get_usage_summary()
    assert "Codex" in summary
    assert "## Codex Usage Summary" in summary
    assert "2 calls" in summary
    assert "1" in summary
    assert "100" in summary and "25" in summary
    assert "25 / 12" in summary
    # (90 fresh * $2 + 10 cached * $0.20 + 25 output * $10) / 1M
    assert "$0.000432" in summary
    assert "API-equivalent" in summary


def test_usage_summary_uses_singular_call_label():
    client = make_client()
    client.usage_stats["calls"] = 1

    assert "**1 call**," in client.get_usage_summary()
