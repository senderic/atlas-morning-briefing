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


# ---------------------------------------------------------------------------
# Chain role: Codex as a rung of llm.chains
# ---------------------------------------------------------------------------


def make_chain_client(**config):
    values = {"enabled": True, "executable": CODEX}
    values.update(config)
    return CodexClient(values, role="chain")


def failed_jsonl(message):
    """The stream `codex exec` writes (to stdout, exit 1) when the API refuses."""
    return "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "t1"}),
            json.dumps({"type": "error", "message": message}),
            json.dumps({"type": "turn.failed", "error": {"message": message}}),
        ]
    ) + "\n"


UNSUPPORTED_MODEL = json.dumps(
    {
        "type": "error",
        "status": 400,
        "error": {"type": "invalid_request_error", "message": "model is not supported"},
    }
)


def model_and_effort(run_call):
    cmd = run_call.args[0]
    return cmd[cmd.index("-m") + 1], cmd[cmd.index("-c") + 1]


class TestChainModelSelection:
    def test_routing_prefix_is_stripped_before_the_cli(self):
        """`-m codex/gpt-5.6-terra` is not a model the CLI knows."""
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())) as run,
        ):
            client = make_chain_client()
            assert client.invoke("p", tier="medium", model="codex/gpt-5.6-terra") == "A useful report."

        cmd = run.call_args.args[0]
        assert cmd[cmd.index("-m") + 1] == "gpt-5.6-terra"
        assert not any("codex/" in part for part in cmd[1:])

    def test_serves_exactly_the_model_it_is_handed(self):
        """A backend that substituted its own model would jump the chain's queue."""
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())) as run,
        ):
            client = make_chain_client(model="gpt-5.6-sol")
            client.invoke("p", tier="light", model="codex/gpt-5.6-luna")
            client.invoke("p", tier="heavy", model="codex/gpt-5.6-luna")

        assert [model_and_effort(c)[0] for c in run.call_args_list] == [
            "gpt-5.6-luna", "gpt-5.6-luna",
        ]

    def test_a_failed_model_is_not_replaced_by_another(self):
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch(
                "scripts.codex_client.subprocess.run",
                return_value=completed(failed_jsonl(UNSUPPORTED_MODEL), returncode=1),
            ) as run,
        ):
            client = make_chain_client()
            assert client.invoke("p", tier="medium", model="codex/gpt-5.6-terra") is None

        assert {model_and_effort(c)[0] for c in run.call_args_list} == {"gpt-5.6-terra"}


class TestChainReasoningEffort:
    def _efforts(self, client, tiers, **kwargs):
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())) as run,
        ):
            for tier in tiers:
                client.invoke("p", tier=tier, model="codex/gpt-5.6-sol", **kwargs)
        return [model_and_effort(c)[1] for c in run.call_args_list]

    def test_effort_scales_with_the_tier_by_default(self):
        assert self._efforts(make_chain_client(), ["light", "medium", "heavy"]) == [
            'model_reasoning_effort="low"',
            'model_reasoning_effort="medium"',
            'model_reasoning_effort="high"',
        ]

    def test_writer_effort_does_not_leak_into_the_chain(self):
        """codex.reasoning_effort is the writer's; the chain has its own map."""
        client = make_chain_client(reasoning_effort="xhigh")
        assert self._efforts(client, ["light"]) == ['model_reasoning_effort="low"']

    def test_per_tier_effort_is_configurable(self):
        client = make_chain_client(chain={"reasoning_effort": {"light": "medium"}})
        assert self._efforts(client, ["light", "heavy"]) == [
            'model_reasoning_effort="medium"',
            'model_reasoning_effort="high"',
        ]

    def test_reasoning_disabled_drops_to_the_registered_floor(self):
        """Codex cannot switch reasoning off; the registry names the lowest effort."""
        efforts = self._efforts(make_chain_client(), ["heavy"], reasoning_enabled=False)
        assert efforts == ['model_reasoning_effort="low"']

    def test_writer_keeps_its_single_effort_whatever_the_tier(self):
        client = make_client(reasoning_effort="high")
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())) as run,
        ):
            client.invoke("p", tier="light")
            client.invoke("p", tier="medium", reasoning_enabled=False)
        assert [model_and_effort(c) for c in run.call_args_list] == [
            ("gpt-5.6-sol", 'model_reasoning_effort="high"'),
            ("gpt-5.6-sol", 'model_reasoning_effort="high"'),
        ]


class TestChainBudget:
    def test_chain_budget_is_separate_from_the_writer_budget(self):
        """The writer's 5 calls must not be the chain's ceiling, or vice versa."""
        config = {
            "enabled": True,
            "executable": CODEX,
            "max_calls_per_run": 1,
            "chain": {"max_calls_per_run": 3},
        }
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())) as run,
        ):
            writer = CodexClient(config)
            chain = CodexClient(config, role="chain")
            results = [chain.invoke("p", model="codex/gpt-5.6-luna") for _ in range(4)]
            assert results == ["A useful report."] * 3 + [None]
            # Draining the chain budget leaves the writer's untouched.
            assert writer.invoke("prose") == "A useful report."
            assert writer.invoke("prose") is None
        assert run.call_count == 4

    def test_defaults(self):
        chain = make_chain_client()
        assert chain.max_calls == 40
        assert make_client().max_calls == 5

    def test_writer_settings_are_untouched_by_a_chain_block(self):
        writer = make_client(
            timeout_seconds=300,
            max_calls_per_run=5,
            chain={"timeout_seconds": 60, "max_calls_per_run": 9, "max_retries": 4},
        )
        assert (writer.timeout, writer.max_calls, writer.max_retries) == (300, 5, 0)


class TestChainRetries:
    def test_transient_failure_is_retried_on_the_same_model(self):
        responses = [subprocess.TimeoutExpired([CODEX], 3), completed(success_jsonl())]
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", side_effect=responses) as run,
            patch("scripts.codex_client.time.sleep"),
        ):
            client = make_chain_client()
            assert client.invoke("p", model="codex/gpt-5.6-terra") == "A useful report."
        assert run.call_count == 2
        assert client.usage_stats["calls"] == 2
        assert client.usage_stats["failures"] == 1

    def test_retries_are_bounded(self):
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch(
                "scripts.codex_client.subprocess.run",
                side_effect=subprocess.TimeoutExpired([CODEX], 3),
            ) as run,
            patch("scripts.codex_client.time.sleep"),
        ):
            client = make_chain_client(chain={"max_retries": 1})
            assert client.invoke("p", model="codex/gpt-5.6-terra") is None
        assert run.call_count == 2

    def test_non_recoverable_error_fails_the_rung_without_a_retry(self):
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch(
                "scripts.codex_client.subprocess.run",
                return_value=completed(failed_jsonl(UNSUPPORTED_MODEL), returncode=1),
            ) as run,
            patch("scripts.codex_client.time.sleep") as sleep,
        ):
            assert make_chain_client().invoke("p", model="codex/nope") is None
        run.assert_called_once()
        sleep.assert_not_called()

    def test_writer_never_retries(self):
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch(
                "scripts.codex_client.subprocess.run",
                side_effect=subprocess.TimeoutExpired([CODEX], 3),
            ) as run,
        ):
            assert make_client().invoke("p") is None
        run.assert_called_once()

    def test_worst_case_fits_the_rung_window(self):
        """CompositeClient reads `_timeout` and `max_retries` for its startup check."""
        from scripts.composite_client import CompositeClient

        client = make_chain_client()
        assert CompositeClient._worst_case_seconds(client) == 360
        assert client.queue_timeout + 360 < 500


class TestChainOutage:
    def test_repeated_failures_take_the_transport_out_of_the_run(self):
        """A dead Codex must cost a few timeouts, not one per remaining call."""
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch(
                "scripts.codex_client.subprocess.run",
                return_value=completed("", returncode=1, stderr="boom"),
            ) as run,
        ):
            client = make_chain_client(chain={"max_consecutive_failures": 2})
            assert client.available is True
            assert client.invoke("p", model="codex/gpt-5.6-luna") is None
            assert client.available is True
            assert client.invoke("p", model="codex/gpt-5.6-luna") is None
            assert client.available is False
            assert client.invoke("p", model="codex/gpt-5.6-luna") is None
        assert run.call_count == 2

    def test_a_success_resets_the_failure_streak(self):
        responses = [
            completed("", returncode=1),
            completed(success_jsonl()),
            completed("", returncode=1),
        ]
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", side_effect=responses),
        ):
            client = make_chain_client(chain={"max_consecutive_failures": 2})
            for _ in responses:
                client.invoke("p", model="codex/gpt-5.6-luna")
            assert client.available is True

    def test_missing_binary_and_exhausted_budget_fail_quietly(self):
        with patch("scripts.codex_client.shutil.which", return_value=None):
            assert make_chain_client().invoke("p", model="codex/gpt-5.6-luna") is None
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())),
        ):
            client = make_chain_client(chain={"max_calls_per_run": 1})
            client.invoke("p", model="codex/gpt-5.6-luna")
            assert client.invoke("p", model="codex/gpt-5.6-luna") is None
            # Running out of budget is not an outage.
            assert client.available is True


class TestChainConcurrency:
    def test_parallel_calls_are_capped_and_counted_exactly(self, tmp_path):
        import threading
        import time as real_time

        log_path = tmp_path / "calls.jsonl"
        state = {"now": 0, "peak": 0}
        guard = threading.Lock()

        def fake_run(*args, **kwargs):
            with guard:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            real_time.sleep(0.05)
            with guard:
                state["now"] -= 1
            return completed(success_jsonl())

        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", side_effect=fake_run),
        ):
            client = make_chain_client(
                call_log_path=str(log_path),
                chain={"max_concurrent_requests": 3, "max_calls_per_run": 12},
            )
            assert client.available
            threads = [
                threading.Thread(
                    target=client.invoke, args=("p",),
                    kwargs={"tier": "light", "model": "codex/gpt-5.6-luna"},
                )
                for _ in range(12)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # The budget was 12 and 12 ran: a 13th must be refused.
            assert client.invoke("p", model="codex/gpt-5.6-luna") is None

        assert state["peak"] <= 3
        assert client.usage_stats["calls"] == 12
        assert client.usage_stats["input_tokens"] == 12 * 101
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(records) == 12

    def test_default_cap_is_three(self):
        assert make_chain_client().max_concurrent == 3

    def test_a_call_that_cannot_get_a_slot_gives_up_without_spending_budget(self):
        import threading

        release = threading.Event()
        started = threading.Event()

        def blocking_run(*args, **kwargs):
            started.set()
            release.wait(5)
            return completed(success_jsonl())

        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", side_effect=blocking_run) as run,
        ):
            client = make_chain_client(
                chain={"max_concurrent_requests": 1, "queue_timeout_seconds": 0.05}
            )
            holder = threading.Thread(
                target=client.invoke, args=("p",), kwargs={"model": "codex/gpt-5.6-luna"}
            )
            holder.start()
            assert started.wait(5)
            assert client.invoke("p", model="codex/gpt-5.6-luna") is None
            release.set()
            holder.join()

        assert run.call_count == 1
        assert client.usage_stats["calls"] == 1


class TestChainCallLog:
    def test_chain_and_writer_records_are_distinguishable(self, tmp_path):
        log_path = tmp_path / "calls.jsonl"
        config = {"enabled": True, "executable": CODEX, "call_log_path": str(log_path)}
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())),
        ):
            CodexClient(config).invoke("prose", tier="heavy")
            CodexClient(config, role="chain").invoke(
                "rank", tier="medium", model="codex/gpt-5.6-terra"
            )

        writer, chain = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert writer["caller"] == "writer"
        assert writer["model"] == "gpt-5.6-sol"
        assert chain["caller"] == "chain"
        assert chain["tier"] == "medium"
        assert chain["model"] == "gpt-5.6-terra"
        assert chain["reasoning_effort"] == "medium"

    def test_error_text_from_the_cli_is_never_logged(self, tmp_path):
        log_path = tmp_path / "calls.jsonl"
        hostile = json.dumps({"status": 400, "error": {"message": "secret-feed-text"}})
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch(
                "scripts.codex_client.subprocess.run",
                return_value=completed(failed_jsonl(hostile), returncode=1),
            ),
        ):
            client = make_chain_client(call_log_path=str(log_path))
            assert client.invoke("p", model="codex/gpt-5.6-luna") is None

        record = json.loads(log_path.read_text().strip())
        assert record["error_category"] == "nonzero_exit"
        assert record["http_status"] == 400
        assert "secret-feed-text" not in log_path.read_text()


class TestPerModelPricing:
    def _served(self, client, *models):
        usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0}
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch(
                "scripts.codex_client.subprocess.run",
                return_value=completed(success_jsonl(**usage)),
            ),
        ):
            for model in models:
                client.invoke("p", model=model)
        return client.get_usage_summary()

    def test_unpriced_model_is_reported_as_unknown_not_at_sol_rates(self):
        summary = self._served(make_chain_client(), "codex/gpt-5.6-sol", "codex/gpt-5.6-terra")
        rows = {line.split("|")[1].strip(): line for line in summary.splitlines() if line.startswith("| `")}
        assert "$2.000000" in rows["`gpt-5.6-sol`"]
        assert "unknown" in rows["`gpt-5.6-terra`"]
        assert "$" not in rows["`gpt-5.6-terra`"]

    def test_rates_come_from_config_per_model(self):
        client = make_chain_client(
            pricing={"gpt-5.6-terra": {"input_per_million": 0.5, "output_per_million": 3.0}}
        )
        assert "$0.500000" in self._served(client, "codex/gpt-5.6-terra")

    def test_legacy_flat_rates_apply_to_the_configured_model_only(self):
        client = make_chain_client(
            model="gpt-5.6-sol",
            pricing={"input_per_million": 4.0, "cached_input_per_million": 0.4, "output_per_million": 9.0},
        )
        summary = self._served(client, "codex/gpt-5.6-sol", "codex/gpt-5.6-luna")
        assert "$4.000000" in summary
        assert summary.count("unknown") >= 1

    def test_a_model_can_be_explicitly_unpriced(self):
        client = make_chain_client(pricing={"gpt-5.6-sol": None})
        assert "$" not in self._served(client, "codex/gpt-5.6-sol").split("|\n")[-2]


class TestChainUsageSummary:
    def test_silent_when_the_chain_made_no_codex_calls(self):
        assert make_chain_client().get_usage_summary() == ""

    def test_is_not_presented_as_a_charge(self):
        client = make_chain_client()
        with (
            patch("scripts.codex_client.shutil.which", return_value=CODEX),
            patch("scripts.codex_client.subprocess.run", return_value=completed(success_jsonl())),
        ):
            client.invoke("p", tier="light", model="codex/gpt-5.6-luna")
        summary = client.get_usage_summary()
        assert "## Codex Chain Usage Summary" in summary
        assert "subscription" in summary
        assert "API-equivalent" in summary
        assert "billed" not in summary.lower().replace("not billed", "")
        assert "⚠" not in summary

    def test_heading_is_recognised_as_a_usage_appendix(self):
        """report_invariants skips degraded-content checks under these headings."""
        from scripts.report_invariants import _is_usage_appendix_heading

        assert _is_usage_appendix_heading("Codex Chain Usage Summary")
