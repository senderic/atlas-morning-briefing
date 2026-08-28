"""Tests for the OpenRouterClient (OpenAI-compatible API fallback backend)."""

import time
import os
from unittest.mock import Mock, patch

import pytest

from scripts.openrouter_client import (
    OpenRouterClient,
    API_BASE_URL,
    DEFAULT_MODELS,
)


class _Resp:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data
        self.text = "err" if status_code != 200 else ""

    def json(self):
        return self._json


def _ok_response(content="Hello"):
    return _Resp(200, {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


@pytest.fixture
def key_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


class TestOpenRouterClient:
    def test_default_models(self):
        c = OpenRouterClient({})
        assert c.models["heavy"] == DEFAULT_MODELS["heavy"]
        assert c.models["medium"] == DEFAULT_MODELS["medium"]
        assert c.models["light"] == DEFAULT_MODELS["light"]

    def test_custom_model_override(self):
        c = OpenRouterClient({"models": {"heavy": "other/model"}})
        assert c.models["heavy"] == "other/model"

    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        c = OpenRouterClient({"enabled": True})
        assert c.available is False

    def test_available_with_env_key(self, key_env):
        c = OpenRouterClient({"enabled": True})
        assert c.available is True

    def test_uses_openai_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
        c = OpenRouterClient({"enabled": True})
        assert c.api_key == "sk-openai-key"
        assert c.available is True

    def test_disabled(self, key_env):
        c = OpenRouterClient({"enabled": False})
        assert c.available is False

    @patch("scripts.openrouter_client.requests.post")
    def test_successful_invoke(self, mock_post, key_env):
        mock_post.return_value = _ok_response("Hello there")
        c = OpenRouterClient({"enabled": True})
        result = c.invoke("hi", tier="light")
        assert result == "Hello there"
        assert c._call_count == 1
        assert c._tier_calls["light"] == 1
        assert c._tier_failures["light"] == 0
        args, kwargs = mock_post.call_args
        assert args[0] == API_BASE_URL
        assert "Bearer test-key" in kwargs["headers"]["Authorization"]

    @patch("scripts.openrouter_client.requests.post")
    @patch("scripts.openrouter_client.time.sleep")
    def test_http_error_fallback(self, mock_sleep, mock_post, key_env):
        # 429 is transient → retried (bounded), then all models exhausted → None.
        mock_post.return_value = _Resp(429, {})
        c = OpenRouterClient({"enabled": True, "fallback_models": {"light": []}})
        result = c.invoke("hi", tier="light")
        assert result is None
        assert c._tier_failures["light"] == 2  # initial + 1 retry (default max_retries_per_model=1)

    @patch("scripts.openrouter_client.requests.post")
    def test_out_of_usage_skips_straight_to_next_model(self, mock_post, key_env):
        # 402 = payment required → out of usage → no retry, straight to next model.
        mock_post.side_effect = [
            _Resp(402, {"error": {"message": "insufficient balance"}}),
            _ok_response("from-fallback"),
        ]
        c = OpenRouterClient({
            "enabled": True,
            "models": {"light": "primary/model"},
            "fallback_models": {"light": ["fallback/model"]},
        })
        result = c.invoke("hi", tier="light")
        assert result == "from-fallback"
        # primary (1) + fallback (1) — no retries in between
        assert mock_post.call_count == 2

    @patch("scripts.openrouter_client.requests.post")
    @patch("scripts.openrouter_client.time.sleep")
    def test_primary_fails_fallback_succeeds(self, mock_sleep, mock_post, key_env):
        mock_post.side_effect = [
            _Resp(500, {}),
            _ok_response("from-fallback"),
        ]
        c = OpenRouterClient({
            "enabled": True,
            "models": {"light": "primary/model"},
            "fallback_models": {"light": ["fallback/model"]},
        })
        result = c.invoke("hi", tier="light")
        assert result == "from-fallback"
        assert mock_post.call_count == 2

    @patch("scripts.openrouter_client.requests.post")
    def test_call_budget(self, mock_post, key_env):
        c = OpenRouterClient({"enabled": True, "max_calls_per_run": 1})
        c._call_count = 1
        result = c.invoke("hi")
        assert result is None
        mock_post.assert_not_called()

    @patch("scripts.openrouter_client.requests.post")
    def test_invoke_passes_system_prompt(self, mock_post, key_env):
        mock_post.return_value = _ok_response()
        c = OpenRouterClient({"enabled": True})
        c.invoke("hi", system_prompt="SYS")
        messages = mock_post.call_args.kwargs["json"]["messages"]
        assert messages[0] == {"role": "system", "content": "SYS"}
        assert messages[1]["content"] == "hi"

    def test_usage_summary_empty_when_no_activity(self, key_env):
        c = OpenRouterClient({"enabled": True})
        assert c.get_usage_summary() == ""


class TestFreeTierRegressions:
    """Regressions for the free-model migration.

    Each test pins a behaviour that was verified against the live OpenRouter
    API on 2026-08-27, where the previous implementation got it wrong.
    """

    def test_tiers_are_distinct_models(self):
        """heavy/medium/light must never collapse onto the same model."""
        c = OpenRouterClient({})
        assert len({c.models["heavy"], c.models["medium"], c.models["light"]}) == 3

    def test_every_default_model_is_free(self):
        """The whole point of the roster: no slug may route to a paid model."""
        c = OpenRouterClient({})
        for tier in ("heavy", "medium", "light"):
            for model in [c.models[tier]] + c.fallback_models[tier]:
                assert model.endswith(":free"), f"{tier} chain has non-free {model}"
                assert "openrouter/auto" not in model

    def test_openrouter_free_does_not_become_paid_auto(self):
        """`openrouter/free` is a real free model; `openrouter/auto` is billed.

        The previous code rewrote one into the other, silently spending money.
        """
        assert OpenRouterClient._to_api_model("openrouter/openrouter/free") == "openrouter/free"

    def test_only_one_routing_prefix_is_stripped(self):
        assert (
            OpenRouterClient._to_api_model("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free")
            == "nvidia/nemotron-3-ultra-550b-a55b:free"
        )

    def test_reasoning_suppression_uses_the_parameter_that_works(self, key_env):
        """`reasoning_effort` is accepted but is a no-op; `reasoning` is not."""
        c = OpenRouterClient({})
        payload, _, sent = c._build_payload(
            c.models["heavy"], "hi", None, reasoning_enabled=False
        )
        assert sent
        assert payload.get("reasoning") == {"enabled": False}
        assert "reasoning_effort" not in payload

    def test_no_reasoning_param_when_reasoning_enabled(self, key_env):
        c = OpenRouterClient({})
        payload, _, sent = c._build_payload(
            c.models["heavy"], "hi", None, reasoning_enabled=True
        )
        assert not sent
        assert "reasoning" not in payload

    def test_empty_content_with_finish_length_is_reasoning_overflow(self, key_env):
        """200 + empty content because reasoning ate max_tokens is recoverable."""
        c = OpenRouterClient({})
        resp = _Resp(200, {
            "choices": [{"finish_reason": "length",
                         "message": {"content": "", "reasoning": "x" * 500}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4096},
        })
        with patch("scripts.openrouter_client.requests.post", return_value=resp):
            result, action = c._single_call(
                c.models["heavy"], "hi", "heavy", None, reasoning_enabled=True
            )
        assert result is None
        assert action == "reasoning_overflow"

    def test_reasoning_overflow_retries_same_model_with_reasoning_off(self, key_env):
        """The retry must stay on the same model rather than burning the chain."""
        c = OpenRouterClient({})
        overflow = _Resp(200, {
            "choices": [{"finish_reason": "length",
                         "message": {"content": "", "reasoning": "x" * 500}}],
            "usage": {},
        })
        with patch("scripts.openrouter_client.requests.post",
                   side_effect=[overflow, _ok_response("recovered")]) as post:
            result = c.invoke("hi", tier="heavy")
        assert result == "recovered"
        assert post.call_count == 2
        first, second = (call.kwargs["json"] for call in post.call_args_list)
        assert first["model"] == second["model"], "retry must stay on the same model"
        assert "reasoning" not in first
        assert second["reasoning"] == {"enabled": False}

    def test_reasoning_mandatory_400_resends_without_the_parameter(self, key_env):
        """Some endpoints refuse suppression; drop the param, keep the model."""
        c = OpenRouterClient({})
        refusal = _Resp(400, {})
        refusal.text = "Reasoning is mandatory for this endpoint and cannot be disabled."
        with patch("scripts.openrouter_client.requests.post",
                   side_effect=[refusal, _ok_response("fine")]) as post:
            result, action = c._single_call(
                c.models["heavy"], "hi", "heavy", None, reasoning_enabled=False
            )
        assert result == "fine"
        assert post.call_count == 2
        first, second = (call.kwargs["json"] for call in post.call_args_list)
        assert first["model"] == second["model"]
        assert "reasoning" not in second

    def test_error_nested_in_a_200_body_is_classified(self, key_env):
        """OpenRouter returns upstream 502s inside an HTTP 200 envelope."""
        c = OpenRouterClient({})
        resp = _Resp(200, {"error": {"message": "Upstream error: Service temporarily "
                                                "overloaded", "code": 502}})
        with patch("scripts.openrouter_client.requests.post", return_value=resp):
            result, action = c._single_call(
                c.models["heavy"], "hi", "heavy", None, reasoning_enabled=True
            )
        assert result is None
        assert action == "retry"

    def test_model_swap_capability_does_not_raise(self, key_env):
        """apply_reasoning_control may return a str; that used to be a TypeError."""
        c = OpenRouterClient({})
        with patch.object(c, "apply_reasoning_control", return_value="openrouter/other:free"):
            payload, model, _ = c._build_payload("m", "hi", None, reasoning_enabled=False)
        assert model == "openrouter/other:free"
        assert payload["model"] == "other:free"

    def test_call_budget_is_atomic_across_threads(self):
        """max_workers>1 must not be able to over-run the per-run call budget."""
        from concurrent.futures import ThreadPoolExecutor

        c = OpenRouterClient({"max_calls_per_run": 5})
        with ThreadPoolExecutor(20) as ex:
            granted = list(ex.map(lambda _: c._reserve_call(), range(50)))
        assert sum(granted) == 5
        assert c._call_count == 5

    def test_concurrency_is_capped_by_semaphore(self, key_env):
        """Parallel enrichment must not stampede the free tier."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        c = OpenRouterClient({"max_concurrent_requests": 3})
        in_flight = 0
        peak = 0
        lock = threading.Lock()

        def fake_post(*args, **kwargs):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            time.sleep(0.05)
            with lock:
                in_flight -= 1
            return _ok_response("ok")

        with patch("scripts.openrouter_client.requests.post", side_effect=fake_post):
            with ThreadPoolExecutor(12) as ex:
                results = list(ex.map(lambda i: c.invoke("hi", tier="medium"), range(12)))
        assert all(results)
        assert peak <= 3, f"peak in-flight {peak} exceeded the cap"

    def test_tier_counters_survive_concurrent_calls(self, key_env):
        from concurrent.futures import ThreadPoolExecutor

        c = OpenRouterClient({})
        with patch("scripts.openrouter_client.requests.post", return_value=_ok_response()):
            with ThreadPoolExecutor(8) as ex:
                list(ex.map(lambda i: c.invoke("hi", tier="light"), range(40)))
        assert c._call_count == sum(c._tier_calls.values()) == 40

    def test_preflight_may_not_move_a_model_between_tiers(self):
        """A preflight record naming another tier must be ignored, not applied."""
        c = OpenRouterClient({}, preflight_models={
            "heavy": {"available": True, "tier": "light",
                      "model": "openrouter/tiny:free"},
        })
        assert c.models["heavy"] == DEFAULT_MODELS["heavy"]

    def test_preflight_override_applies_within_its_own_tier(self):
        c = OpenRouterClient({}, preflight_models={
            "heavy": {"available": True, "tier": "heavy",
                      "model": "openrouter/dots-studio/dots-3-note-preview:free"},
        })
        assert c.models["heavy"] == "openrouter/dots-studio/dots-3-note-preview:free"
        # The configured primary stays in the chain so a preflight blip is recoverable.
        assert DEFAULT_MODELS["heavy"] in c.fallback_models["heavy"]

    def test_unavailable_preflight_keeps_the_configured_model(self):
        c = OpenRouterClient({}, preflight_models={
            "heavy": {"available": False, "tier": "heavy", "model": "openrouter/dead:free"},
        })
        assert c.models["heavy"] == DEFAULT_MODELS["heavy"]

    def test_usage_summary_flags_a_billed_run(self, key_env):
        c = OpenRouterClient({})
        c._tier_calls["heavy"] = 1
        c._billed_cost = 0.0123
        summary = c.get_usage_summary()
        assert "OpenRouter billed $0.012300" in summary

    def test_usage_summary_reports_zero_for_a_free_run(self, key_env):
        c = OpenRouterClient({})
        c._tier_calls["heavy"] = 1
        assert "billed **$0.00**" in c.get_usage_summary()

    def test_retry_semantics_match_opencode(self, key_env):
        """max_retries_per_model = N means N+1 attempts, as in OpencodeClient."""
        with patch("scripts.openrouter_client.requests.post",
                   return_value=_Resp(429, {})) as post, \
             patch("scripts.openrouter_client.time.sleep"):
            c = OpenRouterClient({"max_retries_per_model": 2,
                                  "fallback_models": {"light": []}})
            c.invoke("hi", tier="light")
        assert post.call_count == 3

    def test_max_retries_is_exposed_for_the_composite_budget_guard(self):
        """CompositeClient reads .max_retries to size this backend's window."""
        from scripts.composite_client import CompositeClient

        c = OpenRouterClient({"timeout": 75, "max_retries_per_model": 1})
        chain = 1 + len(c.fallback_models["heavy"])
        assert CompositeClient._worst_case_seconds(c) == 75 * 2 * chain
