"""Tests for the OpenRouterClient (OpenAI-compatible API fallback backend)."""

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
        assert c._tier_failures["light"] == 2  # initial + 1 retry (MAX_RETRIES_PER_MODEL=2)

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
