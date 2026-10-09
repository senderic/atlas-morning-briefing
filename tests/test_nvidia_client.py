"""Behavior tests for the direct NVIDIA NIM transport."""

from unittest.mock import patch

from scripts.nvidia_client import NvidiaClient


def test_super_reasoning_is_bounded_and_can_be_disabled(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = NvidiaClient({"enabled": True, "reasoning_budget": 512})
    model = "nvidia-direct/nvidia/nemotron-3-super-120b-a12b"
    payload, _, _ = client._build_payload(model, "rank these stories", None, True)
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True, "low_effort": True, "reasoning_budget": 512,
    }
    payload, _, sent = client._build_payload(model, "rank these stories", None, False)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent is True


def test_omni_reasoning_retry_sends_the_actual_switch(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = NvidiaClient({"enabled": True})
    payload, _, sent = client._build_payload(
        "nvidia-direct/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "hello", None, False,
    )
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent is True


def test_omni_reasoning_reserves_tokens_for_the_answer(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = NvidiaClient({"enabled": True, "reasoning_budget": 512})
    payload, _, _ = client._build_payload(
        "nvidia-direct/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "hello", None, True,
    )
    assert payload["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_budget": 512}
    # Hosted NIM rejects this self-hosted vLLM-only parameter with HTTP 400.
    assert "thinking_token_budget" not in payload


def test_direct_prefix_is_removed_from_the_api_model(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = NvidiaClient({"enabled": True})

    payload, model, _ = client._build_payload(
        "nvidia-direct/nvidia/nemotron-3-super-120b-a12b",
        "hello",
        None,
        reasoning_enabled=True,
    )

    assert model == "nvidia-direct/nvidia/nemotron-3-super-120b-a12b"
    assert payload["model"] == "nvidia/nemotron-3-super-120b-a12b"


def test_nvidia_key_controls_availability(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with patch("scripts.nvidia_client.HAS_REQUESTS", True):
        assert NvidiaClient({"enabled": True}).available is False

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    with patch("scripts.nvidia_client.HAS_REQUESTS", True):
        assert NvidiaClient({"enabled": True}).available is True


def test_usage_summary_is_labeled_nvidia_nim(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = NvidiaClient({"enabled": True})
    client._tier_calls["medium"] = 1

    summary = client.get_usage_summary()

    assert "## NVIDIA NIM Usage Summary" in summary
    assert "OpenRouter Usage Summary" not in summary
