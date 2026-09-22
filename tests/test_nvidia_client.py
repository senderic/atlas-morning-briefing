"""Behavior tests for the direct NVIDIA NIM transport."""

from unittest.mock import patch

from scripts.nvidia_client import NvidiaClient


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
