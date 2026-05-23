"""Tests for BaseWorker shared client + token-counting helpers."""

from unittest.mock import MagicMock

import pytest

from scripts.workers.base_worker import BaseWorker


class _DummyWorker(BaseWorker):
    """Concrete subclass so we can instantiate BaseWorker for testing."""

    def execute(self):
        return self._create_finding(status="success", items=[])


@pytest.fixture
def stub_llm():
    llm = MagicMock()
    llm.usage_stats = {
        "heavy": {"in_tokens": 0, "out_tokens": 0},
        "medium": {"in_tokens": 0, "out_tokens": 0},
        "light": {"in_tokens": 0, "out_tokens": 0},
    }
    return llm


def test_get_llm_client_returns_injected(stub_llm):
    worker = _DummyWorker({}, "dummy", llm_client=stub_llm)
    assert worker._get_llm_client() is stub_llm


def test_get_llm_client_falls_back_to_gemini_when_enabled(monkeypatch):
    config = {"gemini": {"enabled": True}}
    worker = _DummyWorker(config, "dummy")  # no llm_client injected

    sentinel = object()

    class FakeGemini:
        def __init__(self, gemini_config):
            assert gemini_config == {"enabled": True}
            self.tag = sentinel

    import scripts.gemini_client as gc
    monkeypatch.setattr(gc, "GeminiCLIClient", FakeGemini)
    client = worker._get_llm_client()
    assert client.tag is sentinel


def test_count_client_tokens_sums_in_and_out(stub_llm):
    stub_llm.usage_stats["heavy"]["in_tokens"] = 100
    stub_llm.usage_stats["heavy"]["out_tokens"] = 50
    stub_llm.usage_stats["medium"]["in_tokens"] = 200
    stub_llm.usage_stats["medium"]["out_tokens"] = 75
    assert BaseWorker._count_client_tokens(stub_llm) == 100 + 50 + 200 + 75


def test_count_client_tokens_returns_zero_for_client_without_stats():
    bare = MagicMock(spec=[])  # no usage_stats attribute
    assert BaseWorker._count_client_tokens(bare) == 0


def test_count_client_tokens_returns_zero_when_stats_not_dict():
    bad = MagicMock()
    bad.usage_stats = "not a dict"
    assert BaseWorker._count_client_tokens(bad) == 0


def test_create_finding_records_processing_time(stub_llm):
    worker = _DummyWorker({}, "dummy", llm_client=stub_llm)
    worker._start_timing()
    finding = worker._create_finding(
        status="success",
        items=[{"a": 1}, {"b": 2}],
        synthesis="ok",
        token_count=42,
        items_found=10,
    )
    assert finding["worker"] == "dummy"
    assert finding["status"] == "success"
    assert finding["items"] == [{"a": 1}, {"b": 2}]
    assert finding["synthesis"] == "ok"
    assert finding["metadata"]["token_count"] == 42
    assert finding["metadata"]["items_found"] == 10
    assert finding["metadata"]["items_kept"] == 2
    assert finding["metadata"]["processing_time"] >= 0


def test_create_finding_defaults_items_found_to_kept_count(stub_llm):
    worker = _DummyWorker({}, "dummy", llm_client=stub_llm)
    worker._start_timing()
    finding = worker._create_finding(status="success", items=[1, 2, 3])
    # When items_found not passed, falls back to len(items).
    assert finding["metadata"]["items_found"] == 3
    assert finding["metadata"]["items_kept"] == 3


def test_create_finding_error_path(stub_llm):
    worker = _DummyWorker({}, "dummy", llm_client=stub_llm)
    worker._start_timing()
    finding = worker._create_finding(
        status="error", items=[], error="boom"
    )
    assert finding["status"] == "error"
    assert finding["error"] == "boom"
    assert finding["items"] == []
