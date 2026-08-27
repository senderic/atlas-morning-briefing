"""Tests for the CompositeClient (multi-backend fallback chain)."""

import pytest

from scripts.composite_client import CompositeClient


class _FakeClient:
    """Minimal fake implementing BaseLLMClient's interface for chaining tests."""

    def __init__(self, name, available=True, result=None):
        self.name = name
        self._available = available
        self.result = result
        self.calls = []
        self.usage = f"usage-{name}"

    @property
    def available(self):
        return self._available

    def invoke(self, prompt, tier="medium", system_prompt=None, **kwargs):
        self.calls.append((prompt, tier, system_prompt))
        return self.result

    def get_usage_summary(self, start_time=None, end_time=None):
        return self.usage


class TestCompositeClient:
    def test_requires_at_least_one_client(self):
        with pytest.raises(ValueError):
            CompositeClient([])

    def test_available_true_if_any_available(self):
        c = CompositeClient([_FakeClient("a", available=False), _FakeClient("b", available=True)])
        assert c.available is True

    def test_available_false_if_none_available(self):
        c = CompositeClient([_FakeClient("a", available=False)])
        assert c.available is False

    def test_returns_first_success(self):
        first = _FakeClient("first", result="from-first")
        second = _FakeClient("second", result="from-second")
        c = CompositeClient([first, second])
        assert c.invoke("hi") == "from-first"
        assert len(second.calls) == 0

    def test_falls_back_to_second_when_first_returns_none(self):
        first = _FakeClient("first", result=None)
        second = _FakeClient("second", result="from-second")
        c = CompositeClient([first, second])
        assert c.invoke("hi", tier="light") == "from-second"
        assert first.calls == [("hi", "light", None)]
        assert second.calls == [("hi", "light", None)]

    def test_skips_unavailable_client(self):
        first = _FakeClient("first", available=False, result="should-not-run")
        second = _FakeClient("second", result="from-second")
        c = CompositeClient([first, second])
        assert c.invoke("hi") == "from-second"
        assert first.calls == []  # unavailable client never invoked

    def test_invoke_passes_system_prompt(self):
        second = _FakeClient("second", result="ok")
        c = CompositeClient([_FakeClient("first", result=None), second])
        c.invoke("hi", tier="heavy", system_prompt="SYS")
        assert second.calls[0][2] == "SYS"

    def test_returns_none_when_all_fail(self):
        c = CompositeClient([
            _FakeClient("a", result=None),
            _FakeClient("b", result=None),
        ])
        assert c.invoke("hi") is None

    def test_get_usage_summary_merges_nonempty(self):
        c = CompositeClient([_FakeClient("a"), _FakeClient("b")])
        out = c.get_usage_summary()
        assert "usage-a" in out
        assert "usage-b" in out

    def test_get_usage_summary_empty_when_none(self):
        c = CompositeClient([_FakeClient("a", result=None)])
        c.clients[0].usage = ""
        assert c.get_usage_summary() == ""

    def test_unified_key_rotation_summary_has_provider_column(self):
        class _KeyClient(_FakeClient):
            def __init__(self, name, rows):
                super().__init__(name)
                self._rows = rows
                self.usage = f"usage-{name}"
                self.render_key_rotation = True

            def get_key_rotation_rows(self):
                return self._rows

            def get_usage_summary(self, start_time=None, end_time=None):
                return self.usage

        a = _KeyClient("a", [("gemini", 0, "AIza...1234", 5, 1)])
        b = _KeyClient("b", [("opencode-go", "medium", "opencode-go/deepseek-v4-flash", 3, 0)])
        c = CompositeClient([a, b])
        out = c.get_usage_summary()
        assert "## API Key Rotation Summary" in out
        assert "| Provider | Key | Preview / Model | Success | Failures |" in out
        # Provider column present and populated
        assert "| gemini | 0 | `AIza...1234` | 5 | 1 |" in out
        assert "| opencode-go | medium | `opencode-go/deepseek-v4-flash` | 3 | 0 |" in out
        # Clients were told to suppress their own inline key tables
        assert a.render_key_rotation is False
        assert b.render_key_rotation is False

    def test_no_key_rotation_when_no_rows(self):
        c = CompositeClient([_FakeClient("a", result=None)])
        c.clients[0].usage = "some usage"
        out = c.get_usage_summary()
        assert "API Key Rotation Summary" not in out

    def test_hanging_client_is_skipped_and_falls_through(self):
        class _HangingClient(_FakeClient):
            def invoke(self, prompt, tier="medium", system_prompt=None, **kwargs):
                import time
                time.sleep(30)
                return "too-late"

        fast = _FakeClient("fast", result="from-fast")
        c = CompositeClient([_HangingClient("hanging"), fast], timeout=0.2)
        result = c.invoke("hi")
        assert result == "from-fast"

    def test_timeout_returns_from_second_client(self):
        class _SlowThenOk(_FakeClient):
            def invoke(self, prompt, tier="medium", system_prompt=None, **kwargs):
                import time
                time.sleep(20)
                return "slow-result"

        ok = _FakeClient("ok", result="ok-result")
        c = CompositeClient([_SlowThenOk("slow"), ok], timeout=0.1)
        assert c.invoke("hi") == "ok-result"

    def test_custom_timeout_is_respected(self):
        class _QuickClient(_FakeClient):
            def invoke(self, prompt, tier="medium", system_prompt=None, **kwargs):
                return "quick"

        c = CompositeClient([_QuickClient("q")], timeout=5)
        assert c._timeout == 5
        assert c.invoke("hi") == "quick"
