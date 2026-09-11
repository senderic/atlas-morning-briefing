"""Tests for the CompositeClient (model-rung fallback chain)."""

import pytest

from scripts.composite_client import CompositeClient
from scripts.llm_chain import Rung


class _FakeClient:
    """Minimal fake implementing BaseLLMClient's interface for chaining tests."""

    def __init__(self, name, available=True, result=None, results=None):
        self.name = name
        self._available = available
        self.result = result
        # Per-model results, so one fake can stand in for a backend serving
        # several rungs of the same chain.
        self.results = results or {}
        self.calls = []
        self.usage = f"usage-{name}"

    @property
    def available(self):
        return self._available

    def invoke(self, prompt, tier="medium", system_prompt=None, model=None, **kwargs):
        self.calls.append((prompt, tier, system_prompt, model))
        if model in self.results:
            return self.results[model]
        return self.result

    def get_usage_summary(self, start_time=None, end_time=None):
        return self.usage


def _chain(*models, tier="medium"):
    from scripts.llm_chain import resolve_backend

    return {tier: [Rung(model=m, backend=resolve_backend(m) or "openrouter") for m in models]}


class TestCompositeClient:
    def test_requires_at_least_one_client(self):
        with pytest.raises(ValueError):
            CompositeClient({}, {})

    def test_available_true_if_any_available(self):
        c = CompositeClient(
            {"a": _FakeClient("a", available=False), "b": _FakeClient("b")}, {}
        )
        assert c.available is True

    def test_available_false_if_none_available(self):
        c = CompositeClient({"a": _FakeClient("a", available=False)}, {})
        assert c.available is False

    def test_returns_first_rung_that_answers(self):
        client = _FakeClient("or", results={"openrouter/m1": "from-m1"})
        c = CompositeClient({"openrouter": client}, _chain("openrouter/m1", "openrouter/m2"))
        assert c.invoke("hi") == "from-m1"
        assert [call[3] for call in client.calls] == ["openrouter/m1"]

    def test_falls_through_to_the_next_rung(self):
        client = _FakeClient("or", results={"openrouter/m1": None, "openrouter/m2": "from-m2"})
        c = CompositeClient({"openrouter": client}, _chain("openrouter/m1", "openrouter/m2"))
        assert c.invoke("hi") == "from-m2"
        assert [call[3] for call in client.calls] == ["openrouter/m1", "openrouter/m2"]

    def test_chain_may_interleave_backends(self):
        """The regression the refactor exists for.

        Backend-ordered fallback could visit each backend once, so a chain
        shaped opencode -> openrouter -> opencode was inexpressible and the
        free `opencode/muse-spark-*` heavy rung was never reached.
        """
        oc = _FakeClient(
            "oc",
            results={"opencode/muse-spark": None, "opencode-go/deepseek-v4-pro": "paid"},
        )
        orc = _FakeClient("or", results={"openrouter/nemotron": None})
        c = CompositeClient(
            {"opencode": oc, "openrouter": orc},
            _chain("opencode/muse-spark", "openrouter/nemotron", "opencode-go/deepseek-v4-pro"),
        )
        assert c.invoke("hi") == "paid"
        assert [call[3] for call in oc.calls] == [
            "opencode/muse-spark",
            "opencode-go/deepseek-v4-pro",
        ]
        assert [call[3] for call in orc.calls] == ["openrouter/nemotron"]

    def test_free_rung_is_tried_before_the_paid_one_on_the_same_backend(self):
        oc = _FakeClient("oc", results={"opencode/muse-spark": "free-answer"})
        c = CompositeClient(
            {"opencode": oc},
            _chain("opencode/muse-spark", "opencode-go/deepseek-v4-pro"),
        )
        assert c.invoke("hi") == "free-answer"
        assert "opencode-go/deepseek-v4-pro" not in [call[3] for call in oc.calls]

    def test_skips_rung_whose_backend_is_not_enabled(self):
        orc = _FakeClient("or", results={"openrouter/m2": "from-m2"})
        c = CompositeClient(
            {"openrouter": orc}, _chain("opencode/muse-spark", "openrouter/m2")
        )
        assert c.invoke("hi") == "from-m2"

    def test_skips_unavailable_backend(self):
        oc = _FakeClient("oc", available=False, result="should-not-run")
        orc = _FakeClient("or", results={"openrouter/m2": "from-m2"})
        c = CompositeClient(
            {"opencode": oc, "openrouter": orc},
            _chain("opencode/muse-spark", "openrouter/m2"),
        )
        assert c.invoke("hi") == "from-m2"
        assert oc.calls == []

    def test_invoke_passes_system_prompt_and_tier(self):
        client = _FakeClient("or", result="ok")
        c = CompositeClient({"openrouter": client}, _chain("openrouter/m1", tier="heavy"))
        c.invoke("hi", tier="heavy", system_prompt="SYS")
        assert client.calls[0][1] == "heavy"
        assert client.calls[0][2] == "SYS"

    def test_returns_none_when_every_rung_fails(self):
        client = _FakeClient("or", result=None)
        c = CompositeClient({"openrouter": client}, _chain("openrouter/m1", "openrouter/m2"))
        assert c.invoke("hi") is None

    def test_returns_none_when_the_tier_has_no_chain(self):
        c = CompositeClient({"openrouter": _FakeClient("or", result="x")}, {})
        assert c.invoke("hi", tier="heavy") is None

    def test_explicit_model_pins_one_rung(self):
        client = _FakeClient("or", results={"openrouter/m2": "from-m2"})
        c = CompositeClient({"openrouter": client}, _chain("openrouter/m1", "openrouter/m2"))
        assert c.invoke("hi", model="openrouter/m2") == "from-m2"
        assert [call[3] for call in client.calls] == ["openrouter/m2"]

    def test_get_usage_summary_merges_nonempty(self):
        c = CompositeClient({"a": _FakeClient("a"), "b": _FakeClient("b")}, {})
        out = c.get_usage_summary()
        assert "usage-a" in out
        assert "usage-b" in out

    def test_get_usage_summary_empty_when_none(self):
        client = _FakeClient("a", result=None)
        client.usage = ""
        assert CompositeClient({"a": client}, {}).get_usage_summary() == ""

    def test_unified_key_rotation_summary_has_provider_column(self):
        class _KeyClient(_FakeClient):
            def __init__(self, name, rows):
                super().__init__(name)
                self._rows = rows
                self.render_key_rotation = True

            def get_key_rotation_rows(self):
                return self._rows

        a = _KeyClient("a", [("gemini", 0, "AIza...1234", 5, 1)])
        b = _KeyClient(
            "b", [("opencode-go", "medium", "opencode-go/deepseek-v4-flash", 3, 0)]
        )
        out = CompositeClient({"a": a, "b": b}, {}).get_usage_summary()
        assert "## API Key Rotation Summary" in out
        assert "| Provider | Key | Preview / Model | Success | Failures |" in out
        assert "| gemini | 0 | `AIza...1234` | 5 | 1 |" in out
        assert "| opencode-go | medium | `opencode-go/deepseek-v4-flash` | 3 | 0 |" in out
        # Clients were told to suppress their own inline key tables
        assert a.render_key_rotation is False
        assert b.render_key_rotation is False

    def test_no_key_rotation_when_no_rows(self):
        client = _FakeClient("a", result=None)
        client.usage = "some usage"
        out = CompositeClient({"a": client}, {}).get_usage_summary()
        assert "API Key Rotation Summary" not in out


class TestPerRungTimeout:
    """Each rung gets its own window, so one hanging model costs one window."""

    def test_hanging_rung_is_skipped_and_the_chain_continues(self):
        class _Hanging(_FakeClient):
            def invoke(self, prompt, tier="medium", system_prompt=None, model=None, **kwargs):
                import time

                time.sleep(30)
                return "too-late"

        c = CompositeClient(
            {"opencode": _Hanging("hanging"), "openrouter": _FakeClient("fast", result="from-fast")},
            _chain("opencode/slow", "openrouter/fast"),
            timeout=0.2,
        )
        assert c.invoke("hi") == "from-fast"

    def test_custom_timeout_is_respected(self):
        c = CompositeClient(
            {"openrouter": _FakeClient("q", result="quick")},
            _chain("openrouter/q"),
            timeout=5,
        )
        assert c._timeout == 5
        assert c.invoke("hi") == "quick"


class TestRungBudgetGuard:
    """A rung that cannot finish inside its window never serves.

    Under the old backend-ordered shape the window had to cover a backend's
    ENTIRE internal chain — 300s timeout x 3 retries x 3 models = 2730s inside
    a 400s window — which is how the paid backstop was found dead on arrival.
    A rung is one model, so the multiplier is only its own retries.
    """

    class _Backend(_FakeClient):
        def __init__(self, timeout, retries):
            super().__init__("budgeted")
            self._timeout = timeout
            self.max_retries = retries

    def test_worst_case_counts_retries_only_not_a_chain(self):
        b = self._Backend(timeout=100, retries=2)
        assert CompositeClient._worst_case_seconds(b) == 300

    def test_warns_when_a_rung_cannot_finish_in_its_window(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            CompositeClient(
                {"openrouter": self._Backend(300, 2)}, _chain("openrouter/m"), timeout=400
            )
        assert "will be cut off" in caplog.text

    def test_silent_when_the_budget_fits(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            CompositeClient(
                {"openrouter": self._Backend(150, 0)}, _chain("openrouter/m"), timeout=350
            )
        assert "will be cut off" not in caplog.text

    def test_warns_about_a_rung_whose_backend_is_not_enabled(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            CompositeClient(
                {"openrouter": _FakeClient("or")},
                _chain("opencode/muse-spark", "openrouter/m"),
            )
        assert "not enabled" in caplog.text

    def test_backend_without_a_timeout_is_not_guessed_at(self):
        assert CompositeClient._worst_case_seconds(_FakeClient("plain")) is None
