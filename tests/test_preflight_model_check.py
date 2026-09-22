"""Tests for the pre-flight model availability check.

The failures these pin were all observed live: preflight marked a working
model dead (token budget too small for reasoning), probed a mangled slug, and
logged "ALL models failed" while writing available: true.
"""

from unittest.mock import patch

import pytest

from scripts import preflight_model_check as pf


@pytest.fixture(autouse=True)
def _stub_api_key(monkeypatch):
    """Give the OpenRouter probe a key so its network call is the mocked one.

    Without this the probe short-circuits on "No API key" before reaching the
    patched requests.post, so these tests passed locally only because a real
    key was present in .env — and failed in CI, which has none. Pinning a dummy
    value here makes them hermetic and independent of the developer's
    environment.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used-for-network")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class TestChainComesFromConfig:
    """A hardcoded model table in preflight drifts from config and crosses tiers."""

    def test_matrix_uses_the_configured_chain(self):
        config = {
            "openrouter": {"enabled": True},
            "llm": {"chains": {
                "heavy": ["openrouter/h:free", "openrouter/hf:free"],
                "medium": ["openrouter/m:free"],
                "light": ["openrouter/l:free", "openrouter/lf:free"],
            }},
        }
        by_tier = {tier: [r.model for r in rungs]
                   for tier, rungs in pf.build_test_matrix(config)}
        assert by_tier["heavy"] == ["openrouter/h:free", "openrouter/hf:free"]
        assert by_tier["medium"] == ["openrouter/m:free"]
        assert by_tier["light"] == ["openrouter/l:free", "openrouter/lf:free"]

    def test_disabled_backends_are_skipped(self):
        assert pf.build_test_matrix({"openrouter": {"enabled": False}}) == []
        assert pf.build_test_matrix({}) == []

    def test_rung_whose_backend_is_disabled_is_dropped(self):
        config = {
            "openrouter": {"enabled": True},
            "opencode": {"enabled": False},
            "llm": {"chains": {"heavy": ["opencode/muse-spark", "openrouter/h:free"]}},
        }
        by_tier = dict(pf.build_test_matrix(config))
        assert [r.model for r in by_tier["heavy"]] == ["openrouter/h:free"]

    def test_all_three_tiers_are_probed(self):
        matrix = pf.build_test_matrix({"openrouter": {"enabled": True}})
        assert {t for t, _ in matrix} == {"heavy", "medium", "light"}

    def test_direct_nvidia_backend_is_probeable(self):
        config = {
            "nvidia": {"enabled": True},
            "llm": {"chains": {
                "medium": ["nvidia-direct/nvidia/nemotron-3-super-120b-a12b"],
            }},
        }
        by_tier = dict(pf.build_test_matrix(config))
        assert by_tier["medium"][0].backend == "nvidia"


class TestPaidRungsAreNotProbed:
    """Probing a paid rung daily is the only thing that ever bills it.

    Skipping was per BACKEND before, which is how a free model sharing the
    paid transport (`opencode/muse-spark-*`) went unprobed and unused.
    """

    _CONFIG = {
        "openrouter": {"enabled": True},
        "opencode": {"enabled": True},
        "llm": {
            "chains": {"heavy": [
                "opencode/muse-spark-1.3-contributor-free",
                "openrouter/h:free",
                "opencode-go/deepseek-v4-pro",
            ]},
            "preflight_skip": ["opencode-go/"],
        },
    }

    def test_paid_rung_is_skipped(self):
        by_tier = dict(pf.build_test_matrix(self._CONFIG))
        assert "opencode-go/deepseek-v4-pro" not in [r.model for r in by_tier["heavy"]]

    def test_free_rung_on_the_same_transport_is_still_probed(self):
        by_tier = dict(pf.build_test_matrix(self._CONFIG))
        models = [r.model for r in by_tier["heavy"]]
        assert "opencode/muse-spark-1.3-contributor-free" in models

    def test_nothing_is_skipped_without_preflight_skip(self):
        config = dict(self._CONFIG, llm={"chains": self._CONFIG["llm"]["chains"]})
        by_tier = dict(pf.build_test_matrix(config))
        assert len(by_tier["heavy"]) == 3


class TestTokenBudget:
    def test_budget_leaves_room_for_a_reasoning_trace(self):
        """max_tokens=10 made every reasoning model look dead."""
        assert pf.TEST_MAX_TOKENS >= 512


class TestProbeSemantics:
    def _probe(self, content, reasoning="", finish="stop"):
        class R:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"finish_reason": finish,
                                     "message": {"content": content,
                                                 "reasoning": reasoning}}]}
        return R()

    def test_empty_content_is_not_available(self):
        with patch("requests.post", return_value=self._probe("", "x" * 400, "length")):
            result = pf.test_openrouter_model("openrouter/x:free")
        assert result["available"] is False
        assert "Empty content" in result["error"]

    def test_real_content_is_available(self):
        with patch("requests.post", return_value=self._probe("A briefing orients you.")):
            result = pf.test_openrouter_model("openrouter/unknown-model:free")
        assert result["available"] is True
        assert result["error"] is None

    def test_error_nested_in_a_200_body_is_not_available(self):
        class R:
            status_code = 200
            text = ""

            def json(self):
                return {"error": {"message": "Service temporarily overloaded", "code": 502}}

        with patch("requests.post", return_value=R()):
            result = pf.test_openrouter_model("openrouter/x:free")
        assert result["available"] is False


class TestChainReporting:
    """The old version logged 'ALL models failed' and wrote available: true."""

    @staticmethod
    def _rungs(*models, backend="openrouter"):
        from scripts.llm_chain import Rung

        return [Rung(model=m, backend=backend) for m in models]

    def test_available_result_names_the_rung_that_answered(self):
        calls = []

        def fake(model, timeout=pf.TEST_TIMEOUT):
            calls.append(model)
            return pf._probe_result(model == "openrouter/b:free", 10.0,
                                    None if model == "openrouter/b:free" else "dead")

        with patch.object(pf, "test_openrouter_model", side_effect=fake):
            result = pf.probe_chain(
                "heavy", self._rungs("openrouter/a:free", "openrouter/b:free")
            )
        assert result["available"] is True
        assert result["model"] == "openrouter/b:free"
        assert result["rung_index"] == 1
        assert result["fallback_used"] is True
        assert calls == ["openrouter/a:free", "openrouter/b:free"]

    def test_total_failure_is_reported_as_unavailable(self):
        with patch.object(pf, "test_openrouter_model",
                          side_effect=lambda m, timeout=None: pf._probe_result(False, 1.0, "dead")):
            result = pf.probe_chain(
                "heavy", self._rungs("openrouter/a:free", "openrouter/b:free")
            )
        assert result["available"] is False
        assert result["rung_index"] is None
        assert len(result["attempts"]) == 2

    def test_result_never_names_a_model_outside_the_tier_chain(self):
        with patch.object(pf, "test_openrouter_model",
                          side_effect=lambda m, timeout=None: pf._probe_result(True, 1.0)):
            result = pf.probe_chain(
                "light", self._rungs("openrouter/l:free", "openrouter/lf:free")
            )
        assert result["model"] in ("openrouter/l:free", "openrouter/lf:free")
        assert result["tier"] == "light"

    def test_first_rung_success_skips_the_rest(self):
        calls = []

        def fake(model, timeout=pf.TEST_TIMEOUT):
            calls.append(model)
            return pf._probe_result(True, 1.0)

        with patch.object(pf, "test_openrouter_model", side_effect=fake):
            pf.probe_chain(
                "heavy",
                self._rungs("openrouter/a:free", "openrouter/b:free", "openrouter/c:free"),
            )
        assert calls == ["openrouter/a:free"]

    def test_each_rung_is_probed_through_its_own_backend(self):
        from scripts.llm_chain import Rung

        rungs = [
            Rung(model="opencode/muse-spark", backend="opencode"),
            Rung(model="openrouter/h:free", backend="openrouter"),
        ]
        with patch.object(pf, "test_opencode_model",
                          side_effect=lambda m, timeout=None: pf._probe_result(False, 1.0, "dead")) as oc, \
             patch.object(pf, "test_openrouter_model",
                          side_effect=lambda m, timeout=None: pf._probe_result(True, 1.0)) as orc:
            result = pf.probe_chain("heavy", rungs)
        assert oc.call_count == 1 and orc.call_count == 1
        assert result["model"] == "openrouter/h:free"
        assert result["backend"] == "openrouter"

    def test_direct_nvidia_rung_uses_nvidia_probe(self):
        from scripts.llm_chain import Rung

        rungs = [
            Rung(
                model="nvidia-direct/nvidia/nemotron-3-super-120b-a12b",
                backend="nvidia",
            )
        ]
        with patch.object(
            pf,
            "test_nvidia_model",
            side_effect=lambda m, timeout=None: pf._probe_result(True, 1.0),
        ) as probe:
            result = pf.probe_chain("medium", rungs)

        probe.assert_called_once()
        assert result["backend"] == "nvidia"


class TestReasoningProbeIsTriState:
    """A failed probe is not evidence the endpoint refuses suppression.

    On 2026-08-28 a transient Nvidia 502 during the probe made preflight log
    nemotron-3-ultra as reasoning-off:UNSUPPORTED, which points the reader at
    a config problem that does not exist.
    """

    def _run(self, off_error_text):
        class R:
            status_code = 200
            text = ""

            def __init__(self, body):
                self._b = body

            def json(self):
                return self._b

        ok = R({"choices": [{"finish_reason": "stop",
                             "message": {"content": "A briefing orients you.",
                                         "reasoning": ""}}]})
        if off_error_text is None:
            second = ok
        else:
            second = R({"error": {"message": off_error_text, "code": 502}})
        caps = {"supports_reasoning_control": True,
                "reasoning_control_method": "api_param",
                "api_param_name": "reasoning",
                "api_param_value": {"enabled": False}}
        with patch("requests.post", side_effect=[ok, second]), \
             patch.object(pf, "get_model_capabilities", return_value=caps):
            return pf.test_openrouter_model("openrouter/x:free")

    def test_transient_probe_failure_is_unknown_not_unsupported(self):
        r = self._run("Upstream error from Nvidia: Service temporarily overloaded")
        assert r["available"] is True
        assert r["reasoning_disabled_ok"] is None
        assert r["reasoning_probe_inconclusive"] is True

    def test_genuine_refusal_is_recorded_as_false(self):
        r = self._run("Reasoning is mandatory for this endpoint and cannot be disabled.")
        assert r["reasoning_disabled_ok"] is False
        assert r["reasoning_probe_inconclusive"] is False

    def test_successful_suppression_is_true(self):
        r = self._run(None)
        assert r["reasoning_disabled_ok"] is True
        assert r["reasoning_probe_inconclusive"] is False
