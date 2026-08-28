"""Tests for the pre-flight model availability check.

The failures these pin were all observed live: preflight marked a working
model dead (token budget too small for reasoning), probed a mangled slug, and
logged "ALL models failed" while writing available: true.
"""

from unittest.mock import patch

import pytest

from scripts import preflight_model_check as pf


class TestRosterComesFromConfig:
    """A hardcoded model table in preflight drifts from config and crosses tiers."""

    def test_matrix_uses_configured_models(self):
        config = {
            "openrouter": {
                "enabled": True,
                "models": {"heavy": "openrouter/h:free", "medium": "openrouter/m:free",
                           "light": "openrouter/l:free"},
                "fallback_models": {"heavy": ["openrouter/hf:free"], "medium": [],
                                    "light": ["openrouter/lf:free"]},
            }
        }
        matrix = pf.build_test_matrix(config)
        by_tier = {tier: (primary, chain) for _, tier, primary, chain in matrix}
        assert by_tier["heavy"] == ("openrouter/h:free", ["openrouter/hf:free"])
        assert by_tier["medium"] == ("openrouter/m:free", [])
        assert by_tier["light"] == ("openrouter/l:free", ["openrouter/lf:free"])

    def test_disabled_providers_are_skipped(self):
        assert pf.build_test_matrix({"openrouter": {"enabled": False}}) == []
        assert pf.build_test_matrix({}) == []

    def test_falls_back_to_client_defaults_not_a_local_table(self):
        from scripts.openrouter_client import DEFAULT_MODELS

        matrix = pf.build_test_matrix({"openrouter": {"enabled": True}})
        heavy = next(p for _, t, p, _ in matrix if t == "heavy")
        assert heavy == DEFAULT_MODELS["heavy"]

    def test_all_three_tiers_are_probed(self):
        matrix = pf.build_test_matrix({"openrouter": {"enabled": True}})
        assert {t for _, t, _, _ in matrix} == {"heavy", "medium", "light"}


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

    def test_available_result_names_the_model_that_answered(self):
        calls = []

        def fake(model, timeout=pf.TEST_TIMEOUT):
            calls.append(model)
            return pf._probe_result(model == "openrouter/b:free", 10.0,
                                    None if model == "openrouter/b:free" else "dead")

        with patch.object(pf, "test_openrouter_model", side_effect=fake):
            result = pf.test_model_chain("openrouter", "heavy", "openrouter/a:free",
                                         ["openrouter/b:free"])
        assert result["available"] is True
        assert result["model"] == "openrouter/b:free"
        assert result["fallback_used"] is True
        assert calls == ["openrouter/a:free", "openrouter/b:free"]

    def test_total_failure_is_reported_as_unavailable(self):
        with patch.object(pf, "test_openrouter_model",
                          side_effect=lambda m, timeout=None: pf._probe_result(False, 1.0, "dead")):
            result = pf.test_model_chain("openrouter", "heavy", "openrouter/a:free",
                                         ["openrouter/b:free"])
        assert result["available"] is False
        assert len(result["attempts"]) == 2

    def test_result_never_names_a_model_outside_the_tier_chain(self):
        with patch.object(pf, "test_openrouter_model",
                          side_effect=lambda m, timeout=None: pf._probe_result(True, 1.0)):
            result = pf.test_model_chain("openrouter", "light", "openrouter/l:free",
                                         ["openrouter/lf:free"])
        assert result["model"] in ("openrouter/l:free", "openrouter/lf:free")
        assert result["tier"] == "light"

    def test_primary_success_skips_the_fallbacks(self):
        calls = []

        def fake(model, timeout=pf.TEST_TIMEOUT):
            calls.append(model)
            return pf._probe_result(True, 1.0)

        with patch.object(pf, "test_openrouter_model", side_effect=fake):
            pf.test_model_chain("openrouter", "heavy", "openrouter/a:free",
                                ["openrouter/b:free", "openrouter/c:free"])
        assert calls == ["openrouter/a:free"]


class TestPaidBackendIsNotProbed:
    """Probing a paid last-resort backend daily is the only thing that bills it."""

    def test_preflight_check_false_skips_the_provider(self):
        config = {"opencode": {"enabled": True, "preflight_check": False},
                  "openrouter": {"enabled": True}}
        assert {p for p, _, _, _ in pf.build_test_matrix(config)} == {"openrouter"}

    def test_preflight_check_defaults_to_true(self):
        config = {"opencode": {"enabled": True}}
        assert {p for p, _, _, _ in pf.build_test_matrix(config)} == {"opencode"}


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
