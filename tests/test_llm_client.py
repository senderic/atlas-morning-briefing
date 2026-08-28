"""Tests for BaseLLMClient ABC."""

import pytest
from unittest.mock import patch
from scripts.llm_client import BaseLLMClient


def test_abc_cannot_instantiate():
    """BaseLLMClient should not be instantiable directly."""
    try:
        BaseLLMClient()
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "abstract" in str(e).lower()


def test_abc_subclass_must_implement_all():
    """A subclass that omits methods should also raise."""
    class Incomplete(BaseLLMClient):
        pass

    try:
        Incomplete()
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "abstract" in str(e).lower()


class TestConcreteSubclass:
    """Verify that a fully implemented subclass works."""

    def test_can_instantiate(self):
        class Full(BaseLLMClient):
            @property
            def available(self):
                return True

            def invoke(self, prompt, tier="medium", system_prompt=None):
                return "ok"

            def get_usage_summary(self, start_time=None, end_time=None):
                return ""

        instance = Full()
        assert instance.available is True
        assert instance.invoke("hello") == "ok"
        assert instance.get_usage_summary() == ""


class TestCoTLeakageMarkers:
    """The briefing is ABOUT AI research, so topic words are not leakage.

    Matching bare terms like "chain of thought" discards legitimate paper
    summaries; the detector must match the scaffolding's phrasing instead.
    """

    @pytest.fixture
    def detector(self):
        from scripts.llm_client import ReasoningControlMixin

        return ReasoningControlMixin()

    @pytest.mark.parametrize("text", [
        "This paper improves chain of thought prompting for ISR mission planning.",
        "Researchers released a reasoning trace benchmark for agent evaluation.",
        "The thinking process model from NVIDIA shows gains on long-horizon tasks.",
        "An internal monologue architecture is proposed for embodied agents.",
    ])
    def test_topical_mentions_are_not_leakage(self, detector, text):
        assert detector.detect_cot_leakage(text) is False

    @pytest.mark.parametrize("text", [
        "Strict grounding: check verbatim entities/facts before writing.",
        "Okay, so the user wants me to rank these headlines.",
        "The user is asking for a synthesis. First, I need to group the items.",
        "Let me think through what matters here.",
        "Grounding verification step: is verbatim quote present?",
    ])
    def test_scaffolding_is_leakage(self, detector, text):
        assert detector.detect_cot_leakage(text) is True

    def test_empty_text_is_not_leakage(self, detector):
        assert detector.detect_cot_leakage("") is False
        assert detector.detect_cot_leakage(None) is False


class TestCapabilityRegistry:
    def test_openrouter_models_use_the_parameter_that_actually_works(self):
        """`reasoning_effort` is accepted by the API but suppresses nothing."""
        from scripts.llm_client import _load_capabilities

        caps = _load_capabilities()["model_capabilities"]
        for model, entry in caps.items():
            if not model.startswith("openrouter/"):
                continue
            if entry.get("reasoning_control_method") != "api_param":
                continue
            assert entry["api_param_name"] == "reasoning", model
            assert entry["api_param_value"] == {"enabled": False}, model

    def test_default_roster_models_are_all_registered(self):
        """An unregistered model silently loses reasoning control."""
        from scripts.llm_client import _load_capabilities
        from scripts.openrouter_client import DEFAULT_FALLBACK_MODELS, DEFAULT_MODELS

        caps = _load_capabilities()["model_capabilities"]
        roster = set(DEFAULT_MODELS.values())
        for chain in DEFAULT_FALLBACK_MODELS.values():
            roster.update(chain)
        missing = roster - set(caps)
        assert not missing, f"unregistered models in the default roster: {missing}"


class TestApiParamDefaults:
    def test_api_param_default_is_the_parameter_that_works(self):
        """A registry entry omitting the param keys must still suppress reasoning."""
        from scripts.llm_client import ReasoningControlMixin

        mixin = ReasoningControlMixin()
        payload = {"model": "m"}
        with patch.object(mixin, "get_model_capabilities", return_value={
            "supports_reasoning_control": True,
            "reasoning_control_method": "api_param",
        }):
            result = mixin.apply_reasoning_control("m", payload, reasoning_enabled=False)
        assert result["reasoning"] == {"enabled": False}
        assert "reasoning_effort" not in result
