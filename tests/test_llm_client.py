"""Tests for BaseLLMClient ABC."""

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
