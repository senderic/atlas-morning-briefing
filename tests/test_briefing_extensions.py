# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for config-declared briefing sections (scripts/briefing_extensions.py)."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from scripts.briefing_extensions import (
    ExtensionSection,
    build_prompt,
    build_signals,
    generate_section,
    load_extension_sections,
)


SECTION_CONFIG = {
    "key": "pipeline_efficiency_play",
    "heading": "Pipeline Efficiency Play",
    "tier": "heavy",
    "persona": "You advise an engineer running a cron-driven LLM pipeline.",
    "task": "Propose ONE change.",
    "fields": ["**Play:** <one sentence>", "**Watch-out:** <one sentence>"],
    "rules": ["Reference TODAY's signals."],
}


@pytest.fixture
def section():
    return load_extension_sections({"extension_sections": [SECTION_CONFIG]})[0]


# ---------- load_extension_sections ----------


class TestLoadExtensionSections:
    def test_absent_key_yields_nothing(self):
        assert load_extension_sections({}) == []

    def test_parses_declaration(self, section):
        assert section.key == "pipeline_efficiency_play"
        assert section.heading == "Pipeline Efficiency Play"
        assert section.tier == "heavy"
        assert section.fields == (
            "**Play:** <one sentence>",
            "**Watch-out:** <one sentence>",
        )

    def test_declaration_order_is_preserved(self):
        cfg = {
            "extension_sections": [
                {"key": "b", "heading": "B"},
                {"key": "a", "heading": "A"},
            ]
        }
        assert [s.key for s in load_extension_sections(cfg)] == ["b", "a"]

    def test_enabled_false_skips_section(self):
        cfg = {"extension_sections": [{**SECTION_CONFIG, "enabled": False}]}
        assert load_extension_sections(cfg) == []

    def test_feature_flag_skips_section(self):
        """The pre-existing features.<key> flags keep working."""
        cfg = {
            "extension_sections": [SECTION_CONFIG],
            "features": {"pipeline_efficiency_play": False},
        }
        assert load_extension_sections(cfg) == []

    def test_entry_without_key_is_skipped(self):
        cfg = {"extension_sections": [{"heading": "Nameless"}, SECTION_CONFIG]}
        assert [s.key for s in load_extension_sections(cfg)] == [
            "pipeline_efficiency_play"
        ]

    def test_malformed_entry_is_skipped(self):
        cfg = {"extension_sections": ["not a mapping", SECTION_CONFIG]}
        assert len(load_extension_sections(cfg)) == 1

    def test_non_list_declaration_is_ignored(self):
        assert load_extension_sections({"extension_sections": "nope"}) == []

    def test_heading_defaults_from_key(self):
        cfg = {"extension_sections": [{"key": "my_section"}]}
        assert load_extension_sections(cfg)[0].heading == "My Section"

    def test_string_rules_are_accepted(self):
        cfg = {"extension_sections": [{"key": "k", "rules": "One rule."}]}
        assert load_extension_sections(cfg)[0].rules == ("One rule.",)


# ---------- build_signals ----------


class TestBuildSignals:
    def test_empty_inputs_yield_empty_string(self, section):
        assert build_signals(section, [], [], [], []) == ""

    def test_includes_summaries_not_just_titles(self, section):
        signals = build_signals(
            section,
            papers=[{"title": "P", "brief_summary": "The finding."}],
            blogs=[],
            news=[],
            top_papers=[],
        )
        assert "- P: The finding." in signals

    def test_title_only_item_still_renders(self, section):
        signals = build_signals(
            section, papers=[{"title": "P"}], blogs=[], news=[], top_papers=[]
        )
        assert "- P" in signals

    def test_news_and_blogs_carry_their_source(self, section):
        signals = build_signals(
            section,
            papers=[],
            blogs=[{"source": "HF", "title": "B"}],
            news=[{"source": "CNBC", "title": "N"}],
            top_papers=[],
        )
        assert "- [HF] B" in signals
        assert "- [CNBC] N" in signals

    def test_limits_are_config_driven(self):
        section = load_extension_sections(
            {"extension_sections": [{**SECTION_CONFIG, "limits": {"news": 1}}]}
        )[0]
        signals = build_signals(
            section,
            papers=[],
            blogs=[],
            news=[{"title": "N1"}, {"title": "N2"}],
            top_papers=[],
        )
        assert "N1" in signals
        assert "N2" not in signals


# ---------- build_prompt ----------


class TestBuildPrompt:
    def test_assembles_persona_task_fields_and_rules(self, section):
        prompt = build_prompt(section, "NEWS:\n- N", "September 05, 2026")
        assert section.persona in prompt
        assert section.task in prompt
        assert "**Play:** <one sentence>" in prompt
        assert "- Reference TODAY's signals." in prompt
        assert "<signals>\nNEWS:\n- N\n</signals>" in prompt

    def test_injects_todays_date(self, section):
        prompt = build_prompt(section, "s", "September 05, 2026")
        assert "Today is September 05, 2026." in prompt

    def test_carries_the_shared_grounding_rules(self, section):
        """Without these a section advertises remembered events as current."""
        prompt = build_prompt(section, "s", "September 05, 2026")
        assert "Ground every claim" in prompt
        assert "may be out of date" in prompt
        assert "Never output your internal reasoning" in prompt


# ---------- generate_section ----------


class TestGenerateSection:
    def _client(self, response):
        client = MagicMock()
        client.invoke.return_value = response
        return client

    def test_returns_model_output(self, section):
        client = self._client("**Play:** Trim the context.")
        out = generate_section(
            section, client, "SYS",
            papers=[{"title": "P"}], blogs=[], news=[], top_papers=[],
        )
        assert out == "**Play:** Trim the context."

    def test_uses_the_configured_tier(self, section):
        client = self._client("body")
        generate_section(
            section, client, "SYS",
            papers=[{"title": "P"}], blogs=[], news=[], top_papers=[],
        )
        _, kwargs = client.invoke.call_args
        assert kwargs["tier"] == "heavy"
        assert kwargs["system_prompt"] == "SYS"

    def test_no_signals_means_no_call(self, section):
        client = self._client("body")
        out = generate_section(
            section, client, "SYS", papers=[], blogs=[], news=[], top_papers=[]
        )
        assert out == ""
        client.invoke.assert_not_called()

    def test_empty_model_response_yields_empty(self, section):
        client = self._client(None)
        out = generate_section(
            section, client, "SYS",
            papers=[{"title": "P"}], blogs=[], news=[], top_papers=[],
        )
        assert out == ""

    def test_leaked_scaffolding_is_dropped(self, section):
        client = self._client("Let me think through what the user wants here.")
        out = generate_section(
            section, client, "SYS",
            papers=[{"title": "P"}], blogs=[], news=[], top_papers=[],
        )
        assert out == ""

    def test_date_comes_from_the_injected_clock(self, section):
        client = self._client("body")
        generate_section(
            section, client, "SYS",
            papers=[{"title": "P"}], blogs=[], news=[], top_papers=[],
            now=datetime(2026, 1, 2),
        )
        assert "Today is January 02, 2026." in client.invoke.call_args[0][0]


def test_section_limit_falls_back_to_default():
    section = ExtensionSection(key="k", heading="K")
    assert section.limit("news") == 6
    assert section.limit("unknown_kind") == 0
