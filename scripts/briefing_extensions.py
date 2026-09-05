#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Fork-local briefing sections, declared as config data instead of code.

Upstream deleted the "Solo Founder Angle" and "Agent Cost-Optimization Play"
sections outright (upstream commit ``eef2c10``), so they exist only in this
fork. Carrying their ~185 lines of bespoke prompt text inside
``scripts/intelligence.py`` -- a file upstream keeps changing -- put the
fork's most-edited prose directly in the merge path, and it went stale
without anyone noticing: the cost-optimization section was still writing
Amazon Bedrock advice months after the pipeline moved to OpenRouter.

So the sections live here, and their wording lives in ``config.yaml`` under
``extension_sections``. Retargeting a section at a different audience, stack,
or output shape is a config edit; adding or removing one is a config edit;
``intelligence.py`` and ``briefing_runner.py`` keep only a generic loop that
knows nothing about any particular section.

Config shape (each entry renders one ``## Heading`` section, in order)::

    extension_sections:
      - key: solo_founder_angle      # synthesis key + features.<key> flag
        enabled: true
        heading: "Solo Founder Angle"
        tier: heavy                  # light | medium | heavy
        persona: "You are ..."       # who is writing, and for whom
        task: "Propose ONE ..."      # the single job for this section
        fields:                      # the exact markdown skeleton to fill
          - "**Product:** <one sentence>"
        rules:                       # section-specific constraints
          - "Must be buildable solo."
        limits:                      # optional per-source signal caps
          news: 6

``features.<key>: false`` still disables a section, so the pre-existing
feature flags in ``config_local.yaml`` keep working unchanged.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.leak_detection import is_cot_leak

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# How many items of each kind are quoted into a section's <signals> block.
# Overridable per section via ``limits``.
DEFAULT_LIMITS: Dict[str, int] = {
    "top_papers": 3,
    "papers": 6,
    "blogs": 6,
    "news": 6,
    "emerging_themes": 8,
}

# Grounding rules appended to every extension section. These are the same
# guards the executive-summary prompt carries, and they are the reason a
# section can no longer advertise a remembered product launch as upcoming.
_GROUNDING_RULES: Tuple[str, ...] = (
    "Ground every claim about what exists or what happened in <signals>. The "
    "proposal itself is yours to invent; news, releases, pricing, and "
    "capabilities are not.",
    "Today is {today}. Anything you recall from training may be out of date: "
    "never present a remembered product, version, or event as current, and "
    "never describe a date already past as upcoming.",
    "Never output your internal reasoning, planning, or verification steps -- "
    "output only the section itself.",
)


@dataclass(frozen=True)
class ExtensionSection:
    """One config-declared briefing section."""

    key: str
    heading: str
    tier: str = "heavy"
    persona: str = ""
    task: str = ""
    fields: Tuple[str, ...] = ()
    rules: Tuple[str, ...] = ()
    limits: Dict[str, int] = field(default_factory=dict)

    def limit(self, name: str) -> int:
        """Signal cap for one source kind, falling back to the default."""
        return int(self.limits.get(name, DEFAULT_LIMITS.get(name, 0)))


def _as_tuple(value: Any) -> Tuple[str, ...]:
    """Coerce a config scalar or list into a tuple of non-empty strings."""
    if not value:
        return ()
    if isinstance(value, str):
        value = [value]
    return tuple(str(v).strip() for v in value if str(v).strip())


def load_extension_sections(config: Dict[str, Any]) -> List[ExtensionSection]:
    """
    Build the enabled extension sections from config, in declaration order.

    A section is skipped when its own ``enabled`` is false or when
    ``features.<key>`` is false -- the latter so the feature flags already in
    ``config_local.yaml`` keep switching these sections off.

    Args:
        config: Full config dictionary.

    Returns:
        Enabled sections, in the order they should be rendered.
    """
    declared = config.get("extension_sections") or []
    if not isinstance(declared, list):
        logger.warning("extension_sections is not a list; ignoring")
        return []

    features = config.get("features", {}) or {}
    sections: List[ExtensionSection] = []
    for entry in declared:
        if not isinstance(entry, dict):
            logger.warning("Skipping malformed extension_sections entry: %r", entry)
            continue
        key = str(entry.get("key", "")).strip()
        if not key:
            logger.warning("Skipping extension section with no key: %r", entry)
            continue
        if not entry.get("enabled", True) or not features.get(key, True):
            logger.info("Extension section %s disabled by config", key)
            continue
        sections.append(
            ExtensionSection(
                key=key,
                heading=str(entry.get("heading") or key.replace("_", " ").title()),
                tier=str(entry.get("tier") or "heavy"),
                persona=str(entry.get("persona") or "").strip(),
                task=str(entry.get("task") or "").strip(),
                fields=_as_tuple(entry.get("fields")),
                rules=_as_tuple(entry.get("rules")),
                limits=dict(entry.get("limits") or {}),
            )
        )
    return sections


def _signal_lines(
    section: ExtensionSection,
    label: str,
    items: Sequence[Dict[str, Any]],
    formatter,
) -> str:
    """Render one labelled block of the <signals> payload, or ''."""
    limit = section.limit(label.lower().replace(" ", "_"))
    chosen = list(items or [])[:limit] if limit else []
    if not chosen:
        return ""
    return f"{label}:\n" + "\n".join(formatter(i) for i in chosen)


def build_signals(
    section: ExtensionSection,
    papers: Sequence[Dict[str, Any]],
    blogs: Sequence[Dict[str, Any]],
    news: Sequence[Dict[str, Any]],
    top_papers: Sequence[Dict[str, Any]],
    emerging_themes: Optional[Sequence[str]] = None,
) -> str:
    """
    Assemble the ``<signals>`` payload for a section.

    Each item carries its summary where one exists. The previous versions of
    these prompts passed bare titles, which gave the model nothing to reason
    from and invited it to fill the gap from memory.

    Args:
        section: The section being generated (supplies per-kind limits).
        papers: Today's papers.
        blogs: Today's blog posts.
        news: Today's news.
        top_papers: The top paper picks.
        emerging_themes: Themes detected outside the configured topics.

    Returns:
        The signal text, or "" when there is nothing to say.
    """

    def summarize(item: Dict[str, Any]) -> str:
        return str(
            item.get("brief_summary") or item.get("ai_summary") or ""
        ).strip()

    def plain(item: Dict[str, Any]) -> str:
        title = str(item.get("title", "")).strip()
        summary = summarize(item)
        return f"- {title}: {summary}" if summary else f"- {title}"

    def sourced(item: Dict[str, Any]) -> str:
        source = str(item.get("source", "")).strip()
        title = str(item.get("title", "")).strip()
        summary = summarize(item)
        head = f"- [{source}] {title}" if source else f"- {title}"
        return f"{head}: {summary}" if summary else head

    blocks = [
        _signal_lines(section, "TOP PAPERS", top_papers, plain),
        _signal_lines(section, "PAPERS", papers, plain),
        _signal_lines(section, "BLOGS", blogs, sourced),
        _signal_lines(section, "NEWS", news, sourced),
    ]
    themes = list(emerging_themes or [])[: section.limit("emerging_themes")]
    if themes:
        blocks.append("EMERGING THEMES:\n" + "\n".join(f"- {t}" for t in themes))
    return "\n\n".join(b for b in blocks if b)


def build_prompt(section: ExtensionSection, signals: str, today: str) -> str:
    """
    Assemble a section's full prompt from its config and today's signals.

    Args:
        section: The section definition.
        signals: Output of :func:`build_signals`.
        today: Human-readable date, injected so the model cannot date the
            briefing from training-cutoff cues.

    Returns:
        The prompt string.
    """
    parts: List[str] = []
    if section.persona:
        parts.append(section.persona)
    if section.task:
        parts.append(section.task)
    if section.fields:
        parts.append(
            "Use this exact markdown structure (terse, no fluff):\n\n"
            + "\n".join(section.fields)
        )
    rules = list(section.rules) + [r.format(today=today) for r in _GROUNDING_RULES]
    parts.append("Rules:\n" + "\n".join(f"- {r}" for r in rules))
    parts.append(f"<signals>\n{signals}\n</signals>")
    return "\n\n".join(parts)


def generate_section(
    section: ExtensionSection,
    client,
    system_prompt: str,
    papers: Sequence[Dict[str, Any]],
    blogs: Sequence[Dict[str, Any]],
    news: Sequence[Dict[str, Any]],
    top_papers: Sequence[Dict[str, Any]],
    emerging_themes: Optional[Sequence[str]] = None,
    now: Optional[datetime] = None,
) -> str:
    """
    Generate one extension section's markdown body.

    Args:
        section: The section definition.
        client: An LLM client exposing ``invoke(prompt, tier, system_prompt)``.
        system_prompt: The shared editorial system prompt.
        papers: Today's papers.
        blogs: Today's blog posts.
        news: Today's news.
        top_papers: The top paper picks.
        emerging_themes: Themes detected outside the configured topics.
        now: Override for the current time (tests).

    Returns:
        Markdown body for the section, or "" when there is nothing to render
        or the model leaked its scaffolding instead of answering.
    """
    signals = build_signals(section, papers, blogs, news, top_papers, emerging_themes)
    if not signals:
        logger.info("Extension section %s has no signals today; skipping", section.key)
        return ""

    today = (now or datetime.now()).strftime("%B %d, %Y")
    prompt = build_prompt(section, signals, today)
    result = client.invoke(prompt, tier=section.tier, system_prompt=system_prompt)
    if not result:
        return ""
    result = result.strip()
    if is_cot_leak(result):
        logger.warning(
            "Extension section %s leaked CoT scaffolding; omitting", section.key
        )
        return ""
    logger.info("Extension section %s generated", section.key)
    return result
