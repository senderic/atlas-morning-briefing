#!/usr/bin/env python3
"""Neutralise external text before it is embedded in an LLM prompt.

Feed titles, search snippets and newsletter bodies are written by strangers
and land inside prompts that use tags (`<articles>...</articles>`), fences
(`<<<`/`>>>`) and numbered lines (`[3] ...`) as structure. A field that carries
any of those can close a block early or pose as another candidate. This module
has no project imports so every prompt builder can use it.
"""

import re

# Any tag-shaped token: `<name>`, `</name>`, `<name attr="x">`, `<name/>`.
# Deliberately not a list of names -- prompts gain new delimiters over time and
# a blocklist of five tag names left every one of them forgeable. The name must
# start right after `<` (or `</`), so `a < b`, `<3` and `<https://...>` survive.
_TAG_RE = re.compile(r"</?[A-Za-z_][A-Za-z0-9_-]*(?:\s[^<>]*)?/?>")
# Fence delimiters used to bracket a whole document.
_FENCE_RE = re.compile(r"<{3,}|>{3,}")
# Every character a model may read as a line break.
_LINE_BREAKS = "\r\n\x0b\x0c\x85  "
_NEWLINES_RE = re.compile(rf"[ \t]*[{_LINE_BREAKS}]+[ \t]*")
_ODD_BREAKS_RE = re.compile(r"\r\n?|[\x0b\x0c\x85  ]")
# A line opening with `[12]` is how ranked candidates are numbered.
_LINE_INDEX_RE = re.compile(r"(?m)^([ \t]*)\[(\d+)\]")


def sanitize_prompt_input(
    text: str, max_length: int = 10000, multiline: bool = False
) -> str:
    """
    Sanitize external input before embedding in LLM prompts.

    Truncates, strips anything tag- or fence-shaped, and removes the line
    structure a single field has no business carrying.

    Args:
        text: Raw input text from external source.
        max_length: Maximum allowed character length.
        multiline: Keep line breaks, for excerpts that legitimately span
            paragraphs (a newsletter body, a rendered briefing). Line-leading
            `[n]` markers are then rewritten to `(n)` so a line inside the
            excerpt cannot pass for a numbered candidate. The default collapses
            every line break to a space, which is right for titles, sources,
            snippets and abstracts.

    Returns:
        Sanitized text safe for prompt inclusion.
    """
    if not isinstance(text, str):
        return ""
    text = text[:max_length]
    # Removing one token can splice a new one together (`<sys<b>tem>`), so
    # repeat until nothing changes; each pass only ever shortens the text.
    while True:
        stripped = _FENCE_RE.sub("", _TAG_RE.sub("", text))
        if stripped == text:
            break
        text = stripped
    if multiline:
        text = _ODD_BREAKS_RE.sub("\n", text)
        return _LINE_INDEX_RE.sub(r"\1(\2)", text)
    return _NEWLINES_RE.sub(" ", text)
