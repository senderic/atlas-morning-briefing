#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Intelligence layer for morning briefing.

Uses an LLM client to add reasoning, synthesis, and summarization to the
briefing pipeline. Falls back gracefully when the LLM client is unavailable.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from scripts.llm_client import BaseLLMClient
from scripts.interest_graph import generate_graph_queries, parse_graph
from scripts.leak_detection import is_cot_leak

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are the senior editor of a sharp, well-respected daily intelligence "
    "briefing read by busy professionals who need the signal, not the noise. Your "
    "standard is A+ senior-level journalism: lead with the most important thing, "
    "surface the non-obvious insight, and always answer 'so what?'. "
    "Write in plain, vigorous, active-voice prose. Be specific and concrete -- "
    "name the model, the number, the company, the consequence. "
    "Prize insight over erudition: never use a long word where a short one will "
    "do, cut hedging and throat-clearing, and never pad with jargon to sound "
    "smart. Be factual and skeptical: do not invent facts, numbers, or "
    "citations, and if the evidence is thin, say so plainly. "
    "Use markdown formatting."
)


def _sanitize_prompt_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize external input before embedding in LLM prompts.

    Strips prompt injection markers and truncates to prevent abuse.

    Args:
        text: Raw input text from external source.
        max_length: Maximum allowed character length.

    Returns:
        Sanitized text safe for prompt inclusion.
    """
    if not isinstance(text, str):
        return ""
    text = text[:max_length]
    # Strip characters that could be used for prompt injection / XML tag spoofing
    text = re.sub(r"</?(?:system|human|assistant|instructions?|prompt)[^>]*>", "", text, flags=re.IGNORECASE)
    return text


# Meta-commentary markers a model sometimes appends to a ranked summary,
# leaking its own filtering rationale (e.g. "Dropped: [1], [2] (duplicates)").
# Only matched when they start a new sentence or line -- never mid-sentence --
# so legitimate prose like "Charges were dropped: the DA declined to file"
# survives untouched.
_RATIONALE_MARKERS = (
    "Dropped",
    "Excluded",
    "Omitted",
    "Not selected",
    "Rejected",
    "Skipped",
)

# Rationale leaks the model has phrased WITHOUT the "Marker: ..." shape above
# (e.g. "**The other 17 didn't qualify**, and it's worth saying why...").
# Chasing every new phrasing is a losing game, but these are the wordings
# seen in the wild so far. Same sentence/line-boundary discipline applies.
_RATIONALE_PHRASES = (
    r"didn[’']t qualify",
    r"did not qualify",
    r"the other \d+",
    r"worth saying why",
)

# Markdown emphasis the model sometimes wraps a leak in (e.g. "**The other
# 17 didn't qualify**"). Optional everywhere a leak can start.
_BOLD = r"\*{0,2}"

# Shape-based catch: a rendered summary never legitimately contains a
# bulleted list of bracketed candidate numbers (e.g. "- **[1], [2], [5]**
# Mission Bay ..."). That shape only occurs when the model is explaining
# which candidates it rejected, so it's cut regardless of the wording next
# to it -- this is what keeps a *third* new phrasing from getting through.
_BULLET_INDEX_RE_PART = (
    r"(?:[-•]\s+|\*(?!\*)\s+)"
    + _BOLD + r"\[\d+\]"
    + r"(?:\s*,\s*" + _BOLD + r"\[\d+\]" + _BOLD + r")*"
    + _BOLD
)

_BOUNDARY = r"(?:^|(?<=[.!?])\s+|\n)"

_TRAILING_RATIONALE_RE = re.compile(
    _BOUNDARY
    + r"(?:"
    + _BOLD + r"(?:" + "|".join(re.escape(m) for m in _RATIONALE_MARKERS) + r"):\s"
    + r"|" + _BOLD + r"(?:" + "|".join(_RATIONALE_PHRASES) + r")"
    + r"|" + _BULLET_INDEX_RE_PART
    + r")",
    re.IGNORECASE,
)


def _strip_trailing_rationale(text: str) -> str:
    """
    Strip a trailing meta-commentary clause leaking the model's filtering
    rationale off the end of a reader-facing summary.

    Models occasionally append text like "Dropped: [1], [2] (duplicates)",
    or the same idea in different words (e.g. "**The other 17 didn't
    qualify**, and it's worth saying why..." followed by a bulleted list of
    bracketed candidate numbers), to the last item in a ranked response,
    explaining which candidates they excluded. That rationale is never
    meant for the reader. This trims both a known marker word/phrase and
    the bulleted-bracketed-index shape itself, but only when it begins a
    new sentence or line (case-insensitive), so legitimate mid-sentence
    usage (e.g. "Charges were dropped: the DA declined to file") and
    unrelated uses of the same words, or a stray "[1]" mid-sentence, are
    left intact.

    Args:
        text: Candidate summary text, possibly with a leaked rationale tail.

    Returns:
        The text with any trailing rationale clause removed and trailing
        whitespace trimmed. Returns the input unchanged if no leak is found.
    """
    if not text:
        return text
    match = _TRAILING_RATIONALE_RE.search(text)
    if not match:
        return text
    return text[: match.start()].rstrip()


class BriefingIntelligence:
    """Adds LLM-powered intelligence to the briefing pipeline."""

    def __init__(self, client: BaseLLMClient, config: Dict[str, Any]):
        """
        Initialize BriefingIntelligence.

        Args:
            client: BaseLLMClient instance.
            config: Full config dictionary.
        """
        self.client = client
        self.config = config
        self.topics = config.get("arxiv_topics", [])
        # Domain framing for prompts is config-driven, never hardcoded here.
        # Generic defaults keep the prompts sensible when briefing_profile is omitted.
        profile = config.get("briefing_profile", {}) or {}
        self.briefing_domain = profile.get("domain", "AI and technology")
        self.briefing_audience = profile.get("audience", "an AI researcher or engineer")
        self.briefing_landscape = profile.get(
            "landscape", "the AI and technology landscape"
        )
        # Ordered ranking tiers ("what matters most first"). Empty by default so
        # briefings without a configured order keep their previous behavior.
        self.briefing_priorities = [
            str(p).strip()
            for p in (profile.get("priorities") or [])
            if str(p).strip()
        ]
        # When true, prompts ask for summaries that lead with the reader's
        # action/decision (dates, deadlines, locations, costs) instead of
        # analysis-first prose.
        self.briefing_actionable = bool(profile.get("actionable", False))
        # Cap on items from one outlet per section. A local briefing legitimately
        # leans on two or three dominant papers, so the cap is config-driven.
        self.max_per_source = int(
            (config.get("source_diversity") or {}).get("max_per_source", 2)
        )
        self.source_blurb_cache: Dict[str, str] = {}

    @property
    def available(self) -> bool:
        """Check if intelligence features are available."""
        return self.client.available

    def _priority_block(self) -> str:
        """
        Render ``briefing_profile.priorities`` as a prompt block.

        Returns an empty string when no priority order is configured, so
        prompts stay byte-identical for briefings that don't use the feature.
        """
        if not self.briefing_priorities:
            return ""
        tiers = "\n".join(
            f"{i + 1}. {tier}" for i, tier in enumerate(self.briefing_priorities)
        )
        return (
            "<priority_order>\n"
            "Rank by this order, not by how dramatic a story sounds. Tier 1 "
            "takes the top slots and fills most of the list whenever tier-1 "
            "items exist; lower tiers appear only when they are genuinely "
            "consequential. Follow any per-tier instruction below:\n"
            f"{tiers}\n"
            "</priority_order>\n\n"
        )

    def _rank_directive(self) -> str:
        """Ranking sentence: defer to the configured priority order if set."""
        if self.briefing_priorities:
            return "Rank by the priority order above, then by importance."
        return "Rank by importance."

    def _actionability_note(self) -> str:
        """Render the action-first summary directive when it is enabled."""
        if not self.briefing_actionable:
            return ""
        return (
            "\nACTIONABILITY: write for a reader deciding what to do today. "
            "Lead with the concrete hook -- the date, deadline, location, "
            "street/route, dollar figure, or place to show up -- then the "
            "consequence for them. If an item implies no action, give the one "
            "fact worth knowing in a single sentence instead of padding it."
        )

    def _ranking_interests(self, topics: List[str]) -> List[str]:
        """
        Pick the interest terms used in ranking prompts.

        Falls back to ``interest_profile`` topics (highest weight first) when no
        arxiv topics are configured, so news/blog ranking still has a relevance
        signal in paper-free briefings such as the local one.
        """
        usable = [t for t in (topics or []) if str(t).strip()]
        if usable:
            return usable[:5]
        profile = self.config.get("interest_profile", []) or []
        weighted = [
            (float(item.get("weight", 0.5)), str(item.get("topic", "")).strip())
            for item in profile
            if isinstance(item, dict) and str(item.get("topic", "")).strip()
        ]
        weighted.sort(key=lambda pair: pair[0], reverse=True)
        return [topic for _, topic in weighted[:5]]

    @staticmethod
    def extract_score(text: str) -> Tuple[int, str]:
        """Extract SCORE:X/5 from text, return (score_int, cleaned_text)."""
        text = text.strip()
        match = re.match(r"SCORE:\s*(\d)/5\s*(.*)", text, re.DOTALL)
        if match:
            return int(match.group(1)), match.group(2).strip()
        # Also try "Score: X/5" variant
        match = re.match(r"[Ss]core:\s*(\d)/5\s*(.*)", text, re.DOTALL)
        if match:
            return int(match.group(1)), match.group(2).strip()
        return None, text

    @staticmethod
    def _parse_ranked_response(text: str) -> List[Tuple[int, str]]:
        """
        Parse LLM response with [number] prefixed items.

        Handles bold markers (**[1]**), numbered sub-items, and multi-line entries.

        Args:
            text: Raw LLM response text.

        Returns:
            List of (0-based index, text) tuples.
        """
        items = []
        current_idx = -1
        current_lines: List[str] = []

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Strip markdown bold markers for detection
            clean = line.lstrip("*").strip()

            # Check if line starts a new item: [number] or **[number]**
            if clean.startswith("[") and "]" in clean[:8]:
                # Save previous item
                if current_idx >= 0 and current_lines:
                    full_text = " ".join(l for l in current_lines if l)
                    if full_text.strip():
                        items.append((current_idx, full_text.strip()))

                try:
                    bracket_start = clean.index("[")
                    bracket_end = clean.index("]")
                    current_idx = int(clean[bracket_start + 1:bracket_end]) - 1
                    # Get text after the "]" and any trailing ** or title
                    rest = clean[bracket_end + 1:].strip().rstrip("*").strip()
                    current_lines = [rest] if rest else []
                except (ValueError, IndexError):
                    current_idx = -1
                    current_lines = []
            else:
                # Strip numbered sub-items like "1." "2." that are part of summary
                current_lines.append(
                    line.lstrip("0123456789.").strip()
                    if re.match(r"^\d+\.", line)
                    else line
                )

        # Save last item
        if current_idx >= 0 and current_lines:
            full_text = " ".join(l for l in current_lines if l)
            if full_text.strip():
                items.append((current_idx, full_text.strip()))

        return items

    def filter_papers_by_relevance(
        self, papers: List[Dict[str, Any]], interest_profile: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Two-stage paper filtering using Opus.

        Stage 1: Send ALL paper titles+abstracts to Opus in ONE call.
                Score each paper 1-10 for relevance to interest profile.
                Return ONLY papers scoring >= 7.
        Stage 2: Filtered papers go to existing score_papers_semantically() for deep ranking.

        Args:
            papers: List of paper dictionaries with 'title' and 'summary' keys.
            interest_profile: Optional list of interest topics with weights.
                            If None, falls back to arxiv_topics.

        Returns:
            Filtered papers (target: ~30) that are highly relevant.
        """
        if not self.available or not papers:
            return papers

        # Get interest profile from config or fall back to topics
        if interest_profile is None:
            interest_profile = self.config.get("interest_profile", [])

        # If no interest profile, fall back to current behavior
        if not interest_profile:
            logger.info("No interest_profile configured, skipping relevance filtering")
            return papers

        # Build interest profile string
        profile_lines = []
        for item in interest_profile:
            topic = item.get("topic", "")
            weight = item.get("weight", 1.0)
            profile_lines.append(f"- {topic} (weight: {weight})")
        profile_str = "\n".join(profile_lines)

        # Build paper batch (all papers) with sanitized inputs
        paper_lines = []
        for i, p in enumerate(papers):
            title = _sanitize_prompt_input(p.get("title", "Untitled"), max_length=500)
            abstract = _sanitize_prompt_input(p.get("summary", "")[:400], max_length=500)
            paper_lines.append(f"[{i+1}] {title}\n{abstract}")

        papers_block = "\n\n".join(paper_lines)

        prompt = (
            "You are filtering papers for a daily AI research briefing. "
            "Score each paper 1-10 for relevance to this interest profile:\n\n"
            f"<interest_profile>\n{profile_str}\n</interest_profile>\n\n"
            f"<papers>\n{papers_block}\n</papers>\n\n"
            "Return ONLY papers scoring >= 7. For each relevant paper, respond with:\n"
            "[number] score reason\n"
            "Example: [5] 9 Directly addresses multi-agent systems with novel evaluation methodology\n\n"
            "Be selective. Only include papers that strongly match the profile."
        )

        result = self.client.invoke(
            prompt, tier="medium", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            logger.warning("Stage 1 filtering failed, returning all papers")
            return papers

        # Parse filtered results
        filtered_papers = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            try:
                bracket_end = line.index("]")
                idx = int(line[1:bracket_end]) - 1
                rest = line[bracket_end + 1:].strip()
                parts = rest.split(" ", 1)
                score = float(parts[0])
                reason = parts[1] if len(parts) > 1 else ""

                if 0 <= idx < len(papers) and score >= 7:
                    paper = papers[idx].copy()
                    paper["relevance_score"] = score
                    paper["relevance_reason"] = reason
                    filtered_papers.append(paper)
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse line: {line}, error: {e}")
                continue

        logger.info(f"Stage 1 filtering: {len(papers)} → {len(filtered_papers)} papers (score >= 7)")
        return filtered_papers if filtered_papers else papers[:30]  # Fallback to top 30 if filtering fails

    def generate_dynamic_queries(
        self, previous_briefing_state: Optional[Dict[str, Any]], static_queries: List[str],
        today_blogs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Generate dynamic news queries.

        If the config defines an ``interest_graph``, delegates to the interest
        taxonomy engine: root queries fire as a baseline and leaf queries are
        selected per-run from signal in yesterday's briefing and today's blogs.

        Otherwise (legacy mode), takes yesterday's top stories from state plus
        static queries and generates a few targeted follow-up queries.

        Args:
            previous_briefing_state: Previous briefing state with top stories.
            static_queries: Static queries from config.
            today_blogs: Fresh blog entries for signal scoring (optional).

        Returns:
            Combined list of queries to run.
        """
        graph = parse_graph(self.config)
        if graph is not None:
            return generate_graph_queries(
                graph,
                previous_briefing_state,
                today_blogs=today_blogs,
                llm_client=self.client if self.available else None,
            )

        if not self.available or not previous_briefing_state or not static_queries:
            return static_queries

        # Extract yesterday's top stories
        prev_date = previous_briefing_state.get("date", "unknown")
        prev_paper_titles = previous_briefing_state.get("top_paper_titles", [])[:5]
        prev_blog_titles = previous_briefing_state.get("top_blog_titles", [])[:5]
        prev_news_titles = previous_briefing_state.get("top_news_titles", [])[:5]
        prev_themes = previous_briefing_state.get("emerging_themes", [])

        # Build context from yesterday's briefing
        context_parts = []
        if prev_paper_titles:
            context_parts.append(f"Top Papers ({prev_date}):\n" + "\n".join(f"- {t}" for t in prev_paper_titles))
        if prev_blog_titles:
            context_parts.append(f"Top Blogs ({prev_date}):\n" + "\n".join(f"- {t}" for t in prev_blog_titles))
        if prev_news_titles:
            context_parts.append(f"Top News ({prev_date}):\n" + "\n".join(f"- {t}" for t in prev_news_titles))
        if prev_themes:
            context_parts.append(f"Emerging Themes: {', '.join(prev_themes)}")

        if not context_parts:
            logger.info("No previous briefing context, using static queries only")
            return static_queries

        context_str = "\n\n".join(context_parts)
        static_queries_str = "\n".join(f"- {q}" for q in static_queries)

        prompt = (
            "You are generating follow-up news queries based on yesterday's AI research briefing.\n\n"
            f"<yesterday_briefing>\n{context_str}\n</yesterday_briefing>\n\n"
            f"<static_queries>\n{static_queries_str}\n</static_queries>\n\n"
            "Generate 3 targeted follow-up queries to track developments in yesterday's hot topics. "
            "Return ONLY the new queries, one per line, no numbering or bullets. "
            "Make them specific and actionable for news search.\n\n"
            "Example outputs:\n"
            "- Claude 3.5 Sonnet benchmark results\n"
            "- AWS Trainium chip adoption enterprise\n"
            "- Multi-agent orchestration frameworks release"
        )

        result = self.client.invoke(prompt, tier="light")
        if not result:
            logger.info("Dynamic query generation failed, using static queries only")
            return static_queries

        # Parse new queries
        new_queries = [
            line.strip().strip("- *")
            for line in result.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Deduplicate and limit
        existing_lower = {q.lower() for q in static_queries}
        dynamic_queries = [
            q for q in new_queries
            if q.lower() not in existing_lower and len(q) > 10
        ][:3]

        if dynamic_queries:
            logger.info(f"Generated {len(dynamic_queries)} dynamic queries: {dynamic_queries}")

        return static_queries + dynamic_queries

    def expand_topics(self, topics: List[str]) -> List[str]:
        """
        Expand user-configured topics with semantically related queries.

        Uses the light tier model for simple brainstorming.

        Args:
            topics: Original topic list from config.

        Returns:
            Expanded topic list (original + new suggestions).
        """
        if not self.available or not topics:
            return topics

        topic_list = "\n".join(f"- {t}" for t in topics)
        prompt = (
            "Given these research topics, suggest 2-3 additional related search "
            "queries that would find relevant papers on arxiv. Return ONLY the "
            "new queries, one per line, no numbering or bullets.\n\n"
            f"<topics>\n{topic_list}\n</topics>"
        )

        result = self.client.invoke(prompt, tier="light")
        if not result:
            return topics

        new_topics = [
            line.strip().strip("- ")
            for line in result.strip().split("\n")
            if line.strip() and line.strip() != "-"
        ]

        # Deduplicate and limit additions
        existing_lower = {t.lower() for t in topics}
        additions = [
            t for t in new_topics
            if t.lower() not in existing_lower and len(t) > 3
        ][:3]

        if additions:
            logger.info(f"Expanded topics with {len(additions)} suggestions: {additions}")

        return topics + additions

    def generate_author_blurbs(
        self, items: List[Dict[str, Any]], item_type: str = "item"
    ) -> List[Dict[str, Any]]:
        """
        Generate blurbs about authors or organizations for a list of items.

        Uses the medium tier model to research and summarize author background,
        past work, and trustworthiness.

        Args:
            items: List of item dictionaries (papers, news, or blogs).
            item_type: String describing the type of items (e.g., "papers").

        Returns:
            Items with added 'author_blurb' key.
        """
        if not self.available or not items:
            return items

        # Attempt to extract missing authors using the light model first
        items = self.extract_missing_authors(items, item_type)

        # 1. Map each item to its source_info and check cache
        item_source_infos = []
        for item in items:
            # Handle different ways authors/sources are stored
            if item_type == "papers":
                authors_list = item.get("authors", [])
                if authors_list:
                    # Clean each author name
                    authors_list = [str(a).strip() for a in authors_list if str(a).strip()]
                    authors = ", ".join(authors_list[:3])
                    s_info = f"Authors: {authors}"
                else:
                    s_info = "Authors: Unknown"
            elif item_type == "blogs":
                author = str(item.get("author", "")).strip()
                source = str(item.get("source", "")).strip()
                if author:
                    s_info = f"Author: {author}, Blog: {source}"
                else:
                    s_info = f"Blog: {source}"
            else:  # news or generic
                source = str(item.get("source", "")).strip()
                s_info = f"Source/Organization: {source}"
            
            item_source_infos.append(s_info)

        # 2. Identify missing blurbs (deduplicated by normalized source)
        missing_sources = {}  # norm_source -> (original_s_info, sample_title)
        for i, (item, s_info) in enumerate(zip(items, item_source_infos)):
            norm_source = s_info.strip().lower()
            if norm_source in self.source_blurb_cache:
                item["author_blurb"] = self.source_blurb_cache[norm_source]
            elif norm_source not in missing_sources:
                title = _sanitize_prompt_input(item.get("title", "Untitled"), max_length=200)
                missing_sources[norm_source] = (s_info, title)

        if not missing_sources:
            return items

        # 3. Prepare batch prompt for missing items only
        fetch_list = list(missing_sources.items()) # list of (norm_source, (s_info, title))
        item_texts = []
        for i, (norm_source, (s_info, title)) in enumerate(fetch_list):
            item_texts.append(f"[{i+1}] {title}\n{s_info}")

        items_block = "\n\n".join(item_texts)
        prompt = (
            f"For each {item_type[:-1] if item_type.endswith('s') else item_type} below, "
            "provide a concise blurb (2-3 sentences) about the author(s) or the organization. "
            "Include who they are, their past work, what they are known for, and how trustworthy they seem. "
            "Base your assessment of trustworthiness on reputable sources such as PBS, NPR, NYT, "
            "or university publications. If the author is an organization or if an individual author "
            "cannot be definitively determined, provide a blurb about the organization's reputation.\n\n"
            "Make each blurb specific to THIS source: lead with a concrete detail "
            "(a notable product, beat, founder, ownership, or known slant) rather "
            "than a generic 'is a leading organization known for...' opener. The "
            "reader should learn something they could not have guessed from the "
            "name alone. Vary sentence structure across items -- no template.\n\n"
            f"<items>\n{items_block}\n</items>\n\n"
            "Respond ONLY with a numbered list matching the input numbering. "
            "Do not include any introductory text, thought process, or preamble. "
            "Each item must start with [n] or n."
        )

        result = self.client.invoke(
            prompt, tier="light", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            return items

        # Parse numbered blurbs and update cache
        blurbs = _parse_numbered_list(result, len(fetch_list))
        for i, blurb in enumerate(blurbs):
            if i < len(fetch_list):
                norm_source = fetch_list[i][0]
                self.source_blurb_cache[norm_source] = blurb.strip()

        # 4. Apply newly fetched blurbs to all items
        for i, (item, s_info) in enumerate(zip(items, item_source_infos)):
            norm_source = s_info.strip().lower()
            if norm_source in self.source_blurb_cache:
                items[i]["author_blurb"] = self.source_blurb_cache[norm_source]

        logger.info(f"Generated {len(blurbs)} new source blurbs for {item_type}")
        return items

    def extract_missing_authors(
        self, items: List[Dict[str, Any]], item_type: str = "item"
    ) -> List[Dict[str, Any]]:
        """
        Use the light tier model to extract missing author names from text.

        Args:
            items: List of item dictionaries.
            item_type: Type of items.

        Returns:
            Items with updated 'author' or 'authors' keys where they were missing.
        """
        if not self.available:
            return items

        to_extract = []
        indices = []
        for i, item in enumerate(items):
            if item_type == "papers":
                if not item.get("authors"):
                    to_extract.append(item)
                    indices.append(i)
            elif item_type == "blogs":
                if not item.get("author"):
                    to_extract.append(item)
                    indices.append(i)
            # News usually only has 'source' (organization), but might have an author in snippet
            elif item_type == "news":
                if "author" not in item:
                    to_extract.append(item)
                    indices.append(i)

        if not to_extract:
            return items

        item_texts = []
        for i, item in enumerate(to_extract):
            title = _sanitize_prompt_input(item.get("title", ""), max_length=200)
            summary = _sanitize_prompt_input(
                item.get("summary", item.get("description", item.get("snippet", "")))[:300],
                max_length=350
            )
            item_texts.append(f"[{i+1}] Title: {title}\nText: {summary}")

        prompt = (
            f"Extract the primary author(s) name from these {item_type}. "
            "If no individual author is mentioned, identify the organization or source. "
            "Be very concise. Return ONLY the name(s), one per line, matching input numbering.\n\n"
            f"<items>\n" + "\n\n".join(item_texts) + "\n</items>"
        )

        result = self.client.invoke(prompt, tier="light")
        if not result:
            return items

        extracted = _parse_numbered_list(result, len(to_extract))
        for i, name in enumerate(extracted):
            idx = indices[i]
            if item_type == "papers":
                items[idx]["authors"] = [name]
            else:
                items[idx]["author"] = name

        logger.info(f"Extracted missing authors for {len(extracted)} {item_type}")
        return items

    def summarize_papers(
        self, papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate 1-2 sentence summaries for papers.

        Uses the medium tier model for factual summarization.
        Processes in a single batched call to minimize cost.

        Args:
            papers: List of paper dictionaries with 'title' and 'summary' keys.

        Returns:
            Papers with added 'brief_summary' key.
        """
        if not self.available or not papers:
            return papers

        # Batch papers into a single prompt (limit to top 10 for cost)
        batch = papers[:10]
        paper_texts = []
        for i, p in enumerate(batch):
            title = _sanitize_prompt_input(p.get("title", "Untitled"), max_length=500)
            abstract = _sanitize_prompt_input(p.get("summary", "")[:500], max_length=600)
            paper_texts.append(f"[{i+1}] {title}\n{abstract}")

        papers_block = "\n\n".join(paper_texts)
        prompt = (
            "For each paper below, write a 1-2 sentence summary that extracts the "
            "GEM -- the single most important, non-obvious thing the paper "
            "delivers and why a researcher should care. Lead with the result or "
            "new capability, not the methodology, and cite the concrete number or "
            "benchmark if the abstract gives one. Plain, active voice -- skip the "
            "academic throat-clearing ('In this paper, we...'). "
            "Return a numbered list matching the input numbering. "
            "Be factual -- use only what the abstract states; do not invent "
            "results.\n\n"
            f"<papers>\n{papers_block}\n</papers>"
        )

        result = self.client.invoke(
            prompt, tier="heavy", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            return papers

        # Parse numbered summaries back to papers
        summaries = _parse_numbered_list(result, len(batch))
        for i, summary in enumerate(summaries):
            if i < len(papers):
                papers[i]["brief_summary"] = summary

        logger.info(f"Generated summaries for {len(summaries)} papers")
        return papers

    def score_papers_semantically(
        self, papers: List[Dict[str, Any]], topics: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Score papers using semantic understanding instead of TF-IDF.

        Uses the medium tier model to assess relevance.

        Args:
            papers: List of paper dictionaries.
            topics: User's research topics.

        Returns:
            Papers with added 'semantic_score' (0-10) and 'relevance_reason' keys.
        """
        if not self.available or not papers or not topics:
            return papers

        batch = papers[:15]
        paper_lines = []
        for i, p in enumerate(batch):
            title = _sanitize_prompt_input(p.get("title", "Untitled"), max_length=500)
            abstract = _sanitize_prompt_input(p.get("summary", "")[:300], max_length=400)
            paper_lines.append(f"[{i+1}] {title}: {abstract}")

        papers_block = "\n".join(paper_lines)
        prompt = (
            "Rate each paper's relevance to these research interests on a 0-10 scale.\n\n"
            f"<interests>{', '.join(topics)}</interests>\n\n"
            f"<papers>\n{papers_block}\n</papers>\n\n"
            "For each paper, respond with ONLY this format, one per line:\n"
            "[number] score reason\n"
            "Example: [1] 8 Directly addresses agent evaluation methodology"
        )

        result = self.client.invoke(
            prompt, tier="medium", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            return papers

        # Parse scores
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            try:
                bracket_end = line.index("]")
                idx = int(line[1:bracket_end]) - 1
                rest = line[bracket_end + 1:].strip()
                parts = rest.split(" ", 1)
                score = float(parts[0])
                reason = parts[1] if len(parts) > 1 else ""
                if 0 <= idx < len(papers):
                    papers[idx]["semantic_score"] = min(10.0, max(0.0, score))
                    papers[idx]["relevance_reason"] = reason
            except (ValueError, IndexError):
                continue

        logger.info("Semantic scoring complete")
        return papers

    def assess_reproduction_feasibility(
        self, papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Assess reproduction feasibility and re-rank papers by actionability.

        Uses structured scoring across 5 dimensions to filter out papers
        that are not practically reproducible on our setup (single EC2,
        Amazon Bedrock, no GPU cluster).

        Scoring dimensions (each 1-5):
          1. code_available — Is code open-source and runnable?
          2. data_accessible — Is data open/downloadable (<10GB)?
          3. infra_fit — Can run on single EC2 + Bedrock (no GPU cluster)?
          4. bedrock_ready — Can use Bedrock models (Claude/Titan) directly?
          5. effort — Time to reproduce (5=weekend, 1=months)

        Papers scoring < 15/25 are demoted (moved below higher-scoring ones).
        Papers scoring < 10/25 are dropped from top picks entirely.

        Args:
            papers: Top-scored papers (typically 3-10).

        Returns:
            Papers re-ranked by reproduction feasibility, with structured scores.
        """
        if not self.available or not papers:
            return papers

        paper_texts = []
        for i, p in enumerate(papers):
            title = _sanitize_prompt_input(p.get("title", "Untitled"), max_length=500)
            abstract = _sanitize_prompt_input(p.get("summary", "")[:400], max_length=500)
            has_code = p.get("score_breakdown", {}).get("has_code", False)
            paper_texts.append(
                f"[{i+1}] {title}\nCode available: {has_code}\n{abstract}"
            )

        papers_block = "\n\n".join(paper_texts)
        prompt = (
            "You are evaluating papers for PRACTICAL reproduction on this setup:\n"
            "- Single EC2 GPU instance available (g5.xlarge = 1x A10G 24GB, or trn1.2xlarge = AWS Trainium)\n"
            "- Amazon Bedrock API (Claude Sonnet/Opus, Titan Embeddings)\n"
            "- Python + standard ML libraries, Kubernetes OK if single-node\n"
            "- Budget: <$50 per paper, <1 week effort\n\n"
            "Score each paper on 5 dimensions (1-5 each, 25 max):\n"
            "1. code_available: 5=open repo+README, 3=partial code, 1=no code\n"
            "2. data_accessible: 5=open data <50GB, 3=needs request/large, 1=proprietary\n"
            "3. infra_fit: 5=CPU/API only, 4=single GPU(A10G/Trainium), 3=multi-GPU single node, 2=multi-node cluster, 1=datacenter/TPU pod\n"
            "4. bedrock_ready: 5=can swap in Bedrock models directly, 3=needs adapter, 1=incompatible\n"
            "5. effort: 5=weekend(S), 4=1week(M), 3=2weeks(L), 2=month(XL), 1=impossible\n\n"
            "For each paper respond in this EXACT format (one line each):\n"
            "[number] code:X data:X infra:X bedrock:X effort:X | verdict\n\n"
            "Example: [1] code:5 data:4 infra:5 bedrock:5 effort:4 | Open benchmark + Bedrock RAG, easy to reproduce\n"
            "Example: [2] code:1 data:1 infra:1 bedrock:2 effort:1 | No code, needs GPU cluster, skip\n\n"
            f"<papers>\n{papers_block}\n</papers>"
        )

        result = self.client.invoke(
            prompt, tier="heavy", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            return papers

        # Parse structured scores
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            try:
                bracket_end = line.index("]")
                idx = int(line[1:bracket_end]) - 1
                rest = line[bracket_end + 1:].strip()

                # Parse scores: code:X data:X infra:X bedrock:X effort:X | verdict
                scores = {}
                verdict = ""
                if "|" in rest:
                    scores_part, verdict = rest.split("|", 1)
                    verdict = verdict.strip()
                else:
                    scores_part = rest

                for dim in ["code", "data", "infra", "bedrock", "effort"]:
                    match = re.search(rf"{dim}:(\d)", scores_part)
                    if match:
                        scores[dim] = int(match.group(1))

                if 0 <= idx < len(papers) and scores:
                    total = sum(scores.values())
                    papers[idx]["repro_scores"] = scores
                    papers[idx]["repro_total"] = total
                    papers[idx]["repro_verdict"] = verdict
                    papers[idx]["reproduction_assessment"] = (
                        f"Score: {total}/25 "
                        f"(code:{scores.get('code',0)} data:{scores.get('data',0)} "
                        f"infra:{scores.get('infra',0)} bedrock:{scores.get('bedrock',0)} "
                        f"effort:{scores.get('effort',0)}) — {verdict}"
                    )
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse repro line: {line}, error: {e}")
                continue

        # Re-rank: sort by repro_total descending, drop papers below threshold
        min_score = self.config.get("repro_min_score", 12)
        scored = [p for p in papers if p.get("repro_total", 0) >= min_score]
        unscored = [p for p in papers if "repro_total" not in p]
        scored.sort(key=lambda x: x.get("repro_total", 0), reverse=True)

        dropped = len(papers) - len(scored) - len(unscored)
        if dropped:
            logger.info(f"Repro gate: dropped {dropped} papers scoring <{min_score}/25")

        result_papers = scored + unscored
        logger.info(
            f"Reproduction feasibility: {len(papers)} assessed, "
            f"{len(scored)} passed gate (≥{min_score}/25), top score: "
            f"{scored[0].get('repro_total', 0) if scored else 'N/A'}/25"
        )
        return result_papers

    def rank_and_summarize_news(
        self, news: List[Dict[str, Any]], topics: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Rank news by relevance and generate 2-3 sentence summaries for top items.

        Uses the medium tier model.

        Args:
            news: List of news article dictionaries.
            topics: User's research topics for relevance ranking.

        Returns:
            Top 5 news articles, ranked and summarized.
        """
        if not self.available or not news:
            logger.info(f"News ranking skipped: available={self.available}, news_count={len(news)}")
            return news[:5]

        news_lines = []
        for i, article in enumerate(news[:20]):
            title = _sanitize_prompt_input(article.get("title", ""), max_length=300)
            source = _sanitize_prompt_input(article.get("source", ""), max_length=100)
            snippet = _sanitize_prompt_input(
                article.get("description", article.get("snippet", ""))[:200], max_length=300
            )
            news_lines.append(f"[{i+1}] {title} ({source}): {snippet}")

        articles_block = "\n".join(news_lines)
        prompt = (
            f"You are curating a daily {self.briefing_domain} intelligence briefing. "
            f"From these news articles, select the TOP 5 that genuinely matter to "
            f"{self.briefing_audience} -- prioritize real signal (concrete "
            "decisions, dates, deals, policy, money, hard numbers) over "
            "press-release noise.\n\n"
            f"{self._priority_block()}"
            f"<interests>{', '.join(self._ranking_interests(topics))}</interests>\n\n"
            f"<articles>\n{articles_block}\n</articles>\n\n"
            "For each of your top 5 picks, respond in this exact format:\n"
            "[original_number] 2-3 sentence summary.\n\n"
            "Return only the numbered picks in that exact format -- never "
            "explain, list, or justify which articles you left out.\n\n"
            "In each summary, lead with what actually happened and the concrete "
            "stakes, then deliver the 'so what' -- the implication or what it "
            "signals. Active voice, specific names and numbers, no hype, no "
            f"filler. {self._rank_directive()} Be factual. Do not invent details."
            f"{self._actionability_note()}"
        )

        result = self.client.invoke(
            prompt, tier="medium", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            return news[:5]

        logger.debug(f"News LLM response:\n{result[:500]}")

        # Parse ranked results using shared parser
        parsed = self._parse_ranked_response(result)
        logger.info(f"News parsing: {len(parsed)} items parsed from LLM response")
        ranked_news = []
        for idx, text in parsed:
            if 0 <= idx < len(news):
                article = news[idx].copy()
                article["brief_summary"] = _strip_trailing_rationale(text)
                ranked_news.append(article)

        if ranked_news:
            diversified = self._enforce_source_diversity(ranked_news, max_per_source=self.max_per_source)
            logger.info(f"Ranked and summarized {len(diversified)} news articles")
            return diversified[:5]

        # Retry once with simpler prompt
        logger.warning(f"News ranking parse failed (attempt 1). LLM response: {result[:300]}")
        retry_result = self.client.invoke(
            f"From these articles, pick the 5 most important for an AI researcher. "
            f"Format EXACTLY as: [number] summary sentence.\n\n{articles_block}",
            tier="medium", system_prompt=SYSTEM_PROMPT
        )
        if retry_result:
            parsed_retry = self._parse_ranked_response(retry_result)
            for idx, text in parsed_retry:
                if 0 <= idx < len(news):
                    article = news[idx].copy()
                    article["brief_summary"] = _strip_trailing_rationale(text)
                    ranked_news.append(article)
            if ranked_news:
                diversified = self._enforce_source_diversity(ranked_news, max_per_source=self.max_per_source)
                logger.info(f"Ranked {len(diversified)} news on retry")
                return diversified[:5]

        logger.warning("News ranking failed after retry, using description fallback")
        fallback = []
        for article in news[:5]:
            a = article.copy()
            a["brief_summary"] = a.get("description", a.get("snippet", ""))
            fallback.append(a)
        return fallback

    def rank_and_summarize_happenings(
        self, happenings: List[Dict[str, Any]], max_items: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Rank neighborhood happenings and write concise, date-led summaries.

        Filters the fetched articles down to genuine upcoming events around
        Pacific Beach / Mission Beach / Crown Point / Kate Sessions Park and
        writes short blurbs that lead with the date/venue where available.

        Args:
            happenings: List of event/happening article dictionaries.
            max_items: Maximum number of happenings to return.

        Returns:
            Top happenings, ranked and summarized.
        """
        if not self.available or not happenings:
            fallback = []
            for article in happenings[:max_items]:
                a = article.copy()
                a["brief_summary"] = a.get("description", a.get("snippet", ""))
                fallback.append(a)
            return fallback

        lines = []
        for i, article in enumerate(happenings[:20]):
            title = _sanitize_prompt_input(article.get("title", ""), max_length=300)
            source = _sanitize_prompt_input(article.get("source", ""), max_length=100)
            snippet = _sanitize_prompt_input(
                article.get("description", article.get("snippet", ""))[:250], max_length=300
            )
            lines.append(f"[{i+1}] {title} ({source}): {snippet}")

        articles_block = "\n".join(lines)

        # Inject today's date so the LLM doesn't hallucinate one and can tell
        # which events have already happened.
        from datetime import datetime as _dt
        _today_str = _dt.now().strftime("%B %d, %Y")

        prompt = (
            "You curate an 'upcoming happenings' section for a resident of "
            "Pacific Beach, San Diego (near Crown Point and Kate Sessions Park, "
            "covering the Pacific Beach / Mission Beach area). "
            f"Today's date is {_today_str}.\n"
            "From these articles, select the TOP "
            f"{max_items} that represent genuine upcoming events or community "
            "happenings nearby -- festivals, concerts, farmers markets, beach "
            "cleanups, park events, planning/community meetings, restaurant "
            "openings, etc. DROP pure policy news, crime stories, and anything "
            "not in or near the beach communities. DROP any item whose event "
            f"date has already passed as of {_today_str} -- never describe a "
            "past event as upcoming. If an item's date is unstated, keep it "
            "only if it is plausibly still upcoming (e.g. an ongoing exhibit "
            "or a recurring weekly market); otherwise drop it.\n\n"
            f"<articles>\n{articles_block}\n</articles>\n\n"
            "For each pick, respond in this exact format:\n"
            "[original_number] 1-2 sentence summary.\n\n"
            "Return only the numbered picks in that exact format -- never "
            "explain, list, or justify which items you left out.\n\n"
            "In each summary, LEAD WITH the concrete details: what it is, when "
            "(date/time if mentioned), and where (venue/neighborhood if "
            "mentioned), then the 'so what' -- why it's worth the resident's "
            "attention. Active voice, specific names, no hype, no filler. "
            "Do not invent details. If an item has no clear date/venue and is "
            "not clearly in the area, drop it."
            f"{self._actionability_note()}"
        )

        result = self.client.invoke(
            prompt, tier="medium", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            fallback = []
            for article in happenings[:max_items]:
                a = article.copy()
                a["brief_summary"] = a.get("description", a.get("snippet", ""))
                fallback.append(a)
            return fallback

        logger.debug(f"Happenings LLM response:\n{result[:500]}")
        parsed = self._parse_ranked_response(result)
        ranked = []
        for idx, text in parsed:
            if 0 <= idx < len(happenings):
                article = happenings[idx].copy()
                article["brief_summary"] = _strip_trailing_rationale(text)
                ranked.append(article)

        if ranked:
            diversified = self._enforce_source_diversity(ranked, max_per_source=self.max_per_source)
            logger.info(f"Ranked and summarized {len(diversified)} happenings")
            return diversified[:max_items]

        logger.warning("Happenings ranking parse failed, using description fallback")
        fallback = []
        for article in happenings[:max_items]:
            a = article.copy()
            a["brief_summary"] = a.get("description", a.get("snippet", ""))
            fallback.append(a)
        return fallback

    def rank_and_summarize_blogs(
        self, blogs: List[Dict[str, Any]], topics: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Rank blogs by relevance and generate 1-2 sentence summaries for top items.

        Uses the light tier model to save cost.

        Args:
            blogs: List of blog article dictionaries.
            topics: User's research topics for relevance ranking.

        Returns:
            Top 5 blog articles, ranked and summarized.
        """
        if not self.available or not blogs:
            return blogs[:5]

        blog_lines = []
        for i, article in enumerate(blogs[:15]):
            title = _sanitize_prompt_input(article.get("title", ""), max_length=300)
            source = _sanitize_prompt_input(article.get("source", ""), max_length=100)
            summary = _sanitize_prompt_input(article.get("summary", "")[:200], max_length=300)
            blog_lines.append(f"[{i+1}] {title} ({source}): {summary}")

        blogs_block = "\n".join(blog_lines)
        prompt = (
            f"You are curating a daily {self.briefing_domain} briefing. From these "
            f"blog posts, select the TOP 5 most relevant for {self.briefing_audience}.\n\n"
            f"{self._priority_block()}"
            f"<interests>{', '.join(self._ranking_interests(topics))}</interests>\n\n"
            f"<blogs>\n{blogs_block}\n</blogs>\n\n"
            "For each of your top 5 picks, respond in this exact format:\n"
            "[original_number] SCORE:X/5 1-2 sentence summary.\n\n"
            "Return only the numbered picks in that exact format -- never "
            "explain, list, or justify which posts you left out.\n\n"
            "In the summary, name the specific thing the post is about and state "
            "what it concretely claims, builds, or shows -- the actual takeaway a "
            "reader would quote. Lead with the subject (the product, method, or "
            "finding), not an abstraction.\n"
            "BANNED: hollow 'X is critical/essential/important for Y' framings and "
            "any sentence that would read the same for a dozen different posts. If "
            "the provided text is too thin to say anything specific, summarize the "
            "headline plainly rather than inflating it into a generic principle.\n"
            "Specific, plainspoken, active voice.\n"
            "SCORE is a combined rating (1-5) of impact, complexity, and innovation. "
            "5 = groundbreaking, 1 = routine.\n"
            "Rank by relevance. Be concise."
            f"{self._actionability_note()}"
        )

        result = self.client.invoke(
            prompt, tier="medium", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            return blogs[:5]

        # Parse ranked results using shared parser
        parsed = self._parse_ranked_response(result)
        ranked_blogs = []
        for idx, text in parsed:
            if 0 <= idx < len(blogs):
                article = blogs[idx].copy()
                score, summary = self.extract_score(text)
                article["brief_summary"] = _strip_trailing_rationale(summary)
                if score:
                    article["score_combined"] = score
                if article["brief_summary"]:
                    ranked_blogs.append(article)

        if ranked_blogs:
            # Enforce source diversity: max 2 per source
            diversified = self._enforce_source_diversity(ranked_blogs, max_per_source=self.max_per_source)
            logger.info(f"Ranked and summarized {len(diversified)} blog articles")
            return diversified[:5]

        return self._enforce_source_diversity(blogs, max_per_source=self.max_per_source)[:5]

    @staticmethod
    def _enforce_source_diversity(
        items: List[Dict[str, Any]], max_per_source: int = 2
    ) -> List[Dict[str, Any]]:
        """Cap items per source to ensure diversity. Overflow items are dropped."""
        source_count: Dict[str, int] = {}
        result = []
        for item in items:
            source = item.get("source", "unknown")
            count = source_count.get(source, 0)
            if count < max_per_source:
                result.append(item)
                source_count[source] = count + 1
        return result

    def correlate_stocks_and_news(
        self,
        stocks: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Correlate stock movements with news headlines.

        Uses the medium tier model.

        Args:
            stocks: Stock data with price changes.
            news: News articles.

        Returns:
            Stocks with added 'news_correlation' key.
        """
        if not self.available or not stocks or not news:
            return stocks

        stock_lines = []
        for s in stocks:
            if "error" in s:
                continue
            symbol = s.get("symbol", "")
            name = s.get("name", symbol)
            pct = s.get("percent_change", 0)
            sign = "+" if pct >= 0 else ""
            stock_lines.append(f"{name} ({symbol}): {sign}{pct:.1f}%")

        if not stock_lines:
            return stocks

        news_lines = [
            f"- {_sanitize_prompt_input(n.get('title', ''), max_length=300)}" for n in news[:15]
        ]

        stocks_block = "\n".join(stock_lines)
        headlines_block = "\n".join(news_lines)
        prompt = (
            "These stocks moved today:\n"
            f"<stocks>\n{stocks_block}\n</stocks>\n\n"
            "Today's headlines:\n"
            f"<headlines>\n{headlines_block}\n</headlines>\n\n"
            "For EVERY stock, write a short driver (max 4 words). "
            "Use the headlines if related, otherwise use general market context "
            "(e.g. 'Broad tech selloff', 'Sector rotation').\n"
            "Respond with one line per stock:\n"
            "SYMBOL | short driver\n"
            "Every stock MUST have a driver. Never leave blank."
        )

        result = self.client.invoke(
            prompt, tier="heavy", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            return stocks

        # Parse correlations
        correlations = {}
        for line in result.strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 1)
                symbol = parts[0].strip().upper()
                correlation = parts[1].strip()
                if correlation and correlation.lower() != "no clear driver":
                    correlations[symbol] = correlation

        for stock in stocks:
            symbol = stock.get("symbol", "")
            if symbol in correlations:
                stock["news_correlation"] = correlations[symbol]

        logger.info(f"Correlated {len(correlations)} stocks with news")
        return stocks

    def detect_emerging_themes(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify emerging themes across today's content not in configured topics.

        Args:
            papers: ArXiv papers.
            blogs: Blog articles.
            news: News articles.

        Returns:
            List of emerging theme descriptions (may be empty).
        """
        if not self.available:
            return []

        titles = []
        for p in papers[:15]:
            titles.append(f"[paper] {p.get('title', '')}")
        for b in blogs[:10]:
            titles.append(f"[blog] {b.get('title', '')}")
        for n in news[:10]:
            titles.append(f"[news] {n.get('title', '')}")

        if not titles:
            return []

        titles_block = "\n".join(titles)
        topics_str = ", ".join(self.topics)
        prompt = (
            "Given today's papers, blogs, and news, identify 2-3 emerging themes "
            "or trends that are NOT already covered by these configured topics:\n\n"
            f"<configured_topics>{topics_str}</configured_topics>\n\n"
            f"<content>\n{titles_block}\n</content>\n\n"
            "For each theme, write one line: THEME: brief description\n"
            "Only list genuinely new/emerging themes. If nothing stands out, "
            "respond with NONE."
        )

        result = self.client.invoke(prompt, tier="heavy")
        if not result or "NONE" in result.upper():
            return []

        themes = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("THEME:"):
                themes.append(line[6:].strip())

        if themes:
            logger.info(f"Detected emerging themes: {themes}")
        return themes

    def synthesize_briefing(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        stocks: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        top_papers: List[Dict[str, Any]],
        emerging_themes: Optional[List[str]] = None,
        previous_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Synthesize cross-section connections and generate editorial content.

        Uses the heavy tier model for deep reasoning.

        Args:
            papers: ArXiv papers (full list for context).
            blogs: Blog articles.
            stocks: Stock data.
            news: News articles.
            top_papers: Top-scored papers.
            emerging_themes: Emerging themes detected from today's content.
            previous_state: Previous briefing state for trend tracking.

        Returns:
            Dictionary with key:
              - 'editorial_intro': Executive summary paragraph for the briefing.
        """
        if not self.available:
            return {}

        # Build a compact summary of all data for the synthesis prompt
        sections = []

        if papers:
            paper_items = []
            for p in papers[:10]:
                title = p.get("title", "")
                summary = p.get("brief_summary", p.get("ai_summary", ""))
                if summary:
                    paper_items.append(f"- {title}: {summary}")
                else:
                    paper_items.append(f"- {title}")
            sections.append(
                "PAPERS (" + str(len(papers)) + " total):\n"
                + "\n".join(paper_items)
            )

        if blogs:
            blog_items = []
            for b in blogs[:8]:
                source = b.get("source", "")
                title = b.get("title", "")
                summary = b.get("brief_summary", "")
                if summary:
                    blog_items.append(f"- [{source}] {title}: {summary}")
                else:
                    blog_items.append(f"- [{source}] {title}")
            sections.append("BLOGS:\n" + "\n".join(blog_items))

        if stocks:
            stock_items = []
            for s in stocks:
                if "error" not in s:
                    pct = s.get("percent_change", 0)
                    sign = "+" if pct >= 0 else ""
                    corr = s.get("news_correlation", "")
                    line = f"- {s.get('symbol', '')}: {sign}{pct:.1f}%"
                    if corr:
                        line += f" ({corr})"
                    stock_items.append(line)
            if stock_items:
                sections.append("STOCKS:\n" + "\n".join(stock_items))

        if news:
            news_items = []
            for n in news[:10]:
                title = n.get("title", "")
                summary = n.get("brief_summary", "")
                if summary:
                    news_items.append(f"- {title}: {summary}")
                else:
                    news_items.append(f"- {title}")
            sections.append("NEWS:\n" + "\n".join(news_items))

        if top_papers:
            top_items = []
            for p in top_papers:
                reason = p.get("relevance_reason", "")
                top_items.append(
                    f"- {p.get('title', '')} (score: {p.get('score', 0):.1f})"
                    + (f" -- {reason}" if reason else "")
                )
            sections.append("TOP PAPERS FOR REPRODUCTION:\n" + "\n".join(top_items))

        if emerging_themes:
            sections.append(
                "EMERGING THEMES (not in configured topics):\n"
                + "\n".join(f"- {t}" for t in emerging_themes)
            )

        if previous_state:
            prev_parts = []
            prev_date = previous_state.get("date", "unknown")
            prev_parts.append(f"PREVIOUS BRIEFING ({prev_date}):")
            prev_themes = previous_state.get("emerging_themes", [])
            if prev_themes:
                prev_parts.append(f"Themes: {', '.join(prev_themes)}")
            prev_stocks = previous_state.get("stock_closes", {})
            if prev_stocks and stocks:
                trend_lines = []
                for s in stocks:
                    sym = s.get("symbol", "")
                    if sym in prev_stocks and "error" not in s:
                        prev_price = prev_stocks[sym]
                        curr_price = s.get("current_price", 0)
                        if prev_price and prev_price > 0:
                            multi_day_pct = ((curr_price - prev_price) / prev_price) * 100
                            trend_lines.append(f"{sym}: {multi_day_pct:+.1f}% over 2 days")
                if trend_lines:
                    prev_parts.append("Multi-day trends: " + "; ".join(trend_lines))
            if len(prev_parts) > 1:
                sections.append("\n".join(prev_parts))

        if not sections:
            return {}

        all_data = "\n\n".join(sections)

        # Detect cross-source correlations
        cross_source_signals = self._detect_cross_source_signals(papers, blogs, news)
        cross_source_note = ""
        if cross_source_signals:
            cross_source_note = (
                "\n\n<cross_source_signals>\n"
                "These topics appear in 2+ sources (PRIORITIZE in summary):\n"
                + "\n".join(f"- {s}" for s in cross_source_signals)
                + "\n</cross_source_signals>"
            )

        # Inject today's date so the LLM doesn't hallucinate one from training-cutoff cues.
        from datetime import datetime as _dt
        _today_str = _dt.now().strftime("%B %d, %Y")

        prompt = (
            "You are writing the lead editorial -- the Executive Summary -- for "
            f"today's {self.briefing_domain} intelligence briefing. "
            f"The date is {_today_str}. "
            "This is "
            "the one section every reader reads, so it must earn an A+ in a senior "
            "journalism seminar.\n\n"
            "Your job is NOT to list what happened. It is to CONNECT THE DOTS: "
            "find the single most important through-line across today's data and "
            "tell the reader what it means and why it matters now.\n\n"
            "Write the Executive Summary in a format optimized for quick "
            "scanning — a reader glancing at it should absorb ~80% of the "
            "depth in 5 seconds:\n\n"
            "STRUCTURE:\n"
            "- **Bold a one-sentence headline** whose GRAMMATICAL SUBJECT is "
            f"{self.briefing_audience} — or that reader's plan, assumptions, "
            "budget, or routine. An organization, a product, a market move, or "
            "a trend must NOT be the subject; name the development later in "
            "the sentence, as the reason. State what the reader should now do "
            "differently.\n"
            "  Shape that FAILS — '<Organization> did <thing>, which signals "
            "<trend>.' The reader appears nowhere.\n"
            "  Shape that FAILS — '<Trend> is now <state of the world>.' True, "
            "but it is a thesis about the field, not about the reader.\n"
            "  Shape that PASSES — 'Anyone still <doing X> should <do Y> "
            "<by when> — <development> just <changed the constraint>.'\n"
            "  Two tests before you commit to it: (1) could this sentence "
            "appear verbatim in the press release of whoever is mentioned? If "
            "yes, rewrite. (2) Does a reader learn something to DO, not just "
            "something that HAPPENED? If no, rewrite. "
            "This is the glanceable takeaway. "
            "No throat-clearing, no 'Today's briefing covers...'.\n"
            "- 3-5 short paragraphs (1-2 sentences each) separated by blank "
            "lines. Develop the through-line, connect sources, deliver analysis.\n"
            "- **Bold key tickers, numbers, and entities** to create visual "
            "anchors for skimming (e.g., **NVDA -2.2%**, **$10B**, **Kimi K3**).\n"
            "- At most one brief bullet list (2-3 items) if it genuinely helps "
            "compare or contrast signals — otherwise stick to short paragraphs.\n"
            "- End with **Watch:** or **The bottom line:** in bold, followed by "
            "the single call-out worth tracking.\n\n"
            "Voice: plain, vigorous, active. Concrete nouns and numbers."
            f"{self._actionability_note()}"
            "\n\n"
            "GROUND RULES:\n"
            "- Every specific fact, number, ticker, company name, product name, "
            "or personnel name you state must appear verbatim in the <data> "
            "below. Paraphrase the analysis, never invent specifics.\n"
            "- If a concrete figure is not in <data>, describe the trend "
            "qualitatively. A vaguer true sentence beats a precise invented one.\n"
            f"- Today's date is exactly {_today_str} — do not infer a different "
            "date if you reference one.\n"
            "- Never output your internal reasoning, verification, or grounding "
            "steps; output only the final editorial text.\n\n"
            "IMPORTANT: Topics in <cross_source_signals> appear in multiple "
            "independent sources — treat them as the strongest signal and build "
            "your through-line around them where they fit. If emerging themes or "
            "multi-day trends are present, work them in.\n\n"
            f"{self._priority_block()}"
            f"<data>\n{all_data}\n</data>"
            f"{cross_source_note}"
        )

        result = self.client.invoke(
            prompt,
            tier="heavy",
            system_prompt=SYSTEM_PROMPT,
            reasoning_enabled=True,
        )
        if result and is_cot_leak(result):
            logger.warning(
                "Editorial synthesis leaked CoT scaffolding; "
                "retrying without reasoning."
            )
            result = self.client.invoke(
                prompt,
                tier="heavy",
                system_prompt=SYSTEM_PROMPT,
                reasoning_enabled=False,
            )
            if not result or is_cot_leak(result):
                result = None
        if not result:
            return {}

        logger.info("Briefing synthesis complete")
        return {"editorial_intro": result.strip()}

    def generate_solo_startup_angle(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        top_papers: List[Dict[str, Any]],
        emerging_themes: Optional[List[str]] = None,
    ) -> str:
        """Generate one concrete solo-founder (1-man company) startup angle
        based on today's signals. Inspired by Pieter Levels / Daniel Vassallo style:
        small, focused, AI-native, no-team, ship-fast, paid-from-day-one.

        Returns markdown string (the angle), or empty string when unavailable.
        """
        if not self.available:
            return ""

        sections = []
        if top_papers:
            sections.append(
                "TOP PAPERS:\n"
                + "\n".join(
                    f"- {p.get('title', '')}" for p in top_papers[:3]
                )
            )
        if papers:
            sections.append(
                "OTHER PAPERS:\n"
                + "\n".join(f"- {p.get('title', '')}" for p in papers[:6])
            )
        if blogs:
            sections.append(
                "BLOGS:\n"
                + "\n".join(
                    f"- [{b.get('source', '')}] {b.get('title', '')}"
                    for b in blogs[:6]
                )
            )
        if news:
            sections.append(
                "NEWS:\n"
                + "\n".join(f"- {n.get('title', '')}" for n in news[:6])
            )
        if emerging_themes:
            sections.append(
                "EMERGING THEMES:\n"
                + "\n".join(f"- {t}" for t in emerging_themes)
            )

        if not sections:
            return ""

        all_data = "\n\n".join(sections)

        prompt = (
            "You are a startup scout for a solo founder (1-man company) in the "
            "style of Pieter Levels (Nomad List, Photo AI), Daniel Vassallo, "
            "and the IndieHackers community. The founder is a senior AWS "
            "principal engineer with strong backend / AI infra chops, no team, "
            "and limited free hours per week. They want to build small, "
            "focused, AI-native products that can be shipped in 2-6 weeks and "
            "reach paying customers fast ($500-$10K MRR is great, no VC needed).\n\n"
            "Based ONLY on today's signals below, propose ONE concrete solo "
            "startup idea. Use this exact markdown structure (be terse, no fluff):\n\n"
            "**Product:** <one sentence — what it does>\n"
            "**Who pays:** <ICP — be specific about who and why they pay>\n"
            "**Signal today:** <which paper/blog/news item triggered this and why>\n"
            "**Wedge / unfair advantage:** <why a solo dev can win this niche>\n"
            "**MVP in 2-4 weeks:** <3-5 bullets, concrete tech choices>\n"
            "**Distribution:** <where/how to get the first 10 paying customers>\n"
            "**Pricing:** <starting price, e.g. $19/mo, $99 one-time, etc.>\n"
            "**Risk / why it might fail:** <one honest sentence>\n\n"
            "Rules:\n"
            "- Must be buildable solo. No 'platform', no 'marketplace requiring "
            "liquidity', no enterprise sales cycle.\n"
            "- Must reference TODAY's signals; don't propose generic ideas.\n"
            "- Prefer wedges with painful, niche, willing-to-pay buyers.\n"
            "- Boring beats clever. Distribution beats novelty.\n\n"
            f"<signals>\n{all_data}\n</signals>"
        )

        result = self.client.invoke(
            prompt, tier="heavy", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            return ""
        logger.info("Solo-founder startup angle generated")
        return result.strip()

    def generate_agent_cost_optimization(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        top_papers: List[Dict[str, Any]],
        emerging_themes: Optional[List[str]] = None,
    ) -> str:
        """Generate one concrete agent cost-optimization play grounded in today's signals.
        Audience: AWS principal engineer running agents on Bedrock / Trainium / Inferentia,
        thinking about $/session, latency, throughput, and how to actually move the number.
        Returns markdown (the play), or empty string when unavailable.
        """
        if not self.available:
            return ""

        sections = []
        if top_papers:
            sections.append(
                "TOP PAPERS:\n"
                + "\n".join(
                    f"- {p.get('title', '')}" for p in top_papers[:3]
                )
            )
        if papers:
            sections.append(
                "OTHER PAPERS:\n"
                + "\n".join(f"- {p.get('title', '')}" for p in papers[:6])
            )
        if blogs:
            sections.append(
                "BLOGS:\n"
                + "\n".join(
                    f"- [{b.get('source', '')}] {b.get('title', '')}"
                    for b in blogs[:6]
                )
            )
        if news:
            sections.append(
                "NEWS:\n"
                + "\n".join(f"- {n.get('title', '')}" for n in news[:6])
            )
        if emerging_themes:
            sections.append(
                "EMERGING THEMES:\n"
                + "\n".join(f"- {t}" for t in emerging_themes)
            )

        if not sections:
            return ""

        all_data = "\n\n".join(sections)

        prompt = (
            "You are an AWS principal engineer focused on agent cost "
            "optimization. The reader runs LLM agents on AWS (Bedrock, "
            "SageMaker, Trainium / Inferentia, Neuron SDK) and cares about "
            "$/session, latency P50/P99, throughput, and total monthly spend. "
            "They want ONE concrete cost-optimization play per day, grounded "
            "in today's signals, that they could actually try this week.\n\n"
            "Use this exact markdown structure (terse, no fluff):\n\n"
            "**Play:** <one sentence — the specific tactic>\n"
            "**Signal today:** <which paper/blog/news triggered this and why>\n"
            "**Mechanism:** <how it reduces cost: caching, routing, distillation, "
            "context compression, batching, KV reuse, speculative decoding, "
            "smaller model, tool reduction, etc.>\n"
            "**Estimated impact:** <concrete % or $ range — e.g. '30-50% fewer "
            "input tokens', '$0.012 → $0.003 per session', '2x throughput on "
            "trn1.2xlarge'. Be honest about uncertainty.>\n"
            "**AWS-specific angle:** <Bedrock prompt caching, Trainium NKI, "
            "Inferentia, SageMaker batch, etc. — what to actually use>\n"
            "**Try this week:** <3-5 bullets, concrete steps, time estimate>\n"
            "**Watch-out:** <one honest sentence on where the savings might "
            "not materialize — e.g. cold-cache, quality regression, hidden cost>\n\n"
            "Rules:\n"
            "- Reference TODAY's signals; don't propose generic AWS Well-Architected fluff.\n"
            "- Prefer plays with measurable $/% impact over architectural opinions.\n"
            "- Be specific about model IDs, instance types, or AWS services when relevant.\n"
            "- If today's signals don't suggest a strong play, say so honestly and "
            "propose the smallest useful experiment.\n\n"
            f"<signals>\n{all_data}\n</signals>"
        )

        result = self.client.invoke(
            prompt, tier="heavy", system_prompt=SYSTEM_PROMPT
        )
        if not result:
            return ""
        logger.info("Agent cost-optimization play generated")
        return result.strip()

    def track_trending(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Track topics that appear across multiple days and mark trending items.

        Uses Opus to cluster current items against stored trending topics.
        If a topic reappears on Day 2 or Day 3, increment its counter and mark items
        with "🔥 Day N trending" in their summary.

        Args:
            papers: Today's papers.
            blogs: Today's blogs.
            news: Today's news.
            state: Previous state with trending_topics.

        Returns:
            Tuple of (updated_state, annotated_papers, annotated_blogs, annotated_news).
        """
        if not self.available:
            return state, papers, blogs, news

        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")
        trending_topics = state.get("trending_topics", {})

        # Build list of current items
        current_items = []
        for p in papers[:10]:
            current_items.append(f"[paper] {p.get('title', '')}")
        for b in blogs[:10]:
            current_items.append(f"[blog] {b.get('title', '')}")
        for n in news[:10]:
            current_items.append(f"[news] {n.get('title', '')}")

        if not current_items:
            return state, papers, blogs, news

        # Build trending topics summary
        trending_summary = []
        for topic_key, info in trending_topics.items():
            count = info.get("count", 1)
            first_seen = info.get("first_seen", "")
            last_seen = info.get("last_seen", "")
            trending_summary.append(f"- {topic_key}: count={count}, first={first_seen}, last={last_seen}")

        items_block = "\n".join(current_items)
        trending_block = "\n".join(trending_summary) if trending_summary else "NONE"

        prompt = (
            f"Today is {today}. You are tracking trending topics across days.\n\n"
            "<current_items>\n"
            f"{items_block}\n"
            "</current_items>\n\n"
            "<previous_trending_topics>\n"
            f"{trending_block}\n"
            "</previous_trending_topics>\n\n"
            "For each current item, determine if it matches or is closely related to a previous trending topic. "
            "If it matches, output: [item_index] MATCH topic_key\n"
            "If it's a NEW emerging topic appearing 2+ times today, output: [item_index] NEW topic_keyword\n"
            "If it's neither, skip it.\n\n"
            "Example output:\n"
            "[2] MATCH flash-attention-4\n"
            "[5] NEW claude-3.5-haiku\n"
        )

        result = self.client.invoke(prompt, tier="light", system_prompt=SYSTEM_PROMPT)
        if not result:
            logger.info("Trending tracking skipped (LLM unavailable)")
            return state, papers, blogs, news

        # Parse the result
        matches = {}
        new_topics = {}
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            try:
                bracket_end = line.index("]")
                idx = int(line[1:bracket_end]) - 1
                rest = line[bracket_end + 1:].strip()
                if rest.startswith("MATCH"):
                    topic_key = rest.split("MATCH", 1)[1].strip()
                    matches[idx] = topic_key
                elif rest.startswith("NEW"):
                    topic_key = rest.split("NEW", 1)[1].strip()
                    new_topics[idx] = topic_key
            except (ValueError, IndexError):
                continue

        # Update trending_topics and annotate items
        updated_trending = trending_topics.copy()
        annotated_count = 0

        # Process matches
        for idx, topic_key in matches.items():
            if topic_key in updated_trending:
                updated_trending[topic_key]["count"] += 1
                updated_trending[topic_key]["last_seen"] = today
                day_count = updated_trending[topic_key]["count"]

                # Annotate the item
                if idx < len(current_items):
                    item_type = current_items[idx].split("]")[0][1:]  # Extract type: paper/blog/news
                    if item_type == "paper" and idx < len(papers):
                        orig_summary = papers[idx].get("brief_summary", "")
                        # Annotate as trending but DON'T inject "Day N" into summary text
                        # (user found it confusing in final output)
                        papers[idx]["_trending_days"] = day_count
                        annotated_count += 1
                    elif item_type == "blog":
                        blog_idx = idx - len([i for i in current_items[:idx] if "[paper]" in i])
                        if 0 <= blog_idx < len(blogs):
                            orig_summary = blogs[blog_idx].get("brief_summary", "")
                            blogs[blog_idx]["_trending_days"] = day_count
                            annotated_count += 1
                    elif item_type == "news":
                        news_idx = idx - len([i for i in current_items[:idx] if "[paper]" in i or "[blog]" in i])
                        if 0 <= news_idx < len(news):
                            orig_summary = news[news_idx].get("brief_summary", "")
                            news[news_idx]["_trending_days"] = day_count
                            annotated_count += 1

        # Process new topics
        for idx, topic_key in new_topics.items():
            if topic_key not in updated_trending:
                updated_trending[topic_key] = {
                    "first_seen": today,
                    "count": 1,
                    "last_seen": today,
                }

        # Clean up old trending topics (older than 3 days)
        from datetime import datetime, timedelta
        today_date = datetime.strptime(today, "%Y-%m-%d")
        cleaned_trending = {}
        for topic_key, info in updated_trending.items():
            last_seen = datetime.strptime(info["last_seen"], "%Y-%m-%d")
            if (today_date - last_seen).days <= 3:
                cleaned_trending[topic_key] = info

        state["trending_topics"] = cleaned_trending
        logger.info(f"Trending tracking: {annotated_count} items marked, {len(cleaned_trending)} topics tracked")
        return state, papers, blogs, news

    def detect_entity_mentions(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        tracked_entities: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Detect mentions of tracked entities (companies/people) in content.

        Args:
            papers: Today's papers.
            blogs: Today's blogs.
            news: Today's news.
            tracked_entities: List of entities to track with name and type.

        Returns:
            List of entity mention dicts with name, type, count, and example_titles.
        """
        if not tracked_entities:
            return []

        entity_mentions = {}

        # Scan all items for entity mentions
        all_items = []
        for p in papers:
            all_items.append({
                "title": p.get("title", ""),
                "summary": p.get("brief_summary", "") or p.get("summary", "")[:200],
                "type": "paper",
            })
        for b in blogs:
            all_items.append({
                "title": b.get("title", ""),
                "summary": b.get("brief_summary", "") or b.get("summary", "")[:200],
                "type": "blog",
            })
        for n in news:
            all_items.append({
                "title": n.get("title", ""),
                "summary": n.get("brief_summary", "") or n.get("description", "")[:200],
                "type": "news",
            })

        # Case-insensitive substring matching
        for entity in tracked_entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "")
            if not entity_name:
                continue

            entity_name_lower = entity_name.lower()
            matches = []

            for item in all_items:
                title = item.get("title", "").lower()
                summary = item.get("summary", "").lower()

                if entity_name_lower in title or entity_name_lower in summary:
                    matches.append(item.get("title", ""))

            if matches:
                entity_mentions[entity_name] = {
                    "name": entity_name,
                    "type": entity_type,
                    "count": len(matches),
                    "example_titles": matches[:3],  # Keep top 3 examples
                }

        # Convert to list and sort by count (descending)
        result = list(entity_mentions.values())
        result.sort(key=lambda x: x["count"], reverse=True)

        if result:
            logger.info(f"Entity Watch: detected {len(result)} entities with mentions")
        return result

    def generate_weekly_deep_dive(
        self,
        weekly_items: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a 'This Week in AI' deep dive section.

        Uses Opus (heavy tier) to synthesize a narrative from the week's items,
        identifying the 3 biggest themes, explaining why they matter, and predicting
        what to watch next week.

        Args:
            weekly_items: List of items accumulated over the week with date.

        Returns:
            Markdown string for the "This Week in AI" section (500-800 words).
        """
        if not self.available or not weekly_items:
            return ""

        # Group items by date
        items_by_date = {}
        for item in weekly_items:
            date = item.get("date", "unknown")
            if date not in items_by_date:
                items_by_date[date] = []
            items_by_date[date].append(item)

        # Build context from weekly items
        context_parts = []
        for date in sorted(items_by_date.keys()):
            items = items_by_date[date]
            titles = [f"- {i.get('title', '')} ({i.get('type', 'item')})" for i in items]
            context_parts.append(f"{date}:\n" + "\n".join(titles))

        if not context_parts:
            return ""

        context_str = "\n\n".join(context_parts)

        prompt = (
            "You are writing the 'This Week in AI' essay for a weekly intelligence "
            "briefing -- the marquee long-form piece, held to A+ senior-journalism "
            "standards. Based on this week's papers, blogs, and news below, do not "
            "summarize -- SYNTHESIZE. Connect the dots across the week into a real "
            "argument with a point of view.\n\n"
            "Your essay should:\n"
            "1. Identify the 3 biggest themes of the week and name the through-line "
            "that links them.\n"
            f"2. Explain why each matters -- the concrete implications for "
            f"readers and {self.briefing_landscape}.\n"
            "3. Call your shot: predict what to watch next week, and what would "
            "confirm or kill the trend.\n\n"
            "Write 500-800 words. Open with a strong lede that frames the week -- "
            "no 'This week saw...'. Be analytical, opinionated, and "
            "forward-looking. Plain, vigorous, active-voice prose; insight over "
            "erudition.\n\n"
            "GROUNDING (critical): the items below are headlines/titles ONLY -- "
            "you do not have the article bodies. Build your argument strictly from "
            "what these titles state. Do NOT invent or recall specific dollar "
            "figures, funding amounts, percentages, contract values, product "
            "names, or company names that do not appear verbatim in the titles "
            "below; fabricated precision (e.g. an '$80B' or '1,775%' that is not "
            "shown) is the worst failure mode here. When a title implies a trend "
            "but gives no number, describe it qualitatively. A vaguer true "
            "sentence beats a precise invented one.\n\n"
            f"<week_items>\n{context_str}\n</week_items>"
        )

        result = self.client.invoke(
            prompt, tier="heavy", system_prompt=SYSTEM_PROMPT
        )
        if result and is_cot_leak(result):
            logger.warning(
                "Weekly Deep Dive leaked CoT scaffolding; "
                "retrying without reasoning."
            )
            result = self.client.invoke(
                prompt,
                tier="heavy",
                system_prompt=SYSTEM_PROMPT,
                reasoning_enabled=False,
            )
            if not result or is_cot_leak(result):
                result = None

        if result:
            logger.info("Weekly Deep Dive generated successfully")
            return result.strip()

        logger.warning("Weekly Deep Dive generation failed")
        return ""

    def _detect_cross_source_signals(
        self,
        papers: List[Dict[str, Any]],
        blogs: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Detect topics that appear in 2+ sources (papers, blogs, news).

        Uses simple keyword matching to find cross-source correlations.

        Args:
            papers: ArXiv papers.
            blogs: Blog articles.
            news: News articles.

        Returns:
            List of cross-source topics (e.g., "Claude 3.5", "Trainium", etc.)
        """
        # Extract key terms from each source
        def extract_terms(items: List[Dict[str, Any]], key: str = "title") -> set:
            """Extract significant terms (2+ words) from titles."""
            terms = set()
            for item in items[:15]:  # Top 15 items per source
                text = item.get(key, "").lower()
                # Extract multi-word phrases (simple approach)
                words = text.split()
                # Look for capitalized phrases and specific keywords
                for i in range(len(words)):
                    # 2-word phrases
                    if i + 1 < len(words):
                        phrase = f"{words[i]} {words[i+1]}"
                        if len(phrase) > 6:  # Filter short phrases
                            terms.add(phrase)
                    # 3-word phrases
                    if i + 2 < len(words):
                        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                        if len(phrase) > 10:
                            terms.add(phrase)
            return terms

        paper_terms = extract_terms(papers[:15])
        blog_terms = extract_terms(blogs[:10])
        news_terms = extract_terms(news[:10])

        # Find terms appearing in 2+ sources
        cross_source = []
        all_sources = [
            ("papers", paper_terms),
            ("blogs", blog_terms),
            ("news", news_terms),
        ]

        checked_terms = set()
        for source_name, terms in all_sources:
            for term in terms:
                if term in checked_terms:
                    continue
                checked_terms.add(term)

                # Check if this term appears in other sources
                match_count = 0
                matched_sources = []
                for other_name, other_terms in all_sources:
                    # Use fuzzy matching: check if term is substring of any other term
                    if any(term in other_term or other_term in term for other_term in other_terms):
                        match_count += 1
                        matched_sources.append(other_name)

                if match_count >= 2:
                    cross_source.append(f"{term.title()} ({', '.join(matched_sources)})")

        # Limit to top 5 cross-source signals
        if cross_source:
            logger.info(f"Detected {len(cross_source)} cross-source signals")
        return cross_source[:5]


def _parse_numbered_list(text: str, expected_count: int) -> List[str]:
    """
    Parse a numbered list response from the model.
    Ignores conversational preambles before the first numbered item.

    Args:
        text: Model response text.
        expected_count: Expected number of items.

    Returns:
        List of parsed items.
    """
    items = []
    current_lines = []
    current_num = -1

    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Check for numbered item start: [1], 1., 1)
        new_num = None
        if stripped.startswith("[") and "]" in stripped:
            try:
                bracket_end = stripped.index("]")
                new_num = int(stripped[1:bracket_end])
                stripped = stripped[bracket_end + 1:].strip()
            except (ValueError, IndexError):
                pass
        elif stripped and stripped[0].isdigit():
            for sep in [".", ")", ":"]:
                if sep in stripped[:4]:
                    try:
                        new_num = int(stripped[: stripped.index(sep)])
                        stripped = stripped[stripped.index(sep) + 1:].strip()
                    except (ValueError, IndexError):
                        pass
                    break

        if new_num is not None and new_num != current_num:
            # Only append if we've already started a numbered section
            if current_lines and current_num != -1:
                items.append(" ".join(current_lines))
            current_num = new_num
            current_lines = [stripped] if stripped else []
        else:
            current_lines.append(stripped)

    # Append the last item if it exists and had a number
    if current_lines and current_num != -1:
        items.append(" ".join(current_lines))

    # If we found no numbered items, we might have received one giant block.
    # We'll allow it only if we were expecting exactly one item.
    if not items and expected_count == 1 and current_lines:
        items.append(" ".join(current_lines))

    return items[:expected_count]
