#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Interest taxonomy DAG for dynamic news query generation.

Models the user's interest domains as a tree: root nodes are broad seed
queries that fire every run as a baseline, and leaf nodes are specific
sub-topic queries selected per-run based on signal from yesterday's briefing
and today's blogs, capped by a per-run budget.

The selection is deterministic (term-overlap signal scoring) with an optional
LLM pass that re-ranks the top candidates. When the LLM is unavailable or
there is no signal, only the roots fire.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "over", "under", "that",
    "this", "are", "was", "were", "has", "have", "its", "new", "via", "per",
}


class Query(str):
    """
    A query string that carries its retrieval parameters.

    Subclasses ``str`` so every existing consumer (dedup sets, logging,
    ``NewsAggregator``) keeps working unchanged, while the runner can read the
    ``freshness`` the owning graph node asked for.
    """

    __slots__ = ("freshness", "node_id")

    def __new__(cls, text: str, freshness: Optional[str] = None, node_id: str = ""):
        obj = super().__new__(cls, text)
        obj.freshness = freshness
        obj.node_id = node_id
        return obj


def query_freshness(query: Any, default: str) -> str:
    """Read a query's freshness binding, falling back to the pipeline default."""
    return getattr(query, "freshness", None) or default


@dataclass
class Node:
    """A node in the interest taxonomy tree."""

    id: str
    query: str
    priority: float = 1.0
    # Brave freshness window for this branch ("pd", "pw", "pm"). Inherited from
    # the parent when unset: hyperlocal branches need a wider window than
    # national/market branches, which is a property of the topic, not the run.
    freshness: Optional[str] = None
    children: List["Node"] = field(default_factory=list)
    parent: Optional["Node"] = None

    def is_leaf(self) -> bool:
        return not self.children

    def as_query(self) -> "Query":
        """Bind this node's query text to its retrieval parameters."""
        return Query(self.query, freshness=self.freshness, node_id=self.id)


class InterestGraph:
    """A tree of interest queries with a per-run dynamic-query budget."""

    def __init__(self, roots: List[Node], max_dynamic_queries: int = 3):
        self.roots = roots
        self.max_dynamic_queries = max_dynamic_queries

    def collect_roots(self) -> List[str]:
        return [n.query for n in self.roots]

    def collect_root_queries(self) -> List["Query"]:
        """Root queries bound to their retrieval parameters."""
        return [n.as_query() for n in self.roots]

    def leaves(self) -> List[Node]:
        """
        Dig candidates: leaf nodes below the roots.

        A childless root is excluded -- it already fires every run as a
        baseline query, so selecting it again would burn a dynamic slot and a
        API call on a duplicate.
        """
        out = []
        root_ids = {id(r) for r in self.roots}

        def walk(node: Node) -> None:
            if not node.children:
                if id(node) not in root_ids:
                    out.append(node)
            else:
                for child in node.children:
                    walk(child)

        for root in self.roots:
            walk(root)
        return out

    def all_nodes(self) -> List[Node]:
        out = []

        def walk(node: Node) -> None:
            out.append(node)
            for child in node.children:
                walk(child)

        for root in self.roots:
            walk(root)
        return out


def parse_graph(config: Dict[str, Any]) -> Optional[InterestGraph]:
    graph_cfg = config.get("interest_graph")
    if not graph_cfg:
        return None
    roots_cfg = graph_cfg.get("roots", [])
    if not roots_cfg:
        return None
    max_dynamic = int(graph_cfg.get("max_dynamic_queries", 3))
    roots = [_parse_node(cfg) for cfg in roots_cfg]
    return InterestGraph(roots, max_dynamic)


def _parse_node(cfg: Dict[str, Any], parent: Optional[Node] = None) -> Node:
    # Freshness flows down the tree unless a node overrides it.
    freshness = cfg.get("freshness")
    if freshness is None and parent is not None:
        freshness = parent.freshness
    node = Node(
        id=str(cfg.get("id", "")),
        query=str(cfg.get("query", "")),
        priority=float(cfg.get("priority", 1.0)),
        freshness=str(freshness) if freshness else None,
        parent=parent,
    )
    for child_cfg in cfg.get("children", []) or []:
        node.children.append(_parse_node(child_cfg, parent=node))
    return node


def term_overlap(query: str, corpus: str) -> float:
    if not query or not corpus:
        return 0.0
    terms = [
        t for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) > 2 and t not in _STOPWORDS
    ]
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if t in corpus)
    return hits / len(terms)


def score_signal(
    nodes: List[Node],
    previous_state: Optional[Dict[str, Any]],
    today_blogs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    parts: List[str] = []
    state = previous_state or {}
    for key in ("top_paper_titles", "top_blog_titles", "top_news_titles"):
        parts.extend(state.get(key, []) or [])
    parts.extend(state.get("emerging_themes", []) or [])
    for blog in today_blogs or []:
        title = blog.get("title") if isinstance(blog, dict) else None
        if title:
            parts.append(title)
    corpus = " ".join(parts).lower()
    return {node.id: term_overlap(node.query, corpus) for node in nodes}


def rank_leaves(graph: InterestGraph, signals: Dict[str, float]) -> List[Node]:
    return sorted(
        graph.leaves(),
        key=lambda n: signals.get(n.id, 0.0) * n.priority,
        reverse=True,
    )


def _llm_select(
    candidates: List[Node],
    previous_state: Optional[Dict[str, Any]],
    llm_client: Any,
    max_n: int,
) -> Optional[List[Node]]:
    state = previous_state or {}
    headlines: List[str] = []
    for key in ("top_news_titles", "top_blog_titles", "top_paper_titles"):
        headlines.extend(state.get(key, []) or [])
    candidate_lines = "\n".join(f"- {n.query}" for n in candidates)
    prompt = (
        "You select which specific news queries are most relevant today.\n\n"
        f"<yesterday_headlines>\n{chr(10).join(headlines[:10]) or '(none)'}\n</yesterday_headlines>\n\n"
        f"<candidate_queries>\n{candidate_lines}\n</candidate_queries>\n\n"
        f"Return the {max_n} most relevant queries verbatim from the candidate "
        "list, one per line, no numbering or bullets. Do not invent new queries."
    )
    result = llm_client.invoke(prompt, tier="light")
    if not result:
        return None
    chosen = [
        line.strip().strip("- *")
        for line in result.strip().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    candidate_map = {n.query.lower(): n for n in candidates}
    out = []
    for query in chosen:
        node = candidate_map.get(query.lower())
        if node:
            out.append(node)
    return out[:max_n] or None


def generate_graph_queries(
    graph: InterestGraph,
    previous_state: Optional[Dict[str, Any]],
    today_blogs: Optional[List[Dict[str, Any]]] = None,
    llm_client: Optional[Any] = None,
) -> List[str]:
    roots = graph.collect_root_queries()
    leaves = graph.leaves()
    if not leaves:
        return roots
    signals = score_signal(leaves, previous_state, today_blogs)
    if not any(signals.values()):
        return roots
    ranked = rank_leaves(graph, signals)
    selected = ranked[: graph.max_dynamic_queries]
    if llm_client is not None and getattr(llm_client, "available", False):
        window = ranked[: max(graph.max_dynamic_queries * 3, graph.max_dynamic_queries)]
        chosen = _llm_select(window, previous_state, llm_client, graph.max_dynamic_queries)
        if chosen:
            selected = chosen
    logger.info(
        "Interest graph: %d roots + %d dig queries (%d leaves available)",
        len(roots), len(selected), len(leaves),
    )
    return roots + [n.as_query() for n in selected]
