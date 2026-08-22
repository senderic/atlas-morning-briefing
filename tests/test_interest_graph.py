# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for the interest taxonomy DAG engine (scripts/interest_graph.py)."""

from unittest.mock import MagicMock

import pytest

from scripts.interest_graph import (
    InterestGraph,
    Node,
    generate_graph_queries,
    parse_graph,
    rank_leaves,
    score_signal,
    term_overlap,
)


def _make_graph() -> InterestGraph:
    return InterestGraph(
        roots=[
            Node(
                "r1", "defense ai contracts",
                children=[
                    Node("palantir", "Palantir AIP defense implementation", priority=1.0),
                    Node("lunar", "Lunar mining extraction", priority=1.0),
                    Node("benchmarks", "AI model benchmarks", priority=1.0),
                ],
            ),
            Node(
                "r2", "space infrastructure",
                children=[
                    Node("rocket", "Rocket Lab orbital services", priority=1.0),
                ],
            ),
        ],
        max_dynamic_queries=2,
    )


def _config_with_graph() -> dict:
    return {
        "interest_graph": {
            "max_dynamic_queries": 2,
            "roots": [
                {
                    "id": "r1",
                    "query": "defense ai contracts",
                    "children": [
                        {"id": "palantir", "query": "Palantir AIP defense implementation"},
                        {"id": "lunar", "query": "Lunar mining extraction"},
                    ],
                },
                {
                    "id": "r2",
                    "query": "space infrastructure",
                    "children": [{"id": "rocket", "query": "Rocket Lab orbital services"}],
                },
            ],
        }
    }


# ---------- parse_graph ----------


class TestParseGraph:
    def test_returns_none_when_missing(self):
        assert parse_graph({}) is None
        assert parse_graph({"interest_graph": {}}) is None
        assert parse_graph({"interest_graph": {"roots": []}}) is None

    def test_builds_tree(self):
        graph = parse_graph(_config_with_graph())
        assert graph is not None
        assert graph.max_dynamic_queries == 2
        assert [n.id for n in graph.roots] == ["r1", "r2"]
        root = graph.roots[0]
        assert root.parent is None
        assert [c.id for c in root.children] == ["palantir", "lunar"]
        assert root.children[0].parent is root

    def test_priority_defaults_to_one(self):
        graph = parse_graph(_config_with_graph())
        assert graph.roots[0].children[0].priority == 1.0


# ---------- tree traversal ----------


class TestTraversal:
    def test_collect_roots(self):
        graph = _make_graph()
        assert graph.collect_roots() == ["defense ai contracts", "space infrastructure"]

    def test_leaves(self):
        graph = _make_graph()
        assert [n.id for n in graph.leaves()] == ["palantir", "lunar", "benchmarks", "rocket"]

    def test_all_nodes(self):
        graph = _make_graph()
        assert len(graph.all_nodes()) == 6


# ---------- signal scoring ----------


class TestTermOverlap:
    def test_empty_inputs(self):
        assert term_overlap("", "") == 0.0
        assert term_overlap("palantir", "") == 0.0
        assert term_overlap("", "corpus") == 0.0

    def test_full_match(self):
        assert term_overlap("palantir defense", "palantir defense") == 1.0

    def test_partial_match(self):
        assert term_overlap("palantir defense", "palantir wins contract") == 0.5

    def test_ignores_stopwords_and_short_terms(self):
        # "the", "and", "ai" (len<=2) are filtered
        assert term_overlap("the ai", "the ai") == 0.0


class TestScoreSignal:
    def test_score_from_previous_state(self):
        graph = _make_graph()
        leaves = graph.leaves()
        state = {
            "top_news_titles": ["Palantir wins new defense contract"],
            "emerging_themes": ["defense ai"],
        }
        signals = score_signal(leaves, state)
        assert signals["palantir"] > signals["lunar"]
        assert signals["palantir"] > 0

    def test_score_from_today_blogs(self):
        graph = _make_graph()
        leaves = graph.leaves()
        blogs = [{"title": "Rocket Lab launches orbital mission"}]
        signals = score_signal(leaves, {}, blogs)
        assert signals["rocket"] > 0
        assert signals["rocket"] > signals["palantir"]

    def test_empty_state_returns_zeros(self):
        graph = _make_graph()
        signals = score_signal(graph.leaves(), None)
        assert all(v == 0.0 for v in signals.values())


# ---------- selection ----------


class TestRankLeaves:
    def test_sorted_by_signal(self):
        graph = _make_graph()
        signals = {"palantir": 0.9, "lunar": 0.2, "benchmarks": 0.5, "rocket": 0.0}
        ranked = rank_leaves(graph, signals)
        assert [n.id for n in ranked] == ["palantir", "benchmarks", "lunar", "rocket"]

    def test_priority_breaks_ties(self):
        graph = _make_graph()
        graph.roots[0].children[1].priority = 2.0  # lunar
        signals = {"palantir": 0.5, "lunar": 0.5, "benchmarks": 0.5, "rocket": 0.5}
        ranked = rank_leaves(graph, signals)
        assert ranked[0].id == "lunar"


# ---------- generate_graph_queries ----------


class TestGenerateGraphQueries:
    def test_no_signal_returns_roots_only(self):
        graph = _make_graph()
        result = generate_graph_queries(graph, None)
        assert result == ["defense ai contracts", "space infrastructure"]

    def test_no_leaves_returns_roots(self):
        graph = InterestGraph([Node("only", "solo root query")], max_dynamic_queries=2)
        assert generate_graph_queries(graph, {"top_news_titles": ["x"]}) == ["solo root query"]

    def test_deterministic_selection_respects_budget(self):
        graph = _make_graph()
        state = {"top_news_titles": ["Palantir defense contract", "AI model benchmark"]}
        result = generate_graph_queries(graph, state)
        roots = graph.collect_roots()
        assert result[: len(roots)] == roots
        dig = result[len(roots):]
        assert len(dig) == graph.max_dynamic_queries
        assert "Palantir AIP defense implementation" in dig

    def test_llm_selection_overrides(self):
        graph = _make_graph()
        state = {"top_news_titles": ["Lunar mining news"]}
        llm = MagicMock()
        llm.available = True
        llm.invoke.return_value = "Lunar mining extraction\nAI model benchmarks"
        result = generate_graph_queries(graph, state, llm_client=llm)
        dig = result[len(graph.collect_roots()):]
        assert dig == ["Lunar mining extraction", "AI model benchmarks"]

    def test_llm_invented_queries_ignored(self):
        graph = _make_graph()
        state = {"top_news_titles": ["Palantir defense contract"]}
        llm = MagicMock()
        llm.available = True
        llm.invoke.return_value = "Totally invented query\nAnother fake query"
        result = generate_graph_queries(graph, state, llm_client=llm)
        # fallback to deterministic selection (no invented queries leak through)
        for q in result:
            assert q in {n.query for n in graph.all_nodes()}

    def test_llm_unavailable_falls_back(self):
        graph = _make_graph()
        state = {"top_news_titles": ["Palantir defense contract"]}
        llm = MagicMock()
        llm.available = False
        result = generate_graph_queries(graph, state, llm_client=llm)
        assert "Palantir AIP defense implementation" in result
        llm.invoke.assert_not_called()
