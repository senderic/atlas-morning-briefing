# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for the interest graph exporter (dot + mermaid)."""

import shutil

import pytest

from scripts.export_interest_graph import (
    main as export_main,
    render_dot,
    render_image,
    render_mermaid,
)
from scripts.interest_graph import InterestGraph, Node


def _graph() -> InterestGraph:
    return InterestGraph(
        roots=[
            Node("r1", "defense ai contracts", children=[
                Node("l1", "Palantir AIP defense implementation"),
                Node("l2", 'quotes "in" query'),
            ]),
            Node("r2", "space infrastructure"),
        ],
        max_dynamic_queries=2,
    )


class TestRenderDot:
    def test_header_and_nodes(self):
        text = render_dot(_graph())
        assert text.startswith("digraph interest {")
        for nid in ("r1", "r2", "l1", "l2"):
            assert f'"{nid}"' in text

    def test_edges(self):
        text = render_dot(_graph())
        assert '"r1" -> "l1"' in text
        assert '"r1" -> "l2"' in text

    def test_escapes_quotes(self):
        text = render_dot(_graph())
        assert r"quotes \"in\" query" in text


class TestRenderMermaid:
    def test_header_and_edges(self):
        text = render_mermaid(_graph())
        assert text.startswith("graph TD")
        assert 'r1["defense ai contracts"] --> l1["Palantir AIP defense implementation"]' in text


class TestExportCli:
    def test_writes_dot_and_mmd(self, tmp_path, monkeypatch):
        import yaml

        cfg = {
            "interest_graph": {
                "max_dynamic_queries": 1,
                "roots": [
                    {"id": "r1", "query": "defense", "children": [{"id": "l1", "query": "palantir"}]}
                ],
            }
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        outdir = tmp_path / "out"
        monkeypatch.setattr("sys.argv", ["export", "--config", str(cfg_path), "--outdir", str(outdir)])
        assert export_main() == 0
        assert (outdir / "interest-graph.dot").exists()
        assert (outdir / "interest-graph.mmd").exists()

    def test_missing_graph_returns_2(self, tmp_path, monkeypatch):
        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"arxiv_topics": []}))
        monkeypatch.setattr("sys.argv", ["export", "--config", str(cfg_path)])
        assert export_main() == 2


class TestRenderImage:
    @pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz 'dot' not installed")
    def test_renders_png(self, tmp_path):
        out = tmp_path / "g.png"
        assert render_image(render_dot(_graph()), out, "png") is True
        assert out.exists() and out.stat().st_size > 0
