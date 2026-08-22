#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Export the interest taxonomy DAG to Graphviz (.dot) and Mermaid (.mmd).

Reads the ``interest_graph`` section from a config file and emits two
visualization formats, plus an optional raster render via the system
``dot`` binary.

Usage:
    python3 scripts/export_interest_graph.py --config config.yaml
    python3 scripts/export_interest_graph.py --config config.yaml --render png
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from scripts.interest_graph import InterestGraph, Node, parse_graph


def _quote(text: str) -> str:
    return text.replace('"', '\\"')


def _edges(node: Node) -> List[tuple]:
    out = []
    for child in node.children:
        out.append((node, child))
        out.extend(_edges(child))
    return out


def render_dot(graph: InterestGraph) -> str:
    lines = [
        "digraph interest {",
        "  rankdir=TB;",
        "  node [shape=box, fontsize=10];",
        '  edge [color="#888888"];',
    ]
    for node in graph.all_nodes():
        if node.parent is None:
            lines.append(
                f'  "{node.id}" [label="{_quote(node.query)}", style=filled, fillcolor=lightblue];'
            )
        else:
            lines.append(f'  "{node.id}" [label="{_quote(node.query)}"];')
    for parent, child in _edges_from_roots(graph):
        lines.append(f'  "{parent.id}" -> "{child.id}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _edges_from_roots(graph: InterestGraph) -> List[tuple]:
    out = []
    for root in graph.roots:
        out.extend(_edges(root))
    return out


def render_mermaid(graph: InterestGraph) -> str:
    lines = ["graph TD"]
    for node in graph.all_nodes():
        if node.parent is None:
            lines.append(f'  {node.id}["{_quote(node.query)}"]')
    for parent, child in _edges_from_roots(graph):
        lines.append(f'  {parent.id}["{_quote(parent.query)}"] --> {child.id}["{_quote(child.query)}"]')
    return "\n".join(lines) + "\n"


def render_image(dot_text: str, out_path: Path, fmt: str) -> bool:
    try:
        subprocess.run(
            ["dot", f"-T{fmt}", "-o", str(out_path)],
            input=dot_text,
            text=True,
            check=True,
            capture_output=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"render failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Export interest taxonomy DAG")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--outdir", type=Path, default=Path("logs"), help="Output directory")
    parser.add_argument(
        "--render",
        choices=["png", "svg"],
        default=None,
        help="Render via system 'dot' to this format",
    )
    args = parser.parse_args()

    config: Dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    graph = parse_graph(config)
    if graph is None:
        print(f"no interest_graph found in {args.config}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    dot_text = render_dot(graph)
    mermaid_text = render_mermaid(graph)

    dot_path = args.outdir / "interest-graph.dot"
    mmd_path = args.outdir / "interest-graph.mmd"
    dot_path.write_text(dot_text)
    mmd_path.write_text(mermaid_text)
    print(f"wrote {dot_path}")
    print(f"wrote {mmd_path}")

    if args.render:
        img_path = args.outdir / f"interest-graph.{args.render}"
        if render_image(dot_text, img_path, args.render):
            print(f"wrote {img_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
