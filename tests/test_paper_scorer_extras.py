# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Extra tests for paper_scorer: infrastructure penalties + main() CLI."""

import json
from unittest.mock import patch

import pytest

from scripts.paper_scorer import PaperScorer, load_config, main


@pytest.fixture
def scorer():
    return PaperScorer(
        topics=["Agents"],
        weights={"has_code": 7, "topic_match": 3, "recency": 2, "citation_count": 1},
        num_picks=3,
    )


class TestInfraPenalty:
    def test_no_penalty_normal_paper(self, scorer):
        paper = {
            "title": "Single-GPU model",
            "summary": "We train with code at github.com/x/y on one GPU",
        }
        penalty = scorer.calculate_infra_penalty(paper)
        assert penalty == 0.0

    def test_penalizes_tpu_pod(self, scorer):
        paper = {
            "title": "Big model",
            "summary": "We use a tpu pod with petabyte data, code at github.com/x/y",
        }
        penalty = scorer.calculate_infra_penalty(paper)
        # -2 for infra impossible
        assert penalty <= -2.0

    def test_penalizes_theory_only(self, scorer):
        paper = {
            "title": "Theory paper",
            "summary": "We prove the theorem with theoretical analysis at github.com/x/y",
        }
        penalty = scorer.calculate_infra_penalty(paper)
        # -1.5 for theory-only
        assert penalty <= -1.5

    def test_penalizes_no_code(self, scorer):
        paper = {"title": "Method", "summary": "We propose a new approach."}
        # No code → -1.0
        penalty = scorer.calculate_infra_penalty(paper)
        assert penalty <= -1.0

    def test_caps_at_minus_5(self, scorer):
        paper = {
            "title": "Bad",
            "summary": "We prove theorems with tpu pod (no code provided)",
        }
        penalty = scorer.calculate_infra_penalty(paper)
        # Even with multiple penalties, caps at -5
        assert penalty >= -5.0


class TestCalculateTopicMatchEdgeCases:
    def test_handles_vectorizer_failure(self, scorer, monkeypatch):
        # Force the vectorizer to raise
        monkeypatch.setattr(
            scorer.vectorizer, "fit_transform",
            lambda x: (_ for _ in ()).throw(ValueError("boom"))
        )
        scores = scorer.calculate_topic_match([{"title": "x", "summary": "y"}])
        assert scores == [0.0]


class TestEstimateReproductionDifficultyExtras:
    def test_xl_keywords(self, scorer):
        for kw in ["petabyte", "exascale", "tpu pod", "kubernetes", "multi-node"]:
            paper = {"summary": f"This uses {kw} compute"}
            assert scorer.estimate_reproduction_difficulty(paper) == "XL"

    def test_l_keywords(self, scorer):
        for kw in ["a100", "h100", "8 gpu", "16 gpu"]:
            paper = {"summary": f"Training with {kw}"}
            assert scorer.estimate_reproduction_difficulty(paper) == "L"

    def test_s_keywords(self, scorer):
        for kw in ["lightweight", "efficient", "api"]:
            paper = {"summary": f"This is a {kw} method"}
            assert scorer.estimate_reproduction_difficulty(paper) == "S"


class TestLoadConfig:
    def test_loads_valid(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text("arxiv_topics:\n  - X\npaper_scoring:\n  has_code: 5\n")
        cfg = load_config(str(f))
        assert cfg["paper_scoring"]["has_code"] == 5

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_config(str(tmp_path / "nope.yaml"))

    def test_bad_yaml_exits(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text(": invalid :\n  -[\n}")
        with pytest.raises(SystemExit):
            load_config(str(f))


class TestPaperScorerMain:
    def _setup(self, tmp_path, papers):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "arxiv_topics:\n  - X\n"
            "paper_scoring:\n  has_code: 5\n  topic_match: 3\n"
            "num_paper_picks: 2\n"
        )
        inp = tmp_path / "papers.json"
        inp.write_text(json.dumps(papers))
        out = tmp_path / "scored.json"
        return cfg, inp, out

    def test_main_writes_output(self, tmp_path, monkeypatch):
        papers = [
            {"id": "1", "title": "A", "summary": "abs", "published": "",
             "authors": []},
        ]
        cfg, inp, out = self._setup(tmp_path, papers)
        monkeypatch.setattr("sys.argv", [
            "scorer.py", "--config", str(cfg),
            "--input", str(inp), "--output", str(out)
        ])
        assert main() == 0
        data = json.loads(out.read_text())
        assert "score" in data[0]

    def test_main_missing_input(self, tmp_path, monkeypatch):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("arxiv_topics:\n  - X\n")
        monkeypatch.setattr("sys.argv", [
            "scorer.py", "--config", str(cfg),
            "--input", str(tmp_path / "nope.json"),
            "--output", str(tmp_path / "o.json")
        ])
        assert main() == 2

    def test_main_bad_input_json(self, tmp_path, monkeypatch):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("arxiv_topics:\n  - X\n")
        inp = tmp_path / "bad.json"
        inp.write_text("not json")
        monkeypatch.setattr("sys.argv", [
            "scorer.py", "--config", str(cfg),
            "--input", str(inp), "--output", str(tmp_path / "o.json")
        ])
        assert main() == 2

    def test_main_empty_input_returns_1(self, tmp_path, monkeypatch):
        cfg, inp, out = self._setup(tmp_path, [])
        monkeypatch.setattr("sys.argv", [
            "scorer.py", "--config", str(cfg),
            "--input", str(inp), "--output", str(out)
        ])
        assert main() == 1

    def test_main_no_topics_warning_only(self, tmp_path, monkeypatch):
        """No topics is warning, not error."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("arxiv_topics: []\n")
        papers = [{"id": "1", "title": "P", "summary": "a", "published": "",
                   "authors": []}]
        inp = tmp_path / "p.json"
        inp.write_text(json.dumps(papers))
        out = tmp_path / "o.json"
        monkeypatch.setattr("sys.argv", [
            "scorer.py", "--config", str(cfg),
            "--input", str(inp), "--output", str(out)
        ])
        assert main() == 0
