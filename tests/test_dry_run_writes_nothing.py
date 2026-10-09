"""`--dry-run` must leave the day's real artifacts exactly as it found them.

A dry run used to skip only the email: it still rewrote the state file, the
status file, the raw-data snapshots and that day's briefing. Cron reads all of
those, so a midday dry run changed what the next real run deduplicated against
and what downstream consumers imported.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.briefing_runner import BriefingRunner


PAPERS = [
    {
        "title": "A Paper About Agents",
        "summary": "We study agents.",
        "authors": ["A. Author"],
        "url": "https://arxiv.org/abs/2610.00001",
        "published": "2026-10-07",
    }
]
BLOGS = [
    {
        "title": "A Blog Post",
        "summary": "Something happened.",
        "source": "Example Blog",
        "link": "https://example.com/post",
        "published": "2026-10-07",
    }
]
STOCKS = [
    {"symbol": "ABC", "name": "ABC Corp", "current_price": 10.0, "percent_change": 1.2}
]
NEWS = [
    {
        "title": "A Headline",
        "description": "Details.",
        "source": "Example News",
        "url": "https://news.example.com/a",
    }
]

# One entry per pipeline: they are the same runner driven by a different
# config, so what differs is only where each keeps its files.
PIPELINES = {
    "main": {"status": "status.json", "state": ".atlas-state.json", "out": "briefings", "snap": "snapshots"},
    "local": {"status": "status-local.json", "state": ".local-state.json", "out": "briefings/local", "snap": "snapshots/local"},
    "finance": {"status": "status-finance.json", "state": ".finance-state.json", "out": "briefings/finance", "snap": "snapshots/finance"},
}


def make_config(root: Path, pipeline: str) -> dict:
    paths = PIPELINES[pipeline]
    return {
        "pipeline_name": pipeline,
        "arxiv_topics": ["AI"],
        "blog_feeds": [],
        "stocks": [],
        "news_queries": [],
        "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
        "num_paper_picks": 2,
        "file_naming": "Test-{yyyy}.{mm}.{dd}",
        "pdf": {"enabled": False},
        "gemini": {"enabled": False, "call_log_path": str(root / "logs" / "gemini-calls.jsonl")},
        "codex": {"enabled": True, "call_log_path": str(root / "logs" / "codex-calls.jsonl")},
        "output_dir": str(root / paths["out"]),
        "state_file_path": str(root / paths["state"]),
        "status_file_path": paths["status"],
        "snapshot": {"enabled": True, "dir": str(root / paths["snap"])},
    }


def seed_real_artifacts(root: Path, config: dict) -> None:
    """What a real run earlier the same day would have left behind."""
    today = datetime.now()
    out = Path(config["output_dir"])
    out.mkdir(parents=True)
    name = f"Test-{today:%Y}.{today:%m}.{today:%d}"
    (out / f"{name}.md").write_text("# the real briefing\n")
    (out / f"{name}.epub").write_bytes(b"real epub")
    Path(config["state_file_path"]).write_text('{"date": "real", "weekly_items": [1]}')
    (root / config["status_file_path"]).write_text('{"real": true}')
    snap = Path(config["snapshot"]["dir"]) / "2026-10-08"
    snap.mkdir(parents=True)
    (snap / "brave_news.json").write_text("[]")
    (root / "logs").mkdir()
    (root / "logs" / "codex-calls.jsonl").write_text('{"real": 1}\n')


def tree(root: Path) -> dict:
    """Every file under root -> (content hash, mtime)."""
    return {
        str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def run_pipeline(root: Path, config: dict, dry_run: bool) -> int:
    runner = BriefingRunner(config, dry_run=dry_run)
    with (
        patch.object(runner, "run_arxiv_scan", return_value=[dict(p) for p in PAPERS]),
        patch.object(runner, "run_blog_scan", return_value=[dict(b) for b in BLOGS]),
        patch.object(runner, "run_stock_fetch", return_value=[dict(s) for s in STOCKS]),
        patch.object(runner, "run_alerts_scan", return_value=[]),
        patch.object(runner, "run_news_aggregation", return_value=[dict(n) for n in NEWS]),
        patch.object(runner, "_load_or_fetch_happenings", return_value=[]),
    ):
        return runner.run()


@pytest.fixture
def root(tmp_path, monkeypatch):
    # The status file is written relative to the working directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GMAIL_USER", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    return tmp_path


@pytest.mark.parametrize("pipeline", sorted(PIPELINES))
class TestDryRunWritesNothing:
    def test_nothing_outside_the_dry_run_directory_is_touched(self, root, pipeline):
        config = make_config(root, pipeline)
        seed_real_artifacts(root, config)
        before = tree(root)

        assert run_pipeline(root, config, dry_run=True) in (0, 1)

        after = tree(root)
        dry_dir = (Path(config["output_dir"]) / "dry-run").relative_to(root)
        for path, fingerprint in before.items():
            assert after.get(path) == fingerprint, f"dry run modified {path}"
        created = sorted(set(after) - set(before))
        assert created, "a dry run should still render the briefing somewhere"
        outside = [p for p in created if not Path(p).is_relative_to(dry_dir)]
        assert outside == [], f"dry run created files outside {dry_dir}: {outside}"

    def test_rendered_briefing_lands_in_the_dry_run_directory(self, root, pipeline):
        config = make_config(root, pipeline)
        seed_real_artifacts(root, config)

        run_pipeline(root, config, dry_run=True)

        today = datetime.now()
        dry_dir = Path(config["output_dir"]) / "dry-run"
        rendered = dry_dir / f"Test-{today:%Y}.{today:%m}.{today:%d}.md"
        assert rendered.is_file()
        assert "A Headline" in rendered.read_text()
        # The run's own status is kept beside it, for whoever is inspecting.
        assert (dry_dir / config["status_file_path"]).is_file()

    def test_a_real_run_still_writes_everything(self, root, pipeline):
        """The control: without it, the test above could pass on a runner
        that had simply stopped writing."""
        config = make_config(root, pipeline)
        seed_real_artifacts(root, config)
        before = tree(root)

        assert run_pipeline(root, config, dry_run=False) in (0, 1)

        after = tree(root)
        today = datetime.now()
        out = Path(config["output_dir"]).relative_to(root)
        changed = {p for p in after if after[p] != before.get(p)}
        assert str(out / f"Test-{today:%Y}.{today:%m}.{today:%d}.md") in changed
        assert str(Path(config["state_file_path"]).relative_to(root)) in changed
        assert config["status_file_path"] in changed
        assert any(Path(p).name == "snapshot_manifest.json" for p in changed)
        assert not (Path(config["output_dir"]) / "dry-run").exists()


def test_dry_run_call_logs_are_redirected(root):
    config = make_config(root, "main")
    runner = BriefingRunner(config, dry_run=True)
    dry_dir = Path(config["output_dir"]) / "dry-run"
    assert runner.codex_client.call_log_path == dry_dir / "codex-calls.jsonl"
    assert runner.config["gemini"]["call_log_path"] == str(dry_dir / "gemini-calls.jsonl")
    # The caller's config object is not rewritten underneath it.
    assert config["codex"]["call_log_path"] == str(root / "logs" / "codex-calls.jsonl")
    assert config["output_dir"] == str(root / "briefings")


def test_real_run_call_logs_are_untouched(root):
    config = make_config(root, "main")
    runner = BriefingRunner(config, dry_run=False)
    assert runner.codex_client.call_log_path == root / "logs" / "codex-calls.jsonl"


def test_an_absolute_status_path_is_still_kept_out_of_reach(root):
    config = make_config(root, "main")
    real = root / "elsewhere" / "status.json"
    real.parent.mkdir()
    real.write_text('{"real": true}')
    config["status_file_path"] = str(real)

    run_pipeline(root, config, dry_run=True)

    assert real.read_text() == '{"real": true}'
    assert (Path(config["output_dir"]) / "dry-run" / "status.json").is_file()
