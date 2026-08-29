"""Shared pytest guards.

The quality checker appends judged scores to logs/quality-scores.jsonl, and the
regression detector reads that file back to decide whether a dimension has
degraded. A test that forgets to redirect the path therefore does not just
leave litter: it injects synthetic rows into the history that real alerts are
computed against. One test did exactly that, seeding hundreds of perfect
`why: "ok"` rows that became the "baseline" a genuine regression was measured
from.
"""

import pytest

import scripts.quality_check as quality_check


@pytest.fixture(autouse=True)
def _never_write_the_real_score_log(tmp_path, monkeypatch):
    """Point the default score-log path at a per-test temp file.

    Autouse so a test cannot opt out by omission — the failure mode this
    guards against is precisely forgetting to pass scores_path.
    """
    monkeypatch.setattr(
        quality_check, "DEFAULT_SCORES_PATH", str(tmp_path / "quality-scores.jsonl")
    )
