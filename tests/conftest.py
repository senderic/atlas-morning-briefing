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

import smtplib

import scripts.codex_client as codex_client
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
    # The same hazard for the judge-skip streak counter and the saved raw
    # judge output: both feed what the real checker reports tomorrow.
    monkeypatch.setattr(
        quality_check, "DEFAULT_STREAKS_PATH", str(tmp_path / "quality-streaks.json")
    )
    monkeypatch.setattr(
        quality_check, "DEFAULT_JUDGE_RAW_DIR", str(tmp_path / "judge-raw")
    )


@pytest.fixture(autouse=True)
def _never_discover_real_codex_cli(monkeypatch):
    """Keep runner tests from invoking the machine's Codex executable.

    Codex-client unit tests explicitly patch this discovery seam and mock
    subprocess execution, so they continue to exercise their intended paths.
    """
    monkeypatch.setattr(codex_client.shutil, "which", lambda executable: None)


@pytest.fixture(autouse=True)
def _never_send_real_email(monkeypatch):
    """Make delivery impossible from a test, whatever flags it passes.

    Importing the runner loads the machine's real `.env`, Gmail app password
    included. Tests used to rely on `dry_run=True` to keep that inert, which
    stopped being a neutral default once a dry run also stopped writing state:
    a test of what a real run writes has to construct a real run. So the
    credentials are removed and the SMTP constructors refuse to connect. A
    test that exercises delivery patches these itself, inside this guard.
    """
    for name in ("GMAIL_USER", "GMAIL_APP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a real SMTP connection")

    monkeypatch.setattr(smtplib, "SMTP", refuse)
    monkeypatch.setattr(smtplib, "SMTP_SSL", refuse)
