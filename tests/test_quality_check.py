# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Tests for the daily quality check orchestrator.

Every layer callable (source health, report invariants, the LLM judge, its
client) is injected as a fake -- nothing here touches journald, the network,
or a real LLM. Two tests exercise the lazy-import fallback path itself by
stubbing the sibling modules in sys.modules, since those modules
(scripts.source_health, scripts.report_invariants) are being written by
other agents in parallel and may not exist yet.
"""

import json
import sys
import types
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from scripts.quality_findings import CRITICAL, INFO, WARN, Finding
from scripts import quality_check as qc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_client(available=True, response="{}"):
    class _Client:
        def __init__(self):
            self.available = available
            self.calls = []

        def invoke(self, prompt, tier="medium", system_prompt=None, **kw):
            self.calls.append((prompt, tier, system_prompt))
            return response

    return _Client()


def _judge_json(scores=None):
    scores = scores or {}
    base = {
        "tier_1_share": 2,
        "lead_alignment": 2,
        "actionability": 2,
        "locality": 2,
        "specificity": 2,
        "freshness": 2,
    }
    base.update(scores)
    return json.dumps({k: {"score": v, "why": "ok"} for k, v in base.items()})


def _scores_json(dims_scores):
    """Build a judge JSON payload containing exactly the given dimensions
    (unlike _judge_json, does not default in the other unmentioned ones)."""
    return json.dumps({k: {"score": v, "why": "ok"} for k, v in dims_scores.items()})


def _no_op_layer1(harvest_records=None):
    """Injectable fakes for run_checks' Layer 1 params that do nothing."""
    return {
        "harvest_journal": lambda **kw: harvest_records or [],
        "append_history": lambda records, path=None: len(records or []),
        "load_history": lambda path=None, since=None: harvest_records or [],
        "detect_rot": lambda history, probes=None, rules=None: [],
    }


def _no_op_layer2():
    return {"check_report": lambda markdown, config, today=None, pipeline="": []}


def _stub_sibling_modules(monkeypatch, detect_rot_findings=None, check_report_findings=None):
    """Stub scripts.source_health / scripts.report_invariants in sys.modules.

    Used for tests that go through qc.main() (which has no injection points
    of its own) to guarantee they never touch journald, the network, or an
    LLM even now that the real sibling modules exist on disk.
    """
    sh = types.ModuleType("scripts.source_health")
    sh.harvest_journal = lambda since="-1d", units=(), timeout=180: []
    sh.append_history = lambda records, path="logs/source-health.jsonl": len(records)
    sh.load_history = lambda path="logs/source-health.jsonl", since=None: []
    sh.probe_feeds = lambda feeds, timeout=20: []
    sh.detect_rot = lambda history, probes=None, rules=None: detect_rot_findings or []
    monkeypatch.setitem(sys.modules, "scripts.source_health", sh)

    ri = types.ModuleType("scripts.report_invariants")
    ri.check_report = lambda markdown, config, today=None, pipeline="": check_report_findings or []
    monkeypatch.setitem(sys.modules, "scripts.report_invariants", ri)


# ---------------------------------------------------------------------------
# Finding merge / sort across layers
# ---------------------------------------------------------------------------


class TestRunChecksMerging:
    def test_findings_from_all_layers_merged_and_sorted(self, tmp_path):
        configs = {
            "atlas": {"output_dir": str(tmp_path), "file_naming": "Atlas-{yyyy}.{mm}.{dd}"},
        }
        briefing = tmp_path / "Atlas-2026.08.25.md"
        briefing.write_text("# Briefing\n\nSome content.\n")

        layer1 = _no_op_layer1()
        layer1["detect_rot"] = lambda history, probes=None, rules=None: [
            Finding(WARN, "feed-dead", "Anthropic feed is dead", source="Anthropic", pipeline="atlas"),
        ]
        layer2 = {
            "check_report": lambda markdown, config, today=None, pipeline="": [
                Finding(CRITICAL, "scaffolding-leak", "Dropped: leaked", pipeline=pipeline),
            ]
        }

        findings, scores = qc.run_checks(
            configs,
            today=date(2026, 8, 25),
            no_judge=True,
            **layer1,
            **layer2,
        )

        assert [f.code for f in findings] == ["scaffolding-leak", "feed-dead"]
        assert findings[0].severity == CRITICAL
        assert findings[1].severity == WARN
        assert scores == {}


# ---------------------------------------------------------------------------
# Layer 2: missing briefing
# ---------------------------------------------------------------------------


class TestBriefingMissing:
    def test_missing_file_is_critical(self, tmp_path):
        configs = {"atlas": {"output_dir": str(tmp_path), "file_naming": "Atlas-{yyyy}.{mm}.{dd}"}}
        findings, scores = qc.run_checks(
            configs,
            today=date(2026, 8, 25),
            no_judge=True,
            **_no_op_layer1(),
            **_no_op_layer2(),
        )
        missing = [f for f in findings if f.code == "briefing-missing"]
        assert len(missing) == 1
        assert missing[0].severity == CRITICAL
        assert missing[0].pipeline == "atlas"
        assert scores == {}

    def test_present_file_not_flagged(self, tmp_path):
        configs = {"atlas": {"output_dir": str(tmp_path), "file_naming": "Atlas-{yyyy}.{mm}.{dd}"}}
        (tmp_path / "Atlas-2026.08.25.md").write_text("# Briefing\n")
        findings, _ = qc.run_checks(
            configs,
            today=date(2026, 8, 25),
            no_judge=True,
            **_no_op_layer1(),
            **_no_op_layer2(),
        )
        assert not [f for f in findings if f.code == "briefing-missing"]


# ---------------------------------------------------------------------------
# Layer 3: judge parsing
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_clean_json(self):
        parsed = qc.parse_judge_response(_judge_json())
        assert parsed["tier_1_share"]["score"] == 2
        assert set(parsed) == set(qc.RUBRIC_DIMENSIONS)

    def test_fenced_json(self):
        text = "Here is my scoring:\n```json\n" + _judge_json({"freshness": 0}) + "\n```\nHope that helps."
        parsed = qc.parse_judge_response(text)
        assert parsed["freshness"]["score"] == 0

    def test_prose_wrapped_json_no_fence(self):
        text = "Sure, here you go: " + _judge_json({"locality": 1}) + " let me know if you need more."
        parsed = qc.parse_judge_response(text)
        assert parsed["locality"]["score"] == 1

    def test_missing_dimension_returns_none(self):
        data = json.loads(_judge_json())
        del data["freshness"]
        assert qc.parse_judge_response(json.dumps(data)) is None

    def test_score_out_of_range_returns_none(self):
        assert qc.parse_judge_response(_judge_json({"tier_1_share": 5})) is None

    def test_garbage_returns_none(self):
        assert qc.parse_judge_response("not json at all, sorry") is None

    def test_empty_returns_none(self):
        assert qc.parse_judge_response("") is None
        assert qc.parse_judge_response(None) is None


class TestJudgeBriefing:
    def test_malformed_output_degrades_to_judge_skipped(self):
        client = _fake_client(response="I cannot comply with that request.")
        findings, record = qc.judge_briefing("# Briefing", {}, "atlas", date(2026, 8, 25), client)
        assert record is None
        assert len(findings) == 1
        assert findings[0].code == "judge-skipped"
        assert findings[0].severity == INFO

    def test_none_response_degrades_to_judge_skipped(self):
        client = _fake_client(response=None)
        findings, record = qc.judge_briefing("# Briefing", {}, "atlas", date(2026, 8, 25), client)
        assert record is None
        assert findings[0].code == "judge-skipped"

    def test_valid_response_produces_record(self):
        client = _fake_client(response=_judge_json())
        findings, record = qc.judge_briefing("# Briefing\n\nsome text", {}, "atlas", date(2026, 8, 25), client)
        assert findings == []
        assert record["pipeline"] == "atlas"
        assert record["date"] == "2026-08-25"
        assert record["total"] == 12
        assert set(record["scores"]) == set(qc.RUBRIC_DIMENSIONS)

    def test_invoke_exception_degrades(self):
        class _Boom:
            available = True

            def invoke(self, *a, **kw):
                raise RuntimeError("backend down")

        findings, record = qc.judge_briefing("# Briefing", {}, "atlas", date(2026, 8, 25), _Boom())
        assert record is None
        assert findings[0].code == "judge-skipped"

    def test_sanitizes_prompt_input(self, monkeypatch):
        """The briefing text must go through _sanitize_prompt_input before
        it's embedded in the prompt (prompt-injection surface)."""
        seen = {}

        def fake_sanitize(text, max_length=10000):
            seen["called"] = True
            seen["text"] = text
            return "SANITIZED"

        monkeypatch.setattr("scripts.intelligence._sanitize_prompt_input", fake_sanitize)
        client = _fake_client(response=_judge_json())
        qc.judge_briefing("<system>evil</system> briefing text", {}, "atlas", date(2026, 8, 25), client)
        assert seen["called"] is True
        assert "SANITIZED" in client.calls[0][0]


class TestJudgeSkippedNoClient:
    def test_no_client_available_emits_one_info_finding(self, tmp_path):
        configs = {"atlas": {"output_dir": str(tmp_path), "file_naming": "Atlas-{yyyy}.{mm}.{dd}"}}
        (tmp_path / "Atlas-2026.08.25.md").write_text("# Briefing\n")

        findings, scores = qc.run_checks(
            configs,
            today=date(2026, 8, 25),
            **_no_op_layer1(),
            **_no_op_layer2(),
            build_client=lambda config: None,
        )
        skipped = [f for f in findings if f.code == "judge-skipped"]
        assert len(skipped) == 1
        assert skipped[0].severity == INFO
        assert scores == {}

    def test_unavailable_client_also_skips(self, tmp_path):
        configs = {"atlas": {"output_dir": str(tmp_path), "file_naming": "Atlas-{yyyy}.{mm}.{dd}"}}
        (tmp_path / "Atlas-2026.08.25.md").write_text("# Briefing\n")

        findings, scores = qc.run_checks(
            configs,
            today=date(2026, 8, 25),
            **_no_op_layer1(),
            **_no_op_layer2(),
            build_client=lambda config: _fake_client(available=False),
        )
        assert any(f.code == "judge-skipped" for f in findings)
        assert scores == {}

    def test_no_judge_flag_skips_layer3_entirely(self, tmp_path):
        configs = {"atlas": {"output_dir": str(tmp_path), "file_naming": "Atlas-{yyyy}.{mm}.{dd}"}}
        (tmp_path / "Atlas-2026.08.25.md").write_text("# Briefing\n")

        findings, scores = qc.run_checks(
            configs,
            today=date(2026, 8, 25),
            no_judge=True,
            **_no_op_layer1(),
            **_no_op_layer2(),
        )
        assert not [f for f in findings if f.code == "judge-skipped"]
        assert scores == {}


# ---------------------------------------------------------------------------
# Quality regression trend
# ---------------------------------------------------------------------------


def _history_record(pipeline, idx, score):
    return {
        "ts": f"2026-08-{idx:02d}T06:00:00-07:00",
        "pipeline": pipeline,
        "date": f"2026-08-{idx:02d}",
        "scores": {dim: {"score": score, "why": ""} for dim in qc.RUBRIC_DIMENSIONS},
        "total": score * 6,
        "notes": "",
    }


class TestQualityRegression:
    def test_sustained_drop_fires(self):
        # 10 baseline runs at 2, then 3 recent runs at 0 -> full 2.0 drop.
        history = [_history_record("atlas", i, 2) for i in range(1, 11)]
        history += [_history_record("atlas", i, 0) for i in range(11, 14)]
        findings = qc.detect_quality_regression(history, "atlas")
        codes = {f.code for f in findings}
        assert codes == {"quality-regression"}
        assert len(findings) == len(qc.RUBRIC_DIMENSIONS)
        assert all(f.severity == WARN for f in findings)
        assert all(f.pipeline == "atlas" for f in findings)

    def test_single_bad_day_stays_silent(self):
        # 10 baseline runs at 2, then 2 good recent runs and 1 bad one.
        # Mean of the last 3 is (2+2+0)/3 = 1.33, drop of 0.67 < 1.0 threshold.
        history = [_history_record("atlas", i, 2) for i in range(1, 11)]
        history += [_history_record("atlas", 11, 2), _history_record("atlas", 12, 2), _history_record("atlas", 13, 0)]
        findings = qc.detect_quality_regression(history, "atlas")
        assert findings == []

    def test_insufficient_history_stays_silent(self):
        history = [_history_record("atlas", i, 0) for i in range(1, 5)]
        assert qc.detect_quality_regression(history, "atlas") == []

    def test_run_checks_wires_regression_from_loader(self, tmp_path):
        configs = {"atlas": {"output_dir": str(tmp_path), "file_naming": "Atlas-{yyyy}.{mm}.{dd}"}}
        (tmp_path / "Atlas-2026.08.25.md").write_text("# Briefing\n")

        prior = [_history_record("atlas", i, 2) for i in range(1, 11)]
        prior += [_history_record("atlas", 11, 0), _history_record("atlas", 12, 0)]

        findings, scores = qc.run_checks(
            configs,
            today=date(2026, 8, 25),
            scores_path=str(tmp_path / "scores.jsonl"),
            **_no_op_layer1(),
            **_no_op_layer2(),
            build_client=lambda config: _fake_client(response=_judge_json({dim: 0 for dim in qc.RUBRIC_DIMENSIONS})),
            score_history_loader=lambda pipeline, path=None: prior,
        )
        assert scores["atlas"]["total"] == 0
        assert any(f.code == "quality-regression" for f in findings)
        assert (tmp_path / "scores.jsonl").exists()


# ---------------------------------------------------------------------------
# Digest rendering
# ---------------------------------------------------------------------------


class TestRenderDigest:
    def test_headline_and_table(self):
        findings = [
            Finding(CRITICAL, "briefing-missing", "No briefing", pipeline="atlas"),
            Finding(WARN, "feed-dead", "dead feed", source="Anthropic", pipeline="atlas"),
        ]
        digest = qc.render_digest(findings, {}, date(2026, 8, 25))
        assert "CRITICAL" in digest.splitlines()[2]
        assert "briefing-missing" in digest
        assert "feed-dead" in digest
        assert "| Severity | Pipeline | Code | Source | Message |" in digest

    def test_empty_findings_says_so(self):
        digest = qc.render_digest([], {}, date(2026, 8, 25))
        assert "No findings." in digest

    def test_judge_scores_section(self):
        record = {
            "pipeline": "atlas",
            "total": 10,
            "scores": {dim: {"score": 2, "why": "good"} for dim in qc.RUBRIC_DIMENSIONS},
        }
        digest = qc.render_digest([], {"atlas": record}, date(2026, 8, 25))
        assert "## Judge scores" in digest
        assert "atlas" in digest
        assert "tier_1_share" in digest


# ---------------------------------------------------------------------------
# Alert dedupe
# ---------------------------------------------------------------------------


class TestRouteAlerts:
    def test_critical_pushes_once(self, tmp_path):
        pushed = []
        findings = [Finding(CRITICAL, "briefing-missing", "gone", pipeline="atlas")]
        state = qc.route_alerts(findings, state={}, notify_fn=pushed.append, path=str(tmp_path / "alerts.json"))
        assert len(pushed) == 1
        key = findings[0].dedupe_key
        assert state[key]["count"] == 1
        assert state[key]["last_alerted"] is not None

    def test_second_push_within_24h_suppressed(self, tmp_path):
        pushed = []
        findings = [Finding(CRITICAL, "feed-dead", "dead", source="Anthropic", pipeline="atlas")]
        now = datetime(2026, 8, 25, 6, 0, 0)
        alerts_path = str(tmp_path / "alerts.json")

        state = qc.route_alerts(findings, state={}, notify_fn=pushed.append, now=now, path=alerts_path)
        assert len(pushed) == 1

        later = now + timedelta(hours=2)
        state = qc.route_alerts(findings, state=state, notify_fn=pushed.append, now=later, path=alerts_path)
        assert len(pushed) == 1  # still one -- suppressed
        assert state[findings[0].dedupe_key]["count"] == 2

    def test_push_allowed_after_24h(self, tmp_path):
        pushed = []
        findings = [Finding(CRITICAL, "feed-dead", "dead", source="Anthropic", pipeline="atlas")]
        now = datetime(2026, 8, 25, 6, 0, 0)
        alerts_path = str(tmp_path / "alerts.json")

        state = qc.route_alerts(findings, state={}, notify_fn=pushed.append, now=now, path=alerts_path)
        assert len(pushed) == 1

        next_day = now + timedelta(hours=25)
        state = qc.route_alerts(findings, state=state, notify_fn=pushed.append, now=next_day, path=alerts_path)
        assert len(pushed) == 2

    def test_warn_and_info_never_push(self, tmp_path):
        pushed = []
        findings = [
            Finding(WARN, "feed-dead", "dead", pipeline="atlas"),
            Finding(INFO, "judge-skipped", "skip", pipeline="atlas"),
        ]
        qc.route_alerts(findings, state={}, notify_fn=pushed.append, path=str(tmp_path / "alerts.json"))
        assert pushed == []

    def test_dry_run_does_not_notify_or_persist(self, tmp_path):
        pushed = []
        findings = [Finding(CRITICAL, "briefing-missing", "gone", pipeline="atlas")]
        alerts_path = tmp_path / "quality-alerts.json"
        qc.route_alerts(findings, state={}, notify_fn=pushed.append, dry_run=True, path=str(alerts_path))
        assert pushed == []
        assert not alerts_path.exists()

    def test_notify_missing_env_logs_and_continues(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert qc.notify("hello") is False

    def test_notify_posts_via_urllib(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")

        calls = []

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"ok": true}'

        def fake_urlopen(req, timeout=15):
            calls.append(req.full_url)
            return _Resp()

        monkeypatch.setattr(qc.urllib.request, "urlopen", fake_urlopen)
        assert qc.notify("hello") is True
        assert calls and "api.telegram.org" in calls[0]


# ---------------------------------------------------------------------------
# --dry-run writes nothing
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_writes_no_files(self, tmp_path, monkeypatch):
        _stub_sibling_modules(monkeypatch)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "briefings").mkdir()
        (tmp_path / "briefings" / "Atlas-2026.08.25.md").write_text("# Briefing\n")

        config_path = tmp_path / "config.yaml"
        config_path.write_text("output_dir: briefings\nfile_naming: 'Atlas-{yyyy}.{mm}.{dd}'\n")

        rc = qc.main(["--config", str(config_path), "--date", "2026-08-25", "--dry-run", "--no-judge"])
        assert rc in (0, 1)

        # Nothing should have been created outside the pre-existing fixtures.
        assert not (tmp_path / "logs").exists()

    def test_run_checks_dry_run_skips_history_and_score_writes(self, tmp_path):
        configs = {"atlas": {"output_dir": str(tmp_path), "file_naming": "Atlas-{yyyy}.{mm}.{dd}"}}
        (tmp_path / "Atlas-2026.08.25.md").write_text("# Briefing\n")

        append_calls = []

        def fake_append_history(records, path=None):
            append_calls.append(records)
            return len(records)

        findings, scores = qc.run_checks(
            configs,
            today=date(2026, 8, 25),
            dry_run=True,
            scores_path=str(tmp_path / "scores.jsonl"),
            harvest_journal=lambda **kw: [],
            append_history=fake_append_history,
            load_history=lambda path=None, since=None: [],
            detect_rot=lambda history, probes=None, rules=None: [],
            **_no_op_layer2(),
            build_client=lambda config: _fake_client(response=_judge_json()),
        )
        assert append_calls == []  # never called in dry-run
        assert scores["atlas"]["total"] == 12
        assert not (tmp_path / "scores.jsonl").exists()  # dry-run: never persisted


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_exit_zero_on_clean_run(self, tmp_path, monkeypatch):
        _stub_sibling_modules(monkeypatch)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "briefings").mkdir()
        (tmp_path / "briefings" / "Atlas-2026.08.25.md").write_text("# Briefing\n")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("output_dir: briefings\nfile_naming: 'Atlas-{yyyy}.{mm}.{dd}'\n")

        rc = qc.main(["--config", str(config_path), "--date", "2026-08-25", "--no-judge"])
        assert rc == 0

    def test_exit_one_on_critical(self, tmp_path, monkeypatch):
        _stub_sibling_modules(monkeypatch)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "briefings").mkdir()
        # briefing file deliberately absent -> briefing-missing CRITICAL
        config_path = tmp_path / "config.yaml"
        config_path.write_text("output_dir: briefings\nfile_naming: 'Atlas-{yyyy}.{mm}.{dd}'\n")

        rc = qc.main(["--config", str(config_path), "--date", "2026-08-25", "--no-judge"])
        assert rc == 1

    def test_exit_two_on_bad_config(self, tmp_path, monkeypatch):
        _stub_sibling_modules(monkeypatch)
        monkeypatch.chdir(tmp_path)
        missing_config = tmp_path / "does-not-exist.yaml"
        rc = qc.main(["--config", str(missing_config), "--date", "2026-08-25", "--no-judge"])
        assert rc == 2

    def test_exit_two_on_bad_date(self, tmp_path, monkeypatch):
        _stub_sibling_modules(monkeypatch)
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("output_dir: briefings\n")
        rc = qc.main(["--config", str(config_path), "--date", "not-a-date"])
        assert rc == 2

    def test_run_checks_raising_surfaces_as_exit_two(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("output_dir: briefings\n")

        def boom(*a, **kw):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(qc, "run_checks", boom)
        rc = qc.main(["--config", str(config_path), "--date", "2026-08-25"])
        assert rc == 2


# ---------------------------------------------------------------------------
# Lazy-import mechanism itself (sibling modules stubbed via sys.modules,
# since scripts.source_health / scripts.report_invariants are being written
# by other agents in parallel and may not exist at test time).
# ---------------------------------------------------------------------------


class TestLazyImportFallback:
    def test_picks_up_stubbed_sibling_modules(self, tmp_path, monkeypatch):
        _stub_sibling_modules(
            monkeypatch,
            detect_rot_findings=[Finding(WARN, "feed-dead", "dead", pipeline="atlas")],
            check_report_findings=[Finding(WARN, "thin-section", "thin", pipeline="atlas")],
        )

        configs = {"atlas": {"output_dir": str(tmp_path), "file_naming": "Atlas-{yyyy}.{mm}.{dd}"}}
        (tmp_path / "Atlas-2026.08.25.md").write_text("# Briefing\n")

        findings, _ = qc.run_checks(configs, today=date(2026, 8, 25), no_judge=True)
        codes = {f.code for f in findings}
        assert "feed-dead" in codes
        assert "thin-section" in codes

    def test_missing_sibling_module_degrades_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "scripts.source_health", None)
        monkeypatch.setitem(sys.modules, "scripts.report_invariants", None)

        configs = {"atlas": {"output_dir": str(tmp_path), "file_naming": "Atlas-{yyyy}.{mm}.{dd}"}}
        (tmp_path / "Atlas-2026.08.25.md").write_text("# Briefing\n")

        findings, scores = qc.run_checks(configs, today=date(2026, 8, 25), no_judge=True)
        codes = {f.code for f in findings}
        assert "source-health-unavailable" in codes
        # report_invariants missing means no findings from it, but no crash
        # and the missing briefing check (pure Layer 2 file logic) still runs.
        assert scores == {}


# ---------------------------------------------------------------------------
# locate_briefing_path / config loading now delegate to briefing_runner.py
# (the source of truth) instead of a second, driftable copy. These tests
# assert the coupling rather than assume it.
# ---------------------------------------------------------------------------


class TestLocateBriefingPathAgreesWithRunner:
    def test_non_default_pattern_matches_format_briefing_filename(self, tmp_path):
        from scripts.briefing_runner import format_briefing_filename

        today = date(2026, 8, 25)
        file_naming = "Local-Briefing-{yyyy}.{mm}.{dd}"
        config = {"output_dir": str(tmp_path), "file_naming": file_naming}

        path = qc.locate_briefing_path(config, today)
        expected_name = format_briefing_filename(file_naming, today)

        assert path == Path(tmp_path) / f"{expected_name}.md"
        assert str(path).endswith("Local-Briefing-2026.08.25.md")

    def test_type_token_resolves_without_raising(self, tmp_path):
        # A naive `.format()` (rather than the runner's format_map with a
        # known_vars dict that includes "type") would KeyError on this.
        today = date(2026, 8, 25)
        config = {"output_dir": str(tmp_path), "file_naming": "{type}-Briefing-{yyyy}.{mm}.{dd}"}

        path = qc.locate_briefing_path(config, today)

        assert path.name == "Daily-Briefing-2026.08.25.md"

    def test_default_file_naming_used_when_config_omits_it(self, tmp_path):
        from scripts.briefing_runner import DEFAULT_FILE_NAMING, format_briefing_filename

        today = date(2026, 8, 25)
        config = {"output_dir": str(tmp_path)}  # no file_naming key at all

        path = qc.locate_briefing_path(config, today)
        expected_name = format_briefing_filename(DEFAULT_FILE_NAMING, today)

        assert path.name == f"{expected_name}.md"


class TestConfigInterpolationUsesRunnerLoader:
    """main() loads pipeline configs through scripts.briefing_runner.load_config,
    so ${VAR:-default} interpolation must go through that exact function --
    not a local reimplementation."""

    def _write_config(self, tmp_path, output_dir_expr):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f'output_dir: "{output_dir_expr}"\n'
            "file_naming: 'Atlas-{yyyy}.{mm}.{dd}'\n"
        )
        return config_path

    def test_fallback_used_when_env_var_unset(self, tmp_path, monkeypatch):
        _stub_sibling_modules(monkeypatch)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("QC_TEST_OUTPUT_DIR", raising=False)
        monkeypatch.chdir(tmp_path)

        (tmp_path / "fallback-dir").mkdir()
        (tmp_path / "fallback-dir" / "Atlas-2026.08.25.md").write_text("# Briefing\n")

        config_path = self._write_config(tmp_path, "${QC_TEST_OUTPUT_DIR:-fallback-dir}")

        rc = qc.main(["--config", str(config_path), "--date", "2026-08-25", "--no-judge"])
        # The briefing is only found if output_dir resolved to "fallback-dir".
        assert rc == 0

    def test_env_value_used_when_set(self, tmp_path, monkeypatch):
        _stub_sibling_modules(monkeypatch)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.setenv("QC_TEST_OUTPUT_DIR", "real-dir")
        monkeypatch.chdir(tmp_path)

        (tmp_path / "real-dir").mkdir()
        (tmp_path / "real-dir" / "Atlas-2026.08.25.md").write_text("# Briefing\n")
        # Deliberately do NOT create fallback-dir, to prove the env value
        # won rather than the default.

        config_path = self._write_config(tmp_path, "${QC_TEST_OUTPUT_DIR:-fallback-dir}")

        rc = qc.main(["--config", str(config_path), "--date", "2026-08-25", "--no-judge"])
        assert rc == 0

    def test_env_value_wins_produces_critical_when_absent(self, tmp_path, monkeypatch):
        """Negative control: if the env var resolved to the wrong directory,
        the briefing would be reported missing (CRITICAL -> exit 1). Proves
        the previous two tests aren't passing by accident (e.g. both dirs
        existing, or interpolation being skipped)."""
        _stub_sibling_modules(monkeypatch)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.setenv("QC_TEST_OUTPUT_DIR", "real-dir")
        monkeypatch.chdir(tmp_path)

        (tmp_path / "fallback-dir").mkdir()
        (tmp_path / "fallback-dir" / "Atlas-2026.08.25.md").write_text("# Briefing\n")
        # "real-dir" (what QC_TEST_OUTPUT_DIR resolves to) is deliberately
        # never created.

        config_path = self._write_config(tmp_path, "${QC_TEST_OUTPUT_DIR:-fallback-dir}")

        rc = qc.main(["--config", str(config_path), "--date", "2026-08-25", "--no-judge"])
        assert rc == 1


class TestBadConfigExitsTwoViaRunnerLoadConfig:
    def test_malformed_yaml_exits_two_without_hard_process_exit(self, tmp_path, monkeypatch):
        """briefing_runner.load_config calls sys.exit(2) internally on bad
        YAML; main() must catch that SystemExit and return 2, not let it
        propagate and kill the interpreter (pytest would otherwise abort)."""
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("output_dir: [unclosed\n")  # malformed YAML

        rc = qc.main(["--config", str(config_path), "--date", "2026-08-25", "--no-judge"])
        assert rc == 2


# ---------------------------------------------------------------------------
# Config-driven judge dimensions.
#
# The six-dimension rubric doesn't universally apply: a pipeline like Atlas
# (defense/AI/space, deliberately not neighborhood-local) has no meaningful
# answer for tier_1_share or locality, and scoring it on those forever
# reports a permanently-red number nobody can act on -- and drags
# detect_quality_regression's baseline around for no reason. A pipeline
# opts into a narrower rubric via config["quality_check"]["judge"]["dimensions"].
# ---------------------------------------------------------------------------


class TestConfigDrivenJudgeDimensions:
    FOUR = ["lead_alignment", "actionability", "specificity", "freshness"]

    @staticmethod
    def _config(dims=None):
        if dims is None:
            return {}
        return {"quality_check": {"judge": {"dimensions": dims}}}

    def test_four_dim_config_prompt_asks_for_exactly_those_four(self):
        client = _fake_client(response=_scores_json({d: 2 for d in self.FOUR}))
        config = self._config(self.FOUR)

        findings, record = qc.judge_briefing("# Briefing", config, "atlas", date(2026, 8, 25), client)

        assert findings == []
        prompt = client.calls[0][0]
        for d in self.FOUR:
            assert d in prompt
        for excluded in ("tier_1_share", "locality"):
            assert excluded not in prompt
        assert set(record["scores"]) == set(self.FOUR)

    def test_total_denominator_reflects_configured_count(self):
        client = _fake_client(response=_scores_json({d: 2 for d in self.FOUR}))
        config = self._config(self.FOUR)

        _, record = qc.judge_briefing("# Briefing", config, "atlas", date(2026, 8, 25), client)

        assert record["total"] == 8
        assert record["max_total"] == 8  # 4 dims * 2 -- not the full 12

    def test_absent_config_scores_all_six_unchanged(self):
        client = _fake_client(response=_judge_json())

        _, record = qc.judge_briefing("# Briefing", {}, "local", date(2026, 8, 25), client)

        assert set(record["scores"]) == set(qc.RUBRIC_DIMENSIONS)
        assert record["total"] == 12
        assert record["max_total"] == 12
        prompt = client.calls[0][0]
        for d in qc.RUBRIC_DIMENSIONS:
            assert d in prompt

    def test_extra_unrequested_dimension_is_ignored(self):
        payload = {d: 2 for d in self.FOUR}
        payload["locality"] = 0  # model volunteers a dimension we didn't ask for
        client = _fake_client(response=_scores_json(payload))
        config = self._config(self.FOUR)

        findings, record = qc.judge_briefing("# Briefing", config, "atlas", date(2026, 8, 25), client)

        assert findings == []
        assert record is not None
        assert "locality" not in record["scores"]
        assert record["total"] == 8  # the volunteered locality:0 must not drag the total down
        assert record["max_total"] == 8

    def test_missing_requested_dimension_is_parse_failure(self):
        incomplete = {d: 2 for d in self.FOUR[:-1]}  # "freshness" never shows up
        client = _fake_client(response=_scores_json(incomplete))
        config = self._config(self.FOUR)

        findings, record = qc.judge_briefing("# Briefing", config, "atlas", date(2026, 8, 25), client)

        assert record is None
        assert len(findings) == 1
        assert findings[0].code == "judge-skipped"
        assert findings[0].severity == INFO

    def test_parse_judge_response_respects_dimensions_param_directly(self):
        payload = _scores_json({d: 1 for d in self.FOUR})
        parsed = qc.parse_judge_response(payload, dimensions=self.FOUR)
        assert set(parsed) == set(self.FOUR)

        # Requesting a superset than what's in the payload is a parse failure.
        assert qc.parse_judge_response(payload, dimensions=list(qc.RUBRIC_DIMENSIONS)) is None

    def test_unknown_dimension_name_raises_from_resolver(self):
        config = self._config(["lead_alignment", "made_up_dimension"])
        with pytest.raises(qc.InvalidJudgeDimension):
            qc.resolve_judge_dimensions(config)

    def test_unknown_dimension_via_judge_briefing_degrades_to_finding(self):
        client = _fake_client(response=_judge_json())
        config = self._config(["lead_alignment", "made_up_dimension"])

        findings, record = qc.judge_briefing("# Briefing", config, "atlas", date(2026, 8, 25), client)

        assert record is None
        assert len(findings) == 1
        assert findings[0].code == "judge-config-invalid"
        assert findings[0].severity == WARN
        assert findings[0].pipeline == "atlas"
        assert client.calls == []  # never even asked the model to score something undefined

    def test_run_checks_wires_per_pipeline_dimensions(self, tmp_path):
        """Two pipelines, two different configured rubrics, sharing one LLM
        client (client/backend selection is shared infra -- run_checks
        builds it once -- but judge_briefing resolves each pipeline's own
        dimensions fresh from its own config on every call, so the prompt
        content still differs per pipeline)."""
        atlas_dir = tmp_path / "atlas"
        local_dir = tmp_path / "local"
        atlas_dir.mkdir()
        local_dir.mkdir()
        (atlas_dir / "Atlas-2026.08.25.md").write_text("# Atlas Briefing\n")
        (local_dir / "Local-2026.08.25.md").write_text("# Local Briefing\n")

        configs = {
            "atlas": {
                "output_dir": str(atlas_dir),
                "file_naming": "Atlas-{yyyy}.{mm}.{dd}",
                "quality_check": {"judge": {"dimensions": self.FOUR}},
            },
            "local": {
                "output_dir": str(local_dir),
                "file_naming": "Local-{yyyy}.{mm}.{dd}",
            },
        }

        # A response covering all six dimensions -- parse_judge_response
        # picks out whichever subset each pipeline's own config requested
        # and ignores the rest, so one canned response serves both prompts.
        shared_client = _fake_client(response=_judge_json())

        findings, scores = qc.run_checks(
            configs,
            today=date(2026, 8, 25),
            **_no_op_layer1(),
            **_no_op_layer2(),
            build_client=lambda config: shared_client,
            score_history_loader=lambda pipeline, path=None: [],
        )

        assert set(scores["atlas"]["scores"]) == set(self.FOUR)
        assert scores["atlas"]["max_total"] == 8
        assert set(scores["local"]["scores"]) == set(qc.RUBRIC_DIMENSIONS)
        assert scores["local"]["max_total"] == 12

        assert len(shared_client.calls) == 2
        prompts = [c[0] for c in shared_client.calls]
        assert any("tier_1_share" not in p and "locality" not in p for p in prompts), "atlas prompt should exclude tier_1_share/locality"
        assert any("tier_1_share" in p and "locality" in p for p in prompts), "local prompt should request all six"

    def test_regression_skips_dimension_absent_from_older_history(self):
        def rec(idx, dims_scores):
            return {
                "ts": f"2026-08-{idx:02d}T06:00:00-07:00",
                "pipeline": "atlas",
                "date": f"2026-08-{idx:02d}",
                "scores": {d: {"score": s, "why": ""} for d, s in dims_scores.items()},
                "total": sum(dims_scores.values()),
                "max_total": len(dims_scores) * 2,
                "notes": "",
            }

        # 10 baseline runs scored only on the 4-dim Atlas rubric --
        # tier_1_share never appears in any of them.
        history = [rec(i, {d: 2 for d in self.FOUR}) for i in range(1, 11)]
        # 3 recent runs where tier_1_share suddenly appears (e.g. a config
        # change) scored catastrophically low.
        history += [
            rec(i, {**{d: 2 for d in self.FOUR}, "tier_1_share": 0}) for i in range(11, 14)
        ]

        findings = qc.detect_quality_regression(history, "atlas")

        fired = {(f.code, f.source) for f in findings}
        assert ("quality-regression", "tier_1_share") not in fired
        # The four dimensions with full history are flat (2 -> 2 the whole
        # way), so nothing should fire at all.
        assert findings == []
