"""The quality checker must notice when the LLM layer itself is failing.

For three weeks every medium- and light-tier call failed in every run, the
judge produced no score, and the only trace was a daily INFO line. These tests
cover the signals that should have caught it: the pipelines' own status files,
an escalating `judge-skipped`, a judge that can read a whole briefing and whose
unparseable output is kept, and the finance pipeline being checked at all.
"""

import json
from datetime import date, datetime
from pathlib import Path

import pytest
import yaml

import scripts.quality_check as qc
from scripts.quality_findings import CRITICAL, INFO, WARN, Finding


TODAY = date(2026, 10, 9)
ROOT = Path(__file__).resolve().parent.parent


def write_status(directory, name="status.json", when=TODAY, **fields):
    status = {
        "timestamp": datetime(when.year, when.month, when.day, 6, 0, 30).isoformat(),
        "intelligence_calls": {"calls": 14, "successful": 14, "failed": 0},
        "synthesis_degraded": False,
        "writer_fallback_count": 0,
        "email_sent": True,
        "errors": [],
    }
    status.update(fields)
    (Path(directory) / name).write_text(json.dumps(status))
    return status


def calls(total, failed):
    return {"calls": total, "successful": total - failed, "failed": failed}


def codes(findings):
    return [(f.severity, f.code) for f in findings]


# ---------------------------------------------------------------------------
# a. status files
# ---------------------------------------------------------------------------


class TestStatusHealth:
    def check(self, tmp_path, config=None, pipeline="atlas"):
        return qc.check_status_health(
            config or {}, pipeline, TODAY, status_dir=str(tmp_path)
        )

    def test_a_healthy_status_file_raises_nothing(self, tmp_path):
        write_status(tmp_path)
        assert self.check(tmp_path) == []

    @pytest.mark.parametrize(
        "total, failed, expected",
        [
            (20, 4, None),           # 20%
            (20, 5, WARN),           # exactly 25%
            (20, 9, WARN),           # 45%
            (20, 10, CRITICAL),      # exactly 50%
            (14, 12, CRITICAL),
            (3, 3, CRITICAL),        # every call failed
            (1, 1, CRITICAL),
            (0, 0, None),            # nothing was attempted
        ],
    )
    def test_failed_call_ratio_thresholds(self, tmp_path, total, failed, expected):
        write_status(tmp_path, intelligence_calls=calls(total, failed))
        findings = self.check(tmp_path)
        if expected is None:
            assert findings == []
        else:
            assert codes(findings) == [(expected, "llm-calls-failing")]
            assert f"{failed} of {total}" in findings[0].message
            assert findings[0].pipeline == "atlas"

    def test_thresholds_are_configurable(self, tmp_path):
        write_status(tmp_path, intelligence_calls=calls(20, 2))
        config = {"quality_check": {"status_health": {
            "failed_ratio_warn": 0.05, "failed_ratio_critical": 0.10,
        }}}
        assert codes(self.check(tmp_path, config)) == [(CRITICAL, "llm-calls-failing")]

    def test_can_be_switched_off(self, tmp_path):
        write_status(tmp_path, intelligence_calls=calls(3, 3))
        config = {"quality_check": {"status_health": {"enabled": False}}}
        assert self.check(tmp_path, config) == []

    def test_degraded_synthesis_is_critical(self, tmp_path):
        write_status(tmp_path, synthesis_degraded=True)
        assert codes(self.check(tmp_path)) == [(CRITICAL, "synthesis-degraded")]

    def test_writer_fallback_is_a_warning(self, tmp_path):
        write_status(tmp_path, writer_fallback_count=2)
        findings = self.check(tmp_path)
        assert codes(findings) == [(WARN, "writer-fallback")]
        assert "2" in findings[0].message

    def test_missing_status_file(self, tmp_path):
        findings = self.check(tmp_path)
        assert codes(findings) == [(WARN, "status-missing")]

    def test_unreadable_status_file(self, tmp_path):
        (tmp_path / "status.json").write_text("{not json")
        assert codes(self.check(tmp_path)) == [(WARN, "status-missing")]

    def test_status_from_an_earlier_day_is_stale_and_not_trusted(self, tmp_path):
        # Yesterday's file says everything failed; that describes yesterday.
        write_status(tmp_path, when=date(2026, 10, 8), intelligence_calls=calls(3, 3))
        findings = self.check(tmp_path)
        assert codes(findings) == [(WARN, "status-stale")]
        assert "2026-10-08" in findings[0].message

    def test_replaying_a_past_day_does_not_judge_it_by_a_newer_status(self, tmp_path):
        write_status(tmp_path, when=date(2026, 10, 12), intelligence_calls=calls(3, 3))
        assert self.check(tmp_path) == []

    def test_status_path_comes_from_the_pipelines_config(self, tmp_path):
        write_status(tmp_path, name="status-finance.json", writer_fallback_count=1)
        config = {"status_file_path": "status-finance.json"}
        findings = self.check(tmp_path, config, pipeline="finance")
        assert codes(findings) == [(WARN, "writer-fallback")]
        assert findings[0].pipeline == "finance"

    def test_no_email_is_not_a_fault(self, tmp_path):
        # Finance is written to disk only, by design.
        write_status(tmp_path, email_sent=False)
        assert self.check(tmp_path) == []


def base_run(tmp_path, pipelines=("atlas",), **overrides):
    """run_checks arguments for briefings that exist and layers that are quiet."""
    configs = {}
    for name in pipelines:
        out = tmp_path / name
        out.mkdir(exist_ok=True)
        (out / "B-2026.10.09.md").write_text("# Briefing\n\n## Executive Summary\n\nText.\n")
        configs[name] = {
            "output_dir": str(out),
            "file_naming": "B-{yyyy}.{mm}.{dd}",
            "status_file_path": f"status-{name}.json",
        }
    kwargs = dict(
        today=TODAY,
        harvest_journal=lambda **kw: [],
        append_history=lambda records, path=None: 0,
        load_history=lambda path=None, since=None: [],
        detect_rot=lambda history, probes=None, rules=None, live_sources=None: [],
        check_report=lambda markdown, config, today=None, pipeline="": [],
        streaks_path=str(tmp_path / "streaks.json"),
        judge_raw_dir=str(tmp_path / "raw"),
        scores_path=str(tmp_path / "scores.jsonl"),
    )
    kwargs.update(overrides)
    return configs, kwargs


class TestStatusHealthInRunChecks:
    def test_status_findings_reach_the_digest(self, tmp_path):
        configs, kwargs = base_run(tmp_path, no_judge=True, status_dir=str(tmp_path))
        write_status(tmp_path, name="status-atlas.json", intelligence_calls=calls(10, 10))
        findings, _ = qc.run_checks(configs, **kwargs)
        assert (CRITICAL, "llm-calls-failing") in codes(findings)

    def test_status_layer_is_off_unless_given_a_directory(self, tmp_path):
        configs, kwargs = base_run(tmp_path, no_judge=True)
        findings, _ = qc.run_checks(configs, **kwargs)
        assert findings == []

    def test_degraded_synthesis_is_not_reported_twice(self, tmp_path):
        """The rendered placeholder and the status flag are one defect."""
        placeholder = Finding(
            CRITICAL, "degraded-content", "placeholder", source="Executive Summary",
            pipeline="atlas",
        )
        configs, kwargs = base_run(
            tmp_path, no_judge=True, status_dir=str(tmp_path),
            check_report=lambda markdown, config, today=None, pipeline="": [placeholder],
        )
        write_status(tmp_path, name="status-atlas.json", synthesis_degraded=True)
        findings, _ = qc.run_checks(configs, **kwargs)
        assert codes(findings) == [(CRITICAL, "degraded-content")]

    def test_a_missing_briefing_with_a_stale_status_is_one_finding(self, tmp_path):
        configs, kwargs = base_run(tmp_path, no_judge=True, status_dir=str(tmp_path))
        (tmp_path / "atlas" / "B-2026.10.09.md").unlink()
        write_status(tmp_path, name="status-atlas.json", when=date(2026, 10, 8))
        findings, _ = qc.run_checks(configs, **kwargs)
        assert codes(findings) == [(CRITICAL, "briefing-missing")]

    def test_main_reads_status_files_from_the_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "briefings").mkdir()
        today = date.today()
        (tmp_path / "briefings" / f"A-{today:%Y.%m.%d}.md").write_text("# B\n")
        (tmp_path / "config.yaml").write_text(
            "output_dir: briefings\nfile_naming: 'A-{yyyy}.{mm}.{dd}'\n"
        )
        write_status(tmp_path, when=today, intelligence_calls=calls(4, 4))
        seen = {}
        real = qc.run_checks

        def spy(configs, **kwargs):
            seen.update(kwargs)
            return real(
                configs, **{**kwargs, "harvest_journal": lambda **kw: [],
                            "detect_rot": lambda *a, **k: []}
            )

        monkeypatch.setattr(qc, "run_checks", spy)
        rc = qc.main(["--config", str(tmp_path / "config.yaml"), "--dry-run", "--no-judge"])
        assert seen["status_dir"] == "."
        assert rc == 1  # the CRITICAL from the status file


# ---------------------------------------------------------------------------
# b. judge-skipped escalates
# ---------------------------------------------------------------------------


class _Client:
    available = True

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt, tier="medium", system_prompt=None, **kw):
        self.prompts.append(prompt)
        return self.response


GOOD = json.dumps({
    dim: {"score": 2, "why": "ok"} for dim in qc.RUBRIC_DIMENSIONS
})


def run_day(tmp_path, day, response, pipelines=("atlas",), **overrides):
    configs, kwargs = base_run(tmp_path, pipelines=pipelines, **overrides)
    for name in pipelines:
        (tmp_path / name / f"B-{day:%Y.%m.%d}.md").write_text("# Briefing\n")
    kwargs["today"] = day
    if "build_client" not in kwargs:
        kwargs["build_client"] = lambda config: _Client(response)
    findings, records = qc.run_checks(configs, **kwargs)
    return [f for f in findings if f.code == "judge-skipped"], records


class TestJudgeSkippedEscalation:
    def test_info_then_warn_then_critical(self, tmp_path):
        severities = []
        for offset in range(4):
            skipped, _ = run_day(tmp_path, date(2026, 10, 9 + offset), "not json")
            assert len(skipped) == 1
            severities.append(skipped[0].severity)
        assert severities == [INFO, WARN, CRITICAL, CRITICAL]

    def test_message_says_how_long(self, tmp_path):
        run_day(tmp_path, date(2026, 10, 9), "not json")
        skipped, _ = run_day(tmp_path, date(2026, 10, 10), "not json")
        assert "2 consecutive" in skipped[0].message

    def test_a_scored_day_resets_the_count(self, tmp_path):
        run_day(tmp_path, date(2026, 10, 9), "not json")
        run_day(tmp_path, date(2026, 10, 10), "not json")
        skipped, records = run_day(tmp_path, date(2026, 10, 11), GOOD)
        assert skipped == [] and "atlas" in records
        skipped, _ = run_day(tmp_path, date(2026, 10, 12), "not json")
        assert skipped[0].severity == INFO

    def test_a_second_run_on_the_same_day_does_not_count_twice(self, tmp_path):
        for _ in range(3):
            skipped, _ = run_day(tmp_path, date(2026, 10, 9), "not json")
        assert skipped[0].severity == INFO

    def test_counted_per_pipeline(self, tmp_path):
        run_day(tmp_path, date(2026, 10, 9), "not json", pipelines=("atlas",))
        skipped, _ = run_day(
            tmp_path, date(2026, 10, 10), "not json", pipelines=("atlas", "local")
        )
        assert {f.pipeline: f.severity for f in skipped} == {"atlas": WARN, "local": INFO}

    def test_no_client_at_all_escalates_too(self, tmp_path):
        severities = []
        for offset in range(3):
            skipped, _ = run_day(
                tmp_path, date(2026, 10, 9 + offset), None,
                build_client=lambda config: None,
            )
            assert [f.pipeline for f in skipped] == ["atlas"]
            severities.append(skipped[0].severity)
        assert severities == [INFO, WARN, CRITICAL]

    def test_thresholds_are_configurable(self, tmp_path):
        configs, kwargs = base_run(tmp_path)
        configs["atlas"]["quality_check"] = {
            "judge": {"skip_warn_days": 1, "skip_critical_days": 2}
        }
        kwargs["build_client"] = lambda config: _Client("not json")
        findings, _ = qc.run_checks(configs, **kwargs)
        assert [f.severity for f in findings if f.code == "judge-skipped"] == [WARN]

    def test_first_run_starts_from_one_whatever_history_says(self, tmp_path):
        """No backfill: weeks of old digests must not page on day one."""
        scores = tmp_path / "scores.jsonl"
        scores.write_text(json.dumps({
            "pipeline": "atlas", "date": "2026-09-12", "scores": {}, "total": 8,
        }) + "\n")
        skipped, _ = run_day(tmp_path, date(2026, 10, 9), "not json")
        assert skipped[0].severity == INFO

    def test_dry_run_does_not_advance_the_count(self, tmp_path):
        for _ in range(3):
            skipped, _ = run_day(
                tmp_path, date(2026, 10, 9), "not json", dry_run=True
            )
        assert not (tmp_path / "streaks.json").exists()
        skipped, _ = run_day(tmp_path, date(2026, 10, 10), "not json")
        assert skipped[0].severity == INFO

    def test_a_pipeline_with_no_briefing_keeps_its_count(self, tmp_path):
        run_day(tmp_path, date(2026, 10, 9), "not json")
        # Day two: nothing to judge. That is briefing-missing's job to report.
        configs, kwargs = base_run(tmp_path)
        (tmp_path / "atlas" / "B-2026.10.09.md").unlink()
        kwargs["today"] = date(2026, 10, 10)
        kwargs["build_client"] = lambda config: _Client("not json")
        qc.run_checks(configs, **kwargs)
        skipped, _ = run_day(tmp_path, date(2026, 10, 11), "not json")
        assert skipped[0].severity == WARN


# ---------------------------------------------------------------------------
# c. the judge's input and output
# ---------------------------------------------------------------------------


def briefing(body_chars=0, tail="LAST CONTENT LINE"):
    filler = "\n".join(f"Line {i} of ordinary briefing prose." for i in range(body_chars // 36))
    return (
        "# Atlas Briefing\n\n## Executive Summary\n\nLead.\n\n## News\n\n"
        f"{filler}\n{tail}\n\n\n---\n\n"
        "## OpenRouter Usage Summary\n\n| Tier | Model |\n|---|---|\n| heavy | x |\n\n"
        "---\n\n## Codex Chain Usage Summary\n\n**14 calls**, 0 failed.\n\n"
        "---\n\n## Codex Usage Summary\n\n**1 call**.\n\n"
        "## API Key Rotation Summary\n\n| key | uses |\n"
    )


class TestJudgeExcerpt:
    def test_usage_footers_are_left_out(self):
        excerpt = qc.judge_excerpt(briefing())
        assert "LAST CONTENT LINE" in excerpt
        for footer in ("Usage Summary", "Key Rotation", "14 calls", "| heavy |"):
            assert footer not in excerpt
        assert not excerpt.rstrip().endswith("---")

    def test_a_normal_briefing_fits_whole(self):
        # The main briefing runs 14-26k characters before its footers.
        text = briefing(body_chars=26_000)
        excerpt = qc.judge_excerpt(text)
        assert "LAST CONTENT LINE" in excerpt
        assert "Line 0 of" in excerpt

    def test_an_oversized_briefing_is_cut_at_a_line_boundary(self):
        text = briefing(body_chars=30_000)
        excerpt = qc.judge_excerpt(text, max_chars=5_000)
        assert len(excerpt) <= 5_000
        assert excerpt.endswith("of ordinary briefing prose.")

    def test_sections_after_a_footer_are_kept(self):
        text = "## A\n\none\n\n## Codex Usage Summary\n\nnumbers\n\n## Errors\n\nfeed X failed\n"
        excerpt = qc.judge_excerpt(text)
        assert "feed X failed" in excerpt and "numbers" not in excerpt

    def test_the_fence_survives_in_the_prompt(self):
        client = _Client(GOOD)
        qc.judge_briefing(briefing(body_chars=20_000), {}, "atlas", TODAY, client)
        prompt = client.prompts[0]
        assert prompt.rstrip().endswith(">>>")
        assert "LAST CONTENT LINE" in prompt
        assert "Usage Summary" not in prompt

    def test_window_is_configurable(self):
        client = _Client(GOOD)
        config = {"quality_check": {"judge": {"max_chars": 2_000}}}
        qc.judge_briefing(briefing(body_chars=20_000), config, "atlas", TODAY, client)
        assert "LAST CONTENT LINE" not in client.prompts[0]


class TestExtractJsonObject:
    def test_plain_object(self):
        assert json.loads(qc._extract_json_object('{"a": 1}')) == {"a": 1}

    def test_takes_the_last_complete_object_not_first_brace_to_last_brace(self):
        text = 'Shape: {"a": {"score": 0}} and my answer: {"a": {"score": 2}} done'
        assert json.loads(qc._extract_json_object(text)) == {"a": {"score": 2}}

    def test_trailing_stray_brace_does_not_poison_the_object(self):
        text = '{"a": {"score": 1, "why": "x"}}\n\nNote: unmatched } in prose'
        assert json.loads(qc._extract_json_object(text)) == {"a": {"score": 1, "why": "x"}}

    def test_braces_inside_strings_are_not_structure(self):
        text = 'Answer {"a": {"score": 2, "why": "uses {braces} and \\"quotes\\" }"}}'
        assert json.loads(qc._extract_json_object(text))["a"]["score"] == 2

    def test_fenced_object(self):
        text = '```json\n{"a": {"score": 2}}\n```'
        assert json.loads(qc._extract_json_object(text)) == {"a": {"score": 2}}

    def test_truncated_final_object_falls_back_to_the_last_complete_one(self):
        text = '{"a": 1} then {"b": {"score": '
        assert json.loads(qc._extract_json_object(text)) == {"a": 1}

    def test_brace_balanced_prose_is_not_mistaken_for_the_answer(self):
        text = '{"a": {"score": 2}} (see rubric {tier one})'
        assert json.loads(qc._extract_json_object(text)) == {"a": {"score": 2}}

    def test_nothing_usable(self):
        assert qc._extract_json_object("no json here") is None
        assert qc._extract_json_object("") is None
        assert qc._extract_json_object("{ never closed") is None


class TestJudgeRawOutputIsKept:
    def test_unparseable_output_is_written_and_its_path_reported(self, tmp_path, caplog):
        raw = "I think this briefing is quite good overall! " * 400  # ~18KB
        skipped, _ = run_day(tmp_path, TODAY, raw)
        saved = list((tmp_path / "raw").glob("judge-raw-atlas-2026-10-09*.txt"))
        assert len(saved) == 1
        text = saved[0].read_text()
        assert text.startswith("I think this briefing")
        assert len(text.encode()) <= 8_500  # truncated to a few KB
        assert str(saved[0]) in skipped[0].message
        assert str(saved[0]) in caplog.text

    def test_nothing_is_written_when_the_judge_scores(self, tmp_path):
        run_day(tmp_path, TODAY, GOOD)
        assert not (tmp_path / "raw").exists()

    def test_nothing_is_written_when_the_call_returns_nothing(self, tmp_path):
        skipped, _ = run_day(tmp_path, TODAY, None)
        assert skipped and not (tmp_path / "raw").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        skipped, _ = run_day(tmp_path, TODAY, "not json", dry_run=True)
        assert skipped and not (tmp_path / "raw").exists()

    def test_raw_text_does_not_leak_into_the_digest(self, tmp_path):
        skipped, _ = run_day(tmp_path, TODAY, "SECRET-ISH MODEL RAMBLING")
        digest = qc.render_digest(skipped, {}, TODAY)
        assert "SECRET-ISH" not in digest
        assert "raw" not in skipped[0].detail


# ---------------------------------------------------------------------------
# d. finance is covered
# ---------------------------------------------------------------------------


class TestFinanceIsCovered:
    @pytest.fixture(scope="class")
    def finance(self):
        return yaml.safe_load((ROOT / "config_finance.yaml").read_text())

    def test_finance_has_a_quality_check_block(self, finance):
        block = finance["quality_check"]
        dims = qc.resolve_judge_dimensions(finance)
        # No events and no neighbourhood: the local-only dimensions are out.
        assert "locality" not in dims and "tier_1_share" not in dims
        assert set(dims) <= set(qc.RUBRIC_DIMENSIONS) and len(dims) >= 3
        assert block["alert_email"]["recipients"]

    def test_finance_floors_name_sections_it_renders(self, finance):
        floors = finance["quality_check"].get("section_floors", {})
        assert set(floors) <= set(finance["section_order"])

    def test_finance_alerts_have_somewhere_to_go(self, finance, monkeypatch):
        """Its briefing is emailed to nobody, by design; its alerts are not."""
        assert finance["email_recipients"] == []
        raw = finance["quality_check"]["alert_email"]["recipients"]
        assert any("RECIPIENT_EMAIL" in str(entry) for entry in raw)

    def test_wrapper_checks_all_three_pipelines_by_default(self):
        script = (ROOT / "run_quality_check.sh").read_text()
        for name in ("config.yaml", "config_local.yaml", "config_finance.yaml"):
            assert f'"$DIR/{name}"' in script

    def test_finance_run_chains_its_own_check(self):
        """The 07:00 chained check finishes before finance (07:10) has run, so
        finance is audited when finance finishes -- not on a clock."""
        finance = (ROOT / "run_finance_briefing.sh").read_text()
        assert "run_quality_check.sh" in finance
        assert "config_finance.yaml" in finance
        main = (ROOT / "run_briefing.sh").read_text()
        call = main[main.index('"$DIR/run_quality_check.sh"'):]
        assert "config_finance.yaml" not in call.split("\n\n")[0]

    def test_local_only_checks_skip_cleanly_on_a_finance_briefing(self, finance):
        from scripts.report_invariants import check_report

        markdown = (
            "# Personal Finance Briefing\n\n## Executive Summary\n\n**Rates held.** "
            "On Sept. 3 the Fed kept rates steady.\n\n## Finance News\n\n"
            "**[A](https://a.example/1)** *(Src)*\nOne.\n\n"
            "**[B](https://b.example/2)** *(Src)*\nTwo.\n\n"
            "**[C](https://c.example/3)** *(Src)*\nThree.\n"
        )
        findings = check_report(markdown, finance, today=TODAY, pipeline="finance")
        assert [f for f in findings if f.code in ("stale-event", "out-of-area")] == []
        assert [f for f in findings if f.severity == CRITICAL] == []

    def test_journal_units_follow_the_pipelines_being_checked(self, tmp_path):
        seen = {}

        def harvest(**kwargs):
            seen.update(kwargs)
            return []

        configs, kwargs = base_run(
            tmp_path, pipelines=("finance",), no_judge=True, harvest_journal=harvest
        )
        qc.run_checks(configs, **kwargs)
        assert list(seen["units"]) == ["finance-briefing"]

        configs, kwargs = base_run(
            tmp_path, pipelines=("atlas", "local"), no_judge=True, harvest_journal=harvest
        )
        qc.run_checks(configs, **kwargs)
        assert list(seen["units"]) == ["atlas-briefing", "local-briefing"]

    def test_another_pipelines_history_is_not_reported_in_this_run(self, tmp_path):
        history = [
            {"ts": "2026-10-09", "pipeline": "atlas", "kind": "feed", "name": "A"},
            {"ts": "2026-10-09", "pipeline": "finance", "kind": "feed", "name": "F"},
        ]
        seen = {}

        def detect(history, probes=None, rules=None, live_sources=None):
            seen["pipelines"] = sorted({r["pipeline"] for r in history})
            return []

        configs, kwargs = base_run(
            tmp_path, pipelines=("finance",), no_judge=True,
            load_history=lambda path=None, since=None: history, detect_rot=detect,
        )
        qc.run_checks(configs, **kwargs)
        assert seen["pipelines"] == ["finance"]

    def test_a_subset_run_keeps_its_own_digest(self, tmp_path, monkeypatch):
        """The finance check runs minutes after the main one and must not
        overwrite that morning's digest."""
        assert qc.digest_path_for(TODAY) == "logs/quality-digest-2026-10-09.md"
        assert (
            qc.digest_path_for(TODAY, suffix="finance")
            == "logs/quality-digest-2026-10-09-finance.md"
        )
        args = qc.build_arg_parser().parse_args(["--digest-suffix", "finance"])
        assert args.digest_suffix == "finance"
