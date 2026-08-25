# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for Layer 1 of the daily quality check: source health."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.quality_findings import CRITICAL, INFO, WARN
from scripts.source_health import (
    append_history,
    detect_rot,
    harvest_journal,
    load_history,
    main,
    probe_feed,
    probe_feeds,
)

# ---------------------------------------------------------------------------
# Fixtures: real log lines, verified live on 2026-08-25
# ---------------------------------------------------------------------------

LOG_FEED_YIELD = (
    "2026-08-25T06:15:23-0700 eric-NUC7i7BNHX local-briefing[2251581]: "
    "INFO: Found 8 articles from FOX 5 San Diego"
)
LOG_QUERY_YIELD = (
    "2026-08-25T06:15:59-0700 eric-NUC7i7BNHX local-briefing[2251581]: "
    "INFO: Found 3 articles for query: Crown Point San Diego"
)
LOG_FEED_PARSE_ISSUE = (
    "2026-08-25T06:01:14-0700 eric-NUC7i7BNHX atlas-briefing[2237141]: "
    "WARNING: Feed parsing issue for Anthropic: text/html; charset=utf-8 is not an XML media type"
)
LOG_FEED_SCAN_FAILED = (
    "2026-08-25T06:02:32-0700 eric-NUC7i7BNHX atlas-briefing[2237141]: "
    "WARNING: Feed scan failed for SomeFeed: timeout"
)


def _mock_completed(stdout="", returncode=0, stderr=""):
    return SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)


# ---------------------------------------------------------------------------
# Journald harvesting / parsing
# ---------------------------------------------------------------------------


class TestHarvestJournalParsing:
    def test_feed_yield_line(self):
        with patch("subprocess.run", return_value=_mock_completed(LOG_FEED_YIELD)):
            records = harvest_journal()
        assert len(records) == 1
        r = records[0]
        assert r["kind"] == "feed"
        assert r["name"] == "FOX 5 San Diego"
        assert r["yield"] == 8
        assert r["hard_error"] is None
        assert r["pipeline"] == "local"

    def test_query_yield_line(self):
        with patch("subprocess.run", return_value=_mock_completed(LOG_QUERY_YIELD)):
            records = harvest_journal()
        assert len(records) == 1
        r = records[0]
        assert r["kind"] == "query"
        assert r["name"] == "Crown Point San Diego"
        assert r["yield"] == 3
        assert r["pipeline"] == "local"

    def test_feed_parsing_issue_line(self):
        with patch("subprocess.run", return_value=_mock_completed(LOG_FEED_PARSE_ISSUE)):
            records = harvest_journal()
        assert len(records) == 1
        r = records[0]
        assert r["kind"] == "feed"
        assert r["name"] == "Anthropic"
        assert r["yield"] is None
        assert "not an XML media type" in r["hard_error"]
        assert r["pipeline"] == "atlas"

    def test_feed_scan_failed_line(self):
        with patch("subprocess.run", return_value=_mock_completed(LOG_FEED_SCAN_FAILED)):
            records = harvest_journal()
        assert len(records) == 1
        r = records[0]
        assert r["kind"] == "feed"
        assert r["name"] == "SomeFeed"
        assert r["hard_error"] == "timeout"
        assert r["pipeline"] == "atlas"

    def test_pipeline_mapping_from_systemd_tag(self):
        text = LOG_FEED_YIELD + "\n" + LOG_FEED_PARSE_ISSUE
        with patch("subprocess.run", return_value=_mock_completed(text)):
            records = harvest_journal()
        pipelines = {r["pipeline"] for r in records}
        assert pipelines == {"local", "atlas"}

    def test_malformed_lines_ignored(self):
        text = "\n".join(
            [
                "this is not a journald line at all",
                "2026-08-25T06:15:23-0700 host unknown-service[123]: INFO: Found 5 articles from X",
                "",
                LOG_FEED_YIELD,
            ]
        )
        with patch("subprocess.run", return_value=_mock_completed(text)):
            records = harvest_journal()
        # Only the real local-briefing line should survive; the unknown
        # systemd unit and the garbage line are silently dropped.
        assert len(records) == 1
        assert records[0]["name"] == "FOX 5 San Diego"

    def test_parse_issue_and_yield_merge_into_one_record(self):
        """Same feed, same run: a bozo warning followed by a Found line."""
        text = (
            "2026-08-25T06:01:14-0700 h atlas-briefing[1]: WARNING: "
            "Feed parsing issue for Anthropic: bad xml\n"
            "2026-08-25T06:01:15-0700 h atlas-briefing[1]: INFO: "
            "Found 0 articles from Anthropic"
        )
        with patch("subprocess.run", return_value=_mock_completed(text)):
            records = harvest_journal()
        assert len(records) == 1
        r = records[0]
        assert r["name"] == "Anthropic"
        assert r["yield"] == 0
        assert r["hard_error"] == "bad xml"

    def test_different_days_stay_separate_records(self):
        text = (
            "2026-08-24T06:15:23-0700 h local-briefing[1]: INFO: Found 2 articles from X\n"
            "2026-08-25T06:15:23-0700 h local-briefing[1]: INFO: Found 4 articles from X"
        )
        with patch("subprocess.run", return_value=_mock_completed(text)):
            records = harvest_journal()
        assert len(records) == 2
        assert {r["yield"] for r in records} == {2, 4}


class TestHarvestJournalGracefulDegradation:
    def test_missing_journalctl_returns_empty_list(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            records = harvest_journal()
        assert records == []

    def test_timeout_returns_empty_list(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="journalctl", timeout=180)):
            records = harvest_journal()
        assert records == []

    def test_nonzero_exit_returns_empty_list(self):
        with patch("subprocess.run", return_value=_mock_completed("", returncode=1, stderr="boom")):
            records = harvest_journal()
        assert records == []

    def test_empty_stdout_returns_empty_list(self):
        with patch("subprocess.run", return_value=_mock_completed("")):
            records = harvest_journal()
        assert records == []

    def test_never_raises_on_generic_os_error(self):
        with patch("subprocess.run", side_effect=OSError("weird failure")):
            records = harvest_journal()  # must not raise
        assert records == []


# ---------------------------------------------------------------------------
# History round-trip
# ---------------------------------------------------------------------------


class TestHistoryRoundTrip:
    def test_append_then_load(self, tmp_path):
        path = tmp_path / "source-health.jsonl"
        records = [
            {"ts": "2026-08-25T06:15:23-07:00", "pipeline": "local", "kind": "feed",
             "name": "X", "yield": 3, "hard_error": None, "newest_entry": None},
            {"ts": "2026-08-25T06:15:24-07:00", "pipeline": "local", "kind": "feed",
             "name": "Y", "yield": 0, "hard_error": "timeout", "newest_entry": None},
        ]
        n = append_history(records, path=str(path))
        assert n == 2
        loaded = load_history(path=str(path))
        assert loaded == records

    def test_append_is_cumulative(self, tmp_path):
        path = tmp_path / "h.jsonl"
        append_history([{"ts": "2026-08-20T00:00:00-07:00", "pipeline": "atlas", "kind": "feed",
                          "name": "A", "yield": 1, "hard_error": None, "newest_entry": None}], path=str(path))
        append_history([{"ts": "2026-08-21T00:00:00-07:00", "pipeline": "atlas", "kind": "feed",
                          "name": "A", "yield": 2, "hard_error": None, "newest_entry": None}], path=str(path))
        loaded = load_history(path=str(path))
        assert len(loaded) == 2

    def test_missing_file_returns_empty_list(self, tmp_path):
        assert load_history(path=str(tmp_path / "nope.jsonl")) == []

    def test_malformed_line_skipped(self, tmp_path):
        path = tmp_path / "h.jsonl"
        path.write_text(
            '{"ts": "2026-08-25T00:00:00-07:00", "pipeline": "atlas", "kind": "feed", '
            '"name": "A", "yield": 1, "hard_error": null, "newest_entry": null}\n'
            "not valid json at all\n"
        )
        loaded = load_history(path=str(path))
        assert len(loaded) == 1
        assert loaded[0]["name"] == "A"

    def test_empty_append_is_noop(self, tmp_path):
        path = tmp_path / "h.jsonl"
        assert append_history([], path=str(path)) == 0
        assert not path.exists()

    def test_since_filter(self, tmp_path):
        path = tmp_path / "h.jsonl"
        records = [
            {"ts": "2026-01-01T00:00:00-07:00", "pipeline": "atlas", "kind": "feed",
             "name": "old", "yield": 1, "hard_error": None, "newest_entry": None},
            {"ts": "2026-08-25T00:00:00-07:00", "pipeline": "atlas", "kind": "feed",
             "name": "new", "yield": 1, "hard_error": None, "newest_entry": None},
        ]
        append_history(records, path=str(path))
        loaded = load_history(path=str(path), since="2026-08-01T00:00:00-07:00")
        assert [r["name"] for r in loaded] == ["new"]


# ---------------------------------------------------------------------------
# probe_feed / probe_feeds
# ---------------------------------------------------------------------------


class TestProbeFeed:
    def test_dead_url_404(self):
        resp = SimpleNamespace(status_code=404, headers={"Content-Type": "text/html"}, content=b"")
        with patch("requests.get", return_value=resp), \
             patch("feedparser.parse", return_value=SimpleNamespace(entries=[])):
            result = probe_feed("Anthropic", "https://example.com/rss.xml")
        assert result["status"] == 404
        assert result["error"] is None
        assert result["entries"] == 0

    def test_healthy_feed_with_newest_entry(self):
        resp = SimpleNamespace(status_code=200, headers={"Content-Type": "application/rss+xml"}, content=b"<rss/>")
        entry = {"published_parsed": (2024, 3, 29, 11, 3, 0, 0, 0, 0)}
        with patch("requests.get", return_value=resp), \
             patch("feedparser.parse", return_value=SimpleNamespace(entries=[entry])):
            result = probe_feed("Google AI Blog", "https://example.com/rss")
        assert result["status"] == 200
        assert result["entries"] == 1
        assert result["newest_entry"].startswith("2024-03-29")
        assert result["error"] is None

    def test_network_exception_never_raises(self):
        import requests as requests_module

        with patch("requests.get", side_effect=requests_module.exceptions.ConnectionError("refused")):
            result = probe_feed("Broken", "https://example.com/rss")
        assert result["error"] == "refused" or "refused" in result["error"]
        assert result["status"] is None

    def test_probe_feeds_iterates_all(self):
        resp = SimpleNamespace(status_code=200, headers={"Content-Type": "application/xml"}, content=b"<rss/>")
        with patch("requests.get", return_value=resp), \
             patch("feedparser.parse", return_value=SimpleNamespace(entries=[])):
            results = probe_feeds([{"name": "A", "url": "u1"}, {"name": "B", "url": "u2"}])
        assert [r["name"] for r in results] == ["A", "B"]


# ---------------------------------------------------------------------------
# detect_rot — Mode A: dead URL
# ---------------------------------------------------------------------------


def _feed_record(ts, name, yield_=None, hard_error=None, pipeline="atlas"):
    return {"ts": ts, "pipeline": pipeline, "kind": "feed", "name": name,
            "yield": yield_, "hard_error": hard_error, "newest_entry": None}


def _query_record(ts, name, yield_, pipeline="atlas"):
    return {"ts": ts, "pipeline": pipeline, "kind": "query", "name": name,
            "yield": yield_, "hard_error": None, "newest_entry": None}


def _zone_record(ts, yield_=None, hard_error=None, pipeline="atlas"):
    return {"ts": ts, "pipeline": pipeline, "kind": "alerts_zone", "name": "alerts",
            "yield": yield_, "hard_error": hard_error, "newest_entry": None}


class TestDetectRotDeadUrl:
    def test_probe_bad_first_sighting_is_warn(self):
        history = [_feed_record("2026-08-25T06:00:00-07:00", "Anthropic", yield_=0)]
        probes = [{"name": "Anthropic", "url": "u", "status": 404, "content_type": "text/html",
                   "entries": 0, "newest_entry": None, "error": None}]
        findings = detect_rot(history, probes=probes)
        dead = [f for f in findings if f.code == "feed-dead-url"]
        assert len(dead) == 1
        assert dead[0].severity == WARN
        assert "404" in dead[0].message

    def test_three_consecutive_hard_errors_is_critical(self):
        history = [
            _feed_record("2026-08-23T06:00:00-07:00", "Anthropic", hard_error="404 Not Found"),
            _feed_record("2026-08-24T06:00:00-07:00", "Anthropic", hard_error="404 Not Found"),
            _feed_record("2026-08-25T06:00:00-07:00", "Anthropic", hard_error="404 Not Found"),
        ]
        findings = detect_rot(history)
        dead = [f for f in findings if f.code == "feed-dead-url"]
        assert len(dead) == 1
        assert dead[0].severity == CRITICAL
        assert dead[0].source == "Anthropic"

    def test_two_consecutive_hard_errors_no_probe_does_not_fire(self):
        history = [
            _feed_record("2026-08-24T06:00:00-07:00", "Anthropic", hard_error="timeout"),
            _feed_record("2026-08-25T06:00:00-07:00", "Anthropic", hard_error="timeout"),
        ]
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "feed-dead-url"] == []

    def test_error_streak_broken_by_success_does_not_fire(self):
        history = [
            _feed_record("2026-08-22T06:00:00-07:00", "X", hard_error="timeout"),
            _feed_record("2026-08-23T06:00:00-07:00", "X", hard_error="timeout"),
            _feed_record("2026-08-24T06:00:00-07:00", "X", yield_=5),
            _feed_record("2026-08-25T06:00:00-07:00", "X", hard_error="timeout"),
        ]
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "feed-dead-url"] == []

    def test_healthy_probe_no_history_produces_nothing(self):
        probes = [{"name": "Karpathy", "url": "u", "status": 200, "content_type": "application/xml",
                   "entries": 3, "newest_entry": "2026-08-01T00:00:00+00:00", "error": None}]
        findings = detect_rot([], probes=probes)
        assert [f for f in findings if f.code == "feed-dead-url"] == []

    def test_deepmind_case_cosmetic_error_every_run_but_recent_yield_is_silent(self):
        """
        A feed that logs a harmless encoding-mismatch warning on every run
        while still delivering articles must NOT be flagged dead. The
        harvester merges the bozo warning and the "Found N articles" line
        from the same run into one record carrying both hard_error and a
        nonzero yield — that run is a success and must reset the streak.
        """
        history = [
            _feed_record("2026-08-20T06:00:00-07:00", "DeepMind",
                         yield_=0, hard_error="document declared as us-ascii, but parsed as utf-8"),
            _feed_record("2026-08-21T06:00:00-07:00", "DeepMind",
                         yield_=0, hard_error="document declared as us-ascii, but parsed as utf-8"),
            _feed_record("2026-08-22T06:00:00-07:00", "DeepMind",
                         yield_=0, hard_error="document declared as us-ascii, but parsed as utf-8"),
            _feed_record("2026-08-23T06:00:00-07:00", "DeepMind",
                         yield_=2, hard_error="document declared as us-ascii, but parsed as utf-8"),
            _feed_record("2026-08-24T06:00:00-07:00", "DeepMind",
                         yield_=0, hard_error="document declared as us-ascii, but parsed as utf-8"),
            _feed_record("2026-08-25T06:00:00-07:00", "DeepMind",
                         yield_=1, hard_error="document declared as us-ascii, but parsed as utf-8"),
        ]
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "feed-dead-url"] == []

    def test_anthropic_case_error_and_zero_yield_every_run_is_critical(self):
        """Error AND zero yield, every single run — genuinely dead."""
        history = [
            _feed_record(f"2026-08-{d:02d}T06:00:00-07:00", "Anthropic",
                         yield_=0, hard_error="text/html; charset=utf-8 is not an XML media type")
            for d in range(1, 24)
        ]
        findings = detect_rot(history)
        dead = [f for f in findings if f.code == "feed-dead-url"]
        assert len(dead) == 1
        assert dead[0].severity == CRITICAL
        assert dead[0].source == "Anthropic"
        assert dead[0].detail["consecutive_error_runs"] == 23

    def test_streak_only_counts_from_the_break_forward(self):
        """
        error+zero on every run except one successful run mid-window: the
        streak must be measured only from the break forward, so with a
        default threshold of 3 and only 2 error+zero runs after the
        success, this stays below threshold and does not fire.
        """
        history = [
            _feed_record("2026-08-20T06:00:00-07:00", "Y", yield_=0, hard_error="timeout"),
            _feed_record("2026-08-21T06:00:00-07:00", "Y", yield_=0, hard_error="timeout"),
            _feed_record("2026-08-22T06:00:00-07:00", "Y", yield_=0, hard_error="timeout"),
            _feed_record("2026-08-23T06:00:00-07:00", "Y", yield_=4),  # the break
            _feed_record("2026-08-24T06:00:00-07:00", "Y", yield_=0, hard_error="timeout"),
            _feed_record("2026-08-25T06:00:00-07:00", "Y", yield_=0, hard_error="timeout"),
        ]
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "feed-dead-url"] == []

    def test_short_streak_with_bad_probe_still_fires_warn_via_probe_path(self):
        """
        Only 2 error+zero runs (below the 3-run streak threshold) but a bad
        live probe: Mode A must still fire, at WARN (first-sighting), via
        the probe path rather than the history-streak path.
        """
        history = [
            _feed_record("2026-08-24T06:00:00-07:00", "Broken", yield_=0, hard_error="timeout"),
            _feed_record("2026-08-25T06:00:00-07:00", "Broken", yield_=0, hard_error="timeout"),
        ]
        probes = [{"name": "Broken", "url": "u", "status": 500, "content_type": "text/plain",
                   "entries": 0, "newest_entry": None, "error": None}]
        findings = detect_rot(history, probes=probes)
        dead = [f for f in findings if f.code == "feed-dead-url"]
        assert len(dead) == 1
        assert dead[0].severity == WARN


# ---------------------------------------------------------------------------
# detect_rot — Mode B: frozen feed
# ---------------------------------------------------------------------------


class TestDetectRotFrozenFeed:
    def test_frozen_feed_fires_with_date_in_message(self):
        probes = [{"name": "Google AI Blog", "url": "u", "status": 200, "content_type": "application/rss+xml",
                   "entries": 5, "newest_entry": "2024-03-29T11:03:00+00:00", "error": None}]
        findings = detect_rot([], probes=probes)
        frozen = [f for f in findings if f.code == "feed-frozen"]
        assert len(frozen) == 1
        assert frozen[0].severity == WARN
        assert "2024-03-29" in frozen[0].message

    def test_recently_updated_feed_does_not_fire(self):
        probes = [{"name": "Karpathy", "url": "u", "status": 200, "content_type": "application/xml",
                   "entries": 3, "newest_entry": "2026-08-20T00:00:00+00:00", "error": None}]
        findings = detect_rot([], probes=probes)
        assert [f for f in findings if f.code == "feed-frozen"] == []

    def test_per_feed_stale_override_exempts_slow_blogger(self):
        probes = [{"name": "Lilian Weng", "url": "u", "status": 200, "content_type": "application/xml",
                   "entries": 1, "newest_entry": "2025-01-01T00:00:00+00:00", "error": None}]
        rules = {"stale_after_days": 90, "feed_overrides": {"Lilian Weng": {"stale_after_days": 3650}}}
        findings = detect_rot([], probes=probes, rules=rules)
        assert [f for f in findings if f.code == "feed-frozen"] == []

    def test_dead_feed_is_not_also_reported_frozen(self):
        probes = [{"name": "Anthropic", "url": "u", "status": 404, "content_type": "text/html",
                   "entries": 0, "newest_entry": None, "error": None}]
        findings = detect_rot([], probes=probes)
        assert [f for f in findings if f.code == "feed-frozen"] == []


# ---------------------------------------------------------------------------
# detect_rot — Mode C: yield collapse
#
# The single most important behavior in this module: the same "zero this
# week" means different things for a firehose blog that broke vs. a real
# blogger who just posts a few times a year.
# ---------------------------------------------------------------------------


def _dates(n, start="2026-08-01"):
    from datetime import datetime, timedelta
    base = datetime.fromisoformat(start)
    return [(base + timedelta(days=i)).strftime("%Y-%m-%dT06:00:00-07:00") for i in range(n)]


class TestDetectRotYieldCollapse:
    def test_venturebeat_style_healthy_then_collapsed_fires(self):
        # 16 healthy runs, then 7 consecutive zero-yield runs (>= threshold).
        ts = _dates(23)
        yields = [2, 3, 1, 4, 2, 3, 5, 1, 2, 3, 4, 1, 2, 3, 2, 1] + [0] * 7
        history = [_feed_record(t, "VentureBeat AI", yield_=y) for t, y in zip(ts, yields)]
        findings = detect_rot(history)
        collapse = [f for f in findings if f.code == "feed-yield-collapse"]
        assert len(collapse) == 1
        assert collapse[0].source == "VentureBeat AI"
        assert collapse[0].severity == WARN

    def test_lilian_weng_style_always_quiet_never_fires(self):
        # 23 runs, every single one zero — a real blogger who just doesn't
        # post often. Trailing median is 0, so this must stay silent.
        ts = _dates(23)
        history = [_feed_record(t, "Lilian Weng", yield_=0) for t in ts]
        findings = detect_rot(history)
        collapse = [f for f in findings if f.code == "feed-yield-collapse"]
        assert collapse == []

    def test_short_zero_streak_below_threshold_does_not_fire(self):
        ts = _dates(10)
        yields = [3, 2, 4, 3, 2, 1, 3, 0, 0, 0]  # only 3 zero runs, threshold is 7
        history = [_feed_record(t, "X", yield_=y) for t, y in zip(ts, yields)]
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "feed-yield-collapse"] == []

    def test_error_only_records_do_not_count_as_zero_yield(self):
        # None-yield (error, no Found line that run) must not be treated as
        # a confirmed zero for streak/median purposes.
        ts = _dates(10)
        history = [_feed_record(t, "Y", hard_error="timeout") for t in ts[:9]]
        history.append(_feed_record(ts[9], "Y", yield_=0))
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "feed-yield-collapse"] == []

    def test_custom_thresholds_respected(self):
        ts = _dates(10)
        yields = [5, 5, 5, 5, 5, 0, 0, 0]
        history = [_feed_record(t, "Z", yield_=y) for t, y in zip(ts[:8], yields)]
        rules = {"zero_run_threshold": 3, "median_window": 8}
        findings = detect_rot(history, rules=rules)
        collapse = [f for f in findings if f.code == "feed-yield-collapse"]
        assert len(collapse) == 1


# ---------------------------------------------------------------------------
# detect_rot — query dead weight
# ---------------------------------------------------------------------------


class TestDetectRotQueryDeadWeight:
    def test_dead_query_fires_info(self):
        ts = _dates(14)
        history = [_query_record(t, "condition-style query", 0) for t in ts]
        findings = detect_rot(history)
        dead = [f for f in findings if f.code == "query-dead-weight"]
        assert len(dead) == 1
        assert dead[0].severity == INFO

    def test_productive_query_does_not_fire(self):
        ts = _dates(14)
        history = [_query_record(t, "good query", 5) for t in ts]
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "query-dead-weight"] == []

    def test_new_query_with_too_little_history_does_not_fire(self):
        history = [_query_record(_dates(1)[0], "brand new query", 0)]
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "query-dead-weight"] == []


# ---------------------------------------------------------------------------
# detect_rot — zone unreachable
# ---------------------------------------------------------------------------


class TestDetectRotZoneUnreachable:
    def test_hard_error_fires_warn(self):
        history = [_zone_record("2026-08-25T06:00:00-07:00", hard_error="503 Service Unavailable")]
        findings = detect_rot(history)
        zone = [f for f in findings if f.code == "zone-unreachable"]
        assert len(zone) == 1
        assert zone[0].severity == WARN

    def test_zero_active_alerts_is_normal_never_fires(self):
        history = [_zone_record("2026-08-25T06:00:00-07:00", yield_=0)]
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "zone-unreachable"] == []

    def test_healthy_alerts_present_never_fires(self):
        history = [_zone_record("2026-08-25T06:00:00-07:00", yield_=2)]
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "zone-unreachable"] == []

    def test_only_latest_run_considered(self):
        history = [
            _zone_record("2026-08-24T06:00:00-07:00", hard_error="503"),
            _zone_record("2026-08-25T06:00:00-07:00", yield_=0),
        ]
        findings = detect_rot(history)
        assert [f for f in findings if f.code == "zone-unreachable"] == []


# ---------------------------------------------------------------------------
# CLI: --since=-3d must parse with standard argparse, no private patching
# ---------------------------------------------------------------------------


class TestCLISinceParsing:
    def test_since_equals_form_parses(self, tmp_path, monkeypatch):
        history_path = tmp_path / "source-health.jsonl"
        missing_config = tmp_path / "no-such-config.yaml"
        argv = [
            "source_health.py",
            "--since=-3d",
            "--history", str(history_path),
            "--config", str(missing_config),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with patch("subprocess.run", return_value=_mock_completed(LOG_FEED_YIELD)):
            rc = main()
        assert rc == 0
        assert history_path.exists()
        loaded = load_history(path=str(history_path))
        assert len(loaded) == 1
        assert loaded[0]["name"] == "FOX 5 San Diego"
