# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for weekly happenings caching in BriefingRunner.

Happenings are fetched only on a configured weekday (default Saturday) and
reused from state for the rest of the week.
"""

from datetime import datetime as _datetime
from unittest.mock import MagicMock, patch

import pytest

from scripts.briefing_runner import BriefingRunner


def _fake_datetime(day: int) -> type:
    class FakeDateTime(_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, day, 6, 0, 0)

    return FakeDateTime


def _make_runner(tmp_path, monkeypatch, fetch_weekday=5):
    monkeypatch.chdir(tmp_path)
    config = {
        "arxiv_topics": [],
        "stocks": [],
        "blog_feeds": [],
        "news_queries": [],
        "happenings_queries": ["Pacific Beach San Diego events upcoming"],
        "happenings_fetch_weekday": fetch_weekday,
        "state_file_path": ".local-state.json",
        "output_dir": "briefings",
        "section_order": ["news"],
        "features": {},
        "bedrock": {"enabled": False},
        "gemini": {"enabled": False},
        "pdf": {"enabled": False},
    }
    return BriefingRunner(config=config, dry_run=True)


def _sample_happenings(n=3):
    return [{"title": f"Event {i}", "url": f"https://e/{i}"} for i in range(n)]


class TestLoadOrFetchHappenings:
    def test_fetch_on_configured_weekday(self, tmp_path, monkeypatch):
        runner = _make_runner(tmp_path, monkeypatch)
        fetched = _sample_happenings()
        with patch("scripts.briefing_runner.datetime", _fake_datetime(8)), \
             patch.object(runner, "run_happenings_aggregation", return_value=fetched):
            result = runner._load_or_fetch_happenings({})
        assert result == fetched
        assert runner._happenings_cache == fetched
        assert runner._happenings_cache_date == "2026-08-08"

    def test_reuse_on_non_fetch_day(self, tmp_path, monkeypatch):
        runner = _make_runner(tmp_path, monkeypatch)
        cached = _sample_happenings(2)
        state = {"cached_happenings": cached, "cached_happenings_date": "2026-08-08"}
        with patch("scripts.briefing_runner.datetime", _fake_datetime(10)), \
             patch.object(runner, "run_happenings_aggregation") as mock_fetch:
            result = runner._load_or_fetch_happenings(state)
        assert result == cached
        mock_fetch.assert_not_called()
        assert runner._happenings_cache == cached
        assert runner._happenings_cache_date == "2026-08-08"

    def test_no_cache_non_fetch_day_returns_empty(self, tmp_path, monkeypatch):
        runner = _make_runner(tmp_path, monkeypatch)
        with patch("scripts.briefing_runner.datetime", _fake_datetime(10)), \
             patch.object(runner, "run_happenings_aggregation") as mock_fetch:
            result = runner._load_or_fetch_happenings({})
        assert result == []
        mock_fetch.assert_not_called()

    def test_fetch_empty_falls_back_to_cache(self, tmp_path, monkeypatch):
        runner = _make_runner(tmp_path, monkeypatch)
        cached = _sample_happenings(2)
        state = {"cached_happenings": cached, "cached_happenings_date": "2026-08-01"}
        with patch("scripts.briefing_runner.datetime", _fake_datetime(8)), \
             patch.object(runner, "run_happenings_aggregation", return_value=[]):
            result = runner._load_or_fetch_happenings(state)
        assert result == cached

    def test_no_weekday_configured_fetches_every_day(self, tmp_path, monkeypatch):
        runner = _make_runner(tmp_path, monkeypatch, fetch_weekday=None)
        fetched = _sample_happenings()
        with patch("scripts.briefing_runner.datetime", _fake_datetime(10)), \
             patch.object(runner, "run_happenings_aggregation", return_value=fetched) as mock_fetch:
            result = runner._load_or_fetch_happenings({})
        assert result == fetched
        mock_fetch.assert_called_once()


class TestSaveStateCache:
    def test_cached_happenings_persisted(self, tmp_path, monkeypatch):
        runner = _make_runner(tmp_path, monkeypatch)
        cached = _sample_happenings(2)
        runner._save_state(
            [], [], [], [], [],
            cached_happenings=cached,
            cached_happenings_date="2026-08-08",
        )
        import json
        state = json.loads((tmp_path / ".local-state.json").read_text())
        assert state["cached_happenings"] == cached
        assert state["cached_happenings_date"] == "2026-08-08"

    def test_no_cache_omits_keys(self, tmp_path, monkeypatch):
        runner = _make_runner(tmp_path, monkeypatch)
        runner._save_state([], [], [], [], [])
        import json
        state = json.loads((tmp_path / ".local-state.json").read_text())
        assert "cached_happenings" not in state


class TestMultiDayHappeningsRefresh:
    """happenings_fetch_weekday accepts one weekday or a list of them."""

    def _runner(self, value, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = {
            "arxiv_topics": [],
            "blog_feeds": [],
            "stocks": [],
            "news_queries": [],
            "paper_scoring": {"has_code": 5, "topic_match": 3, "recency": 2, "citation_count": 1},
            "max_papers": 5, "max_blogs": 5, "max_news": 5, "arxiv_days_back": 3,
            "output_format": "kindle", "file_naming": "T-{yyyy}",
            "pdf": {"enabled": False}, "bedrock": {"enabled": False}, "gemini": {"enabled": False},
        }
        if value is not None:
            config["happenings_fetch_weekday"] = value
        return BriefingRunner(config, dry_run=True)

    def test_absent_key_means_every_run(self, tmp_path, monkeypatch):
        assert self._runner(None, tmp_path, monkeypatch)._happenings_fetch_days() is None

    def test_single_int_is_normalized_to_a_set(self, tmp_path, monkeypatch):
        assert self._runner(3, tmp_path, monkeypatch)._happenings_fetch_days() == {3}

    def test_list_of_weekdays(self, tmp_path, monkeypatch):
        assert self._runner([0, 3], tmp_path, monkeypatch)._happenings_fetch_days() == {0, 3}

    def test_string_digits_are_coerced(self, tmp_path, monkeypatch):
        assert self._runner(["0", "3"], tmp_path, monkeypatch)._happenings_fetch_days() == {0, 3}

    def test_empty_list_falls_back_to_every_run(self, tmp_path, monkeypatch):
        assert self._runner([], tmp_path, monkeypatch)._happenings_fetch_days() is None

    def test_fetches_on_any_listed_day(self, tmp_path, monkeypatch):
        runner = self._runner([0, 3], tmp_path, monkeypatch)
        fetched = [{"title": "Fresh event"}]
        for weekday, expect_fetch in ((0, True), (3, True), (1, False), (5, False)):
            fake_now = MagicMock()
            fake_now.weekday.return_value = weekday
            fake_now.strftime.return_value = "2026-08-27"
            with patch("scripts.briefing_runner.datetime") as dt:
                dt.now.return_value = fake_now
                with patch.object(runner, "run_happenings_aggregation", return_value=fetched) as agg:
                    result = runner._load_or_fetch_happenings(
                        {"cached_happenings": [{"title": "Stale event"}],
                         "cached_happenings_date": "2026-08-20"}
                    )
            assert agg.called is expect_fetch, f"weekday {weekday}"
            assert result[0]["title"] == ("Fresh event" if expect_fetch else "Stale event")
