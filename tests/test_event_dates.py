"""Calendar cases observed in the October local briefs."""

from datetime import date

import pytest

from scripts.event_dates import extract_dates, has_only_past_dates


@pytest.mark.parametrize("text,expected", [
    ("From Oct. 1 to Oct. 8, local communities hold events.", [date(2026, 10, 8)]),
    ("October 1–8, 2026", [date(2026, 10, 8)]),
    ("Tue Oct 6, 4 PM → Fri Oct 9, 11 PM", [date(2026, 10, 9)]),
    ("Dec. 30-Jan. 2", [date(2027, 1, 2)]),
])
def test_ranges_use_the_end_date(text, expected):
    today = date(2026, 12, 31) if "Dec." in text else date(2026, 10, 6)
    assert extract_dates(text, today) == expected


def test_explicit_previous_year_is_not_current():
    assert has_only_past_dates("Festival October 8, 2025", date(2026, 10, 6))
