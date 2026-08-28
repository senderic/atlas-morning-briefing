#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Calendar-date extraction for event copy.

Shared by the briefing pipeline, which DROPS happenings whose dates have
already passed, and by the quality checker, which REPORTS any that slip
through. Both must agree on what counts as a date: if the filter and the
check drift, the checker either cries wolf about items the pipeline handled
or stays silent about ones it missed.

Handles arrow ranges ("Tue Aug 25, 10:00 AM -> Fri Aug 28"), dash ranges
("Aug. 21-23", "Aug 30-Sep 2") and single dates ("Wed Aug 26"). For a range,
staleness is decided by the END of the range. A bare month with no day
("in September") is never treated as a date.
"""

import re
from datetime import date
from typing import List, Optional, Tuple

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_MONTH_ALT = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember|t)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"

# "Tue Aug 25, 10:00 AM -> Fri Aug 28, 8:00 PM" -- an explicit arrow range.
# Staleness is decided by the second (end) date; only that half is captured.
_ARROW_RANGE_RE = re.compile(
    rf"(?:{_MONTH_ALT})\.?\s+{_DAY}[^\n]{{0,60}}?->\s*(?:\w+\s+)?"
    rf"(?P<m2>{_MONTH_ALT})\.?\s+(?P<d2>{_DAY})",
    re.IGNORECASE,
)
# "Aug. 21-23" (same month) or "Aug 30-Sep 2" (cross month). Staleness is
# decided by the second (end) date.
_DASH_RANGE_RE = re.compile(
    rf"(?P<m1>{_MONTH_ALT})\.?\s+(?P<d1>{_DAY})\s*-\s*"
    rf"(?:(?P<m2>{_MONTH_ALT})\.?\s+)?(?P<d2>{_DAY})",
    re.IGNORECASE,
)
# A single explicit date with no range: "Wed Aug 26". A bare month with no
# day ("in September") never matches this -- the day group is mandatory.
_SINGLE_DATE_RE = re.compile(
    rf"(?P<m1>{_MONTH_ALT})\.?\s+(?P<d1>{_DAY})",
    re.IGNORECASE,
)

_STALE_WINDOW_DAYS = 180


def _month_num(token: Optional[str]) -> Optional[int]:
    if not token:
        return None
    return _MONTHS.get(token.lower().rstrip("."))


def _safe_date(year: int, month: int, day_token: str) -> Optional[date]:
    try:
        day = int(re.match(r"\d+", day_token).group())
        return date(year, month, day)
    except (ValueError, AttributeError):
        return None


def _mask(text: str, span: Tuple[int, int]) -> str:
    start, end = span
    return text[:start] + (" " * (end - start)) + text[end:]


def _extract_dates_from_line(line: str, today: date) -> List[date]:
    """
    Every explicit calendar date on this line, resolved to the current year,
    within the +/-180 day sanity window. For a range, only the end date is
    returned -- staleness is decided by when the range ends.
    """
    dates: List[date] = []
    work = line

    for m in list(_ARROW_RANGE_RE.finditer(work)):
        month = _month_num(m.group("m2"))
        if month:
            d = _safe_date(today.year, month, m.group("d2"))
            if d is not None and abs((d - today).days) <= _STALE_WINDOW_DAYS:
                dates.append(d)
        work = _mask(work, m.span())

    for m in list(_DASH_RANGE_RE.finditer(work)):
        month = _month_num(m.group("m2") or m.group("m1"))
        if month:
            d = _safe_date(today.year, month, m.group("d2"))
            if d is not None and abs((d - today).days) <= _STALE_WINDOW_DAYS:
                dates.append(d)
        work = _mask(work, m.span())

    for m in list(_SINGLE_DATE_RE.finditer(work)):
        month = _month_num(m.group("m1"))
        if month:
            d = _safe_date(today.year, month, m.group("d1"))
            if d is not None and abs((d - today).days) <= _STALE_WINDOW_DAYS:
                dates.append(d)

    return dates


def extract_dates(line: str, today: date) -> List[date]:
    """Public name for :func:`_extract_dates_from_line`."""
    return _extract_dates_from_line(line, today)


def has_only_past_dates(text: str, today: date) -> bool:
    """True when ``text`` carries at least one date and every one is past.

    Undated copy returns False: an evergreen entry (a venue's events calendar,
    a standing rule change) is not stale just because it names no day.
    """
    dates = []
    for line in (text or "").splitlines():
        dates.extend(_extract_dates_from_line(line, today))
    if not dates:
        return False
    return all(d < today for d in dates)
