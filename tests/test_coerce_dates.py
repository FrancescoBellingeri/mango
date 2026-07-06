"""Tests for _parse_iso in mango.tools.mongo_tools."""

from __future__ import annotations

from datetime import datetime

from mango.tools.mongo_tools import _parse_iso


# ---------------------------------------------------------------------------
# _parse_iso
# ---------------------------------------------------------------------------


class TestParseIso:
    def test_date_only(self):
        dt = _parse_iso("2024-01-15")
        assert isinstance(dt, datetime)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_datetime_with_time(self):
        dt = _parse_iso("2024-03-20T10:30:00")
        assert dt.hour == 10
        assert dt.minute == 30

    def test_datetime_with_milliseconds(self):
        dt = _parse_iso("2024-06-01T12:00:00.123")
        assert dt.microsecond == 123000

    def test_datetime_with_z_suffix(self):
        dt = _parse_iso("2024-01-01T00:00:00Z")
        assert isinstance(dt, datetime)
        # Timezone is stripped — pymongo expects naive UTC datetimes.
        assert dt.tzinfo is None

    def test_datetime_with_offset(self):
        dt = _parse_iso("2024-01-01T00:00:00+02:00")
        assert isinstance(dt, datetime)
        assert dt.tzinfo is None  # offset stripped

    def test_invalid_string_returns_none(self):
        assert _parse_iso("not-a-date") is None

    def test_non_string_returns_none(self):
        assert _parse_iso(12345) is None
        assert _parse_iso(None) is None

    def test_partial_date_with_space_separator(self):
        dt = _parse_iso("2024-01-15 08:00:00")
        assert isinstance(dt, datetime)
        assert dt.hour == 8
