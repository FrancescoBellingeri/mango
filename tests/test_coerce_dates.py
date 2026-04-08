"""Tests for _coerce_dates and _parse_iso in mango.tools.mongo_tools."""

from __future__ import annotations

from datetime import datetime

import pytest

from mango.tools.mongo_tools import _coerce_dates, _parse_iso


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


# ---------------------------------------------------------------------------
# _coerce_dates
# ---------------------------------------------------------------------------


class TestCoerceDates:
    def test_none_returns_none(self):
        assert _coerce_dates(None) is None

    def test_non_date_string_unchanged(self):
        assert _coerce_dates("hello") == "hello"
        assert _coerce_dates("active") == "active"

    def test_iso_string_converted_to_datetime(self):
        result = _coerce_dates("2024-01-15")
        assert isinstance(result, datetime)

    def test_iso_string_with_time_converted(self):
        result = _coerce_dates("2024-01-15T00:00:00")
        assert isinstance(result, datetime)

    def test_extended_json_date_converted(self):
        result = _coerce_dates({"$date": "2024-01-15T00:00:00Z"})
        assert isinstance(result, datetime)

    def test_nested_dict_converted(self):
        result = _coerce_dates({
            "$gte": "2024-01-01",
            "$lt": "2024-12-31",
        })
        assert isinstance(result["$gte"], datetime)
        assert isinstance(result["$lt"], datetime)

    def test_list_of_dates_converted(self):
        result = _coerce_dates(["2024-01-01", "2024-06-01"])
        assert all(isinstance(d, datetime) for d in result)

    def test_non_date_values_in_dict_unchanged(self):
        result = _coerce_dates({"status": "active", "count": 42})
        assert result["status"] == "active"
        assert result["count"] == 42

    def test_pipeline_with_date_filter(self):
        pipeline = [
            {"$match": {"created_at": {"$gte": "2024-01-01T00:00:00"}}}
        ]
        result = _coerce_dates(pipeline)
        assert isinstance(result[0]["$match"]["created_at"]["$gte"], datetime)

    def test_already_datetime_unchanged(self):
        dt = datetime(2024, 1, 1)
        result = _coerce_dates(dt)
        assert result is dt

    def test_integer_unchanged(self):
        assert _coerce_dates(42) == 42

    def test_empty_dict_unchanged(self):
        assert _coerce_dates({}) == {}

    def test_empty_list_unchanged(self):
        assert _coerce_dates([]) == []

    def test_extended_json_only_one_key(self):
        # {"$date": ...} with extra keys is NOT treated as Extended JSON.
        obj = {"$date": "2024-01-01", "extra": "key"}
        result = _coerce_dates(obj)
        # Should be recursed into, not converted whole
        assert isinstance(result, dict)
        assert isinstance(result["$date"], datetime)
