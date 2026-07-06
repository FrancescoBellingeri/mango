"""Schema-aware literal coercion for run_mql filters and pipelines.

The agent emits query literals as JSON strings. Whether a string like
"2020-01-01" or a 24-char hex id must be coerced to a BSON datetime / ObjectId
depends on how the *target field* is actually stored — which the introspected
schema knows. Blind coercion (convert every ISO-looking string to a datetime,
never touch ObjectIds) produces silent wrong/empty results:

  - a date filter on a STRING-typed field is broken by coercing to datetime;
  - a tz-offset string is mis-converted (offset truncated, not converted);
  - an _id / ObjectId filter written as a hex string never matches.

These tests assert the schema-aware behaviour.
"""

from __future__ import annotations

from datetime import datetime

from bson import ObjectId

from mango.tools.mongo_tools import (
    RunMQLTool,
    _coerce_filter,
    _coerce_pipeline_matches,
    _parse_iso,
    _target_type,
)


# ---------------------------------------------------------------------------
# _parse_iso — timezone correctness
# ---------------------------------------------------------------------------


class TestParseIsoTimezone:
    def test_offset_converted_to_utc_not_truncated(self):
        # 00:00 at +02:00 == 22:00 UTC of the previous day.
        dt = _parse_iso("2024-01-01T00:00:00+02:00")
        assert dt.tzinfo is None            # naive (pymongo stores naive as UTC)
        assert (dt.year, dt.month, dt.day) == (2023, 12, 31)
        assert dt.hour == 22

    def test_z_suffix_is_utc(self):
        dt = _parse_iso("2024-06-01T10:00:00Z")
        assert dt.tzinfo is None
        assert dt.hour == 10

    def test_naive_left_as_is(self):
        dt = _parse_iso("2024-06-01T10:00:00")
        assert dt.hour == 10


# ---------------------------------------------------------------------------
# _target_type — unambiguous type resolution
# ---------------------------------------------------------------------------


class TestTargetType:
    def test_pure_date(self):
        assert _target_type({"date"}) == "date"
        assert _target_type({"date", "null"}) == "date"

    def test_pure_objectid(self):
        assert _target_type({"ObjectId"}) == "ObjectId"

    def test_string_is_not_coercible(self):
        assert _target_type({"string"}) is None

    def test_mixed_is_not_coercible(self):
        assert _target_type({"date", "string"}) is None
        assert _target_type({"ObjectId", "string"}) is None

    def test_empty(self):
        assert _target_type(set()) is None


# ---------------------------------------------------------------------------
# _coerce_filter — value coercion driven by field types
# ---------------------------------------------------------------------------


class TestCoerceFilter:
    def test_date_field_string_coerced(self):
        ftypes = {"released": {"date"}}
        out = _coerce_filter({"released": {"$gte": "2020-01-01"}}, ftypes)
        assert isinstance(out["released"]["$gte"], datetime)

    def test_string_date_field_left_untouched(self):
        ftypes = {"lastupdated": {"string"}}
        out = _coerce_filter({"lastupdated": {"$gte": "2020-01-01"}}, ftypes)
        assert out["lastupdated"]["$gte"] == "2020-01-01"  # still a string

    def test_objectid_field_hex_coerced(self):
        oid = "5a9427648b0beebeb69579e7"
        ftypes = {"_id": {"ObjectId"}}
        out = _coerce_filter({"_id": oid}, ftypes)
        assert out["_id"] == ObjectId(oid)

    def test_string_field_hex_left_untouched(self):
        hexish = "5a9427648b0beebeb69579e7"
        ftypes = {"code": {"string"}}
        out = _coerce_filter({"code": hexish}, ftypes)
        assert out["code"] == hexish

    def test_in_list_coerced(self):
        ftypes = {"_id": {"ObjectId"}}
        a, b = "5a9427648b0beebeb69579e7", "5a9427648b0beebeb69579e8"
        out = _coerce_filter({"_id": {"$in": [a, b]}}, ftypes)
        assert out["_id"]["$in"] == [ObjectId(a), ObjectId(b)]

    def test_logical_and_recurses(self):
        ftypes = {"released": {"date"}}
        out = _coerce_filter(
            {"$and": [{"released": {"$gte": "2020-01-01"}}]}, ftypes
        )
        assert isinstance(out["$and"][0]["released"]["$gte"], datetime)

    def test_unknown_field_left_untouched(self):
        out = _coerce_filter({"whatever": "2020-01-01"}, {})
        assert out["whatever"] == "2020-01-01"

    def test_extended_json_date_on_date_field(self):
        ftypes = {"released": {"date"}}
        out = _coerce_filter({"released": {"$date": "2020-01-01T00:00:00Z"}}, ftypes)
        assert isinstance(out["released"], datetime)

    def test_extended_json_oid_coerced(self):
        # Models frequently emit {"$oid": "..."} — the standard Extended JSON
        # form for ObjectId, analogous to {"$date": ...}. It must coerce to a
        # bare ObjectId, not {"$oid": ObjectId(...)}.
        oid = "5a9427648b0beebeb69579e7"
        ftypes = {"_id": {"ObjectId"}}
        out = _coerce_filter({"_id": {"$oid": oid}}, ftypes)
        assert out["_id"] == ObjectId(oid)

    def test_extended_json_oid_in_list(self):
        a, b = "5a9427648b0beebeb69579e7", "5a9427648b0beebeb69579e8"
        ftypes = {"_id": {"ObjectId"}}
        out = _coerce_filter({"_id": {"$in": [{"$oid": a}, {"$oid": b}]}}, ftypes)
        assert out["_id"]["$in"] == [ObjectId(a), ObjectId(b)]

    def test_oid_on_string_field_left_untouched(self):
        # $oid wrapper on a genuinely string field: unambiguous type is not
        # ObjectId, so we do not coerce.
        oid = "5a9427648b0beebeb69579e7"
        out = _coerce_filter({"code": {"$oid": oid}}, {"code": {"string"}})
        assert out["code"] == {"$oid": oid}


# ---------------------------------------------------------------------------
# _coerce_pipeline_matches — only $match stages, base field types
# ---------------------------------------------------------------------------


class TestCoercePipeline:
    def test_match_stage_coerced(self):
        ftypes = {"released": {"date"}}
        out = _coerce_pipeline_matches(
            [{"$match": {"released": {"$gte": "2020-01-01"}}}], ftypes
        )
        assert isinstance(out[0]["$match"]["released"]["$gte"], datetime)

    def test_group_stage_untouched(self):
        ftypes = {"released": {"date"}}
        pipe = [{"$group": {"_id": "$released", "n": {"$sum": 1}}}]
        out = _coerce_pipeline_matches(pipe, ftypes)
        assert out == pipe  # non-$match stages pass through unchanged

    def test_expr_match_left_untouched(self):
        ftypes = {"released": {"date"}}
        pipe = [{"$match": {"$expr": {"$gt": ["$a", "$b"]}}}]
        out = _coerce_pipeline_matches(pipe, ftypes)
        assert out == pipe


# ---------------------------------------------------------------------------
# Backend field-type map
# ---------------------------------------------------------------------------


class TestFieldTypes:
    def test_field_types_map(self, mongo_backend):
        # orders has created_at (datetime), product (string), qty (int)
        ftypes = mongo_backend.field_types("orders")
        assert "date" in ftypes["created_at"]
        assert "string" in ftypes["product"]


# ---------------------------------------------------------------------------
# End-to-end through RunMQLTool
# ---------------------------------------------------------------------------


class TestEndToEnd:
    async def test_objectid_lookup_returns_document(self, mongo_backend):
        # Grab a real _id, then query it back as a hex string via the tool.
        doc = mongo_backend._database["users"].find_one({"name": "Alice"})
        oid_hex = str(doc["_id"])

        tool = RunMQLTool(mongo_backend, max_rows=100)
        result = await tool.execute(
            operation="find", collection="users", filter={"_id": oid_hex}
        )
        assert result.success is True
        assert result.data["row_count"] == 1
        assert result.data["rows"][0]["name"] == "Alice"

    async def test_string_date_filter_matches_string_field(self, mongo_backend):
        # A collection whose date field is stored as an ISO *string*.
        mongo_backend._database["logs"].insert_many(
            [{"d": "2020-06-01"}, {"d": "2021-06-01"}, {"d": "2019-06-01"}]
        )
        tool = RunMQLTool(mongo_backend, max_rows=100)
        result = await tool.execute(
            operation="find", collection="logs", filter={"d": {"$gte": "2020-01-01"}}
        )
        assert result.success is True
        # Blind coercion would turn "2020-01-01" into a datetime and match nothing.
        assert result.data["row_count"] == 2
