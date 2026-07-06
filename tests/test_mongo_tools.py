"""Tests for mango.tools.mongo_tools — ListCollectionsTool, DescribeCollectionTool, RunMQLTool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mango.core.types import FieldInfo, SchemaInfo
from mango.tools.mongo_tools import (
    DescribeCollectionTool,
    InspectFieldTool,
    ListCollectionsTool,
    RunMQLTool,
)


# ---------------------------------------------------------------------------
# ListCollectionsTool
# ---------------------------------------------------------------------------


class TestListCollectionsTool:
    async def test_returns_collection_names(self, mongo_backend):
        tool = ListCollectionsTool(mongo_backend)
        result = await tool.execute()
        assert result.success is True
        assert "users" in result.data["collections"]
        assert "orders" in result.data["collections"]

    async def test_count_matches(self, mongo_backend):
        tool = ListCollectionsTool(mongo_backend)
        result = await tool.execute()
        assert result.data["count"] == len(result.data["collections"])

    def test_definition_name(self, mongo_backend):
        tool = ListCollectionsTool(mongo_backend)
        assert tool.definition.name == "list_collections"


# ---------------------------------------------------------------------------
# DescribeCollectionTool
# ---------------------------------------------------------------------------


class TestDescribeCollectionTool:
    def _make_schema(self) -> SchemaInfo:
        return SchemaInfo(
            collection_name="users",
            document_count=3,
            fields=[
                FieldInfo(name="name", path="name", types=["string"], frequency=1.0),
                FieldInfo(name="age", path="age", types=["int"], frequency=1.0),
            ],
            indexes=[],
            sample_documents=[{"name": "Alice", "age": 30}],
        )

    async def test_returns_schema_payload(self, mongo_backend):
        tool = DescribeCollectionTool(mongo_backend)
        schema = self._make_schema()
        with patch.object(mongo_backend, "_introspect_collection", return_value=schema):
            result = await tool.execute(collection="users")
        assert result.success is True
        assert result.data["collection"] == "users"
        assert result.data["document_count"] == 3
        assert len(result.data["fields"]) == 2

    async def test_fields_contain_path_and_types(self, mongo_backend):
        tool = DescribeCollectionTool(mongo_backend)
        schema = self._make_schema()
        with patch.object(mongo_backend, "_introspect_collection", return_value=schema):
            result = await tool.execute(collection="users")
        field_paths = {f["path"] for f in result.data["fields"]}
        assert "name" in field_paths
        assert "age" in field_paths

    async def test_sample_documents_capped_at_3(self, mongo_backend):
        tool = DescribeCollectionTool(mongo_backend)
        schema = SchemaInfo(
            collection_name="users",
            document_count=10,
            fields=[],
            indexes=[],
            sample_documents=[{"x": i} for i in range(10)],
        )
        with patch.object(mongo_backend, "_introspect_collection", return_value=schema):
            result = await tool.execute(collection="users")
        assert len(result.data["sample_documents"]) <= 3

    def test_definition_name(self, mongo_backend):
        tool = DescribeCollectionTool(mongo_backend)
        assert tool.definition.name == "describe_collection"

    async def test_nonexistent_collection_errors(self, mongo_backend):
        # A wrong name must be a clear error, not success with an empty schema
        # (which the agent would misread as "the collection is empty").
        tool = DescribeCollectionTool(mongo_backend)
        result = await tool.execute(collection="ghost_collection")
        assert result.success is False
        assert "does not exist" in (result.error or "")

    async def test_nonexistent_suggests_similar_name(self, mongo_backend):
        # "user" is a near-miss for the real "users" collection.
        tool = DescribeCollectionTool(mongo_backend)
        result = await tool.execute(collection="user")
        assert result.success is False
        assert "users" in (result.error or "")

    async def test_existing_collection_still_succeeds(self, mongo_backend):
        # Control: the existence check must not break the happy path.
        tool = DescribeCollectionTool(mongo_backend)
        result = await tool.execute(collection="users")
        assert result.success is True
        assert result.data["collection"] == "users"


# ---------------------------------------------------------------------------
# RunMQLTool
# ---------------------------------------------------------------------------


class TestRunMQLTool:
    async def test_count_all(self, mongo_backend):
        tool = RunMQLTool(mongo_backend)
        result = await tool.execute(operation="count", collection="users")
        assert result.success is True
        assert result.data["rows"][0]["count"] == 3

    async def test_find_returns_rows(self, mongo_backend):
        tool = RunMQLTool(mongo_backend)
        result = await tool.execute(operation="find", collection="users")
        assert result.success is True
        assert result.data["row_count"] == 3

    async def test_find_with_filter(self, mongo_backend):
        tool = RunMQLTool(mongo_backend)
        result = await tool.execute(
            operation="find",
            collection="users",
            filter={"name": "Bob"},
        )
        assert result.success is True
        assert result.data["row_count"] == 1
        assert result.data["rows"][0]["name"] == "Bob"

    async def test_aggregate_returns_rows(self, mongo_backend):
        tool = RunMQLTool(mongo_backend)
        result = await tool.execute(
            operation="aggregate",
            collection="orders",
            pipeline=[{"$group": {"_id": "$product", "n": {"$sum": 1}}}],
        )
        assert result.success is True
        assert result.data["row_count"] == 2

    async def test_max_rows_cap(self, mongo_backend):
        tool = RunMQLTool(mongo_backend, max_rows=2)
        result = await tool.execute(operation="find", collection="users", limit=100)
        assert result.success is True
        assert result.data["row_count"] <= 2

    async def test_empty_result_returns_empty_rows(self, mongo_backend):
        tool = RunMQLTool(mongo_backend)
        result = await tool.execute(
            operation="find",
            collection="users",
            filter={"name": "DoesNotExist"},
        )
        assert result.success is True
        assert result.data["rows"] == []
        assert result.data["row_count"] == 0

    async def test_pipeline_as_json_string(self, mongo_backend):
        """RunMQLTool must accept pipeline as a JSON string (LLM sometimes sends it that way)."""
        import json
        pipeline_str = json.dumps([{"$match": {"name": "Alice"}}])
        tool = RunMQLTool(mongo_backend)
        result = await tool.execute(
            operation="aggregate",
            collection="users",
            pipeline=pipeline_str,
        )
        assert result.success is True
        assert result.data["row_count"] == 1

    async def test_date_string_in_filter_coerced(self, mongo_backend):
        """ISO date strings in filters should be coerced to datetime."""
        tool = RunMQLTool(mongo_backend)
        result = await tool.execute(
            operation="find",
            collection="orders",
            filter={"created_at": {"$gte": "2024-01-01"}},
        )
        assert result.success is True
        assert result.data["row_count"] == 3

    def test_definition_name(self, mongo_backend):
        tool = RunMQLTool(mongo_backend)
        assert tool.definition.name == "run_mql"


# ---------------------------------------------------------------------------
# InspectFieldTool
# ---------------------------------------------------------------------------


def _seed_drift(mongo_backend):
    """Insert a collection whose 'status' field carries value drift."""
    db = mongo_backend._db
    db["accounts"].insert_many(
        [{"status": "active"} for _ in range(8)]
        + [{"status": "ACTIVE"} for _ in range(2)]
        + [{"status": "inactive"}]
        + [{"tier": {"name": "gold"}}, {"tier": {"name": "gold"}}, {"tier": {"name": "silver"}}]
    )


class TestInspectFieldTool:
    async def test_reveals_value_drift_ordered_by_frequency(self, mongo_backend):
        _seed_drift(mongo_backend)
        tool = InspectFieldTool(mongo_backend)
        result = await tool.execute(collection="accounts", field="status")
        assert result.success is True
        vals = {v["value"]: v["count"] for v in result.data["top_values"]}
        # Both encodings surface with their real counts — the whole point.
        assert vals.get("active") == 8
        assert vals.get("ACTIVE") == 2
        # Ordered by frequency descending.
        counts = [v["count"] for v in result.data["top_values"]]
        assert counts == sorted(counts, reverse=True)

    async def test_payload_shape(self, mongo_backend):
        _seed_drift(mongo_backend)
        tool = InspectFieldTool(mongo_backend)
        result = await tool.execute(collection="accounts", field="status")
        data = result.data
        for key in ("collection", "field", "distinct_count", "type_breakdown",
                    "top_values", "truncated", "sampled", "scanned_docs"):
            assert key in data
        assert data["distinct_count"] >= 3  # active, ACTIVE, inactive (+ missing)

    async def test_dotted_path(self, mongo_backend):
        _seed_drift(mongo_backend)
        tool = InspectFieldTool(mongo_backend)
        result = await tool.execute(collection="accounts", field="tier.name")
        vals = {v["value"]: v["count"] for v in result.data["top_values"]}
        assert vals.get("gold") == 2
        assert vals.get("silver") == 1

    async def test_top_k_limits_and_flags_truncation(self, mongo_backend):
        db = mongo_backend._db
        db["many"].insert_many([{"v": f"val{i}"} for i in range(10)])
        tool = InspectFieldTool(mongo_backend)
        result = await tool.execute(collection="many", field="v", top_k=3)
        assert len(result.data["top_values"]) == 3
        assert result.data["distinct_count"] == 10
        assert result.data["truncated"] is True

    async def test_missing_args_returns_error(self, mongo_backend):
        tool = InspectFieldTool(mongo_backend)
        result = await tool.execute(collection="accounts")
        assert result.success is False

    def test_definition_name(self, mongo_backend):
        tool = InspectFieldTool(mongo_backend)
        assert tool.definition.name == "inspect_field"


# ---------------------------------------------------------------------------
# Introspection cache (avoids re-sampling ~100 docs on every describe_collection)
# ---------------------------------------------------------------------------


class TestIntrospectCache:
    def test_second_call_is_served_from_cache(self, mongo_backend):
        # Fresh introspection builds a new SchemaInfo each time, so identity
        # equality proves the second call returned the cached object.
        first = mongo_backend._introspect_collection("users", {"users", "orders"})
        second = mongo_backend._introspect_collection("users", {"users", "orders"})
        assert first is second

    def test_expired_entry_is_re_sampled(self, mongo_backend):
        first = mongo_backend._introspect_collection("users", {"users", "orders"})
        # Age the cached timestamp past the TTL.
        schema, ts = mongo_backend._introspect_cache["users"]
        mongo_backend._introspect_cache["users"] = (schema, ts - 10_000)
        refreshed = mongo_backend._introspect_collection("users", {"users", "orders"})
        assert refreshed is not first

    def test_ttl_zero_disables_cache(self, mongo_client, mongo_db):
        from mango.integrations.mongodb import MongoRunner

        backend = MongoRunner(introspect_ttl_s=0)
        backend._client = mongo_client
        backend._db = mongo_db
        first = backend._introspect_collection("users", {"users", "orders"})
        second = backend._introspect_collection("users", {"users", "orders"})
        assert first is not second
        assert backend._introspect_cache == {}
