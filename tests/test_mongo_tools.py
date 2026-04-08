"""Tests for mango.tools.mongo_tools — ListCollectionsTool, DescribeCollectionTool, RunMQLTool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mango.core.types import FieldInfo, SchemaInfo
from mango.tools.mongo_tools import (
    DescribeCollectionTool,
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
