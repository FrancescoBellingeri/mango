"""Tests for ExplainQueryTool and its helpers."""

from __future__ import annotations

import pytest

from mango.core.types import QueryRequest
from mango.tools.mongo_tools import (
    ExplainQueryTool,
    _describe_request,
    _describe_stage,
    _parse_explain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool(mongo_backend, validate: bool = True) -> ExplainQueryTool:
    return ExplainQueryTool(mongo_backend, validate=validate)


def make_request(**kwargs) -> QueryRequest:
    defaults = dict(operation="find", collection="users")
    defaults.update(kwargs)
    return QueryRequest(**defaults)


# ---------------------------------------------------------------------------
# _describe_stage
# ---------------------------------------------------------------------------


class TestDescribeStage:
    def test_match(self):
        d = _describe_stage("$match", {"age": {"$gt": 25}})
        assert "$match" not in d or "Filter" in d

    def test_group(self):
        d = _describe_stage("$group", {"_id": "$product", "total": {"$sum": "$qty"}})
        assert "product" in d
        assert "total" in d

    def test_sort_desc(self):
        d = _describe_stage("$sort", {"created_at": -1})
        assert "desc" in d
        assert "created_at" in d

    def test_sort_asc(self):
        d = _describe_stage("$sort", {"name": 1})
        assert "asc" in d

    def test_limit(self):
        d = _describe_stage("$limit", 10)
        assert "10" in d

    def test_skip(self):
        d = _describe_stage("$skip", 5)
        assert "5" in d

    def test_project_include(self):
        d = _describe_stage("$project", {"title": 1, "year": 1})
        assert "title" in d
        assert "year" in d

    def test_project_exclude(self):
        d = _describe_stage("$project", {"_id": 0})
        assert "Exclude" in d or "exclude" in d

    def test_unwind_string(self):
        d = _describe_stage("$unwind", "$genres")
        assert "genres" in d

    def test_unwind_object(self):
        d = _describe_stage("$unwind", {"path": "$tags", "preserveNullAndEmptyArrays": True})
        assert "tags" in d
        assert "null" in d.lower() or "keep" in d.lower()

    def test_lookup(self):
        d = _describe_stage("$lookup", {
            "from": "orders", "localField": "_id",
            "foreignField": "user_id", "as": "user_orders",
        })
        assert "orders" in d
        assert "user_orders" in d

    def test_count(self):
        d = _describe_stage("$count", "total")
        assert "total" in d

    def test_addfields(self):
        d = _describe_stage("$addFields", {"ratio": {"$divide": ["$a", "$b"]}})
        assert "ratio" in d

    def test_out(self):
        d = _describe_stage("$out", "result_collection")
        assert "result_collection" in d

    def test_unknown_stage_fallback(self):
        d = _describe_stage("$someNewStage", {"x": 1})
        assert "$someNewStage" in d


# ---------------------------------------------------------------------------
# _describe_request
# ---------------------------------------------------------------------------


class TestDescribeRequest:
    def test_find_no_filter(self):
        stages = _describe_request(make_request())
        assert any("all documents" in s["description"].lower() for s in stages)

    def test_find_with_filter(self):
        stages = _describe_request(make_request(filter={"age": {"$gte": 18}}))
        assert any("age" in s["description"] for s in stages)

    def test_find_with_sort(self):
        stages = _describe_request(make_request(sort={"name": 1}))
        assert any("name" in s["description"] for s in stages)

    def test_find_with_projection(self):
        stages = _describe_request(make_request(projection={"name": 1, "email": 1}))
        assert any("name" in s["description"] for s in stages)

    def test_count_has_count_step(self):
        stages = _describe_request(make_request(operation="count"))
        assert any(s["step"] == "count" for s in stages)

    def test_distinct(self):
        stages = _describe_request(make_request(operation="distinct", distinct_field="email"))
        assert any("email" in s["description"] for s in stages)

    def test_aggregate_pipeline(self):
        stages = _describe_request(make_request(
            operation="aggregate",
            pipeline=[
                {"$match": {"qty": {"$gt": 0}}},
                {"$group": {"_id": "$product", "total": {"$sum": "$qty"}}},
                {"$sort": {"total": -1}},
            ],
        ))
        assert len(stages) == 3
        assert stages[0]["step"] == "$match"
        assert stages[1]["step"] == "$group"
        assert stages[2]["step"] == "$sort"


# ---------------------------------------------------------------------------
# _parse_explain
# ---------------------------------------------------------------------------


class TestParseExplain:
    def test_empty_raw(self):
        assert _parse_explain({}) == {}

    def test_execution_stats(self):
        raw = {
            "executionStats": {
                "nReturned": 5,
                "totalDocsExamined": 100,
                "totalKeysExamined": 5,
                "executionTimeMillis": 12,
            },
            "queryPlanner": {
                "winningPlan": {
                    "stage": "FETCH",
                    "inputStage": {"stage": "IXSCAN", "keyPattern": {"name": 1}},
                }
            },
        }
        result = _parse_explain(raw)
        assert result["docs_returned"] == 5
        assert result["docs_examined"] == 100
        assert result["execution_time_ms"] == 12
        assert result["scan_type"] == "IXSCAN"
        assert "name" in result["index_used"]

    def test_collscan(self):
        raw = {
            "executionStats": {"nReturned": 3, "totalDocsExamined": 3,
                                "totalKeysExamined": 0, "executionTimeMillis": 1},
            "queryPlanner": {"winningPlan": {"stage": "COLLSCAN"}},
        }
        result = _parse_explain(raw)
        assert result["scan_type"] == "COLLSCAN"
        assert "index_used" not in result

    def test_aggregate_stages_count(self):
        raw = {"stages": [{"$cursor": {}}, {"$group": {}}, {"$sort": {}}]}
        result = _parse_explain(raw)
        assert result["pipeline_stages_executed"] == 3


# ---------------------------------------------------------------------------
# ExplainQueryTool — integration
# ---------------------------------------------------------------------------


class TestExplainQueryTool:
    async def test_valid_find_returns_stages(self, mongo_backend):
        tool = make_tool(mongo_backend)
        result = await tool.execute(operation="find", collection="users")
        assert result.success
        assert "stages" in result.data
        assert len(result.data["stages"]) >= 1

    async def test_valid_aggregate_returns_stages(self, mongo_backend):
        tool = make_tool(mongo_backend)
        result = await tool.execute(
            operation="aggregate",
            collection="orders",
            pipeline=[
                {"$match": {"qty": {"$gt": 1}}},
                {"$group": {"_id": "$product", "total": {"$sum": "$qty"}}},
            ],
        )
        assert result.success
        assert len(result.data["stages"]) == 2
        assert result.data["stages"][0]["step"] == "$match"
        assert result.data["stages"][1]["step"] == "$group"

    async def test_invalid_collection_blocked(self, mongo_backend):
        tool = make_tool(mongo_backend)
        result = await tool.execute(operation="find", collection="ghost")
        assert not result.success
        assert "does not exist" in result.error

    async def test_aggregate_without_pipeline_blocked(self, mongo_backend):
        tool = make_tool(mongo_backend)
        result = await tool.execute(operation="aggregate", collection="users")
        assert not result.success
        assert "pipeline" in result.error

    async def test_validate_false_skips_validation(self, mongo_backend):
        tool = make_tool(mongo_backend, validate=False)
        result = await tool.execute(operation="find", collection="ghost")
        assert result.success

    async def test_collection_in_result(self, mongo_backend):
        tool = make_tool(mongo_backend)
        result = await tool.execute(operation="find", collection="users")
        assert result.data["collection"] == "users"

    async def test_operation_in_result(self, mongo_backend):
        tool = make_tool(mongo_backend)
        result = await tool.execute(operation="count", collection="users")
        assert result.data["operation"] == "count"

    async def test_find_with_filter_described(self, mongo_backend):
        tool = make_tool(mongo_backend)
        result = await tool.execute(
            operation="find",
            collection="users",
            filter={"name": "Alice"},
        )
        assert result.success
        descriptions = " ".join(s["description"] for s in result.data["stages"])
        assert "Alice" in descriptions

    async def test_find_with_sort_described(self, mongo_backend):
        tool = make_tool(mongo_backend)
        result = await tool.execute(
            operation="find",
            collection="users",
            sort={"name": -1},
        )
        assert result.success
        descriptions = " ".join(s["description"] for s in result.data["stages"])
        assert "name" in descriptions
        assert "desc" in descriptions
