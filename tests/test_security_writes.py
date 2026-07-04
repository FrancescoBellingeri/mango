"""Security tests for the read-only guarantee.

The agent must not be coercible into WRITE or server-side-JS operations
through aggregation pipeline stages that the operation-level allowlist never
inspects ($out, $merge, $where, $function, ...).

These tests assert the post-fix behaviour: such stages are blocked both by
the pre-execution validator and by the backend runner (defence-in-depth,
active even when validation is disabled).
"""

from __future__ import annotations

import os

import pytest

from mango.core.security import find_forbidden_operators
from mango.core.types import QueryRequest, ValidationError
from mango.tools.mongo_tools import RunMQLTool
from mango.tools.validator import MQLValidator


# ---------------------------------------------------------------------------
# Execution-level: $out must be blocked, no collection written
# ---------------------------------------------------------------------------


class TestOutWriteBypassBlocked:
    async def test_out_stage_is_blocked_and_writes_nothing(self, mongo_backend):
        assert "victims" not in mongo_backend.list_collections()

        tool = RunMQLTool(mongo_backend, validate=True)
        result = await tool.execute(
            operation="aggregate",
            collection="users",
            pipeline=[{"$match": {}}, {"$out": "victims"}],
        )

        assert result.success is False
        assert "victims" not in mongo_backend.list_collections()

    async def test_out_blocked_even_with_validation_disabled(self, mongo_backend):
        """Runner-level defence: the read-only guarantee is not optional."""
        assert "victims" not in mongo_backend.list_collections()
        req = QueryRequest(
            operation="aggregate",
            collection="users",
            pipeline=[{"$match": {}}, {"$out": "victims"}],
        )
        with pytest.raises(ValidationError):
            mongo_backend.execute_query(req)
        assert "victims" not in mongo_backend.list_collections()


# ---------------------------------------------------------------------------
# Validation-level: dangerous stages rejected at any depth
# ---------------------------------------------------------------------------


class TestValidatorBlocksDangerousStages:
    def _validate(self, backend, pipeline=None, filter=None):
        v = MQLValidator(backend)
        op = "aggregate" if pipeline else "find"
        return v.validate(
            QueryRequest(operation=op, collection="users", pipeline=pipeline, filter=filter)
        )

    def test_out_blocked(self, mongo_backend):
        assert self._validate(mongo_backend, [{"$out": "x"}]).valid is False

    def test_merge_blocked(self, mongo_backend):
        assert self._validate(mongo_backend, [{"$merge": {"into": "x"}}]).valid is False

    def test_where_js_blocked(self, mongo_backend):
        r = self._validate(mongo_backend, filter={"$where": "function(){ return true; }"})
        assert r.valid is False

    def test_function_operator_blocked_when_nested(self, mongo_backend):
        r = self._validate(
            mongo_backend,
            [{"$match": {"$expr": {"$function": {"body": "f", "args": [], "lang": "js"}}}}],
        )
        assert r.valid is False

    def test_merge_blocked_in_lookup_subpipeline(self, mongo_backend):
        r = self._validate(
            mongo_backend,
            [{"$lookup": {"from": "orders", "as": "o", "pipeline": [{"$out": "sink"}]}}],
        )
        assert r.valid is False

    def test_change_stream_blocked(self, mongo_backend):
        assert self._validate(mongo_backend, [{"$changeStream": {}}]).valid is False

    @pytest.mark.parametrize(
        "stage",
        [
            {"$collStats": {"storageStats": {}}},
            {"$indexStats": {}},
            {"$planCacheStats": {}},
            {"$listSessions": {}},
            {"$listLocalSessions": {}},
        ],
    )
    def test_admin_stages_blocked(self, mongo_backend, stage):
        assert self._validate(mongo_backend, [stage]).valid is False

    def test_accumulator_operator_blocked(self, mongo_backend):
        # $accumulator (server-side JS) nested inside a $group.
        r = self._validate(
            mongo_backend,
            [{"$group": {"_id": None, "v": {"$accumulator": {"lang": "js", "init": "f"}}}}],
        )
        assert r.valid is False

    def test_ordinary_pipeline_still_valid(self, mongo_backend):
        """No false positives: a normal read pipeline still passes."""
        r = self._validate(
            mongo_backend,
            [
                {"$match": {"age": {"$gt": 25}}},
                {"$group": {"_id": "$name", "n": {"$sum": 1}}},
                {"$sort": {"n": -1}},
            ],
        )
        assert r.valid is True


# ---------------------------------------------------------------------------
# Pure scanner unit tests
# ---------------------------------------------------------------------------


class TestFindForbiddenOperators:
    def test_clean_returns_empty(self):
        assert find_forbidden_operators({"x": 1}, [{"$match": {"x": 1}}]) == []

    def test_finds_top_level_stage(self):
        assert find_forbidden_operators(None, [{"$out": "x"}]) == ["$out"]

    def test_finds_deeply_nested_and_dedups(self):
        pipeline = [
            {"$facet": {"a": [{"$out": "s"}], "b": [{"$match": {"$where": "1"}}]}},
        ]
        assert find_forbidden_operators(None, pipeline) == ["$out", "$where"]


# ---------------------------------------------------------------------------
# Real MongoDB (opt-in): confirm nothing is written / executed on a live server
# ---------------------------------------------------------------------------


@pytest.fixture
def real_backend():
    from mango.integrations.mongodb import MongoRunner
    import pymongo

    uri = os.getenv("MANGO_TEST_MONGO_URI", "mongodb://localhost:27017/")
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip("No local MongoDB reachable")

    db_name = "mango_sectest_scratch"
    client.drop_database(db_name)
    db = client[db_name]
    db["src"].insert_many([{"x": 1}, {"x": 2}, {"x": 3}])

    backend = MongoRunner()
    backend._client = client
    backend._db = db
    try:
        yield backend
    finally:
        client.drop_database(db_name)


@pytest.mark.mongodb
class TestRealMongoWriteAndJsBlocked:
    async def test_merge_is_blocked_on_real_mongo(self, real_backend):
        tool = RunMQLTool(real_backend, validate=True)
        result = await tool.execute(
            operation="aggregate",
            collection="src",
            pipeline=[{"$match": {}}, {"$merge": {"into": "sink"}}],
        )
        assert result.success is False
        assert "sink" not in real_backend.list_collections()

    async def test_where_js_is_blocked_on_real_mongo(self, real_backend):
        tool = RunMQLTool(real_backend, validate=True)
        result = await tool.execute(
            operation="find",
            collection="src",
            filter={"$where": "this.x > 1"},
        )
        assert result.success is False
