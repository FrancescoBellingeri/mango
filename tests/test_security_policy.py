"""Read-only policy: projection/sort scanning, stage allowlist, collection access.

Complements ``test_security_writes.py`` (which covers $out/$merge/$where in
filter and pipeline). These tests pin the three rings of
``mango.core.security``:

1. forbidden operators are caught in *every* expression-bearing argument —
   a ``find`` projection can carry ``$function`` (aggregation expressions are
   accepted there since MongoDB 4.4);
2. the stage allowlist is enforced by the runner too, so ``validate=False``
   cannot smuggle an unknown/administrative stage, at top level or nested;
3. collection allow/deny lists apply to listing, introspection and execution,
   including collections pulled in through ``$lookup`` / ``$unionWith``.
"""

from __future__ import annotations

import pytest

from mango.core.security import (
    CollectionPolicy,
    enforce_read_only_policy,
    find_disallowed_stages,
    find_forbidden_operators,
    referenced_collections,
)
from mango.core.types import QueryRequest, ValidationError
from mango.integrations.mongodb import MongoRunner
from mango.tools.mongo_tools import DescribeCollectionTool, RunMQLTool
from mango.tools.validator import MQLValidator

_JS_PROJECTION = {
    "name": 1,
    "evil": {"$function": {"body": "function(){ while(true){} }", "args": [], "lang": "js"}},
}


# ---------------------------------------------------------------------------
# Ring 1 — projection / sort are scanned
# ---------------------------------------------------------------------------


class TestProjectionAndSortScanned:
    def test_helper_sees_projection_and_sort(self):
        assert find_forbidden_operators(None, None, projection=_JS_PROJECTION) == ["$function"]
        assert find_forbidden_operators(None, None, sort={"$where": 1}) == ["$where"]

    def test_validator_rejects_js_in_projection(self, mongo_backend):
        req = QueryRequest(operation="find", collection="users", projection=_JS_PROJECTION)
        result = MQLValidator(mongo_backend).validate(req)
        assert not result.valid
        assert any("$function" in e for e in result.errors)

    def test_runner_rejects_js_in_projection_without_validation(self, mongo_backend):
        req = QueryRequest(operation="find", collection="users", projection=_JS_PROJECTION)
        with pytest.raises(ValidationError, match=r"\$function"):
            mongo_backend.execute_query(req)

    async def test_run_mql_tool_blocks_js_projection(self, mongo_backend):
        # With validation on, the tool returns a failed ToolResult.
        tool = RunMQLTool(mongo_backend, validate=True)
        result = await tool.execute(
            operation="find", collection="users", projection=_JS_PROJECTION
        )
        assert result.success is False
        assert "$function" in (result.error or "")
        # With validation off, the runner still refuses (raises; the registry
        # turns it into a failed ToolResult for the LLM loop).
        with pytest.raises(ValidationError, match=r"\$function"):
            await RunMQLTool(mongo_backend, validate=False).execute(
                operation="find", collection="users", projection=_JS_PROJECTION
            )


# ---------------------------------------------------------------------------
# Ring 2 — stage allowlist in the runner, top level and nested
# ---------------------------------------------------------------------------


class TestStageAllowlist:
    def test_unknown_top_level_stage_is_reported(self):
        assert find_disallowed_stages([{"$match": {}}, {"$currentOp": {}}]) == ["$currentOp"]

    def test_nested_stage_inside_lookup_and_facet(self):
        pipeline = [
            {"$lookup": {"from": "orders", "pipeline": [{"$listSearchIndexes": {}}], "as": "o"}},
            {"$facet": {"a": [{"$match": {}}], "b": [{"$bogusStage": 1}]}},
        ]
        assert find_disallowed_stages(pipeline) == ["$bogusStage", "$listSearchIndexes"]

    def test_runner_blocks_admin_stage_even_with_validation_off(self, mongo_backend):
        req = QueryRequest(
            operation="aggregate", collection="users", pipeline=[{"$currentOp": {}}]
        )
        with pytest.raises(ValidationError, match=r"\$currentOp"):
            mongo_backend.execute_query(req)

    def test_runner_blocks_nested_unknown_stage(self, mongo_backend):
        req = QueryRequest(
            operation="aggregate",
            collection="users",
            pipeline=[{"$facet": {"x": [{"$notAStage": {}}]}}],
        )
        with pytest.raises(ValidationError, match=r"\$notAStage"):
            mongo_backend.execute_query(req)

    def test_validator_reports_nested_unknown_stage(self, mongo_backend):
        req = QueryRequest(
            operation="aggregate",
            collection="users",
            pipeline=[{"$lookup": {"from": "orders", "pipeline": [{"$typo": {}}], "as": "o"}}],
        )
        result = MQLValidator(mongo_backend).validate(req)
        assert not result.valid
        assert any("$typo" in e for e in result.errors)

    def test_legitimate_nested_pipeline_passes(self, mongo_backend):
        req = QueryRequest(
            operation="aggregate",
            collection="users",
            pipeline=[
                {"$lookup": {"from": "orders", "pipeline": [{"$match": {"qty": {"$gt": 1}}}], "as": "o"}},
                {"$limit": 5},
            ],
        )
        assert MQLValidator(mongo_backend).validate(req).valid
        enforce_read_only_policy(
            collection="users", filter_doc=None, pipeline=req.pipeline
        )  # does not raise


# ---------------------------------------------------------------------------
# Ring 3 — collection access policy
# ---------------------------------------------------------------------------


class TestCollectionPolicyUnit:
    def test_system_collections_always_denied(self):
        assert CollectionPolicy().is_allowed("system.users") is False
        assert CollectionPolicy(allowed=["system.users"]).is_allowed("system.users") is False

    def test_allow_and_deny_lists(self):
        pol = CollectionPolicy(allowed=["a", "b"], denied=["b"])
        assert pol.filter(["a", "b", "c"]) == ["a"]
        with pytest.raises(ValidationError):
            pol.check("c")

    def test_referenced_collections_are_collected(self):
        pipeline = [
            {"$lookup": {"from": "orders", "as": "o", "pipeline": [
                {"$unionWith": {"coll": "refunds"}},
            ]}},
            {"$unionWith": "archive"},
            {"$graphLookup": {"from": "users", "startWith": "$x", "connectFromField": "a",
                              "connectToField": "b", "as": "g"}},
        ]
        assert referenced_collections(pipeline) == {"orders", "refunds", "archive", "users"}


@pytest.fixture
def scoped_backend(mongo_client, mongo_db):
    backend = MongoRunner(allowed_collections=["users"])
    backend._client = mongo_client
    backend._db = mongo_db
    return backend


class TestCollectionPolicyOnRunner:
    def test_listing_is_filtered(self, scoped_backend):
        assert scoped_backend.list_collections() == ["users"]

    def test_denied_target_is_rejected(self, scoped_backend):
        req = QueryRequest(operation="find", collection="orders")
        with pytest.raises(ValidationError, match="not accessible"):
            scoped_backend.execute_query(req)

    def test_lookup_into_denied_collection_is_rejected(self, scoped_backend):
        req = QueryRequest(
            operation="aggregate",
            collection="users",
            pipeline=[{"$lookup": {"from": "orders", "localField": "_id",
                                   "foreignField": "user_id", "as": "o"}}],
        )
        with pytest.raises(ValidationError, match="'orders' is not accessible"):
            scoped_backend.execute_query(req)

    def test_allowed_target_still_works(self, scoped_backend):
        rows = scoped_backend.execute_query(
            QueryRequest(operation="count", collection="users")
        )
        assert rows == [{"count": 3}]

    def test_introspection_and_profiling_respect_policy(self, scoped_backend):
        with pytest.raises(ValidationError):
            scoped_backend._introspect_collection("orders", {"users", "orders"})
        with pytest.raises(ValidationError):
            scoped_backend.profile_field("orders", "product")
        assert scoped_backend.field_types("orders") == {}
        assert "orders" not in scoped_backend.introspect_schema()

    async def test_describe_tool_reports_denied_collection_as_missing(self, scoped_backend):
        result = await DescribeCollectionTool(scoped_backend).execute(collection="orders")
        assert result.success is False

    def test_deny_list_variant(self, mongo_client, mongo_db):
        backend = MongoRunner(denied_collections=["orders"])
        backend._client = mongo_client
        backend._db = mongo_db
        assert backend.list_collections() == ["users"]


# ---------------------------------------------------------------------------
# Real MongoDB (opt-in via MONGODB_URI): prove the vector exists on mongod and
# that the runner closes it; check the role warning against real users.
# ---------------------------------------------------------------------------


@pytest.fixture
def real_scratch():
    import os

    import pymongo

    uri = os.getenv("MANGO_TEST_MONGO_URI", "mongodb://localhost:27017/")
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip("No local MongoDB reachable")
    db_name = "mango_sectest_policy"
    client.drop_database(db_name)
    db = client[db_name]
    db["src"].insert_many([{"x": 1}, {"x": 2}])
    db["secret"].insert_many([{"token": "s3"}])
    try:
        yield client, db
    finally:
        client.drop_database(db_name)


_JS_PROJ = {"_id": 0, "x": 1, "js": {"$function": {
    "body": "function(x){ return 'JS-RAN:' + (x * 100); }", "args": ["$x"], "lang": "js",
}}}


@pytest.mark.mongodb
class TestRealMongoPolicy:
    def test_projection_js_runs_raw_but_is_blocked_by_runner(self, real_scratch):
        client, db = real_scratch
        # The vector is real: mongod evaluates $function inside a find projection.
        raw = list(db["src"].find({}, _JS_PROJ))
        assert raw and raw[0]["js"] == "JS-RAN:100"

        backend = MongoRunner()
        backend._client, backend._db = client, db
        with pytest.raises(ValidationError, match=r"\$function"):
            backend.execute_query(
                QueryRequest(operation="find", collection="src", projection=_JS_PROJ)
            )

    def test_admin_stage_blocked_before_reaching_server(self, real_scratch):
        client, db = real_scratch
        backend = MongoRunner()
        backend._client, backend._db = client, db
        with pytest.raises(ValidationError, match=r"\$currentOp"):
            backend.execute_query(
                QueryRequest(operation="aggregate", collection="src",
                             pipeline=[{"$currentOp": {}}])
            )

    def test_lookup_into_denied_collection_on_real_server(self, real_scratch):
        client, db = real_scratch
        backend = MongoRunner(denied_collections=["secret"])
        backend._client, backend._db = client, db
        assert "secret" not in backend.list_collections()
        with pytest.raises(ValidationError, match="'secret' is not accessible"):
            backend.execute_query(QueryRequest(
                operation="aggregate", collection="src",
                pipeline=[{"$lookup": {"from": "secret", "pipeline": [], "as": "s"}}],
            ))
        # Sanity: a legitimate query on the same server still works.
        assert backend.execute_query(
            QueryRequest(operation="count", collection="src")
        ) == [{"count": 2}]

    @pytest.mark.parametrize(
        "role, expect_warning",
        [("readWrite", True), ("read", False)],
    )
    def test_role_warning_reflects_real_user_roles(self, real_scratch, caplog, role, expect_warning):
        import logging

        import pymongo

        client, db = real_scratch
        user = f"mango_policy_{role}"
        try:
            db.command("dropUser", user)
        except pymongo.errors.OperationFailure:
            pass
        db.command("createUser", user, pwd="pw", roles=[{"role": role, "db": db.name}])
        try:
            backend = MongoRunner()
            uri = f"mongodb://{user}:pw@localhost:27017/{db.name}?authSource={db.name}"
            with caplog.at_level(logging.WARNING, logger="mango.integrations.mongodb"):
                try:
                    backend.connect(uri, serverSelectionTimeoutMS=2000)
                except Exception as exc:  # server without auth support for users
                    pytest.skip(f"cannot authenticate as created user: {exc}")
            warned = any("write-capable role" in r.getMessage() for r in caplog.records)
            assert warned is expect_warning
            if warned:
                assert role in caplog.text
        finally:
            db.command("dropUser", user)
