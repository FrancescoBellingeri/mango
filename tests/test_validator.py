"""Tests for mango.tools.validator — MQLValidator and ValidationResult."""

from __future__ import annotations

from mango.core.types import QueryRequest
from mango.tools.validator import MQLValidator, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_validator(mongo_backend) -> MQLValidator:
    return MQLValidator(mongo_backend)


def make_request(**kwargs) -> QueryRequest:
    defaults = dict(operation="find", collection="users")
    defaults.update(kwargs)
    return QueryRequest(**defaults)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_valid_when_no_errors(self):
        r = ValidationResult(valid=True)
        assert r.valid is True
        assert r.errors == []
        assert r.warnings == []

    def test_as_tool_error_includes_errors_and_warnings(self):
        r = ValidationResult(
            valid=False,
            errors=["Collection 'x' does not exist."],
            warnings=["Unknown operator '$typo'."],
        )
        text = r.as_tool_error()
        assert "ERROR" in text
        assert "Collection 'x' does not exist." in text
        assert "WARNING" in text
        assert "'$typo'" in text

    def test_as_tool_error_only_warnings(self):
        r = ValidationResult(valid=True, warnings=["Unknown operator '$foo'."])
        text = r.as_tool_error()
        assert "WARNING" in text
        assert "ERROR" not in text


# ---------------------------------------------------------------------------
# Operation check
# ---------------------------------------------------------------------------


class TestOperationCheck:
    def test_valid_operations_pass(self, mongo_backend):
        v = make_validator(mongo_backend)
        for op in ("find", "count", "distinct", "aggregate"):
            extra = {}
            if op == "aggregate":
                extra["pipeline"] = [{"$match": {}}]
            if op == "distinct":
                extra["distinct_field"] = "name"
            r = v.validate(make_request(operation=op, **extra))
            assert op not in [e for e in r.errors if "not allowed" in e]

    def test_write_operation_blocked(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(operation="insert"))
        assert not r.valid
        assert any("not allowed" in e for e in r.errors)

    def test_delete_operation_blocked(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(operation="delete"))
        assert not r.valid

    def test_drop_operation_blocked(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(operation="drop"))
        assert not r.valid


# ---------------------------------------------------------------------------
# Collection check
# ---------------------------------------------------------------------------


class TestCollectionCheck:
    def test_existing_collection_passes(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(collection="users"))
        assert not any("does not exist" in e for e in r.errors)

    def test_nonexistent_collection_errors(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(collection="nonexistent_xyz"))
        assert not r.valid
        assert any("does not exist" in e for e in r.errors)

    def test_suggestion_for_similar_name(self, mongo_backend):
        v = make_validator(mongo_backend)
        # "user" is close to "users"
        r = v.validate(make_request(collection="user"))
        assert not r.valid
        error_text = " ".join(r.errors)
        assert "users" in error_text  # suggestion included


# ---------------------------------------------------------------------------
# Required args check
# ---------------------------------------------------------------------------


class TestRequiredArgsCheck:
    def test_aggregate_without_pipeline_errors(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(operation="aggregate", collection="users"))
        assert not r.valid
        assert any("pipeline" in e for e in r.errors)

    def test_aggregate_with_pipeline_passes(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            operation="aggregate",
            collection="users",
            pipeline=[{"$match": {"name": "Alice"}}],
        ))
        assert not any("pipeline" in e for e in r.errors)

    def test_distinct_without_field_errors(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(operation="distinct", collection="users"))
        assert not r.valid
        assert any("distinct_field" in e for e in r.errors)

    def test_distinct_with_field_passes(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            operation="distinct",
            collection="users",
            distinct_field="name",
        ))
        assert not any("distinct_field" in e for e in r.errors)


# ---------------------------------------------------------------------------
# Pipeline stage check
# ---------------------------------------------------------------------------


class TestPipelineStageCheck:
    def test_valid_stages_pass(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            operation="aggregate",
            collection="orders",
            pipeline=[
                {"$match": {"qty": {"$gt": 1}}},
                {"$group": {"_id": "$product", "total": {"$sum": "$qty"}}},
                {"$sort": {"total": -1}},
                {"$limit": 10},
            ],
        ))
        assert not any("Unknown aggregation stage" in e for e in r.errors)

    def test_unknown_stage_errors(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            operation="aggregate",
            collection="users",
            pipeline=[{"$fakeStage": {}}],
        ))
        assert not r.valid
        assert any("Unknown aggregation stage" in e for e in r.errors)

    def test_misspelled_stage_errors(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            operation="aggregate",
            collection="users",
            pipeline=[{"$metch": {}}],  # typo: $metch instead of $match
        ))
        assert not r.valid
        assert any("$metch" in e for e in r.errors)

    def test_non_dict_stage_errors(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            operation="aggregate",
            collection="users",
            pipeline=["not_a_dict"],
        ))
        assert not r.valid

    def test_multi_key_stage_errors(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            operation="aggregate",
            collection="users",
            pipeline=[{"$match": {}, "$limit": 5}],
        ))
        assert not r.valid
        assert any("has 2 keys" in e for e in r.errors)


# ---------------------------------------------------------------------------
# Operator check (warnings)
# ---------------------------------------------------------------------------


class TestOperatorCheck:
    def test_valid_filter_operators_no_warnings(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            collection="users",
            filter={"age": {"$gt": 25, "$lte": 40}},
        ))
        assert r.warnings == []

    def test_unknown_operator_produces_warning(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            collection="users",
            filter={"name": {"$typoOp": "Alice"}},
        ))
        assert r.valid  # warnings don't block
        assert any("$typoOp" in w for w in r.warnings)

    def test_unknown_operator_in_pipeline_body_produces_warning(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            operation="aggregate",
            collection="orders",
            pipeline=[{"$match": {"qty": {"$notRealOp": 5}}}],
        ))
        assert r.valid
        assert any("$notRealOp" in w for w in r.warnings)

    def test_duplicate_unknown_operators_deduplicated(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            collection="users",
            filter={
                "a": {"$fake": 1},
                "b": {"$fake": 2},
            },
        ))
        assert sum(1 for w in r.warnings if "$fake" in w) == 1

    def test_known_accumulator_in_group_no_warning(self, mongo_backend):
        v = make_validator(mongo_backend)
        r = v.validate(make_request(
            operation="aggregate",
            collection="orders",
            pipeline=[
                {"$group": {"_id": "$product", "total": {"$sum": "$qty"}, "avg": {"$avg": "$qty"}}},
            ],
        ))
        assert r.warnings == []


# ---------------------------------------------------------------------------
# Multiple errors collected
# ---------------------------------------------------------------------------


class TestMultipleErrors:
    def test_all_errors_collected(self, mongo_backend):
        v = make_validator(mongo_backend)
        # Bad operation + bad collection + no pipeline
        r = v.validate(QueryRequest(
            operation="drop",
            collection="nonexistent_xyz",
            pipeline=None,
        ))
        assert not r.valid
        assert len(r.errors) >= 2  # operation error + collection error


# ---------------------------------------------------------------------------
# RunMQLTool integration
# ---------------------------------------------------------------------------


class TestRunMQLToolValidation:
    async def test_valid_query_executes(self, mongo_backend):
        from mango.tools.mongo_tools import RunMQLTool
        tool = RunMQLTool(mongo_backend, validate=True)
        result = await tool.execute(operation="find", collection="users")
        assert result.success is True

    async def test_invalid_collection_blocked(self, mongo_backend):
        from mango.tools.mongo_tools import RunMQLTool
        tool = RunMQLTool(mongo_backend, validate=True)
        result = await tool.execute(operation="find", collection="ghost_collection")
        assert result.success is False
        assert "does not exist" in result.error

    async def test_validate_false_skips_validation(self, mongo_backend):
        from mango.tools.mongo_tools import RunMQLTool
        # With validate=False, a bad collection name goes straight to the backend.
        tool = RunMQLTool(mongo_backend, validate=False)
        # mongomock returns empty for unknown collections rather than raising.
        result = await tool.execute(operation="find", collection="ghost_collection")
        assert result.success is True
        assert result.data["row_count"] == 0

    async def test_aggregate_without_pipeline_blocked(self, mongo_backend):
        from mango.tools.mongo_tools import RunMQLTool
        tool = RunMQLTool(mongo_backend, validate=True)
        result = await tool.execute(operation="aggregate", collection="users")
        assert result.success is False
        assert "pipeline" in result.error

    async def test_unknown_stage_blocked(self, mongo_backend):
        from mango.tools.mongo_tools import RunMQLTool
        tool = RunMQLTool(mongo_backend, validate=True)
        result = await tool.execute(
            operation="aggregate",
            collection="users",
            pipeline=[{"$fakeStage": {}}],
        )
        assert result.success is False
        assert "fakeStage" in result.error

    async def test_warnings_surfaced_in_result(self, mongo_backend):
        from unittest.mock import patch
        from mango.tools.mongo_tools import RunMQLTool
        from mango.tools.validator import ValidationResult
        tool = RunMQLTool(mongo_backend, validate=True)
        # Inject a warning via mock — the query itself is valid so mongomock executes it.
        # This tests that RunMQLTool includes validator warnings in the result data
        # without needing a query that would actually fail at the backend level.
        fake_result = ValidationResult(
            valid=True,
            warnings=["Unrecognised operator '$unknownOp' — verify this is a valid MQL operator."],
        )
        with patch.object(tool._validator, "validate", return_value=fake_result):
            result = await tool.execute(operation="find", collection="users")
        assert result.success is True
        assert "validation_warnings" in result.data
        assert any("$unknownOp" in w for w in result.data["validation_warnings"])

    async def test_no_warnings_key_when_clean(self, mongo_backend):
        from mango.tools.mongo_tools import RunMQLTool
        tool = RunMQLTool(mongo_backend, validate=True)
        result = await tool.execute(operation="find", collection="users")
        assert "validation_warnings" not in result.data
