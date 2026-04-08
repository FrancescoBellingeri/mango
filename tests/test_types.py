"""Tests for mango.core.types — dataclasses and exception hierarchy."""

from __future__ import annotations

import pytest

from mango.core.types import (
    BackendError,
    FieldInfo,
    LLMError,
    MangoError,
    QueryError,
    QueryRequest,
    SchemaInfo,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_mango_error_is_exception(self):
        assert issubclass(MangoError, Exception)

    def test_query_error_inherits_mango_error(self):
        assert issubclass(QueryError, MangoError)

    def test_validation_error_inherits_mango_error(self):
        assert issubclass(ValidationError, MangoError)

    def test_backend_error_inherits_mango_error(self):
        assert issubclass(BackendError, MangoError)

    def test_llm_error_inherits_mango_error(self):
        assert issubclass(LLMError, MangoError)

    def test_raise_query_error(self):
        with pytest.raises(MangoError):
            raise QueryError("bad query")

    def test_raise_validation_error(self):
        with pytest.raises(MangoError):
            raise ValidationError("invalid operation")

    def test_error_message(self):
        exc = BackendError("connection refused")
        assert "connection refused" in str(exc)


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------


class TestQueryRequest:
    def test_required_fields(self):
        req = QueryRequest(operation="find", collection="users")
        assert req.operation == "find"
        assert req.collection == "users"

    def test_defaults(self):
        req = QueryRequest(operation="count", collection="orders")
        assert req.filter is None
        assert req.pipeline is None
        assert req.projection is None
        assert req.sort is None
        assert req.limit is None
        assert req.distinct_field is None

    def test_with_filter(self):
        req = QueryRequest(
            operation="find",
            collection="users",
            filter={"age": {"$gt": 25}},
        )
        assert req.filter == {"age": {"$gt": 25}}

    def test_with_pipeline(self):
        pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        req = QueryRequest(operation="aggregate", collection="orders", pipeline=pipeline)
        assert req.pipeline == pipeline


# ---------------------------------------------------------------------------
# FieldInfo
# ---------------------------------------------------------------------------


class TestFieldInfo:
    def test_defaults(self):
        f = FieldInfo(name="age", path="age", types=["int"], frequency=1.0)
        assert f.is_indexed is False
        assert f.is_unique is False
        assert f.is_reference is False
        assert f.reference_collection is None
        assert f.array_element_types is None
        assert f.sub_fields is None

    def test_with_all_fields(self):
        f = FieldInfo(
            name="user_id",
            path="user_id",
            types=["string"],
            frequency=0.95,
            is_indexed=True,
            is_unique=False,
            is_reference=True,
            reference_collection="users",
        )
        assert f.is_reference is True
        assert f.reference_collection == "users"


# ---------------------------------------------------------------------------
# SchemaInfo
# ---------------------------------------------------------------------------


class TestSchemaInfo:
    def test_basic(self):
        fields = [FieldInfo(name="_id", path="_id", types=["ObjectId"], frequency=1.0)]
        schema = SchemaInfo(
            collection_name="users",
            document_count=42,
            fields=fields,
            indexes=[],
            sample_documents=[],
        )
        assert schema.collection_name == "users"
        assert schema.document_count == 42
        assert len(schema.fields) == 1
