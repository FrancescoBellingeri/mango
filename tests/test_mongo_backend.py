"""Tests for mango.backends.mongodb — MongoBackend with mongomock."""

from __future__ import annotations

import pytest

from mango.integrations.mongodb import (
    MongoRunner as MongoBackend,
    _annotate_references,
    _bson_type_name,
    _docs_to_dataframe,
    _infer_fields,
    _stringify_bson,
)
from mango.core.types import FieldInfo, QueryRequest, ValidationError


# ---------------------------------------------------------------------------
# _stringify_bson
# ---------------------------------------------------------------------------


class TestStringifyBson:
    def test_primitives_unchanged(self):
        assert _stringify_bson(42) == 42
        assert _stringify_bson("hello") == "hello"
        assert _stringify_bson(3.14) == 3.14
        assert _stringify_bson(True) is True
        assert _stringify_bson(None) is None

    def test_dict_recursed(self):
        result = _stringify_bson({"a": 1, "b": "x"})
        assert result == {"a": 1, "b": "x"}

    def test_list_recursed(self):
        result = _stringify_bson([1, "two", None])
        assert result == [1, "two", None]

    def test_nested_structure(self):
        result = _stringify_bson({"nested": {"value": 99}})
        assert result["nested"]["value"] == 99


class TestBsonTypeName:
    def test_string(self):
        assert _bson_type_name("hello") == "string"

    def test_int(self):
        assert _bson_type_name(42) == "int"

    def test_float(self):
        assert _bson_type_name(3.14) == "float"

    def test_bool(self):
        assert _bson_type_name(True) == "bool"

    def test_none(self):
        assert _bson_type_name(None) == "null"

    def test_list(self):
        assert _bson_type_name([]) == "array"

    def test_dict(self):
        assert _bson_type_name({}) == "subdocument"

    def test_datetime(self):
        from datetime import datetime
        assert _bson_type_name(datetime.now()) == "date"


# ---------------------------------------------------------------------------
# _docs_to_dataframe
# ---------------------------------------------------------------------------


class TestDocsToDataframe:
    def test_empty_list_returns_empty_dataframe(self):
        df = _docs_to_dataframe([])
        assert df.empty

    def test_single_doc(self):
        df = _docs_to_dataframe([{"name": "Alice", "age": 30}])
        assert len(df) == 1
        assert list(df.columns) == ["name", "age"]

    def test_multiple_docs(self):
        docs = [{"x": 1}, {"x": 2}, {"x": 3}]
        df = _docs_to_dataframe(docs)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# _infer_fields
# ---------------------------------------------------------------------------


class TestInferFields:
    def test_empty_returns_empty(self):
        assert _infer_fields([]) == []

    def test_single_doc(self):
        docs = [{"name": "Alice", "age": 30}]
        fields = _infer_fields(docs)
        paths = {f.path for f in fields}
        assert "name" in paths
        assert "age" in paths

    def test_type_inference(self):
        docs = [{"score": 95}]
        fields = _infer_fields(docs)
        score_field = next(f for f in fields if f.path == "score")
        assert "int" in score_field.types

    def test_frequency_all_present(self):
        docs = [{"a": 1}, {"a": 2}]
        fields = _infer_fields(docs)
        f = next(x for x in fields if x.path == "a")
        assert f.frequency == 1.0

    def test_frequency_partial(self):
        docs = [{"a": 1}, {"b": 2}]
        fields = _infer_fields(docs)
        f_a = next((x for x in fields if x.path == "a"), None)
        assert f_a is not None
        assert f_a.frequency == 0.5

    def test_nested_subdocument(self):
        docs = [{"address": {"city": "Rome", "zip": "00100"}}]
        fields = _infer_fields(docs)
        addr = next(f for f in fields if f.path == "address")
        assert addr.sub_fields is not None
        sub_paths = {f.path for f in addr.sub_fields}
        assert "address.city" in sub_paths

    def test_array_element_types(self):
        docs = [{"tags": ["python", "mongodb"]}]
        fields = _infer_fields(docs)
        tags = next(f for f in fields if f.path == "tags")
        assert tags.array_element_types is not None
        assert "string" in tags.array_element_types


# ---------------------------------------------------------------------------
# _annotate_references
# ---------------------------------------------------------------------------


class TestAnnotateReferences:
    def test_user_id_detected(self):
        fields = [FieldInfo(name="user_id", path="user_id", types=["string"], frequency=1.0)]
        _annotate_references(fields, {"users", "orders"})
        assert fields[0].is_reference is True
        assert fields[0].reference_collection == "users"

    def test_no_matching_collection(self):
        fields = [FieldInfo(name="foo_id", path="foo_id", types=["string"], frequency=1.0)]
        _annotate_references(fields, {"users"})
        assert fields[0].is_reference is False

    def test_plain_id_field_not_flagged(self):
        fields = [FieldInfo(name="_id", path="_id", types=["ObjectId"], frequency=1.0)]
        _annotate_references(fields, {"users"})
        assert fields[0].is_reference is False


# ---------------------------------------------------------------------------
# MongoBackend — query execution
# ---------------------------------------------------------------------------


class TestMongoBackendExecuteQuery:
    def test_list_collections(self, mongo_backend):
        collections = mongo_backend.list_collections()
        assert "users" in collections
        assert "orders" in collections

    def test_list_collections_sorted(self, mongo_backend):
        collections = mongo_backend.list_collections()
        assert collections == sorted(collections)

    def test_find_returns_all(self, mongo_backend):
        req = QueryRequest(operation="find", collection="users")
        df = mongo_backend.execute_query(req)
        assert len(df) == 3

    def test_find_with_filter(self, mongo_backend):
        req = QueryRequest(
            operation="find",
            collection="users",
            filter={"name": "Alice"},
        )
        df = mongo_backend.execute_query(req)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Alice"

    def test_count_all(self, mongo_backend):
        req = QueryRequest(operation="count", collection="users")
        df = mongo_backend.execute_query(req)
        assert df.iloc[0]["count"] == 3

    def test_count_with_filter(self, mongo_backend):
        req = QueryRequest(
            operation="count",
            collection="users",
            filter={"age": {"$gt": 26}},
        )
        df = mongo_backend.execute_query(req)
        assert df.iloc[0]["count"] == 2  # Alice (30) and Charlie (35)

    def test_aggregate_group(self, mongo_backend):
        pipeline = [{"$group": {"_id": "$product", "total": {"$sum": "$qty"}}}]
        req = QueryRequest(operation="aggregate", collection="orders", pipeline=pipeline)
        df = mongo_backend.execute_query(req)
        assert len(df) == 2  # Widget and Gadget

    def test_distinct(self, mongo_backend):
        req = QueryRequest(
            operation="distinct",
            collection="orders",
            distinct_field="product",
        )
        df = mongo_backend.execute_query(req)
        assert set(df["product"]) == {"Widget", "Gadget"}

    def test_invalid_operation_raises_validation_error(self, mongo_backend):
        req = QueryRequest(operation="delete", collection="users")
        with pytest.raises(ValidationError):
            mongo_backend.execute_query(req)

    def test_not_connected_raises(self):
        backend = MongoBackend()
        with pytest.raises(Exception, match="Not connected"):
            backend.list_collections()
