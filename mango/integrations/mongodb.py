"""MongoDB backend implementation.

Concrete implementation of NoSQLBackend using pymongo.
All operations are read-only: find, aggregate, count, distinct.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd
import pymongo
from pymongo import MongoClient
from pymongo.database import Database

from mango.nosql_runner import NoSQLRunner
from mango.core.security import find_forbidden_operators
from mango.core.types import (
    BackendError,
    FieldInfo,
    QueryError,
    QueryRequest,
    SchemaInfo,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Operations explicitly allowed — allowlist, not blocklist.
_ALLOWED_OPERATIONS = {"find", "aggregate", "count", "distinct"}

# How many documents to sample when inferring schema.
_DEFAULT_SAMPLE_SIZE = 100


class MongoRunner(NoSQLRunner):
    """MongoDB backend using pymongo.

    Connects to a MongoDB instance and exposes read-only query execution
    and schema introspection. No write operations are possible.
    """

    def __init__(self, max_time_ms: int = 30_000) -> None:
        self._client: MongoClient | None = None
        self._db: Database | None = None
        # Server-side time budget applied to every query so a pathological
        # scan (missing index, huge $group) cannot hang the worker forever.
        # 0 disables the limit.
        self._max_time_ms = max_time_ms
        # Cache of collection → {dotted_path: set(bson_type_names)}, used for
        # schema-aware literal coercion. Populated lazily, kept for the runner's
        # lifetime (schema is stable within a session).
        self._field_type_cache: dict[str, dict[str, set[str]]] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, connection_string: str, **kwargs: object) -> None:
        """Connect to a MongoDB instance.

        Args:
            connection_string: MongoDB URI, e.g. 'mongodb://localhost:27017/mydb'.
                The database name must be included in the URI.
            **kwargs: Additional options passed to MongoClient.

        Raises:
            BackendError: If the connection fails or no database name is given.
        """
        try:
            self._client = MongoClient(connection_string, **kwargs)
            db_name = self._client.get_default_database().name
            self._db = self._client[db_name]
            self._client.admin.command("ping")
            logger.info("Connected to MongoDB database '%s'", db_name)
        except pymongo.errors.ConfigurationError as exc:
            raise BackendError(
                "No database name found in connection string. "
                "Include it in the URI: mongodb://host:port/dbname"
            ) from exc
        except pymongo.errors.ConnectionFailure as exc:
            raise BackendError(f"Cannot connect to MongoDB: {exc}") from exc

    @property
    def _database(self) -> Database:
        if self._db is None:
            raise BackendError("Not connected. Call connect() first.")
        return self._db

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute_query(self, operation: QueryRequest) -> list[dict[str, Any]]:
        """Execute a read-only query and return results as a list of rows.

        Rows are JSON-safe dicts (BSON types stringified, dates as ISO strings),
        built directly from the documents — no pandas roundtrip, so absent fields
        stay absent (not null) and ints stay ints (not widened to float).

        Args:
            operation: Standardized query request.

        Returns:
            List of row dicts. Count returns ``[{"count": N}]``.

        Raises:
            ValidationError: If the operation type is not allowed.
            QueryError: If MongoDB raises an error during execution.
        """
        if operation.operation not in _ALLOWED_OPERATIONS:
            raise ValidationError(
                f"Operation '{operation.operation}' is not allowed. "
                f"Allowed: {sorted(_ALLOWED_OPERATIONS)}"
            )

        # Defence-in-depth: block write / server-side-JS / streaming operators
        # regardless of whether pre-execution validation ran. The read-only
        # guarantee must not be optional.
        forbidden = find_forbidden_operators(operation.filter, operation.pipeline)
        if forbidden:
            raise ValidationError(
                f"Forbidden operator(s) {forbidden} are not permitted "
                "(read-only: no $out/$merge, no server-side JavaScript, "
                "no change streams or administrative stages)."
            )

        collection = self._database[operation.collection]

        try:
            if operation.operation == "find":
                return self._execute_find(collection, operation)
            elif operation.operation == "aggregate":
                return self._execute_aggregate(collection, operation)
            elif operation.operation == "count":
                return self._execute_count(collection, operation)
            else:  # distinct
                return self._execute_distinct(collection, operation)
        except pymongo.errors.ConnectionFailure as exc:
            # Infra failure (server unreachable, connection lost, server-selection
            # timeout, auto-reconnect). The LLM cannot fix this by rewriting the
            # query — surface as fatal so the agent stops retrying. Caught before
            # PyMongoError because ConnectionFailure is a subclass of it.
            raise BackendError(
                f"MongoDB connection error on {operation.operation} "
                f"'{operation.collection}': {exc}"
            ) from exc
        except pymongo.errors.PyMongoError as exc:
            # Query-level error (bad operator, execution timeout on a heavy
            # query, etc.). The LLM can retry with a corrected or simpler query.
            raise QueryError(
                f"MongoDB error on {operation.operation} "
                f"'{operation.collection}': {exc}"
            ) from exc

    @property
    def _time_kwargs(self) -> dict:
        """maxTimeMS kwarg for aggregate/count, empty when disabled."""
        return {"maxTimeMS": self._max_time_ms} if self._max_time_ms else {}

    def _execute_find(
        self, collection: pymongo.collection.Collection, req: QueryRequest
    ) -> list[dict[str, Any]]:
        cursor = collection.find(
            filter=req.filter or {},
            projection=req.projection,
            sort=list(req.sort.items()) if req.sort else None,
            limit=req.limit or 0,
            max_time_ms=self._max_time_ms or None,
        )
        return self._docs_to_rows(list(cursor))

    def _execute_aggregate(
        self, collection: pymongo.collection.Collection, req: QueryRequest
    ) -> pd.DataFrame:
        pipeline = list(req.pipeline or [])
        # Enforce the row cap: an aggregation without a terminal $limit could
        # otherwise stream unbounded results into memory and the LLM context.
        # A smaller $limit already in the pipeline runs first and still wins;
        # $limit only truncates, so appending it never changes result meaning.
        if req.limit and req.limit > 0:
            pipeline.append({"$limit": req.limit})
        return self._docs_to_rows(
            list(collection.aggregate(pipeline, **self._time_kwargs))
        )

    def _execute_count(
        self, collection: pymongo.collection.Collection, req: QueryRequest
    ) -> list[dict[str, Any]]:
        count = collection.count_documents(req.filter or {}, **self._time_kwargs)
        return [{"count": count}]

    def _execute_distinct(
        self, collection: pymongo.collection.Collection, req: QueryRequest
    ) -> list[dict[str, Any]]:
        if not req.distinct_field:
            raise ValidationError(
                "distinct_field must be set for 'distinct' operations."
            )
        # Run distinct as a bounded aggregation instead of Collection.distinct():
        # a native distinct on a high-cardinality field builds one unbounded
        # array and can hit the 16 MB BSON limit. $group streams, $limit caps,
        # and this path also carries the server-side time budget.
        field = req.distinct_field
        pipeline: list[dict] = []
        if req.filter:
            pipeline.append({"$match": req.filter})
        pipeline.append({"$group": {"_id": f"${field}"}})
        pipeline.append({"$sort": {"_id": 1}})
        if req.limit and req.limit > 0:
            pipeline.append({"$limit": req.limit})
        docs = list(collection.aggregate(pipeline, **self._time_kwargs))
        return [{field: self._json_safe(d["_id"])} for d in docs]

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def introspect_schema(self) -> dict[str, SchemaInfo]:
        """Infer schema for all collections by sampling documents.

        Returns:
            Mapping of collection name → SchemaInfo.

        Raises:
            BackendError: If introspection fails.
        """
        schema: dict[str, SchemaInfo] = {}
        all_collections = set(self.list_collections())
        for name in all_collections:
            try:
                schema[name] = self._introspect_collection(name, all_collections)
                logger.debug("Introspected schema for '%s'", name)
            except Exception as exc:
                logger.warning("Failed to introspect '%s': %s", name, exc)
        return schema

    def _introspect_collection(
        self, collection_name: str, all_collections: set[str]
    ) -> SchemaInfo:
        collection = self._database[collection_name]

        doc_count = collection.estimated_document_count()

        # Sample documents: mix of sequential + random.
        sequential = list(collection.find().limit(_DEFAULT_SAMPLE_SIZE // 2))
        random_sample = list(
            collection.aggregate(
                [{"$sample": {"size": _DEFAULT_SAMPLE_SIZE // 2}}]
            )
        )
        # Deduplicate by _id.
        seen: set = set()
        samples: list[dict] = []
        for doc in sequential + random_sample:
            doc_id = str(doc.get("_id", id(doc)))
            if doc_id not in seen:
                seen.add(doc_id)
                samples.append(doc)

        fields = self._infer_fields(samples)
        indexes = self.get_indexes(collection_name)
        indexed_fields = self._extract_indexed_fields(indexes)
        unique_fields = self._extract_unique_fields(indexes)
        self._annotate_indexes(fields, indexed_fields, unique_fields)
        self._annotate_references(fields, all_collections)

        return SchemaInfo(
            collection_name=collection_name,
            document_count=doc_count,
            fields=fields,
            indexes=indexes,
            sample_documents=samples[:5],
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_sample_documents(self, collection: str, n: int = 5) -> list[dict]:
        """Return N sample documents from a collection.

        Args:
            collection: Collection name.
            n: Number of documents to return.

        Returns:
            List of raw document dicts.

        Raises:
            BackendError: If the collection does not exist.
        """
        try:
            docs = list(self._database[collection].find().limit(n))
        except pymongo.errors.PyMongoError as exc:
            raise BackendError(f"Cannot sample '{collection}': {exc}") from exc
        if not docs and collection not in self._database.list_collection_names():
            raise BackendError(f"Collection '{collection}' does not exist.")
        return docs

    def field_types(self, collection: str, sample_size: int = 50) -> dict[str, set[str]]:
        """Return {dotted_path: set(bson_type_names)} for a collection.

        Cheap, cached, best-effort: samples up to *sample_size* documents and
        reuses the schema inference. Powers schema-aware literal coercion so a
        query string is only converted to a datetime/ObjectId when the target
        field is actually stored that way. Returns an empty map on any failure
        (coercion then safely leaves values untouched).
        """
        cached = self._field_type_cache.get(collection)
        if cached is not None:
            return cached

        try:
            docs = list(self._database[collection].find().limit(sample_size))
        except Exception:
            docs = []

        out: dict[str, set[str]] = {}

        def _walk(fields: list[FieldInfo]) -> None:
            for f in fields:
                out[f.path] = set(f.types)
                if f.sub_fields:
                    _walk(f.sub_fields)

        _walk(self._infer_fields(docs))
        self._field_type_cache[collection] = out
        return out

    def list_collections(self) -> list[str]:
        """Return sorted list of all collection names.

        Raises:
            BackendError: If listing fails.
        """
        try:
            return sorted(self._database.list_collection_names())
        except pymongo.errors.PyMongoError as exc:
            raise BackendError(f"Cannot list collections: {exc}") from exc

    def get_indexes(self, collection: str) -> list[dict]:
        """Return index definitions for a collection.

        Args:
            collection: Collection name.

        Returns:
            List of index definition dicts.

        Raises:
            BackendError: If the collection does not exist.
        """
        try:
            return list(
                self._database[collection].index_information().values()
            )
        except pymongo.errors.PyMongoError as exc:
            raise BackendError(
                f"Cannot get indexes for '{collection}': {exc}"
            ) from exc

    def profile_field(
        self,
        collection: str,
        field: str,
        top_k: int = 20,
        sample_threshold: int = 2_000_000,
        sample_size: int = 100_000,
    ) -> dict:
        """Profile the actual values stored in one field of a collection.

        Returns the most frequent values (with counts), the distinct-value
        cardinality, and a breakdown of the BSON types observed. This surfaces
        what neither the inferred schema (types only) nor a 3-document sample can:
        that a categorical field may store the same concept under different
        encodings (``active`` vs ``ACTIVE``, ``true`` vs ``"yes"`` vs ``1``).

        Exact by default. For collections above ``sample_threshold`` documents it
        falls back to a ``$sample`` pass and flags ``sampled=True`` (rare values
        may then be missed — the trade-off is documented to the caller).

        ``field`` may be a dotted path (``a.b.c``). Each sub-aggregation degrades
        gracefully: a backend that does not support a stage yields an empty part
        rather than failing the whole call.

        Raises:
            BackendError: If the collection cannot be accessed at all.
        """
        try:
            coll = self._database[collection]
            total = coll.estimated_document_count()
        except pymongo.errors.PyMongoError as exc:
            raise BackendError(
                f"Cannot profile '{field}' on '{collection}': {exc}"
            ) from exc

        sampled = total > sample_threshold
        pre: list[dict] = [{"$sample": {"size": sample_size}}] if sampled else []
        field_ref = f"${field}"

        def _agg(stages: list[dict]) -> list[dict]:
            try:
                return list(coll.aggregate(pre + stages, allowDiskUse=True))
            except Exception as exc:  # backend may lack a stage (e.g. mongomock)
                logger.debug("profile_field sub-aggregation failed: %s", exc)
                return []

        top = _agg([
            {"$group": {"_id": field_ref, "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "_id": 1}},
            {"$limit": top_k},
        ])
        card = _agg([
            {"$group": {"_id": field_ref}},
            {"$count": "n"},
        ])
        types = _agg([
            {"$group": {"_id": {"$type": field_ref}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ])

        distinct_count = card[0]["n"] if card else len(top)
        top_values = [
            {"value": self._stringify_bson(g["_id"]), "count": g["count"]}
            for g in top
        ]
        type_breakdown = {t["_id"]: t["count"] for t in types if t.get("_id")}

        return {
            "collection": collection,
            "field": field,
            "scanned_docs": sample_size if sampled else total,
            "sampled": sampled,
            "distinct_count": distinct_count,
            "type_breakdown": type_breakdown,
            "top_values": top_values,
            "truncated": distinct_count > len(top_values),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _docs_to_rows(docs: list[dict]) -> list[dict[str, Any]]:
        """Convert raw BSON documents to JSON-safe row dicts.

        Per-document conversion: an absent field stays absent (no null injection)
        and ints keep their type — unlike the old DataFrame roundtrip.
        """
        return [MongoRunner._json_safe(doc) for doc in docs]

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        """Recursively convert BSON/datetime values to JSON-serialisable form."""
        if isinstance(obj, dict):
            return {k: MongoRunner._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MongoRunner._json_safe(v) for v in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        type_name = type(obj).__name__
        if type_name in {"ObjectId", "Decimal128", "Binary", "Code", "Regex"}:
            return str(obj)
        return obj

    @staticmethod
    def _docs_to_dataframe(docs: list[dict]) -> pd.DataFrame:
        if not docs:
            return pd.DataFrame()
        return pd.DataFrame([MongoRunner._stringify_bson(doc) for doc in docs])

    @staticmethod
    def _stringify_bson(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: MongoRunner._stringify_bson(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MongoRunner._stringify_bson(v) for v in obj]
        # ObjectId, Decimal128, etc. all have a useful __str__.
        type_name = type(obj).__name__
        if type_name in {"ObjectId", "Decimal128", "Binary", "Code", "Regex"}:
            return str(obj)
        return obj

    @staticmethod
    def _infer_fields(docs: list[dict], prefix: str = "") -> list[FieldInfo]:
        if not docs:
            return []

        field_types: dict[str, set[str]] = defaultdict(set)
        field_counts: dict[str, int] = defaultdict(int)
        sub_docs: dict[str, list[dict]] = defaultdict(list)
        array_element_types: dict[str, set[str]] = defaultdict(set)

        total = len(docs)

        for doc in docs:
            for key, value in doc.items():
                path = key if not prefix else f"{prefix}.{key}"
                field_counts[path] += 1
                type_name = MongoRunner._bson_type_name(value)
                field_types[path].add(type_name)

                if isinstance(value, dict):
                    sub_docs[path].append(value)
                elif isinstance(value, list):
                    for item in value:
                        array_element_types[path].add(MongoRunner._bson_type_name(item))

        fields: list[FieldInfo] = []
        for path, types in field_types.items():
            name = path.split(".")[-1]
            frequency = field_counts[path] / total

            sub_fields = None
            if sub_docs[path]:
                sub_fields = MongoRunner._infer_fields(sub_docs[path], prefix=path)

            arr_types = list(array_element_types[path]) if array_element_types[path] else None

            fields.append(
                FieldInfo(
                    name=name,
                    path=path,
                    types=sorted(types),
                    frequency=round(frequency, 4),
                    array_element_types=arr_types,
                    sub_fields=sub_fields,
                )
            )

        return sorted(fields, key=lambda f: f.path)

    @staticmethod
    def _bson_type_name(value: Any) -> str:
        if value is None:
            return "null"
        type_name = type(value).__name__
        mapping = {
            "str": "string",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "list": "array",
            "dict": "subdocument",
            "datetime": "date",
            "ObjectId": "ObjectId",
            "Decimal128": "Decimal128",
            "bytes": "binary",
        }
        return mapping.get(type_name, type_name)

    @staticmethod
    def _extract_indexed_fields(indexes: list[dict]) -> set[str]:
        fields: set[str] = set()
        for idx in indexes:
            for key_tuple in idx.get("key", []):
                fields.add(key_tuple[0])
        return fields

    @staticmethod
    def _extract_unique_fields(indexes: list[dict]) -> set[str]:
        fields: set[str] = set()
        for idx in indexes:
            if idx.get("unique"):
                for key_tuple in idx.get("key", []):
                    fields.add(key_tuple[0])
        return fields

    @staticmethod
    def _annotate_indexes(
        fields: list[FieldInfo],
        indexed: set[str],
        unique: set[str],
    ) -> None:
        for f in fields:
            f.is_indexed = f.path in indexed
            f.is_unique = f.path in unique
            if f.sub_fields:
                MongoRunner._annotate_indexes(f.sub_fields, indexed, unique)

    @staticmethod
    def _annotate_references(
        fields: list[FieldInfo],
        all_collections: set[str],
    ) -> None:
        for f in fields:
            name = f.name
            candidate: str | None = None

            if name.endswith("_id") and name != "_id":
                candidate = name[:-3] + "s"         # user_id → users
            elif name.endswith("Id") and name != "Id":
                candidate = name[:-2].lower() + "s"  # userId → users

            if candidate and candidate in all_collections:
                f.is_reference = True
                f.reference_collection = candidate

            if f.sub_fields:
                MongoRunner._annotate_references(f.sub_fields, all_collections)
