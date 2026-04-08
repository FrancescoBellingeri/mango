"""MongoDB backend implementation.

Concrete implementation of NoSQLBackend using pymongo.
All operations are read-only: find, aggregate, count, distinct.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import pandas as pd
import pymongo
from pymongo import MongoClient
from pymongo.database import Database

from mango.nosql_runner import NoSQLRunner
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

    def __init__(self) -> None:
        self._client: MongoClient | None = None
        self._db: Database | None = None

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
            # Extract the database name from the URI.
            db_name = self._client.get_default_database().name
            self._db = self._client[db_name]
            # Force a real connection check.
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
        """Return the active database, raising if not connected."""
        if self._db is None:
            raise BackendError("Not connected. Call connect() first.")
        return self._db

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute_query(self, operation: QueryRequest) -> pd.DataFrame:
        """Execute a read-only query and return results as a DataFrame.

        Args:
            operation: Standardized query request.

        Returns:
            DataFrame with query results. Count returns a single-row
            DataFrame with a 'count' column.

        Raises:
            ValidationError: If the operation type is not allowed.
            QueryError: If MongoDB raises an error during execution.
        """
        if operation.operation not in _ALLOWED_OPERATIONS:
            raise ValidationError(
                f"Operation '{operation.operation}' is not allowed. "
                f"Allowed: {sorted(_ALLOWED_OPERATIONS)}"
            )

        collection = self._database[operation.collection]

        try:
            if operation.operation == "find":
                return self._execute_find(collection, operation)
            elif operation.operation == "aggregate":
                return self._execute_aggregate(collection, operation)
            elif operation.operation == "count":
                return self._execute_count(collection, operation)
            elif operation.operation == "distinct":
                return self._execute_distinct(collection, operation)
        except pymongo.errors.PyMongoError as exc:
            raise QueryError(
                f"MongoDB error on {operation.operation} "
                f"'{operation.collection}': {exc}"
            ) from exc

        # Should never reach here given the allowlist check above.
        raise ValidationError(f"Unhandled operation: {operation.operation}")

    def _execute_find(
        self, collection: pymongo.collection.Collection, req: QueryRequest
    ) -> pd.DataFrame:
        cursor = collection.find(
            filter=req.filter or {},
            projection=req.projection,
            sort=list(req.sort.items()) if req.sort else None,
            limit=req.limit or 0,
        )
        docs = list(cursor)
        return _docs_to_dataframe(docs)

    def _execute_aggregate(
        self, collection: pymongo.collection.Collection, req: QueryRequest
    ) -> pd.DataFrame:
        pipeline = req.pipeline or []
        docs = list(collection.aggregate(pipeline))
        return _docs_to_dataframe(docs)

    def _execute_count(
        self, collection: pymongo.collection.Collection, req: QueryRequest
    ) -> pd.DataFrame:
        count = collection.count_documents(req.filter or {})
        return pd.DataFrame([{"count": count}])

    def _execute_distinct(
        self, collection: pymongo.collection.Collection, req: QueryRequest
    ) -> pd.DataFrame:
        if not req.distinct_field:
            raise ValidationError(
                "distinct_field must be set for 'distinct' operations."
            )
        values = collection.distinct(req.distinct_field, filter=req.filter or {})
        return pd.DataFrame({req.distinct_field: values})

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
        for name in self.list_collections():
            try:
                schema[name] = self._introspect_collection(name)
                logger.debug("Introspected schema for '%s'", name)
            except Exception as exc:
                logger.warning("Failed to introspect '%s': %s", name, exc)
        return schema

    def _introspect_collection(self, collection_name: str) -> SchemaInfo:
        collection = self._database[collection_name]

        # Document count (approximate for speed on large collections).
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

        # Infer field metadata from sampled documents.
        fields = _infer_fields(samples)

        # Index information.
        indexes = self.get_indexes(collection_name)
        indexed_fields = _extract_indexed_fields(indexes)
        unique_fields = _extract_unique_fields(indexes)

        # Annotate fields with index information.
        _annotate_indexes(fields, indexed_fields, unique_fields)

        # Reference detection.
        all_collections = set(self.list_collections())
        _annotate_references(fields, all_collections)

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

    def get_sample_documents(
        self, collection: str, n: int = 5
    ) -> list[dict]:
        """Return N sample documents from a collection.

        Args:
            collection: Collection name.
            n: Number of documents to return.

        Returns:
            List of raw document dicts.

        Raises:
            BackendError: If the collection does not exist.
        """
        if collection not in self.list_collections():
            raise BackendError(f"Collection '{collection}' does not exist.")
        return list(self._database[collection].find().limit(n))

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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _docs_to_dataframe(docs: list[dict]) -> pd.DataFrame:
    """Convert a list of MongoDB documents to a DataFrame.

    ObjectId and other BSON types are converted to strings so pandas
    can handle them without issues.
    """
    if not docs:
        return pd.DataFrame()
    # Stringify non-serialisable BSON types (_id, ObjectId, etc.).
    clean = [_stringify_bson(doc) for doc in docs]
    return pd.DataFrame(clean)


def _stringify_bson(obj: Any) -> Any:
    """Recursively convert BSON types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _stringify_bson(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_bson(v) for v in obj]
    # ObjectId, Decimal128, etc. all have a useful __str__.
    type_name = type(obj).__name__
    if type_name in {"ObjectId", "Decimal128", "Binary", "Code", "Regex"}:
        return str(obj)
    return obj


def _infer_fields(docs: list[dict], prefix: str = "") -> list[FieldInfo]:
    """Infer FieldInfo list from a list of sampled documents.

    Recursively handles nested subdocuments.
    """
    if not docs:
        return []

    # Aggregate type info per field path.
    field_types: dict[str, set[str]] = defaultdict(set)
    field_counts: dict[str, int] = defaultdict(int)
    sub_docs: dict[str, list[dict]] = defaultdict(list)
    array_element_types: dict[str, set[str]] = defaultdict(set)

    total = len(docs)

    for doc in docs:
        for key, value in doc.items():
            path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            field_counts[path] += 1
            type_name = _bson_type_name(value)
            field_types[path].add(type_name)

            if isinstance(value, dict):
                sub_docs[path].append(value)
            elif isinstance(value, list):
                for item in value:
                    array_element_types[path].add(_bson_type_name(item))

    fields: list[FieldInfo] = []
    for path, types in field_types.items():
        name = path.split(".")[-1]
        frequency = field_counts[path] / total

        sub_fields = None
        if sub_docs[path]:
            sub_fields = _infer_fields(sub_docs[path], prefix=path)

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


def _bson_type_name(value: Any) -> str:
    """Return a human-readable type name for a BSON/Python value."""
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


def _extract_indexed_fields(indexes: list[dict]) -> set[str]:
    """Extract all field names that are part of any index."""
    fields: set[str] = set()
    for idx in indexes:
        for key_tuple in idx.get("key", []):
            fields.add(key_tuple[0])
    return fields


def _extract_unique_fields(indexes: list[dict]) -> set[str]:
    """Extract field names that are part of a unique index."""
    fields: set[str] = set()
    for idx in indexes:
        if idx.get("unique"):
            for key_tuple in idx.get("key", []):
                fields.add(key_tuple[0])
    return fields


def _annotate_indexes(
    fields: list[FieldInfo],
    indexed: set[str],
    unique: set[str],
) -> None:
    """Set is_indexed and is_unique on FieldInfo objects in place."""
    for f in fields:
        f.is_indexed = f.path in indexed
        f.is_unique = f.path in unique
        if f.sub_fields:
            _annotate_indexes(f.sub_fields, indexed, unique)


def _annotate_references(
    fields: list[FieldInfo],
    all_collections: set[str],
) -> None:
    """Heuristic reference detection.

    A field named 'user_id' is flagged as a reference to 'users' if
    'users' exists as a collection. Handles both '_id' suffix and 'Id' suffix.
    """
    for f in fields:
        name = f.name
        candidate: str | None = None

        if name.endswith("_id") and name != "_id":
            candidate = name[:-3] + "s"     # user_id → users
        elif name.endswith("Id") and name != "Id":
            candidate = name[:-2].lower() + "s"   # userId → users

        if candidate and candidate in all_collections:
            f.is_reference = True
            f.reference_collection = candidate

        if f.sub_fields:
            _annotate_references(f.sub_fields, all_collections)
