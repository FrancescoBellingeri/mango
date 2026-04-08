"""Abstract base class for all NoSQL backends.

Every concrete backend (MongoDB, Redis, Cassandra, ...) must implement
this interface. No tool should import a specific driver directly —
all database interaction goes through this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from mango.core.types import QueryRequest, SchemaInfo


class NoSQLRunner(ABC):
    """Abstract interface for NoSQL database backends.

    Implementations must be read-only: no insert, update, delete, or drop
    operations are allowed. All query execution goes through execute_query()
    which receives a QueryRequest with an explicit allowlist of operations.
    """

    @abstractmethod
    def connect(self, connection_string: str, **kwargs: object) -> None:
        """Establish a connection to the database.

        Args:
            connection_string: URI or DSN for the target database.
            **kwargs: Additional driver-specific options.

        Raises:
            BackendError: If the connection cannot be established.
        """

    @abstractmethod
    def execute_query(self, operation: QueryRequest) -> pd.DataFrame:
        """Execute a read-only query and return results as a DataFrame.

        Args:
            operation: Standardized query request. Only find, aggregate,
                count, and distinct operations are allowed.

        Returns:
            A pandas DataFrame with the query results. For scalar results
            (count), a single-row DataFrame with a 'result' column is returned.

        Raises:
            QueryError: If the query fails to execute.
            ValidationError: If the operation type is not in the allowlist.
        """

    @abstractmethod
    def introspect_schema(self) -> dict[str, SchemaInfo]:
        """Infer and return the schema for all collections.

        Returns:
            Mapping of collection name → SchemaInfo with inferred fields,
            types, frequencies, indexes, and sample documents.

        Raises:
            BackendError: If schema introspection fails.
        """

    @abstractmethod
    def get_sample_documents(
        self, collection: str, n: int = 5
    ) -> list[dict]:
        """Return a sample of documents from the given collection.

        Args:
            collection: Name of the collection to sample.
            n: Number of documents to return.

        Returns:
            List of raw document dicts.

        Raises:
            BackendError: If the collection does not exist or sampling fails.
        """

    @abstractmethod
    def list_collections(self) -> list[str]:
        """Return the names of all collections in the database.

        Returns:
            Sorted list of collection names.

        Raises:
            BackendError: If the listing fails.
        """

    @abstractmethod
    def get_indexes(self, collection: str) -> list[dict]:
        """Return index definitions for the given collection.

        Args:
            collection: Name of the collection.

        Returns:
            List of index definition dicts as returned by the driver.

        Raises:
            BackendError: If the collection does not exist.
        """
