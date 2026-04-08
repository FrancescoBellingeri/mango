"""Core data structures for Mango.

All dataclasses used across the system are defined here.
No external dependencies — only Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


@dataclass
class QueryRequest:
    """Standardized query format passed to all NoSQL backends.

    This is the single source of truth for query structure.
    No tool should accept raw dicts instead of this dataclass.
    """

    operation: Literal["find", "aggregate", "count", "distinct"]
    collection: str
    filter: dict[str, Any] | None = None
    pipeline: list[dict[str, Any]] | None = None
    projection: dict[str, Any] | None = None
    sort: dict[str, Any] | None = None
    limit: int | None = None
    distinct_field: str | None = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class FieldInfo:
    """Inferred metadata for a single field in a collection."""

    name: str
    path: str                                   # dotted path e.g. 'address.city'
    types: list[str]                            # e.g. ['string', 'null']
    frequency: float                            # 0.0 - 1.0, presence rate in sampled docs
    is_indexed: bool = False
    is_unique: bool = False
    is_reference: bool = False
    reference_collection: str | None = None
    sub_fields: list[FieldInfo] | None = None   # for subdocument fields
    array_element_types: list[str] | None = None


@dataclass
class SchemaInfo:
    """Inferred schema for a single collection."""

    collection_name: str
    document_count: int
    fields: list[FieldInfo]
    indexes: list[dict[str, Any]]
    sample_documents: list[dict[str, Any]]      # 3-5 representative docs


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A stored successful interaction in agent memory."""

    id: str
    question: str                       # original natural language question
    tool_name: str                      # e.g. 'run_mql'
    tool_args: dict[str, Any]           # the exact args that produced a correct result
    result_summary: str                 # brief human-readable summary of results
    created_at: datetime = field(default_factory=datetime.utcnow)
    confirmed_by: str | None = None     # user who confirmed correctness


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MangoError(Exception):
    """Base exception for all Mango errors."""


class QueryError(MangoError):
    """Raised when a query fails to execute."""


class ValidationError(MangoError):
    """Raised when a query fails pre-execution validation."""


class BackendError(MangoError):
    """Raised when the database backend encounters an error."""


class LLMError(MangoError):
    """Raised when the LLM service encounters an error."""


class MemoryError(MangoError):
    """Raised when the agent memory encounters an error."""
