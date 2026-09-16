"""Shared security policy for MQL execution.

Defines the read-only policy that every query must satisfy, and helpers to
enforce it at any depth of a filter document, projection, sort spec or
aggregation pipeline.

Three independent rings:

1. **Forbidden operators** (denylist): stages/operators that write, run
   server-side JavaScript, open unbounded cursors or leak server metadata.
   Scanned recursively in *every* expression-bearing argument of a query —
   not only ``filter`` and ``pipeline``: since MongoDB 4.4 a ``find``
   projection accepts aggregation expressions, so ``$function`` hidden in a
   projection would otherwise slip through.
2. **Allowed stages** (allowlist): the only aggregation stages a pipeline (or
   a sub-pipeline in ``$lookup``/``$unionWith``/``$facet``) may contain.
3. **Collection access**: which collections a query may touch, including the
   ones referenced by ``$lookup``/``$graphLookup``/``$unionWith``.

This lives in ``core`` (no driver imports) so that both the pre-execution
validator (``mango.tools.validator``) and the backend runner
(``mango.integrations.mongodb``) can enforce the same policy independently —
the runner check is defence-in-depth and is NOT disabled when validation is
turned off.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mango.core.types import ValidationError

# Operators/stages that break the read-only guarantee or open a code-execution
# / resource-exhaustion vector. Blocked as hard errors, at any depth.
FORBIDDEN_OPERATORS: frozenset[str] = frozenset({
    # --- Writes: escape the read-only allowlist entirely ---
    "$out",            # overwrites/replaces a whole collection
    "$merge",          # writes (and can target another database)
    # --- Server-side JavaScript: arbitrary code execution + DoS ---
    "$where",
    "$function",
    "$accumulator",
    # --- Unbounded / streaming cursor: hangs the worker ---
    "$changeStream",
    # --- Administrative / metadata leakage; no place in a NL query agent ---
    "$collStats",
    "$indexStats",
    "$planCacheStats",
    "$listSessions",
    "$listLocalSessions",
    "$currentOp",
    "$listSearchIndexes",
    "$querySettings",
    "$shardedDataDistribution",
})

# The only aggregation stages a pipeline may contain. Anything else — unknown,
# misspelled, or a stage deliberately left out (see FORBIDDEN_OPERATORS) — is
# rejected before it reaches the server.
ALLOWED_STAGES: frozenset[str] = frozenset({
    "$addFields", "$bucket", "$bucketAuto", "$count", "$densify", "$documents",
    "$facet", "$fill", "$geoNear", "$graphLookup", "$group", "$limit",
    "$lookup", "$match", "$project", "$redact", "$replaceRoot", "$replaceWith",
    "$sample", "$search", "$searchMeta", "$set", "$setWindowFields", "$skip",
    "$sort", "$sortByCount", "$unionWith", "$unset", "$unwind", "$vectorSearch",
})

# Collections that must never be exposed, regardless of any allowlist.
_SYSTEM_PREFIX = "system."


# ---------------------------------------------------------------------------
# Ring 1 — forbidden operators
# ---------------------------------------------------------------------------


def find_forbidden_operators(
    filter_doc: dict[str, Any] | None,
    pipeline: list[dict[str, Any]] | None,
    projection: dict[str, Any] | None = None,
    sort: dict[str, Any] | None = None,
) -> list[str]:
    """Return the sorted, de-duplicated forbidden operators present.

    Scans the filter document, every pipeline stage (keys *and* bodies), the
    projection and the sort spec recursively, so operators hidden inside
    ``$expr``, ``$lookup`` sub-pipelines, ``$facet`` branches or a projection
    expression are still caught.
    """
    found: set[str] = set()
    for doc in (filter_doc, pipeline, projection, sort):
        if doc:
            _walk(doc, found)
    return sorted(found)


def _walk(obj: Any, found: set[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key in FORBIDDEN_OPERATORS:
                found.add(key)
            _walk(value, found)
    elif isinstance(obj, list):
        for item in obj:
            _walk(item, found)


# ---------------------------------------------------------------------------
# Ring 2 — stage allowlist (top level and nested sub-pipelines)
# ---------------------------------------------------------------------------


def find_disallowed_stages(pipeline: list[dict[str, Any]] | None) -> list[str]:
    """Return the sorted stage names not in :data:`ALLOWED_STAGES`.

    Walks the top-level pipeline and every nested sub-pipeline
    (``$lookup.pipeline``, ``$unionWith.pipeline``, each ``$facet`` branch).
    Malformed stages (non-dict, or not exactly one key) are skipped here: the
    validator reports them with a helpful message and MongoDB rejects them
    anyway — they cannot execute anything.
    """
    found: set[str] = set()
    _walk_stages(pipeline, found)
    return sorted(found)


def _walk_stages(pipeline: Any, found: set[str]) -> None:
    if not isinstance(pipeline, list):
        return
    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1:
            continue
        op, body = next(iter(stage.items()))
        if op not in ALLOWED_STAGES:
            found.add(str(op))
            continue
        if op in ("$lookup", "$unionWith") and isinstance(body, dict):
            _walk_stages(body.get("pipeline"), found)
        elif op == "$facet" and isinstance(body, dict):
            for branch in body.values():
                _walk_stages(branch, found)


# ---------------------------------------------------------------------------
# Ring 3 — collection access
# ---------------------------------------------------------------------------


def referenced_collections(pipeline: list[dict[str, Any]] | None) -> set[str]:
    """Return every collection name a pipeline reads besides its target.

    Covers ``$lookup.from``, ``$graphLookup.from`` and ``$unionWith`` (string
    or ``{coll: ...}`` form), recursing into nested sub-pipelines.
    """
    found: set[str] = set()
    _walk_refs(pipeline, found)
    return found


def _walk_refs(pipeline: Any, found: set[str]) -> None:
    if not isinstance(pipeline, list):
        return
    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1:
            continue
        op, body = next(iter(stage.items()))
        if op in ("$lookup", "$graphLookup") and isinstance(body, dict):
            src = body.get("from")
            if isinstance(src, str):
                found.add(src)
            _walk_refs(body.get("pipeline"), found)
        elif op == "$unionWith":
            if isinstance(body, str):
                found.add(body)
            elif isinstance(body, dict):
                src = body.get("coll")
                if isinstance(src, str):
                    found.add(src)
                _walk_refs(body.get("pipeline"), found)
        elif op == "$facet" and isinstance(body, dict):
            for branch in body.values():
                _walk_refs(branch, found)


class CollectionPolicy:
    """Decides which collections a runner may expose and query.

    Args:
        allowed: If given, *only* these collections are accessible.
        denied: Collections that are never accessible (wins over ``allowed``).

    ``system.*`` collections are always denied.
    """

    def __init__(
        self,
        allowed: Iterable[str] | None = None,
        denied: Iterable[str] | None = None,
    ) -> None:
        self._allowed: frozenset[str] | None = (
            frozenset(allowed) if allowed is not None else None
        )
        self._denied: frozenset[str] = frozenset(denied or ())

    @property
    def restricted(self) -> bool:
        """True when an explicit allow/deny list is configured."""
        return self._allowed is not None or bool(self._denied)

    def is_allowed(self, name: str) -> bool:
        if not isinstance(name, str) or name.startswith(_SYSTEM_PREFIX):
            return False
        if name in self._denied:
            return False
        if self._allowed is not None and name not in self._allowed:
            return False
        return True

    def filter(self, names: Iterable[str]) -> list[str]:
        return [n for n in names if self.is_allowed(n)]

    def combined(self, other: "CollectionPolicy | None") -> "CollectionPolicy":
        """Return a policy that allows a name only if *both* policies allow it."""
        if other is None:
            return self
        if self._allowed is None:
            allowed = other._allowed
        elif other._allowed is None:
            allowed = self._allowed
        else:
            allowed = self._allowed & other._allowed
        return CollectionPolicy(allowed=allowed, denied=self._denied | other._denied)

    def check(self, name: str) -> None:
        """Raise :class:`ValidationError` when *name* is not accessible."""
        if not self.is_allowed(name):
            raise ValidationError(
                f"Collection '{name}' is not accessible to this agent."
            )


# ---------------------------------------------------------------------------
# One-shot policy check used by the runner (defence-in-depth)
# ---------------------------------------------------------------------------


def enforce_read_only_policy(
    *,
    collection: str,
    filter_doc: dict[str, Any] | None,
    pipeline: list[dict[str, Any]] | None,
    projection: dict[str, Any] | None = None,
    sort: dict[str, Any] | None = None,
    collection_policy: CollectionPolicy | None = None,
) -> None:
    """Raise :class:`ValidationError` when the query violates the policy.

    Applied by the backend runner on every execution, independently of the
    pre-execution validator, so the read-only guarantee cannot be switched
    off by constructing a tool with ``validate=False``.
    """
    forbidden = find_forbidden_operators(filter_doc, pipeline, projection, sort)
    if forbidden:
        raise ValidationError(
            f"Forbidden operator(s) {forbidden} are not permitted "
            "(read-only: no $out/$merge, no server-side JavaScript, "
            "no change streams or administrative stages)."
        )

    disallowed = find_disallowed_stages(pipeline)
    if disallowed:
        raise ValidationError(
            f"Aggregation stage(s) {disallowed} are not permitted. "
            f"Allowed stages: {sorted(ALLOWED_STAGES)}."
        )

    policy = collection_policy or CollectionPolicy()
    policy.check(collection)
    for ref in sorted(referenced_collections(pipeline)):
        policy.check(ref)
