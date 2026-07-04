"""Shared security policy for MQL execution.

Defines the set of aggregation stages / operators that must never run under
Mango's read-only guarantee, and a recursive scanner that finds them at any
depth inside a filter document or aggregation pipeline.

This lives in ``core`` (no driver imports) so that both the pre-execution
validator (``mango.tools.validator``) and the backend runner
(``mango.integrations.mongodb``) can enforce the same policy independently —
the runner check is defence-in-depth and is NOT disabled when validation is
turned off.
"""

from __future__ import annotations

from typing import Any

# Operators/stages that break the read-only guarantee or open a code-execution
# / resource-exhaustion vector. Blocked as hard errors, at any pipeline depth.
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
})


def find_forbidden_operators(
    filter_doc: dict[str, Any] | None,
    pipeline: list[dict[str, Any]] | None,
) -> list[str]:
    """Return the sorted, de-duplicated forbidden operators present.

    Scans both the filter document and every pipeline stage (keys *and*
    bodies) recursively, so operators hidden inside ``$expr``, ``$lookup``
    sub-pipelines, ``$facet`` branches, etc. are still caught.
    """
    found: set[str] = set()
    if filter_doc:
        _walk(filter_doc, found)
    if pipeline:
        _walk(pipeline, found)
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
