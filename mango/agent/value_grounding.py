"""Proactive value-grounding: surface known categorical value encodings up front.

Without this, the agent only discovers that a value is stored as "ACTIVE" instead
of "active" if the LLM decides to call inspect_field first — an optional, easy to
skip step. This module builds an index of distinct string values observed during
schema introspection (FieldInfo.sample_values, collected for free during the
existing sampling pass — no extra DB round trip) and matches question tokens
against it, so common drift is resolved before the LLM writes a filter.

Scope: lexical matching only — exact after normalisation (case/punctuation), or
fuzzy (difflib) for near-miss spelling/typos. It does NOT catch cross-language or
synonym drift (e.g. "sicilia" vs "Sicily"); inspect_field remains the tool for
that, and for fields with more distinct values than _CATEGORICAL_MAX_DISTINCT.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict

from mango.core.types import FieldInfo, SchemaInfo

_MIN_TOKEN_LEN = 3
_FUZZY_CUTOFF = 0.85
_MAX_HINTS = 5

# normalized_value -> [(collection, field_path, original_value), ...]
ValueIndex = dict[str, list[tuple[str, str, str]]]


def _normalize(s: str) -> str:
    """Lowercase and strip non-alphanumeric characters for lookup purposes."""
    return re.sub(r"[^\w]", "", s.lower())


def build_value_index(schema: dict[str, SchemaInfo]) -> ValueIndex:
    """Build a normalized-value lookup index from introspected schema.

    Values that normalize to the same key (e.g. "active" and "ACTIVE") land
    under one entry, which is exactly the signal used later to detect drift.
    """
    index: ValueIndex = defaultdict(list)
    for collection, info in schema.items():
        _index_fields(info.fields, collection, index)
    return dict(index)


def _index_fields(fields: list[FieldInfo], collection: str, index: ValueIndex) -> None:
    for f in fields:
        if f.sample_values:
            for value in f.sample_values:
                index.setdefault(_normalize(value), []).append((collection, f.path, value))
        if f.sub_fields:
            _index_fields(f.sub_fields, collection, index)


def find_value_hints(question: str, index: ValueIndex, max_hints: int = _MAX_HINTS) -> list[str]:
    """Return plain-language hints for question tokens matching a known value.

    A hint is only emitted when there is something to correct: either the field
    stores a single variant whose casing/spelling differs from what the question
    typed, or the field stores multiple variants of the same concept (real
    drift) — in which case the hint tells the model to match all of them.
    Tokens typed exactly as the (only) stored variant produce no hint.
    """
    if not index:
        return []

    raw_words = re.findall(r"[A-Za-z0-9]+", question)
    raw_bigrams = [f"{a} {b}" for a, b in zip(raw_words, raw_words[1:])]
    candidates = [w for w in raw_words if len(w) >= _MIN_TOKEN_LEN] + raw_bigrams

    index_keys = list(index.keys())
    hints: list[str] = []
    seen_fields: set[tuple[str, str]] = set()

    for token in candidates:
        norm = _normalize(token)
        if not norm:
            continue

        matched_key: str | None = norm if norm in index else None
        if matched_key is None:
            close = difflib.get_close_matches(norm, index_keys, n=1, cutoff=_FUZZY_CUTOFF)
            matched_key = close[0] if close else None
        if matched_key is None:
            continue

        by_field: dict[tuple[str, str], list[str]] = defaultdict(list)
        for collection, path, original in index[matched_key]:
            by_field[(collection, path)].append(original)

        for (collection, path), variants in by_field.items():
            field_key = (collection, path)
            if field_key in seen_fields:
                continue
            if len(variants) == 1 and variants[0] == token:
                continue  # already typed exactly as stored — nothing to correct
            seen_fields.add(field_key)

            if len(variants) == 1:
                hints.append(
                    f"- \"{token}\" -> in `{collection}.{path}` it is stored as {variants[0]!r}"
                )
            else:
                shown = ", ".join(repr(v) for v in sorted(variants)[:5])
                hints.append(
                    f"- \"{token}\" -> in `{collection}.{path}` multiple forms exist: {shown} "
                    "— match all of them (e.g. $in or a case-insensitive $regex)"
                )
            if len(hints) >= max_hints:
                return hints

    return hints
