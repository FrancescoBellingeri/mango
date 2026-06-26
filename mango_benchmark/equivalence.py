"""Canonical result-set equivalence — the locked core of the new scorer.

This module is the heart of the execution-accuracy metric (DESIGN.md §4). It is
pure, deterministic and offline: no DB, no network, no LLM. Given the agent's
executed result-set and the gold result-set it answers one binary question —
*are these the same answer?* — plus diagnostic row-level precision / recall / F1.

The rules (DESIGN.md §4), summarised:

* **Envelope unwrap** — ``{"rows":[...],"row_count":N}`` → the row list.
* **Scalar reconciliation** — a single-row / single-non-``_id``-field result and a
  bare scalar both reduce to that scalar (so the agent's ``{"count":395}`` is
  comparable to gold ``395``). Not applied to grouped results.
* **Empty vs zero** — an empty result-set ``[]`` and a scalar ``0`` are distinct.
* **``_id`` — default KEEP.** Strip only to reconcile projection asymmetry on
  *non-grouped* results (one side has ``_id``, the other was projected
  ``{_id:0}``). For ``$group``, ``_id`` is the grouping key and is always kept.
  ``$group`` is detected from the gold MQL, never from data shape.
* **Type normalization** — ObjectId→str, Decimal128/Decimal→float,
  datetime/Timestamp→canonical epoch, ``None``/``NaN`` explicit, deep on
  nested structures.
* **Numbers** — ints exact; floats within a relative tolerance (default ``1e-6``)
  plus an absolute floor (default ``1e-9``).
* **Ordering** — multiset by default; order-sensitive only when the **gold** query
  carries an explicit sort. Tie-runs (equal sort-key) compare as multisets within
  the run, ordered across runs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

# Default tolerances (DESIGN.md §4.6). Tight: agent and gold execute the same
# deterministic dataset, so the only legitimate float divergence is repr (~1e-6).
DEFAULT_REL_TOL = 1e-6
DEFAULT_ABS_TOL = 1e-9

# Aggregation stages after which document order is no longer guaranteed, so a
# preceding $sort no longer makes the question order-sensitive.
_ORDER_DESTROYING_STAGES = frozenset(
    {"$group", "$bucket", "$bucketAuto", "$sample", "$unwind", "$facet"}
)

# A scalar sentinel for an empty result-set (distinct from any value incl. 0).
_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$"
)


# ---------------------------------------------------------------------------
# MQL introspection (DESIGN.md §4.4 / §4.7) — read intent from the gold query.
# ---------------------------------------------------------------------------


def _pipeline_of(mql: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not isinstance(mql, dict):
        return None
    pipe = mql.get("pipeline")
    return pipe if isinstance(pipe, list) else None


def _group_id_is_meaningful(group_id: Any) -> bool:
    """True when a ``$group`` ``_id`` is a real grouping key (load-bearing).

    A grouping *key* groups documents by data — a field reference (``"$status"``)
    or an expression over fields. A ``$group`` with ``_id: null`` (or a literal
    constant) is a whole-collection aggregate: a single bucket with no key, whose
    output is effectively a scalar/aggregate row. Only the former makes ``_id``
    load-bearing (DESIGN.md §4.4); the latter must reconcile like a scalar so a
    one-value aggregate isn't false-failed on an aliased field name.
    """
    if group_id is None:
        return False
    if isinstance(group_id, str):
        return group_id.startswith("$")
    if isinstance(group_id, dict):
        # An expression document referencing any field path is a grouping key.
        return any(isinstance(k, str) and k.startswith("$") for k in _walk_str_values(group_id)) or bool(group_id)
    return False  # literal constant (number/bool) → single bucket


def _walk_str_values(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_str_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_str_values(v)
    elif isinstance(obj, str):
        yield obj


def is_group_query(mql: dict[str, Any] | None) -> bool:
    """True when the MQL aggregates with a ``$group`` carrying a meaningful key.

    Detected structurally from the MQL — never from the data shape, because a
    group key can itself be an ObjectId (group-by on a reference field) which a
    shape heuristic would mistake for a strippable ``_id``. A ``$group`` keyed on
    ``_id: null`` is treated as non-grouping (see :func:`_group_id_is_meaningful`).
    """
    pipe = _pipeline_of(mql)
    if pipe is None:
        return False
    for stage in pipe:
        if isinstance(stage, dict) and "$group" in stage:
            spec = stage["$group"]
            if isinstance(spec, dict) and _group_id_is_meaningful(spec.get("_id")):
                return True
    return False


def sort_spec(mql: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the *effective* sort spec of the gold query, or None.

    For find: the ``sort`` field. For aggregate: the last ``$sort`` stage whose
    ordering survives to the output (i.e. not undone by a later order-destroying
    stage such as ``$group``). None ⇒ the question is order-insensitive.
    """
    if not isinstance(mql, dict):
        return None
    pipe = _pipeline_of(mql)
    if pipe is not None:
        effective: dict[str, Any] | None = None
        for stage in pipe:
            if not isinstance(stage, dict):
                continue
            if "$sort" in stage and isinstance(stage["$sort"], dict):
                effective = stage["$sort"]
            elif any(k in _ORDER_DESTROYING_STAGES for k in stage):
                effective = None
        return effective or None
    sort = mql.get("sort")
    return sort if isinstance(sort, dict) and sort else None


# ---------------------------------------------------------------------------
# Scalar / value normalization (DESIGN.md §4.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Epoch:
    """Canonical datetime marker so a BSON datetime and its ISO string match."""

    seconds: float


def _parse_iso(s: str) -> _Epoch | None:
    if not _ISO_RE.match(s):
        return None
    txt = s.strip().replace(" ", "T")
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        return _Epoch(datetime.fromisoformat(txt).timestamp())
    except ValueError:
        return None


def _normalize(value: Any) -> Any:
    """Recursively normalise a value to a canonical, comparable form.

    ObjectId→str, Decimal128/Decimal→float, datetime/Timestamp→``_Epoch``,
    ISO-looking strings→``_Epoch``, ``NaN`` preserved as float('nan'). Dicts and
    lists are normalised element-wise (list order is preserved — array fields are
    ordered values, distinct from row ordering).
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        dt = value if isinstance(value, datetime) else datetime(value.year, value.month, value.day)
        return _Epoch(dt.timestamp())
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]

    type_name = type(value).__name__
    if type_name == "ObjectId":
        return str(value)
    if type_name == "Decimal128":
        return float(str(value))
    if type_name == "Timestamp":  # bson.Timestamp
        return _Epoch(float(getattr(value, "time", lambda: 0)() if callable(getattr(value, "time", None)) else getattr(value, "time", 0)))

    if isinstance(value, str):
        iso = _parse_iso(value)
        return iso if iso is not None else value

    return value


# ---------------------------------------------------------------------------
# Equality with numeric tolerance (DESIGN.md §4.6)
# ---------------------------------------------------------------------------


def _decimals(x: Any) -> int:
    """Number of decimal places displayed by a float (0 for ints/integers).

    ``2.487`` → 3, ``116063.95`` → 2, ``1000.0`` → 1, scientific/huge → 12 (treat
    as full precision, no rounding)."""
    if isinstance(x, bool) or isinstance(x, int):
        return 0
    s = repr(float(x))
    if "e" in s or "E" in s:
        return 12
    return len(s.split(".")[1]) if "." in s else 0


def _nums_equal(a: float | int, b: float | int, rel_tol: float, abs_tol: float) -> bool:
    # Both genuine ints → exact.
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    if math.isinf(fa) or math.isinf(fb):
        return fa == fb
    # Path 1: relative/absolute tolerance (float-repr noise on identical data).
    if abs(fa - fb) <= max(rel_tol * max(abs(fa), abs(fb)), abs_tol):
        return True
    # Path 2: round-to-gold-precision. ``b`` is always the gold value (every
    # call site passes eq(agent, gold)); when the gold is rounded (e.g. a $round
    # to 3 dp), compare at that precision so an unrounded agent value matches.
    # Skipped for integer-precision gold to avoid truncating real differences.
    gd = _decimals(b)
    if 0 < gd < 12:
        return abs(round(fa, gd) - round(fb, gd)) <= abs_tol
    return False


def values_equal(a: Any, b: Any, *, rel_tol: float = DEFAULT_REL_TOL, abs_tol: float = DEFAULT_ABS_TOL) -> bool:
    """Deep equality of two *already-normalised* values under the tolerance rules."""
    # bool is a subclass of int — keep it distinct so True != 1.
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _nums_equal(a, b, rel_tol, abs_tol)
    if isinstance(a, _Epoch) and isinstance(b, _Epoch):
        return _nums_equal(a.seconds, b.seconds, rel_tol, abs_tol)
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(values_equal(a[k], b[k], rel_tol=rel_tol, abs_tol=abs_tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(values_equal(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------
# Canonical form (DESIGN.md §4.1 - §4.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalForm:
    """A result reduced to one of three kinds. ``kind`` ∈ {scalar, empty, rows}."""

    kind: str
    scalar: Any = None
    rows: tuple[Any, ...] = ()


_ENVELOPE = ("rows", "result", "results", "data")


def _unwrap_envelope(result: Any) -> Any:
    if isinstance(result, dict):
        for key in _ENVELOPE:
            if key in result and isinstance(result[key], list):
                return result[key]
    return result


def _row_has_only_id_and_one(row: dict[str, Any]) -> bool:
    keys = [k for k in row if k != "_id"]
    return len(keys) == 1


def _scalarize_row(row: Any) -> Any:
    """Reduce a single-key row to its bare value (non-grouped results only).

    A ``distinct`` executed via ``run_mql`` wraps each value in a one-field doc
    (``{"method": "card"}``) or a one-key ``$group`` (``{"_id": "Anker"}``), while
    the gold ``distinct`` is a list of bare scalars. Unwrapping single-key rows
    makes the two shapes comparable. Applied to both sides, so a one-field
    projection stays consistent. Multi-field rows are left untouched.
    """
    if isinstance(row, dict):
        if len(row) == 1:
            # {"method": "card"} or {"_id": "Anker"} → the lone value.
            return next(iter(row.values()))
        non_id = [k for k in row if k != "_id"]
        if "_id" in row and len(non_id) == 1:
            # Whole-collection aggregate {"_id": null, "revenue": X} → X.
            return row[non_id[0]]
    return row


def canonicalize(
    result: Any,
    *,
    is_group: bool = False,
) -> CanonicalForm:
    """Reduce a raw result-set to a :class:`CanonicalForm` (DESIGN.md §4.1-§4.5).

    ``_id`` is *not* stripped here (kept by default); projection-asymmetry
    reconciliation is pairwise and lives in :func:`equivalent`. ``is_group``
    (from the gold MQL) disables scalar reconciliation so a one-row ``$group`` is
    never collapsed to a bare scalar.
    """
    result = _unwrap_envelope(result)

    # Bare scalar (incl. count results and the count-0 case).
    if result is None or isinstance(result, (int, float, bool, str)):
        return CanonicalForm("scalar", scalar=_normalize(result))

    if isinstance(result, dict):
        # A single document — treat as a one-row result, possibly reconcilable.
        if not is_group and _row_has_only_id_and_one(result):
            key = next(k for k in result if k != "_id")
            return CanonicalForm("scalar", scalar=_normalize(result[key]))
        return CanonicalForm("rows", rows=(_normalize(result),))

    if isinstance(result, list):
        if not result:
            return CanonicalForm("empty")
        norm = [_normalize(r) for r in result]
        if not is_group:
            # Unwrap single-key rows (distinct shape) so scalar lists compare.
            norm = [_scalarize_row(r) for r in norm]
            # Scalar reconciliation: a lone scalar row is a scalar answer (count,
            # single distinct value) — distinct from an empty set.
            if len(norm) == 1 and not isinstance(norm[0], (dict, list)):
                return CanonicalForm("scalar", scalar=norm[0])
        return CanonicalForm("rows", rows=tuple(norm))

    return CanonicalForm("scalar", scalar=_normalize(result))


# ---------------------------------------------------------------------------
# Row-set comparison: matching, ordering, precision/recall (§4.7, §4 diagnostics)
# ---------------------------------------------------------------------------


def _max_matching(a_rows: list[Any], b_rows: list[Any], eq) -> int:
    """Size of a maximum bipartite matching between two row lists under ``eq``.

    Tolerance equality is not transitive, so a greedy pass can undercount; this
    augmenting-path matcher (Kuhn's algorithm) gives the true multiset overlap.
    """
    adj: list[list[int]] = [[j for j, b in enumerate(b_rows) if eq(a, b)] for a in a_rows]
    match_b = [-1] * len(b_rows)

    def _augment(u: int, seen: list[bool]) -> bool:
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                if match_b[v] == -1 or _augment(match_b[v], seen):
                    match_b[v] = u
                    return True
        return False

    matched = 0
    for u in range(len(a_rows)):
        if _augment(u, [False] * len(b_rows)):
            matched += 1
    return matched


def _group_row_eq(a: Any, b: Any, eq) -> bool:
    """Equality for grouped rows (DESIGN.md §4.4): two rows match iff their full
    multiset of *values* matches, ignoring every field name (including ``_id``).

    The group-key value participates in the bag, so it still anchors each row —
    cross-row value-swaps and different groupings are caught (the keys differ) —
    while tolerating both metric aliasing (gold ``total`` vs agent ``revenue``)
    **and** group-key naming (gold raw ``_id`` vs agent-aliased ``network``). The
    accepted tradeoffs: differing field *counts* are a mismatch (a genuinely
    different projection), and two values swapped between fields *within one row*
    are not distinguished.
    """
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return eq(a, b)
    a_vals = list(a.values())
    b_vals = list(b.values())
    if len(a_vals) != len(b_vals):
        return False
    return _max_matching(a_vals, b_vals, eq) == len(a_vals)


def _strip_id(rows: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "_id"} if isinstance(r, dict) else r for r in rows]


def _reconcile_ids(
    a_rows: tuple[Any, ...], b_rows: tuple[Any, ...], is_group: bool
) -> tuple[list[Any], list[Any]]:
    """Apply the ``_id`` rule (DESIGN.md §4.4) and return comparable row lists.

    Group results keep ``_id`` (load-bearing). Non-grouped results keep ``_id``
    only when *both* sides carry it on every row; otherwise (projection
    asymmetry) it is stripped from both.
    """
    if is_group:
        return list(a_rows), list(b_rows)

    def _all_have_id(rows: tuple[Any, ...]) -> bool:
        return bool(rows) and all(isinstance(r, dict) and "_id" in r for r in rows)

    if _all_have_id(a_rows) and _all_have_id(b_rows):
        return list(a_rows), list(b_rows)
    return _strip_id(a_rows), _strip_id(b_rows)


def _sort_key_tuple(row: Any, keys: list[str]) -> tuple[Any, ...] | None:
    if not isinstance(row, dict):
        return None
    out: list[Any] = []
    for k in keys:
        if k not in row:
            return None
        out.append(row[k])
    return tuple(out)


def _runs_by_sort_key(rows: list[Any], keys: list[str], eq) -> list[list[Any]] | None:
    """Partition rows into maximal runs of equal sort-key value, or None if a key
    is missing from any row (sort key unrecoverable → caller falls back)."""
    runs: list[list[Any]] = []
    prev: tuple[Any, ...] | None = None
    for row in rows:
        kt = _sort_key_tuple(row, keys)
        if kt is None:
            return None
        if prev is not None and all(eq(x, y) for x, y in zip(kt, prev)):
            runs[-1].append(row)
        else:
            runs.append([row])
            prev = kt
    return runs


@dataclass
class ComparisonResult:
    """Outcome of comparing one agent result-set against gold (DESIGN.md §4/§5)."""

    equivalent: bool
    precision: float
    recall: float
    f1: float
    order_sensitive: bool
    is_group: bool
    note: str = ""
    agent_kind: str = ""
    gold_kind: str = ""


def _prf(matched: int, n_agent: int, n_gold: int) -> tuple[float, float, float]:
    precision = matched / n_agent if n_agent else (1.0 if n_gold == 0 else 0.0)
    recall = matched / n_gold if n_gold else (1.0 if n_agent == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def compare(
    agent_result: Any,
    gold_result: Any,
    *,
    gold_mql: dict[str, Any] | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> ComparisonResult:
    """Compare an agent result-set against gold under canonical equivalence.

    ``gold_mql`` (the structured gold query) is the source of truth for
    ``is_group`` (``_id`` handling) and ``order_sensitive`` (sort handling). When
    absent or body-only (legacy CSVs) the comparison falls back to non-grouped,
    order-insensitive — the safe, debuggable direction.
    """
    is_group = is_group_query(gold_mql)
    spec = sort_spec(gold_mql)
    order_sensitive = spec is not None

    a = canonicalize(agent_result, is_group=is_group)
    g = canonicalize(gold_result, is_group=is_group)

    def eq(x: Any, y: Any) -> bool:
        return values_equal(x, y, rel_tol=rel_tol, abs_tol=abs_tol)

    # --- scalar / empty kinds ---
    if a.kind != "rows" or g.kind != "rows":
        if a.kind == "scalar" and g.kind == "scalar":
            ok = eq(a.scalar, g.scalar)
        elif a.kind == "empty" and g.kind == "empty":
            ok = True
        else:
            ok = False  # empty vs scalar, scalar vs rows, etc. — distinct (§4.3)
        return ComparisonResult(
            equivalent=ok, precision=1.0 if ok else 0.0, recall=1.0 if ok else 0.0,
            f1=1.0 if ok else 0.0, order_sensitive=order_sensitive, is_group=is_group,
            agent_kind=a.kind, gold_kind=g.kind,
        )

    # --- both are row-sets ---
    a_rows, g_rows = _reconcile_ids(a.rows, g.rows, is_group)

    # Grouped rows match by (strict _id key + value-bag); plain rows match strictly.
    def row_eq(x: Any, y: Any) -> bool:
        return _group_row_eq(x, y, eq) if is_group else eq(x, y)

    matched = _max_matching(a_rows, g_rows, row_eq)
    precision, recall, f1 = _prf(matched, len(a_rows), len(g_rows))

    note = ""
    if not order_sensitive:
        equiv = len(a_rows) == len(g_rows) == matched
    else:
        keys = list(spec.keys())
        equiv, note = _ordered_equivalent(a_rows, g_rows, keys, eq, row_eq)

    return ComparisonResult(
        equivalent=equiv, precision=precision, recall=recall, f1=f1,
        order_sensitive=order_sensitive, is_group=is_group, note=note,
        agent_kind=a.kind, gold_kind=g.kind,
    )


def _ordered_equivalent(a_rows: list[Any], g_rows: list[Any], keys: list[str], eq, row_eq) -> tuple[bool, str]:
    """Order-sensitive comparison with tie-run handling (DESIGN.md §4.7).

    ``eq`` compares sort-key values; ``row_eq`` compares whole rows (value-bag for
    grouped results, strict otherwise).
    """
    if len(a_rows) != len(g_rows):
        return False, "row-count mismatch"

    a_runs = _runs_by_sort_key(a_rows, keys, eq)
    g_runs = _runs_by_sort_key(g_rows, keys, eq)

    # Sort key unrecoverable from the rows → strict ordered compare, flagged.
    if a_runs is None or g_runs is None:
        ok = all(row_eq(x, y) for x, y in zip(a_rows, g_rows))
        return ok, "sort-key not in rows; strict ordered fallback"

    if len(a_runs) != len(g_runs):
        return False, "tie-run structure mismatch"

    for ar, gr in zip(a_runs, g_runs):
        # Across runs: ordered (enforced by run alignment). Within run: multiset.
        if len(ar) != len(gr) or _max_matching(ar, gr, row_eq) != len(gr):
            return False, "within-tie content mismatch"
    return True, ""


def equivalent(
    agent_result: Any,
    gold_result: Any,
    *,
    gold_mql: dict[str, Any] | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> bool:
    """Binary PASS gate: are the two result-sets equivalent? (DESIGN.md §4)."""
    return compare(
        agent_result, gold_result, gold_mql=gold_mql, rel_tol=rel_tol, abs_tol=abs_tol
    ).equivalent
