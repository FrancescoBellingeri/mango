"""Shared case model, gold execution and serialization for bench_datasets.

Design contract (DATASET_DESIGN.md §4):

* ``GoldMQL`` mirrors ``mango.core.types.QueryRequest`` — the same eight keys
  ``dataset.py`` already parses (``_GOLD_MQL_KEYS``).
* ``order_sensitive`` / ``grouped`` are **derived, never authored**: they come
  from :func:`mango_benchmark.equivalence.sort_spec` /
  :func:`mango_benchmark.equivalence.is_group_query` — the exact code path the
  scorecard uses — and are stored only as caches.
* ``expected.result`` is produced exclusively by executing the gold against
  the seeded DB (:func:`execute_gold`); it is never hand-typed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from bson import Decimal128, ObjectId

from mango_benchmark.equivalence import is_group_query, sort_spec

# ---------------------------------------------------------------------------
# Registries (taxonomy §2)
# ---------------------------------------------------------------------------

DATABASES = ("bench_logistics", "bench_workforce", "bench_meters")

# category code -> block. AMB is the only non-gating block (§2.3).
CATEGORIES: dict[str, str] = {
    "CAP-FIND": "capability",
    "CAP-FILTER": "capability",
    "CAP-PROJ": "capability",
    "CAP-SORT": "capability",
    "CAP-PAGE": "capability",
    "CAP-COUNT": "capability",
    "CAP-GROUP": "capability",
    "CAP-MULTI": "capability",
    "CAP-TIME": "capability",
    "CAP-NEST": "capability",
    "EDGE-NUM": "edge",
    "EDGE-EMPTY": "edge",
    "EDGE-SCALE": "edge",
    "SEC-TRAP": "behavioral",
    "OOS": "behavioral",
    "AMB": "ambiguity",
    "NLR": "robustness",
    "STRETCH": "robustness",
}

# §2.6 — core categories must span ≥2 databases, no database >60% of cases.
CORE_CATEGORIES = frozenset(
    {"CAP-FILTER", "CAP-SORT", "CAP-GROUP", "CAP-MULTI", "CAP-TIME", "CAP-NEST"}
)
DOMAIN_SPREAD_MAX_SHARE = 0.60

DIFFICULTIES = ("easy", "medium", "hard")
BEHAVIORS = ("answer", "refuse", "safe_subset", "any_of")
GENERATIONS = ("hand", "template", "template+llm")

_ID_RE = re.compile(r"^(LOG|WF|MET|XD)-[A-Z]+(-[A-Z]+)?-\d{3}$")

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# BSON-safe JSON serialization (same semantics as seed_hard.ground_truth,
# plus Decimal128 which the meters domain uses)
# ---------------------------------------------------------------------------


def serialize(obj: Any) -> Any:
    """Convert BSON types to JSON-safe equivalents (recursive)."""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal128):
        return float(obj.to_decimal())
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    return obj


def result_sha256(result: Any) -> str:
    return hashlib.sha256(
        json.dumps(result, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Gold MQL
# ---------------------------------------------------------------------------


@dataclass
class GoldMQL:
    """Structured gold query — mirrors ``dataset.py::_GOLD_MQL_KEYS``."""

    operation: str  # find | aggregate | count | distinct
    collection: str
    filter: dict[str, Any] | None = None
    pipeline: list[dict[str, Any]] | None = None
    projection: dict[str, Any] | None = None
    sort: dict[str, int] | None = None
    limit: int | None = None
    distinct_field: str | None = None

    def __post_init__(self) -> None:
        if self.operation not in ("find", "aggregate", "count", "distinct"):
            raise ValueError(f"unknown operation {self.operation!r}")
        if self.operation == "aggregate" and self.pipeline is None:
            raise ValueError("aggregate gold needs a pipeline")
        if self.operation != "aggregate" and self.pipeline is not None:
            raise ValueError(f"{self.operation} gold must not carry a pipeline")
        if self.operation == "distinct" and not self.distinct_field:
            raise ValueError("distinct gold needs distinct_field")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "collection": self.collection,
            "filter": serialize(self.filter),
            "pipeline": serialize(self.pipeline),
            "projection": serialize(self.projection),
            "sort": serialize(self.sort),
            "limit": self.limit,
            "distinct_field": self.distinct_field,
        }


def derive_order_sensitive(gold: GoldMQL) -> bool:
    return sort_spec(gold.to_dict()) is not None


def derive_grouped(gold: GoldMQL) -> bool:
    return is_group_query(gold.to_dict())


def execute_gold(db: Any, gold: GoldMQL) -> Any:
    """Execute a gold query live. Same execution semantics as
    ``seed_hard.ground_truth._run_question``: count -> int scalar, distinct ->
    sorted scalar list, find/aggregate -> serialized row list. No row cap."""
    col = db[gold.collection]

    if gold.operation == "count":
        return col.count_documents(gold.filter or {})

    if gold.operation == "distinct":
        values = col.distinct(gold.distinct_field, gold.filter or {})
        return sorted(
            [serialize(v) for v in values], key=lambda x: (str(type(x)), str(x))
        )

    if gold.operation == "find":
        cursor = col.find(gold.filter or {}, gold.projection or None)
        if gold.sort:
            cursor = cursor.sort(list(gold.sort.items()))
        if gold.limit:
            cursor = cursor.limit(gold.limit)
        return [serialize(doc) for doc in cursor]

    if gold.operation == "aggregate":
        return [serialize(doc) for doc in col.aggregate(gold.pipeline, allowDiskUse=True)]

    raise ValueError(f"unknown operation {gold.operation!r}")


# ---------------------------------------------------------------------------
# Bench case
# ---------------------------------------------------------------------------


@dataclass
class BenchCase:
    """One benchmark case (DATASET_DESIGN.md §4).

    ``wrong_mql`` is build-side only (never exported to the runner CSV): the
    *documented plausible-wrong query* that per-category validation executes
    to prove the case has teeth — the naive `$elemMatch` rewrite for CAP-NEST,
    the superset average for EDGE-NUM, the one-predicate-relaxed filter for
    EDGE-EMPTY (§6.2).
    """

    id: str
    database: str
    category: str
    subcategory: str
    difficulty: str
    nl_question: str
    gold: GoldMQL | None = None
    gold_alternatives: list[GoldMQL] = field(default_factory=list)
    expected_behavior: str = "answer"
    tags: list[str] = field(default_factory=list)
    nl_variant_of: str | None = None
    sibling_of: str | None = None
    safety_trap: bool = False
    tolerance_override: dict[str, float] | None = None
    generation: str = "hand"
    swap_waiver: bool = False
    wrong_mql: GoldMQL | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.id):
            raise ValueError(f"{self.id}: id does not match DOM-CAT-NNN convention")
        if self.database not in DATABASES:
            raise ValueError(f"{self.id}: unknown database {self.database!r}")
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"{self.id}: unknown difficulty {self.difficulty!r}")
        if self.expected_behavior not in BEHAVIORS:
            raise ValueError(f"{self.id}: unknown behavior {self.expected_behavior!r}")
        if self.generation not in GENERATIONS:
            raise ValueError(f"{self.id}: unknown generation {self.generation!r}")
        if self.expected_behavior == "answer" and self.gold is None and self.nl_variant_of is None:
            raise ValueError(f"{self.id}: answer case needs a gold (or a base via nl_variant_of)")
        if self.expected_behavior == "refuse" and self.gold is not None:
            raise ValueError(f"{self.id}: refuse case must not carry a gold")
        if self.expected_behavior == "any_of" and not self.gold_alternatives:
            raise ValueError(f"{self.id}: any_of case needs gold_alternatives")
        if self.category == "EDGE-EMPTY" and self.sibling_of is None:
            raise ValueError(f"{self.id}: EDGE-EMPTY requires sibling_of (§2.2)")

    # -- export ------------------------------------------------------------

    def to_record(
        self, gold: GoldMQL | None, expected_result: Any, alt_results: list[Any]
    ) -> dict[str, Any]:
        """The §4 JSONL record. ``gold`` is the *effective* gold — the base
        case's for NLR variants (resolved by build.py, so variant and base
        cannot drift apart). ``expected_result`` comes from execute_gold."""
        return {
            "id": self.id,
            "schema_version": SCHEMA_VERSION,
            "database": self.database,
            "nl_question": self.nl_question,
            "nl_variant_of": self.nl_variant_of,
            "sibling_of": self.sibling_of,
            "gold_mql": gold.to_dict() if gold else None,
            "gold_alternatives": [g.to_dict() for g in self.gold_alternatives],
            "alternative_results": alt_results,
            "expected": None
            if gold is None
            else {
                "result": expected_result,
                "row_count": len(expected_result) if isinstance(expected_result, list) else 1,
                "result_sha256": result_sha256(expected_result),
            },
            "expected_behavior": self.expected_behavior,
            "metadata": {
                "category": self.category,
                "subcategory": self.subcategory,
                "difficulty": self.difficulty,
                "tags": self.tags,
                "safety_trap": self.safety_trap,
                "order_sensitive": derive_order_sensitive(gold) if gold else None,
                "grouped": derive_grouped(gold) if gold else None,
                "tolerance_override": self.tolerance_override,
                "verified": "author-trusted",
                "generation": self.generation,
                "swap_waiver": self.swap_waiver,
                "contamination_max_sim": None,  # filled by contamination.py
                "notes": self.notes,
            },
        }

    def to_csv_row(self, gold: GoldMQL | None, expected_result: Any) -> dict[str, str]:
        """Runner-compatible Braintrust row (same shape seed_hard emits)."""
        gold_json = json.dumps(gold.to_dict(), ensure_ascii=False) if gold else ""
        return {
            "input": json.dumps(
                {"nlQuery": self.nl_question, "databaseName": self.database},
                ensure_ascii=False,
            ),
            "expected": json.dumps(
                {"dbQuery": gold_json, "result": expected_result}, ensure_ascii=False
            ),
            "tags": "|".join(
                [self.id, self.category, self.subcategory, self.difficulty, *self.tags]
            ),
        }


def check_unique_ids(cases: list[BenchCase]) -> None:
    seen: dict[str, str] = {}
    for c in cases:
        if c.id in seen:
            raise ValueError(f"duplicate case id {c.id}")
        seen[c.id] = c.database


def resolve_base(case: BenchCase, by_id: dict[str, BenchCase]) -> BenchCase:
    """Resolve an NLR variant to its base (variants carry no own gold)."""
    if case.nl_variant_of is None:
        return case
    base = by_id.get(case.nl_variant_of)
    if base is None:
        raise ValueError(f"{case.id}: nl_variant_of {case.nl_variant_of!r} not found")
    if base.nl_variant_of is not None:
        raise ValueError(f"{case.id}: variant chains to another variant {base.id}")
    return base
