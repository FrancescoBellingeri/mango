"""Compute ground truth answers for all benchmark questions and write the CSV.

Requires:
  1. A running MongoDB with the mango_ecommerce database already populated
     (run generate.py first).
  2. The CSV is written in the Braintrust format consumed by dataset.py:
       input:    {"nlQuery": "...", "databaseName": "mango_ecommerce"}
       expected: {"dbQuery": <serialized_query>, "result": <computed_result>}
       tags:     "easy|count"

Usage:
    python -m mango_benchmark.seed.ground_truth --uri mongodb://localhost:27017
    python -m mango_benchmark.seed.ground_truth --uri mongodb://localhost:27017 \\
        --out mango_benchmark/mango_ecommerce_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import os

from bson import ObjectId
from pymongo import MongoClient

from mango_benchmark.seed.questions import QUESTIONS, LITE_QUESTIONS, BenchmarkQuestion
from mango_benchmark.seed.schema import DB_NAME

from dotenv import load_dotenv

load_dotenv()

DEFAULT_OUT = Path(__file__).parent.parent / "mango_ecommerce_benchmark.csv"
DEFAULT_LITE_OUT = Path(__file__).parent.parent / "mango_ecommerce_benchmark_lite.csv"


# ---------------------------------------------------------------------------
# BSON / datetime JSON serialisation
# ---------------------------------------------------------------------------


class _BSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def _serialize(obj: Any) -> Any:
    """Convert BSON types to JSON-safe equivalents (recursive)."""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _query_to_json(q: BenchmarkQuestion) -> str:
    """Serialize the MQL query to a compact JSON string for the CSV."""
    return json.dumps(_serialize(q.query), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def _key_tuple(row: dict, keys: list[str]) -> tuple:
    return tuple(row.get(k) for k in keys)


def _boundary_tie(rows: list[dict], limit: int, keys: list[str]) -> bool:
    """True when a ``$limit`` cuts through a group of equal-sort-key rows.

    The row just inside the cut (index ``limit-1``) ties, on the sort keys, with
    the row just outside (index ``limit``). When that happens, "top-N" has more
    than one valid answer and re-generating the gold yields a different set — so
    the query needs a unique tie-breaker to be well-posed. A clean cut (no tie at
    the boundary) is left untouched, preserving the scorer's tie-aware comparison.
    """
    if len(rows) <= limit:
        return False
    return _key_tuple(rows[limit - 1], keys) == _key_tuple(rows[limit], keys)


def _agg_tiebreak(col: Any, pipeline: list[dict]) -> list[dict]:
    """Append ``_id`` as a deterministic final sort key to a sort→limit pipeline,
    but only when the limit actually cuts a tie group. Mutates the ``$sort`` stage
    in place so the serialised gold query reflects the tie-breaker. No-op otherwise.
    """
    sort_idx = next((i for i, s in enumerate(pipeline) if isinstance(s, dict) and "$sort" in s), None)
    limit_idx = next((i for i, s in enumerate(pipeline) if isinstance(s, dict) and "$limit" in s), None)
    if sort_idx is None or limit_idx is None or limit_idx < sort_idx:
        return pipeline
    sort_spec = pipeline[sort_idx]["$sort"]
    if "_id" in sort_spec:
        return pipeline
    limit_n = pipeline[limit_idx]["$limit"]
    # Probe the rows around the boundary BEFORE any later $project, where the sort
    # keys are still present. Same prefix + one extra row past the limit.
    probe = list(col.aggregate(pipeline[:limit_idx] + [{"$limit": limit_n + 1}], allowDiskUse=True))
    if _boundary_tie(probe, limit_n, list(sort_spec.keys())):
        pipeline[sort_idx]["$sort"] = {**sort_spec, "_id": 1}
    return pipeline


def _find_sort_tiebreak(col: Any, q: BenchmarkQuestion) -> dict | None:
    """Return ``q.sort`` plus ``_id`` when a sort+limit find cuts a tie at the
    boundary; otherwise ``q.sort`` unchanged."""
    if not (q.sort and q.limit) or "_id" in q.sort:
        return q.sort
    # Probe full documents (no projection) so the sort keys are always present.
    probe = list(col.find(q.query).sort(list(q.sort.items())).limit(q.limit + 1))
    if _boundary_tie(probe, q.limit, list(q.sort.keys())):
        return {**q.sort, "_id": 1}
    return q.sort


def _run_question(db: Any, q: BenchmarkQuestion) -> Any:
    """Execute the question against MongoDB and return a JSON-safe result."""
    col = db[q.collection]

    if q.operation == "count":
        result = col.count_documents(q.query)
        return result  # plain int → scored as count query

    if q.operation == "distinct":
        values = col.distinct(q.distinct_field, q.query)
        return sorted([_serialize(v) for v in values])

    if q.operation == "find":
        sort = _find_sort_tiebreak(col, q)
        cursor = col.find(q.query, q.projection or {})
        if sort:
            cursor = cursor.sort(list(sort.items()))
        if q.limit:
            cursor = cursor.limit(q.limit)
        return [_serialize(doc) for doc in cursor]

    if q.operation == "aggregate":
        pipeline = _agg_tiebreak(col, q.query)
        docs = list(col.aggregate(pipeline, allowDiskUse=True))
        return [_serialize(doc) for doc in docs]

    raise ValueError(f"Unknown operation: {q.operation!r}")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def _make_row(q: BenchmarkQuestion, result: Any, idx: int) -> dict[str, str]:
    input_payload = json.dumps(
        {"nlQuery": q.nl_query, "databaseName": DB_NAME},
        ensure_ascii=False,
    )
    expected_payload = json.dumps(
        {"dbQuery": _query_to_json(q), "result": result},
        ensure_ascii=False,
    )
    return {
        "input": input_payload,
        "expected": expected_payload,
        "tags": q.tags,
    }


def build_ground_truth(uri: str, out_path: Path, *, lite: bool = False) -> None:
    questions = LITE_QUESTIONS if lite else QUESTIONS
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    client.admin.command("ping")
    db = client[DB_NAME]

    col_names = db.list_collection_names()
    if not col_names:
        print(f"ERROR: database '{DB_NAME}' is empty. Run generate.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to {DB_NAME} @ {uri}")
    print(f"Collections: {', '.join(sorted(col_names))}")
    print(f"Processing {len(questions)} questions...\n")

    rows: list[dict] = []
    errors: list[tuple[int, str, str]] = []

    for idx, q in enumerate(questions, start=1):
        label = f"Q{idx:03d}"
        try:
            result = _run_question(db, q)
            row = _make_row(q, result, idx)
            rows.append(row)
            # Brief preview
            if isinstance(result, int):
                preview = str(result)
            elif isinstance(result, list):
                preview = f"[{len(result)} rows]"
            else:
                preview = str(result)[:60]
            print(f"  {label} ✓  {q.nl_query[:60]:<60}  → {preview}")
        except Exception as exc:
            errors.append((idx, q.nl_query, str(exc)))
            print(f"  {label} ✗  {q.nl_query[:60]:<60}  ERROR: {exc}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "expected", "tags"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {out_path}")
    if errors:
        print(f"WARNING: {len(errors)} questions failed:")
        for idx, nl, err in errors:
            print(f"  Q{idx:03d}: {err}")
        sys.exit(1)

    client.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mango_ecommerce benchmark CSV with ground truth answers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"), help="MongoDB URI")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Write the 50-question lite subset instead of the full 170-question set.",
    )
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else (DEFAULT_LITE_OUT if args.lite else DEFAULT_OUT)
    build_ground_truth(uri=args.uri, out_path=out_path, lite=args.lite)


if __name__ == "__main__":
    main()
