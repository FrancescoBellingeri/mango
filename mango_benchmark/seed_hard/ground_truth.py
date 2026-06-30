"""Compute ground-truth answers for the *mango_marketplace* HARD benchmark.

Requires a running MongoDB with the ``mango_marketplace`` database already
populated (run ``generate.py`` first).

The CSV is written in the same Braintrust format consumed by dataset.py:
    input:    {"nlQuery": "...", "databaseName": "mango_marketplace"}
    expected: {"dbQuery": <serialized_query>, "result": <computed_result>}
    tags:     "hard|lookup|orders"

Usage:
    python -m mango_benchmark.seed_hard.ground_truth --uri mongodb://localhost:27017
    python -m mango_benchmark.seed_hard.ground_truth --uri mongodb://localhost:27017 \\
        --out mango_benchmark/mango_marketplace_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from bson import ObjectId
from pymongo import MongoClient

from mango_benchmark.seed_hard.questions import QUESTIONS, BenchmarkQuestion
from mango_benchmark.seed_hard.schema import DB_NAME

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv optional
    pass

DEFAULT_OUT = Path(__file__).parent.parent / "mango_marketplace_benchmark.csv"


# ---------------------------------------------------------------------------
# BSON / datetime JSON serialisation
# ---------------------------------------------------------------------------


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
    """Serialize the FULL gold query, not just the filter/pipeline body.

    The previous version emitted only ``q.query`` (the filter for find/count/
    distinct, the pipeline for aggregate). That silently dropped ``sort``,
    ``limit``, ``projection`` and ``distinct_field`` for find-style queries — so
    a ranking like "the 10 highest-commission merchants" lost its sort, and the
    scorer could not tell the question was order-sensitive (→ false PASS).

    The structured form mirrors ``mango.core.types.QueryRequest`` so the gold and
    the agent's MQL share one schema:

        {"operation","collection","filter","pipeline",
         "projection","sort","limit","distinct_field"}

    ``filter`` carries the body for find/count/distinct; ``pipeline`` carries it
    for aggregate. The two are mutually exclusive.
    """
    is_pipeline = q.operation == "aggregate"
    structured = {
        "operation": q.operation,
        "collection": q.collection,
        "filter": None if is_pipeline else _serialize(q.query),
        "pipeline": _serialize(q.query) if is_pipeline else None,
        "projection": _serialize(q.projection),
        "sort": _serialize(q.sort),
        "limit": q.limit,
        "distinct_field": q.distinct_field,
    }
    return json.dumps(structured, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def _run_question(db: Any, q: BenchmarkQuestion) -> Any:
    col = db[q.collection]

    if q.operation == "count":
        return col.count_documents(q.query)

    if q.operation == "distinct":
        values = col.distinct(q.distinct_field, q.query)
        return sorted([_serialize(v) for v in values], key=lambda x: (str(type(x)), str(x)))

    if q.operation == "find":
        cursor = col.find(q.query, q.projection or {})
        if q.sort:
            cursor = cursor.sort(list(q.sort.items()))
        if q.limit:
            cursor = cursor.limit(q.limit)
        return [_serialize(doc) for doc in cursor]

    if q.operation == "aggregate":
        docs = list(col.aggregate(q.query, allowDiskUse=True))
        return [_serialize(doc) for doc in docs]

    raise ValueError(f"Unknown operation: {q.operation!r}")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def _make_row(q: BenchmarkQuestion, result: Any) -> dict[str, str]:
    return {
        "input": json.dumps({"nlQuery": q.nl_query, "databaseName": DB_NAME}, ensure_ascii=False),
        "expected": json.dumps({"dbQuery": _query_to_json(q), "result": result}, ensure_ascii=False),
        "tags": q.tags,
    }


def build_ground_truth(uri: str, out_path: Path) -> None:
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    client.admin.command("ping")
    db = client[DB_NAME]

    col_names = db.list_collection_names()
    if not col_names:
        print(f"ERROR: database '{DB_NAME}' is empty. Run generate.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to {DB_NAME} @ {uri}")
    print(f"Processing {len(QUESTIONS)} questions...\n")

    rows: list[dict] = []
    errors: list[tuple[int, str, str]] = []

    for idx, q in enumerate(QUESTIONS, start=1):
        label = f"Q{idx:03d}"
        try:
            result = _run_question(db, q)
            rows.append(_make_row(q, result))
            if isinstance(result, int):
                preview = str(result)
            elif isinstance(result, list):
                preview = f"[{len(result)} rows]"
            else:
                preview = str(result)[:60]
            print(f"  {label} ✓  {q.nl_query[:58]:<58}  → {preview}")
        except Exception as exc:
            errors.append((idx, q.nl_query, str(exc)))
            print(f"  {label} ✗  {q.nl_query[:58]:<58}  ERROR: {exc}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "expected", "tags"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {out_path}")
    if errors:
        print(f"WARNING: {len(errors)} questions failed:")
        for idx, _nl, err in errors:
            print(f"  Q{idx:03d}: {err}")
        sys.exit(1)

    client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mango_marketplace HARD benchmark CSV with ground-truth answers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"), help="MongoDB URI")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    args = parser.parse_args()
    build_ground_truth(uri=args.uri, out_path=Path(args.out))


if __name__ == "__main__":
    main()
