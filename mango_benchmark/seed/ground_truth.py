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

from mango_benchmark.seed.questions import QUESTIONS, BenchmarkQuestion
from mango_benchmark.seed.schema import DB_NAME

from dotenv import load_dotenv

load_dotenv()

DEFAULT_OUT = Path(__file__).parent.parent / "mango_ecommerce_benchmark.csv"


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


def build_ground_truth(uri: str, out_path: Path) -> None:
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    client.admin.command("ping")
    db = client[DB_NAME]

    col_names = db.list_collection_names()
    if not col_names:
        print(f"ERROR: database '{DB_NAME}' is empty. Run generate.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to {DB_NAME} @ {uri}")
    print(f"Collections: {', '.join(sorted(col_names))}")
    print(f"Processing {len(QUESTIONS)} questions...\n")

    rows: list[dict] = []
    errors: list[tuple[int, str, str]] = []

    for idx, q in enumerate(QUESTIONS, start=1):
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
        default=str(DEFAULT_OUT),
        help="Output CSV path",
    )
    args = parser.parse_args()
    build_ground_truth(uri=args.uri, out_path=Path(args.out))


if __name__ == "__main__":
    main()
