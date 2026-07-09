"""Build the benchmark artifacts from the authored cases (DATASET_DESIGN.md §7.3).

Executes every gold (and every alternative) against the live seeded DBs and
emits:

  * ``cases.jsonl``  — the canonical §4 records
  * ``bench_v1.csv`` — runner-compatible Braintrust CSV (all domains combined)

Usage:
    python -m mango_benchmark.bench_datasets.build
    python -m mango_benchmark.bench_datasets.build --uri mongodb://localhost:27017 --regen
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from pymongo import MongoClient

from mango_benchmark.bench_datasets.common import (
    BenchCase,
    check_unique_ids,
    execute_gold,
    resolve_base,
)

CASE_MODULES = [
    "mango_benchmark.bench_datasets.logistics.questions",
    "mango_benchmark.bench_datasets.workforce.questions",
    "mango_benchmark.bench_datasets.meters.questions",
]

GENERATOR_MODULES = [
    "mango_benchmark.bench_datasets.logistics.generate",
    "mango_benchmark.bench_datasets.workforce.generate",
    "mango_benchmark.bench_datasets.meters.generate",
]

OUT_DIR = Path(__file__).parent
CASES_JSONL = OUT_DIR / "cases.jsonl"
BENCH_CSV = OUT_DIR / "bench_v1.csv"


def load_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []
    for mod_name in CASE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name == mod_name:
                print(f"  (no question module {mod_name} yet — skipped)")
                continue
            raise
        cases.extend(mod.CASES)
    check_unique_ids(cases)
    return cases


def build(uri: str, regen: bool) -> int:
    if regen:
        for mod_name in GENERATOR_MODULES:
            print(f"regenerating via {mod_name} ...")
            importlib.import_module(mod_name).generate(uri)

    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    client.admin.command("ping")

    cases = load_cases()
    by_id = {c.id: c for c in cases}
    print(f"building {len(cases)} cases ...")

    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, str]] = []
    errors: list[tuple[str, str]] = []

    for case in cases:
        base = resolve_base(case, by_id)
        gold = base.gold
        db = client[base.database]
        try:
            result = execute_gold(db, gold) if gold else None
            alt_results = [execute_gold(db, alt) for alt in base.gold_alternatives]
        except Exception as exc:
            errors.append((case.id, str(exc)))
            print(f"  {case.id} ✗ gold execution failed: {exc}", file=sys.stderr)
            continue
        records.append(case.to_record(gold, result, alt_results))
        csv_rows.append(case.to_csv_row(gold, result))

    with open(CASES_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(BENCH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "expected", "tags"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"wrote {len(records)} records -> {CASES_JSONL.name}, {BENCH_CSV.name}")
    if errors:
        print(f"BUILD FAILED: {len(errors)} gold executions failed", file=sys.stderr)
        return 1
    client.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bench_datasets artifacts")
    parser.add_argument("--uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    parser.add_argument("--regen", action="store_true", help="re-run the DB generators first")
    args = parser.parse_args()
    sys.exit(build(args.uri, args.regen))


if __name__ == "__main__":
    main()
