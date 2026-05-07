"""Stratified honest split for training vs benchmark ablation.

Logic:
  - 169 benchmark questions, 79 with a matching training entry in the JSONL.
  - The 79 "covered" questions are joined with their training entry and split
    into train/test stratified by difficulty tag (easy / medium / hard).
  - The 90 "uncovered" questions (no training entry) always go to the test set.
  - Zero overlap by construction: a question is either in training_train OR
    in benchmark_test, never both.

Outputs (in mango_benchmark/splits/):
  training_train.jsonl  — text notes + training entries for the train split
  benchmark_test.csv    — test questions (covered test-split + all uncovered)

Usage:
    python -m mango_benchmark.split
    python -m mango_benchmark.split --test-ratio 0.25 --seed 42

Ablation runs:
    # 1. No memory baseline
    python -m mango_benchmark.runner \\
        --csv-path mango_benchmark/splits/benchmark_test.csv --no-memory --sample 0

    # 2. Memory auto-save only
    python -m mango_benchmark.runner \\
        --csv-path mango_benchmark/splits/benchmark_test.csv --sample 0

    # 3. Memory + training (no leakage)
    python -m mango_benchmark.runner \\
        --csv-path mango_benchmark/splits/benchmark_test.csv \\
        --training-file mango_benchmark/splits/training_train.jsonl --sample 0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_ROOT          = Path(__file__).parent.parent
_TRAINING_JSONL = _ROOT / "examples" / "trainingset_example.jsonl"
_BENCHMARK_CSV  = _ROOT / "mango_benchmark" / "mango_ecommerce_benchmark.csv"
_OUT_DIR        = _ROOT / "mango_benchmark" / "splits"

_DIFFICULTY_TAGS = ("easy", "medium", "hard")


def _difficulty(tags: str) -> str:
    for d in _DIFFICULTY_TAGS:
        if d in tags.split("|"):
            return d
    return "medium"


def split(
    test_ratio: float = 0.25,
    seed: int = 42,
    training_jsonl: Path = _TRAINING_JSONL,
    benchmark_csv: Path = _BENCHMARK_CSV,
    out_dir: Path = _OUT_DIR,
) -> None:
    rng = random.Random(seed)

    # --- Load training JSONL ---
    with open(training_jsonl) as f:
        all_lines = [json.loads(l) for l in f if l.strip()]
    text_entries = [e for e in all_lines if e.get("type") != "training"]
    # Index training entries by normalised question text
    training_by_q: dict[str, dict] = {
        e["question"].strip().lower(): e
        for e in all_lines if e.get("type") == "training"
    }

    # --- Load benchmark CSV ---
    with open(benchmark_csv, newline="") as f:
        bench_rows = list(csv.DictReader(f))

    # --- Join: covered vs uncovered ---
    covered:   list[dict] = []  # have a training entry
    uncovered: list[dict] = []  # no training entry → always go to test

    for row in bench_rows:
        try:
            q = json.loads(row["input"])["nlQuery"].strip().lower()
        except Exception:
            uncovered.append(row)
            continue
        tags = row.get("tags", "")
        if q in training_by_q:
            covered.append({"row": row, "entry": training_by_q[q], "difficulty": _difficulty(tags)})
        else:
            uncovered.append(row)

    # --- Stratified split of covered questions ---
    # Group by difficulty, shuffle each group, then take test_ratio from each.
    by_difficulty: dict[str, list] = defaultdict(list)
    for item in covered:
        by_difficulty[item["difficulty"]].append(item)

    train_items: list[dict] = []
    test_items:  list[dict] = []

    for diff in _DIFFICULTY_TAGS:
        group = by_difficulty[diff]
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_ratio)) if group else 0
        test_items.extend(group[:n_test])
        train_items.extend(group[n_test:])

    # --- Write outputs ---
    out_dir.mkdir(parents=True, exist_ok=True)

    # training_train.jsonl: text notes + training entries for train questions
    with open(out_dir / "training_train.jsonl", "w") as f:
        for e in text_entries:
            f.write(json.dumps(e) + "\n")
        for item in train_items:
            f.write(json.dumps(item["entry"]) + "\n")

    # benchmark_test.csv: test-split covered + all uncovered
    test_bench_rows = [item["row"] for item in test_items] + uncovered
    fieldnames = list(bench_rows[0].keys())
    with open(out_dir / "benchmark_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_bench_rows)

    # --- Report ---
    n_covered   = len(covered)
    n_uncovered = len(uncovered)
    n_train     = len(train_items)
    n_test_cov  = len(test_items)
    n_test_tot  = len(test_bench_rows)

    print(f"Seed {seed}  |  test_ratio {test_ratio}")
    print()
    print(f"  Benchmark total   : {len(bench_rows)} questions")
    print(f"  Covered (w/ entry): {n_covered}  →  {n_train} train / {n_test_cov} test")
    print(f"  Uncovered         : {n_uncovered}  →  all go to test")
    print()
    print("  Train split by difficulty:")
    for d in _DIFFICULTY_TAGS:
        n = sum(1 for i in train_items if i["difficulty"] == d)
        print(f"    {d:<8} {n}")
    print("  Test split by difficulty (covered only):")
    for d in _DIFFICULTY_TAGS:
        n = sum(1 for i in test_items if i["difficulty"] == d)
        print(f"    {d:<8} {n}")
    print()
    print(f"  training_train.jsonl : {n_train} training entries + {len(text_entries)} text notes")
    print(f"  benchmark_test.csv   : {n_test_tot} questions ({n_test_cov} covered + {n_uncovered} uncovered)")
    print(f"  Overlap              : 0  (guaranteed by construction)")
    print()
    print(f"Output → {out_dir}/")
    print()
    print("Ablation runs:")
    print(f"  python -m mango_benchmark.runner --csv-path {out_dir}/benchmark_test.csv --no-memory --sample 0")
    print(f"  python -m mango_benchmark.runner --csv-path {out_dir}/benchmark_test.csv --sample 0")
    print(f"  python -m mango_benchmark.runner --csv-path {out_dir}/benchmark_test.csv --training-file {out_dir}/training_train.jsonl --sample 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified honest split for mango ablation")
    parser.add_argument("--test-ratio", type=float, default=0.25,
                        help="Fraction of covered questions for test (default: 0.25)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split(test_ratio=args.test_ratio, seed=args.seed)
