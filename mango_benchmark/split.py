"""K-fold honest split for training vs benchmark ablation.

Takes the 79 training entries (trainingset_example.jsonl) and 79 benchmark
rows (mango_ecommerce_benchmark.csv) and splits them independently into
train/test by index using a fixed random seed.

Outputs:
  mango_benchmark/splits/training_train.jsonl   — entries to load into ChromaDB
  mango_benchmark/splits/training_test.jsonl    — held-out training entries (not loaded)
  mango_benchmark/splits/benchmark_test.csv     — benchmark rows to evaluate on

The test indices are THE SAME for both files, so the 19 benchmark questions
correspond (by position/topic) to the 19 held-out training entries.

Usage:
    python -m mango_benchmark.split
    python -m mango_benchmark.split --test-ratio 0.25 --seed 42

Then run the 3 ablation comparisons:
    # 1. No memory baseline
    python -m mango_benchmark.runner \\
        --csv-path mango_benchmark/splits/benchmark_test.csv --no-memory

    # 2. Memory, no training
    python -m mango_benchmark.runner \\
        --csv-path mango_benchmark/splits/benchmark_test.csv

    # 3. Memory + training (60 entries, test questions not included)
    python -m mango_benchmark.runner \\
        --csv-path mango_benchmark/splits/benchmark_test.csv \\
        --training-file mango_benchmark/splits/training_train.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_ROOT = Path(__file__).parent.parent
_TRAINING_JSONL = _ROOT / "examples" / "trainingset_example.jsonl"
_BENCHMARK_CSV  = _ROOT / "mango_benchmark" / "mango_ecommerce_benchmark.csv"
_OUT_DIR        = _ROOT / "mango_benchmark" / "splits"


def split(
    test_ratio: float = 0.35,
    seed: int = 42,
    training_jsonl: Path = _TRAINING_JSONL,
    benchmark_csv: Path = _BENCHMARK_CSV,
    out_dir: Path = _OUT_DIR,
) -> None:
    rng = random.Random(seed)

    # Load training entries (type=training only, preserve text entries separately)
    with open(training_jsonl) as f:
        all_lines = [json.loads(l) for l in f if l.strip()]
    text_entries   = [e for e in all_lines if e.get("type") != "training"]
    training_entries = [e for e in all_lines if e.get("type") == "training"]

    # Load benchmark rows
    with open(benchmark_csv, newline="") as f:
        bench_rows = list(csv.DictReader(f))

    n_train_total = len(training_entries)
    n_bench_total = len(bench_rows)

    # Split training entries independently
    train_indices = list(range(n_train_total))
    rng.shuffle(train_indices)
    n_train_test  = max(1, round(n_train_total * test_ratio))
    n_train_train = n_train_total - n_train_test
    train_test_idx  = set(train_indices[:n_train_test])
    train_train_idx = set(train_indices[n_train_test:])

    # Split benchmark rows independently
    bench_indices = list(range(n_bench_total))
    rng.shuffle(bench_indices)
    n_bench_test = max(1, round(n_bench_total * test_ratio))
    bench_test_idx = bench_indices[:n_bench_test]

    train_entries_train = [training_entries[i] for i in sorted(train_train_idx)]
    bench_test          = [bench_rows[i]        for i in sorted(bench_test_idx)]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write training_train.jsonl (text entries + train split)
    with open(out_dir / "training_train.jsonl", "w") as f:
        for e in text_entries:
            f.write(json.dumps(e) + "\n")
        for e in train_entries_train:
            f.write(json.dumps(e) + "\n")

    # Write training_test.jsonl (held-out — not loaded, just for reference)
    with open(out_dir / "training_test.jsonl", "w") as f:
        for i in sorted(train_test_idx):
            f.write(json.dumps(training_entries[i]) + "\n")

    # Write benchmark_test.csv
    with open(out_dir / "benchmark_test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bench_test)

    print(f"Seed: {seed}  |  Training: {n_train_total} total → {n_train_train} train / {n_train_test} held-out")
    print(f"           |  Benchmark: {n_bench_total} total → {n_bench_test} test questions")
    print(f"Output → {out_dir}/")
    print(f"  training_train.jsonl  — {n_train_train} entries + {len(text_entries)} text notes")
    print(f"  training_test.jsonl   — {n_train_test} held-out entries (reference only)")
    print(f"  benchmark_test.csv    — {n_bench_test} benchmark questions to evaluate")
    print()
    print("Run ablation:")
    print("  # 1. No memory")
    print(f"  python -m mango_benchmark.runner --csv-path {out_dir}/benchmark_test.csv --no-memory --sample 0")
    print("  # 2. Memory, no training")
    print(f"  python -m mango_benchmark.runner --csv-path {out_dir}/benchmark_test.csv --sample 0")
    print("  # 3. Memory + training")
    print(f"  python -m mango_benchmark.runner --csv-path {out_dir}/benchmark_test.csv --training-file {out_dir}/training_train.jsonl --sample 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split training/benchmark for honest ablation")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Fraction for test set (default: 0.25 → ~20 questions)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split(test_ratio=args.test_ratio, seed=args.seed)
