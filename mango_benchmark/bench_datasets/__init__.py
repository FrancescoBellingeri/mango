"""Primary NL-to-MQL benchmark dataset (DATASET_DESIGN.md).

Three synthetic domains (`logistics`, `workforce`, `meters`), a shared case
model (`common.py`), and the build/validation pipeline:

    python -m mango_benchmark.bench_datasets.build      # seed DBs + emit cases
    python -m mango_benchmark.bench_datasets.validate   # §6 gates (hard-fail)
    python -m mango_benchmark.bench_datasets.contamination
"""
