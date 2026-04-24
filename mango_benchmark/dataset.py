"""Load the NL-to-MQL benchmark dataset from the bundled CSV."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_CSV_PATH = Path(__file__).parent / "atlas_sample_data_benchmark.braintrust.csv"


def load_dataset(
    csv_path: Path | str | None = None,
    tags_filter: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load and parse the benchmark CSV.

    Args:
        csv_path: Path to CSV file. Defaults to bundled atlas dataset.
        tags_filter: If set, only include rows whose ``tags`` field matches
            one of the given strings (substring match).
        limit: Maximum number of rows to return (sampled from the start).

    Returns:
        List of dicts with keys: ``nl_query``, ``db``, ``mongo_query``,
        ``expected_result``, ``tags``, ``_idx`` (1-based row index).
    """
    path = Path(csv_path) if csv_path else _CSV_PATH
    dataset: list[dict[str, Any]] = []
    idx = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                input_data = json.loads(row["input"])
                expected_data = json.loads(row["expected"])
            except Exception:
                continue

            idx += 1
            tags = row.get("tags", "")

            if tags_filter and not any(t in tags for t in tags_filter):
                continue

            dataset.append({
                "_idx": idx,
                "nl_query": input_data["nlQuery"],
                "db": input_data["databaseName"],
                "mongo_query": expected_data["dbQuery"],
                "expected_result": expected_data.get("result"),
                "tags": tags,
            })

            if limit and len(dataset) >= limit:
                break

    return dataset
