"""Benchmark report generator.

Reads one or more result CSV files produced by runner.py and prints
a comparison table of XMaNeR metrics per model, with optional per-DB breakdown.

Usage:
    python -m mango_benchmark.report mango_benchmark/results/results_20250417_120000.csv
    python -m mango_benchmark.report results_a.csv results_b.csv   # compare runs
    python -m mango_benchmark.report results.csv --by-db           # per-DB breakdown
    python -m mango_benchmark.report results.csv --out report.md   # save to file
"""

from __future__ import annotations

import argparse
import csv
import sys

csv.field_size_limit(sys.maxsize)
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast numeric columns
            for col in ("successful_execution", "non_empty_output", "reasonable_output",
                        "correct_output_fuzzy", "xmaner", "latency_seconds"):
                try:
                    row[col] = float(row[col])  # type: ignore[assignment]
                except (ValueError, KeyError):
                    row[col] = 0.0  # type: ignore[assignment]
            for col in ("token_input", "token_output", "iterations", "memory_hits", "retries"):
                try:
                    row[col] = int(row[col])  # type: ignore[assignment]
                except (ValueError, KeyError):
                    row[col] = 0  # type: ignore[assignment]
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _avg(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(r[key] for r in rows) / len(rows)  # type: ignore[operator]


def _summarise(rows: list[dict]) -> dict[str, float | int]:
    n = len(rows)
    return {
        "n": n,
        "xmaner": _avg(rows, "xmaner"),
        "se": _avg(rows, "successful_execution"),
        "neo": _avg(rows, "non_empty_output"),
        "ro": _avg(rows, "reasonable_output"),
        "co": _avg(rows, "correct_output_fuzzy"),
        "latency": _avg(rows, "latency_seconds"),
        "tok_in": _avg(rows, "token_input"),
        "tok_out": _avg(rows, "token_output"),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _bar(value: float, width: int = 10) -> str:
    """Simple ASCII progress bar."""
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def _fmt(value: float | int, decimals: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"


def _table(
    headers: list[str],
    rows: list[list[str]],
    out: TextIO,
    markdown: bool = False,
) -> None:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    if markdown:
        sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for cell, w in zip(cells, col_widths):
            parts.append(f" {str(cell):<{w}} ")
        return "|" + "|".join(parts) + "|"

    if not markdown:
        out.write(sep + "\n")
    out.write(fmt_row(headers) + "\n")
    if markdown:
        out.write("|" + "|".join(f" {'-'*w} " for w in col_widths) + "|\n")
    else:
        out.write(sep + "\n")
    for row in rows:
        out.write(fmt_row(row) + "\n")
    if not markdown:
        out.write(sep + "\n")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    csv_paths: list[str | Path],
    *,
    by_db: bool = False,
    out: TextIO | None = None,
    markdown: bool = False,
) -> None:
    if out is None:
        out = sys.stdout

    # Load all rows; tag with source file if multiple files given
    all_rows: list[dict] = []
    for path in csv_paths:
        rows = _load_csv(path)
        if len(csv_paths) > 1:
            src = Path(path).stem
            for r in rows:
                # Prefix model label with source file stem to distinguish runs
                r["_source"] = src
        all_rows.extend(rows)

    if not all_rows:
        out.write("No data found in provided CSV files.\n")
        return

    # Group by model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        model_key = row.get("model", "unknown")
        if "_source" in row:
            model_key = f"{row['_source']} / {model_key}"
        by_model[model_key].append(row)

    # --- Overall comparison table ---
    out.write("\n")
    if markdown:
        out.write("## Benchmark Results\n\n")
    else:
        out.write("=== Benchmark Results ===\n\n")

    headers = ["Model", "N", "XMaNeR", "SE", "NEO", "RO", "CO", "Latency(s)", "Tokens(in/out)"]
    table_rows: list[list[str]] = []

    for model_label, rows in sorted(by_model.items(), key=lambda x: -_avg(x[1], "xmaner")):
        s = _summarise(rows)
        table_rows.append([
            model_label,
            str(s["n"]),
            _fmt(s["xmaner"]),
            _fmt(s["se"]),
            _fmt(s["neo"]),
            _fmt(s["ro"]),
            _fmt(s["co"]),
            f"{s['latency']:.2f}",
            f"{s['tok_in']:.0f}/{s['tok_out']:.0f}",
        ])

    _table(headers, table_rows, out, markdown=markdown)

    # --- Reference scores ---
    out.write("\n")
    if markdown:
        out.write("_Official reference scores (for comparison): "
                  "Claude 3.7 Sonnet 0.8671 · GPT-4o 0.8253_\n")
    else:
        out.write("Reference: Claude 3.7 Sonnet = 0.8671 | GPT-4o = 0.8253\n")

    # --- Per-DB breakdown ---
    if by_db:
        out.write("\n")
        if markdown:
            out.write("## Per-Database Breakdown\n\n")
        else:
            out.write("\n=== Per-Database Breakdown ===\n")

        all_dbs = sorted({r["database"] for r in all_rows})

        for db_name in all_dbs:
            out.write(f"\n{'  ' if not markdown else '### '}{db_name}\n")
            headers_db = ["Model", "N", "XMaNeR", "SE", "NEO", "RO", "CO"]
            db_rows: list[list[str]] = []
            for model_label, rows in sorted(by_model.items(), key=lambda x: -_avg(x[1], "xmaner")):
                db_subset = [r for r in rows if r["database"] == db_name]
                if not db_subset:
                    continue
                s = _summarise(db_subset)
                db_rows.append([
                    model_label,
                    str(s["n"]),
                    _fmt(s["xmaner"]),
                    _fmt(s["se"]),
                    _fmt(s["neo"]),
                    _fmt(s["ro"]),
                    _fmt(s["co"]),
                ])
            if db_rows:
                if markdown:
                    out.write("\n")
                _table(headers_db, db_rows, out, markdown=markdown)

    out.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a benchmark comparison report from result CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        metavar="CSV",
        help="Result CSV file(s) produced by runner.py.",
    )
    parser.add_argument(
        "--by-db",
        action="store_true",
        help="Include per-database breakdown.",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="Write report to FILE instead of stdout.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output in Markdown table format.",
    )

    args = parser.parse_args()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            generate_report(args.csv_files, by_db=args.by_db, out=f, markdown=args.markdown)
        print(f"Report saved to: {args.out}")
    else:
        generate_report(args.csv_files, by_db=args.by_db, markdown=args.markdown)


if __name__ == "__main__":
    main()
