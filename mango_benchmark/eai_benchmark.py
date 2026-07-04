"""EAI benchmark runner — evaluate MangoAgent on the Atlas sample-data dataset.

Reads the *flat* CSV (`atlas_sample_data_benchmark.flat.csv`) where each field is
its own column (``input.databaseName``, ``input.nlQuery``, ``expected.dbQuery``,
``expected.result``, ``metadata.complexity``) rather than the nested-JSON
``.braintrust.csv`` consumed by :mod:`mango_benchmark.runner`.

It reuses runner's battle-tested agent factory and XMaNeR scorer; this module only
adds the flat-CSV loader, the ``--db`` / ``--limit`` filters, the weatherdata skip,
and the per-database / per-complexity summary the task asks for.

Usage:
    python -m mango_benchmark.eai_benchmark --limit 10
    python -m mango_benchmark.eai_benchmark --db sample_mflix
    python -m mango_benchmark.eai_benchmark --provider openai --model gpt-4o
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Reuse the agent factory + per-question scorer from the main runner so the
# XMaNeR definitions stay identical across both entry points.
from mango_benchmark.runner import _build_agent, _run_question

load_dotenv()

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# sample_weatherdata is excluded: its documents are not loaded locally, so every
# query against it would false-FAIL. (Task requirement.)
_SKIP_DBS = {"sample_weatherdata"}

_DEFAULT_CSV = (
    Path(__file__).resolve().parent.parent
    / "natural-language-to-mongosh"
    / "atlas_sample_data_benchmark.flat.csv"
)

# Output CSV columns (ordered). Metrics + the agent's generated query.
_COLUMNS = [
    "question_id",
    "database",
    "complexity",
    "natural_language_query",
    "generated_query",
    "reference_output",
    "successful_execution",
    "non_empty_output",
    "reasonable_output",
    "correct_output_fuzzy",
    "xmaner",
    "latency_seconds",
    "error_detail",
]


# ---------------------------------------------------------------------------
# Flat-CSV loader
# ---------------------------------------------------------------------------


def _parse_result(raw: str) -> Any:
    """Parse an ``expected.result`` cell.

    The flat CSV stores results as Python ``repr`` strings (single-quoted keys),
    which are not strict JSON. ``ast.literal_eval`` round-trips 764/766 rows; the
    two failures are empty cells → ``None``. We fall back to ``json.loads`` for any
    double-quoted variant before giving up.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return None


def load_flat_dataset(
    csv_path: Path | str,
    *,
    db_filter: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load the flat benchmark CSV into the item shape ``_run_question`` expects.

    Skips ``_SKIP_DBS`` rows. ``_idx`` is the original 1-based CSV row number so
    results stay traceable back to the source even after filtering.
    """
    path = Path(csv_path)
    dataset: list[dict[str, Any]] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            db = (row.get("input.databaseName") or "").strip()
            if db in _SKIP_DBS:
                continue
            if db_filter and db != db_filter:
                continue

            nl_query = (row.get("input.nlQuery") or "").strip()
            if not nl_query:
                continue

            dataset.append(
                {
                    "_idx": idx,
                    "nl_query": nl_query,
                    "db": db,
                    "complexity": (row.get("metadata.complexity") or "").strip(),
                    # gold_mql is mongosh source (not structured JSON) in this CSV;
                    # the scorer uses expected_result, so we keep the raw string for
                    # the debug trail only.
                    "gold_mql": row.get("expected.dbQuery", ""),
                    "expected_result": _parse_result(row.get("expected.result", "")),
                }
            )

            if limit and len(dataset) >= limit:
                break

    return dataset


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    "successful_execution",
    "non_empty_output",
    "reasonable_output",
    "correct_output_fuzzy",
    "xmaner",
)


def _avg(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def _print_breakdown(title: str, groups: dict[str, list[dict]]) -> None:
    print(f"\n  {title}")
    print(f"    {'group':<22} {'n':>4} {'XMaNeR':>8} {'SE':>6} {'NEO':>6} {'RO':>6} {'CO':>6}")
    for name in sorted(groups):
        rows = groups[name]
        print(
            f"    {name:<22} {len(rows):>4} "
            f"{_avg(rows, 'xmaner'):>8.3f} "
            f"{_avg(rows, 'successful_execution'):>6.2f} "
            f"{_avg(rows, 'non_empty_output'):>6.2f} "
            f"{_avg(rows, 'reasonable_output'):>6.2f} "
            f"{_avg(rows, 'correct_output_fuzzy'):>6.2f}"
        )


def print_summary(rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        print("\nNo results.")
        return

    print(f"\n{'=' * 64}")
    print("SUMMARY")
    print(f"{'=' * 64}")
    print(f"  Cases     : {n}")
    print(f"  XMaNeR    : {_avg(rows, 'xmaner'):.4f}   <-- primary metric")
    print(f"  SE        : {_avg(rows, 'successful_execution'):.4f}")
    print(f"  NEO       : {_avg(rows, 'non_empty_output'):.4f}")
    print(f"  RO        : {_avg(rows, 'reasonable_output'):.4f}")
    print(f"  CO (fuzzy): {_avg(rows, 'correct_output_fuzzy'):.4f}")

    by_db: dict[str, list[dict]] = defaultdict(list)
    by_cx: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_db[r["database"]].append(r)
        by_cx[r["complexity"] or "unknown"].append(r)

    _print_breakdown("Per-database", by_db)
    _print_breakdown("Per-complexity", by_cx)

    errs = [r for r in rows if r["error_detail"]]
    if errs:
        print(f"\n  Errors: {len(errs)}/{n}")
        for r in errs[:5]:
            print(f"    Q{r['question_id']}: {r['error_detail'][:80]}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    dataset = load_flat_dataset(
        args.csv_path, db_filter=args.db, limit=args.limit if args.limit > 0 else None
    )
    if not dataset:
        print("No benchmark cases matched the filters.")
        return
    print(f"Loaded {len(dataset)} cases (weatherdata skipped).")

    # Group by db so schema introspection / agent build happens once per database.
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in dataset:
        groups[item["db"]].append(item)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for db_name, items in groups.items():
            print(f"\n=== DB: {db_name} ({len(items)} cases) ===")
            try:
                base_agent = _build_agent(
                    provider=args.provider,
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    db_name=db_name,
                    mongo_uri=args.mongo_uri,
                    memory_enabled=not args.no_memory,
                    memory_base_dir=str(out_path.parent / ".eai_memory"),
                    max_iterations=args.max_iterations,
                    max_rows=args.max_rows,
                    temperature=args.temperature,
                )
            except Exception as exc:
                print(f"  ERROR building agent for {db_name}: {exc}")
                for item in items:
                    row = _error_row(item, f"agent init failed: {exc}")
                    all_rows.append(row)
                    writer.writerow(row)
                out_f.flush()
                continue

            for i, item in enumerate(items, 1):
                print(f"  [{i}/{len(items)}] {item['nl_query'][:70]}", end="\r", flush=True)
                t0 = time.perf_counter()
                session = base_agent.new_session()
                raw = await _run_question(session, item, args.timeout, model_label=args.model or args.provider)

                row = {
                    "question_id": item["_idx"],
                    "database": item["db"],
                    "complexity": item["complexity"],
                    "natural_language_query": item["nl_query"],
                    "generated_query": json.dumps(raw.get("agent_mql"), default=str, ensure_ascii=False),
                    "reference_output": raw["reference_output"],
                    "successful_execution": raw["successful_execution"],
                    "non_empty_output": raw["non_empty_output"],
                    "reasonable_output": raw["reasonable_output"],
                    "correct_output_fuzzy": raw["correct_output_fuzzy"],
                    "xmaner": raw["xmaner"],
                    "latency_seconds": raw["latency_seconds"],
                    "error_detail": raw["error_detail"],
                }
                if args.verbose:
                    print(
                        f"\n    Q{item['_idx']} [{item['complexity']}] xmaner={row['xmaner']:.2f} "
                        f"se={row['successful_execution']:.0f} neo={row['non_empty_output']:.0f} "
                        f"ro={row['reasonable_output']:.0f} co={row['correct_output_fuzzy']:.0f} "
                        f"({time.perf_counter() - t0:.1f}s)"
                    )
                all_rows.append(row)
                writer.writerow(row)
                out_f.flush()
            print()

    print(f"\nResults saved to: {out_path}")
    print_summary(all_rows)


def _error_row(item: dict, detail: str) -> dict:
    return {
        "question_id": item["_idx"],
        "database": item["db"],
        "complexity": item["complexity"],
        "natural_language_query": item["nl_query"],
        "generated_query": "",
        "reference_output": json.dumps(item["expected_result"], default=str, ensure_ascii=False)
        if item["expected_result"] is not None
        else "",
        "successful_execution": 0.0,
        "non_empty_output": 0.0,
        "reasonable_output": 0.0,
        "correct_output_fuzzy": 0.0,
        "xmaner": 0.0,
        "latency_seconds": 0.0,
        "error_detail": detail,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate MangoAgent on the Atlas EAI benchmark (flat CSV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv-path", default=str(_DEFAULT_CSV), help="Path to the flat benchmark CSV.")
    p.add_argument("--output", default="benchmark_results.csv", help="Where to write per-case results.")
    p.add_argument("--limit", type=int, default=0, metavar="N", help="Run only the first N cases (0 = all).")
    p.add_argument("--db", default=None, metavar="NAME", help="Run only cases for this database.")
    p.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        help="MongoDB connection URI.",
    )
    p.add_argument("--provider", default=os.getenv("MANGO_PROVIDER", "openai"), help="LLM provider.")
    p.add_argument("--model", default=os.getenv("MANGO_MODEL"), help="Model ID.")
    p.add_argument("--api-key", default=os.getenv("REGOLO_API_KEY"), help="LLM API key (else provider env var).")
    p.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or os.getenv("REGOLO_URL"),
        help="Base URL for OpenAI-compatible providers.",
    )
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature (omit for provider default).")
    p.add_argument("--timeout", type=float, default=60.0, help="Per-case timeout in seconds.")
    p.add_argument("--max-iterations", type=int, default=8, help="Max agent tool-call iterations per case.")
    p.add_argument("--max-rows", type=int, default=100_000, help="RunMQLTool row cap (high = no one-sided truncation).")
    p.add_argument("--no-memory", action="store_true", help="Disable ChromaDB memory layer.")
    p.add_argument("--verbose", action="store_true", help="Print per-case scores.")

    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
