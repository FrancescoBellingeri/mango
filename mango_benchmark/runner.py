"""Benchmark runner — evaluates MangoAgent against the NL-to-MQL dataset.

Usage:
    python -m mango_benchmark.runner                          # 50 questions, config.py defaults
    python -m mango_benchmark.runner --sample 0               # all 766 questions
    python -m mango_benchmark.runner --provider openai --model gpt-4o
    python -m mango_benchmark.runner --providers anthropic:claude-sonnet-4-6 openai:gpt-4o
    python -m mango_benchmark.runner --no-memory
    python -m mango_benchmark.runner --dry-run                # validate dataset, no DB needed
    python -m mango_benchmark.runner --help
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from mango import MangoAgent
from mango.integrations.mongodb import MongoRunner
from mango.llm.factory import build_llm
from mango.tools import (
    CollectionStatsTool,
    DescribeCollectionTool,
    ListCollectionsTool,
    RunMQLTool,
    SearchCollectionsTool,
    ToolRegistry,
)

from mango_benchmark.dataset import load_dataset
import mango_benchmark.config as _cfg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# CSV output columns (ordered)
_COLUMNS = [
    "model",
    "question_id",
    "database",
    "natural_language_query",
    "mango_result",
    "reference_output",
    "successful_execution",
    "non_empty_output",
    "reasonable_output",
    "correct_output_fuzzy",
    "xmaner",
    "token_input",
    "token_output",
    "latency_seconds",
    "tool_calls",
    "iterations",
    "memory_hits",
    "retries",
    "error_detail",
]

# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------


def _uri_for_db(base_uri: str, db_name: str) -> str:
    """Inject a database name into a MongoDB URI."""
    p = urlparse(base_uri)
    return urlunparse(p._replace(path=f"/{db_name}"))


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _build_agent(
    *,
    provider: str,
    model: str | None,
    api_key: str | None,
    db_name: str,
    mongo_uri: str,
    memory_enabled: bool,
    memory_base_dir: str,
    max_iterations: int,
    training_file: str | None = None,
    mongo_kwargs: dict | None = None,
) -> MangoAgent:
    """Build and return a ready MangoAgent connected to *db_name*."""
    llm = build_llm(provider=provider, model=model, api_key=api_key)

    db = MongoRunner()
    db.connect(_uri_for_db(mongo_uri, db_name), **(mongo_kwargs or {}))

    tools = ToolRegistry()
    tools.register(ListCollectionsTool(db))
    tools.register(SearchCollectionsTool(db))
    tools.register(DescribeCollectionTool(db))
    tools.register(CollectionStatsTool(db))

    agent_memory = None
    if memory_enabled:
        from mango.integrations.chromadb import ChromaAgentMemory
        from mango.tools import SaveTextMemoryTool, SearchSavedCorrectToolUsesTool

        safe_label = (model or provider).replace("/", "_").replace(":", "_")
        training_tag = "_trained" if training_file else ""
        persist_dir = os.path.join(memory_base_dir, f"{safe_label}_{db_name}{training_tag}")
        agent_memory = ChromaAgentMemory(
            collection_name="benchmark_memory",
            persist_dir=persist_dir,
        )

        tools.register(SearchSavedCorrectToolUsesTool(agent_memory))
        tools.register(SaveTextMemoryTool(agent_memory))

    tools.register(RunMQLTool(db))

    agent = MangoAgent(
        llm_service=llm,
        tool_registry=tools,
        db=db,
        agent_memory=agent_memory,
        introspect=True,
        max_iterations=max_iterations,
    )
    agent.setup()
    return agent


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _parse_mql_result(result_text: str) -> dict[str, Any] | None:
    """Parse a RunMQLTool result_text into a Python dict, or None on failure."""
    if not result_text:
        return None
    # Error or retry messages injected by the agent loop
    if result_text.startswith(("ERROR:", "[RETRY", "[FATAL]", "[MAX RETRIES")):
        return None
    try:
        return json.loads(result_text)
    except (json.JSONDecodeError, ValueError):
        return None


def _flatten_values(obj: Any) -> list:
    """Recursively collect all primitive (non-dict, non-list) values."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        out: list = []
        for v in obj.values():
            out.extend(_flatten_values(v))
        return out
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_flatten_values(item))
        return out
    return [obj]


def _nums_close(a: Any, b: Any, tol: float = 0.01) -> bool:
    """Return True when both values are numbers and within *tol* relative."""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if fa == fb == 0:
        return True
    denom = max(abs(fa), abs(fb))
    return abs(fa - fb) / denom <= tol


def _values_match(a: Any, b: Any) -> bool:
    """Fuzzy equality: strings case-insensitive, numbers within 1%."""
    if a == b:
        return True
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return _nums_close(a, b)


def _fuzzy_f1(actual: list, expected: list) -> float:
    """F1 between actual and expected value lists using fuzzy matching."""
    if not actual and not expected:
        return 1.0
    if not actual or not expected:
        return 0.0

    matched = 0
    used: set[int] = set()
    for a_val in actual:
        for i, e_val in enumerate(expected):
            if i not in used and _values_match(a_val, e_val):
                matched += 1
                used.add(i)
                break

    precision = matched / len(actual)
    recall = matched / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _leaf_fields(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Return {dotted.path: value} for all leaf fields in a dict."""
    if not isinstance(obj, dict):
        return {prefix: obj} if prefix else {}
    out: dict = {}
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_leaf_fields(v, path))
        elif isinstance(v, list):
            # Flatten list of scalars; skip nested lists of dicts (too noisy)
            for item in v:
                if not isinstance(item, (dict, list)):
                    out.setdefault(path, [])
                    if isinstance(out[path], list):
                        out[path].append(item)
        else:
            out[path] = v
    return out


def _score_co(parsed: dict[str, Any] | None, expected_result: Any) -> float:
    """CorrectOutputFuzzy: order-insensitive, numbers within 1%.

    Strategy by result type:
    - Count queries (expected is a number): actual count within 1%.
    - Single-row aggregation (expected is a 1-element list): compare all
      leaf field values fuzzy (F1 ≥ 0.8).
    - Multi-row list queries (expected has >1 element): compare row count
      within 20% AND field-name overlap ≥ 0.7.  We don't compare exact
      document values because any valid subset is a correct answer.
    """
    if parsed is None or expected_result is None:
        return 0.0

    rows: list = parsed.get("rows", [])
    row_count: int = parsed.get("row_count", len(rows))

    # --- Count queries ---
    if isinstance(expected_result, (int, float)):
        if len(rows) == 1 and "count" in rows[0]:
            actual_n = rows[0]["count"]
            exp_n = expected_result
            if exp_n == 0:
                return 1.0 if actual_n == 0 else 0.0
            return 1.0 if _nums_close(actual_n, exp_n) else 0.0
        return 0.0

    if not isinstance(expected_result, list) or not expected_result:
        return 0.0

    exp_count = len(expected_result)

    # --- Single-row aggregation (e.g. average, max, total) ---
    if exp_count == 1 and isinstance(expected_result[0], dict):
        if not rows:
            return 0.0
        actual_leaf = _leaf_fields(rows[0])
        expected_leaf = _leaf_fields(expected_result[0])
        actual_vals = [v for v in actual_leaf.values() if v is not None]
        expected_vals = [v for v in expected_leaf.values() if v is not None]
        f1 = _fuzzy_f1(actual_vals, expected_vals)
        return 1.0 if f1 >= 0.8 else 0.0

    # --- Multi-row list queries ---
    # 1. Row count within 20 %
    if exp_count == 0:
        return 1.0 if row_count == 0 else 0.0
    count_ratio = abs(row_count - exp_count) / exp_count
    if count_ratio > 0.20:
        return 0.0

    # 2. Field-name overlap between actual and reference documents
    if rows and isinstance(rows[0], dict) and isinstance(expected_result[0], dict):
        actual_fields = set(_leaf_fields(rows[0]).keys())
        expected_fields = set(_leaf_fields(expected_result[0]).keys())
        if expected_fields:
            overlap = len(actual_fields & expected_fields) / len(expected_fields)
            return 1.0 if overlap >= 0.7 else 0.0

    return 1.0


def _score(
    *,
    last_mql_result: str | None,
    expected_result: Any,
    tool_calls_made: list[str],
    timed_out: bool,
    error_detail: str,
) -> dict[str, float]:
    """Compute XMaNeR metrics for one question."""
    zeros = dict(
        successful_execution=0.0,
        non_empty_output=0.0,
        reasonable_output=0.0,
        correct_output_fuzzy=0.0,
        xmaner=0.0,
    )

    if timed_out or error_detail:
        return zeros

    # SE: run_mql was called at least once with a non-error result
    parsed = _parse_mql_result(last_mql_result) if last_mql_result else None
    se = 1.0 if (parsed is not None and "run_mql" in tool_calls_made) else 0.0

    if se == 0.0:
        return zeros

    rows: list = parsed.get("rows", [])  # type: ignore[union-attr]
    row_count: int = parsed.get("row_count", len(rows))

    # NEO: non-empty result
    neo = 1.0 if row_count > 0 else 0.0

    # RO: fewer than 50% of non-_id leaf values are null/empty.
    # Atlas documents legitimately have some null optional fields, so we use
    # a tolerance rather than a strict zero-null requirement.
    ro = 0.0
    if neo:
        leaf_vals = [
            v for path, v in _leaf_fields(rows[0] if rows else {}).items()
            if not path.startswith("_id")
        ] if rows and isinstance(rows[0], dict) else _flatten_values(rows)
        if leaf_vals:
            null_ratio = sum(1 for v in leaf_vals if v is None or v == "") / len(leaf_vals)
            ro = 1.0 if null_ratio < 0.5 else 0.0
        else:
            ro = 1.0

    # CO: fuzzy match vs reference
    co = _score_co(parsed, expected_result)

    xmaner = (se + neo + ro + co) / 4

    return dict(
        successful_execution=se,
        non_empty_output=neo,
        reasonable_output=ro,
        correct_output_fuzzy=co,
        xmaner=xmaner,
    )


# ---------------------------------------------------------------------------
# Per-question async runner
# ---------------------------------------------------------------------------


async def _run_question(
    session: MangoAgent,
    item: dict[str, Any],
    timeout: float,
    model_label: str,
) -> dict[str, Any]:
    """Ask one question and return a result row dict."""
    last_mql_result: str | None = None
    mql_was_success: bool = False

    def _on_tool_call(tool_name: str, _tool_args: dict, result_text: str) -> None:
        nonlocal last_mql_result, mql_was_success
        if tool_name == "run_mql":
            last_mql_result = result_text
            mql_was_success = _parse_mql_result(result_text) is not None

    t0 = time.perf_counter()
    error_detail = ""
    timed_out = False
    answer = ""
    tool_calls_made: list[str] = []
    input_tokens = output_tokens = iterations = memory_hits = retries = 0

    try:
        response = await asyncio.wait_for(
            session.ask(item["nl_query"], on_tool_call=_on_tool_call),
            timeout=timeout,
        )
        answer = response.answer
        tool_calls_made = response.tool_calls_made
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
        iterations = response.iterations
        memory_hits = response.memory_hits
        retries = response.retries_made
    except asyncio.TimeoutError:
        timed_out = True
        error_detail = f"timeout after {timeout:.0f}s"
    except Exception as exc:
        error_detail = str(exc)

    latency = time.perf_counter() - t0

    mql_for_scoring = last_mql_result if mql_was_success else None

    metrics = _score(
        last_mql_result=mql_for_scoring,
        expected_result=item["expected_result"],
        tool_calls_made=tool_calls_made,
        timed_out=timed_out,
        error_detail=error_detail,
    )

    ref_output = ""
    if item["expected_result"] is not None:
        ref_output = json.dumps(item["expected_result"], default=str, ensure_ascii=False)

    return {
        "model": model_label,
        "question_id": item["_idx"],
        "database": item["db"],
        "natural_language_query": item["nl_query"],
        "mango_result": last_mql_result or error_detail or "",
        "reference_output": ref_output,
        **metrics,
        "token_input": input_tokens,
        "token_output": output_tokens,
        "latency_seconds": round(latency, 3),
        "tool_calls": ",".join(tool_calls_made),
        "iterations": iterations,
        "memory_hits": memory_hits,
        "retries": retries,
        "error_detail": error_detail,
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


async def run_benchmark(
    *,
    mongo_uri: str,
    models: list[dict],
    sample_size: int | None,
    timeout: float,
    memory_enabled: bool,
    max_iterations: int,
    results_dir: str,
    tags_filter: list[str] | None,
    dry_run: bool,
    verbose: bool,
    csv_path: str | None = None,
    mongo_kwargs: dict | None = None,
    training_file: str | None = None,
) -> list[dict[str, Any]]:
    """Run the full benchmark and return all result rows."""
    # --- Load dataset ---
    dataset = load_dataset(csv_path=csv_path, tags_filter=tags_filter, limit=sample_size)
    print(f"Dataset: {len(dataset)} questions loaded")

    if dry_run:
        dbs = sorted({item["db"] for item in dataset})
        print(f"Dry run — {len(dataset)} questions across {len(dbs)} databases:")
        for db in dbs:
            n = sum(1 for item in dataset if item["db"] == db)
            print(f"  {db}: {n} questions")
        return []

    # --- Setup output ---
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"results_{ts}.csv"
    jsonl_path = out_dir / f"results_{ts}_debug.jsonl"

    # --- Progress indicator ---
    try:
        from tqdm import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    # Group dataset by database (schema introspection happens once per db/model)
    db_groups: dict[str, list[dict]] = {}
    for item in dataset:
        db_groups.setdefault(item["db"], []).append(item)

    all_rows: list[dict[str, Any]] = []
    memory_dir = str(out_dir / ".benchmark_memory")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_f, \
         open(jsonl_path, "w", encoding="utf-8") as jsonl_f:

        writer = csv.DictWriter(csv_f, fieldnames=_COLUMNS)
        writer.writeheader()

        for model_cfg in models:
            provider = model_cfg["provider"]
            model = model_cfg.get("model")
            api_key = model_cfg.get("api_key")
            training_tag = " [+training]" if training_file else ""
            model_label = f"{model or provider} ({provider}){training_tag}"

            print(f"\n{'='*60}")
            print(f"Model: {model_label}")
            print(f"{'='*60}")

            model_rows: list[dict] = []

            for db_name, items in db_groups.items():
                print(f"\n  DB: {db_name} ({len(items)} questions)")

                try:
                    base_agent = _build_agent(
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        db_name=db_name,
                        mongo_uri=mongo_uri,
                        memory_enabled=memory_enabled,
                        memory_base_dir=memory_dir,
                        max_iterations=max_iterations,
                        training_file=training_file,
                        mongo_kwargs=mongo_kwargs,
                    )
                    if training_file and base_agent.agent_memory is not None:
                        from mango.servers.cli.main import _load_training_file
                        print(f"  Loading training data from {training_file}…")
                        await _load_training_file(base_agent.agent_memory, training_file)
                except Exception as exc:
                    print(f"  ERROR building agent for {db_name}: {exc}")
                    for item in items:
                        row = {
                            "model": model_label,
                            "question_id": item["_idx"],
                            "database": db_name,
                            "natural_language_query": item["nl_query"],
                            "mango_result": "",
                            "reference_output": "",
                            "successful_execution": 0.0,
                            "non_empty_output": 0.0,
                            "reasonable_output": 0.0,
                            "correct_output_fuzzy": 0.0,
                            "xmaner": 0.0,
                            "token_input": 0,
                            "token_output": 0,
                            "latency_seconds": 0.0,
                            "tool_calls": "",
                            "iterations": 0,
                            "memory_hits": 0,
                            "retries": 0,
                            "error_detail": f"agent init failed: {exc}",
                        }
                        model_rows.append(row)
                        writer.writerow(row)
                        jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        jsonl_f.flush()
                    continue

                iterator = enumerate(items, 1)
                if _has_tqdm:
                    from tqdm import tqdm
                    iterator = enumerate(tqdm(items, desc=f"    {db_name}", leave=False), 1)

                for i, item in iterator:
                    if not _has_tqdm:
                        print(f"    [{i}/{len(items)}] {item['nl_query'][:70]}", end="\r")

                    session = base_agent.new_session()
                    row = await _run_question(session, item, timeout, model_label)

                    model_rows.append(row)
                    all_rows.append(row)
                    writer.writerow(row)
                    csv_f.flush()
                    jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    jsonl_f.flush()

                    if verbose:
                        print(
                            f"\n    Q{item['_idx']}: xmaner={row['xmaner']:.2f} "
                            f"se={row['successful_execution']:.0f} "
                            f"neo={row['non_empty_output']:.0f} "
                            f"ro={row['reasonable_output']:.0f} "
                            f"co={row['correct_output_fuzzy']:.0f} "
                            f"lat={row['latency_seconds']}s"
                        )

            # Print model summary
            if model_rows:
                _print_model_summary(model_label, model_rows)

    print(f"\nResults saved to: {csv_path}")
    print(f"Debug log:        {jsonl_path}")
    return all_rows


def _print_model_summary(label: str, rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        return
    avg = lambda key: sum(r[key] for r in rows) / n  # noqa: E731
    print(f"\n  Summary — {label}")
    print(f"  Questions : {n}")
    print(f"  XMaNeR    : {avg('xmaner'):.4f}")
    print(f"  SE        : {avg('successful_execution'):.4f}")
    print(f"  NEO       : {avg('non_empty_output'):.4f}")
    print(f"  RO        : {avg('reasonable_output'):.4f}")
    print(f"  CO        : {avg('correct_output_fuzzy'):.4f}")
    print(f"  Avg latency: {avg('latency_seconds'):.2f}s")
    print(f"  Avg tokens: {avg('token_input'):.0f} in / {avg('token_output'):.0f} out")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_model_spec(spec: str) -> dict:
    """Parse 'provider:model' or 'provider' into a model config dict."""
    parts = spec.split(":", 1)
    provider = parts[0]
    model = parts[1] if len(parts) > 1 else None
    return {"provider": provider, "model": model, "api_key": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MangoAgent against the NL-to-MQL benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGODB_URI", ""),
        help="MongoDB connection URI (overrides MONGODB_URI env var).",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("MANGO_PROVIDER", "gemini"),
        help="LLM provider for single-model run (anthropic/openai/gemini).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MANGO_MODEL", "gemini-3.1-pro-preview"),
        help="Model ID for single-model run.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        metavar="PROVIDER[:MODEL]",
        help=(
            "One or more provider:model specs for multi-model comparison. "
            "Overrides config.py MODELS. Example: anthropic:claude-sonnet-4-6 openai:gpt-4o"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=_cfg.SAMPLE_SIZE if _cfg.SAMPLE_SIZE else 0,
        metavar="N",
        help="Number of questions to run. 0 = all 766.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_cfg.TIMEOUT_SECONDS,
        help="Per-question timeout in seconds.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=_cfg.MAX_ITERATIONS,
        help="Max agent tool-call iterations per question.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable ChromaDB memory layer.",
    )
    parser.add_argument(
        "--training-file",
        default=_cfg.TRAINING_FILE,
        metavar="PATH",
        help="JSONL training file to pre-load before the benchmark (ablation: with vs without training).",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        metavar="TAG",
        help="Only include questions whose tags field contains one of these strings.",
    )
    parser.add_argument(
        "--results-dir",
        default=_cfg.RESULTS_DIR,
        help="Directory to write result CSV and debug JSONL.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        metavar="PATH",
        help="Path to benchmark CSV. Defaults to the bundled Atlas dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the dataset without connecting to MongoDB.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-question scores.",
    )
    parser.add_argument(
        "--tls-insecure",
        action="store_true",
        help=(
            "Pass tlsInsecure=True to MongoClient. "
            "Useful when Atlas returns TLSV1_ALERT_INTERNAL_ERROR due to "
            "OpenSSL incompatibility. Note: disables certificate verification."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    # Resolve model list
    if args.providers:
        models = [_parse_model_spec(s) for s in args.providers]
    elif args.provider:
        models = [{"provider": args.provider, "model": args.model, "api_key": None}]
    else:
        models = _cfg.MODELS

    mongo_uri = args.mongo_uri
    if not mongo_uri and not args.dry_run:
        parser.error("--mongo-uri or MONGODB_URI env var is required (unless --dry-run).")

    sample_size = args.sample if args.sample > 0 else None

    mongo_kwargs = {"tlsInsecure": True} if args.tls_insecure else None

    asyncio.run(
        run_benchmark(
            mongo_uri=mongo_uri,
            models=models,
            sample_size=sample_size,
            timeout=args.timeout,
            memory_enabled=not args.no_memory,
            max_iterations=args.max_iterations,
            results_dir=args.results_dir,
            tags_filter=args.tags,
            dry_run=args.dry_run,
            verbose=args.verbose,
            csv_path=args.csv_path,
            mongo_kwargs=mongo_kwargs,
            training_file=args.training_file,
        )
    )


if __name__ == "__main__":
    main()
