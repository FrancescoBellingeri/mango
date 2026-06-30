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
import re
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
    InspectFieldTool,
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
    "tool_arg_valid",
    "output_accuracy",
    "run_mql_calls",
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
    base_url: str | None = None,
    db_name: str,
    mongo_uri: str,
    memory_enabled: bool,
    memory_base_dir: str,
    max_iterations: int,
    training_file: str | None = None,
    mongo_kwargs: dict | None = None,
    max_rows: int = 100_000,
    temperature: float | None = None,
) -> MangoAgent:
    """Build and return a ready MangoAgent connected to *db_name*.

    ``max_rows`` raises RunMQLTool's row cap (production default 100). For
    benchmark runs it must be high enough that the agent's result-set is complete:
    a 100-row cap silently truncates large results and turns a correct query into
    a false FAIL on the execution-accuracy metric. The gold side is uncapped, so
    capping only the agent side is a one-sided truncation.
    """
    llm = build_llm(
        provider=provider, model=model, api_key=api_key, base_url=base_url,
        temperature=temperature,
    )

    db = MongoRunner()
    db.connect(_uri_for_db(mongo_uri, db_name), **(mongo_kwargs or {}))

    tools = ToolRegistry()
    tools.register(ListCollectionsTool(db))
    tools.register(SearchCollectionsTool(db))
    tools.register(DescribeCollectionTool(db))
    tools.register(InspectFieldTool(db))
    tools.register(CollectionStatsTool(db))

    agent_memory = None
    if memory_enabled:
        from mango.integrations.chromadb import ChromaAgentMemory
        from mango.tools import SaveTextMemoryTool

        safe_label = (model or provider).replace("/", "_").replace(":", "_")
        training_tag = "_trained" if training_file else ""
        persist_dir = os.path.join(memory_base_dir, f"{safe_label}_{db_name}{training_tag}")
        agent_memory = ChromaAgentMemory(
            collection_name="benchmark_memory",
            persist_dir=persist_dir,
        )

        tools.register(SaveTextMemoryTool(agent_memory))

    tools.register(RunMQLTool(db, max_rows=max_rows))

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
    """Parse a RunMQLTool result_text into a Python dict, or None on failure.

    The agent loop may prepend an ``[AUTO-SCHEMA for '<col>']\\n…\\n\\n`` block to
    the run_mql result when the schema is large, so the actual ``{"rows":…}`` JSON
    is not at offset 0. We therefore extract the trailing result object rather
    than parsing the whole string (the previous version returned None on every
    auto-schema-prefixed result, silently corrupting success detection).
    """
    if not result_text:
        return None
    # Error or retry messages injected by the agent loop
    if result_text.startswith(("ERROR:", "[RETRY", "[FATAL]", "[MAX RETRIES")):
        return None
    try:
        return json.loads(result_text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Strip any AUTO-SCHEMA prefix: the run_mql payload is the final {"rows":…}.
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{\s*"rows"', result_text):
        try:
            obj, _ = decoder.raw_decode(result_text[m.start():])
            return obj
        except (json.JSONDecodeError, ValueError):
            continue
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
    - Single-row aggregation (expected is a 1-element list): all expected
      leaf values present in actual values (recall ≥ 0.8). Extra fields in
      the actual row are ignored — a more descriptive field name is correct.
    - Multi-row list queries (expected has >1 element): row count within 20%
      AND value overlap of the first row ≥ 0.7. Field names are ignored
      because the LLM legitimately uses more descriptive aliases than the
      reference (e.g. "customer_count" vs "count").
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
        if not expected_vals:
            return 1.0
        # Use recall: are all expected values present in actual?
        # Extra fields in actual (more descriptive aliases) do not penalise.
        matched = 0
        used: set[int] = set()
        for exp_v in expected_vals:
            for i, act_v in enumerate(actual_vals):
                if i not in used and _values_match(exp_v, act_v):
                    matched += 1
                    used.add(i)
                    break
        recall = matched / len(expected_vals)
        return 1.0 if recall >= 0.8 else 0.0

    # --- Multi-row list queries ---
    # 1. Row count within 20 %
    if exp_count == 0:
        return 1.0 if row_count == 0 else 0.0
    count_ratio = abs(row_count - exp_count) / exp_count
    if count_ratio > 0.20:
        return 0.0

    # 2. Value overlap of first row (field-name-agnostic).
    # Compares the set of leaf values rather than field names so that
    # descriptive aliases ("customer_count") score the same as generic
    # ones ("count") when the underlying data is identical.
    if rows and isinstance(rows[0], dict) and isinstance(expected_result[0], dict):
        actual_vals = [v for v in _leaf_fields(rows[0]).values() if v is not None]
        expected_vals = [v for v in _leaf_fields(expected_result[0]).values() if v is not None]
        if expected_vals:
            f1 = _fuzzy_f1(actual_vals, expected_vals)
            return 1.0 if f1 >= 0.7 else 0.0

    return 1.0


def _score(
    *,
    last_mql_result: str | None,
    expected_result: Any,
    tool_calls_made: list[str],
    timed_out: bool,
    error_detail: str,
    run_mql_first_success: bool = False,
) -> dict[str, float]:
    """Compute XMaNeR metrics for one question.

    New metrics (inspired by the structured-output reliability paper):
    - tool_arg_valid (TAV): first run_mql call succeeded without retry/error.
    - output_accuracy (OA): TAV × CO — joint event of format compliance + correct answer.
    """
    zeros = dict(
        successful_execution=0.0,
        non_empty_output=0.0,
        reasonable_output=0.0,
        correct_output_fuzzy=0.0,
        xmaner=0.0,
        tool_arg_valid=0.0,
        output_accuracy=0.0,
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

    # TAV: first run_mql call succeeded without needing a retry.
    tav = 1.0 if run_mql_first_success else 0.0
    # OA: joint event — format-compliant on first try AND correct answer.
    oa = tav * co

    return dict(
        successful_execution=se,
        non_empty_output=neo,
        reasonable_output=ro,
        correct_output_fuzzy=co,
        xmaner=xmaner,
        tool_arg_valid=tav,
        output_accuracy=oa,
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
    run_mql_call_count: int = 0
    run_mql_first_success: bool = False
    # Every run_mql invocation, in call order: its generated QueryRequest args and
    # its parsed result. The scorer needs ALL calls, not just the last: a
    # multi-call agent may run the answer-bearing query (e.g. a count) and then a
    # supplementary one (e.g. a breakdown), so "last result" is not "the answer".
    # execution_accuracy = does ANY successful call's result-set equal gold.
    # Args also drive structural validity, read-only safety and tool correctness.
    agent_mql_calls: list[dict[str, Any]] = []
    agent_mql_results: list[dict[str, Any]] = []

    def _on_tool_call(tool_name: str, tool_args: dict, result_text: str) -> None:
        nonlocal last_mql_result, mql_was_success, run_mql_call_count, run_mql_first_success
        if tool_name == "run_mql":
            run_mql_call_count += 1
            parsed = _parse_mql_result(result_text)
            is_success = parsed is not None
            agent_mql_calls.append(dict(tool_args))
            agent_mql_results.append({"success": is_success, "result": parsed})
            if run_mql_call_count == 1 and is_success:
                run_mql_first_success = True
            last_mql_result = result_text
            mql_was_success = is_success

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
        run_mql_first_success=run_mql_first_success,
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
        # --- New harness inputs (scored offline by scorecard.py) ---
        "nl_answer": answer,                       # final NL answer (was discarded)
        "agent_mql": agent_mql_calls,              # agent's generated QueryRequest(s)
        "agent_results": agent_mql_results,        # parsed result-set per run_mql call
        "gold_mql": item.get("gold_mql"),          # full structured gold query
        **metrics,
        "run_mql_calls": run_mql_call_count,
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
    max_rows: int = 100_000,
    temperature: float | None = None,
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
    # Name files after model + database + timestamp for quick identification.
    # ("multi" / "multi-db" when several models or databases share one run.)
    if len(models) == 1:
        raw = models[0].get("model") or models[0]["provider"]
    else:
        raw = "multi"
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw)).strip("-") or "model"
    dbs = sorted({item["db"] for item in dataset})
    if len(dbs) == 1:
        db_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", dbs[0]).strip("-") or "db"
    else:
        db_slug = "multi-db"
    out_csv_path = out_dir / f"{model_slug}_{db_slug}_{ts}.csv"
    out_jsonl_path = out_dir / f"{model_slug}_{db_slug}_{ts}_debug.jsonl"

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

    with open(out_csv_path, "w", newline="", encoding="utf-8") as csv_f, \
         open(out_jsonl_path, "w", encoding="utf-8") as jsonl_f:

        # extrasaction="ignore": the new harness fields (nl_answer, agent_mql,
        # gold_mql) are debug-only and go to the JSONL, not the lean human CSV.
        writer = csv.DictWriter(csv_f, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for model_cfg in models:
            provider = model_cfg["provider"]
            model = model_cfg.get("model")
            api_key = model_cfg.get("api_key")
            base_url = model_cfg.get("base_url")
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
                        base_url=base_url,
                        db_name=db_name,
                        mongo_uri=mongo_uri,
                        memory_enabled=memory_enabled,
                        memory_base_dir=memory_dir,
                        max_iterations=max_iterations,
                        training_file=training_file,
                        mongo_kwargs=mongo_kwargs,
                        max_rows=max_rows,
                        temperature=temperature,
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
                        err = f" ERR={row['error_detail']}" if row["error_detail"] else ""
                        print(
                            f"\n    Q{item['_idx']}: xmaner={row['xmaner']:.2f} "
                            f"se={row['successful_execution']:.0f} "
                            f"neo={row['non_empty_output']:.0f} "
                            f"ro={row['reasonable_output']:.0f} "
                            f"co={row['correct_output_fuzzy']:.0f} "
                            f"lat={row['latency_seconds']}s{err}"
                        )

            # Print model summary
            if model_rows:
                _print_model_summary(model_label, model_rows)

    print(f"\nResults saved to: {out_csv_path}")
    print(f"Debug log:        {out_jsonl_path}")
    return all_rows


def _print_model_summary(label: str, rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        return
    avg = lambda key: sum(r[key] for r in rows) / n  # noqa: E731
    se_rows = [r for r in rows if r["successful_execution"] == 1.0]
    pta = sum(r["correct_output_fuzzy"] for r in se_rows) / len(se_rows) if se_rows else 0.0
    tav = avg("tool_arg_valid")
    oa = avg("output_accuracy")
    gap = pta - oa
    print(f"\n  Summary — {label}")
    print(f"  Questions : {n}")
    print(f"  XMaNeR    : {avg('xmaner'):.4f}")
    print(f"  SE        : {avg('successful_execution'):.4f}")
    print(f"  NEO       : {avg('non_empty_output'):.4f}")
    print(f"  RO        : {avg('reasonable_output'):.4f}")
    print(f"  CO        : {avg('correct_output_fuzzy'):.4f}")
    print(f"  --- output accuracy (paper-inspired) ---")
    print(f"  TAV       : {tav:.4f}  (first run_mql call OK, no retry needed)")
    print(f"  OA        : {oa:.4f}  (TAV × CO — format + correttezza)")
    print(f"  PTA       : {pta:.4f}  (CO | SE=1 — tetto se formato fosse perfetto)")
    print(f"  Gap PTA-OA: {gap:.4f}  (accuratezza persa per problemi di formato)")
    print(f"  Avg MQL calls: {avg('run_mql_calls'):.2f}")
    print(f"  Avg latency: {avg('latency_seconds'):.2f}s")
    print(f"  Avg tokens: {avg('token_input'):.0f} in / {avg('token_output'):.0f} out")
    error_rows = [r for r in rows if r["error_detail"]]
    if error_rows:
        print(f"  Errors: {len(error_rows)}/{n}")
        for r in error_rows[:5]:
            print(f"    Q{r['question_id']}: {r['error_detail']}")


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
        "--max-rows",
        type=int,
        default=100_000,
        help="RunMQLTool row cap for benchmark runs (default 100000, effectively "
             "uncapped). Prevents one-sided truncation against the uncapped gold "
             "result-set, which would false-FAIL large-result questions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="T",
        help="Sampling temperature. Omit to use the provider default; set 0 for "
             "(near-)deterministic ablation runs where the only behavioural delta "
             "between two configs is the feature under test.",
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
        "--api-key",
        default=os.getenv("REGOLO_API_KEY"),
        metavar="KEY",
        help=(
            "API key for the LLM provider. Overrides provider-specific env vars "
            "(ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, etc.)."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help=(
            "Base URL for OpenAI-compatible providers (e.g. Together AI: https://api.together.xyz/v1). "
            "Required when using --provider openai with a non-gpt model."
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
        for m in models:
            if args.api_key:
                m["api_key"] = args.api_key
            if args.base_url:
                m["base_url"] = args.base_url
    elif args.provider:
        models = [{"provider": args.provider, "model": args.model, "api_key": args.api_key, "base_url": args.base_url}]
    else:
        models = _cfg.MODELS
        for m in models:
            if args.api_key:
                m["api_key"] = args.api_key
            if args.base_url:
                m["base_url"] = args.base_url

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
            max_rows=args.max_rows,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
