"""Validation gates for the benchmark dataset (DATASET_DESIGN.md §6).

Hard-fails (non-zero exit) on any violation: a case that does not *earn* its
category label is rejected, never demoted (§6.2).

Usage:
    python -m mango_benchmark.bench_datasets.validate
    python -m mango_benchmark.bench_datasets.validate --only LOG-SORT-001
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter
from typing import Any

# equivalence.py's multiset row-matching (_max_matching) is a recursive DFS
# augmenting-path search — a real scaling gap for any gold/agent result-set of
# a few hundred rows (a genuine finding for scorecard maintainers, out of
# scope to fix here since equivalence.py is locked). Raised defensively so
# validate.py itself doesn't crash on the dataset's larger legitimate cohorts
# (e.g. EDGE-SCALE); the production runner will hit the same wall on any
# large agent result-set.
sys.setrecursionlimit(20_000)

from pymongo import MongoClient

from mango_benchmark.bench_datasets.build import load_cases
from mango_benchmark.bench_datasets.common import (
    CORE_CATEGORIES,
    DOMAIN_SPREAD_MAX_SHARE,
    BenchCase,
    GoldMQL,
    derive_grouped,
    derive_order_sensitive,
    execute_gold,
    resolve_base,
)
from mango_benchmark.bench_datasets.probes import passing_probes
from mango_benchmark.equivalence import compare, sort_spec
from mango_benchmark.scorecard import read_only_safe, structurally_valid

TOL_FAIL_LOW, TOL_FAIL_HIGH = 1e-6, 5e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flip_boundaries(obj: Any) -> Any:
    """Swap $gte<->$gt and $lte<->$lt everywhere (CAP-TIME §6.2 mutation)."""
    swap = {"$gte": "$gt", "$gt": "$gte", "$lte": "$lt", "$lt": "$lte"}
    if isinstance(obj, dict):
        return {swap.get(k, k): _flip_boundaries(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_flip_boundaries(v) for v in obj]
    return obj


def _flipped_gold(gold: GoldMQL) -> GoldMQL:
    g = copy.deepcopy(gold)
    if g.filter is not None:
        g.filter = _flip_boundaries(g.filter)
    if g.pipeline is not None:
        g.pipeline = [
            {"$match": _flip_boundaries(stage["$match"])} if "$match" in stage else stage
            for stage in g.pipeline
        ]
    return g


def _without_limit(gold: GoldMQL) -> tuple[GoldMQL, int | None]:
    """Strip the row cut (find limit / trailing $limit) for tie inspection."""
    g = copy.deepcopy(gold)
    cut: int | None = None
    if g.operation == "find":
        cut, g.limit = g.limit, None
    elif g.pipeline and "$limit" in g.pipeline[-1]:
        cut = g.pipeline[-1]["$limit"]
        g.pipeline = g.pipeline[:-1]
    return g, cut


def _toggle_unwind_preserve(gold: GoldMQL) -> GoldMQL:
    """Flip preserveNullAndEmptyArrays on the first $unwind stage (§6.2)."""
    g = copy.deepcopy(gold)
    for i, stage in enumerate(g.pipeline or []):
        if "$unwind" in stage:
            u = stage["$unwind"]
            if isinstance(u, str):
                path, current = u, False
            else:
                path, current = u["path"], u.get("preserveNullAndEmptyArrays", False)
            g.pipeline[i] = {"$unwind": {"path": path, "preserveNullAndEmptyArrays": not current}}
            return g
    raise ValueError("no $unwind stage in pipeline")


def _fast_multiset_key(row: Any) -> Any:
    """Hashable canonical key for a row, used only to fast-path the
    determinism check (§6.1). ``equivalence.compare`` is O(V^2) deep-equality
    matching with no bucketing — fine for small result-sets, but several
    realistic capability filters here return thousands of full nested
    documents (e.g. LOG-FILTER-003's ~7200-row priority filter), where a
    naive determinism re-check costs minutes. A repeated run of the exact
    same gold query is checking for LITERAL multiset identity, not semantic
    equivalence (no tolerance, no _id reconciliation needed) — so a plain
    hashable-repr multiset compare is a correct, much cheaper substitute for
    this one purpose. Falls back to the real `compare()` (still called
    downstream via other checks) whenever a row isn't hashable this way.
    """
    def freeze(x):
        if isinstance(x, dict):
            return tuple(sorted((k, freeze(v)) for k, v in x.items()))
        if isinstance(x, list):
            return tuple(freeze(v) for v in x)
        return x
    return freeze(row)


def _fast_multiset_equal(a: Any, b: Any) -> bool | None:
    """True/False if a fast multiset-identity check is conclusive; None to
    signal 'fall back to compare()' (non-list results, unhashable rows)."""
    if not (isinstance(a, list) and isinstance(b, list)):
        return None
    try:
        ca = Counter(_fast_multiset_key(r) for r in a)
        cb = Counter(_fast_multiset_key(r) for r in b)
    except TypeError:
        return None
    return ca == cb


def _sort_keys(gold: GoldMQL) -> list[str]:
    spec = sort_spec(gold.to_dict())
    return list(spec.keys()) if spec else []


def _key_tuple(row: Any, keys: list[str]) -> tuple[Any, ...] | None:
    if not isinstance(row, dict):
        return None
    return tuple(row.get(k) for k in keys)


def _as_scalar(result: Any) -> float | None:
    """Collapse a scalar-shaped result (bare number or single-row/single-value
    rowset) to a float, for EDGE-NUM tolerance-window checks."""
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return float(result)
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        vals = [v for k, v in result[0].items() if k != "_id"]
        if len(vals) == 1 and isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
            return float(vals[0])
    return None


_NUM = (int, float)


def _type_class(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, _NUM):
        return "number"
    return type(v).__name__


# ---------------------------------------------------------------------------
# Per-case validation
# ---------------------------------------------------------------------------


def validate_case(case: BenchCase, by_id: dict[str, BenchCase], client: MongoClient) -> list[str]:
    errs: list[str] = []
    base = resolve_base(case, by_id)
    gold = base.gold
    db = client[base.database]

    # --- behavioral shape ---------------------------------------------------
    if case.category == "SEC-TRAP" and not case.safety_trap:
        errs.append("SEC-TRAP case must set safety_trap")
    if case.safety_trap and case.category != "SEC-TRAP":
        errs.append("safety_trap set outside SEC-TRAP")
    if case.category == "OOS" and case.expected_behavior != "refuse":
        errs.append("OOS must be expected_behavior=refuse")
    if case.category == "AMB" and case.expected_behavior != "any_of":
        errs.append("AMB must be expected_behavior=any_of")
    if case.nl_variant_of and case.nl_question.strip() == base.nl_question.strip():
        errs.append("NLR variant NL is identical to its base")

    if gold is None:
        return errs  # refuse cases: nothing executable to validate

    gold_dict = gold.to_dict()

    # --- universal (§6.1) ----------------------------------------------------
    if not structurally_valid(gold_dict):
        errs.append("gold fails offline structural validity")
    if not read_only_safe(gold_dict):
        errs.append("gold fails read-only safety")
    for alt in base.gold_alternatives:
        if not structurally_valid(alt.to_dict()) or not read_only_safe(alt.to_dict()):
            errs.append("gold_alternative fails structural/read-only checks")

    try:
        runs = [execute_gold(db, gold) for _ in range(3)]
    except Exception as exc:
        return errs + [f"gold execution failed: {exc}"]
    gold_result = runs[0]
    # Fast path only when order is not semantically part of the answer: for
    # an order_sensitive gold, determinism means "same rows in the same
    # order", which a multiset check cannot see — those cases fall through
    # to the real compare() (they are also never the large-result-set ones;
    # CAP-SORT/CAP-PAGE cases carry a $limit).
    use_fast = not derive_order_sensitive(gold)
    for i, r in enumerate(runs[1:], 2):
        fast = _fast_multiset_equal(r, gold_result) if use_fast else None
        ok = fast if fast is not None else compare(r, gold_result, gold_mql=gold_dict).equivalent
        if not ok:
            errs.append(f"gold is nondeterministic (run {i} differs)")

    if case.nl_variant_of is None:  # probes are meaningful once per gold
        exempt = frozenset({"empty"}) if case.category == "EDGE-EMPTY" else frozenset()
        for probe in passing_probes(db, gold, gold_result, exempt=exempt):
            errs.append(f"degenerate: trivial probe '{probe}' passes this case")

    # --- tag-driven mutations (§6.2) -----------------------------------------
    if "boundary" in case.tags:
        flipped = execute_gold(db, _flipped_gold(gold))
        if compare(flipped, gold_result, gold_mql=gold_dict).equivalent:
            errs.append("boundary: flipping range inclusivity does not change the result")

    if "unwind_empty" in case.tags:
        mutated = execute_gold(db, _toggle_unwind_preserve(gold))
        if compare(mutated, gold_result, gold_mql=gold_dict).equivalent:
            errs.append("unwind_empty: toggling preserveNullAndEmptyArrays does not change the result")

    if "ties" in case.tags:
        keys = _sort_keys(gold)
        if not keys:
            errs.append("ties: gold has no sort spec")
        else:
            uncut, cut = _without_limit(gold)
            rows = execute_gold(db, uncut)
            kts = [_key_tuple(r, keys) for r in rows]
            window = kts[:cut] if cut else kts
            if not any(a == b for a, b in zip(window, window[1:])):
                errs.append("ties: no tie-run inside the returned window")
            if cut and cut < len(kts) and kts[cut - 1] == kts[cut]:
                errs.append("ties: tie straddles the limit cut (gold nondeterministic)")

    # --- wrong_mql (documented plausible-wrong query) ------------------------
    if case.wrong_mql is not None:
        if not structurally_valid(case.wrong_mql.to_dict()):
            errs.append("wrong_mql fails structural validity")
        try:
            wrong = execute_gold(db, case.wrong_mql)
        except Exception as exc:
            return errs + [f"wrong_mql execution failed: {exc}"]
        cmp = compare(wrong, gold_result, gold_mql=gold_dict)

        if "tol_pass" in case.tags:
            if not cmp.equivalent:
                errs.append("tol_pass: alternative computation is NOT within tolerance")
        elif "tol_fail" in case.tags:
            if cmp.equivalent:
                errs.append("tol_fail: near-miss value passes (tolerance too loose)")
            w, g = _as_scalar(wrong), _as_scalar(gold_result)
            if w is None or g is None or g == 0:
                errs.append("tol_fail: results are not scalar-shaped")
            else:
                rel = abs(w - g) / abs(g)
                if not (TOL_FAIL_LOW < rel <= TOL_FAIL_HIGH):
                    errs.append(f"tol_fail: rel error {rel:.2e} outside (1e-6, 5e-3]")
        elif case.category == "EDGE-EMPTY":
            if not (isinstance(wrong, list) and len(wrong) > 0) and not (
                isinstance(wrong, int) and wrong > 0
            ):
                errs.append("EDGE-EMPTY: relaxed variant (wrong_mql) is also empty")
        elif case.category == "CAP-NEST":
            if cmp.equivalent:
                errs.append("CAP-NEST: naive rewrite is equivalent to gold")
            elif isinstance(wrong, list) and isinstance(gold_result, list):
                if abs(len(wrong) - len(gold_result)) < 2:
                    errs.append("CAP-NEST: naive-vs-gold margin < 2 rows (lucky superset could mask)")
            elif isinstance(wrong, int) and isinstance(gold_result, int):
                if abs(wrong - gold_result) < 2:
                    errs.append("CAP-NEST: naive-vs-gold count margin < 2")
        else:
            if cmp.equivalent:
                errs.append("wrong_mql is equivalent to gold (case has no teeth)")

    # --- category-specific shapes (§6.2) --------------------------------------
    if case.category == "EDGE-EMPTY":
        # A count's earned-empty is the scalar 0, not []; find/aggregate's is
        # []. Both are the correct zero-result shape for their operation (§4.3
        # distinguishes [] from scalar 0 — this is not a case of conflating them,
        # just accepting whichever shape the operation actually produces).
        empty_ok = (gold_result == 0) if gold.operation == "count" else (gold_result == [])
        if not empty_ok:
            errs.append(f"EDGE-EMPTY: gold result is not earned-empty for a {gold.operation} (got {gold_result!r})")
        sib = by_id.get(case.sibling_of or "")
        if sib is None:
            errs.append("EDGE-EMPTY: sibling_of does not resolve")
        else:
            sib_gold = resolve_base(sib, by_id).gold
            if sib_gold is None:
                errs.append("EDGE-EMPTY: sibling has no gold")
            else:
                sib_result = execute_gold(client[sib.database], sib_gold)
                if isinstance(sib_result, list) and not sib_result:
                    errs.append("EDGE-EMPTY: sibling result is empty too")

    if case.category == "EDGE-SCALE" and "scale" in case.tags:
        if not isinstance(gold_result, list) or len(gold_result) <= 100:
            errs.append("EDGE-SCALE: result does not exceed 100 rows")

    if case.category == "AMB":
        alt_results = [execute_gold(db, a) for a in base.gold_alternatives]
        distinct = False
        for i in range(len(alt_results)):
            for j in range(i + 1, len(alt_results)):
                if not compare(
                    alt_results[i], alt_results[j], gold_mql=base.gold_alternatives[j].to_dict()
                ).equivalent:
                    distinct = True
        if not distinct:
            errs.append("AMB: all alternatives yield equivalent results (ambiguity is moot)")

    if derive_grouped(gold):
        if "having" not in case.tags and isinstance(gold_result, list) and len(gold_result) < 2:
            errs.append("grouped gold returns <2 groups")
        if case.category == "CAP-GROUP" and not case.swap_waiver:
            for row in gold_result if isinstance(gold_result, list) else []:
                classes = Counter(
                    _type_class(v) for k, v in row.items() if k != "_id" and v is not None
                )
                if any(n >= 2 for n in classes.values()):
                    errs.append(
                        "CAP-GROUP: two same-typed non-key fields in a gold row "
                        "(intra-row swap exposure) — waive explicitly or redesign"
                    )
                    break

    return errs


# ---------------------------------------------------------------------------
# Whole-set checks
# ---------------------------------------------------------------------------


def validate_set(cases: list[BenchCase]) -> list[str]:
    errs: list[str] = []

    nls = Counter(c.nl_question.strip().lower() for c in cases)
    for nl, n in nls.items():
        if n > 1:
            errs.append(f"duplicate nl_question x{n}: {nl[:60]!r}")

    for cat in sorted(CORE_CATEGORIES):
        rows = [c for c in cases if c.category == cat]
        if not rows:
            continue
        by_db = Counter(c.database for c in rows)
        if len(by_db) < 2:
            errs.append(f"{cat}: spans only {len(by_db)} database (§2.6 needs ≥2)")
        top_share = max(by_db.values()) / len(rows)
        if top_share > DOMAIN_SPREAD_MAX_SHARE:
            errs.append(f"{cat}: {top_share:.0%} of cases in one database (max {DOMAIN_SPREAD_MAX_SHARE:.0%})")

    return errs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate bench_datasets cases (§6)")
    parser.add_argument("--uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    parser.add_argument("--only", help="validate a single case id")
    parser.add_argument(
        "--partial", action="store_true",
        help="skip whole-set checks (authoring in progress; per-case gates only)",
    )
    args = parser.parse_args()

    client: MongoClient = MongoClient(args.uri, serverSelectionTimeoutMS=5_000)
    client.admin.command("ping")

    cases = load_cases()
    by_id = {c.id: c for c in cases}
    if args.only:
        cases = [c for c in cases if c.id == args.only]

    n_err = 0
    for case in cases:
        errs = validate_case(case, by_id, client)
        for e in errs:
            print(f"  {case.id}: {e}")
        n_err += len(errs)

    if not args.only and not args.partial:
        set_errs = validate_set(load_cases())
        for e in set_errs:
            print(f"  SET: {e}")
        n_err += len(set_errs)

    print(f"\n{len(cases)} cases validated, {n_err} violations")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
