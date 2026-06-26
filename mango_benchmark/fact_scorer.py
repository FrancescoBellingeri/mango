"""Fact scorer — dataset-agnostic fact-containment check (SECONDARY signal).

This is the demoted presentation-layer scorer (DESIGN.md §3, layer 4): a useful
smoke test but **not** a correctness gate. The primary scorer is
``scorecard.py``, which reuses ``derive_facts``/``check_facts`` from here for the
``presentation_recall`` metric only.

Approach: every debug-log row carries ``reference_output`` (the gold answer). It
*derives* the salient facts from that reference automatically and checks they
appear in the agent's answer — works for ANY database with no per-question
maintenance.

Fact derivation (salience rules):
  * scalar (int/float/str)         -> the value itself
  * empty list                     -> an EMPTY marker (answer must be empty)
  * list of scalars (distinct)     -> element count + every distinct value
  * list of dicts (rows)           -> row count + all scalar leaf values of the
                                       FIRST and LAST row (top/bottom of a ranking)
  * single dict / one-row result   -> all its scalar leaf values

Matching:
  * numbers match within a relative tolerance (default 0.5%), so 2.7 ~ 2.704
  * strings match case-insensitively as whole tokens inside the agent output

Standalone usage (secondary check only):
    python -m mango_benchmark.fact_scorer \\
        --results mango_benchmark/results/<model>_<ts>_debug.jsonl
    python -m mango_benchmark.fact_scorer --results <file> --only 1,5,23 --verbose
    python -m mango_benchmark.fact_scorer --results <file> --min-coverage 0.5
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any

# Maximum scalar facts to extract from a single row (avoid combinatorial blow-up).
_MAX_ROW_FACTS = 8
# Maximum distinct values to require for a distinct-style result.
_MAX_DISTINCT_FACTS = 25

_EMPTY = "<EMPTY>"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_value(raw: Any) -> Any:
    """Best-effort parse of a stringified reference/result into a Python object."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if s == "":
        return None
    # plain int / float
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(s)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return s  # leave as string


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float, str, bool)) or v is None


def _scalar_leaves(obj: Any) -> list[Any]:
    """Collect scalar leaf values from a dict/list, skipping the noisy _id."""
    out: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("_id", "id"):
                continue
            out.extend(_scalar_leaves(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_scalar_leaves(item))
    elif _is_scalar(obj) and obj is not None and obj != "":
        out.append(obj)
    return out


# ---------------------------------------------------------------------------
# Fact derivation
# ---------------------------------------------------------------------------


def derive_facts(expected: Any) -> list[Any]:
    """Derive the salient must-appear facts from a ground-truth result."""
    if _is_scalar(expected):
        return [expected] if expected is not None and expected != "" else []

    if isinstance(expected, dict):
        return _scalar_leaves(expected)[:_MAX_ROW_FACTS]

    if isinstance(expected, list):
        if not expected:
            return [_EMPTY]
        # distinct-style (list of scalars): require the values themselves.
        # (Row/element COUNTS are deliberately NOT required facts — they rarely
        #  appear verbatim in the answer and would spuriously match any number.)
        if all(_is_scalar(x) for x in expected):
            return list(expected[:_MAX_DISTINCT_FACTS])
        # rows (list of dicts): scalar leaves of the FIRST and LAST row.
        facts: list[Any] = list(_scalar_leaves(expected[0])[:_MAX_ROW_FACTS])
        if len(expected) > 1:
            facts.extend(_scalar_leaves(expected[-1])[:_MAX_ROW_FACTS])
        return facts

    return []


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _numbers_in(s: str) -> list[float]:
    out: list[float] = []
    for m in re.findall(r"-?\d+\.?\d*", s):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def _num_match(ref: float, nums: list[float], rel_tol: float) -> bool:
    for n in nums:
        if ref == 0:
            if abs(n) <= 1e-9:
                return True
        elif abs(n - ref) / max(abs(ref), 1e-9) <= rel_tol:
            return True
    return False


def _str_match(tok: str, norm_answer: str) -> bool:
    t = _norm(tok)
    if not t:
        return True
    # whole-token-ish containment
    return re.search(rf"(^|[^a-z0-9]){re.escape(t)}([^a-z0-9]|$)", norm_answer) is not None or t in norm_answer


_ENVELOPE_KEYS = ("result", "results", "rows", "data", "output", "value")


def _payload_empty(answer_obj: Any) -> bool:
    """True when the agent's answer carries no data, unwrapping common envelopes
    like ``{"result": []}`` / ``{"rows": [], "row_count": 0}``."""
    if answer_obj in (None, "", [], {}):
        return True
    obj = answer_obj
    if isinstance(obj, dict):
        for k in _ENVELOPE_KEYS:
            if k in obj:
                v = obj[k]
                return v in (None, "", [], {}, 0)
    return False


@dataclass
class FactResult:
    passed: int = 0
    total: int = 0
    missing: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def coverage(self) -> float:
        return self.passed / self.total if self.total else 1.0


def check_facts(facts: list[Any], answer_obj: Any, rel_tol: float) -> FactResult:
    fr = FactResult(total=len(facts))
    if not facts:
        fr.note = "no derivable facts"
        return fr

    answer_text = _norm(json.dumps(answer_obj, default=str, ensure_ascii=False)) if answer_obj is not None else ""
    answer_nums = _numbers_in(answer_text)
    answer_empty = _payload_empty(answer_obj)

    for fact in facts:
        if fact == _EMPTY:
            ok = answer_empty
        elif isinstance(fact, bool):
            ok = _str_match(str(fact).lower(), answer_text)
        elif isinstance(fact, (int, float)):
            ok = _num_match(float(fact), answer_nums, rel_tol)
        else:
            ok = _str_match(str(fact), answer_text)
        if ok:
            fr.passed += 1
        else:
            fr.missing.append(repr(fact))
    return fr


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def score_file(path: str, only: set[int] | None, verbose: bool, min_cov: float, rel_tol: float) -> int:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rows.sort(key=lambda r: int(r.get("question_id", 0)))

    model = rows[0].get("model", "?") if rows else "?"
    print(f"Fact scorer v2 | model: {model} | {len(rows)} questions"
          f" | tol={rel_tol:.1%} | pass>=cov {min_cov:.0%}\n")
    print(f"{'Q':>4}  {'result':6}  {'facts':8}  note / missing")
    print("-" * 78)

    total_facts = 0
    total_passed = 0
    full_pass = 0
    scored = 0

    for r in rows:
        qid = int(r.get("question_id", 0))
        if only and qid not in only:
            continue
        scored += 1

        expected = _parse_value(r.get("reference_output"))
        answer = _parse_value(r.get("mango_result"))
        facts = derive_facts(expected)
        fr = check_facts(facts, answer, rel_tol)

        total_facts += fr.total
        total_passed += fr.passed
        passed = fr.coverage >= min_cov
        if passed:
            full_pass += 1

        verdict = "PASS" if passed else "FAIL"
        print(f"Q{qid:>3}  {verdict:6}  {fr.passed}/{fr.total:<6}  {fr.note}")
        if fr.missing and not passed:
            print(f"            missing: {', '.join(fr.missing[:8])}")
        if verbose:
            print(f"            expected: {str(expected)[:90]}")
            print(f"            answer  : {str(answer)[:90]}")

    print("-" * 78)
    print(f"Questions passed (cov>={min_cov:.0%}) : {full_pass}/{scored}")
    pct = 100 * total_passed / (total_facts or 1)
    print(f"Fact coverage                  : {total_passed}/{total_facts} ({pct:.1f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Dataset-agnostic fact scorer (v2).")
    ap.add_argument("--results", required=True, help="Path to a results *_debug.jsonl file.")
    ap.add_argument("--only", default=None, help="Comma-separated question ids to score.")
    ap.add_argument("--verbose", action="store_true", help="Print expected vs answer per row.")
    ap.add_argument("--min-coverage", type=float, default=0.999,
                    help="Fraction of facts required to PASS a question (default: all).")
    ap.add_argument("--rel-tol", type=float, default=0.005, help="Relative tolerance for numeric facts.")
    args = ap.parse_args()

    only = {int(x) for x in args.only.split(",")} if args.only else None
    return score_file(args.results, only, args.verbose, args.min_coverage, args.rel_tol)


if __name__ == "__main__":
    sys.exit(main())
