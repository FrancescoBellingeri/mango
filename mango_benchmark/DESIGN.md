# Mango Benchmark — Evaluation Harness Design

Status: **locked for review** (Phase 1). No implementation until sign-off.

This document defines the new primary scorer for mango-bench. It replaces
`fact_scorer.py` as the gate and demotes fact-containment to a labeled
secondary signal.

---

## 1. Why this exists

The current scorer (`fact_scorer.py`) is a recall-only, structure-blind,
fact-containment check. It systematically overstates accuracy:

- **Recall-only / superset-blind** — a missing filter or forgotten `$limit`
  returns extra rows that still contain the gold facts → false PASS.
- **Structure discarded** — value-swaps across rows (right values, wrong
  association) still match.
- **Numeric containment with tolerance** — small-integer facts match any
  number in a row dump.
- **Ranking** — only first/last row checked; ordering ignored.
- **Layer conflation** — query correctness and NL presentation collapsed into
  one number.

The new harness fixes these by scoring **executed result-sets**, not prose, and
by separating three independent layers that are never collapsed into one number.

---

## 2. What the debug log persists (Phase 0 findings)

Per-row debug schema (`*_debug.jsonl`), as currently written by
`runner.py::run_one`:

| Field | Meaning |
|---|---|
| `model`, `question_id`, `database`, `natural_language_query` | identity |
| `mango_result` | last **successful** `run_mql` `result_text` = JSON `{"rows":[...],"row_count":N}`, **capped at 100 rows** |
| `reference_output` | JSON of gold `expected.result` |
| `successful_execution`, `non_empty_output`, `reasonable_output`, `correct_output_fuzzy`, `xmaner`, `tool_arg_valid`, `output_accuracy` | **legacy heuristic scores — to be deprecated (§8)** |
| `run_mql_calls`, `token_input`, `token_output`, `latency_seconds`, `tool_calls` (names only), `iterations`, `memory_hits`, `retries`, `error_detail` | operating envelope |

Critical gaps (drive the runner patch in §7):

- **Agent MQL is discarded.** `_on_tool_call` receives `_tool_args` and throws
  it away; `tool_calls` keeps only tool *names*. → structural validity,
  read-only safety, sort-detection, `$group` detection, and tool-correctness are
  all unscorable from current logs.
- **NL answer is computed but never written.** `mango_result` is the raw rowset,
  not prose → presentation layer has nothing to score.
- **100-row cap** (`RunMQLTool.max_rows = 100`) silently truncates the rowset →
  corrupts the primary metric when gold > 100 rows.

### Ground-truth provenance

Gold is produced by `seed_hard/ground_truth.py`: a **human-authored**
`BenchmarkQuestion` (operation + query/pipeline/sort/limit/distinct) executed
**deterministically against a real populated MongoDB**. So it is
**author-trusted query execution**, not per-row human-audited gold. The gold MQL
*is* serialized into the CSV (`expected.dbQuery`) and loaded by `dataset.py` as
`mongo_query`, but the runner does not currently write it to the debug row.

---

## 3. Three layers — never one number

| # | Layer | Type | LLM? | Gate? |
|---|---|---|---|---|
| 1 | `structural_validity` | binary | no | yes |
| 2 | `read_only_safety` | binary | no | yes |
| 3 | `execution_accuracy` | binary | no | **yes — PRIMARY** |
| — | `row_precision / recall / f1` | float | no | diagnostic only |
| — | `tool_correctness` | binary | no | diagnostic |
| — | operating-envelope (iters/tokens/latency + budget breaches) | mixed | no | diagnostic |
| 4 | `presentation_recall` | float | optional | **never a gate** |

**Aggregate PASS = `execution_accuracy ∧ structural_validity ∧ read_only_safety`.**

- **Layer 1 — structural validity.** Reconstruct `QueryRequest` from the logged
  agent args; run `MQLValidator` (operation allowlist, pipeline-stage allowlist,
  operator allowlist, required-args). Offline mode skips collection/field checks
  that need a live schema; op/stage/operator/read-only checks run with no DB.
  **Coupling note:** this gate is only as complete as `MQLValidator`. A
  false-negative in the validator (rejecting a legitimate stage/operator) becomes
  a false-FAIL here. Accepted under the risk hierarchy — a loud, debuggable FAIL,
  not a silent PASS — but the coupling is acknowledged so validator gaps are
  fixed at the source rather than papered over in the scorer.
- **Layer 2 — read-only safety.** Reject any write or server-side-JS stage:
  `$out, $merge, $where, $function, $accumulator, mapReduce`. Tracked as its own
  metric — it doubles as a security/robustness signal.
- **Layer 3 — execution accuracy (PRIMARY).**
  `equivalent(canonicalize(agent_rows), canonicalize(gold))`. Immune to
  supersets and value-swaps. Emits `row_precision/recall/f1` as multiset
  diagnostics for near-misses.
- **Layer 4 — presentation faithfulness (SECONDARY).** Demoted
  `fact_scorer`, run against the now-logged NL answer, explicitly labeled
  secondary. If an LLM judge is ever added it must be calibrated against human
  gold and must not gate by default.

---

## 4. Canonicalization rules (`canonicalize(result) -> CanonicalForm`)

Locked defaults. `equivalent(a, b)` returns true iff the canonical forms are
equal under these rules.

1. **Envelope unwrap.** `{"rows":[...], "row_count":N}` → the row list.
2. **Scalar reconciliation.** A single-row / single-non-`_id`-field result and a
   bare scalar both canonicalize to that scalar. This makes the agent's
   `{"rows":[{"count":395}]}` comparable to gold `395`.
3. **Empty vs zero.** An empty result-set (`[]`) and a scalar `0` are
   **distinct**. A count of 0 is a valid answer, not "no data".
3b. **`distinct` shape unwrap (Phase 2).** A `distinct` executed via `run_mql`
   wraps each value in a one-key doc (`{"method":"card"}` or a one-key `$group`
   `{"_id":"Anker"}`), while the gold `distinct` is a list of bare scalars.
   Single-key rows are unwrapped to their value (both sides) so the two shapes
   compare. Multi-field rows are untouched.
4. **`_id` handling — default KEEP (corrected).** The error profile is
   asymmetric: wrong-strip on a grouped result is a *silent false PASS* (two
   different groupings whose value tuples coincide look equal); wrong-keep is at
   worst a *loud, debuggable false FAIL*. Therefore:
   - **Default: keep `_id`.**
   - Strip `_id` **only** to reconcile projection asymmetry on **non-grouped**
     results — i.e. one side carries `_id`, the other was projected with
     `{_id: 0}` → drop from both.
   - For `$group`, `_id` is the grouping key and is **load-bearing → always
     preserve**.
   - **`$group` is detected by reading the logged MQL, never by data shape.**
     A shape rule ("value is an ObjectId → strip") breaks when the group key is
     itself an ObjectId (group-by on a reference field). The MQL is the source of
     truth.
   - **`$group: {_id: null}` is NOT a grouping key (Phase 2 refinement).** A
     whole-collection aggregate has a single bucket with no key; its `_id` is not
     load-bearing. Such results are treated as non-grouped so a one-value
     aggregate (e.g. total revenue) reconciles like a scalar instead of being
     false-failed on an aliased output field name. "Meaningful `_id`" = a field
     reference (`"$status"`) or an expression over fields; `null`/literal = not.
   - **Output-field aliasing on grouped rows → whole-row value-bag (Phase 2,
     approved).** For a genuinely grouped result, two rows match iff their full
     **multiset of values matches, ignoring every field name (including `_id`)**.
     The group-key *value* participates in the bag, so it still anchors each row —
     cross-row value-swaps and different groupings are caught (the key values
     differ) — while tolerating both metric aliasing (gold `total` vs agent
     `revenue`) **and** group-key naming (gold raw `_id` vs agent-aliased
     `network`). Accepted tradeoffs: a differing field *count* is a mismatch (a
     genuinely different projection), and two values swapped between fields
     *within one row* are not distinguished. Non-grouped (find) rows keep strict
     key matching.
5. **Type normalization** (applied deeply to nested dicts/arrays):
   - `ObjectId` → canonical `str`.
   - `Decimal128` / `float` / `int` → number.
   - `datetime` / `Timestamp` → normalized ISO-8601 (UTC).
   - `None` / `NaN` handled explicitly and distinctly.
6. **Numeric comparison.** Integers compared exactly. Floats are equal if
   **either** (a) they are within a **relative tolerance, default `1e-6`**
   (`--rel-tol`) plus an **absolute floor (`--abs-tol`, default `1e-9`)`, **or**
   (b) **rounding both to the gold value's decimal precision** makes them equal
   (Phase 2). Path (b) handles hand-authored golds that apply `$round` (e.g. an
   average rounded to 3 dp): the unrounded agent value `2.4868333` matches gold
   `2.487` at gold's precision, without loosening the global tolerance for
   high-precision golds. Skipped when the gold is integer-precision, so a real
   difference (`395.4` vs `395`) is never truncated away. Mixed int/float
   compared as floats. **Rationale:**
   agent and gold execute the same deterministic dataset; the only expected
   divergence is float serialization (`df.to_json` vs Python repr, ~1e-6), not
   measurement noise — so the default must be tight. The old 0.5 % was a
   noisy-containment tolerance and would silently pass an aggregation genuinely
   wrong by 0.3 % (e.g. a sum of money). Widen per-metric only when a specific
   metric demonstrably needs it.
7. **Ordering.**
   - `order_sensitive` is determined **solely by whether the GOLD query carries
     an explicit sort** (`$sort` stage or `sort=`), read from the logged
     `gold_mql`. The agent's own sort declaration is **never consulted** for this
     decision — the gold defines the question (same principle as `_id` in §4.4).
     - *Why not the agent's sort:* using `agent AND gold` lets an agent that
       omits a required `$sort` drop to multiset compare → false PASS of a
       ranking that was never produced. Using `agent OR gold` lets a spurious
       agent `$sort` on a set-style question force an ordered compare → false
       FAIL of a correct answer. Both are exactly the silent-asymmetry failures
       this harness exists to kill.
   - `order_sensitive == False` → compare as a **multiset** (order-insensitive,
     duplicates significant).
   - `order_sensitive == True` → compare the **actual order of rows the agent
     returned** against the gold's row order. What matters is the sequence the
     agent *produced*, not whether its MQL declared a sort.
   - **Tie handling (required).** Read the sort key(s) from `gold_mql`. Partition
     both result-sets into maximal runs of equal sort-key value. Compare
     **within each run as a multiset** (intra-tie order is arbitrary and
     semantically irrelevant) and **across runs as an ordered sequence**. This
     catches genuine ordering errors without false-failing rankings that contain
     ties (e.g. two customers tied on spend).
   - With `gold_mql` logged (§7) there is no "sort-unknown" fallback. If the sort
     key is somehow unrecoverable, fall back to strict ordered compare and flag
     the row.
8. **Truncation safety net.** With the cap raised/disabled (§7), truncation
   should not occur; retain a `row_count == cap` flag purely as a residual
   safety net, **not** as the verdict mechanism.

`equivalent` is the binary PASS gate. `row_precision/recall/f1` are computed as a
multiset over canonicalized rows for debugging partial credit; they never gate.

---

## 5. Metrics emitted (per-question + aggregate)

- `execution_accuracy` — binary, **primary**.
- `structural_validity` — binary.
- `read_only_safety` — binary.
- `row_precision`, `row_recall`, `row_f1` — diagnostic partial credit.
- `tool_correctness` — right tool, right collection, introspected when needed
  (from the logged trajectory/args).
- Operating-envelope — `iterations`, `tokens`, `latency`; flag breaches of
  configurable budgets even when the answer is correct.
- `presentation_recall` — secondary fact-style check against the NL answer.
- `verified` — provenance marker (§9), so the report separates "passed against
  verified gold" from "passed against unverified gold".

**Aggregate PASS** gated on the three binary layers (§3).
**Multi-run** (**inactive until the runner repeats questions k times**): report
`pass@k` and all-runs consistency, not best-case — agents are non-deterministic
and best-case overstates reliability. The runner is single-run today, so this has
no data to operate on. The reporting code is kept, but **question repetition is
not added in this phase** — the section must not be mistaken for a live metric.

---

## 6. Module layout

- `mango_benchmark/equivalence.py` — `canonicalize`, `equivalent`, multiset
  P/R/F1. The locked core. Pure, deterministic, no DB/network.
- `mango_benchmark/scorecard.py` — metric assembly + CLI. **Primary entrypoint.**
  Mirrors the old scorer's flags (`--results`, `--only`, `--verbose`,
  `--rel-tol`, budget thresholds); tabular + summary output; **non-zero exit on
  failure for CI gating**.
- `runner.py` — patched (§7) to log agent MQL, gold MQL, NL answer.
- `tests/test_equivalence.py` — adversarial cases (§10).
- `fact_scorer.py` — retained but demoted to the presentation layer; no longer
  the gate.

Conventions: `from __future__ import annotations`, dataclasses, full type hints,
`argparse`, `python -m` entrypoint, exit codes. Matches existing
`mango_benchmark/` modules.

---

## 7. Sequencing — runner patch FIRST (corrected)

"Score from existing logs" is true only for non-grouped queries. Aggregation
verdicts depend on `_id`, which depends on the MQL, which is currently discarded.
Since the runner is being patched anyway (for structural + read-only), the order
is:

1. **Patch `runner.py` to log, per question:**
   - `agent_mql` — the `run_mql` tool args (the agent's `QueryRequest`(s)) for
     every call, not just names. Enables structural/read-only/sort/`$group`/tool
     layers.
   - `gold_mql` — the **full** gold query (operation/collection/filter/pipeline/
     sort/limit/projection/distinct_field). **Correction (Phase 2):** the gold
     `expected.dbQuery` was *not* sufficient as shipped — it serialized only the
     query *body* (`q.query`: the filter for find/count/distinct, the pipeline
     for aggregate), silently dropping `sort`/`limit`/`projection` for find-style
     queries. Three find rankings (Q11 `name:1`, Q12 `mrr_eur:-1`, Q16
     `commission_pct:-1, merchant_code:1`) lost their sort → §4.7 would mis-detect
     them as order-insensitive → **silent false-PASS of a ranking**. So the patch
     also enriches `seed_hard/ground_truth.py` to serialize the full structured
     query, and `dataset.py` parses it into a structured `gold_mql` dict (with a
     legacy body-only fallback for older CSVs). The marketplace CSV was
     regenerated against the live DB: gold *results* are byte-identical, only
     `dbQuery` is enriched.
   - `nl_answer` — `response.answer`, currently computed but never written.
     Enables the presentation layer.
   - **Raise/disable `RunMQLTool.max_rows` for benchmark runs** so both sides
     carry complete result-sets (corrects the cap that silently corrupts the
     primary metric). Keep the `row_count == cap` flag only as a residual safety
     net.
2. **Regenerate the 50 debug rows once** with the patched runner. **Check the
   gold side for an independent cap during regeneration:** verify the gold
   result-set is not separately capped at 100 anywhere in `ground_truth.py` /
   `dataset.py`. If the cap is lifted on the agent side only, large-result
   questions still false-FAIL on a one-sided truncation.
3. **Build** equivalence core → metrics → CLI → tests.

This collapses the three former "unknown" fallbacks (sort-unknown,
group-unknown, tool-correctness-unavailable) into solved cases from day one and
gives a trustworthy primary metric on aggregations immediately. Correctness over
speed.

---

## 8. Legacy column deprecation

The debug schema currently carries `correct_output_fuzzy`, `output_accuracy`,
`reasonable_output`, `tool_arg_valid`, `xmaner` — existing runner-side scoring
logic and a likely source of inflated numbers. To avoid two conflicting
"accuracy" signals in one file, the new harness **deprecates these columns**:
either stop writing them, or clearly prefix them `legacy_`. The new
`scorecard.py` ignores them entirely and treats `execution_accuracy` as the only
correctness gate. Called out here so the migration is explicit.

---

## 9. Gold-set integrity

- `verified: bool` per question. **Default:**
  - `seed_hard`-derived = `True`, labeled honestly as **"author-trusted query
    execution"** (not "human gold").
  - atlas-default dataset = `False` (provenance unknown).
- The report separates pass-rate against verified vs unverified gold.

**Follow-up (not a blocker):** the benchmark's ceiling is gold correctness, and a
hand-written gold query can encode the very semantic error the benchmark exists
to catch. Plan a one-time human audit of the `seed_hard` golds before treating
the numbers as absolute.

**Phase 2 audit finding — over-specified golds.** The first clean run surfaced
four golds that return a column the NL question does not ask for (an aggregate
**count** alongside the requested metric), which false-fails correct agents under
strict projection matching:

| Q | NL asks | Gold returns | Fix |
|---|---|---|---|
| Q24 | revenue by channel | + `orders` | drop `orders` |
| Q26 | captured amount per method | + `txns` | drop `txns` |
| Q28 | MRR per plan | + `subs` | drop `subs` |
| Q39 | units returned per reason | + `lines` | drop `lines` |

**Applied:** these gold queries were amended in `seed_hard/questions.py` (the
unrequested count dropped from `$group`/`$project`) and the CSV regenerated; all
four now pass and execution_accuracy rose 62% → 70% (§12). Separately, Q27/Q46
are the agent *over-answering* (extra columns) and Q38 is a genuine agent error
(under-grouping) — those stay FAIL.

---

## 10. Tests (adversarial — must break the old scorer's false PASSes)

`tests/test_equivalence.py` covers:

- **Superset** — extra rows beyond gold → FAIL (old scorer: false PASS).
- **Value-swap** — correct values, wrong key/row association → FAIL.
- **Ordering — sensitivity from gold only.**
  - gold has `$sort`, agent omits sort and returns a different order → **FAIL**
    (even though the agent's MQL declares no sort — the false-PASS guard).
  - gold has no sort, agent adds a spurious `$sort` → **PASS** (multiset; the
    agent's sort is ignored).
- **Ordering — tie handling.**
  - ranking with tied sort-key values, intra-tie order differs between agent and
    gold → **PASS**.
  - cross-tie order wrong (a higher-key row after a lower-key row) → **FAIL**.
- **Type coercion** — `ObjectId` / `Decimal128` / `datetime` normalize and
  compare equal across BSON and JSON-string forms.
- **Empty vs count-0** — `[]` ≠ scalar `0`.
- **Numeric tolerance.**
  - aggregation off by 0.3 % → **FAIL** (would have falsely passed at the old
    0.5 % default).
  - float-repr divergence ~1e-7 → **PASS**; boundary just inside / outside
    `1e-6`.
- **`$group` `_id` preservation** — same value tuples under different group keys
  → FAIL (the wrong-strip false-PASS guard); group key that is itself an
  `ObjectId` handled correctly.
- **Scalar reconciliation** — `{"rows":[{"count":N}]}` ≡ gold `N`.
- **Grouped value-bag (Phase 2)** — aliased metric field passes; whole-collection
  `$group:{_id:null}` reconciles to scalar; cross-row value-swap on grouped rows
  still FAILs.
- **`distinct` shape (Phase 2)** — gold scalar list ≡ agent single-key rows.
- **Round-to-gold-precision (Phase 2)** — gold `2.487` ≡ agent `2.4868333`;
  integer-precision gold does not truncate (`395.4` ≠ `395`).

---

## 11. Open decisions — locked

1. **Re-run vs log-only** → **log-only**, but only after the cap is raised
   (§7). Log-only on capped logs would freeze truncated results into the primary
   metric.
2. **100-row cap** → **raise/disable** for benchmark runs (§7). Truncation flag
   is a residual safety net, not the mechanism.
3. **`_id` group-key heuristic** → **keep `_id` by default; detect `$group` from
   the logged MQL** (§4.4).
4. **`verified` default** → `seed_hard` = `True` ("author-trusted query
   execution"); atlas-default = `False`; one-time human audit of seed_hard golds
   as a follow-up (§9).

---

## 12. Phase 2 results

First clean run (`results/qwen3.7_debug.jsonl`, model `qwen3.6-27b`, 50 questions,
patched runner):

| Scorer | PASS |
|---|---|
| old `fact_scorer` (cov ≥ 0.5) | **82%** (41/50) |
| **new scorecard** (exec ∧ struct ∧ read_only) | **62%** (31/50) |
| new scorecard, after correcting the 4 over-specified golds | **70%** (35/50) |

The four over-specified golds (Q24/26/28/39) were fixed in
`seed_hard/questions.py` and the CSV regenerated; Q24/26/28/39 then pass (the
agent was correct). The corrected-gold number is reproducible from the *same*
debug log via `scorecard --dataset <csv>`, which re-reads gold by `question_id`
without re-running the agent (agent output is independent of gold).

`execution_accuracy 62% · structural_validity 94% · read_only_safety 100% ·
tool_correctness 96% · presentation_recall 0.39 (secondary)`.

The 20-point gap = false positives the old scorer hid. 14 old-PASS→new-FAIL
flips. Poster child: **Q16** — agent returned 800 rows (no `$limit`); gold is 10;
old scorer PASSED (the 10 gold rows' facts appear among the 800), new scorer
FAILS. That is the dominant text-to-MQL failure mode (missing limit → superset).

Flip breakdown: ~7 genuine catches (Q11, Q16, Q35, Q36, Q37, Q38, Q48), 4
over-specified golds (Q24/26/28/39 — see §9), 2 agent over-answering (Q27/46), 1
aliased-sort near-miss (Q20).

**Scorer bugs found via real data and fixed** (none caught by the synthetic
tests alone — the value of running on real output):
1. `run_mql` results carry an `[AUTO-SCHEMA …]` prefix → result JSON wasn't at
   offset 0 → parsing returned `None` (also corrupted the old success metric).
2. `distinct` shape (scalar list vs single-key rows) — §4.3b.
3. grouped group-key naming asymmetry (`_id` vs alias) → whole-row value-bag —
   §4.4.
4. scorecard index misalignment: successful-call results were filtered but the
   match index was used against the unfiltered `agent_mql`, validating the wrong
   query (false STRUCT fails).
