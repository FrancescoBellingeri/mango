# mango-bench — evaluating a text-to-MQL agent / model

How to run a full test of the Mango agent against a set of natural-language
questions, and how to read the results.

For the **scoring rules** (result-set equivalence, metrics, design decisions) see
[`DESIGN.md`](DESIGN.md). This README is the operational guide.

> **Primary scorer: `scorecard.py`.** The old `XMaNeR` /
> `correct_output_fuzzy` / `output_accuracy` metrics written by `runner.py` are
> **deprecated** (they overstated accuracy — see DESIGN §8): the scorecard ignores
> them. `runner.py` still writes them as columns, but the scorecard is the
> reference.

---

## What it measures

The agent receives a set of natural-language questions, generates MongoDB queries
(`find`/`aggregate`/`count`/`distinct`) and answers. The scorer compares the
agent's **executed result-set** against the **gold** (a hand-written query
executed against the real DB), across three independent layers:

1. `execution_accuracy` — is the result-set equivalent to gold? **(primary metric)**
2. `structural_validity` — is the generated query well-formed?
3. `read_only_safety` — no writes / server-side JS?

**PASS = execution_accuracy ∧ structural_validity ∧ read_only_safety.**

`presentation_recall` (the old fact-containment check) remains as a **secondary**
signal, never a gate.

---

## Prerequisites

- **MongoDB** running with the benchmark database populated. For the "hard"
  benchmark used here that is `mango_marketplace` (generate it with
  `seed_hard/generate.py`). The original Atlas dataset (766 questions) uses the
  Atlas `sample_*` databases.
- **venv** active (`.venv` with dependencies installed).
- **API key** for the LLM provider under test. Export it instead of inlining it:
  ```bash
  export REGOLO_API_KEY=sk-...
  ```

---

## Full test in 2 steps

### 1. The agent answers the questions

```bash
.venv/bin/python -m mango_benchmark.runner \
  --csv-path mango_benchmark/mango_marketplace_benchmark.csv \
  --training-file examples/trainingset_marketplace.jsonl \
  --provider openai --model <MODEL> \
  --base-url <BASE_URL> --api-key "$REGOLO_API_KEY"
```

- `--provider openai` for OpenAI-compatible endpoints (e.g. regolo.ai, even when
  serving GLM/Qwen models). For native Gemini: `--provider gemini`.
- `--max-rows` defaults to 100000 (cap raised on purpose: do not truncate the
  agent's result-set, otherwise questions with many rows fail unfairly).
- `--no-memory` disables the ChromaDB memory layer. `--help` for all options.

It produces two files in `mango_benchmark/results/`, **named after the model +
date**:
```
<model>_<YYYYMMDD_HHMMSS>.csv          # human-readable summary
<model>_<YYYYMMDD_HHMMSS>_debug.jsonl  # full log (this is what you score)
```
E.g. `qwen3.6-27b_20260626_213000_debug.jsonl`. (With multiple models in one run
the name is `multi_...`; the per-row `model` column distinguishes them.)

### 2. Score

```bash
F=$(ls -t mango_benchmark/results/*_debug.jsonl | head -1)   # most recent
.venv/bin/python -m mango_benchmark.scorecard --results "$F" \
  --dataset mango_benchmark/mango_marketplace_benchmark.csv
```

Useful options:
- `--dataset <csv>` — re-reads gold from the CSV by `question_id`. **If you fix a
  gold you don't have to re-run**: re-score the existing debug log against the
  updated gold (the agent's output is independent of the gold).
- `--verbose` — per-question FAIL detail (which layer fell).
- `--only 11,16,24` — score only some questions.
- `--rel-tol` / `--abs-tol` — numeric tolerances (default 1e-6 / 1e-9, tight).
- `--iter-budget` / `--token-budget` / `--latency-budget` — flag overruns.
- `--min-pass-rate 0.7` — non-zero exit code if the PASS rate drops below the
  threshold (for CI gating).

---

## Reading the output

Per-question table + aggregate summary:

```
PASS (exec ∧ struct ∧ read_only) : 70.0%     <- the number that matters
execution_accuracy (PRIMARY)     : 70.0%
structural_validity              : 94.0%
read_only_safety                 : 100.0%
tool_correctness                 : 96.0%
row_f1 (mean, diagnostic)        : 0.63
presentation_recall (SECONDARY)  : 0.39
```

Per-question columns: `exec / strc / ro` (the three gates), `P/R/F1` (row-level
precision/recall/F1, diagnostic), `ord` (is the question order-sensitive?), `grp`
(is it a `$group`?), `pres` (presentation_recall).

### Interpreting a FAIL

- **fail = EXEC, high f1 (~0.9)** → almost right: a few rows/order off.
- **fail = EXEC, f1 = 0.00** → completely different result-set (wrong
  filter/grouping, or different columns from gold).
- **`[final-only]`** → the last query didn't match but an earlier one did.
- **fail = STRUCT** → malformed query (invalid operator/stage).
- **fail = RO** → write / server-side-JS attempt (a red flag).
- **`note: sort-key not in rows`** → gold sorts by a field the agent renamed:
  ordering compared via positional fallback.

> ⚠️ A FAIL is not always the agent's fault: it may be an **over-specified gold**
> (returns columns the question doesn't ask for). When in doubt, compare the NL
> question against the gold's columns before concluding.

---

## Regenerating the gold (only if you change the questions/queries)

The golds live in `seed_hard/questions.py` (hand-written queries). To regenerate
the CSV by executing them against the DB:

```bash
.venv/bin/python -m mango_benchmark.seed_hard.ground_truth \
  --uri mongodb://localhost:27017 \
  --out mango_benchmark/mango_marketplace_benchmark.csv
```

Gold provenance: **hand-written** queries executed against the real DB —
"author-trusted query execution", not audited row by row. Questions on the
`mango_marketplace` DB are marked `verified=True` in the scorecard.

---

## Files in the flow

**Run these (entry points):**
- `runner.py` — runs the agent against the benchmark.
- `scorecard.py` — primary evaluator.
- `seed_hard/ground_truth.py` — (re)generates the gold CSV.

**Libraries (imported, not run directly):**
- `equivalence.py` — result-set equivalence core.
- `dataset.py` — loads the CSV + the structured `gold_mql`.
- `fact_scorer.py` — used by the scorecard for `presentation_recall` (secondary).

**Evaluator tests:**
```bash
.venv/bin/python -m pytest tests/test_equivalence.py -q
```

**Config:** defaults in [`config.py`](config.py) (`SAMPLE_SIZE`, `MAX_ITERATIONS`,
`TIMEOUT_SECONDS`, `MEMORY_ENABLED`, `MODELS`, `TRAINING_FILE`, `RESULTS_DIR`).

---

## Available datasets

- **`mango_marketplace`** (hard, used above) — generated locally from `seed_hard/`;
  hand-written golds executed against the real DB.
- **Atlas sample (766 questions)** — the original
  [NL-to-mongosh](https://huggingface.co/datasets/mongodb-eai/natural-language-to-mongosh)
  benchmark over the Atlas `sample_*` databases. It is `dataset.py`'s default, so
  it's used when you run **without** `--csv-path` — but its CSV
  (`atlas_sample_data_benchmark.braintrust.csv`) is not committed: place it next
  to `dataset.py` (or pass it via `--csv-path`). Requires an Atlas cluster with
  the sample datasets loaded and `MONGODB_URI` set.
