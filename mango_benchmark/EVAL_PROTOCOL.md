# Eval protocol — dev / test firewall

Discipline for measuring agent improvements **honestly**, i.e. distinguishing a
real capability gain from benchmark overfit (Goodhart). Read before iterating.

## Split

| Role | Benchmark | Size | Use |
|------|-----------|------|-----|
| **DEV**  | `mango_ecommerce_benchmark.csv`   | 169 | Iterate freely. Read per-question failures, design fixes against them. |
| **TEST** | `mango_marketplace_benchmark.csv` |  50 | Cross-schema firewall. Frozen. Consult **rarely** (milestones only). |

Different schemas on purpose: marketplace is the "schema never seen during
development" judge. That is the only signal that separates real generalization
from dev-set overfit.

## The loop

1. **Baseline both, now.** Freeze the numbers (see commands below).
2. Develop one change. Score on **DEV** only.
3. DEV did not improve → discard, try another change. Do **not** touch TEST.
4. DEV improved → score **TEST once** as a yes/no transfer check.
   - Transfers → real gain, keep it.
   - Improves DEV only → overfit, treat with suspicion.

## Firewall rules (what keeps TEST honest)

- **TEST is read aggregate-only.** Look at the pass-rate number, not *which*
  marketplace questions failed. The moment you read its per-question failures to
  design a fix, marketplace stops being a firewall and becomes a second dev set.
- **No benchmark-specific aids in the firewall path.** Any help given to the
  agent (memory, training files, few-shot) must be schema-agnostic. Hand-authored
  cheat sheets cued to one benchmark are teaching-to-the-test.
  - `examples/trainingset_marketplace.jsonl` is exactly this — a hand-written
    crib of marketplace's drift quirks. **It must NOT be loaded** in baseline or
    transfer runs. It may live only as a separate "upper bound with hand-holding"
    experiment, never as a dev→test measurement.
- **Fresh memory per measured run.** Use a dedicated `--results-dir` so the
  `.benchmark_memory` is fresh and prior runs' accumulated correct queries don't
  leak in.
- **Same conditions on both sides.** Whatever flags you run on DEV, run identically
  on TEST. A transfer comparison across different conditions is meaningless.

## Baseline commands (clean: no benchmark-specific training, fresh memory)

```bash
set -a; source .env; set +a   # loads REGOLO_URL, REGOLO_API_KEY, MONGODB_URI

# DEV baseline — ecommerce (169 q)
.venv/bin/python -m mango_benchmark.runner \
  --csv-path mango_benchmark/mango_ecommerce_benchmark.csv \
  --provider openai --model glm5.2-beta \
  --base-url "$REGOLO_URL" --api-key "$REGOLO_API_KEY" \
  --sample 0 --results-dir mango_benchmark/results/baseline

# TEST baseline — marketplace (50 q), SAME conditions, NO --training-file
.venv/bin/python -m mango_benchmark.runner \
  --csv-path mango_benchmark/mango_marketplace_benchmark.csv \
  --provider openai --model glm5.2-beta \
  --base-url "$REGOLO_URL" --api-key "$REGOLO_API_KEY" \
  --sample 0 --results-dir mango_benchmark/results/baseline
```

## Scoring

`runner.py`'s inline metrics are deprecated (DESIGN §8). Score each run's
`_debug.jsonl` with the scorecard — `execution_accuracy` / PASS is the number
that matters:

```bash
.venv/bin/python -m mango_benchmark.scorecard \
  --results mango_benchmark/results/baseline/<model>_<ts>_debug.jsonl
```

Record the two PASS rates. They are the frozen reference every later change is
measured against.
