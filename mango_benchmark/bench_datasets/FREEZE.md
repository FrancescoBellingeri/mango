# Freeze record — bench-dataset-v1

Status: **content-complete, NOT yet frozen.** All 292 cases build and pass
`validate.py` (structural/read-only/determinism/tie/boundary/exposure/sibling
gates — 0 violations). What's outstanding before this becomes the actual
frozen v1 is the human-review and contamination-remediation pass specified in
DATASET_DESIGN.md §7.5 and §5, which this agent cannot complete alone (it
requires human judgment calls, not mechanical checks). This file is the
honest snapshot at handoff, not a signed-off artifact.

Generated: 2026-07-07T15:10:02Z, repo commit `cb59d2a`.

---

## 1. Case counts

**Total: 294** (design target 293; final count after review remediation and
the two post-review EDGE-NUM additions).

| Category | n | Target | Domain split (logistics/workforce/meters) |
|---|---|---|---|
| CAP-FIND | 10 | 10 | 4/4/2 |
| CAP-FILTER | 25 | 25 | 9/9/7 |
| CAP-PROJ | 12 | 12 | 5/4/3 |
| CAP-SORT | 25 | 25 | 8/8/9 |
| CAP-PAGE | 8 | 8 | 3/3/2 |
| CAP-COUNT | 21 | 15 | 7/7/7 (over target — absorbed EDGE-EMPTY sibling anchors) |
| CAP-GROUP | 30 | 30 | 11/11/8 |
| CAP-MULTI | 25 | 25 | 9/8/8 |
| CAP-TIME | 20 | 20 | 6/8/6 |
| CAP-NEST | 20 | 20 | 9/5/6 |
| EDGE-NUM | 7 | 12 | 2/2/3 — resolved at 7, see §3 |
| EDGE-EMPTY | 10 | 10 | 3/3/4 |
| EDGE-SCALE | 6 | 6 | 2/2/2 |
| SEC-TRAP | 20 | 20 | 8/6/6 |
| OOS | 15 | 15 | 5/5/5 |
| AMB | 12 | 12 | 4/4/4 |
| NLR | 20 | 20 | 7/7/6 (15 para + 5 IT) |
| STRETCH | 8 | 8 | 4/2/2 |

`expected_behavior` split: `answer` 247, `refuse` 27, `safe_subset` 8, `any_of` 12.

Core-category domain spread (§2.6: ≥2 domains, no domain >60%) — verified by
`validate.py`'s whole-set check, currently green.

## 2. Artifacts

**FINAL — pinned at freeze.** Post rounds 1+2 review remediation (16 cases
changed, 1 replaced) plus the 2 post-review EDGE-NUM additions (294 cases
total — see `REVIEW.md`); `validate.py` 0 violations, contamination gate
green:

```
9197e0a3566125f86f3ab905a29a59736d6c57df131c6ed7bb43ec6d6f5464b3  cases.jsonl
47e906aaae59c919c3967a6ecd0843f932fe50dca1521a54cea4d921faa6d88f  bench_v1.csv
```

Superseded hashes (history, not for scoring): `c33bef2a…`/`67a51bcc…`
(pre-review, 292 cases), `5193a579…`/`da53887d…` (post round 1, 292 cases),
`afb7dd0a…`/`6f712ef1…` (post round 2, still 292 cases, pre-EDGE-NUM).

## 3. EDGE-NUM — RESOLVED at n=7 (owner decision, REVIEW.md)

Originally shipped at 5/12 (the must-FAIL "near-miss in (1e-6, 5e-3]"
pattern needs a hand-tuned numeric sentinel per instance; padding with
unverified numbers would have produced cases that pass or fail for the wrong
reason). Post-review the owner chose to add sentinels rather than accept 5:

- **LOG-NUM-002** — reserved vehicle VH-0100: 10 delivered shipments
  (avg weight exactly 503.007) + 4 cancelled tuned so the status-filter-
  forgotten average lands at rel ~2.0e-4.
- **WF-NUM-002** — reserved role "compliance auditor": 10 active
  (avg salary exactly 60750) + 3 terminated at 60802 so the
  exclusion-forgotten average lands at rel ~1.98e-4.

Both generator-asserted into the must-FAIL window with rounding-path guards
(the MET-NUM-001 lesson). **Final v1 count: 7** (3 must-FAIL, 4 in the
must-PASS family); the residual gap to the design target of 12 is accepted
as final for v1 — a v2 extension point, not a silent disagreement.

## 4. Contamination calibration (§5)

Corpus: 343 texts — `seed_hard.questions.QUESTIONS` (50) + `seed.questions.QUESTIONS`
(170, verified importing and populated) + `trainingset_marketplace.jsonl` +
`trainingset_example.jsonl` entries (123 combined).

Character 4-gram Jaccard distribution over the 292 candidate NLs:

| Percentile | Value |
|---|---|
| p50 | 0.137 |
| p75 | 0.221 |
| p90 | **0.281 ← frozen threshold** |
| p95 | 0.328 |
| p99 | 0.587 |
| p100 | 0.628 |

`JACCARD_THRESHOLD = 0.281` is now a **frozen absolute constant** in
`contamination.py` (calibrated once on this distribution, per §5.1 — it will
not move if a future corpus grows).

**Embedding cosine signal did not run**: `sentence-transformers` is not
installed in this environment. This is a disclosed gap, not a silent pass —
the Jaccard signal alone is the only contamination gate active on this build.
Before treating contamination clearance as complete, install
`sentence-transformers`, pin `EMBEDDING_MODEL` (already set to
`all-MiniLM-L6-v2` in code), calibrate `COSINE_THRESHOLD` the same way, and
re-run.

**RESOLVED (round-1 review)**: all 29 flagged cases were independently judged
by two AI reviewers — with adjudicator concurrence — to be short-template
convergence, zero genuine near-duplicates. Recorded in
`contamination.py::CONTAM_REVIEWED_OK`; the gate now exits 0 with the waiver
count printed. Full rationale in `REVIEW.md`. Original flag list kept below
for the record.

### 29 cases over the Jaccard threshold — adjudicated (originally: need review/rewrite/reject)

```
LOG-COUNT-001 (0.36)  LOG-GROUP-001 (0.62)  LOG-MULTI-001 (0.39)
LOG-TIME-001 (0.46)   LOG-FIND-002 (0.33)   LOG-COUNT-002 (0.63)
LOG-COUNT-004 (0.33)  LOG-GROUP-006 (0.30)  LOG-GROUP-007 (0.28)
LOG-GROUP-009 (0.38)  LOG-MULTI-003 (0.29)  LOG-MULTI-006 (0.34)
LOG-TIME-002 (0.33)   LOG-TIME-006 (0.34)   LOG-NEST-002 (0.30)
LOG-NEST-003 (0.29)   LOG-COUNT-006 (0.30)  WF-GROUP-001 (0.29)
WF-GROUP-010 (0.51)   WF-AMB-006 (0.30)     MET-MULTI-001 (0.28)
MET-COUNT-002 (0.59)  MET-GROUP-003 (0.33)  MET-GROUP-005 (0.36)
MET-MULTI-004 (0.30)  MET-COUNT-007 (0.31)  MET-EMPTY-003 (0.37)
MET-EMPTY-004 (0.31)  MET-OOS-009 (0.32)
```

**Read on these, honestly**: spot-checking the worst offenders (e.g.
`LOG-COUNT-002` "How many depots are currently active?" vs seed_hard's "How
many merchants are active?") suggests most of this list is **short-template
convergence** ("How many X are Y?" / "How many X per Y?"), not literal
memorized content — the entities and domains differ. Character 4-gram
Jaccard is oversensitive on short strings by construction. That said, the
gate exists precisely to force a human look at this instead of assuming it
away, per the design's own reasoning for using an absolute threshold. A
reviewer should: confirm each is template-convergence (rephrase to
diversify wording — cheap) vs a genuine near-duplicate of a *specific*
trainingset/seed example (reject/replace — the real risk this gate defends
against). None of these are in SEC-TRAP/OOS/AMB.

Corpus composition verified directly (not assumed): all three sources import
and populate correctly, so this calibration is against the full intended
corpus, not a partial one.

## 5. Mandatory human review (§7.5) — not done by this agent

Per DATASET_DESIGN.md, these require 100% human sign-off before the dataset
gates anything:

- **SEC-TRAP (20 cases)** — does each trap read as a realistic thing a user
  might actually type (not a strawman), and is the `safe_subset` reformulation
  genuinely the right read-only fallback?
- **OOS (15 cases)** — same realism check; is each truly unanswerable from
  this schema (not a case where the field just has a different name)?
- **AMB (12 cases)** — are both `gold_alternatives` genuinely defensible
  readings of the NL, not one obviously-right and one straw alternative?
- **NLR (20 cases)** — do the paraphrase/typo/colloquial/Italian variants
  preserve the base's meaning (a variant that accidentally changes the
  question invalidates the paired-flip design)?
- **20% stratified sample of the template-generated volume categories**
  (CAP-FILTER/SORT/GROUP/MULTI/TIME/NEST) — does the gold MQL actually answer
  the NL as a careful human reads it (the Q24/26/28/39 over-specified-gold
  lesson from the legacy suite, DESIGN.md §9).

Track reviewer/verdict per case in a `REVIEW.md` (not yet created) as cases
flip from `verified: author-trusted` to `verified: human-audited`.

## 6. Findings for Mango engineering (out of scope here, worth relaying)

Discovered incidentally while building this dataset — not dataset bugs, but
real gaps in shared infrastructure this dataset exercises harder than the
legacy 50-question suite did:

1. **`equivalence.py::_max_matching` doesn't scale — crashes small, hangs
   large.** The multiset row-matcher is a recursive DFS augmenting-path
   search over the full O(V²) pairwise deep-equality graph, no bucketing, no
   iterative fallback. Two independent symptoms of the same root cause,
   found at different result-set sizes:
   - **Crash**: a few-hundred-row result-set (encountered validating a
     since-narrowed CAP-FIND case) blew Python's default recursion limit.
     `validate.py` locally raises `sys.setrecursionlimit(20_000)`.
   - **Hang**: several realistic unordered `find` filters in this dataset
     (e.g. `LOG-FILTER-003`, a plain "priority in [express, critical]" query
     matching ~30% of a 24k-document collection) return several-thousand
     full nested documents. Determinism-checking such a case by re-running
     `compare()` took **up to 435 seconds per case** — confirmed with clean,
     non-suspend-confounded timing (an earlier run that appeared hung for
     14+ hours turned out to be a laptop-sleep artifact, not this; a clean
     rerun surfaced the real, finite-but-severe slowdown independently).
   Neither symptom is a dataset defect — these are realistic, legitimately
   selective filters a user would actually ask, and narrowing them to
   protect the scorer's performance would be exactly the kind of
   softening-for-the-score this benchmark exists to refuse. **Mitigated
   locally, not at the source**: `validate.py`'s own determinism re-check
   now fast-paths on a plain hashable-multiset comparison instead of calling
   `compare()` when the gold isn't order-sensitive (determinism only needs
   literal content identity, not full semantic equivalence) — this dropped
   full-suite validation from crashing/hanging to **48 seconds for all 294
   cases**. This is dataset-tooling-side and does not touch locked
   `equivalence.py`; the production scorecard path has no equivalent
   guard and will hit the same wall on any real agent answer that returns a
   comparably large, unsorted result-set — worth a bucketed-equality fix
   (e.g. group by a fast canonical hash key first, deep-compare only within
   matching buckets) at the source.
2. **`MQLValidator`'s stage/operator allowlist has no window-ranking
   functions.** `$setWindowFields` itself, `$geoNear`/`$geoWithin`/
   `$centerSphere` are all present, but `$rank`, `$denseRank`,
   `$documentNumber`, `$shift`, `$expMovingAvg`, `$locf` are absent from both
   `_ACCUMULATOR_OPERATORS` and `_EXPRESSION_OPERATORS` in
   `mango/tools/validator.py`. A correct agent answer to a genuine
   "rank X within Y" question is structurally unreachable — it would FAIL
   `structural_validity` regardless of reasoning quality. Two STRETCH cases
   were deliberately redesigned around this (`$max`-based partition
   comparison instead of `$rank`) so they test the agent, not the validator
   gap; the gap itself is untested by this dataset and should be fixed at the
   source per DESIGN.md §3's own risk-hierarchy framing.

## 7. What "frozen" requires from here

1. ~~Resolve the 29 flagged NLs~~ **DONE** — adjudicated as template
   convergence with dual-AI concurrence (`REVIEW.md`), gate green.
2. Review pass — **CLOSED** (rounds 1+2 dual AI, 16 fixes; the owner
   formally accepted AI-proxy review as the v1 standard — decision recorded
   in `REVIEW.md`; `verified` stays `author-trusted` everywhere so no case
   claims a human audit it never had).
3. (Optional but recommended) install `sentence-transformers`, calibrate and
   freeze `COSINE_THRESHOLD`, re-run contamination with the embedding signal
   active.
4. ~~EDGE-NUM sentinels~~ **DONE** — LOG-NUM-002 + WF-NUM-002 added, n=7
   accepted as final (owner decision, REVIEW.md; §3 above).
5. Final `build.py` + `validate.py` + re-hash + `git tag bench-dataset-v1`.
6. Only then does DATASET_DESIGN.md §9 (first run on Mango) begin.
