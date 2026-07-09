"""Contamination gate (DATASET_DESIGN.md §5).

Scores every benchmark NL question against the assembled contamination
corpus (trainingset JSONLs + seed/seed_hard questions) and rejects cases
above the frozen absolute thresholds.

Similarity signals:
  * character 4-gram Jaccard (always available, deterministic)
  * embedding cosine — only if ``sentence-transformers`` is installed; the
    model name is pinned below and must be recorded in FREEZE.md.

Thresholds are calibrated once on the v1 candidate distribution
(``--calibrate`` prints the percentiles) and then FROZEN — they are absolute
constants, not moving percentiles (§5.1).

Usage:
    python -m mango_benchmark.bench_datasets.contamination --calibrate
    python -m mango_benchmark.bench_datasets.contamination            # gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mango_benchmark.bench_datasets.build import load_cases

# --- frozen gate constants (recorded in FREEZE.md at sign-off, DATASET_DESIGN.md §5) --
# Calibrated ONCE on the full v1 candidate set (292 cases vs a 343-text corpus of
# seed_hard questions + trainingset entries): p90 of the observed Jaccard
# distribution was 0.281. From this point the constant is FROZEN — absolute,
# not a moving percentile — so v2 growing the corpus cannot silently shift it.
JACCARD_THRESHOLD: float | None = 0.281
# COSINE_THRESHOLD left unset: sentence-transformers is not installed in this
# environment, so the embedding signal never ran for v1. This is a disclosed
# gap, not a silent pass — see FREEZE.md. Calibrate + freeze before relying on
# cosine similarity as a gate.
COSINE_THRESHOLD: float | None = None
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cases flagged over threshold and ADJUDICATED as short-template convergence
# (not near-duplicates) in the v1 review pass — see REVIEW.md for the two
# independent AI reviewers' per-case verdicts and the adjudication rationale.
# Character 4-gram Jaccard is oversensitive on short analytic questions
# ("How many X are Y?"); every entry here shares only a stem with its corpus
# match while entities, domain, and query class differ. The gate's purpose is
# to FORCE this review, not to mechanically reject — an entry may only be
# added here with a REVIEW.md record behind it.
CONTAM_REVIEWED_OK = frozenset({
    "LOG-COUNT-001", "LOG-COUNT-002", "LOG-COUNT-004", "LOG-COUNT-006",
    "LOG-FIND-002", "LOG-GROUP-001", "LOG-GROUP-006", "LOG-GROUP-007",
    "LOG-GROUP-009", "LOG-MULTI-001", "LOG-MULTI-003", "LOG-MULTI-006",
    "LOG-NEST-002", "LOG-NEST-003", "LOG-TIME-001", "LOG-TIME-002",
    "LOG-TIME-006", "WF-GROUP-001", "WF-GROUP-010", "WF-AMB-006",
    "MET-COUNT-002", "MET-COUNT-007", "MET-EMPTY-003", "MET-EMPTY-004",
    "MET-GROUP-003", "MET-GROUP-005", "MET-MULTI-001", "MET-MULTI-004",
    "MET-OOS-009",
})

_ROOT = Path(__file__).parent.parent.parent
TRAININGSET_PATHS = [
    _ROOT / "examples" / "trainingset_marketplace.jsonl",
    _ROOT / "examples" / "trainingset_example.jsonl",
]


def _ngrams(text: str, n: int = 4) -> set[str]:
    t = " ".join(text.lower().split())
    return {t[i : i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a: str, b: str) -> float:
    na, nb = _ngrams(a), _ngrams(b)
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def load_corpus() -> list[str]:
    corpus: list[str] = []

    from mango_benchmark.seed_hard.questions import QUESTIONS as HARD_QS

    corpus.extend(q.nl_query for q in HARD_QS)
    try:
        from mango_benchmark.seed.questions import QUESTIONS as SEED_QS

        corpus.extend(q.nl_query for q in SEED_QS)
    except (ImportError, AttributeError):
        print("  (seed questions not importable — skipped)", file=sys.stderr)

    for path in TRAININGSET_PATHS:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("question", "text", "nl_query"):
                    if isinstance(entry.get(key), str):
                        corpus.append(entry[key])
                        break
    return corpus


def max_similarities(nls: list[str], corpus: list[str]) -> list[tuple[float, float | None]]:
    """Per NL: (max 4-gram Jaccard, max embedding cosine or None)."""
    cosines: list[float] | None = None
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore

        model = SentenceTransformer(EMBEDDING_MODEL)
        emb_nl = model.encode(nls, convert_to_tensor=True, normalize_embeddings=True)
        emb_corpus = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
        cosines = util.cos_sim(emb_nl, emb_corpus).max(dim=1).values.tolist()
    except ImportError:
        print("  (sentence-transformers not installed — Jaccard-only)", file=sys.stderr)

    out = []
    for i, nl in enumerate(nls):
        jac = max((jaccard(nl, c) for c in corpus), default=0.0)
        out.append((jac, cosines[i] if cosines else None))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Contamination gate (§5)")
    parser.add_argument("--calibrate", action="store_true", help="print percentiles, no gating")
    args = parser.parse_args()

    cases = load_cases()
    corpus = load_corpus()
    print(f"{len(cases)} candidate NLs vs corpus of {len(corpus)} texts")

    sims = max_similarities([c.nl_question for c in cases], corpus)

    if args.calibrate:
        jacs = sorted(s[0] for s in sims)
        for p in (50, 75, 90, 95, 99, 100):
            idx = min(len(jacs) - 1, int(len(jacs) * p / 100))
            print(f"  jaccard p{p}: {jacs[idx]:.3f}")
        worst = sorted(zip((s[0] for s in sims), (c.id for c in cases)), reverse=True)[:10]
        print("  worst 10:", ", ".join(f"{cid}={j:.2f}" for j, cid in worst))
        return

    if JACCARD_THRESHOLD is None:
        print("thresholds not frozen yet — run --calibrate on the full v1 set, "
              "set JACCARD_THRESHOLD/COSINE_THRESHOLD, record them in FREEZE.md")
        sys.exit(1)

    n_bad = 0
    n_waived = 0
    for case, (jac, cos) in zip(cases, sims):
        over = jac > JACCARD_THRESHOLD or (
            cos is not None and COSINE_THRESHOLD is not None and cos > COSINE_THRESHOLD
        )
        if not over:
            continue
        if case.id in CONTAM_REVIEWED_OK:
            n_waived += 1
            continue
        n_bad += 1
        print(f"  {case.id}: jaccard={jac:.3f} cosine={cos} — rewrite or reject (or adjudicate via REVIEW.md)")
    print(f"{n_bad} cases over threshold ({n_waived} over-threshold but review-adjudicated as convergence)")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
