"""Trivial-agent probes (DATASET_DESIGN.md §6.1).

Three cheap baselines every case must FAIL (i.e. the probe's answer must NOT
be equivalent to gold):

  (i)   always-empty answer
  (ii)  ``find({})`` + limit 100 on the gold collection
  (iii) whole-collection count

A case a probe *passes* is degenerate — its gold is reachable without
understanding the question. Exemption: EDGE-EMPTY vs probe (i), by design
(§2.2), covered by the sibling-pair rule instead.
"""

from __future__ import annotations

from typing import Any

from mango_benchmark.bench_datasets.common import GoldMQL, serialize
from mango_benchmark.equivalence import compare

PROBE_NAMES = ("empty", "find_all_100", "collection_count")


def probe_results(db: Any, gold: GoldMQL) -> dict[str, Any]:
    return {
        "empty": [],
        "find_all_100": [serialize(d) for d in db[gold.collection].find({}).limit(100)],
        "collection_count": db[gold.collection].count_documents({}),
    }


def passing_probes(
    db: Any, gold: GoldMQL, gold_result: Any, *, exempt: frozenset[str] = frozenset()
) -> list[str]:
    """Names of probes whose answer is equivalent to gold (should be none)."""
    gold_dict = gold.to_dict()
    hits = []
    for name, result in probe_results(db, gold).items():
        if name in exempt:
            continue
        if compare(result, gold_result, gold_mql=gold_dict).equivalent:
            hits.append(name)
    return hits
