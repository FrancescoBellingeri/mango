"""Adversarial tests for the canonical equivalence core (DESIGN.md §10).

Each test targets a failure mode the old fact-containment scorer hid: supersets,
value-swaps, ordering, type coercion, empty-vs-zero, numeric tolerance and
``$group`` ``_id`` preservation. These are the cases the new primary metric must
get right.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from mango_benchmark.equivalence import (
    DEFAULT_REL_TOL,
    canonicalize,
    compare,
    equivalent,
)

try:
    from bson import Decimal128, ObjectId

    _HAS_BSON = True
except ImportError:  # pragma: no cover - bson ships with pymongo
    _HAS_BSON = False


def _find(sort: dict | None = None) -> dict:
    return {"operation": "find", "collection": "c", "filter": {}, "pipeline": None,
            "projection": None, "sort": sort, "limit": None, "distinct_field": None}


def _agg(pipeline: list) -> dict:
    return {"operation": "aggregate", "collection": "c", "filter": None,
            "pipeline": pipeline, "projection": None, "sort": None,
            "limit": None, "distinct_field": None}


def _count() -> dict:
    return {"operation": "count", "collection": "c", "filter": {}, "pipeline": None,
            "projection": None, "sort": None, "limit": None, "distinct_field": None}


# ---------------------------------------------------------------------------
# Superset — the dominant text-to-MQL failure (missing filter/limit → extra rows)
# ---------------------------------------------------------------------------


def test_superset_fails():
    # Multi-field, multi-row gold so it stays row-shaped (no scalar reconciliation)
    # and the partial-credit diagnostic is exercised.
    gold = [{"id": 1, "x": 1}, {"id": 2, "x": 2}]
    agent = {"rows": [{"id": 1, "x": 1}, {"id": 2, "x": 2}, {"id": 3, "x": 3}]}
    res = compare(agent, gold, gold_mql=_find())
    assert res.equivalent is False
    assert res.recall == 1.0                      # all gold rows covered
    assert res.precision == pytest.approx(2 / 3)  # agent has an extra row


def test_single_row_single_field_reconciles_to_scalar_then_superset_fails():
    # A 1-row/1-field result reduces to a scalar (§4.2); a superset can't match it.
    res = compare({"rows": [{"x": 1}, {"x": 2}]}, [{"x": 1}], gold_mql=_find())
    assert res.equivalent is False


def test_exact_set_passes():
    gold = [{"x": 1}, {"x": 2}]
    agent = {"rows": [{"x": 2}, {"x": 1}], "row_count": 2}  # order differs, no sort
    assert equivalent(agent, gold, gold_mql=_find()) is True


# ---------------------------------------------------------------------------
# Value-swap — right values, wrong association
# ---------------------------------------------------------------------------


def test_value_swap_fails():
    gold = [{"name": "A", "city": "NY"}, {"name": "B", "city": "LA"}]
    agent = {"rows": [{"name": "A", "city": "LA"}, {"name": "B", "city": "NY"}]}
    assert equivalent(agent, gold, gold_mql=_find()) is False


# ---------------------------------------------------------------------------
# Ordering — sensitivity decided by the GOLD query only (DESIGN.md §4.7)
# ---------------------------------------------------------------------------


def test_gold_sorted_agent_wrong_order_fails():
    gold = [{"v": 3}, {"v": 2}, {"v": 1}]
    agent = {"rows": [{"v": 1}, {"v": 2}, {"v": 3}]}  # agent MQL declares no sort
    assert equivalent(agent, gold, gold_mql=_find(sort={"v": -1})) is False


def test_gold_unsorted_agent_spurious_sort_passes():
    gold = [{"v": 1}, {"v": 2}]
    agent = {"rows": [{"v": 2}, {"v": 1}]}  # agent added a sort; gold has none → ignored
    assert equivalent(agent, gold, gold_mql=_find(sort=None)) is True


def test_aggregate_final_sort_is_order_sensitive():
    pipe = [{"$group": {"_id": "$c", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}]
    gold = [{"_id": "A", "n": 5}, {"_id": "B", "n": 2}]
    agent_ok = {"rows": [{"_id": "A", "n": 5}, {"_id": "B", "n": 2}]}
    agent_bad = {"rows": [{"_id": "B", "n": 2}, {"_id": "A", "n": 5}]}
    assert equivalent(agent_ok, gold, gold_mql=_agg(pipe)) is True
    assert equivalent(agent_bad, gold, gold_mql=_agg(pipe)) is False


# ---------------------------------------------------------------------------
# Ordering — tie handling (DESIGN.md §4.7)
# ---------------------------------------------------------------------------


def test_intra_tie_order_irrelevant_passes():
    spec = {"spend": -1}
    gold = [{"spend": 100, "c": "A"}, {"spend": 100, "c": "B"}, {"spend": 50, "c": "C"}]
    agent = {"rows": [{"spend": 100, "c": "B"}, {"spend": 100, "c": "A"}, {"spend": 50, "c": "C"}]}
    assert equivalent(agent, gold, gold_mql=_find(sort=spec)) is True


def test_cross_tie_order_wrong_fails():
    spec = {"spend": -1}
    gold = [{"spend": 100, "c": "A"}, {"spend": 100, "c": "B"}, {"spend": 50, "c": "C"}]
    agent = {"rows": [{"spend": 50, "c": "C"}, {"spend": 100, "c": "A"}, {"spend": 100, "c": "B"}]}
    assert equivalent(agent, gold, gold_mql=_find(sort=spec)) is False


# ---------------------------------------------------------------------------
# Type coercion — ObjectId / Decimal128 / datetime across BSON and JSON forms
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_BSON, reason="bson not installed")
def test_objectid_bson_vs_string():
    oid = ObjectId()
    gold = [{"_id": str(oid), "v": 1}]
    agent = {"rows": [{"_id": oid, "v": 1}]}
    assert equivalent(agent, gold, gold_mql=_find()) is True


@pytest.mark.skipif(not _HAS_BSON, reason="bson not installed")
def test_decimal128_vs_float():
    gold = [{"amt": 2.5}]
    agent = {"rows": [{"amt": Decimal128("2.5")}]}
    assert equivalent(agent, gold, gold_mql=_find()) is True


def test_datetime_vs_iso_string():
    gold = [{"t": "2024-01-01T00:00:00"}]
    agent = {"rows": [{"t": datetime(2024, 1, 1, 0, 0, 0)}]}
    assert equivalent(agent, gold, gold_mql=_find()) is True


# ---------------------------------------------------------------------------
# Empty result-set vs scalar 0 (DESIGN.md §4.3)
# ---------------------------------------------------------------------------


def test_empty_not_equal_to_count_zero():
    agent = {"rows": [], "row_count": 0}
    assert equivalent(agent, 0, gold_mql=_count()) is False


def test_empty_equals_empty():
    agent = {"rows": [], "row_count": 0}
    assert equivalent(agent, [], gold_mql=_find()) is True


def test_count_zero_reconciles_to_scalar():
    agent = {"rows": [{"count": 0}], "row_count": 1}
    assert equivalent(agent, 0, gold_mql=_count()) is True


# ---------------------------------------------------------------------------
# Scalar reconciliation — {"rows":[{"count":N}]} ≡ gold N
# ---------------------------------------------------------------------------


def test_count_scalar_reconciliation():
    agent = {"rows": [{"count": 395}], "row_count": 1}
    assert equivalent(agent, 395, gold_mql=_count()) is True
    assert canonicalize(agent, is_group=False).kind == "scalar"


# ---------------------------------------------------------------------------
# Numeric tolerance (DESIGN.md §4.6) — tight default 1e-6
# ---------------------------------------------------------------------------


def test_aggregation_off_by_third_percent_fails():
    # Would have falsely passed at the old 0.5% default.
    assert equivalent(1003.0, 1000.0, gold_mql=_count()) is False


def test_float_repr_divergence_passes():
    assert equivalent(1000.0000001, 1000.0, gold_mql=_count()) is True


def test_tolerance_boundary():
    # High-precision gold so round-to-gold-precision doesn't mask the rel_tol test.
    base = 1000.000123
    just_inside = base * (1 + DEFAULT_REL_TOL * 0.9)
    just_outside = base * (1 + DEFAULT_REL_TOL * 2.0)
    assert equivalent(just_inside, base, gold_mql=_count()) is True
    assert equivalent(just_outside, base, gold_mql=_count()) is False


def test_round_to_gold_precision_passes():
    # Gold rounded to 3 dp ($round); agent unrounded → equal at gold precision.
    assert equivalent(2.4868333333, 2.487, gold_mql=_count()) is True
    # Money gold at 2 dp.
    assert equivalent(116063.948, 116063.95, gold_mql=_count()) is True


def test_round_to_gold_precision_does_not_mask_real_error():
    # Gold 3 dp; agent differs in the 2nd decimal → still fails.
    assert equivalent(2.51, 2.487, gold_mql=_count()) is False
    # Integer-precision gold must not truncate the agent value.
    assert equivalent(395.4, 395, gold_mql=_count()) is False


def test_integers_compared_exactly():
    assert equivalent(100, 100, gold_mql=_count()) is True
    assert equivalent(101, 100, gold_mql=_count()) is False


# ---------------------------------------------------------------------------
# $group _id preservation (DESIGN.md §4.4) — the silent false-PASS guard
# ---------------------------------------------------------------------------


def test_group_id_preserved_different_keys_fail():
    pipe = [{"$group": {"_id": "$cat", "total": {"$sum": 1}}}]
    gold = [{"_id": "X", "total": 5}, {"_id": "Y", "total": 3}]
    # Same value tuples, different grouping → must FAIL (not a coincidental pass).
    agent = {"rows": [{"_id": "P", "total": 5}, {"_id": "Q", "total": 3}]}
    assert equivalent(agent, gold, gold_mql=_agg(pipe)) is False


def test_group_matching_keys_pass():
    pipe = [{"$group": {"_id": "$cat", "total": {"$sum": 1}}}]
    gold = [{"_id": "X", "total": 5}, {"_id": "Y", "total": 3}]
    agent = {"rows": [{"_id": "Y", "total": 3}, {"_id": "X", "total": 5}]}  # unordered group
    assert equivalent(agent, gold, gold_mql=_agg(pipe)) is True


def test_group_aliased_metric_field_passes():
    # Agent aliases the computed metric differently from gold — same value.
    pipe = [{"$group": {"_id": "$cat", "total": {"$sum": "$amt"}}}]
    gold = [{"_id": "A", "total": 5}, {"_id": "B", "total": 3}]
    agent = {"rows": [{"_id": "A", "revenue": 5}, {"_id": "B", "revenue": 3}]}
    assert equivalent(agent, gold, gold_mql=_agg(pipe)) is True


def test_group_aliased_but_cross_row_value_swap_still_fails():
    pipe = [{"$group": {"_id": "$cat", "total": {"$sum": "$amt"}}}]
    gold = [{"_id": "A", "total": 5}, {"_id": "B", "total": 3}]
    # Values swapped across the group keys → _id anchors catch it despite aliasing.
    agent = {"rows": [{"_id": "A", "revenue": 3}, {"_id": "B", "revenue": 5}]}
    assert equivalent(agent, gold, gold_mql=_agg(pipe)) is False


def test_whole_collection_aggregate_null_id_reconciles_alias():
    # $group:{_id:null} → not a grouping key → single-value aggregate reconciles
    # to scalar; aliased field name is irrelevant.
    pipe = [{"$group": {"_id": None, "revenue_eur": {"$sum": "$amt"}}}]
    gold = [{"_id": None, "total_revenue_eur": 2196109211.14}]
    agent = {"rows": [{"_id": None, "revenue_eur": 2196109211.14}]}
    assert equivalent(agent, gold, gold_mql=_agg(pipe)) is True


@pytest.mark.skipif(not _HAS_BSON, reason="bson not installed")
def test_group_key_is_objectid_not_stripped():
    # Group key is itself an ObjectId — a data-shape rule would wrongly strip it.
    oid = ObjectId()
    pipe = [{"$group": {"_id": "$merchant_id", "n": {"$sum": 1}}}]
    gold = [{"_id": str(oid), "n": 2}]
    agent = {"rows": [{"_id": oid, "n": 2}]}
    assert equivalent(agent, gold, gold_mql=_agg(pipe)) is True


# ---------------------------------------------------------------------------
# Legacy gold_mql fallback (body-only / None) — safe direction
# ---------------------------------------------------------------------------


def test_legacy_none_mql_is_order_insensitive():
    gold = [{"v": 1}, {"v": 2}]
    agent = {"rows": [{"v": 2}, {"v": 1}]}
    assert equivalent(agent, gold, gold_mql=None) is True
