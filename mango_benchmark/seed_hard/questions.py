"""Benchmark questions for the *mango_marketplace* HARD database.

Each :class:`BenchmarkQuestion` is the *canonical* solution to its natural
language prompt: ``ground_truth.py`` runs the query live and records the result
as the expected answer. The questions are written to exercise the four
difficulty levers of this DB, so a correct reference query must itself cope
with the mess:

  * **schema drift**      — orders exist as v2 (``status`` / ``user_id`` /
    ``totals.grand_total_eur``) and legacy v1 (``state`` / ``customer_id`` /
    top-level ``grand_total_eur``). Unified queries use ``$ifNull`` / ``$switch``.
  * **boolean dialects**  — ``true`` / ``"Y"`` / ``1`` / ``"yes"`` → ``$in`` sets.
  * **enum case drift**   — merchant ``status`` is ``active`` *and* ``ACTIVE``.
  * **multi-currency**    — native amounts live alongside ``*_eur`` snapshots;
    EUR questions sum the snapshot fields (deterministic), and ``fx_rates`` is
    queried directly for rate questions.
  * **mixed rating scales** — reviews are 1..5 (``rating``) or legacy 1..10
    (``score``); normalised via ``$switch``.
  * **deep nesting / polymorphism** — per-merchant order ``groups[].items[]``,
    polymorphic ``payments.details`` / ``events.payload`` / ``promotions.action``.

NOTE: messy-date fields (``events.ts``, ``audit_logs.at``, ``placed_at``,
``inventory_snapshots.snapshot_at``, ``messages[].at``, ``last_login_at``) mix
ISODate / epoch-int / string encodings, so questions intentionally avoid range
filters on them and use clean datetime fields instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Date boundaries (DB window is 2022-01-01 .. 2025-12-31)
# ---------------------------------------------------------------------------

D2024_START = datetime(2024, 1, 1)
D2024_END = datetime(2024, 12, 31, 23, 59, 59)
D2025_START = datetime(2025, 1, 1)
D2025_END = datetime(2025, 12, 31, 23, 59, 59)
Q1_2025_START = datetime(2025, 1, 1)
Q1_2025_END = datetime(2025, 3, 31, 23, 59, 59)

# Boolean dialects used across "messy" collections.
TRUTHY = [True, "Y", "true", 1, "yes"]


@dataclass
class BenchmarkQuestion:
    nl_query: str
    collection: str
    operation: str  # find | aggregate | count | distinct
    query: Any  # dict (filter) or list (pipeline)
    tags: str
    limit: int | None = None
    distinct_field: str | None = None
    sort: dict | None = None
    projection: dict | None = None


# ===========================================================================
# 1 — Simple counts (easy) — several embed drift/boolean-dialect traps
# ===========================================================================

_Q_COUNTS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="How many merchants are active? Note some records store the status in upper case.",
        collection="merchants",
        operation="count",
        query={"status": {"$in": ["active", "ACTIVE"]}},
        tags="easy|count|drift|merchants",
    ),
    BenchmarkQuestion(
        nl_query="How many support tickets are currently open?",
        collection="support_tickets",
        operation="count",
        query={"status": "open"},
        tags="easy|count|support_tickets",
    ),
    BenchmarkQuestion(
        nl_query="How many catalog products are tangible (physical) goods?",
        collection="catalog",
        operation="count",
        query={"is_tangible": True},
        tags="easy|count|catalog",
    ),
    BenchmarkQuestion(
        nl_query="How many listings are live?",
        collection="listings",
        operation="count",
        query={"status": "live"},
        tags="easy|count|listings",
    ),
    BenchmarkQuestion(
        nl_query="How many users belong to the B2B segment?",
        collection="users",
        operation="count",
        query={"segment": "b2b"},
        tags="easy|count|users",
    ),
    BenchmarkQuestion(
        nl_query="How many orders were cancelled? Remember orders come in two schema versions.",
        collection="orders",
        operation="count",
        query={"$or": [{"status": "cancelled"}, {"state": "CANCELLED"}]},
        tags="medium|count|drift|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many loyalty accounts are in the platinum tier?",
        collection="loyalty_accounts",
        operation="count",
        query={"tier": "platinum"},
        tags="easy|count|loyalty",
    ),
    BenchmarkQuestion(
        nl_query="How many promotions are active?",
        collection="promotions",
        operation="count",
        query={"is_active": True},
        tags="easy|count|promotions",
    ),
    BenchmarkQuestion(
        nl_query="How many listings are the buy-box winner? Beware the flag uses mixed boolean encodings.",
        collection="listings",
        operation="count",
        query={"is_buybox_winner": {"$in": TRUTHY}},
        tags="medium|count|drift|listings",
    ),
    BenchmarkQuestion(
        nl_query="How many shipments are in an exception state?",
        collection="shipments",
        operation="count",
        query={"status": "exception"},
        tags="easy|count|shipments",
    ),
]


# ===========================================================================
# 2 — Field access / find / distinct (easy-medium)
# ===========================================================================

_Q_FIELD: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="List 10 warehouses that support cold-chain storage; show name, city and capabilities.",
        collection="warehouses",
        operation="find",
        query={"capabilities": "cold_chain"},
        projection={"_id": 0, "name": 1, "city": 1, "country": 1, "capabilities": 1},
        sort={"name": 1},
        limit=10,
        tags="easy|find|warehouses",
    ),
    BenchmarkQuestion(
        nl_query="Show 10 past-due subscriptions with their plan and monthly recurring revenue in EUR.",
        collection="subscriptions",
        operation="find",
        query={"status": "past_due"},
        projection={"_id": 0, "plan": 1, "status": 1, "mrr_eur": 1, "interval": 1},
        sort={"mrr_eur": -1},
        limit=10,
        tags="medium|find|subscriptions",
    ),
    BenchmarkQuestion(
        nl_query="What distinct payment methods have been used?",
        collection="payments",
        operation="distinct",
        query={},
        distinct_field="method",
        tags="easy|distinct|payments",
    ),
    BenchmarkQuestion(
        nl_query="What distinct carriers appear in shipments?",
        collection="shipments",
        operation="distinct",
        query={},
        distinct_field="carrier",
        tags="easy|distinct|shipments",
    ),
    BenchmarkQuestion(
        nl_query="What distinct brands exist in the Electronics category tree?",
        collection="catalog",
        operation="distinct",
        query={"category_path": {"$regex": "^Electronics"}},
        distinct_field="brand",
        tags="medium|distinct|catalog",
    ),
    BenchmarkQuestion(
        nl_query="List the 10 highest-commission merchants; show display name, country and commission percentage.",
        collection="merchants",
        operation="find",
        query={"deleted_at": {"$exists": False}},
        projection={"_id": 0, "display_name": 1, "country": 1, "commission_pct": 1, "type": 1},
        sort={"commission_pct": -1, "merchant_code": 1},
        limit=10,
        tags="medium|find|merchants",
    ),
]


# ===========================================================================
# 3 — Arrays / $unwind / $size (medium)
# ===========================================================================

_Q_ARRAY: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="How many orders are split across more than one merchant group?",
        collection="orders",
        operation="count",
        query={"$expr": {"$gt": [{"$size": "$groups"}, 1]}},
        tags="medium|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many listings are there for each condition (new, refurbished, used...)?",
        collection="listings",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$condition", "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "_id": 1}},
        ],
        tags="medium|aggregate|listings",
    ),
    BenchmarkQuestion(
        nl_query="What is the average number of variants per catalog product?",
        collection="catalog",
        operation="aggregate",
        query=[
            {"$group": {"_id": None, "avg_variants": {"$avg": {"$size": "$variants"}}}},
            {"$project": {"_id": 0, "avg_variants": {"$round": ["$avg_variants", 3]}}},
        ],
        tags="medium|array|catalog",
    ),
    BenchmarkQuestion(
        nl_query="Total units sold per product variant across all orders — top 15 variants.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$groups"},
            {"$unwind": "$groups.items"},
            {"$group": {"_id": "$groups.items.variant_id", "units": {"$sum": "$groups.items.qty"}}},
            {"$sort": {"units": -1, "_id": 1}},
            {"$limit": 15},
            {"$project": {"_id": 0, "variant_id": "$_id", "units": 1}},
        ],
        tags="hard|array|unwind|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many shipments contain more than one package?",
        collection="shipments",
        operation="count",
        query={"$expr": {"$gt": [{"$size": "$packages"}, 1]}},
        tags="medium|array|shipments",
    ),
    BenchmarkQuestion(
        nl_query="Count promotions by their action kind (percentage_off, free_shipping, tiered, ...).",
        collection="promotions",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$action.kind", "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "_id": 1}},
        ],
        tags="medium|aggregate|nested|promotions",
    ),
]


# ===========================================================================
# 4 — Aggregation (medium-hard) — drift & multi-currency handling
# ===========================================================================

_Q_AGG: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="Total order revenue in EUR — sum the EUR grand total across both order schema versions.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": None,
                        "revenue_eur": {"$sum": {"$ifNull": ["$grand_total_eur", "$totals.grand_total_eur"]}}}},
            {"$project": {"_id": 0, "revenue_eur": {"$round": ["$revenue_eur", 2]}}},
        ],
        tags="hard|aggregate|drift|currency|orders",
    ),
    BenchmarkQuestion(
        nl_query="Order revenue in EUR broken down by sales channel.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$channel",
                        "revenue_eur": {"$sum": {"$ifNull": ["$grand_total_eur", "$totals.grand_total_eur"]}}}},
            {"$sort": {"revenue_eur": -1, "_id": 1}},
            {"$project": {"_id": 0, "channel": "$_id",
                          "revenue_eur": {"$round": ["$revenue_eur", 2]}}},
        ],
        tags="hard|aggregate|drift|currency|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the average order value in EUR (across both schema versions)?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": None,
                        "aov_eur": {"$avg": {"$ifNull": ["$grand_total_eur", "$totals.grand_total_eur"]}}}},
            {"$project": {"_id": 0, "aov_eur": {"$round": ["$aov_eur", 2]}}},
        ],
        tags="hard|aggregate|drift|currency|orders",
    ),
    BenchmarkQuestion(
        nl_query="Total captured payment amount in EUR per payment method.",
        collection="payments",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$method", "captured_eur": {"$sum": "$captured_eur"}}},
            {"$sort": {"captured_eur": -1, "_id": 1}},
            {"$project": {"_id": 0, "method": "$_id",
                          "captured_eur": {"$round": ["$captured_eur", 2]}}},
        ],
        tags="medium|aggregate|payments",
    ),
    BenchmarkQuestion(
        nl_query="Average review rating on a 1-5 scale, normalising legacy 1-10 reviews (divide their score by 2).",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": None, "avg_rating_5": {"$avg": {"$switch": {"branches": [
                {"case": {"$eq": ["$scale", "five"]}, "then": "$rating"},
                {"case": {"$eq": ["$scale", "ten"]}, "then": {"$divide": ["$score", 2]}},
            ], "default": None}}}}},
            {"$project": {"_id": 0, "avg_rating_5": {"$round": ["$avg_rating_5", 3]}}},
        ],
        tags="hard|aggregate|drift|reviews",
    ),
    BenchmarkQuestion(
        nl_query="Monthly recurring revenue (EUR) summed per subscription plan.",
        collection="subscriptions",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$plan", "mrr_eur": {"$sum": "$mrr_eur"}}},
            {"$sort": {"mrr_eur": -1, "_id": 1}},
            {"$project": {"_id": 0, "plan": "$_id", "mrr_eur": {"$round": ["$mrr_eur", 2]}}},
        ],
        tags="medium|aggregate|subscriptions",
    ),
    BenchmarkQuestion(
        nl_query="Net balance per ledger account, treating credits as positive and debits as negative.",
        collection="ledger_entries",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$account", "net_eur": {"$sum": {
                "$cond": [{"$eq": ["$side", "credit"]}, "$amount_eur", {"$multiply": ["$amount_eur", -1]}]}}}},
            {"$sort": {"net_eur": -1, "_id": 1}},
            {"$project": {"_id": 0, "account": "$_id", "net_eur": {"$round": ["$net_eur", 2]}}},
        ],
        tags="hard|aggregate|nested|ledger",
    ),
    BenchmarkQuestion(
        nl_query="Count events by type.",
        collection="events",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "_id": 1}},
        ],
        tags="medium|aggregate|events",
    ),
    BenchmarkQuestion(
        nl_query="Total captured payment amount in EUR in 2025.",
        collection="payments",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2025_START, "$lte": D2025_END}}},
            {"$group": {"_id": None, "captured_eur": {"$sum": "$captured_eur"}}},
            {"$project": {"_id": 0, "captured_eur": {"$round": ["$captured_eur", 2]}}},
        ],
        tags="medium|aggregate|date|payments",
    ),
]


# ===========================================================================
# 5 — Lookup / multi-hop joins (hard)
# ===========================================================================

_Q_LOOKUP: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="Top 10 merchants by amount payable to them (ledger merchant_payable account); show display name and country.",
        collection="ledger_entries",
        operation="aggregate",
        query=[
            {"$match": {"account": "merchant_payable"}},
            {"$group": {"_id": "$merchant_id", "payable_eur": {"$sum": "$amount_eur"}}},
            {"$sort": {"payable_eur": -1, "_id": 1}},
            {"$limit": 10},
            {"$lookup": {"from": "merchants", "localField": "_id", "foreignField": "_id", "as": "m"}},
            {"$unwind": "$m"},
            {"$project": {"_id": 0, "merchant": "$m.display_name", "country": "$m.country",
                          "payable_eur": {"$round": ["$payable_eur", 2]}}},
        ],
        tags="hard|lookup|ledger|merchants",
    ),
    BenchmarkQuestion(
        nl_query="Top 10 best-selling catalog products by units sold; show product title and brand.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$groups"},
            {"$unwind": "$groups.items"},
            {"$group": {"_id": "$groups.items.product_id", "units": {"$sum": "$groups.items.qty"}}},
            {"$sort": {"units": -1, "_id": 1}},
            {"$limit": 10},
            {"$lookup": {"from": "catalog", "localField": "_id", "foreignField": "_id", "as": "p"}},
            {"$unwind": "$p"},
            {"$project": {"_id": 0, "title": "$p.title", "brand": "$p.brand", "units": 1}},
        ],
        tags="hard|lookup|unwind|orders|catalog",
    ),
    BenchmarkQuestion(
        nl_query="Total order subtotal (EUR) attributed to each merchant type (brand, reseller, ...).",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$groups"},
            {"$group": {"_id": "$groups.merchant_id", "subtotal_eur": {"$sum": "$groups.group_subtotal_eur"}}},
            {"$lookup": {"from": "merchants", "localField": "_id", "foreignField": "_id", "as": "m"}},
            {"$unwind": "$m"},
            {"$group": {"_id": "$m.type", "subtotal_eur": {"$sum": "$subtotal_eur"}}},
            {"$sort": {"subtotal_eur": -1, "_id": 1}},
            {"$project": {"_id": 0, "merchant_type": "$_id", "subtotal_eur": {"$round": ["$subtotal_eur", 2]}}},
        ],
        tags="hard|lookup|multihop|orders|merchants",
    ),
    BenchmarkQuestion(
        nl_query="Average normalised review rating (1-5) per brand; top 15 brands by review count.",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$lookup": {"from": "catalog", "localField": "product_id", "foreignField": "_id", "as": "p"}},
            {"$unwind": "$p"},
            {"$group": {"_id": "$p.brand",
                        "avg_rating_5": {"$avg": {"$switch": {"branches": [
                            {"case": {"$eq": ["$scale", "five"]}, "then": "$rating"},
                            {"case": {"$eq": ["$scale", "ten"]}, "then": {"$divide": ["$score", 2]}},
                        ], "default": None}}},
                        "reviews": {"$sum": 1}}},
            {"$sort": {"reviews": -1, "_id": 1}},
            {"$limit": 15},
            {"$project": {"_id": 0, "brand": "$_id", "reviews": 1,
                          "avg_rating_5": {"$round": ["$avg_rating_5", 3]}}},
        ],
        tags="hard|lookup|drift|reviews|catalog",
    ),
    BenchmarkQuestion(
        nl_query="Top 10 users by number of orders; show their segment, country and order count (orders use two schema versions).",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": {"$ifNull": ["$user_id", "$customer_id"]}, "orders": {"$sum": 1}}},
            {"$sort": {"orders": -1, "_id": 1}},
            {"$limit": 10},
            {"$lookup": {"from": "users", "localField": "_id", "foreignField": "_id", "as": "u"}},
            {"$unwind": "$u"},
            {"$project": {"_id": 0, "segment": "$u.segment", "country": "$u.country", "orders": 1}},
        ],
        tags="hard|lookup|drift|orders|users",
    ),
    BenchmarkQuestion(
        nl_query="Average delivery time in days per carrier for delivered shipments.",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$match": {"status": "delivered", "actual_delivery": {"$ne": None}}},
            {"$project": {"carrier": 1,
                          "days": {"$dateDiff": {"startDate": "$created_at", "endDate": "$actual_delivery", "unit": "day"}}}},
            {"$group": {"_id": "$carrier", "avg_days": {"$avg": "$days"}, "shipments": {"$sum": 1}}},
            {"$sort": {"avg_days": 1, "_id": 1}},
            {"$project": {"_id": 0, "carrier": "$_id", "shipments": 1, "avg_days": {"$round": ["$avg_days", 2]}}},
        ],
        tags="hard|aggregate|date|shipments",
    ),
    BenchmarkQuestion(
        nl_query="For each warehouse, how many inventory snapshots are below their reorder threshold? Show warehouse city.",
        collection="inventory_snapshots",
        operation="aggregate",
        query=[
            {"$match": {"below_threshold": True}},
            {"$group": {"_id": "$warehouse_id", "below": {"$sum": 1}}},
            {"$sort": {"below": -1, "_id": 1}},
            {"$lookup": {"from": "warehouses", "localField": "_id", "foreignField": "_id", "as": "w"}},
            {"$unwind": "$w"},
            {"$project": {"_id": 0, "warehouse": "$w.name", "city": "$w.city", "below": 1}},
        ],
        tags="hard|lookup|inventory|warehouses",
    ),
    BenchmarkQuestion(
        nl_query="Units returned per return reason across all RMAs.",
        collection="returns",
        operation="aggregate",
        query=[
            {"$unwind": "$lines"},
            {"$group": {"_id": "$lines.reason", "units": {"$sum": "$lines.qty"}}},
            {"$sort": {"units": -1, "_id": 1}},
            {"$project": {"_id": 0, "reason": "$_id", "units": 1}},
        ],
        tags="hard|array|unwind|returns",
    ),
]


# ===========================================================================
# 6 — Drift-aware single-value answers (medium-hard)
# ===========================================================================

_Q_DRIFT: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="How many orders have been delivered? Legacy orders record this as state 'COMPLETE'.",
        collection="orders",
        operation="count",
        query={"$or": [{"status": "delivered"}, {"state": "COMPLETE"}]},
        tags="hard|count|drift|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many merchants are verified? The verification flag uses mixed boolean encodings.",
        collection="merchants",
        operation="count",
        query={"is_verified": {"$in": TRUTHY}},
        tags="medium|count|drift|merchants",
    ),
    BenchmarkQuestion(
        nl_query="How many grocery products are organic? Watch for mixed boolean encodings on the attribute.",
        collection="catalog",
        operation="count",
        query={"attr_schema": "grocery", "attributes.organic": {"$in": TRUTHY}},
        tags="hard|count|drift|nested|catalog",
    ),
    BenchmarkQuestion(
        nl_query="How many reviews use the legacy 1-10 rating scale versus the modern 1-5 scale?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$scale", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ],
        tags="medium|aggregate|drift|reviews",
    ),
    BenchmarkQuestion(
        nl_query="How many orders were stored using the legacy v1 schema?",
        collection="orders",
        operation="count",
        query={"_schema_v": 1},
        tags="medium|count|drift|orders",
    ),
    BenchmarkQuestion(
        nl_query="Count orders by a normalised status that merges the legacy state vocabulary into the modern one.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": {"$ifNull": ["$status", {"$switch": {"branches": [
                {"case": {"$eq": ["$state", "NEW"]}, "then": "created"},
                {"case": {"$eq": ["$state", "PROCESSING"]}, "then": "paid"},
                {"case": {"$eq": ["$state", "SENT"]}, "then": "shipped"},
                {"case": {"$eq": ["$state", "COMPLETE"]}, "then": "delivered"},
                {"case": {"$eq": ["$state", "CANCELLED"]}, "then": "cancelled"},
                {"case": {"$eq": ["$state", "REFUND"]}, "then": "refunded"},
            ], "default": {"$toLower": "$state"}}}]}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "_id": 1}},
            {"$project": {"_id": 0, "status": "$_id", "count": 1}},
        ],
        tags="hard|aggregate|drift|orders",
    ),
]


# ===========================================================================
# 7 — Polymorphic / nested payloads (medium-hard)
# ===========================================================================

_Q_POLY: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="Distribution of card payments by card network (visa, mastercard, ...).",
        collection="payments",
        operation="aggregate",
        query=[
            {"$match": {"method": "card"}},
            {"$group": {"_id": "$details.network", "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "_id": 1}},
            {"$project": {"_id": 0, "network": "$_id", "count": 1}},
        ],
        tags="hard|aggregate|polymorphic|payments",
    ),
    BenchmarkQuestion(
        nl_query="Top 10 search queries by frequency from the events stream.",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"type": "search"}},
            {"$group": {"_id": "$payload.q", "searches": {"$sum": 1}}},
            {"$sort": {"searches": -1, "_id": 1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "query": "$_id", "searches": 1}},
        ],
        tags="hard|aggregate|polymorphic|events",
    ),
    BenchmarkQuestion(
        nl_query="Average first-response time in minutes per support ticket priority (only tickets that got a response).",
        collection="support_tickets",
        operation="aggregate",
        query=[
            {"$match": {"first_response_minutes": {"$ne": None}}},
            {"$group": {"_id": "$priority", "avg_first_response_min": {"$avg": "$first_response_minutes"},
                        "tickets": {"$sum": 1}}},
            {"$sort": {"avg_first_response_min": -1, "_id": 1}},
            {"$project": {"_id": 0, "priority": "$_id", "tickets": 1,
                          "avg_first_response_min": {"$round": ["$avg_first_response_min", 1]}}},
        ],
        tags="medium|aggregate|support_tickets",
    ),
    BenchmarkQuestion(
        nl_query="For Klarna payments, how many use each installment plan length?",
        collection="payments",
        operation="aggregate",
        query=[
            {"$match": {"method": "klarna"}},
            {"$group": {"_id": "$details.installments", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "installments": "$_id", "count": 1}},
        ],
        tags="hard|aggregate|polymorphic|payments",
    ),
    BenchmarkQuestion(
        nl_query="What are the latest EUR->USD and EUR->GBP exchange rates by period?",
        collection="fx_rates",
        operation="aggregate",
        query=[
            {"$match": {"quote": {"$in": ["USD", "GBP"]}}},
            {"$sort": {"valid_from": -1}},
            {"$group": {"_id": "$quote", "latest_period": {"$first": "$period"}, "rate": {"$first": "$rate"}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "quote": "$_id", "latest_period": 1, "rate": 1}},
        ],
        tags="hard|aggregate|currency|fx_rates",
    ),
]


# ===========================================================================
# Master list
# ===========================================================================

QUESTIONS: list[BenchmarkQuestion] = (
    _Q_COUNTS
    + _Q_FIELD
    + _Q_ARRAY
    + _Q_AGG
    + _Q_LOOKUP
    + _Q_DRIFT
    + _Q_POLY
)

assert len(QUESTIONS) == 50, f"Expected 50 questions, got {len(QUESTIONS)}"
