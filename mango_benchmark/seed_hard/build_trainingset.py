"""Build a trainingset JSONL for the mango_marketplace HARD benchmark.

This produces a file to inject into the agent prompt via ``--training-file``.

Design choice: the training examples are DISTINCT from the 50 benchmark
questions (different collections / filters / phrasings). They teach the messy
conventions of this DB without leaking any benchmark answer, so you can inject
ALL of them and still evaluate on the full 50-question benchmark.

The file contains:
  * ``text`` entries     — schema notes (order drift, boolean dialects, EUR
    snapshot fields, the two review scales, mixed date encodings, ...).
  * ``training`` entries — example ``question -> run_mql`` calls. Each query is
    executed against the live DB so ``result_summary`` is real.

Usage:
    python -m mango_benchmark.seed_hard.build_trainingset
    python -m mango_benchmark.seed_hard.build_trainingset --uri mongodb://localhost:27017
    python -m mango_benchmark.seed_hard.build_trainingset --out examples/trainingset_marketplace.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from pymongo import MongoClient

from mango_benchmark.seed_hard.schema import DB_NAME

_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_OUT = _ROOT / "examples" / "trainingset_marketplace.jsonl"

D2025_START = datetime(2025, 1, 1)
D2025_END = datetime(2025, 12, 31, 23, 59, 59)
TRUTHY = [True, "Y", "true", 1, "yes"]


def _json_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Schema notes — teach the conventions.
# ---------------------------------------------------------------------------

TEXT_NOTES: list[str] = [
    "Database: mango_marketplace — a multi-vendor marketplace + light fintech. "
    "Collections: users, merchants, catalog, listings, categories, warehouses, "
    "currencies, fx_rates, inventory_snapshots, promotions, orders, payments, "
    "shipments, returns, reviews, subscriptions, loyalty_accounts, "
    "loyalty_transactions, support_tickets, ledger_entries, audit_logs, events.",

    "BUYERS live in 'users' (NOT 'customers'). PRODUCTS live in 'catalog' (NOT "
    "'products'); each catalog doc has polymorphic 'attributes' keyed by "
    "'attr_schema' and a nested 'variants' array (each variant has an 'options' "
    "array). SELLERS live in 'merchants' (self-referencing via parent_merchant_id). "
    "'listings' is the M:N link between a merchant and a catalog product/variant.",

    "ORDER SCHEMA DRIFT: ~20% of orders are legacy v1 (_schema_v=1) using fields "
    "'customer_id', 'ccy', 'state', 'placed_at', top-level 'grand_total_eur'. The "
    "rest are v2 (_schema_v=2) using 'user_id', 'currency', 'status', 'created_at', "
    "and nested 'totals.grand_total_eur'. For a user id use "
    "{$ifNull:['$user_id','$customer_id']}; for EUR total use "
    "{$ifNull:['$grand_total_eur','$totals.grand_total_eur']}.",

    "ORDER STATUS DRIFT: v2 'status' = created/paid/partially_shipped/shipped/"
    "delivered/cancelled/refunded/partially_refunded/disputed. Legacy v1 'state' "
    "uses NEW/PROCESSING/SENT/COMPLETE/CANCELLED/REFUND. delivered==state "
    "'COMPLETE'; shipped==state 'SENT'; cancelled==state 'CANCELLED'. Query both, "
    "e.g. {$or:[{status:'delivered'},{state:'COMPLETE'}]}.",

    "MULTI-CURRENCY: monetary fields are in the row's own currency, but every "
    "money field has an EUR snapshot sibling ending in '_eur' (grand_total_eur, "
    "amount_eur, captured_eur, price_eur, line_total_eur, mrr_eur). For EUR "
    "aggregates ALWAYS sum the *_eur fields. The 'fx_rates' collection (base=EUR, "
    "quote=<code>, rate, period 'YYYY-MM', valid_from) holds EUR->currency rates.",

    "BOOLEAN DIALECTS: messy flags may be true / 'Y' / 'true' / 1 / 'yes'. Match "
    "truthy with {$in:[true,'Y','true',1,'yes']}. Affected: merchants.is_verified, "
    "listings.is_buybox_winner, catalog.attributes.organic, subscriptions.auto_renew, "
    "orders.is_gift.",

    "ENUM CASE DRIFT: merchants.status is mostly 'active' but some rows store "
    "'ACTIVE'. Match with {$in:['active','ACTIVE']}.",

    "REVIEW RATING SCALES: reviews have a 'scale' field. scale='five' rows have a "
    "1-5 'rating'; legacy scale='ten' rows have a 1-10 'score' instead. To average "
    "on a 1-5 basis use $switch: five->rating, ten->score/2.",

    "DEEP NESTING: order line items are at groups[].items[] (per-merchant groups) — "
    "use two $unwind stages. payments.details, events.payload and promotions.action "
    "are polymorphic sub-documents whose keys depend on method/type/kind.",

    "MIXED DATE ENCODINGS: events.ts, audit_logs.at, orders.placed_at, "
    "inventory_snapshots.snapshot_at, support_tickets.messages[].at and "
    "users.last_login_at mix ISODate/epoch-int/string — avoid range filters on "
    "them; prefer clean datetime fields (created_at, requested_at, opened_at, "
    "posted_at, listed_at, valid_from, started_at).",

    "DOUBLE-ENTRY LEDGER: ledger_entries have account, side ('debit'|'credit'), "
    "amount_eur, merchant_id, posted_at. Net = sum(credit) - sum(debit) via "
    "{$cond:[{$eq:['$side','credit']},'$amount_eur',{$multiply:['$amount_eur',-1]}]}.",
]


# ---------------------------------------------------------------------------
# Example queries — DISTINCT from the 50 benchmark questions (no leakage).
# (question, tool_args)
# ---------------------------------------------------------------------------

EXAMPLES: list[tuple[str, dict]] = [
    ("How many users are in each segment?", {
        "operation": "aggregate", "collection": "users",
        "pipeline": [{"$group": {"_id": "$segment", "count": {"$sum": 1}}},
                     {"$sort": {"count": -1, "_id": 1}}]}),
    ("How many merchants are there of each type?", {
        "operation": "aggregate", "collection": "merchants",
        "pipeline": [{"$group": {"_id": "$type", "count": {"$sum": 1}}},
                     {"$sort": {"count": -1, "_id": 1}}]}),
    ("How many merchants are suspended?", {
        "operation": "count", "collection": "merchants",
        "filter": {"status": "suspended"}}),
    ("What distinct subscription plans exist?", {
        "operation": "distinct", "collection": "subscriptions", "distinct_field": "plan"}),
    ("What distinct loyalty tiers exist?", {
        "operation": "distinct", "collection": "loyalty_accounts", "distinct_field": "tier"}),
    ("How many catalog products exist per attribute schema?", {
        "operation": "aggregate", "collection": "catalog",
        "pipeline": [{"$group": {"_id": "$attr_schema", "count": {"$sum": 1}}},
                     {"$sort": {"count": -1, "_id": 1}}]}),
    ("How many subscriptions are active?", {
        "operation": "count", "collection": "subscriptions", "filter": {"status": "active"}}),
    ("What is the average merchant commission percentage?", {
        "operation": "aggregate", "collection": "merchants",
        "pipeline": [{"$group": {"_id": None, "avg_commission": {"$avg": "$commission_pct"}}},
                     {"$project": {"_id": 0, "avg_commission": {"$round": ["$avg_commission", 2]}}}]}),
    ("How many payments are there in each payment status?", {
        "operation": "aggregate", "collection": "payments",
        "pipeline": [{"$group": {"_id": "$status", "count": {"$sum": 1}}},
                     {"$sort": {"count": -1, "_id": 1}}]}),
    ("What is the total value in EUR of all payment refunds?", {
        "operation": "aggregate", "collection": "payments",
        "pipeline": [{"$unwind": "$refunds"},
                     {"$group": {"_id": None, "refunded_eur": {"$sum": "$refunds.amount_eur"}}},
                     {"$project": {"_id": 0, "refunded_eur": {"$round": ["$refunded_eur", 2]}}}]}),
    ("How many returns are there in each return status?", {
        "operation": "aggregate", "collection": "returns",
        "pipeline": [{"$group": {"_id": "$status", "count": {"$sum": 1}}},
                     {"$sort": {"count": -1, "_id": 1}}]}),
    ("How many shipments use each service level?", {
        "operation": "aggregate", "collection": "shipments",
        "pipeline": [{"$group": {"_id": "$service_level", "count": {"$sum": 1}}},
                     {"$sort": {"count": -1, "_id": 1}}]}),
    ("How many events come from each device type?", {
        "operation": "aggregate", "collection": "events",
        "pipeline": [{"$group": {"_id": "$context.device", "count": {"$sum": 1}}},
                     {"$sort": {"count": -1, "_id": 1}}]}),
    ("Which 5 brands have the most catalog products?", {
        "operation": "aggregate", "collection": "catalog",
        "pipeline": [{"$group": {"_id": "$brand", "products": {"$sum": 1}}},
                     {"$sort": {"products": -1, "_id": 1}}, {"$limit": 5},
                     {"$project": {"_id": 0, "brand": "$_id", "products": 1}}]}),
    ("How many loyalty transactions are there of each type?", {
        "operation": "aggregate", "collection": "loyalty_transactions",
        "pipeline": [{"$group": {"_id": "$type", "count": {"$sum": 1}}},
                     {"$sort": {"count": -1, "_id": 1}}]}),
    ("What is the average number of helpful votes per review?", {
        "operation": "aggregate", "collection": "reviews",
        "pipeline": [{"$group": {"_id": None, "avg_helpful": {"$avg": "$helpful_votes"}}},
                     {"$project": {"_id": 0, "avg_helpful": {"$round": ["$avg_helpful", 2]}}}]}),
    ("How many support tickets are there in each category?", {
        "operation": "aggregate", "collection": "support_tickets",
        "pipeline": [{"$group": {"_id": "$category", "count": {"$sum": 1}}},
                     {"$sort": {"count": -1, "_id": 1}}]}),
    ("How many inventory snapshots are below their reorder threshold?", {
        "operation": "count", "collection": "inventory_snapshots",
        "filter": {"below_threshold": True}}),
    ("What is the total revenue (EUR) recorded in the ledger revenue account?", {
        "operation": "aggregate", "collection": "ledger_entries",
        "pipeline": [{"$match": {"account": "revenue"}},
                     {"$group": {"_id": None, "revenue_eur": {"$sum": "$amount_eur"}}},
                     {"$project": {"_id": 0, "revenue_eur": {"$round": ["$revenue_eur", 2]}}}]}),
    ("How many orders have been shipped? (legacy orders store this as state 'SENT')", {
        "operation": "count", "collection": "orders",
        "filter": {"$or": [{"status": "shipped"}, {"state": "SENT"}]}}),
    ("How many subscriptions have auto-renew enabled? (the flag uses mixed boolean encodings)", {
        "operation": "count", "collection": "subscriptions",
        "filter": {"auto_renew": {"$in": TRUTHY}}}),
    ("What is the latest EUR to CHF exchange rate?", {
        "operation": "find", "collection": "fx_rates",
        "filter": {"quote": "CHF"}, "projection": {"_id": 0, "quote": 1, "rate": 1, "period": 1},
        "sort": {"valid_from": -1}, "limit": 1}),
    ("What distinct crypto assets have been used in crypto payments?", {
        "operation": "distinct", "collection": "payments",
        "filter": {"method": "crypto"}, "distinct_field": "details.asset"}),
    ("What is the average order value in EUR per currency? (handle both order schema versions)", {
        "operation": "aggregate", "collection": "orders",
        "pipeline": [{"$group": {"_id": {"$ifNull": ["$currency", "$ccy"]},
                                 "aov_eur": {"$avg": {"$ifNull": ["$grand_total_eur", "$totals.grand_total_eur"]}},
                                 "orders": {"$sum": 1}}},
                     {"$sort": {"orders": -1, "_id": 1}},
                     {"$project": {"_id": 0, "currency": "$_id", "orders": 1,
                                   "aov_eur": {"$round": ["$aov_eur", 2]}}}]}),
]


# ---------------------------------------------------------------------------
# Execution + summary
# ---------------------------------------------------------------------------


def _run(db, args: dict):
    col = db[args["collection"]]
    op = args["operation"]
    if op == "count":
        return col.count_documents(args.get("filter", {}))
    if op == "distinct":
        return sorted(str(v) for v in col.distinct(args["distinct_field"], args.get("filter", {})))
    if op == "find":
        cur = col.find(args.get("filter", {}), args.get("projection") or {})
        if args.get("sort"):
            cur = cur.sort(list(args["sort"].items()))
        if args.get("limit"):
            cur = cur.limit(args["limit"])
        return list(cur)
    if op == "aggregate":
        return list(col.aggregate(args["pipeline"], allowDiskUse=True))
    raise ValueError(op)


def _summary(result) -> str:
    if isinstance(result, bool):
        return f"Result: {result}"
    if isinstance(result, (int, float)):
        return f"Result: {result}"
    if isinstance(result, list):
        if not result:
            return "Result: 0 item(s)"
        if all(not isinstance(x, (dict, list)) for x in result):
            sample = ", ".join(str(x) for x in result[:8])
            more = "" if len(result) <= 8 else ", ..."
            return f"Result: {len(result)} value(s): [{sample}{more}]"
        keys = sorted(k for k in result[0].keys() if k != "_id") if isinstance(result[0], dict) else []
        return f"Result: {len(result)} item(s) — fields: {keys}"
    return f"Result: {str(result)[:120]}"


def _structural_summary(args: dict) -> str:
    """Fallback result_summary when the DB is unreachable (no exact numbers)."""
    op = args["operation"]
    return {
        "count": "Result: <count>",
        "distinct": "Result: list of distinct values",
        "find": "Result: matching document(s)",
        "aggregate": "Result: grouped row(s)",
    }.get(op, "Result: …")


def build(uri: str, out_path: Path) -> None:
    db = None
    try:
        client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
        client.admin.command("ping")
        db = client[DB_NAME]
    except Exception as exc:
        print(f"WARNING: DB unreachable ({str(exc)[:60]}…) — writing structural "
              f"result summaries instead of real values.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for note in TEXT_NOTES:
            f.write(json.dumps({"type": "text", "text": note}, ensure_ascii=False) + "\n")
        for question, args in EXAMPLES:
            summary = _summary(_run(db, args)) if db is not None else _structural_summary(args)
            entry = {
                "type": "training",
                "question": question,
                "tool_name": "run_mql",
                "tool_args": args,
                "result_summary": summary,
            }
            f.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")

    print(f"Wrote {len(TEXT_NOTES)} text notes + {len(EXAMPLES)} training examples → {out_path}")
    print("These examples are DISTINCT from the 50 benchmark questions — inject them all "
          "and evaluate on the full mango_marketplace_benchmark.csv (no leakage).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build mango_marketplace trainingset JSONL.")
    ap.add_argument("--uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    ap.add_argument("--out", default=str(_DEFAULT_OUT))
    args = ap.parse_args()
    build(args.uri, Path(args.out))


if __name__ == "__main__":
    main()
