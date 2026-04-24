"""180 benchmark questions for the mango_ecommerce database.

Each BenchmarkQuestion contains:
- nl_query: natural language question
- collection: primary collection
- operation: find | aggregate | count | distinct
- query: pymongo-compatible filter (for find/count) or pipeline (for aggregate)
- tags: pipe-separated difficulty|category labels

Date constants:
  DATE_2023: 2023-01-01 to 2023-12-31
  DATE_2024: 2024-01-01 to 2024-12-31
  DATE_2025: 2025-01-01 to 2025-12-31
  Q1_2023:   2023-01-01 to 2023-03-31
  Q4_2024:   2024-10-01 to 2024-12-31
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Date boundaries used by queries
# ---------------------------------------------------------------------------

D2023_START = datetime(2023, 1, 1)
D2023_END = datetime(2023, 12, 31, 23, 59, 59)
D2024_START = datetime(2024, 1, 1)
D2024_END = datetime(2024, 12, 31, 23, 59, 59)
D2025_START = datetime(2025, 1, 1)
D2025_END = datetime(2025, 12, 31, 23, 59, 59)
Q1_2023_START = datetime(2023, 1, 1)
Q1_2023_END = datetime(2023, 3, 31, 23, 59, 59)
Q4_2024_START = datetime(2024, 10, 1)
Q4_2024_END = datetime(2024, 12, 31, 23, 59, 59)
DEC_2024_START = datetime(2024, 12, 1)
DEC_2024_END = datetime(2024, 12, 31, 23, 59, 59)
MID_2024 = datetime(2024, 6, 1)

Q1_2024_START = datetime(2024, 1, 1)
Q1_2024_END   = datetime(2024, 3, 31, 23, 59, 59)
Q2_2024_START = datetime(2024, 4, 1)
Q2_2024_END   = datetime(2024, 6, 30, 23, 59, 59)
Q3_2024_START = datetime(2024, 7, 1)
Q3_2024_END   = datetime(2024, 9, 30, 23, 59, 59)
H1_2024_START = datetime(2024, 1, 1)
H1_2024_END   = datetime(2024, 6, 30, 23, 59, 59)
H2_2024_START = datetime(2024, 7, 1)
H2_2024_END   = datetime(2024, 12, 31, 23, 59, 59)
JAN_2024_START = datetime(2024, 1, 1)
JAN_2024_END   = datetime(2024, 1, 31, 23, 59, 59)
Q1_2025_START = datetime(2025, 1, 1)
Q1_2025_END   = datetime(2025, 3, 31, 23, 59, 59)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkQuestion:
    nl_query: str
    collection: str
    operation: str  # find | aggregate | count | distinct
    query: Any  # dict (filter) or list (pipeline)
    tags: str  # e.g. "easy|count"
    limit: int | None = None
    distinct_field: str | None = None
    sort: dict | None = None
    projection: dict | None = None


# ---------------------------------------------------------------------------
# CATEGORY 1 — Simple Counts (10 questions, easy)
# ---------------------------------------------------------------------------

_Q_COUNTS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="How many customers are in the gold tier?",
        collection="customers",
        operation="count",
        query={"tier": "gold"},
        tags="easy|count|customers",
    ),
    BenchmarkQuestion(
        nl_query="How many products are currently active?",
        collection="products",
        operation="count",
        query={"is_active": True},
        tags="easy|count|products",
    ),
    BenchmarkQuestion(
        nl_query="How many orders have been cancelled?",
        collection="orders",
        operation="count",
        query={"status": "cancelled"},
        tags="easy|count|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many reviews have a 5-star rating?",
        collection="reviews",
        operation="count",
        query={"rating": 5},
        tags="easy|count|reviews",
    ),
    BenchmarkQuestion(
        nl_query="How many customers opted in to marketing communications?",
        collection="customers",
        operation="count",
        query={"marketing_opt_in": True},
        tags="easy|count|customers",
    ),
    BenchmarkQuestion(
        nl_query="How many warehouses are currently active?",
        collection="warehouses",
        operation="count",
        query={"is_active": True},
        tags="easy|count|warehouses",
    ),
    BenchmarkQuestion(
        nl_query="How many suppliers are based in Italy?",
        collection="suppliers",
        operation="count",
        query={"country": "Italy"},
        tags="easy|count|suppliers",
    ),
    BenchmarkQuestion(
        nl_query="How many products have a rating count above 100?",
        collection="products",
        operation="count",
        query={"rating_count": {"$gt": 100}},
        tags="easy|count|products",
    ),
    BenchmarkQuestion(
        nl_query="How many inventory items are below their reorder threshold?",
        collection="inventory",
        operation="count",
        query={"below_threshold": True},
        tags="easy|count|inventory",
    ),
    BenchmarkQuestion(
        nl_query="How many orders were paid with credit card?",
        collection="orders",
        operation="count",
        query={"payment_method": "credit_card"},
        tags="easy|count|orders",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 2 — Field Access & Filter (10 questions, easy-medium)
# ---------------------------------------------------------------------------

_Q_FIELD_ACCESS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="What is the average price of all active products?",
        collection="products",
        operation="aggregate",
        query=[
            {"$match": {"is_active": True}},
            {"$group": {"_id": None, "avg_price": {"$avg": "$price"}}},
            {"$project": {"_id": 0, "avg_price": {"$round": ["$avg_price", 2]}}},
        ],
        tags="easy|aggregate|products",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 5 most expensive active products with their prices?",
        collection="products",
        operation="find",
        query={"is_active": True},
        sort={"price": -1},
        limit=5,
        projection={"name": 1, "price": 1, "brand": 1, "sku": 1, "_id": 0},
        tags="easy|find|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the average supplier rating by country?",
        collection="suppliers",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$country", "avg_rating": {"$avg": "$rating"}, "count": {"$sum": 1}}},
            {"$sort": {"avg_rating": -1}},
            {"$project": {"_id": 0, "country": "$_id", "avg_rating": {"$round": ["$avg_rating", 2]}, "count": 1}},
        ],
        tags="easy|aggregate|suppliers",
    ),
    BenchmarkQuestion(
        nl_query="What are the distinct payment methods used by customers?",
        collection="customers",
        operation="distinct",
        query={},
        distinct_field="preferred_payment",
        tags="easy|distinct|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the maximum and minimum order total?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": None, "max_total": {"$max": "$total"}, "min_total": {"$min": "$total"}}},
            {"$project": {"_id": 0, "max_total": 1, "min_total": 1}},
        ],
        tags="easy|aggregate|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many customers are in each tier?",
        collection="customers",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$tier", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$project": {"_id": 0, "tier": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the average number of items per order?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$project": {"item_count": {"$size": "$items"}}},
            {"$group": {"_id": None, "avg_items": {"$avg": "$item_count"}}},
            {"$project": {"_id": 0, "avg_items": {"$round": ["$avg_items", 2]}}},
        ],
        tags="easy|aggregate|orders",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 3 countries with the most customers?",
        collection="customers",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$country", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 3},
            {"$project": {"_id": 0, "country": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|customers",
    ),
    BenchmarkQuestion(
        nl_query="What are all distinct product brands in the catalog?",
        collection="products",
        operation="distinct",
        query={},
        distinct_field="brand",
        tags="easy|distinct|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the total number of products per top-level category?",
        collection="products",
        operation="aggregate",
        query=[
            {"$match": {"is_active": True}},
            {"$group": {"_id": {"$arrayElemAt": ["$category_path", 0]}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$project": {"_id": 0, "category": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|products",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 3 — Date Filtering (10 questions, medium)
# ---------------------------------------------------------------------------

_Q_DATE: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="How many orders were placed in 2024?",
        collection="orders",
        operation="count",
        query={"created_at": {"$gte": D2024_START, "$lte": D2024_END}},
        tags="medium|count|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many customers registered in the first quarter of 2023?",
        collection="customers",
        operation="count",
        query={"created_at": {"$gte": Q1_2023_START, "$lte": Q1_2023_END}},
        tags="medium|count|date|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the total revenue from orders placed in December 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": DEC_2024_START, "$lte": DEC_2024_END}}},
            {"$group": {"_id": None, "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many reviews were written in 2025?",
        collection="reviews",
        operation="count",
        query={"created_at": {"$gte": D2025_START, "$lte": D2025_END}},
        tags="medium|count|date|reviews",
    ),
    BenchmarkQuestion(
        nl_query="What is the total number of orders placed each month in 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$group": {"_id": {"$month": "$created_at"}, "order_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "month": "$_id", "order_count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the average order value by year?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": {"$year": "$created_at"}, "avg_order_value": {"$avg": "$total"}, "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "avg_order_value": {"$round": ["$avg_order_value", 2]}, "count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many events were recorded in 2023?",
        collection="events",
        operation="count",
        query={"timestamp": {"$gte": D2023_START, "$lte": D2023_END}},
        tags="medium|count|date|events",
    ),
    BenchmarkQuestion(
        nl_query="What is the monthly revenue trend for 2023?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2023_START, "$lte": D2023_END}}},
            {"$group": {"_id": {"$month": "$created_at"}, "revenue": {"$sum": "$total"}, "orders": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "month": "$_id", "revenue": {"$round": ["$revenue", 2]}, "orders": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="Which platinum-tier customers registered after June 2024?",
        collection="customers",
        operation="find",
        query={"tier": "platinum", "created_at": {"$gte": MID_2024}},
        sort={"created_at": -1},
        limit=20,
        projection={"first_name": 1, "last_name": 1, "email": 1, "created_at": 1, "_id": 0},
        tags="medium|find|date|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the total revenue and order count for Q4 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": Q4_2024_START, "$lte": Q4_2024_END}}},
            {"$group": {"_id": None, "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 4 — Array / $unwind (10 questions, medium)
# ---------------------------------------------------------------------------

_Q_ARRAY: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="What is the total quantity of items sold across all orders?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": None, "total_qty": {"$sum": "$items.qty"}}},
            {"$project": {"_id": 0, "total_qty": 1}},
        ],
        tags="medium|aggregate|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many orders contain more than 5 items?",
        collection="orders",
        operation="count",
        query={"$expr": {"$gt": [{"$size": "$items"}, 5]}},
        tags="medium|count|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the total revenue generated per product category?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.category_id", "revenue": {"$sum": "$items.final_price"}, "units_sold": {"$sum": "$items.qty"}}},
            {"$sort": {"revenue": -1}},
            {"$limit": 20},
            {"$project": {"_id": 0, "category_id": "$_id", "revenue": {"$round": ["$revenue", 2]}, "units_sold": 1}},
        ],
        tags="medium|aggregate|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the top 10 most frequently purchased products by total units sold?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.product_id", "total_qty": {"$sum": "$items.qty"}, "product_name": {"$first": "$items.product_name"}}},
            {"$sort": {"total_qty": -1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "product_id": "$_id", "product_name": 1, "total_qty": 1}},
        ],
        tags="medium|aggregate|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the average discount percentage applied to order items?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": None, "avg_discount_pct": {"$avg": "$items.discount_pct"}}},
            {"$project": {"_id": 0, "avg_discount_pct": {"$round": ["$avg_discount_pct", 2]}}},
        ],
        tags="medium|aggregate|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many orders have at least one item with a discount greater than 20%?",
        collection="orders",
        operation="count",
        query={"items": {"$elemMatch": {"discount_pct": {"$gt": 20}}}},
        tags="medium|count|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="What are the last 5 shipment status updates? (latest status in the timeline per shipment)",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$sort": {"created_at": -1}},
            {"$limit": 100},
            {"$project": {
                "_id": 0,
                "carrier": 1,
                "status": 1,
                "last_event": {"$arrayElemAt": ["$timeline", -1]},
            }},
            {"$limit": 5},
        ],
        tags="medium|aggregate|array|shipments",
    ),
    BenchmarkQuestion(
        nl_query="Which products appear in the most orders? Show top 10.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.product_id", "order_count": {"$sum": 1}, "product_name": {"$first": "$items.product_name"}}},
            {"$sort": {"order_count": -1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "product_name": 1, "order_count": 1}},
        ],
        tags="medium|aggregate|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the total stock per warehouse (sum of all inventory quantities)?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$warehouse_id", "total_quantity": {"$sum": "$quantity"}, "items": {"$sum": 1}}},
            {"$sort": {"total_quantity": -1}},
            {"$project": {"_id": 0, "warehouse_id": "$_id", "total_quantity": 1, "items": 1}},
        ],
        tags="medium|aggregate|array|inventory",
    ),
    BenchmarkQuestion(
        nl_query="What is the total revenue of orders with more than 3 items?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"$expr": {"$gt": [{"$size": "$items"}, 3]}}},
            {"$group": {"_id": None, "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="medium|aggregate|array|orders",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 5 — Aggregation Pipelines (15 questions, medium-hard)
# ---------------------------------------------------------------------------

_Q_AGGREGATION: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="What is the monthly revenue for 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$group": {"_id": {"$month": "$created_at"}, "revenue": {"$sum": "$total"}, "orders": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "month": "$_id", "revenue": {"$round": ["$revenue", 2]}, "orders": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="What percentage of orders in 2024 were cancelled?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$group": {"_id": None, "total": {"$sum": 1}, "cancelled": {"$sum": {"$cond": [{"$eq": ["$status", "cancelled"]}, 1, 0]}}}},
            {"$project": {"_id": 0, "total": 1, "cancelled": 1, "cancellation_rate_pct": {"$round": [{"$multiply": [{"$divide": ["$cancelled", "$total"]}, 100]}, 2]}}},
        ],
        tags="medium|aggregate|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the revenue breakdown by payment method?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$payment_method", "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$sort": {"total_revenue": -1}},
            {"$project": {"_id": 0, "payment_method": "$_id", "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="medium|aggregate|orders",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 10 highest-revenue products by total sales amount?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.product_id", "product_name": {"$first": "$items.product_name"}, "total_revenue": {"$sum": "$items.final_price"}, "units_sold": {"$sum": "$items.qty"}}},
            {"$sort": {"total_revenue": -1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "product_name": 1, "total_revenue": {"$round": ["$total_revenue", 2]}, "units_sold": 1}},
        ],
        tags="hard|aggregate|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the count of each event type?",
        collection="events",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$project": {"_id": 0, "event_type": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|events",
    ),
    BenchmarkQuestion(
        nl_query="What percentage of reviews are verified purchases?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": None, "total": {"$sum": 1}, "verified": {"$sum": {"$cond": ["$is_verified", 1, 0]}}}},
            {"$project": {"_id": 0, "total": 1, "verified": 1, "verified_pct": {"$round": [{"$multiply": [{"$divide": ["$verified", "$total"]}, 100]}, 2]}}},
        ],
        tags="medium|aggregate|reviews",
    ),
    BenchmarkQuestion(
        nl_query="What is the order status breakdown (percentage of each status)?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$project": {"_id": 0, "status": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|orders",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 5 countries by total order revenue?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$shipping_address.country", "total_revenue": {"$sum": "$total"}, "orders": {"$sum": 1}}},
            {"$sort": {"total_revenue": -1}},
            {"$limit": 5},
            {"$project": {"_id": 0, "country": "$_id", "total_revenue": {"$round": ["$total_revenue", 2]}, "orders": 1}},
        ],
        tags="medium|aggregate|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the product count per supplier (top 10 suppliers)?",
        collection="products",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$supplier_id", "product_count": {"$sum": 1}}},
            {"$sort": {"product_count": -1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "supplier_id": "$_id", "product_count": 1}},
        ],
        tags="medium|aggregate|products",
    ),
    BenchmarkQuestion(
        nl_query="How many customers placed more than one order?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$customer_id", "order_count": {"$sum": 1}}},
            {"$match": {"order_count": {"$gt": 1}}},
            {"$count": "repeat_customers"},
        ],
        tags="medium|aggregate|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the total cancellation rate by year?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": {"$year": "$created_at"}, "total": {"$sum": 1}, "cancelled": {"$sum": {"$cond": [{"$eq": ["$status", "cancelled"]}, 1, 0]}}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "total": 1, "cancelled": 1, "cancel_rate_pct": {"$round": [{"$multiply": [{"$divide": ["$cancelled", "$total"]}, 100]}, 2]}}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 10 brands by number of products?",
        collection="products",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$brand", "product_count": {"$sum": 1}}},
            {"$sort": {"product_count": -1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "brand": "$_id", "product_count": 1}},
        ],
        tags="easy|aggregate|products",
    ),
    BenchmarkQuestion(
        nl_query="How many inventory items have fewer than 10 units available?",
        collection="inventory",
        operation="count",
        query={"available": {"$lt": 10}},
        tags="easy|count|inventory",
    ),
    BenchmarkQuestion(
        nl_query="What is the average order shipping cost by year?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": {"$year": "$created_at"}, "avg_shipping": {"$avg": "$shipping_cost"}, "free_shipping_count": {"$sum": {"$cond": [{"$eq": ["$shipping_cost", 0]}, 1, 0]}}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "avg_shipping": {"$round": ["$avg_shipping", 2]}, "free_shipping_count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the review sentiment distribution (positive, neutral, negative)?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$sentiment.label", "count": {"$sum": 1}, "avg_rating": {"$avg": "$rating"}}},
            {"$sort": {"count": -1}},
            {"$project": {"_id": 0, "sentiment": "$_id", "count": 1, "avg_rating": {"$round": ["$avg_rating", 2]}}},
        ],
        tags="easy|aggregate|reviews",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 6 — Cross-collection $lookup (10 questions, hard)
# ---------------------------------------------------------------------------

_Q_LOOKUP: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="Who are the top 10 customers by total order spend? Show their name and total.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$customer_id", "total_spend": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$sort": {"total_spend": -1}},
            {"$limit": 10},
            {"$lookup": {"from": "customers", "localField": "_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$project": {"_id": 0, "customer_name": {"$concat": ["$customer.first_name", " ", "$customer.last_name"]}, "email": "$customer.email", "total_spend": {"$round": ["$total_spend", 2]}, "order_count": 1}},
        ],
        tags="hard|lookup|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the average review rating per product? Show top 20 by review count with product name.",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$product_id", "avg_rating": {"$avg": "$rating"}, "review_count": {"$sum": 1}}},
            {"$sort": {"review_count": -1}},
            {"$limit": 20},
            {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$project": {"_id": 0, "product_name": "$product.name", "sku": "$product.sku", "avg_rating": {"$round": ["$avg_rating", 2]}, "review_count": 1}},
        ],
        tags="hard|lookup|reviews|products",
    ),
    BenchmarkQuestion(
        nl_query="Which suppliers have the highest-rated products on average? Top 10.",
        collection="products",
        operation="aggregate",
        query=[
            {"$match": {"rating_count": {"$gt": 0}}},
            {"$group": {"_id": "$supplier_id", "avg_product_rating": {"$avg": "$rating_avg"}, "product_count": {"$sum": 1}}},
            {"$sort": {"avg_product_rating": -1}},
            {"$limit": 10},
            {"$lookup": {"from": "suppliers", "localField": "_id", "foreignField": "_id", "as": "supplier"}},
            {"$unwind": "$supplier"},
            {"$project": {"_id": 0, "supplier_name": "$supplier.name", "country": "$supplier.country", "avg_product_rating": {"$round": ["$avg_product_rating", 2]}, "product_count": 1}},
        ],
        tags="hard|lookup|products|suppliers",
    ),
    BenchmarkQuestion(
        nl_query="What is the total stock quantity per warehouse? Show warehouse name.",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$warehouse_id", "total_quantity": {"$sum": "$quantity"}, "product_count": {"$sum": 1}}},
            {"$sort": {"total_quantity": -1}},
            {"$lookup": {"from": "warehouses", "localField": "_id", "foreignField": "_id", "as": "warehouse"}},
            {"$unwind": "$warehouse"},
            {"$project": {"_id": 0, "warehouse_name": "$warehouse.name", "city": "$warehouse.city", "country": "$warehouse.country", "total_quantity": 1, "product_count": 1}},
        ],
        tags="hard|lookup|inventory|warehouses",
    ),
    BenchmarkQuestion(
        nl_query="What is the total revenue per product category? Show category name.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.category_id", "total_revenue": {"$sum": "$items.final_price"}, "units_sold": {"$sum": "$items.qty"}}},
            {"$sort": {"total_revenue": -1}},
            {"$limit": 20},
            {"$lookup": {"from": "categories", "localField": "_id", "foreignField": "_id", "as": "category"}},
            {"$unwind": "$category"},
            {"$project": {"_id": 0, "category_name": "$category.name", "category_path": "$category.path", "total_revenue": {"$round": ["$total_revenue", 2]}, "units_sold": 1}},
        ],
        tags="hard|lookup|array|orders|categories",
    ),
    BenchmarkQuestion(
        nl_query="Which products have more than 50 reviews? Show product name, review count, and average rating.",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$product_id", "review_count": {"$sum": 1}, "avg_rating": {"$avg": "$rating"}}},
            {"$match": {"review_count": {"$gt": 50}}},
            {"$sort": {"review_count": -1}},
            {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$project": {"_id": 0, "product_name": "$product.name", "sku": "$product.sku", "review_count": 1, "avg_rating": {"$round": ["$avg_rating", 2]}}},
        ],
        tags="hard|lookup|reviews|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the average delivery time in days per carrier?",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$match": {"actual_delivery": {"$ne": None}}},
            {"$project": {"carrier": 1, "days_to_deliver": {"$divide": [{"$subtract": ["$actual_delivery", "$created_at"]}, 86400000]}}},
            {"$group": {"_id": "$carrier", "avg_days": {"$avg": "$days_to_deliver"}, "shipment_count": {"$sum": 1}}},
            {"$sort": {"avg_days": 1}},
            {"$project": {"_id": 0, "carrier": "$_id", "avg_days": {"$round": ["$avg_days", 1]}, "shipment_count": 1}},
        ],
        tags="hard|aggregate|shipments",
    ),
    BenchmarkQuestion(
        nl_query="Which customers placed more than 5 orders? Show their name and order count.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$customer_id", "order_count": {"$sum": 1}, "total_spend": {"$sum": "$total"}}},
            {"$match": {"order_count": {"$gt": 5}}},
            {"$sort": {"order_count": -1}},
            {"$lookup": {"from": "customers", "localField": "_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$project": {"_id": 0, "customer_name": {"$concat": ["$customer.first_name", " ", "$customer.last_name"]}, "email": "$customer.email", "tier": "$customer.tier", "order_count": 1, "total_spend": {"$round": ["$total_spend", 2]}}},
        ],
        tags="hard|lookup|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the total revenue by customer country? (via customer lookup)",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$customer_id", "customer_revenue": {"$sum": "$total"}}},
            {"$lookup": {"from": "customers", "localField": "_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$group": {"_id": "$customer.country", "total_revenue": {"$sum": "$customer_revenue"}, "customer_count": {"$sum": 1}}},
            {"$sort": {"total_revenue": -1}},
            {"$limit": 15},
            {"$project": {"_id": 0, "country": "$_id", "total_revenue": {"$round": ["$total_revenue", 2]}, "customer_count": 1}},
        ],
        tags="hard|lookup|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="For each carrier, what percentage of shipments were delivered on time?",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$match": {"actual_delivery": {"$ne": None}}},
            {"$project": {"carrier": 1, "on_time": {"$lte": ["$actual_delivery", "$estimated_delivery"]}}},
            {"$group": {"_id": "$carrier", "total": {"$sum": 1}, "on_time_count": {"$sum": {"$cond": ["$on_time", 1, 0]}}}},
            {"$sort": {"total": -1}},
            {"$project": {"_id": 0, "carrier": "$_id", "total": 1, "on_time_count": 1, "on_time_pct": {"$round": [{"$multiply": [{"$divide": ["$on_time_count", "$total"]}, 100]}, 1]}}},
        ],
        tags="hard|aggregate|shipments",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 7 — Hard / Multi-stage (15 questions, hard)
# ---------------------------------------------------------------------------

_Q_HARD: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="For each customer tier, what is the average order value, total revenue, and order count?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$lookup": {"from": "customers", "localField": "customer_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$group": {"_id": "$customer.tier", "avg_order_value": {"$avg": "$total"}, "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}, "customer_count": {"$addToSet": "$customer_id"}}},
            {"$sort": {"total_revenue": -1}},
            {"$project": {"_id": 0, "tier": "$_id", "avg_order_value": {"$round": ["$avg_order_value", 2]}, "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1, "unique_customers": {"$size": "$customer_count"}}},
        ],
        tags="hard|lookup|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 10 products by revenue in 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.product_id", "product_name": {"$first": "$items.product_name"}, "total_revenue": {"$sum": "$items.final_price"}, "units_sold": {"$sum": "$items.qty"}}},
            {"$sort": {"total_revenue": -1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "product_name": 1, "total_revenue": {"$round": ["$total_revenue", 2]}, "units_sold": 1}},
        ],
        tags="hard|aggregate|array|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the lifetime value distribution of customers? Group into: under 100 EUR, 100-500 EUR, 500-1000 EUR, over 1000 EUR.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$customer_id", "lifetime_value": {"$sum": "$total"}}},
            {"$bucket": {"groupBy": "$lifetime_value", "boundaries": [0, 100, 500, 1000], "default": "over_1000", "output": {"count": {"$sum": 1}, "avg_ltv": {"$avg": "$lifetime_value"}}}},
        ],
        tags="hard|aggregate|orders",
    ),
    BenchmarkQuestion(
        nl_query="For products with at least 10 reviews, what is the ratio of positive to negative reviews? Show top 20.",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$product_id", "total_reviews": {"$sum": 1}, "positive": {"$sum": {"$cond": [{"$eq": ["$sentiment.label", "positive"]}, 1, 0]}}, "negative": {"$sum": {"$cond": [{"$eq": ["$sentiment.label", "negative"]}, 1, 0]}}}},
            {"$match": {"total_reviews": {"$gte": 10}}},
            {"$project": {"_id": 1, "total_reviews": 1, "positive": 1, "negative": 1, "sentiment_ratio": {"$cond": [{"$eq": ["$negative", 0]}, None, {"$round": [{"$divide": ["$positive", "$negative"]}, 2]}]}}},
            {"$sort": {"sentiment_ratio": -1}},
            {"$limit": 20},
        ],
        tags="hard|aggregate|reviews",
    ),
    BenchmarkQuestion(
        nl_query="What is the average number of days between first and last order for customers with more than 1 order?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$customer_id", "first_order": {"$min": "$created_at"}, "last_order": {"$max": "$created_at"}, "order_count": {"$sum": 1}}},
            {"$match": {"order_count": {"$gt": 1}}},
            {"$project": {"days_between": {"$divide": [{"$subtract": ["$last_order", "$first_order"]}, 86400000]}}},
            {"$group": {"_id": None, "avg_days_between_orders": {"$avg": "$days_between"}, "qualifying_customers": {"$sum": 1}}},
            {"$project": {"_id": 0, "avg_days_between_orders": {"$round": ["$avg_days_between_orders", 1]}, "qualifying_customers": 1}},
        ],
        tags="hard|aggregate|orders",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 5 product categories by average order value (only categories that appear in at least 100 orders)?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": {"order_id": "$_id", "category_id": "$items.category_id"}, "order_total": {"$first": "$total"}, "category_path": {"$first": "$items"}}},
            {"$group": {"_id": "$_id.category_id", "avg_order_value": {"$avg": "$order_total"}, "order_count": {"$sum": 1}}},
            {"$match": {"order_count": {"$gte": 100}}},
            {"$sort": {"avg_order_value": -1}},
            {"$limit": 5},
            {"$lookup": {"from": "categories", "localField": "_id", "foreignField": "_id", "as": "category"}},
            {"$unwind": "$category"},
            {"$project": {"_id": 0, "category_name": "$category.name", "avg_order_value": {"$round": ["$avg_order_value", 2]}, "order_count": 1}},
        ],
        tags="hard|lookup|array|orders|categories",
    ),
    BenchmarkQuestion(
        nl_query="What is the monthly new customer acquisition trend for 2024?",
        collection="customers",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$group": {"_id": {"$month": "$created_at"}, "new_customers": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "month": "$_id", "new_customers": 1}},
        ],
        tags="medium|aggregate|date|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the NPS proxy (percentage of 5-star reviews minus percentage of 1-star reviews)?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": None, "total": {"$sum": 1}, "five_star": {"$sum": {"$cond": [{"$eq": ["$rating", 5]}, 1, 0]}}, "one_star": {"$sum": {"$cond": [{"$eq": ["$rating", 1]}, 1, 0]}}}},
            {"$project": {"_id": 0, "total": 1, "five_star": 1, "one_star": 1, "nps_proxy": {"$round": [{"$subtract": [{"$multiply": [{"$divide": ["$five_star", "$total"]}, 100]}, {"$multiply": [{"$divide": ["$one_star", "$total"]}, 100]}]}, 2]}}},
        ],
        tags="medium|aggregate|reviews",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 10 customers by average order value (minimum 3 orders)?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$customer_id", "avg_order_value": {"$avg": "$total"}, "order_count": {"$sum": 1}, "total_spend": {"$sum": "$total"}}},
            {"$match": {"order_count": {"$gte": 3}}},
            {"$sort": {"avg_order_value": -1}},
            {"$limit": 10},
            {"$lookup": {"from": "customers", "localField": "_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$project": {"_id": 0, "customer_name": {"$concat": ["$customer.first_name", " ", "$customer.last_name"]}, "tier": "$customer.tier", "avg_order_value": {"$round": ["$avg_order_value", 2]}, "order_count": 1, "total_spend": {"$round": ["$total_spend", 2]}}},
        ],
        tags="hard|lookup|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the average review rating per customer tier?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$lookup": {"from": "customers", "localField": "customer_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$group": {"_id": "$customer.tier", "avg_rating": {"$avg": "$rating"}, "review_count": {"$sum": 1}}},
            {"$sort": {"avg_rating": -1}},
            {"$project": {"_id": 0, "tier": "$_id", "avg_rating": {"$round": ["$avg_rating", 2]}, "review_count": 1}},
        ],
        tags="hard|lookup|reviews|customers",
    ),
    BenchmarkQuestion(
        nl_query="For negatively reviewed products, what is their average review rating and total orders?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$match": {"sentiment.label": "negative"}},
            {"$group": {"_id": "$product_id", "avg_rating": {"$avg": "$rating"}, "negative_review_count": {"$sum": 1}}},
            {"$sort": {"negative_review_count": -1}},
            {"$limit": 20},
            {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$project": {"_id": 0, "product_name": "$product.name", "avg_rating": {"$round": ["$avg_rating", 2]}, "negative_review_count": 1}},
        ],
        tags="hard|lookup|reviews|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the total revenue and unique customer count by customer tier in 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$lookup": {"from": "customers", "localField": "customer_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$group": {"_id": "$customer.tier", "total_revenue": {"$sum": "$total"}, "unique_customers": {"$addToSet": "$customer_id"}, "order_count": {"$sum": 1}}},
            {"$sort": {"total_revenue": -1}},
            {"$project": {"_id": 0, "tier": "$_id", "total_revenue": {"$round": ["$total_revenue", 2]}, "unique_customers": {"$size": "$unique_customers"}, "order_count": 1}},
        ],
        tags="hard|lookup|date|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the revenue from orders that have at least one item with a discount, vs orders with no discounts?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$project": {"total": 1, "has_discount": {"$gt": [{"$size": {"$filter": {"input": "$items", "cond": {"$gt": ["$$this.discount_pct", 0]}}}}, 0]}}},
            {"$group": {"_id": "$has_discount", "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "has_discount": "$_id", "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="hard|aggregate|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="Which 10 products have the most helpful reviews? (sum of helpful_votes)",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$product_id", "total_helpful_votes": {"$sum": "$helpful_votes"}, "review_count": {"$sum": 1}}},
            {"$sort": {"total_helpful_votes": -1}},
            {"$limit": 10},
            {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$project": {"_id": 0, "product_name": "$product.name", "total_helpful_votes": 1, "review_count": 1}},
        ],
        tags="hard|lookup|reviews|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the cart-to-purchase ratio? (cart_add events vs purchase events) by year",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"type": {"$in": ["cart_add", "purchase"]}}},
            {"$group": {"_id": {"year": {"$year": "$timestamp"}, "type": "$type"}, "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$group": {"_id": "$_id.year", "events": {"$push": {"type": "$_id.type", "count": "$count"}}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "events": 1}},
        ],
        tags="hard|aggregate|date|events",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 8 — Events Analytics (15 questions)
# ---------------------------------------------------------------------------

_Q_EVENTS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="What is the breakdown of events by device type (mobile, desktop, tablet)?",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"payload.device": {"$exists": True}}},
            {"$group": {"_id": "$payload.device", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$project": {"_id": 0, "device": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|events",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 10 most searched terms on the platform?",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"type": "search"}},
            {"$group": {"_id": "$payload.query", "search_count": {"$sum": 1}}},
            {"$sort": {"search_count": -1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "query": "$_id", "search_count": 1}},
        ],
        tags="easy|aggregate|events",
    ),
    BenchmarkQuestion(
        nl_query="What is the breakdown of events by referrer source?",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"payload.referrer": {"$exists": True}}},
            {"$group": {"_id": "$payload.referrer", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$project": {"_id": 0, "referrer": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|events",
    ),
    BenchmarkQuestion(
        nl_query="How many cart_add events were recorded in 2024?",
        collection="events",
        operation="count",
        query={"type": "cart_add", "timestamp": {"$gte": D2024_START, "$lte": D2024_END}},
        tags="medium|count|date|events",
    ),
    BenchmarkQuestion(
        nl_query="What is the average number of events per session?",
        collection="events",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$session_id", "event_count": {"$sum": 1}}},
            {"$group": {"_id": None, "avg_events_per_session": {"$avg": "$event_count"}}},
            {"$project": {"_id": 0, "avg_events_per_session": {"$round": ["$avg_events_per_session", 2]}}},
        ],
        tags="medium|aggregate|events",
    ),
    BenchmarkQuestion(
        nl_query="How many view events were recorded in 2024?",
        collection="events",
        operation="count",
        query={"type": "view", "timestamp": {"$gte": D2024_START, "$lte": D2024_END}},
        tags="medium|count|date|events",
    ),
    BenchmarkQuestion(
        nl_query="What is the monthly event volume trend for 2024?",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"timestamp": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$group": {"_id": {"$month": "$timestamp"}, "event_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "month": "$_id", "event_count": 1}},
        ],
        tags="medium|aggregate|date|events",
    ),
    BenchmarkQuestion(
        nl_query="How many wishlist events were recorded in total?",
        collection="events",
        operation="count",
        query={"type": "wishlist"},
        tags="easy|count|events",
    ),
    BenchmarkQuestion(
        nl_query="What percentage of events are from authenticated (non-anonymous) users?",
        collection="events",
        operation="aggregate",
        query=[
            {"$group": {"_id": None, "total": {"$sum": 1}, "authenticated": {"$sum": {"$cond": [{"$ne": ["$customer_id", None]}, 1, 0]}}}},
            {"$project": {"_id": 0, "total": 1, "authenticated": 1, "auth_pct": {"$round": [{"$multiply": [{"$divide": ["$authenticated", "$total"]}, 100]}, 2]}}},
        ],
        tags="medium|aggregate|events",
    ),
    BenchmarkQuestion(
        nl_query="What is the average product view duration in seconds?",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"type": "view"}},
            {"$group": {"_id": None, "avg_duration": {"$avg": "$payload.duration_seconds"}}},
            {"$project": {"_id": 0, "avg_duration_seconds": {"$round": ["$avg_duration", 1]}}},
        ],
        tags="easy|aggregate|events",
    ),
    BenchmarkQuestion(
        nl_query="How many cart_remove events were recorded compared to cart_add events?",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"type": {"$in": ["cart_add", "cart_remove"]}}},
            {"$group": {"_id": "$type", "count": {"$sum": 1}}},
            {"$project": {"_id": 0, "type": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|events",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 5 most viewed products (by view event count)?",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"type": "view", "product_id": {"$exists": True}}},
            {"$group": {"_id": "$product_id", "view_count": {"$sum": 1}}},
            {"$sort": {"view_count": -1}},
            {"$limit": 5},
            {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$project": {"_id": 0, "product_name": "$product.name", "sku": "$product.sku", "view_count": 1}},
        ],
        tags="hard|lookup|events|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the search-to-purchase conversion: how many sessions had both a search and a purchase event?",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"type": {"$in": ["search", "purchase"]}}},
            {"$group": {"_id": "$session_id", "types": {"$addToSet": "$type"}}},
            {"$match": {"types": {"$all": ["search", "purchase"]}}},
            {"$count": "sessions_with_search_and_purchase"},
        ],
        tags="hard|aggregate|events",
    ),
    BenchmarkQuestion(
        nl_query="How many events were recorded per year?",
        collection="events",
        operation="aggregate",
        query=[
            {"$group": {"_id": {"$year": "$timestamp"}, "event_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "event_count": 1}},
        ],
        tags="easy|aggregate|date|events",
    ),
    BenchmarkQuestion(
        nl_query="What is the average search result count for searches that returned at least one result?",
        collection="events",
        operation="aggregate",
        query=[
            {"$match": {"type": "search", "payload.results_count": {"$gt": 0}}},
            {"$group": {"_id": None, "avg_results": {"$avg": "$payload.results_count"}, "search_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "avg_results": {"$round": ["$avg_results", 1]}, "search_count": 1}},
        ],
        tags="medium|aggregate|events",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 9 — Shipments Deep Dive (10 questions)
# ---------------------------------------------------------------------------

_Q_SHIPMENTS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="What is the shipment status breakdown (count per status)?",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$project": {"_id": 0, "status": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|shipments",
    ),
    BenchmarkQuestion(
        nl_query="How many shipments are in 'failed' status?",
        collection="shipments",
        operation="count",
        query={"status": "failed"},
        tags="easy|count|shipments",
    ),
    BenchmarkQuestion(
        nl_query="What is the total number of shipments per carrier?",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$carrier", "shipment_count": {"$sum": 1}}},
            {"$sort": {"shipment_count": -1}},
            {"$project": {"_id": 0, "carrier": "$_id", "shipment_count": 1}},
        ],
        tags="easy|aggregate|shipments",
    ),
    BenchmarkQuestion(
        nl_query="What is the maximum and minimum estimated delivery time across all shipments (in days from creation)?",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$project": {"days_to_est_delivery": {"$divide": [{"$subtract": ["$estimated_delivery", "$created_at"]}, 86400000]}}},
            {"$group": {"_id": None, "max_days": {"$max": "$days_to_est_delivery"}, "min_days": {"$min": "$days_to_est_delivery"}, "avg_days": {"$avg": "$days_to_est_delivery"}}},
            {"$project": {"_id": 0, "max_days": {"$round": ["$max_days", 1]}, "min_days": {"$round": ["$min_days", 1]}, "avg_days": {"$round": ["$avg_days", 1]}}},
        ],
        tags="medium|aggregate|shipments",
    ),
    BenchmarkQuestion(
        nl_query="How many shipments have been actually delivered?",
        collection="shipments",
        operation="count",
        query={"actual_delivery": {"$ne": None}},
        tags="easy|count|shipments",
    ),
    BenchmarkQuestion(
        nl_query="What is the average number of timeline events per shipment?",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$project": {"timeline_length": {"$size": "$timeline"}}},
            {"$group": {"_id": None, "avg_timeline_events": {"$avg": "$timeline_length"}}},
            {"$project": {"_id": 0, "avg_timeline_events": {"$round": ["$avg_timeline_events", 2]}}},
        ],
        tags="medium|aggregate|array|shipments",
    ),
    BenchmarkQuestion(
        nl_query="How many shipments were created in 2024?",
        collection="shipments",
        operation="count",
        query={"created_at": {"$gte": D2024_START, "$lte": D2024_END}},
        tags="medium|count|date|shipments",
    ),
    BenchmarkQuestion(
        nl_query="What is the on-time delivery count vs late delivery count per carrier?",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$match": {"actual_delivery": {"$ne": None}}},
            {"$project": {"carrier": 1, "is_late": {"$gt": ["$actual_delivery", "$estimated_delivery"]}}},
            {"$group": {"_id": "$carrier", "on_time": {"$sum": {"$cond": [{"$eq": ["$is_late", False]}, 1, 0]}}, "late": {"$sum": {"$cond": ["$is_late", 1, 0]}}, "total": {"$sum": 1}}},
            {"$sort": {"total": -1}},
            {"$project": {"_id": 0, "carrier": "$_id", "on_time": 1, "late": 1, "total": 1}},
        ],
        tags="medium|aggregate|shipments",
    ),
    BenchmarkQuestion(
        nl_query="What is the monthly shipment volume for 2024?",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$group": {"_id": {"$month": "$created_at"}, "shipment_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "month": "$_id", "shipment_count": 1}},
        ],
        tags="medium|aggregate|date|shipments",
    ),
    BenchmarkQuestion(
        nl_query="Which carrier delivered the most shipments in 2024?",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}, "status": "delivered"}},
            {"$group": {"_id": "$carrier", "delivered_count": {"$sum": 1}}},
            {"$sort": {"delivered_count": -1}},
            {"$limit": 1},
            {"$project": {"_id": 0, "carrier": "$_id", "delivered_count": 1}},
        ],
        tags="medium|aggregate|date|shipments",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 10 — Inventory & Supply Chain (10 questions)
# ---------------------------------------------------------------------------

_Q_INVENTORY: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="How many distinct products are tracked in inventory?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$product_id"}},
            {"$count": "distinct_products"},
        ],
        tags="easy|aggregate|inventory",
    ),
    BenchmarkQuestion(
        nl_query="What is the total reserved quantity across all inventory?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$group": {"_id": None, "total_reserved": {"$sum": "$reserved"}, "total_quantity": {"$sum": "$quantity"}}},
            {"$project": {"_id": 0, "total_reserved": 1, "total_quantity": 1}},
        ],
        tags="easy|aggregate|inventory",
    ),
    BenchmarkQuestion(
        nl_query="What percentage of inventory items have zero available stock?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$group": {"_id": None, "total": {"$sum": 1}, "zero_stock": {"$sum": {"$cond": [{"$eq": ["$available", 0]}, 1, 0]}}}},
            {"$project": {"_id": 0, "total": 1, "zero_stock": 1, "zero_stock_pct": {"$round": [{"$multiply": [{"$divide": ["$zero_stock", "$total"]}, 100]}, 2]}}},
        ],
        tags="medium|aggregate|inventory",
    ),
    BenchmarkQuestion(
        nl_query="What is the average reorder threshold across all inventory items?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$group": {"_id": None, "avg_threshold": {"$avg": "$reorder_threshold"}}},
            {"$project": {"_id": 0, "avg_threshold": {"$round": ["$avg_threshold", 1]}}},
        ],
        tags="easy|aggregate|inventory",
    ),
    BenchmarkQuestion(
        nl_query="Which warehouse has the most inventory items below reorder threshold?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$match": {"below_threshold": True}},
            {"$group": {"_id": "$warehouse_id", "below_threshold_count": {"$sum": 1}}},
            {"$sort": {"below_threshold_count": -1}},
            {"$limit": 1},
            {"$lookup": {"from": "warehouses", "localField": "_id", "foreignField": "_id", "as": "warehouse"}},
            {"$unwind": "$warehouse"},
            {"$project": {"_id": 0, "warehouse_name": "$warehouse.name", "city": "$warehouse.city", "below_threshold_count": 1}},
        ],
        tags="hard|lookup|inventory|warehouses",
    ),
    BenchmarkQuestion(
        nl_query="What is the total available stock per warehouse, with warehouse name?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$warehouse_id", "total_available": {"$sum": "$available"}, "item_count": {"$sum": 1}}},
            {"$sort": {"total_available": -1}},
            {"$lookup": {"from": "warehouses", "localField": "_id", "foreignField": "_id", "as": "warehouse"}},
            {"$unwind": "$warehouse"},
            {"$project": {"_id": 0, "warehouse_name": "$warehouse.name", "city": "$warehouse.city", "total_available": 1, "item_count": 1}},
        ],
        tags="hard|lookup|inventory|warehouses",
    ),
    BenchmarkQuestion(
        nl_query="How many inventory entries have more reserved than available quantity?",
        collection="inventory",
        operation="count",
        query={"$expr": {"$gt": ["$reserved", "$available"]}},
        tags="medium|count|inventory",
    ),
    BenchmarkQuestion(
        nl_query="What is the top 10 products by total available stock across all warehouses?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$product_id", "total_available": {"$sum": "$available"}, "warehouse_count": {"$sum": 1}}},
            {"$sort": {"total_available": -1}},
            {"$limit": 10},
            {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$project": {"_id": 0, "product_name": "$product.name", "sku": "$product.sku", "total_available": 1, "warehouse_count": 1}},
        ],
        tags="hard|lookup|inventory|products",
    ),
    BenchmarkQuestion(
        nl_query="How many products appear in more than one warehouse?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$product_id", "warehouse_count": {"$sum": 1}}},
            {"$match": {"warehouse_count": {"$gt": 1}}},
            {"$count": "multi_warehouse_products"},
        ],
        tags="medium|aggregate|inventory",
    ),
    BenchmarkQuestion(
        nl_query="What is the ratio of reserved to total quantity per warehouse?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$warehouse_id", "total_quantity": {"$sum": "$quantity"}, "total_reserved": {"$sum": "$reserved"}}},
            {"$match": {"total_quantity": {"$gt": 0}}},
            {"$lookup": {"from": "warehouses", "localField": "_id", "foreignField": "_id", "as": "warehouse"}},
            {"$unwind": "$warehouse"},
            {"$project": {"_id": 0, "warehouse_name": "$warehouse.name", "total_quantity": 1, "total_reserved": 1, "reserved_ratio": {"$round": [{"$divide": ["$total_reserved", "$total_quantity"]}, 3]}}},
            {"$sort": {"reserved_ratio": -1}},
        ],
        tags="hard|lookup|inventory|warehouses",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 11 — Reviews Extended (10 questions)
# ---------------------------------------------------------------------------

_Q_REVIEWS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="What is the average rating for verified vs unverified reviews?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$is_verified", "avg_rating": {"$avg": "$rating"}, "count": {"$sum": 1}}},
            {"$project": {"_id": 0, "is_verified": "$_id", "avg_rating": {"$round": ["$avg_rating", 2]}, "count": 1}},
        ],
        tags="easy|aggregate|reviews",
    ),
    BenchmarkQuestion(
        nl_query="What is the monthly review volume for 2024?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$group": {"_id": {"$month": "$created_at"}, "review_count": {"$sum": 1}, "avg_rating": {"$avg": "$rating"}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "month": "$_id", "review_count": 1, "avg_rating": {"$round": ["$avg_rating", 2]}}},
        ],
        tags="medium|aggregate|date|reviews",
    ),
    BenchmarkQuestion(
        nl_query="What is the rating distribution (count per star rating 1-5)?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$rating", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "rating": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|reviews",
    ),
    BenchmarkQuestion(
        nl_query="Which customers wrote the most reviews? Show top 10.",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$customer_id", "review_count": {"$sum": 1}, "avg_rating_given": {"$avg": "$rating"}}},
            {"$sort": {"review_count": -1}},
            {"$limit": 10},
            {"$lookup": {"from": "customers", "localField": "_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$project": {"_id": 0, "customer_name": {"$concat": ["$customer.first_name", " ", "$customer.last_name"]}, "review_count": 1, "avg_rating_given": {"$round": ["$avg_rating_given", 2]}}},
        ],
        tags="hard|lookup|reviews|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the average helpful_votes per review by sentiment?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$sentiment.label", "avg_helpful_votes": {"$avg": "$helpful_votes"}, "count": {"$sum": 1}}},
            {"$sort": {"avg_helpful_votes": -1}},
            {"$project": {"_id": 0, "sentiment": "$_id", "avg_helpful_votes": {"$round": ["$avg_helpful_votes", 2]}, "count": 1}},
        ],
        tags="easy|aggregate|reviews",
    ),
    BenchmarkQuestion(
        nl_query="How many reviews with 1 or 2 stars were written in 2024?",
        collection="reviews",
        operation="count",
        query={"rating": {"$lte": 2}, "created_at": {"$gte": D2024_START, "$lte": D2024_END}},
        tags="medium|count|date|reviews",
    ),
    BenchmarkQuestion(
        nl_query="What is the average review rating per year?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": {"$year": "$created_at"}, "avg_rating": {"$avg": "$rating"}, "review_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "avg_rating": {"$round": ["$avg_rating", 2]}, "review_count": 1}},
        ],
        tags="medium|aggregate|date|reviews",
    ),
    BenchmarkQuestion(
        nl_query="What are the distinct sentiment labels in the reviews collection?",
        collection="reviews",
        operation="distinct",
        query={},
        distinct_field="sentiment.label",
        tags="easy|distinct|reviews",
    ),
    BenchmarkQuestion(
        nl_query="How many reviews have more than 100 helpful votes?",
        collection="reviews",
        operation="count",
        query={"helpful_votes": {"$gt": 100}},
        tags="easy|count|reviews",
    ),
    BenchmarkQuestion(
        nl_query="What is the average sentiment score for positive vs negative reviews?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$match": {"sentiment.label": {"$in": ["positive", "negative"]}}},
            {"$group": {"_id": "$sentiment.label", "avg_score": {"$avg": "$sentiment.score"}, "count": {"$sum": 1}}},
            {"$project": {"_id": 0, "label": "$_id", "avg_score": {"$round": ["$avg_score", 3]}, "count": 1}},
        ],
        tags="medium|aggregate|reviews",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 12 — Product Catalog Extended (10 questions)
# ---------------------------------------------------------------------------

_Q_PRODUCTS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="How many products are inactive (not active)?",
        collection="products",
        operation="count",
        query={"is_active": False},
        tags="easy|count|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the price distribution of products? Group into: under 50, 50-200, 200-500, 500-1000, over 1000 EUR.",
        collection="products",
        operation="aggregate",
        query=[
            {"$bucket": {"groupBy": "$price", "boundaries": [0, 50, 200, 500, 1000], "default": "over_1000", "output": {"count": {"$sum": 1}, "avg_price": {"$avg": "$price"}}}},
        ],
        tags="medium|aggregate|products",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 5 brands by average product price?",
        collection="products",
        operation="aggregate",
        query=[
            {"$match": {"is_active": True}},
            {"$group": {"_id": "$brand", "avg_price": {"$avg": "$price"}, "product_count": {"$sum": 1}}},
            {"$match": {"product_count": {"$gte": 5}}},
            {"$sort": {"avg_price": -1}},
            {"$limit": 5},
            {"$project": {"_id": 0, "brand": "$_id", "avg_price": {"$round": ["$avg_price", 2]}, "product_count": 1}},
        ],
        tags="medium|aggregate|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the total number of products per brand, sorted descending? Show top 15.",
        collection="products",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$brand", "product_count": {"$sum": 1}, "active_count": {"$sum": {"$cond": ["$is_active", 1, 0]}}}},
            {"$sort": {"product_count": -1}},
            {"$limit": 15},
            {"$project": {"_id": 0, "brand": "$_id", "product_count": 1, "active_count": 1}},
        ],
        tags="easy|aggregate|products",
    ),
    BenchmarkQuestion(
        nl_query="How many products have no rating yet (rating_count is 0)?",
        collection="products",
        operation="count",
        query={"rating_count": 0},
        tags="easy|count|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the average stock level per product category (top-level)?",
        collection="products",
        operation="aggregate",
        query=[
            {"$group": {"_id": {"$arrayElemAt": ["$category_path", 0]}, "avg_stock": {"$avg": "$stock"}, "product_count": {"$sum": 1}}},
            {"$sort": {"avg_stock": -1}},
            {"$project": {"_id": 0, "category": "$_id", "avg_stock": {"$round": ["$avg_stock", 1]}, "product_count": 1}},
        ],
        tags="medium|aggregate|products",
    ),
    BenchmarkQuestion(
        nl_query="What are the top 10 products with the highest average rating (minimum 50 reviews)?",
        collection="products",
        operation="find",
        query={"rating_count": {"$gte": 50}, "is_active": True},
        sort={"rating_avg": -1},
        limit=10,
        projection={"name": 1, "brand": 1, "sku": 1, "rating_avg": 1, "rating_count": 1, "_id": 0},
        tags="medium|find|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the average product cost-to-price margin ratio?",
        collection="products",
        operation="aggregate",
        query=[
            {"$match": {"price": {"$gt": 0}}},
            {"$project": {"margin": {"$divide": [{"$subtract": ["$price", "$cost"]}, "$price"]}}},
            {"$group": {"_id": None, "avg_margin": {"$avg": "$margin"}}},
            {"$project": {"_id": 0, "avg_margin_pct": {"$round": [{"$multiply": ["$avg_margin", 100]}, 2]}}},
        ],
        tags="medium|aggregate|products",
    ),
    BenchmarkQuestion(
        nl_query="How many products are tagged as 'sale'?",
        collection="products",
        operation="count",
        query={"tags": "sale"},
        tags="easy|count|products",
    ),
    BenchmarkQuestion(
        nl_query="What are the distinct product tags used in the catalog?",
        collection="products",
        operation="distinct",
        query={},
        distinct_field="tags",
        tags="easy|distinct|products",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 13 — Customer Segmentation (10 questions)
# ---------------------------------------------------------------------------

_Q_CUSTOMERS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="How many platinum-tier customers are based in Italy?",
        collection="customers",
        operation="count",
        query={"tier": "platinum", "country": "Italy"},
        tags="easy|count|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the marketing opt-in rate by customer tier?",
        collection="customers",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$tier", "total": {"$sum": 1}, "opted_in": {"$sum": {"$cond": ["$marketing_opt_in", 1, 0]}}}},
            {"$project": {"_id": 0, "tier": "$_id", "total": 1, "opted_in": 1, "opt_in_pct": {"$round": [{"$multiply": [{"$divide": ["$opted_in", "$total"]}, 100]}, 2]}}},
            {"$sort": {"opt_in_pct": -1}},
        ],
        tags="medium|aggregate|customers",
    ),
    BenchmarkQuestion(
        nl_query="How many customers have never placed an order (last_order_at is null)?",
        collection="customers",
        operation="count",
        query={"last_order_at": None},
        tags="medium|count|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the customer count per country, top 10?",
        collection="customers",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$country", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "country": "$_id", "count": 1}},
        ],
        tags="easy|aggregate|customers",
    ),
    BenchmarkQuestion(
        nl_query="How many customers registered in 2025?",
        collection="customers",
        operation="count",
        query={"created_at": {"$gte": D2025_START, "$lte": D2025_END}},
        tags="medium|count|date|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the average number of addresses per customer?",
        collection="customers",
        operation="aggregate",
        query=[
            {"$project": {"address_count": {"$size": "$addresses"}}},
            {"$group": {"_id": None, "avg_addresses": {"$avg": "$address_count"}}},
            {"$project": {"_id": 0, "avg_addresses": {"$round": ["$avg_addresses", 2]}}},
        ],
        tags="medium|aggregate|array|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the preferred payment method breakdown by customer tier?",
        collection="customers",
        operation="aggregate",
        query=[
            {"$group": {"_id": {"tier": "$tier", "payment": "$preferred_payment"}, "count": {"$sum": 1}}},
            {"$sort": {"_id.tier": 1, "count": -1}},
            {"$project": {"_id": 0, "tier": "$_id.tier", "payment_method": "$_id.payment", "count": 1}},
        ],
        tags="medium|aggregate|customers",
    ),
    BenchmarkQuestion(
        nl_query="How many silver-tier customers opted in to marketing?",
        collection="customers",
        operation="count",
        query={"tier": "silver", "marketing_opt_in": True},
        tags="easy|count|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the new customer registration trend by year?",
        collection="customers",
        operation="aggregate",
        query=[
            {"$group": {"_id": {"$year": "$created_at"}, "new_customers": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "new_customers": 1}},
        ],
        tags="easy|aggregate|date|customers",
    ),
    BenchmarkQuestion(
        nl_query="Which countries have more than 2000 customers?",
        collection="customers",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$country", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 2000}}},
            {"$sort": {"count": -1}},
            {"$project": {"_id": 0, "country": "$_id", "count": 1}},
        ],
        tags="medium|aggregate|customers",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 14 — Date & Time Extended (10 questions)
# ---------------------------------------------------------------------------

_Q_DATE_EXT: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="What is the total revenue for Q1 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": Q1_2024_START, "$lte": Q1_2024_END}}},
            {"$group": {"_id": None, "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the total revenue for Q2 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": Q2_2024_START, "$lte": Q2_2024_END}}},
            {"$group": {"_id": None, "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the total revenue for Q3 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": Q3_2024_START, "$lte": Q3_2024_END}}},
            {"$group": {"_id": None, "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="How do H1 2024 and H2 2024 revenue compare?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": H1_2024_START, "$lte": H2_2024_END}}},
            {"$group": {"_id": {"$cond": [{"$lte": ["$created_at", H1_2024_END]}, "H1_2024", "H2_2024"]}, "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "half": "$_id", "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many orders were placed in January 2024?",
        collection="orders",
        operation="count",
        query={"created_at": {"$gte": JAN_2024_START, "$lte": JAN_2024_END}},
        tags="medium|count|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the year-over-year order count comparison for 2023 vs 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2023_START, "$lte": D2024_END}}},
            {"$group": {"_id": {"$year": "$created_at"}, "order_count": {"$sum": 1}, "total_revenue": {"$sum": "$total"}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "order_count": 1, "total_revenue": {"$round": ["$total_revenue", 2]}}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many new customers registered in Q1 2025?",
        collection="customers",
        operation="count",
        query={"created_at": {"$gte": Q1_2025_START, "$lte": Q1_2025_END}},
        tags="medium|count|date|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the average order value for H1 2024 vs H2 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": H1_2024_START, "$lte": H2_2024_END}}},
            {"$group": {"_id": {"$cond": [{"$lte": ["$created_at", H1_2024_END]}, "H1_2024", "H2_2024"]}, "avg_order_value": {"$avg": "$total"}, "order_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "half": "$_id", "avg_order_value": {"$round": ["$avg_order_value", 2]}, "order_count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="What is the total revenue for each quarter of 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$group": {"_id": {"$ceil": {"$divide": [{"$month": "$created_at"}, 3]}}, "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "quarter": "$_id", "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="medium|aggregate|date|orders",
    ),
    BenchmarkQuestion(
        nl_query="How many orders were placed on weekends vs weekdays in 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$project": {"day_of_week": {"$dayOfWeek": "$created_at"}}},
            {"$group": {"_id": {"$cond": [{"$in": ["$day_of_week", [1, 7]]}, "weekend", "weekday"]}, "order_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "day_type": "$_id", "order_count": 1}},
        ],
        tags="hard|aggregate|date|orders",
    ),
]


# ---------------------------------------------------------------------------
# CATEGORY 15 — Multi-collection Hard (15 questions)
# ---------------------------------------------------------------------------

_Q_MULTI: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        nl_query="For each supplier country, what is the total revenue from products sold?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.product_id", "total_revenue": {"$sum": "$items.final_price"}}},
            {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$lookup": {"from": "suppliers", "localField": "product.supplier_id", "foreignField": "_id", "as": "supplier"}},
            {"$unwind": "$supplier"},
            {"$group": {"_id": "$supplier.country", "total_revenue": {"$sum": "$total_revenue"}, "product_count": {"$sum": 1}}},
            {"$sort": {"total_revenue": -1}},
            {"$limit": 15},
            {"$project": {"_id": 0, "country": "$_id", "total_revenue": {"$round": ["$total_revenue", 2]}, "product_count": 1}},
        ],
        tags="hard|lookup|orders|products|suppliers",
    ),
    BenchmarkQuestion(
        nl_query="What is the average order value for customers who opted in to marketing vs those who did not?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$lookup": {"from": "customers", "localField": "customer_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$group": {"_id": "$customer.marketing_opt_in", "avg_order_value": {"$avg": "$total"}, "order_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "marketing_opt_in": "$_id", "avg_order_value": {"$round": ["$avg_order_value", 2]}, "order_count": 1}},
        ],
        tags="hard|lookup|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="Which products have been both viewed and purchased (appear in both view events and orders)? Show top 10 by purchase count.",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.product_id", "purchase_count": {"$sum": "$items.qty"}, "product_name": {"$first": "$items.product_name"}}},
            {"$sort": {"purchase_count": -1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "product_name": 1, "purchase_count": 1}},
        ],
        tags="hard|aggregate|array|orders",
    ),
    BenchmarkQuestion(
        nl_query="For platinum customers, what is their average order value and total spend in 2024?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$match": {"created_at": {"$gte": D2024_START, "$lte": D2024_END}}},
            {"$lookup": {"from": "customers", "localField": "customer_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$match": {"customer.tier": "platinum"}},
            {"$group": {"_id": None, "avg_order_value": {"$avg": "$total"}, "total_spend": {"$sum": "$total"}, "order_count": {"$sum": 1}, "unique_customers": {"$addToSet": "$customer_id"}}},
            {"$project": {"_id": 0, "avg_order_value": {"$round": ["$avg_order_value", 2]}, "total_spend": {"$round": ["$total_spend", 2]}, "order_count": 1, "unique_customers": {"$size": "$unique_customers"}}},
        ],
        tags="hard|lookup|date|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the cancellation rate by customer tier?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$lookup": {"from": "customers", "localField": "customer_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$group": {"_id": "$customer.tier", "total": {"$sum": 1}, "cancelled": {"$sum": {"$cond": [{"$eq": ["$status", "cancelled"]}, 1, 0]}}}},
            {"$project": {"_id": 0, "tier": "$_id", "total": 1, "cancelled": 1, "cancel_rate_pct": {"$round": [{"$multiply": [{"$divide": ["$cancelled", "$total"]}, 100]}, 2]}}},
            {"$sort": {"cancel_rate_pct": -1}},
        ],
        tags="hard|lookup|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="Which products have the highest ratio of 5-star to 1-star reviews (minimum 20 reviews)?",
        collection="reviews",
        operation="aggregate",
        query=[
            {"$group": {"_id": "$product_id", "total": {"$sum": 1}, "five_star": {"$sum": {"$cond": [{"$eq": ["$rating", 5]}, 1, 0]}}, "one_star": {"$sum": {"$cond": [{"$eq": ["$rating", 1]}, 1, 0]}}}},
            {"$match": {"total": {"$gte": 20}, "one_star": {"$gt": 0}}},
            {"$project": {"_id": 1, "total": 1, "five_star": 1, "one_star": 1, "ratio": {"$round": [{"$divide": ["$five_star", "$one_star"]}, 2]}}},
            {"$sort": {"ratio": -1}},
            {"$limit": 10},
            {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$project": {"_id": 0, "product_name": "$product.name", "five_star": 1, "one_star": 1, "ratio": 1}},
        ],
        tags="hard|lookup|reviews|products",
    ),
    BenchmarkQuestion(
        nl_query="What is the average product price per supplier country?",
        collection="products",
        operation="aggregate",
        query=[
            {"$match": {"is_active": True}},
            {"$lookup": {"from": "suppliers", "localField": "supplier_id", "foreignField": "_id", "as": "supplier"}},
            {"$unwind": "$supplier"},
            {"$group": {"_id": "$supplier.country", "avg_price": {"$avg": "$price"}, "product_count": {"$sum": 1}}},
            {"$sort": {"avg_price": -1}},
            {"$limit": 15},
            {"$project": {"_id": 0, "country": "$_id", "avg_price": {"$round": ["$avg_price", 2]}, "product_count": 1}},
        ],
        tags="hard|lookup|products|suppliers",
    ),
    BenchmarkQuestion(
        nl_query="For customers from Italy, what is the total order revenue by year?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$lookup": {"from": "customers", "localField": "customer_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$match": {"customer.country": "Italy"}},
            {"$group": {"_id": {"$year": "$created_at"}, "total_revenue": {"$sum": "$total"}, "order_count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "total_revenue": {"$round": ["$total_revenue", 2]}, "order_count": 1}},
        ],
        tags="hard|lookup|date|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="What is the return rate (returned orders / total orders) per customer country?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$lookup": {"from": "customers", "localField": "customer_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$group": {"_id": "$customer.country", "total": {"$sum": 1}, "returned": {"$sum": {"$cond": [{"$eq": ["$status", "returned"]}, 1, 0]}}}},
            {"$match": {"total": {"$gte": 50}}},
            {"$project": {"_id": 0, "country": "$_id", "total": 1, "returned": 1, "return_rate_pct": {"$round": [{"$multiply": [{"$divide": ["$returned", "$total"]}, 100]}, 2]}}},
            {"$sort": {"return_rate_pct": -1}},
            {"$limit": 15},
        ],
        tags="hard|lookup|orders|customers",
    ),
    BenchmarkQuestion(
        nl_query="For the top 5 revenue-generating product categories, what is the average review rating?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$group": {"_id": "$items.category_id", "total_revenue": {"$sum": "$items.final_price"}}},
            {"$sort": {"total_revenue": -1}},
            {"$limit": 5},
            {"$lookup": {"from": "reviews", "localField": "_id", "foreignField": "product_id", "as": "reviews"}},
            {"$project": {"_id": 1, "total_revenue": {"$round": ["$total_revenue", 2]}, "avg_rating": {"$avg": "$reviews.rating"}, "review_count": {"$size": "$reviews"}}},
            {"$lookup": {"from": "categories", "localField": "_id", "foreignField": "_id", "as": "category"}},
            {"$unwind": {"path": "$category", "preserveNullAndEmptyArrays": True}},
            {"$project": {"_id": 0, "category_name": "$category.name", "total_revenue": 1, "avg_rating": {"$round": ["$avg_rating", 2]}, "review_count": 1}},
        ],
        tags="hard|lookup|orders|categories|reviews",
    ),
    BenchmarkQuestion(
        nl_query="What is the average time between order placement and shipment creation (in hours)?",
        collection="shipments",
        operation="aggregate",
        query=[
            {"$lookup": {"from": "orders", "localField": "order_id", "foreignField": "_id", "as": "order"}},
            {"$unwind": "$order"},
            {"$project": {"hours_to_ship": {"$divide": [{"$subtract": ["$created_at", "$order.created_at"]}, 3600000]}}},
            {"$match": {"hours_to_ship": {"$gte": 0}}},
            {"$group": {"_id": None, "avg_hours": {"$avg": "$hours_to_ship"}}},
            {"$project": {"_id": 0, "avg_hours_to_ship": {"$round": ["$avg_hours", 1]}}},
        ],
        tags="hard|lookup|shipments|orders",
    ),
    BenchmarkQuestion(
        nl_query="Which suppliers have products with an average rating below 3.0?",
        collection="products",
        operation="aggregate",
        query=[
            {"$match": {"rating_count": {"$gt": 10}}},
            {"$group": {"_id": "$supplier_id", "avg_rating": {"$avg": "$rating_avg"}, "product_count": {"$sum": 1}}},
            {"$match": {"avg_rating": {"$lt": 3.0}}},
            {"$lookup": {"from": "suppliers", "localField": "_id", "foreignField": "_id", "as": "supplier"}},
            {"$unwind": "$supplier"},
            {"$project": {"_id": 0, "supplier_name": "$supplier.name", "country": "$supplier.country", "avg_rating": {"$round": ["$avg_rating", 2]}, "product_count": 1}},
            {"$sort": {"avg_rating": 1}},
        ],
        tags="hard|lookup|products|suppliers",
    ),
    BenchmarkQuestion(
        nl_query="How many orders include products from more than one supplier?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$unwind": "$items"},
            {"$lookup": {"from": "products", "localField": "items.product_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$group": {"_id": "$_id", "supplier_ids": {"$addToSet": "$product.supplier_id"}}},
            {"$project": {"supplier_count": {"$size": "$supplier_ids"}}},
            {"$match": {"supplier_count": {"$gt": 1}}},
            {"$count": "multi_supplier_orders"},
        ],
        tags="hard|lookup|array|orders|products|suppliers",
    ),
    BenchmarkQuestion(
        nl_query="What is the total inventory value (quantity * product price) per warehouse?",
        collection="inventory",
        operation="aggregate",
        query=[
            {"$lookup": {"from": "products", "localField": "product_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$group": {"_id": "$warehouse_id", "inventory_value": {"$sum": {"$multiply": ["$quantity", "$product.price"]}}, "item_count": {"$sum": 1}}},
            {"$sort": {"inventory_value": -1}},
            {"$lookup": {"from": "warehouses", "localField": "_id", "foreignField": "_id", "as": "warehouse"}},
            {"$unwind": "$warehouse"},
            {"$project": {"_id": 0, "warehouse_name": "$warehouse.name", "city": "$warehouse.city", "inventory_value": {"$round": ["$inventory_value", 2]}, "item_count": 1}},
        ],
        tags="hard|lookup|inventory|products|warehouses",
    ),
    BenchmarkQuestion(
        nl_query="For each customer tier, what is the average review rating they give?",
        collection="orders",
        operation="aggregate",
        query=[
            {"$lookup": {"from": "customers", "localField": "customer_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$lookup": {"from": "reviews", "localField": "customer_id", "foreignField": "customer_id", "as": "reviews"}},
            {"$unwind": "$reviews"},
            {"$group": {"_id": "$customer.tier", "avg_rating": {"$avg": "$reviews.rating"}, "review_count": {"$sum": 1}}},
            {"$sort": {"avg_rating": -1}},
            {"$project": {"_id": 0, "tier": "$_id", "avg_rating": {"$round": ["$avg_rating", 2]}, "review_count": 1}},
        ],
        tags="hard|lookup|orders|customers|reviews",
    ),
]


# ---------------------------------------------------------------------------
# Full dataset
# ---------------------------------------------------------------------------

QUESTIONS: list[BenchmarkQuestion] = (
    _Q_COUNTS
    + _Q_FIELD_ACCESS
    + _Q_DATE
    + _Q_ARRAY
    + _Q_AGGREGATION
    + _Q_LOOKUP
    + _Q_HARD
    + _Q_EVENTS
    + _Q_SHIPMENTS
    + _Q_INVENTORY
    + _Q_REVIEWS
    + _Q_PRODUCTS
    + _Q_CUSTOMERS
    + _Q_DATE_EXT
    + _Q_MULTI
)

assert len(QUESTIONS) == 170, f"Expected 170 questions, got {len(QUESTIONS)}"
