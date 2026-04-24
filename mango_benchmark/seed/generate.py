"""Synthetic data generator for the mango_ecommerce benchmark database.

Usage:
    python -m mango_benchmark.seed.generate --uri mongodb://localhost:27017
    python -m mango_benchmark.seed.generate --uri mongodb://localhost:27017 --scale 0.1 --drop
    python -m mango_benchmark.seed.generate --help

Scale 0.1 → ~158K docs (~1 min)
Scale 1.0 → ~1.58M docs (~10 min)
"""

from __future__ import annotations

import argparse
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

from bson import ObjectId
from faker import Faker
from pymongo import InsertOne, MongoClient
from pymongo.collection import Collection
from tqdm import tqdm

from mango_benchmark.seed.schema import (
    BASE_COUNTS,
    BRANDS_BY_CATEGORY,
    CARRIERS,
    CARRIER_WEIGHTS,
    CATEGORY_TREE,
    COLORS,
    CURRENCIES,
    CUSTOMER_TIERS,
    COUNTRIES,
    DATE_END,
    DATE_START,
    DB_NAME,
    DEFAULT_BRANDS,
    DEVICES,
    EVENT_TYPES,
    EVENT_WEIGHTS,
    GENDERS,
    LANGUAGES,
    MATERIALS_CLOTHING,
    MATERIALS_HOME,
    ORDER_STATUSES,
    ORDER_STATUS_WEIGHTS,
    PAYMENT_METHODS,
    PAYMENT_WEIGHTS,
    REFERRERS,
    REVIEW_SENTIMENTS,
    SHIPMENT_STATUSES,
    SIZES_CLOTHING,
    SKIN_TYPES,
    SPECS_TEMPLATES,
    SPORTS,
    TIER_WEIGHTS,
)

BATCH_SIZE = 10_000
RANDOM_SEED = 42

fake = Faker()
Faker.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_date(start: datetime = DATE_START, end: datetime = DATE_END) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.random() * delta)


def _rand_date_after(after: datetime, max_days: int = 30) -> datetime:
    days = random.randint(1, max_days)
    result = after + timedelta(days=days)
    return min(result, DATE_END)


def _weighted_choice(choices: list, weights: list) -> Any:
    return random.choices(choices, weights=weights, k=1)[0]


def _bulk_insert(col: Collection, docs: list[dict], label: str) -> None:
    total = len(docs)
    with tqdm(total=total, desc=f"  {label}", unit="doc", leave=False) as bar:
        for i in range(0, total, BATCH_SIZE):
            batch = docs[i : i + BATCH_SIZE]
            col.insert_many(batch, ordered=False)
            bar.update(len(batch))


# ---------------------------------------------------------------------------
# Category generation
# ---------------------------------------------------------------------------


def _flatten_category_tree(tree: list[dict]) -> list[dict]:
    """Flatten the nested category tree into a list of category docs."""
    docs: list[dict] = []

    def _recurse(nodes: list[dict], parent_id: ObjectId | None, parent_path: list[str], level: int) -> None:
        for node in nodes:
            oid = ObjectId()
            path = parent_path + [node["name"]]
            doc = {
                "_id": oid,
                "name": node["name"],
                "slug": node["slug"],
                "level": level,
                "parent_id": parent_id,
                "path": path,
                "description": fake.sentence(nb_words=8),
                "is_active": True,
            }
            docs.append(doc)
            if "children" in node:
                _recurse(node["children"], oid, path, level + 1)

    _recurse(tree, None, [], 1)
    return docs


def generate_categories(col: Collection) -> list[dict]:
    docs = _flatten_category_tree(CATEGORY_TREE)
    _bulk_insert(col, docs, "categories")
    col.create_index("slug", unique=True)
    col.create_index("level")
    col.create_index("parent_id")
    return docs


# ---------------------------------------------------------------------------
# Suppliers
# ---------------------------------------------------------------------------


def generate_suppliers(col: Collection, n: int) -> list[dict]:
    docs = []
    for i in range(n):
        country = random.choice(COUNTRIES)
        docs.append({
            "_id": ObjectId(),
            "name": fake.company(),
            "code": f"SUP-{i+1:04d}",
            "country": country,
            "address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "zip": fake.postcode(),
                "country": country,
            },
            "contact": {
                "name": fake.name(),
                "email": fake.company_email(),
                "phone": fake.phone_number(),
            },
            "tags": random.sample(["electronics", "wholesale", "fashion", "books", "home", "organic", "premium", "eco"], k=random.randint(1, 3)),
            "rating": round(random.uniform(2.5, 5.0), 1),
            "is_active": random.random() > 0.05,
            "created_at": _rand_date(DATE_START, datetime(2023, 6, 1)),
        })
    _bulk_insert(col, docs, "suppliers")
    col.create_index("country")
    col.create_index("is_active")
    return docs


# ---------------------------------------------------------------------------
# Warehouses
# ---------------------------------------------------------------------------


def generate_warehouses(col: Collection, n: int) -> list[dict]:
    warehouse_cities = [
        ("Milan", "Italy", 45.46, 9.19),
        ("Rome", "Italy", 41.90, 12.49),
        ("Berlin", "Germany", 52.52, 13.40),
        ("Munich", "Germany", 48.14, 11.58),
        ("Paris", "France", 48.85, 2.35),
        ("Madrid", "Spain", 40.42, -3.70),
        ("London", "United Kingdom", 51.51, -0.13),
        ("Amsterdam", "Netherlands", 52.37, 4.90),
        ("Vienna", "Austria", 48.21, 16.37),
        ("Warsaw", "Poland", 52.23, 21.01),
        ("Zurich", "Switzerland", 47.38, 8.54),
        ("Barcelona", "Spain", 41.39, 2.15),
        ("Brussels", "Belgium", 50.85, 4.35),
        ("Stockholm", "Sweden", 59.33, 18.07),
        ("Lisbon", "Portugal", 38.72, -9.14),
        ("Tokyo", "Japan", 35.69, 139.69),
        ("New York", "United States", 40.71, -74.01),
        ("Los Angeles", "United States", 34.05, -118.24),
        ("Toronto", "Canada", 43.65, -79.38),
        ("Oslo", "Norway", 59.91, 10.75),
        ("Copenhagen", "Denmark", 55.68, 12.57),
        ("Naples", "Italy", 40.85, 14.27),
        ("Turin", "Italy", 45.07, 7.69),
        ("Frankfurt", "Germany", 50.11, 8.68),
        ("Hamburg", "Germany", 53.55, 10.00),
    ]
    docs = []
    for i in range(min(n, len(warehouse_cities))):
        city, country, lat, lng = warehouse_cities[i]
        docs.append({
            "_id": ObjectId(),
            "name": f"{city} Warehouse",
            "code": f"WH-{i+1:02d}",
            "city": city,
            "country": country,
            "address": {
                "street": fake.street_address(),
                "city": city,
                "zip": fake.postcode(),
                "country": country,
            },
            "location": {"type": "Point", "coordinates": [lng, lat]},
            "capacity": random.randint(5_000, 50_000),
            "is_active": True,
            "created_at": _rand_date(DATE_START, datetime(2023, 3, 1)),
        })
    _bulk_insert(col, docs, "warehouses")
    col.create_index("country")
    col.create_index([("location", "2dsphere")])
    return docs


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


def _make_specs(root_category: str) -> dict:
    tpl = SPECS_TEMPLATES.get(root_category, [])
    specs: dict = {}
    for field in tpl:
        if field == "display_inches":
            specs[field] = round(random.choice([5.5, 6.1, 6.4, 6.7, 13.3, 14.0, 15.6, 27, 32, 55, 65]), 1)
        elif field == "battery_mah":
            specs[field] = random.choice([3000, 4000, 5000, 6000, 8000, 10000])
        elif field == "ram_gb":
            specs[field] = random.choice([4, 8, 16, 32, 64])
        elif field == "storage_gb":
            specs[field] = random.choice([64, 128, 256, 512, 1024])
        elif field == "connectivity":
            specs[field] = random.choice(["WiFi 6", "WiFi 6E", "Bluetooth 5.3", "5G", "4G LTE"])
        elif field == "size":
            specs[field] = random.choice(SIZES_CLOTHING)
        elif field == "color":
            specs[field] = random.choice(COLORS)
        elif field == "material":
            specs[field] = random.choice(MATERIALS_CLOTHING if "Clothing" in root_category else MATERIALS_HOME)
        elif field == "gender":
            specs[field] = random.choice(GENDERS)
        elif field == "author":
            specs[field] = fake.name()
        elif field == "pages":
            specs[field] = random.randint(80, 900)
        elif field == "publisher":
            specs[field] = random.choice(["Penguin", "HarperCollins", "Mondadori", "Random House", "Oxford"])
        elif field == "language":
            specs[field] = random.choice(LANGUAGES)
        elif field == "dimensions":
            w, h, d = random.randint(10, 200), random.randint(10, 200), random.randint(5, 100)
            specs[field] = f"{w}x{h}x{d}cm"
        elif field == "weight_kg":
            specs[field] = round(random.uniform(0.1, 30.0), 2)
        elif field == "sport":
            specs[field] = random.choice(SPORTS)
        elif field == "volume_ml":
            specs[field] = random.choice([30, 50, 100, 150, 200, 250, 500])
        elif field == "skin_type":
            specs[field] = random.choice(SKIN_TYPES)
        elif field == "is_vegan":
            specs[field] = random.random() > 0.5
    return specs


def generate_products(
    col: Collection,
    n: int,
    categories: list[dict],
    suppliers: list[dict],
) -> list[dict]:
    leaf_cats = [c for c in categories if c["level"] == 3]
    supplier_ids = [s["_id"] for s in suppliers]

    docs = []
    for i in range(n):
        cat = random.choice(leaf_cats)
        root_cat = cat["path"][0]
        brands = BRANDS_BY_CATEGORY.get(root_cat, DEFAULT_BRANDS)
        brand = random.choice(brands)
        price = round(random.uniform(3.99, 1499.99), 2)
        cost = round(price * random.uniform(0.3, 0.7), 2)
        rating_count = random.randint(0, 2000)
        docs.append({
            "_id": ObjectId(),
            "sku": f"PRD-{i+1:05d}",
            "name": f"{brand} {fake.catch_phrase()[:40]}",
            "brand": brand,
            "category_id": cat["_id"],
            "category_path": cat["path"],
            "supplier_id": random.choice(supplier_ids),
            "price": price,
            "cost": cost,
            "currency": random.choice(CURRENCIES),
            "stock": random.randint(0, 500),
            "tags": random.sample(
                ["sale", "new", "popular", "eco", "premium", "refurbished", "limited", "bestseller"],
                k=random.randint(0, 3),
            ),
            "specs": _make_specs(root_cat),
            "rating_avg": round(random.uniform(1.0, 5.0), 2) if rating_count > 0 else None,
            "rating_count": rating_count,
            "is_active": random.random() > 0.08,
            "created_at": _rand_date(DATE_START, datetime(2024, 6, 1)),
        })
    _bulk_insert(col, docs, "products")
    col.create_index("category_id")
    col.create_index("supplier_id")
    col.create_index("is_active")
    col.create_index("brand")
    col.create_index("price")
    col.create_index("rating_avg")
    return docs


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


def generate_customers(col: Collection, n: int) -> list[dict]:
    docs = []
    for _ in range(n):
        country = random.choice(COUNTRIES)
        tier = _weighted_choice(CUSTOMER_TIERS, TIER_WEIGHTS)
        created_at = _rand_date(DATE_START, datetime(2025, 6, 1))
        addresses = [
            {
                "type": "shipping",
                "street": fake.street_address(),
                "city": fake.city(),
                "zip": fake.postcode(),
                "country": country,
                "is_default": True,
            }
        ]
        if random.random() > 0.6:
            bill_country = country if random.random() > 0.3 else random.choice(COUNTRIES)
            addresses.append({
                "type": "billing",
                "street": fake.street_address(),
                "city": fake.city(),
                "zip": fake.postcode(),
                "country": bill_country,
                "is_default": True,
            })
        docs.append({
            "_id": ObjectId(),
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "email": fake.unique.email(),
            "phone": fake.phone_number(),
            "tier": tier,
            "country": country,
            "addresses": addresses,
            "preferred_payment": _weighted_choice(PAYMENT_METHODS, PAYMENT_WEIGHTS),
            "marketing_opt_in": random.random() > 0.35,
            "created_at": created_at,
            "last_order_at": None,
        })
    _bulk_insert(col, docs, "customers")
    col.create_index("tier")
    col.create_index("country")
    col.create_index("email", unique=True)
    col.create_index("created_at")
    return docs


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def generate_inventory(
    col: Collection,
    n: int,
    products: list[dict],
    warehouses: list[dict],
) -> list[dict]:
    product_ids = [p["_id"] for p in products]
    warehouse_ids = [w["_id"] for w in warehouses]
    seen: set = set()
    docs = []
    attempts = 0
    while len(docs) < n and attempts < n * 3:
        attempts += 1
        pid = random.choice(product_ids)
        wid = random.choice(warehouse_ids)
        key = (pid, wid)
        if key in seen:
            continue
        seen.add(key)
        qty = random.randint(0, 200)
        reserved = random.randint(0, min(qty, 30))
        available = qty - reserved
        threshold = random.randint(10, 50)
        docs.append({
            "_id": ObjectId(),
            "product_id": pid,
            "warehouse_id": wid,
            "quantity": qty,
            "reserved": reserved,
            "available": available,
            "reorder_threshold": threshold,
            "below_threshold": available < threshold,
            "updated_at": _rand_date(datetime(2025, 1, 1), DATE_END),
        })
    _bulk_insert(col, docs, "inventory")
    col.create_index("product_id")
    col.create_index("warehouse_id")
    col.create_index("below_threshold")
    return docs


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def generate_orders(
    col: Collection,
    n: int,
    customers: list[dict],
    products: list[dict],
) -> tuple[list[dict], dict]:
    """Returns (order_docs, customer_last_order_map)."""
    customer_ids = [c["_id"] for c in customers]
    customer_country = {c["_id"]: c["country"] for c in customers}
    product_lookup = {p["_id"]: p for p in products}
    product_ids = list(product_lookup.keys())

    customer_last_order: dict[ObjectId, datetime] = {}
    docs = []

    for k in range(n):
        customer_id = random.choice(customer_ids)
        created_at = _rand_date()
        status = _weighted_choice(ORDER_STATUSES, ORDER_STATUS_WEIGHTS)
        payment_method = _weighted_choice(PAYMENT_METHODS, PAYMENT_WEIGHTS)
        n_items = random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[30, 25, 18, 12, 7, 4, 2, 2])[0]
        chosen_pids = random.sample(product_ids, min(n_items, len(product_ids)))

        items = []
        subtotal = 0.0
        for pid in chosen_pids:
            p = product_lookup[pid]
            qty = random.randint(1, 5)
            unit_price = p["price"]
            discount_pct = random.choice([0, 0, 0, 5, 10, 15, 20, 25])
            final_price = round(unit_price * qty * (1 - discount_pct / 100), 2)
            subtotal += final_price
            items.append({
                "product_id": pid,
                "product_name": p["name"],
                "category_id": p["category_id"],
                "sku": p["sku"],
                "qty": qty,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "final_price": final_price,
            })

        shipping_cost = 0.0 if subtotal > 50 else round(random.uniform(3.99, 9.99), 2)
        total = round(subtotal + shipping_cost, 2)
        country = customer_country.get(customer_id, "Italy")

        docs.append({
            "_id": ObjectId(),
            "order_number": f"ORD-{created_at.strftime('%Y%m%d')}-{k+1:05d}",
            "customer_id": customer_id,
            "status": status,
            "items": items,
            "subtotal": round(subtotal, 2),
            "shipping_cost": shipping_cost,
            "total": total,
            "currency": "EUR",
            "payment_method": payment_method,
            "shipping_address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "zip": fake.postcode(),
                "country": country,
            },
            "created_at": created_at,
            "updated_at": created_at,
        })

        if customer_id not in customer_last_order or created_at > customer_last_order[customer_id]:
            customer_last_order[customer_id] = created_at

    _bulk_insert(col, docs, "orders")
    col.create_index("customer_id")
    col.create_index("status")
    col.create_index("payment_method")
    col.create_index("created_at")
    col.create_index("total")
    return docs, customer_last_order


# ---------------------------------------------------------------------------
# Update customers.last_order_at
# ---------------------------------------------------------------------------


def _update_customer_last_orders(col: Collection, last_order_map: dict) -> None:
    ops = []
    for cid, ts in last_order_map.items():
        from pymongo import UpdateOne
        ops.append(UpdateOne({"_id": cid}, {"$set": {"last_order_at": ts}}))
    for i in range(0, len(ops), BATCH_SIZE):
        col.bulk_write(ops[i : i + BATCH_SIZE], ordered=False)


# ---------------------------------------------------------------------------
# Shipments
# ---------------------------------------------------------------------------


def generate_shipments(col: Collection, n: int, orders: list[dict]) -> list[dict]:
    shippable_orders = [o for o in orders if o["status"] in ("shipped", "delivered", "returned")]
    if not shippable_orders:
        shippable_orders = orders
    order_pool = random.choices(shippable_orders, k=min(n, len(orders)))

    docs = []
    for order in order_pool[:n]:
        carrier = _weighted_choice(CARRIERS, CARRIER_WEIGHTS)
        created_at = order["created_at"]
        est_delivery = _rand_date_after(created_at, max_days=14)
        is_delivered = order["status"] in ("delivered", "returned")
        actual_delivery = _rand_date_after(est_delivery, max_days=5) if is_delivered else None

        timeline = [{"status": "created", "timestamp": created_at, "location": "Origin"}]
        pickup_ts = _rand_date_after(created_at, max_days=2)
        timeline.append({"status": "picked_up", "timestamp": pickup_ts, "location": "Local Hub"})
        transit_ts = _rand_date_after(pickup_ts, max_days=3)
        timeline.append({"status": "in_transit", "timestamp": transit_ts, "location": "Transit Hub"})
        if order["status"] in ("shipped", "delivered", "returned"):
            ofd_ts = _rand_date_after(transit_ts, max_days=2)
            timeline.append({"status": "out_for_delivery", "timestamp": ofd_ts, "location": "Destination Hub"})
        if is_delivered:
            timeline.append({"status": "delivered", "timestamp": actual_delivery, "location": "Destination"})

        shipment_status = timeline[-1]["status"]

        docs.append({
            "_id": ObjectId(),
            "order_id": order["_id"],
            "carrier": carrier,
            "tracking_number": fake.bothify(text="??################").upper(),
            "status": shipment_status,
            "timeline": timeline,
            "estimated_delivery": est_delivery,
            "actual_delivery": actual_delivery,
            "created_at": created_at,
        })

    _bulk_insert(col, docs, "shipments")
    col.create_index("order_id")
    col.create_index("carrier")
    col.create_index("status")
    return docs


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------


def generate_reviews(
    col: Collection,
    n: int,
    customers: list[dict],
    products: list[dict],
    orders: list[dict],
) -> list[dict]:
    customer_ids = [c["_id"] for c in customers]
    product_ids = [p["_id"] for p in products]
    delivered_order_ids = [o["_id"] for o in orders if o["status"] == "delivered"]

    docs = []
    for _ in range(n):
        rating = random.choices([1, 2, 3, 4, 5], weights=[5, 8, 15, 32, 40])[0]
        sentiment_label, _ = random.choices(
            REVIEW_SENTIMENTS,
            weights=[w for _, w in REVIEW_SENTIMENTS],
        )[0]
        if rating >= 4:
            sentiment_label = "positive"
            score = round(random.uniform(0.3, 1.0), 3)
        elif rating == 3:
            sentiment_label = "neutral"
            score = round(random.uniform(-0.2, 0.3), 3)
        else:
            sentiment_label = "negative"
            score = round(random.uniform(-1.0, -0.1), 3)

        is_verified = random.random() > 0.3
        docs.append({
            "_id": ObjectId(),
            "product_id": random.choice(product_ids),
            "customer_id": random.choice(customer_ids),
            "order_id": random.choice(delivered_order_ids) if is_verified and delivered_order_ids else None,
            "rating": rating,
            "title": fake.sentence(nb_words=6).rstrip("."),
            "body": fake.paragraph(nb_sentences=3),
            "helpful_votes": random.randint(0, 200),
            "is_verified": is_verified,
            "sentiment": {"score": score, "label": sentiment_label},
            "created_at": _rand_date(datetime(2023, 3, 1), DATE_END),
        })

    _bulk_insert(col, docs, "reviews")
    col.create_index("product_id")
    col.create_index("customer_id")
    col.create_index("rating")
    col.create_index("is_verified")
    col.create_index("created_at")
    return docs


# ---------------------------------------------------------------------------
# Events (polymorphic)
# ---------------------------------------------------------------------------


def generate_events(
    col: Collection,
    n: int,
    customers: list[dict],
    products: list[dict],
    orders: list[dict],
) -> None:
    customer_ids = [c["_id"] for c in customers]
    product_ids = [p["_id"] for p in products]
    order_ids = [o["_id"] for o in orders]

    search_terms = [
        "laptop", "iphone", "shoes", "jacket", "sofa", "camera", "headphones",
        "book", "watch", "bag", "gaming", "dress", "sneakers", "tablet", "tv",
    ]

    docs = []
    for _ in range(n):
        event_type = _weighted_choice(EVENT_TYPES, EVENT_WEIGHTS)
        cid = random.choice(customer_ids) if random.random() > 0.25 else None
        session_id = fake.uuid4()
        ts = _rand_date()

        if event_type == "view":
            payload: dict = {
                "duration_seconds": random.randint(3, 300),
                "referrer": random.choice(REFERRERS),
                "device": random.choice(DEVICES),
            }
            doc = {
                "_id": ObjectId(),
                "type": event_type,
                "customer_id": cid,
                "session_id": session_id,
                "product_id": random.choice(product_ids),
                "timestamp": ts,
                "payload": payload,
            }
        elif event_type == "search":
            payload = {
                "query": random.choice(search_terms),
                "results_count": random.randint(0, 200),
                "device": random.choice(DEVICES),
            }
            doc = {
                "_id": ObjectId(),
                "type": event_type,
                "customer_id": cid,
                "session_id": session_id,
                "timestamp": ts,
                "payload": payload,
            }
        elif event_type in ("cart_add", "cart_remove"):
            pid = random.choice(product_ids)
            payload = {
                "qty": random.randint(1, 5),
                "unit_price": round(random.uniform(5.0, 999.0), 2),
            }
            doc = {
                "_id": ObjectId(),
                "type": event_type,
                "customer_id": cid,
                "session_id": session_id,
                "product_id": pid,
                "timestamp": ts,
                "payload": payload,
            }
        elif event_type == "purchase":
            payload = {
                "total": round(random.uniform(10.0, 2000.0), 2),
                "items_count": random.randint(1, 8),
            }
            doc = {
                "_id": ObjectId(),
                "type": event_type,
                "customer_id": cid,
                "session_id": session_id,
                "order_id": random.choice(order_ids) if order_ids else None,
                "timestamp": ts,
                "payload": payload,
            }
        else:  # wishlist
            doc = {
                "_id": ObjectId(),
                "type": event_type,
                "customer_id": cid,
                "session_id": session_id,
                "product_id": random.choice(product_ids),
                "timestamp": ts,
                "payload": {},
            }

        docs.append(doc)

        if len(docs) >= BATCH_SIZE:
            col.insert_many(docs, ordered=False)
            docs = []

    if docs:
        col.insert_many(docs, ordered=False)

    col.create_index("type")
    col.create_index("customer_id")
    col.create_index("timestamp")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def generate_all(uri: str, scale: float = 1.0, drop: bool = False) -> None:
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    # Verify connection
    client.admin.command("ping")
    db = client[DB_NAME]

    if drop:
        print(f"Dropping database '{DB_NAME}'...")
        client.drop_database(DB_NAME)

    def count(name: str) -> int:
        base = BASE_COUNTS[name]
        return max(1, math.ceil(base * scale))

    print(f"\nGenerating mango_ecommerce benchmark (scale={scale:.2f})...")
    print(f"Target: ~{int(sum(BASE_COUNTS.values()) * scale):,} documents\n")

    print("Phase 1: Reference data")
    cats = generate_categories(db["categories"])
    print(f"  categories: {len(cats)}")

    suppliers = generate_suppliers(db["suppliers"], count("suppliers"))
    print(f"  suppliers:  {len(suppliers)}")

    warehouses = generate_warehouses(db["warehouses"], count("warehouses"))
    print(f"  warehouses: {len(warehouses)}")

    print("\nPhase 2: Products & customers")
    products = generate_products(db["products"], count("products"), cats, suppliers)
    print(f"  products:   {len(products)}")

    customers = generate_customers(db["customers"], count("customers"))
    print(f"  customers:  {len(customers)}")

    print("\nPhase 3: Inventory")
    generate_inventory(db["inventory"], count("inventory"), products, warehouses)
    print(f"  inventory:  {count('inventory')}")

    print("\nPhase 4: Transactional data")
    orders, last_order_map = generate_orders(db["orders"], count("orders"), customers, products)
    print(f"  orders:     {len(orders)}")

    _update_customer_last_orders(db["customers"], last_order_map)

    generate_shipments(db["shipments"], count("shipments"), orders)
    print(f"  shipments:  {count('shipments')}")

    generate_reviews(db["reviews"], count("reviews"), customers, products, orders)
    print(f"  reviews:    {count('reviews')}")

    print("\nPhase 5: Events (large collection)")
    with tqdm(total=count("events"), desc="  events", unit="doc") as bar:
        n_events = count("events")
        # call generate_events but wrap with progress tracking via chunks
        generate_events(db["events"], n_events, customers, products, orders)
        bar.update(n_events)

    print(f"\nDone! Total documents: ~{int(sum(BASE_COUNTS.values()) * scale):,}")
    print(f"Database: {DB_NAME} @ {uri}")
    client.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic mango_ecommerce benchmark data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default="mongodb://localhost:27017", help="MongoDB URI")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor (0.1 = dev, 1.0 = full ~1.5M docs)",
    )
    parser.add_argument("--drop", action="store_true", help="Drop database before generating")
    args = parser.parse_args()
    generate_all(uri=args.uri, scale=args.scale, drop=args.drop)


if __name__ == "__main__":
    main()
