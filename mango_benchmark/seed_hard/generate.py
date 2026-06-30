"""Synthetic data generator for the *mango_marketplace* HARD benchmark database.

Usage:
    python -m mango_benchmark.seed_hard.generate --uri mongodb://localhost:27017
    python -m mango_benchmark.seed_hard.generate --uri mongodb://localhost:27017 --scale 0.05 --drop
    python -m mango_benchmark.seed_hard.generate --help

Scale 0.05 -> ~200K docs (quick smoke test)
Scale 1.0  -> ~4M docs   (full run, slow)

This DB is intentionally messy. See ``schema.py`` for the four difficulty
levers. Notable on-purpose inconsistencies the agent must cope with:

  * Multi-currency amounts; only ``fx_rates`` lets you normalise to EUR.
  * Mixed date encodings (ISODate / epoch-millis int / ISO string) in some
    collections.
  * Boolean dialects (true / "Y" / 1 / "yes") in messy collections.
  * Order schema drift: ~20% of orders use a legacy v1 shape (different field
    names + a different status vocabulary) than the v2 majority.
  * Reviews use two rating scales (legacy 1..10 ``score`` vs new 1..5 ``rating``).
  * Soft deletes via ``deleted_at`` instead of physical removal.
  * Optional / sometimes-missing fields, nulls, and polymorphic sub-documents.
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta
from typing import Any

from bson import ObjectId
from faker import Faker
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from tqdm import tqdm

from mango_benchmark.seed_hard.schema import (
    ALLERGENS,
    AB_VARIANTS,
    ATTR_TEMPLATES,
    AUDIT_ACTIONS,
    AUDIT_ENTITIES,
    ACTOR_TYPES,
    BASE_COUNTS,
    BASE_CURRENCY,
    BRANDS_BY_ROOT,
    CARRIERS,
    CARRIER_WEIGHTS,
    CARD_NETWORKS,
    CATEGORY_TREE,
    CHANNELS,
    CHANNEL_WEIGHTS,
    COLORS,
    CONSENT_CHANNELS,
    COUNTRIES,
    COUNTRY_ISO,
    CURRENCY_CODES,
    CURRENCY_RATE,
    CURRENCY_WEIGHTS,
    CURRENCIES,
    DATE_END,
    DATE_START,
    DB_NAME,
    DEFAULT_BRANDS,
    DEVICES,
    EVENT_TYPES,
    EVENT_WEIGHTS,
    FULFILLMENT_MODELS,
    GENDERS,
    INTANGIBLE_ROOTS,
    LICENSE_TYPES,
    LISTING_CONDITIONS,
    LISTING_STATUSES,
    LOCALES,
    LOYALTY_TIERS,
    LOYALTY_TIER_WEIGHTS,
    LOYALTY_TXN_TYPES,
    MATERIALS_CLOTHING,
    MATERIALS_HOME,
    MERCHANT_STATUSES,
    MERCHANT_TYPES,
    MERCHANT_TYPE_WEIGHTS,
    OS_LIST,
    ORDER_STATUS_LEGACY_MAP,
    ORDER_STATUS_WEIGHTS,
    ORDER_STATUSES_V2,
    PAYMENT_METHOD_WEIGHTS,
    PAYMENT_METHODS,
    PAYMENT_STATUSES,
    PLATFORMS,
    PROMO_SCOPES,
    PROMO_TYPES,
    PSP_PROVIDERS,
    REFERRERS,
    RETURN_REASONS,
    RETURN_RESOLUTIONS,
    RETURN_STATUSES,
    SEASONS,
    SENDER_ROLES,
    SERVICE_LEVELS,
    SIZES_CLOTHING,
    SUB_INTERVALS,
    SUB_PLANS,
    SUB_STATUSES,
    TICKET_CATEGORIES,
    TICKET_CHANNELS,
    TICKET_PRIORITIES,
    TICKET_STATUSES,
    UOM,
    USER_SEGMENT_WEIGHTS,
    USER_SEGMENTS,
)

BATCH_SIZE = 10_000
RANDOM_SEED = 1337

fake = Faker()
Faker.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ===========================================================================
# Generic helpers
# ===========================================================================


def _rand_date(start: datetime = DATE_START, end: datetime = DATE_END) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.random() * delta)


def _rand_date_after(after: datetime, max_days: int = 30) -> datetime:
    result = after + timedelta(days=random.randint(1, max_days),
                               seconds=random.randint(0, 86_399))
    return min(result, DATE_END)


def _weighted_choice(choices: list, weights: list) -> Any:
    return random.choices(choices, weights=weights, k=1)[0]


def _messy_date(dt: datetime) -> Any:
    """Return a date in one of three encodings to stress type handling.

    ISODate (60%) | epoch-millis int (25%) | ISO-8601 string (15%).
    """
    r = random.random()
    if r < 0.60:
        return dt
    if r < 0.85:
        return int(dt.timestamp() * 1000)
    return dt.isoformat()


def _messy_bool(value: bool) -> Any:
    """Return a boolean in one of several dialects."""
    if random.random() < 0.55:
        return value
    if value:
        return random.choice(["Y", "true", 1, "yes"])
    return random.choice(["N", "false", 0, "no"])


def _maybe(p: float) -> bool:
    """True with probability ``p`` (use to drop optional fields)."""
    return random.random() < p


def _money(eur_amount: float, currency: str) -> float:
    """Convert an EUR base amount into ``currency`` with small per-doc jitter."""
    rate = CURRENCY_RATE[currency] * random.uniform(0.97, 1.03)
    if currency == "JPY":
        return round(eur_amount * rate)
    return round(eur_amount * rate, 2)


def _bulk_insert(col: Collection, docs: list[dict], label: str) -> None:
    if not docs:
        return
    with tqdm(total=len(docs), desc=f"  {label}", unit="doc", leave=False) as bar:
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i: i + BATCH_SIZE]
            col.insert_many(batch, ordered=False)
            bar.update(len(batch))


class _StreamWriter:
    """Buffered streaming inserter for very large collections."""

    def __init__(self, col: Collection, label: str, total: int) -> None:
        self.col = col
        self.buf: list[dict] = []
        self.bar = tqdm(total=total, desc=f"  {label}", unit="doc", leave=False)

    def add(self, doc: dict) -> None:
        self.buf.append(doc)
        if len(self.buf) >= BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        if self.buf:
            self.col.insert_many(self.buf, ordered=False)
            self.bar.update(len(self.buf))
            self.buf = []

    def close(self) -> None:
        self.flush()
        self.bar.close()


# ===========================================================================
# Reference: currencies + fx_rates
# ===========================================================================


def generate_currencies(col: Collection) -> None:
    docs = []
    for code, _w, rate in CURRENCIES:
        docs.append({
            "_id": code,
            "code": code,
            "name": {
                "EUR": "Euro", "USD": "US Dollar", "GBP": "Pound Sterling",
                "CHF": "Swiss Franc", "PLN": "Polish Zloty", "SEK": "Swedish Krona",
                "JPY": "Japanese Yen", "CAD": "Canadian Dollar",
            }[code],
            "symbol": {"EUR": "€", "USD": "$", "GBP": "£", "CHF": "Fr", "PLN": "zł",
                       "SEK": "kr", "JPY": "¥", "CAD": "C$"}[code],
            "decimals": 0 if code == "JPY" else 2,
            "is_base": code == BASE_CURRENCY,
            "indicative_rate_per_eur": rate,
        })
    _bulk_insert(col, docs, "currencies")


def generate_fx_rates(col: Collection) -> None:
    """Monthly fx rates per non-base currency; needed to normalise revenue."""
    docs = []
    month = datetime(DATE_START.year, DATE_START.month, 1)
    while month <= DATE_END:
        for code, _w, base_rate in CURRENCIES:
            if code == BASE_CURRENCY:
                continue
            # drift over time
            drift = 1.0 + random.uniform(-0.08, 0.08)
            rate = round(base_rate * drift, 6)
            docs.append({
                "_id": ObjectId(),
                "base": BASE_CURRENCY,
                "quote": code,
                "rate": rate,                 # 1 EUR = rate * quote
                "inverse_rate": round(1 / rate, 8),
                "period": month.strftime("%Y-%m"),
                "valid_from": month,
                "source": random.choice(["ecb", "openexchange", "internal"]),
            })
        # advance one month
        month = (month.replace(day=28) + timedelta(days=7)).replace(day=1)
    _bulk_insert(col, docs, "fx_rates")
    col.create_index([("quote", ASCENDING), ("period", ASCENDING)])
    col.create_index("valid_from")


# ===========================================================================
# Categories (arbitrary depth, ancestors[] materialised path)
# ===========================================================================


def generate_categories(col: Collection) -> list[dict]:
    docs: list[dict] = []

    def _recurse(nodes, parent_id, parent_path, ancestors, level, attr_schema):
        for node in nodes:
            oid = ObjectId()
            schema = node.get("attr_schema", attr_schema)
            path = parent_path + [node["name"]]
            doc = {
                "_id": oid,
                "name": node["name"],
                "slug": node["slug"],
                "level": level,
                "parent_id": parent_id,
                "ancestors": ancestors,          # array of {_id, name}
                "path_str": " > ".join(path),
                "attr_schema": schema,
                "is_active": True,
            }
            docs.append(doc)
            if node.get("children"):
                _recurse(node["children"], oid, path,
                         ancestors + [{"_id": oid, "name": node["name"]}],
                         level + 1, schema)

    _recurse(CATEGORY_TREE, None, [], [], 1, "generic")
    _bulk_insert(col, docs, "categories")
    col.create_index("slug", unique=True)
    col.create_index("parent_id")
    col.create_index("level")
    col.create_index("ancestors._id")
    return docs


# ===========================================================================
# Merchants (self-referencing parent org hierarchy)
# ===========================================================================


def generate_merchants(col: Collection, n: int) -> list[dict]:
    docs: list[dict] = []
    # First ~15% are "parent" org accounts with no parent; rest may attach to one
    n_parents = max(1, n // 7)
    parent_ids: list[ObjectId] = []

    for i in range(n):
        oid = ObjectId()
        is_parent = i < n_parents
        parent_id = None
        if not is_parent and parent_ids and _maybe(0.45):
            parent_id = random.choice(parent_ids)
        country = random.choice(COUNTRIES)
        created = _rand_date(DATE_START, datetime(2024, 6, 1))
        mtype = _weighted_choice(MERCHANT_TYPES, MERCHANT_TYPE_WEIGHTS)
        status = random.choice(MERCHANT_STATUSES)
        doc = {
            "_id": oid,
            "legal_name": fake.company(),
            "display_name": fake.company().split()[0] + " Store",
            "merchant_code": f"MCH-{i+1:05d}",
            "type": mtype,
            "parent_merchant_id": parent_id,
            "status": status,
            "fulfillment_model": random.choice(FULFILLMENT_MODELS),
            "country": country,
            "country_code": COUNTRY_ISO.get(country, "XX"),
            "default_currency": _weighted_choice(CURRENCY_CODES, CURRENCY_WEIGHTS),
            "commission_pct": round(random.uniform(5, 22), 1),
            "ratings": {
                "fulfillment": round(random.uniform(3.0, 5.0), 2),
                "communication": round(random.uniform(3.0, 5.0), 2),
                "count": random.randint(0, 5000),
            },
            "contact": {
                "email": fake.company_email(),
                "phone": fake.phone_number(),
            },
            "onboarded_at": created,
            "is_verified": _messy_bool(random.random() > 0.25),
        }
        if _maybe(0.06):                       # soft-deleted merchants
            doc["deleted_at"] = _rand_date_after(created, 400)
        if is_parent:
            parent_ids.append(oid)
        docs.append(doc)

    _bulk_insert(col, docs, "merchants")
    col.create_index("merchant_code", unique=True)
    col.create_index("parent_merchant_id")
    col.create_index("status")
    col.create_index("type")
    col.create_index("country")
    return docs


# ===========================================================================
# Warehouses (geo + capabilities)
# ===========================================================================


def generate_warehouses(col: Collection, n: int) -> list[dict]:
    cities = [
        ("Milan", "Italy", 45.46, 9.19), ("Rome", "Italy", 41.90, 12.49),
        ("Naples", "Italy", 40.85, 14.27), ("Berlin", "Germany", 52.52, 13.40),
        ("Munich", "Germany", 48.14, 11.58), ("Frankfurt", "Germany", 50.11, 8.68),
        ("Paris", "France", 48.85, 2.35), ("Lyon", "France", 45.76, 4.84),
        ("Madrid", "Spain", 40.42, -3.70), ("Barcelona", "Spain", 41.39, 2.15),
        ("London", "United Kingdom", 51.51, -0.13), ("Manchester", "United Kingdom", 53.48, -2.24),
        ("Amsterdam", "Netherlands", 52.37, 4.90), ("Brussels", "Belgium", 50.85, 4.35),
        ("Zurich", "Switzerland", 47.38, 8.54), ("Vienna", "Austria", 48.21, 16.37),
        ("Warsaw", "Poland", 52.23, 21.01), ("Lisbon", "Portugal", 38.72, -9.14),
        ("Stockholm", "Sweden", 59.33, 18.07), ("Oslo", "Norway", 59.91, 10.75),
        ("Copenhagen", "Denmark", 55.68, 12.57), ("Tokyo", "Japan", 35.69, 139.69),
        ("New York", "United States", 40.71, -74.01), ("Los Angeles", "United States", 34.05, -118.24),
        ("Chicago", "United States", 41.88, -87.63), ("Toronto", "Canada", 43.65, -79.38),
        ("Hamburg", "Germany", 53.55, 10.00), ("Turin", "Italy", 45.07, 7.69),
        ("Valencia", "Spain", 39.47, -0.38), ("Rotterdam", "Netherlands", 51.92, 4.48),
    ]
    docs = []
    for i in range(min(n, len(cities))):
        city, country, lat, lng = cities[i]
        docs.append({
            "_id": ObjectId(),
            "code": f"WH-{i+1:02d}",
            "name": f"{city} FC",
            "city": city,
            "country": country,
            "location": {"type": "Point", "coordinates": [lng, lat]},
            "capacity_units": random.randint(20_000, 250_000),
            "capabilities": random.sample(
                ["ambient", "cold_chain", "hazmat", "oversized", "returns", "cross_dock"],
                k=random.randint(1, 4)),
            "is_active": True,
            "opened_at": _rand_date(DATE_START, datetime(2023, 6, 1)),
        })
    _bulk_insert(col, docs, "warehouses")
    col.create_index("country")
    col.create_index([("location", "2dsphere")])
    return docs


# ===========================================================================
# Users (messy: name drift, legacy fields, nested consents, mixed dates)
# ===========================================================================


def generate_users(col: Collection, n: int) -> list[dict]:
    """Returns lightweight summaries [{_id, country, currency, created_at, segment}]."""
    summaries: list[dict] = []
    writer = _StreamWriter(col, "users", n)
    for i in range(n):
        oid = ObjectId()
        country = random.choice(COUNTRIES)
        segment = _weighted_choice(USER_SEGMENTS, USER_SEGMENT_WEIGHTS)
        created = _rand_date(DATE_START, datetime(2025, 9, 1))
        currency = _weighted_choice(CURRENCY_CODES, CURRENCY_WEIGHTS)
        first, last = fake.first_name(), fake.last_name()

        doc: dict = {
            "_id": oid,
            "segment": segment,
            "country": country,
            "preferred_currency": currency,
            "locale": random.choice(LOCALES),
            "created_at": created,
        }

        # --- name representation drift -------------------------------------
        r = random.random()
        if r < 0.6:
            doc["name"] = {"first": first, "last": last}
        elif r < 0.85:
            doc["full_name"] = f"{first} {last}"        # legacy flat field
        else:
            doc["first_name"] = first                    # very old shape
            doc["last_name"] = last

        # --- email field drift ---------------------------------------------
        email = fake.unique.email()
        if _maybe(0.8):
            doc["email"] = email
        else:
            doc["email_address"] = email                 # legacy key

        if _maybe(0.7):
            doc["phone"] = fake.phone_number()

        # --- nested addresses array ----------------------------------------
        addresses = [{
            "label": "home",
            "line1": fake.street_address(),
            "city": fake.city(),
            "postcode": fake.postcode(),
            "country": country,
            "country_code": COUNTRY_ISO.get(country, "XX"),
            "is_default": True,
            "geo": {"type": "Point",
                    "coordinates": [round(random.uniform(-10, 25), 4),
                                    round(random.uniform(36, 60), 4)]},
        }]
        if _maybe(0.35):
            addresses.append({
                "label": random.choice(["work", "billing", "other"]),
                "line1": fake.street_address(),
                "city": fake.city(),
                "postcode": fake.postcode(),
                "country": random.choice(COUNTRIES),
                "is_default": False,
            })
        doc["addresses"] = addresses

        # --- marketing consents (nested object, boolean dialects) ----------
        doc["consents"] = {
            ch: _messy_bool(random.random() > 0.5) for ch in
            random.sample(CONSENT_CHANNELS, k=random.randint(2, len(CONSENT_CHANNELS)))
        }

        # --- profile / metrics ---------------------------------------------
        doc["loyalty_tier"] = _weighted_choice(LOYALTY_TIERS, LOYALTY_TIER_WEIGHTS)
        if _maybe(0.9):
            doc["last_login_at"] = _messy_date(_rand_date_after(created, 600))
        if segment == "b2b":
            doc["company"] = {
                "name": fake.company(),
                "vat_id": fake.bothify("??#########").upper(),
                "credit_limit_eur": random.choice([1000, 5000, 10000, 25000, 50000]),
            }
        if _maybe(0.04):
            doc["deleted_at"] = _rand_date_after(created, 700)   # soft delete
        doc["_schema_v"] = random.choice([1, 2, 2, 3])

        writer.add(doc)
        summaries.append({"_id": oid, "country": country, "currency": currency,
                          "created_at": created, "segment": segment})
    writer.close()
    col.create_index("segment")
    col.create_index("country")
    col.create_index("email")
    col.create_index("created_at")
    col.create_index("loyalty_tier")
    return summaries


# ===========================================================================
# Catalog (polymorphic attributes + nested variant trees w/ options arrays)
# ===========================================================================


def _make_attributes(attr_schema: str) -> dict:
    fields = ATTR_TEMPLATES.get(attr_schema, [])
    a: dict = {}
    for f in fields:
        if f == "display_inches":
            a[f] = random.choice([5.5, 6.1, 6.7, 13.3, 14.0, 15.6, 27, 32, 55, 65])
        elif f == "ram_gb":
            a[f] = random.choice([4, 8, 16, 32, 64])
        elif f == "storage_gb":
            a[f] = random.choice([64, 128, 256, 512, 1024, 2048])
        elif f == "battery_mah":
            a[f] = random.choice([3000, 4000, 5000, 6000])
        elif f == "connectivity":
            a[f] = random.sample(["wifi6", "wifi6e", "bt5.3", "5g", "lte", "nfc"], k=random.randint(1, 3))
        elif f == "color":
            a[f] = random.choice(COLORS)
        elif f == "warranty_months":
            a[f] = random.choice([12, 24, 36])
        elif f == "size":
            a[f] = random.choice(SIZES_CLOTHING)
        elif f == "material":
            a[f] = random.choice(MATERIALS_CLOTHING if attr_schema == "fashion" else MATERIALS_HOME)
        elif f == "gender":
            a[f] = random.choice(GENDERS)
        elif f == "season":
            a[f] = random.choice(SEASONS)
        elif f == "care":
            a[f] = random.sample(["machine_wash", "hand_wash", "dry_clean", "tumble_dry", "iron_low"], k=2)
        elif f == "dimensions_cm":
            a[f] = {"w": random.randint(10, 200), "h": random.randint(10, 200), "d": random.randint(5, 100)}
        elif f == "weight_kg":
            a[f] = round(random.uniform(0.05, 40.0), 2)
        elif f == "assembly_required":
            a[f] = _messy_bool(random.random() > 0.5)
        elif f == "net_weight_g":
            a[f] = random.choice([100, 250, 500, 750, 1000])
        elif f == "organic":
            a[f] = _messy_bool(random.random() > 0.6)
        elif f == "country_of_origin":
            a[f] = random.choice(COUNTRIES)
        elif f == "allergens":
            a[f] = random.sample(ALLERGENS, k=random.randint(1, 3))
        elif f == "expiry_days":
            a[f] = random.choice([7, 14, 30, 90, 180, 365])
        elif f == "part_number":
            a[f] = fake.bothify("??-####-##").upper()
        elif f == "compatible_makes":
            a[f] = random.sample(["BMW", "Audi", "VW", "Fiat", "Ford", "Toyota", "Renault"], k=random.randint(1, 3))
        elif f == "oem":
            a[f] = _messy_bool(random.random() > 0.5)
        elif f == "unit_of_measure":
            a[f] = random.choice(UOM)
        elif f == "pack_size":
            a[f] = random.choice([1, 5, 10, 25, 50, 100])
        elif f == "hazardous":
            a[f] = _messy_bool(random.random() > 0.85)
        elif f == "certification":
            a[f] = random.sample(["CE", "ISO9001", "RoHS", "ATEX", "UL"], k=random.randint(0, 2))
        elif f == "lead_time_days":
            a[f] = random.choice([1, 3, 7, 14, 30])
        elif f == "platform":
            a[f] = random.choice(PLATFORMS)
        elif f == "license_type":
            a[f] = random.choice(LICENSE_TYPES)
        elif f == "region_lock":
            a[f] = random.choice(["EU", "NA", "global", "none"])
        elif f == "file_size_mb":
            a[f] = random.choice([5, 50, 500, 5000, 25000])
        elif f == "duration_minutes":
            a[f] = random.choice([30, 60, 120, 240])
        elif f == "onsite":
            a[f] = _messy_bool(random.random() > 0.5)
        elif f == "sla_hours":
            a[f] = random.choice([4, 8, 24, 48, 72])
        elif f == "coverage_region":
            a[f] = random.choice(["national", "regional", "EU-wide"])
    return a


def _make_variants(base_sku: str, attr_schema: str, base_price_eur: float, tangible: bool) -> list[dict]:
    n_variants = random.choices([1, 2, 3, 4, 5, 6], weights=[35, 25, 15, 12, 8, 5])[0]
    variants = []
    for v in range(n_variants):
        options: list[dict] = []
        if attr_schema == "fashion":
            options.append({"name": "size", "value": random.choice(SIZES_CLOTHING)})
            options.append({"name": "color", "value": random.choice(COLORS)})
        elif attr_schema == "electronics":
            options.append({"name": "storage", "value": f"{random.choice([128,256,512,1024])}GB"})
            options.append({"name": "color", "value": random.choice(COLORS)})
        elif attr_schema in ("home", "automotive", "industrial"):
            options.append({"name": "color", "value": random.choice(COLORS)})
        elif attr_schema == "grocery":
            options.append({"name": "pack", "value": random.choice(["single", "x3", "x6", "x12"])})
        else:
            options.append({"name": "edition", "value": random.choice(["standard", "deluxe", "pro"])})
        price = round(base_price_eur * random.uniform(0.9, 1.25), 2)
        variants.append({
            "variant_id": f"{base_sku}-V{v+1}",
            "options": options,
            "barcode": fake.ean13() if tangible else None,
            "price_eur": price,
            "compare_at_eur": round(price * random.uniform(1.05, 1.4), 2) if _maybe(0.4) else None,
            "weight_g": random.randint(50, 8000) if tangible else 0,
            "media": [f"https://cdn.mango.example/{base_sku.lower()}/{v+1}/{j}.jpg"
                      for j in range(random.randint(1, 4))],
            "is_default": v == 0,
        })
    return variants


def generate_catalog(col: Collection, n: int, categories: list[dict]) -> list[dict]:
    """Returns summaries: [{_id, sku, root, tangible, variants:[{variant_id, price_eur}], category_id, brand}]."""
    leaf_cats = [c for c in categories if not any(
        cc["parent_id"] == c["_id"] for cc in categories)]

    def root_of(cat: dict) -> str:
        if cat["ancestors"]:
            return cat["ancestors"][0]["name"]
        return cat["name"]

    summaries: list[dict] = []
    writer = _StreamWriter(col, "catalog", n)
    for i in range(n):
        cat = random.choice(leaf_cats)
        root = root_of(cat)
        attr_schema = cat["attr_schema"]
        tangible = root not in INTANGIBLE_ROOTS
        brands = BRANDS_BY_ROOT.get(root, DEFAULT_BRANDS)
        brand = random.choice(brands)
        base_price = round(random.uniform(2.99, 2499.99), 2)
        sku = f"CAT-{i+1:06d}"
        variants = _make_variants(sku, attr_schema, base_price, tangible)
        created = _rand_date(DATE_START, datetime(2025, 6, 1))

        doc: dict = {
            "_id": ObjectId(),
            "sku": sku,
            "gtin": fake.ean13() if tangible else None,
            "title": f"{brand} {fake.catch_phrase()[:48]}",
            "brand": brand,
            "category_id": cat["_id"],
            "category_path": cat["path_str"],
            "category_ancestors": [a["_id"] for a in cat["ancestors"]] + [cat["_id"]],
            "attr_schema": attr_schema,
            "attributes": _make_attributes(attr_schema),     # polymorphic
            "variants": variants,                            # array of subdocs w/ option arrays
            "is_tangible": tangible,
            "tags": random.sample(["new", "bestseller", "eco", "premium", "clearance",
                                    "exclusive", "bundle", "limited"], k=random.randint(0, 3)),
            "rating_summary": {
                "avg": round(random.uniform(1.0, 5.0), 2),
                "count": random.randint(0, 4000),
                "histogram": {str(s): random.randint(0, 800) for s in range(1, 6)},
            },
            "created_at": created,
            "_schema_v": random.choice([1, 2, 2]),
        }
        # messy: price sometimes stored top-level as string-with-currency (legacy)
        if _maybe(0.15):
            doc["price"] = f"{variants[0]['price_eur']} EUR"
        if _maybe(0.05):
            doc["deleted_at"] = _rand_date_after(created, 400)

        writer.add(doc)
        summaries.append({
            "_id": doc["_id"], "sku": sku, "root": root, "tangible": tangible,
            "category_id": cat["_id"], "brand": brand,
            "variants": [{"variant_id": v["variant_id"], "price_eur": v["price_eur"]} for v in variants],
        })
    writer.close()
    col.create_index("sku", unique=True)
    col.create_index("brand")
    col.create_index("category_id")
    col.create_index("category_ancestors")
    col.create_index("is_tangible")
    col.create_index("attr_schema")
    col.create_index("variants.variant_id")
    return summaries


# ===========================================================================
# Listings (M:N merchant <-> catalog; per-merchant offer of a product variant)
# ===========================================================================


def generate_listings(col: Collection, n: int, catalog: list[dict],
                       merchants: list[dict]) -> list[dict]:
    """Returns summaries [{_id, product_id, variant_id, merchant_id, price_eur, currency, root, tangible}]."""
    active_merchants = [m for m in merchants if "deleted_at" not in m]
    merchant_ids = [m["_id"] for m in active_merchants]
    merchant_cur = {m["_id"]: m["default_currency"] for m in active_merchants}

    summaries: list[dict] = []
    seen: set = set()
    writer = _StreamWriter(col, "listings", n)
    attempts = 0
    while len(summaries) < n and attempts < n * 4:
        attempts += 1
        product = random.choice(catalog)
        variant = random.choice(product["variants"])
        merchant_id = random.choice(merchant_ids)
        key = (product["_id"], variant["variant_id"], merchant_id)
        if key in seen:
            continue
        seen.add(key)

        currency = merchant_cur[merchant_id]
        price_eur = round(variant["price_eur"] * random.uniform(0.85, 1.20), 2)
        oid = ObjectId()
        listed = _rand_date(DATE_START, datetime(2025, 9, 1))
        doc = {
            "_id": oid,
            "product_id": product["_id"],
            "sku": product["sku"],
            "variant_id": variant["variant_id"],
            "merchant_id": merchant_id,
            "condition": random.choice(LISTING_CONDITIONS),
            "status": random.choice(LISTING_STATUSES),
            "currency": currency,
            "price": _money(price_eur, currency),
            "price_eur_snapshot": price_eur,          # denormalised helper
            "msrp": _money(round(price_eur * random.uniform(1.1, 1.5), 2), currency) if _maybe(0.5) else None,
            "stock_on_hand": random.randint(0, 800),
            "handling_days": random.choice([1, 1, 2, 3, 5]),
            "is_buybox_winner": _messy_bool(random.random() > 0.7),
            "fulfilled_by": random.choice(FULFILLMENT_MODELS),
            "listed_at": listed,
            "tags": random.sample(["fast_ship", "free_return", "gift_wrap", "bulk"], k=random.randint(0, 2)),
        }
        writer.add(doc)
        summaries.append({
            "_id": oid, "product_id": product["_id"], "sku": product["sku"],
            "variant_id": variant["variant_id"], "merchant_id": merchant_id,
            "price_eur": price_eur, "currency": currency,
            "root": product["root"], "tangible": product["tangible"],
        })
    writer.close()
    col.create_index([("product_id", ASCENDING), ("merchant_id", ASCENDING)])
    col.create_index("merchant_id")
    col.create_index("status")
    col.create_index("condition")
    return summaries


# ===========================================================================
# Inventory snapshots (time-series per listing/warehouse)
# ===========================================================================


def generate_inventory_snapshots(col: Collection, n: int, listings: list[dict],
                                  warehouses: list[dict]) -> None:
    tangible_listings = [l for l in listings if l["tangible"]]
    if not tangible_listings:
        tangible_listings = listings
    warehouse_ids = [w["_id"] for w in warehouses]
    writer = _StreamWriter(col, "inventory_snapshots", n)
    for _ in range(n):
        listing = random.choice(tangible_listings)
        on_hand = random.randint(0, 600)
        reserved = random.randint(0, min(on_hand, 60))
        threshold = random.randint(5, 60)
        available = on_hand - reserved
        snap = _rand_date(datetime(2024, 1, 1), DATE_END)
        writer.add({
            "_id": ObjectId(),
            "listing_id": listing["_id"],
            "product_id": listing["product_id"],
            "merchant_id": listing["merchant_id"],
            "warehouse_id": random.choice(warehouse_ids),
            "snapshot_at": _messy_date(snap),
            "on_hand": on_hand,
            "reserved": reserved,
            "available": available,
            "incoming": random.randint(0, 200),
            "reorder_threshold": threshold,
            "below_threshold": available < threshold,
            "unit_cost_eur": round(listing["price_eur"] * random.uniform(0.3, 0.7), 2),
        })
    writer.close()
    col.create_index("listing_id")
    col.create_index("product_id")
    col.create_index("warehouse_id")
    col.create_index("below_threshold")


# ===========================================================================
# Promotions (polymorphic rule arrays, tiered arrays-of-arrays)
# ===========================================================================


def generate_promotions(col: Collection, n: int, categories: list[dict],
                         merchants: list[dict]) -> None:
    cat_ids = [c["_id"] for c in categories]
    merchant_ids = [m["_id"] for m in merchants]
    docs = []
    for i in range(n):
        ptype = random.choice(PROMO_TYPES)
        scope = random.choice(PROMO_SCOPES)
        start = _rand_date(DATE_START, datetime(2025, 9, 1))
        end = _rand_date_after(start, 90)

        # polymorphic conditions array
        conditions = []
        if _maybe(0.7):
            conditions.append({"field": "cart_total_eur", "op": "gte",
                               "value": random.choice([25, 50, 75, 100, 150])})
        if scope == "category":
            conditions.append({"field": "category_id", "op": "in",
                               "value": random.sample(cat_ids, k=random.randint(1, 3))})
        if scope == "merchant":
            conditions.append({"field": "merchant_id", "op": "eq",
                               "value": random.choice(merchant_ids)})
        if _maybe(0.3):
            conditions.append({"field": "user_segment", "op": "in",
                               "value": random.sample(USER_SEGMENTS, k=2)})

        # polymorphic action depending on type
        if ptype == "percentage":
            action = {"kind": "percentage_off", "pct": random.choice([5, 10, 15, 20, 25, 30])}
        elif ptype == "fixed_amount":
            action = {"kind": "amount_off_eur", "amount": random.choice([5, 10, 20, 50])}
        elif ptype == "bogo":
            action = {"kind": "buy_x_get_y", "buy": random.randint(1, 3), "get": 1,
                      "get_discount_pct": random.choice([50, 100])}
        elif ptype == "free_shipping":
            action = {"kind": "free_shipping"}
        elif ptype == "bundle":
            action = {"kind": "bundle_price_eur", "bundle_size": random.randint(2, 4),
                      "price": random.choice([29, 49, 99])}
        else:  # tiered  -> array-of-arrays [threshold, pct]
            action = {"kind": "tiered",
                      "tiers": [[t, p] for t, p in
                                zip([50, 100, 200, 500], sorted(random.sample([5, 10, 15, 20, 25, 30], 4)))]}

        docs.append({
            "_id": ObjectId(),
            "code": fake.bothify("PROMO-????##").upper(),
            "name": fake.catch_phrase()[:40],
            "type": ptype,
            "scope": scope,
            "conditions": conditions,        # array of heterogeneous condition objs
            "action": action,                # polymorphic
            "stackable": _messy_bool(random.random() > 0.6),
            "priority": random.randint(1, 10),
            "usage": {"limit": random.choice([None, 100, 1000, 10000]),
                      "redeemed": random.randint(0, 5000)},
            "valid_from": start,
            "valid_to": end,
            "is_active": end > datetime(2025, 6, 19),
        })
    _bulk_insert(col, docs, "promotions")
    col.create_index("code", unique=True)
    col.create_index("type")
    col.create_index("scope")
    col.create_index("is_active")


# ===========================================================================
# Orders (multi-currency, per-merchant groups, nested adjustments, schema drift)
# ===========================================================================


def _order_status() -> str:
    return _weighted_choice(ORDER_STATUSES_V2, ORDER_STATUS_WEIGHTS)


def generate_orders(col: Collection, n: int, users: list[dict],
                    listings: list[dict]) -> list[dict]:
    """Returns order summaries used by downstream collections."""
    user_by_id = {u["_id"]: u for u in users}
    user_ids = [u["_id"] for u in users]
    summaries: list[dict] = []
    writer = _StreamWriter(col, "orders", n)

    for k in range(n):
        user = user_by_id[random.choice(user_ids)]
        created = _rand_date()
        currency = user["currency"]
        status = _order_status()
        is_legacy = random.random() < 0.20      # v1 schema drift
        channel = _weighted_choice(CHANNELS, CHANNEL_WEIGHTS)

        # pick 1-4 distinct merchants -> per-merchant groups
        n_listings = random.choices([1, 2, 3, 4, 5, 6], weights=[34, 26, 18, 12, 6, 4])[0]
        chosen = random.sample(listings, min(n_listings, len(listings)))
        # group by merchant
        groups_map: dict[ObjectId, list[dict]] = {}
        any_tangible = False
        for listing in chosen:
            groups_map.setdefault(listing["merchant_id"], []).append(listing)
            any_tangible = any_tangible or listing["tangible"]

        groups = []
        items_eur_total = 0.0
        item_refs = []
        for merchant_id, mls in groups_map.items():
            items = []
            group_subtotal_eur = 0.0
            for listing in mls:
                qty = random.randint(1, 4)
                unit_eur = listing["price_eur"]
                unit_cur = _money(unit_eur, currency)
                line_eur = round(unit_eur * qty, 2)
                # nested adjustments array (discount / tax / fee)
                adjustments = []
                disc_pct = random.choice([0, 0, 0, 5, 10, 15, 20])
                if disc_pct:
                    adjustments.append({"type": "discount", "code": fake.bothify("SAVE##"),
                                        "pct": disc_pct,
                                        "amount": round(-line_eur * disc_pct / 100, 2)})
                tax_rate = random.choice([0.0, 0.10, 0.20, 0.22])
                if tax_rate:
                    adjustments.append({"type": "tax", "rate": tax_rate,
                                        "amount": round(line_eur * tax_rate, 2)})
                line_eur_net = line_eur + sum(a["amount"] for a in adjustments)
                group_subtotal_eur += line_eur_net
                items.append({
                    "listing_id": listing["_id"],
                    "product_id": listing["product_id"],
                    "sku": listing["sku"],
                    "variant_id": listing["variant_id"],
                    "qty": qty,
                    "unit_price": unit_cur,
                    "unit_price_eur": unit_eur,
                    "adjustments": adjustments,
                    "line_total_eur": round(line_eur_net, 2),
                })
                item_refs.append({"product_id": listing["product_id"],
                                  "listing_id": listing["_id"], "qty": qty})
            shipping_eur = 0.0 if group_subtotal_eur > 75 else round(random.uniform(2.99, 9.99), 2)
            items_eur_total += group_subtotal_eur + shipping_eur
            groups.append({
                "merchant_id": merchant_id,
                "fulfillment": random.choice(FULFILLMENT_MODELS),
                "items": items,
                "group_subtotal": _money(group_subtotal_eur, currency),
                "group_subtotal_eur": round(group_subtotal_eur, 2),
                "shipping": _money(shipping_eur, currency),
            })

        grand_total_eur = round(items_eur_total, 2)
        ship_addr = {
            "line1": fake.street_address(), "city": fake.city(),
            "postcode": fake.postcode(), "country": user["country"],
            "country_code": COUNTRY_ISO.get(user["country"], "XX"),
        }

        oid = ObjectId()
        if is_legacy:
            # ---- v1 legacy shape: different field names + status vocab ----
            legacy_status = ORDER_STATUS_LEGACY_MAP.get(status, status.upper())
            doc = {
                "_id": oid,
                "order_no": f"L-{created.strftime('%y%m')}-{k+1:06d}",
                "customer_id": user["_id"],            # legacy key for user_id
                "channel": channel,
                "ccy": currency,                       # legacy currency key
                "groups": groups,
                "grand_total": _money(grand_total_eur, currency),
                "grand_total_eur": grand_total_eur,
                "state": legacy_status,                # legacy status field/vocab
                "placed_at": _messy_date(created),     # legacy date key, messy enc
                "_schema_v": 1,
            }
        else:
            # ---- v2 modern shape ------------------------------------------
            doc = {
                "_id": oid,
                "order_ref": f"ORD-{created.strftime('%Y%m%d')}-{k+1:06d}",
                "user_id": user["_id"],
                "channel": channel,
                "currency": currency,
                "groups": groups,
                "totals": {
                    "items_eur": grand_total_eur,
                    "grand_total": _money(grand_total_eur, currency),
                    "grand_total_eur": grand_total_eur,
                },
                "status": status,
                "payment_status": random.choice(PAYMENT_STATUSES),
                "shipping_address": ship_addr,
                "is_gift": _messy_bool(random.random() > 0.85),
                "created_at": created,
                "updated_at": _rand_date_after(created, 20),
                "_schema_v": 2,
            }
            if _maybe(0.3):
                doc["promo_codes"] = [fake.bothify("PROMO-????##").upper()
                                      for _ in range(random.randint(1, 2))]

        writer.add(doc)
        summaries.append({
            "_id": oid, "user_id": user["_id"], "currency": currency,
            "created_at": created, "status": status, "is_legacy": is_legacy,
            "grand_total_eur": grand_total_eur,
            "merchant_ids": list(groups_map.keys()),
            "item_refs": item_refs, "tangible": any_tangible,
        })
    writer.close()
    col.create_index("user_id")
    col.create_index("customer_id")
    col.create_index("status")
    col.create_index("state")
    col.create_index("created_at")
    col.create_index("placed_at")
    col.create_index("groups.merchant_id")
    col.create_index("_schema_v")
    return summaries


# ===========================================================================
# Payments (polymorphic by method, partial captures + embedded refunds)
# ===========================================================================


def generate_payments(col: Collection, n: int, orders: list[dict]) -> None:
    writer = _StreamWriter(col, "payments", n)
    pool = orders if len(orders) >= n else orders
    for i in range(n):
        order = random.choice(pool)
        method = _weighted_choice(PAYMENT_METHODS, PAYMENT_METHOD_WEIGHTS)
        currency = order["currency"]
        amount_eur = order["grand_total_eur"]
        amount = _money(amount_eur, currency)
        status = random.choice(PAYMENT_STATUSES)
        created = _rand_date_after(order["created_at"], 2)

        # polymorphic method details
        if method == "card":
            details = {"network": random.choice(CARD_NETWORKS),
                       "last4": f"{random.randint(0,9999):04d}",
                       "exp": f"{random.randint(1,12):02d}/{random.randint(26,30)}",
                       "psp": random.choice(PSP_PROVIDERS),
                       "three_ds": _messy_bool(random.random() > 0.4)}
        elif method == "paypal":
            details = {"payer_email": fake.email(), "psp": "paypal"}
        elif method == "bank_transfer":
            details = {"iban_last4": f"{random.randint(0,9999):04d}",
                       "reference": fake.bothify("REF########")}
        elif method == "klarna":
            details = {"plan": random.choice(["pay_in_3", "pay_in_30", "financing"]),
                       "installments": random.choice([3, 4, 12])}
        elif method == "gift_card":
            details = {"card_code": fake.bothify("GC-????-####").upper(),
                       "remaining_eur": round(random.uniform(0, 50), 2)}
        elif method == "crypto":
            details = {"asset": random.choice(["BTC", "ETH", "USDC"]),
                       "tx_hash": fake.sha256()[:32]}
        else:
            details = {"wallet": method}

        # embedded refunds array (only for refunded statuses)
        refunds = []
        if status in ("refunded", "partially_refunded"):
            n_ref = random.randint(1, 2)
            for _ in range(n_ref):
                frac = 1.0 if status == "refunded" else random.uniform(0.1, 0.6)
                refunds.append({
                    "refund_id": str(ObjectId()),
                    "amount": _money(round(amount_eur * frac, 2), currency),
                    "amount_eur": round(amount_eur * frac, 2),
                    "reason": random.choice(RETURN_REASONS),
                    "at": _rand_date_after(created, 30),
                })

        captured_eur = amount_eur if status in ("captured", "refunded", "partially_refunded") else 0.0
        writer.add({
            "_id": ObjectId(),
            "order_id": order["_id"],
            "user_id": order["user_id"],
            "method": method,
            "details": details,                  # polymorphic
            "currency": currency,
            "amount": amount,
            "amount_eur": amount_eur,
            "captured_eur": round(captured_eur, 2),
            "status": status,
            "refunds": refunds,
            "fee_eur": round(amount_eur * random.uniform(0.012, 0.029), 2),
            "created_at": created,
        })
    writer.close()
    col.create_index("order_id")
    col.create_index("user_id")
    col.create_index("method")
    col.create_index("status")
    col.create_index("created_at")


# ===========================================================================
# Shipments (multi-package, per-package items, timeline) - tangible only
# ===========================================================================


def generate_shipments(col: Collection, n: int, orders: list[dict]) -> None:
    shippable = [o for o in orders if o["tangible"]
                 and o["status"] in ("shipped", "delivered", "partially_shipped", "refunded")]
    if not shippable:
        shippable = [o for o in orders if o["tangible"]] or orders
    writer = _StreamWriter(col, "shipments", n)
    for _ in range(n):
        order = random.choice(shippable)
        carrier = _weighted_choice(CARRIERS, CARRIER_WEIGHTS)
        created = _rand_date_after(order["created_at"], 3)
        delivered = order["status"] in ("delivered", "refunded")
        est = _rand_date_after(created, 10)
        actual = _rand_date_after(est, 4) if delivered else None

        # one shipment can have multiple packages
        n_pkgs = random.choices([1, 1, 1, 2, 3], weights=[60, 0, 20, 15, 5])[0] or 1
        packages = []
        for p in range(n_pkgs):
            packages.append({
                "package_no": f"{p+1}/{n_pkgs}",
                "weight_kg": round(random.uniform(0.2, 25.0), 2),
                "dims_cm": [random.randint(10, 120) for _ in range(3)],
                "items": [{"product_id": r["product_id"], "qty": r["qty"]}
                          for r in random.sample(order["item_refs"],
                                                 min(len(order["item_refs"]), random.randint(1, 3)))],
            })

        timeline = [{"status": "label_created", "at": created, "location": "Origin Hub"}]
        t = _rand_date_after(created, 2)
        timeline.append({"status": "in_transit", "at": t, "location": "Sort Center"})
        if order["status"] in ("shipped", "delivered", "refunded"):
            t = _rand_date_after(t, 2)
            timeline.append({"status": "out_for_delivery", "at": t, "location": "Last Mile"})
        if delivered:
            timeline.append({"status": "delivered", "at": actual, "location": "Recipient"})
        status = timeline[-1]["status"]
        if random.random() < 0.03:
            status = "exception"
            timeline.append({"status": "exception", "at": _rand_date_after(created, 6),
                             "reason": random.choice(["address_issue", "damaged", "customs_hold"])})

        writer.add({
            "_id": ObjectId(),
            "order_id": order["_id"],
            "merchant_id": random.choice(order["merchant_ids"]),
            "carrier": carrier,
            "service_level": random.choice(SERVICE_LEVELS),
            "tracking_number": fake.bothify("??############").upper(),
            "status": status,
            "packages": packages,                  # array of subdocs w/ item arrays
            "timeline": timeline,
            "estimated_delivery": est,
            "actual_delivery": actual,
            "shipping_cost_eur": round(random.uniform(2.0, 18.0), 2),
            "created_at": created,
        })
    writer.close()
    col.create_index("order_id")
    col.create_index("merchant_id")
    col.create_index("carrier")
    col.create_index("status")


# ===========================================================================
# Returns / RMA
# ===========================================================================


def generate_returns(col: Collection, n: int, orders: list[dict]) -> None:
    eligible = [o for o in orders if o["tangible"]
                and o["status"] in ("delivered", "refunded", "partially_refunded")]
    if not eligible:
        eligible = [o for o in orders if o["tangible"]] or orders
    writer = _StreamWriter(col, "returns", n)
    for i in range(n):
        order = random.choice(eligible)
        requested = _rand_date_after(order["created_at"], 40)
        status = random.choice(RETURN_STATUSES)
        n_lines = min(len(order["item_refs"]), random.randint(1, 2))
        lines = []
        refund_eur = 0.0
        for ref in random.sample(order["item_refs"], n_lines):
            qty = random.randint(1, ref["qty"])
            amt = round(random.uniform(5, 400), 2)
            refund_eur += amt
            lines.append({
                "product_id": ref["product_id"],
                "listing_id": ref["listing_id"],
                "qty": qty,
                "reason": random.choice(RETURN_REASONS),
                "refund_amount_eur": amt,
                "restock": _messy_bool(random.random() > 0.4),
            })
        writer.add({
            "_id": ObjectId(),
            "rma_number": f"RMA-{requested.strftime('%Y')}-{i+1:06d}",
            "order_id": order["_id"],
            "user_id": order["user_id"],
            "status": status,
            "resolution": random.choice(RETURN_RESOLUTIONS) if status in ("refunded", "received") else None,
            "lines": lines,
            "total_refund_eur": round(refund_eur, 2) if status == "refunded" else 0.0,
            "requested_at": requested,
            "closed_at": _rand_date_after(requested, 20) if status in ("refunded", "rejected") else None,
        })
    writer.close()
    col.create_index("order_id")
    col.create_index("user_id")
    col.create_index("status")
    col.create_index("requested_at")


# ===========================================================================
# Reviews (mixed rating scales: legacy 1..10 vs new 1..5, nested replies)
# ===========================================================================


def generate_reviews(col: Collection, n: int, users: list[dict],
                     catalog: list[dict], orders: list[dict]) -> None:
    user_ids = [u["_id"] for u in users]
    product_ids = [c["_id"] for c in catalog]
    delivered = [o["_id"] for o in orders if o["status"] == "delivered"]
    writer = _StreamWriter(col, "reviews", n)
    for _ in range(n):
        legacy_scale = random.random() < 0.25
        verified = random.random() > 0.35
        created = _rand_date(datetime(2022, 3, 1), DATE_END)
        doc: dict = {
            "_id": ObjectId(),
            "product_id": random.choice(product_ids),
            "user_id": random.choice(user_ids),
            "order_id": random.choice(delivered) if verified and delivered else None,
            "title": fake.sentence(nb_words=6).rstrip("."),
            "body": fake.paragraph(nb_sentences=random.randint(1, 4)),
            "is_verified": _messy_bool(verified),
            "helpful_votes": random.randint(0, 350),
            "created_at": created,
        }
        if legacy_scale:
            # legacy: 1..10 in "score" field, no normalised rating
            doc["scale"] = "ten"
            doc["score"] = random.randint(1, 10)
        else:
            r = random.choices([1, 2, 3, 4, 5], weights=[5, 8, 14, 33, 40])[0]
            doc["scale"] = "five"
            doc["rating"] = r
            doc["sentiment"] = {
                "label": "positive" if r >= 4 else "neutral" if r == 3 else "negative",
                "score": round(random.uniform(-1, 1), 3),
            }
        # nested merchant reply (optional)
        if _maybe(0.2):
            doc["replies"] = [{
                "by": "merchant",
                "body": fake.sentence(nb_words=10),
                "at": _rand_date_after(created, 14),
            }]
        if _maybe(0.05):
            doc["flagged"] = {"reason": random.choice(["spam", "abuse", "off_topic"]),
                              "at": _rand_date_after(created, 30)}
        writer.add(doc)
    writer.close()
    col.create_index("product_id")
    col.create_index("user_id")
    col.create_index("rating")
    col.create_index("score")
    col.create_index("scale")
    col.create_index("created_at")


# ===========================================================================
# Subscriptions
# ===========================================================================


def generate_subscriptions(col: Collection, n: int, users: list[dict]) -> None:
    user_ids = [u["_id"] for u in users]
    docs = []
    for i in range(n):
        plan = random.choice(SUB_PLANS)
        interval = random.choice(SUB_INTERVALS)
        status = random.choice(SUB_STATUSES)
        started = _rand_date(DATE_START, datetime(2025, 9, 1))
        currency = _weighted_choice(CURRENCY_CODES, CURRENCY_WEIGHTS)
        price_eur = {"mango_plus": 4.99, "mango_pro": 9.99, "mango_business": 29.99,
                     "fresh_box": 19.99, "coffee_club": 14.99}[plan]
        cycles = random.randint(1, 30)
        docs.append({
            "_id": ObjectId(),
            "user_id": random.choice(user_ids),
            "plan": plan,
            "interval": interval,
            "status": status,
            "currency": currency,
            "price": _money(price_eur, currency),
            "price_eur": price_eur,
            "started_at": started,
            "current_period_end": _rand_date_after(started, 365),
            "cancelled_at": _rand_date_after(started, 300) if status == "cancelled" else None,
            "trial": {"is_trial": status == "trialing",
                      "ends_at": _rand_date_after(started, 14) if status == "trialing" else None},
            "billing_cycles_completed": cycles,
            "mrr_eur": round(price_eur if interval == "monthly"
                             else price_eur / 3 if interval == "quarterly"
                             else price_eur / 12 if interval == "annual"
                             else price_eur * 4.3, 2),
            "auto_renew": _messy_bool(status not in ("cancelled", "paused")),
        })
    _bulk_insert(col, docs, "subscriptions")
    col.create_index("user_id")
    col.create_index("plan")
    col.create_index("status")


# ===========================================================================
# Loyalty accounts + transactions (points ledger)
# ===========================================================================


def generate_loyalty(col_acc: Collection, col_txn: Collection,
                     n_acc: int, n_txn: int, users: list[dict],
                     orders: list[dict]) -> None:
    user_ids = [u["_id"] for u in users]
    acc_users = random.sample(user_ids, min(n_acc, len(user_ids)))
    accounts = []
    acc_ids = []
    for uid in acc_users:
        oid = ObjectId()
        acc_ids.append((oid, uid))
        accounts.append({
            "_id": oid,
            "user_id": uid,
            "tier": _weighted_choice(LOYALTY_TIERS, LOYALTY_TIER_WEIGHTS),
            "points_balance": random.randint(0, 20000),
            "lifetime_points": random.randint(0, 80000),
            "enrolled_at": _rand_date(DATE_START, datetime(2025, 6, 1)),
            "tier_expires_at": _rand_date(datetime(2025, 6, 1), DATE_END),
        })
    _bulk_insert(col_acc, accounts, "loyalty_accounts")
    col_acc.create_index("user_id", unique=True)
    col_acc.create_index("tier")

    order_ids = [o["_id"] for o in orders]
    writer = _StreamWriter(col_txn, "loyalty_transactions", n_txn)
    for _ in range(n_txn):
        acc_id, uid = random.choice(acc_ids)
        txn_type = random.choice(LOYALTY_TXN_TYPES)
        if txn_type in ("redeem", "expire"):
            points = -random.randint(50, 2000)
        else:
            points = random.randint(10, 1500)
        writer.add({
            "_id": ObjectId(),
            "account_id": acc_id,
            "user_id": uid,
            "type": txn_type,
            "points": points,
            "order_id": random.choice(order_ids) if txn_type == "earn" and order_ids else None,
            "balance_after": random.randint(0, 25000),
            "created_at": _rand_date(DATE_START, DATE_END),
            "note": fake.sentence(nb_words=5) if _maybe(0.2) else None,
        })
    writer.close()
    col_txn.create_index("account_id")
    col_txn.create_index("user_id")
    col_txn.create_index("type")
    col_txn.create_index("created_at")


# ===========================================================================
# Support tickets (threaded messages, polymorphic channel, mixed dates)
# ===========================================================================


def generate_support_tickets(col: Collection, n: int, users: list[dict],
                             orders: list[dict]) -> None:
    user_ids = [u["_id"] for u in users]
    order_ids = [o["_id"] for o in orders]
    writer = _StreamWriter(col, "support_tickets", n)
    for i in range(n):
        channel = random.choice(TICKET_CHANNELS)
        category = random.choice(TICKET_CATEGORIES)
        status = random.choice(TICKET_STATUSES)
        opened = _rand_date(DATE_START, DATE_END)
        n_msgs = random.randint(1, 6)
        messages = []
        last = opened
        for m in range(n_msgs):
            role = "customer" if m == 0 else random.choice(SENDER_ROLES)
            last = _rand_date_after(last, 3)
            msg = {
                "seq": m,
                "sender_role": role,
                "body": fake.paragraph(nb_sentences=random.randint(1, 3)),
                "at": _messy_date(last),
            }
            if _maybe(0.15):
                msg["attachments"] = [{"name": fake.file_name(),
                                       "size_kb": random.randint(10, 5000)}
                                      for _ in range(random.randint(1, 2))]
            messages.append(msg)

        first_response_mins = random.randint(2, 1440) if n_msgs > 1 else None
        writer.add({
            "_id": ObjectId(),
            "ticket_no": f"TCK-{opened.strftime('%Y')}-{i+1:06d}",
            "user_id": random.choice(user_ids),
            "order_id": random.choice(order_ids) if category in ("order_issue", "returns", "payment") and _maybe(0.7) else None,
            "channel": channel,
            "category": category,
            "priority": random.choice(TICKET_PRIORITIES),
            "status": status,
            "subject": fake.sentence(nb_words=6).rstrip("."),
            "messages": messages,                       # threaded array
            "csat": random.randint(1, 5) if status in ("resolved", "closed") and _maybe(0.6) else None,
            "first_response_minutes": first_response_mins,
            "tags": random.sample(["vip", "escalated", "bug", "billing", "fraud_check"], k=random.randint(0, 2)),
            "opened_at": opened,
            "resolved_at": _rand_date_after(opened, 10) if status in ("resolved", "closed") else None,
        })
    writer.close()
    col.create_index("user_id")
    col.create_index("status")
    col.create_index("category")
    col.create_index("channel")
    col.create_index("opened_at")


# ===========================================================================
# Ledger entries (double-entry; balancing debit/credit pairs per order)
# ===========================================================================


def generate_ledger(col: Collection, n: int, orders: list[dict]) -> None:
    writer = _StreamWriter(col, "ledger_entries", n)
    paid_orders = [o for o in orders if o["status"] in (
        "paid", "shipped", "delivered", "partially_shipped", "refunded", "partially_refunded")]
    if not paid_orders:
        paid_orders = orders
    emitted = 0
    while emitted < n:
        order = random.choice(paid_orders)
        total = order["grand_total_eur"]
        cogs = round(total * random.uniform(0.45, 0.7), 2)
        fee = round(total * random.uniform(0.012, 0.03), 2)
        commission = round(total * random.uniform(0.05, 0.18), 2)
        txn_id = ObjectId()
        ts = _rand_date_after(order["created_at"], 3)
        # balanced set of postings for one order
        postings = [
            ("revenue", "credit", total),
            ("cogs", "debit", cogs),
            ("payment_fee", "debit", fee),
            ("platform_fee", "credit", commission),
            ("merchant_payable", "credit", round(total - commission - fee, 2)),
        ]
        for account, side, amount in postings:
            if emitted >= n:
                break
            writer.add({
                "_id": ObjectId(),
                "txn_id": txn_id,             # links the balancing group
                "order_id": order["_id"],
                "account": account,
                "side": side,                # debit | credit
                "amount_eur": amount,
                "currency": "EUR",
                "merchant_id": random.choice(order["merchant_ids"]),
                "posted_at": ts,
            })
            emitted += 1
    writer.close()
    col.create_index("txn_id")
    col.create_index("order_id")
    col.create_index("account")
    col.create_index("posted_at")


# ===========================================================================
# Audit logs (polymorphic before/after diffs, mixed dates)
# ===========================================================================


def generate_audit_logs(col: Collection, n: int, users: list[dict],
                        merchants: list[dict], orders: list[dict]) -> None:
    user_ids = [u["_id"] for u in users]
    merchant_ids = [m["_id"] for m in merchants]
    order_ids = [o["_id"] for o in orders]
    writer = _StreamWriter(col, "audit_logs", n)
    for _ in range(n):
        entity = random.choice(AUDIT_ENTITIES)
        action = random.choice(AUDIT_ACTIONS)
        actor_type = random.choice(ACTOR_TYPES)
        entity_id = {
            "order": random.choice(order_ids) if order_ids else ObjectId(),
            "merchant": random.choice(merchant_ids),
            "user": random.choice(user_ids),
        }.get(entity, ObjectId())

        changes = {}
        if action in ("update", "price_change", "status_change"):
            if action == "price_change":
                old = round(random.uniform(5, 500), 2)
                changes = {"price": {"from": old, "to": round(old * random.uniform(0.8, 1.2), 2)}}
            elif action == "status_change":
                changes = {"status": {"from": random.choice(["active", "paused"]),
                                      "to": random.choice(["active", "suspended", "delisted"])}}
            else:
                field = random.choice(["title", "stock", "is_active", "tier"])
                changes = {field: {"from": fake.word(), "to": fake.word()}}

        writer.add({
            "_id": ObjectId(),
            "entity_type": entity,
            "entity_id": entity_id,
            "action": action,
            "actor": {"type": actor_type,
                      "id": random.choice(user_ids) if actor_type == "user" else
                            random.choice(merchant_ids) if actor_type == "merchant" else None,
                      "ip": fake.ipv4() if actor_type in ("user", "merchant", "admin") else None},
            "changes": changes,              # polymorphic diff
            "request_id": fake.uuid4(),
            "at": _messy_date(_rand_date()),
        })
    writer.close()
    col.create_index("entity_type")
    col.create_index("entity_id")
    col.create_index("action")
    col.create_index("actor.type")


# ===========================================================================
# Events (heterogeneous clickstream, polymorphic payload, mixed dates)
# ===========================================================================


def generate_events(col: Collection, n: int, users: list[dict],
                    catalog: list[dict], orders: list[dict]) -> None:
    user_ids = [u["_id"] for u in users]
    product_ids = [c["_id"] for c in catalog]
    order_ids = [o["_id"] for o in orders]
    search_terms = ["laptop", "iphone", "sneakers", "coffee", "sofa", "gpu", "headphones",
                    "watch", "dress", "tyres", "wine", "license", "drill", "monitor"]
    writer = _StreamWriter(col, "events", n)
    for _ in range(n):
        etype = _weighted_choice(EVENT_TYPES, EVENT_WEIGHTS)
        uid = random.choice(user_ids) if random.random() > 0.30 else None  # anon traffic
        session = fake.uuid4()
        ts = _rand_date()
        ctx = {
            "device": random.choice(DEVICES),
            "os": random.choice(OS_LIST),
            "referrer": random.choice(REFERRERS),
            "ab_variant": random.choice(AB_VARIANTS),
            "ip": fake.ipv4(),
        }

        if etype == "page_view":
            payload: dict = {"url": fake.uri_path(), "duration_s": random.randint(1, 600),
                             "product_id": random.choice(product_ids) if _maybe(0.5) else None}
        elif etype == "search":
            payload = {"q": random.choice(search_terms), "results": random.randint(0, 500),
                       "filters": random.sample(["price", "brand", "rating", "in_stock"], k=random.randint(0, 2))}
        elif etype in ("add_to_cart", "remove_from_cart", "wishlist_add"):
            payload = {"product_id": random.choice(product_ids), "qty": random.randint(1, 5),
                       "unit_price_eur": round(random.uniform(5, 999), 2)}
        elif etype == "checkout_start":
            payload = {"cart_value_eur": round(random.uniform(10, 2000), 2),
                       "items": random.randint(1, 8)}
        elif etype == "purchase":
            payload = {"order_id": random.choice(order_ids) if order_ids else None,
                       "value_eur": round(random.uniform(10, 2000), 2)}
        elif etype == "review_submit":
            payload = {"product_id": random.choice(product_ids), "rating": random.randint(1, 5)}
        elif etype == "support_open":
            payload = {"category": random.choice(TICKET_CATEGORIES)}
        else:  # app_open
            payload = {"cold_start": _messy_bool(random.random() > 0.5)}

        writer.add({
            "_id": ObjectId(),
            "type": etype,
            "user_id": uid,
            "session_id": session,
            "context": ctx,
            "payload": payload,             # polymorphic per type
            "ts": _messy_date(ts),          # mixed date encoding
        })
    writer.close()
    col.create_index("type")
    col.create_index("user_id")
    col.create_index("session_id")


# ===========================================================================
# Orchestrator
# ===========================================================================


def generate_all(uri: str, scale: float = 1.0, drop: bool = False) -> None:
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    client.admin.command("ping")
    db = client[DB_NAME]

    if drop:
        print(f"Dropping database '{DB_NAME}'...")
        client.drop_database(DB_NAME)

    def count(name: str) -> int:
        return max(1, math.ceil(BASE_COUNTS[name] * scale))

    print(f"\nGenerating {DB_NAME} HARD benchmark (scale={scale:.3f})")
    print(f"Target: ~{int(sum(BASE_COUNTS.values()) * scale):,} documents\n")

    print("Phase 1: Reference")
    generate_currencies(db["currencies"])
    generate_fx_rates(db["fx_rates"])
    cats = generate_categories(db["categories"])
    print(f"  categories: {len(cats)}")
    merchants = generate_merchants(db["merchants"], count("merchants"))
    print(f"  merchants:  {len(merchants)}")
    warehouses = generate_warehouses(db["warehouses"], count("warehouses"))
    print(f"  warehouses: {len(warehouses)}")

    print("\nPhase 2: Catalog & users")
    users = generate_users(db["users"], count("users"))
    print(f"  users:      {len(users)}")
    catalog = generate_catalog(db["catalog"], count("catalog"), cats)
    print(f"  catalog:    {len(catalog)}")
    listings = generate_listings(db["listings"], count("listings"), catalog, merchants)
    print(f"  listings:   {len(listings)}")
    generate_inventory_snapshots(db["inventory_snapshots"], count("inventory_snapshots"),
                                 listings, warehouses)
    print(f"  inventory:  {count('inventory_snapshots')}")
    generate_promotions(db["promotions"], count("promotions"), cats, merchants)
    print(f"  promotions: {count('promotions')}")

    print("\nPhase 3: Orders & financials")
    orders = generate_orders(db["orders"], count("orders"), users, listings)
    print(f"  orders:     {len(orders)}")
    generate_payments(db["payments"], count("payments"), orders)
    print(f"  payments:   {count('payments')}")
    generate_shipments(db["shipments"], count("shipments"), orders)
    print(f"  shipments:  {count('shipments')}")
    generate_returns(db["returns"], count("returns"), orders)
    print(f"  returns:    {count('returns')}")
    generate_ledger(db["ledger_entries"], count("ledger_entries"), orders)
    print(f"  ledger:     {count('ledger_entries')}")

    print("\nPhase 4: Engagement")
    generate_reviews(db["reviews"], count("reviews"), users, catalog, orders)
    print(f"  reviews:    {count('reviews')}")
    generate_subscriptions(db["subscriptions"], count("subscriptions"), users)
    print(f"  subs:       {count('subscriptions')}")
    generate_loyalty(db["loyalty_accounts"], db["loyalty_transactions"],
                     count("loyalty_accounts"), count("loyalty_transactions"), users, orders)
    print(f"  loyalty:    {count('loyalty_accounts')} acc / {count('loyalty_transactions')} txn")
    generate_support_tickets(db["support_tickets"], count("support_tickets"), users, orders)
    print(f"  tickets:    {count('support_tickets')}")
    generate_audit_logs(db["audit_logs"], count("audit_logs"), users, merchants, orders)
    print(f"  audit:      {count('audit_logs')}")

    print("\nPhase 5: Events (largest)")
    generate_events(db["events"], count("events"), users, catalog, orders)
    print(f"  events:     {count('events')}")

    print(f"\nDone. Database '{DB_NAME}' @ {uri}")
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the hard mango_marketplace benchmark database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default="mongodb://localhost:27017", help="MongoDB URI")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor (0.05 = quick, 1.0 = full ~4M docs)")
    parser.add_argument("--drop", action="store_true", help="Drop database before generating")
    args = parser.parse_args()
    generate_all(uri=args.uri, scale=args.scale, drop=args.drop)


if __name__ == "__main__":
    main()
