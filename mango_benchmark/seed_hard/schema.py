"""Domain constants for the *mango_marketplace* synthetic benchmark database.

This is the HARD benchmark DB — deliberately much messier and more relational
than ``mango_ecommerce`` so it stresses a text-to-MongoDB agent on four axes:

1. More domains / collections (multi-vendor marketplace + light fintech).
2. Complex relationships (M:N listings, self-referencing merchant & category
   trees, multi-hop lookups, double-entry ledger, multi-currency via fx_rates).
3. Deep nesting & arrays (variant trees, per-merchant order groups, threaded
   tickets, polymorphic promotion rule arrays).
4. Messy / inconsistent data (schema drift, mixed types in the same field,
   legacy field names, mixed date encodings, multi-currency, nulls, soft
   deletes, boolean dialects).

Everything here is *data*; the actual generation lives in ``generate.py``.
"""

from __future__ import annotations

from datetime import datetime

DB_NAME = "mango_marketplace"

DATE_START = datetime(2022, 1, 1)
DATE_END = datetime(2025, 12, 31, 23, 59, 59)

# ---------------------------------------------------------------------------
# Currencies & FX  (multi-currency is a core difficulty lever)
# ---------------------------------------------------------------------------

BASE_CURRENCY = "EUR"

# code, weight, approx units-per-EUR (the fx_rates collection adds noise/drift)
CURRENCIES: list[tuple[str, float, float]] = [
    ("EUR", 0.34, 1.00),
    ("USD", 0.26, 1.08),
    ("GBP", 0.14, 0.85),
    ("CHF", 0.07, 0.96),
    ("PLN", 0.06, 4.32),
    ("SEK", 0.05, 11.30),
    ("JPY", 0.05, 162.0),
    ("CAD", 0.03, 1.46),
]
CURRENCY_CODES = [c[0] for c in CURRENCIES]
CURRENCY_WEIGHTS = [c[1] for c in CURRENCIES]
CURRENCY_RATE = {c[0]: c[2] for c in CURRENCIES}

# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

COUNTRIES = [
    "Italy", "Italy", "Italy",
    "Germany", "Germany",
    "France", "France",
    "Spain",
    "United Kingdom", "United Kingdom",
    "United States", "United States",
    "Netherlands", "Belgium", "Switzerland", "Austria", "Poland",
    "Portugal", "Sweden", "Norway", "Denmark", "Japan", "Canada",
]

# code used in some collections instead of the full name (drift lever)
COUNTRY_ISO = {
    "Italy": "IT", "Germany": "DE", "France": "FR", "Spain": "ES",
    "United Kingdom": "GB", "United States": "US", "Netherlands": "NL",
    "Belgium": "BE", "Switzerland": "CH", "Austria": "AT", "Poland": "PL",
    "Portugal": "PT", "Sweden": "SE", "Norway": "NO", "Denmark": "DK",
    "Japan": "JP", "Canada": "CA",
}

# ---------------------------------------------------------------------------
# Users / buyers
# ---------------------------------------------------------------------------

USER_SEGMENTS = ["new", "regular", "vip", "at_risk", "churned", "b2b"]
USER_SEGMENT_WEIGHTS = [0.20, 0.42, 0.10, 0.12, 0.10, 0.06]

LOCALES = ["it-IT", "de-DE", "fr-FR", "en-GB", "en-US", "es-ES", "nl-NL", "ja-JP"]

CONSENT_CHANNELS = ["email", "sms", "push", "phone", "postal"]

# ---------------------------------------------------------------------------
# Merchants / sellers  (self-referencing org hierarchy)
# ---------------------------------------------------------------------------

MERCHANT_TYPES = ["brand", "reseller", "marketplace_3p", "dropshipper", "first_party"]
MERCHANT_TYPE_WEIGHTS = [0.18, 0.30, 0.34, 0.12, 0.06]

# status enum that changed over time (legacy values still present)
MERCHANT_STATUSES = ["active", "active", "active", "suspended", "onboarding", "offboarded",
                     "ACTIVE", "pending_review"]  # mixed-case + legacy = drift

FULFILLMENT_MODELS = ["merchant", "platform", "hybrid"]

# ---------------------------------------------------------------------------
# Category tree (ARBITRARY depth; ancestors[] + parent_id, attribute schemas)
# ---------------------------------------------------------------------------

CATEGORY_TREE: list[dict] = [
    {"name": "Electronics", "slug": "electronics", "attr_schema": "electronics", "children": [
        {"name": "Mobile", "slug": "mobile", "children": [
            {"name": "Smartphones", "slug": "smartphones", "children": [
                {"name": "Flagship", "slug": "flagship"},
                {"name": "Mid-range", "slug": "midrange"},
                {"name": "Rugged", "slug": "rugged"},
            ]},
            {"name": "Tablets", "slug": "tablets"},
            {"name": "Wearables", "slug": "wearables", "children": [
                {"name": "Smartwatches", "slug": "smartwatches"},
                {"name": "Fitness Bands", "slug": "fitness-bands"},
            ]},
        ]},
        {"name": "Computing", "slug": "computing", "children": [
            {"name": "Laptops", "slug": "laptops", "children": [
                {"name": "Ultrabooks", "slug": "ultrabooks"},
                {"name": "Gaming Laptops", "slug": "gaming-laptops"},
                {"name": "Workstations", "slug": "workstations"},
            ]},
            {"name": "Components", "slug": "components", "children": [
                {"name": "GPUs", "slug": "gpus"},
                {"name": "CPUs", "slug": "cpus"},
                {"name": "Memory", "slug": "memory"},
            ]},
        ]},
        {"name": "Audio", "slug": "audio", "children": [
            {"name": "Headphones", "slug": "headphones"},
            {"name": "Speakers", "slug": "speakers"},
        ]},
    ]},
    {"name": "Fashion", "slug": "fashion", "attr_schema": "fashion", "children": [
        {"name": "Womenswear", "slug": "womenswear", "children": [
            {"name": "Dresses", "slug": "dresses"},
            {"name": "Knitwear", "slug": "knitwear"},
            {"name": "Outerwear", "slug": "w-outerwear"},
        ]},
        {"name": "Menswear", "slug": "menswear", "children": [
            {"name": "Shirts", "slug": "shirts"},
            {"name": "Trousers", "slug": "trousers"},
            {"name": "Outerwear", "slug": "m-outerwear"},
        ]},
        {"name": "Footwear", "slug": "footwear", "children": [
            {"name": "Sneakers", "slug": "sneakers"},
            {"name": "Formal", "slug": "formal-shoes"},
            {"name": "Boots", "slug": "boots"},
        ]},
        {"name": "Accessories", "slug": "fashion-accessories", "children": [
            {"name": "Bags", "slug": "bags"},
            {"name": "Watches", "slug": "fashion-watches"},
        ]},
    ]},
    {"name": "Home & Living", "slug": "home", "attr_schema": "home", "children": [
        {"name": "Furniture", "slug": "furniture", "children": [
            {"name": "Sofas", "slug": "sofas"},
            {"name": "Storage", "slug": "storage"},
        ]},
        {"name": "Kitchen", "slug": "kitchen", "children": [
            {"name": "Cookware", "slug": "cookware"},
            {"name": "Small Appliances", "slug": "small-appliances"},
        ]},
        {"name": "Decor", "slug": "decor"},
    ]},
    {"name": "Grocery", "slug": "grocery", "attr_schema": "grocery", "children": [
        {"name": "Beverages", "slug": "beverages", "children": [
            {"name": "Coffee", "slug": "coffee"},
            {"name": "Wine", "slug": "wine"},
        ]},
        {"name": "Pantry", "slug": "pantry"},
        {"name": "Fresh", "slug": "fresh", "children": [
            {"name": "Cheese", "slug": "cheese"},
            {"name": "Produce", "slug": "produce"},
        ]},
    ]},
    {"name": "Automotive", "slug": "automotive", "attr_schema": "automotive", "children": [
        {"name": "Parts", "slug": "parts", "children": [
            {"name": "Brakes", "slug": "brakes"},
            {"name": "Filters", "slug": "filters"},
        ]},
        {"name": "Tyres", "slug": "tyres"},
        {"name": "Accessories", "slug": "auto-accessories"},
    ]},
    {"name": "Industrial & B2B", "slug": "industrial", "attr_schema": "industrial", "children": [
        {"name": "MRO Supplies", "slug": "mro"},
        {"name": "Safety", "slug": "safety"},
        {"name": "Packaging", "slug": "packaging"},
    ]},
    {"name": "Digital Goods", "slug": "digital", "attr_schema": "digital", "children": [
        {"name": "Software Licenses", "slug": "software"},
        {"name": "E-books", "slug": "ebooks"},
        {"name": "Game Keys", "slug": "game-keys"},
    ]},
    {"name": "Services", "slug": "services", "attr_schema": "service", "children": [
        {"name": "Installation", "slug": "installation"},
        {"name": "Extended Warranty", "slug": "warranty"},
        {"name": "Subscriptions Box", "slug": "subscription-box"},
    ]},
]

# Which roots are intangible (no shipment, no physical inventory) -> polymorphism
INTANGIBLE_ROOTS = {"Digital Goods", "Services"}

# ---------------------------------------------------------------------------
# Per-attr-schema attribute templates (catalog "attributes" sub-document)
# Heterogeneous keys per category root => polymorphic documents.
# ---------------------------------------------------------------------------

ATTR_TEMPLATES: dict[str, list[str]] = {
    "electronics": ["display_inches", "ram_gb", "storage_gb", "battery_mah", "connectivity", "color", "warranty_months"],
    "fashion":     ["size", "color", "material", "gender", "season", "care"],
    "home":        ["material", "dimensions_cm", "weight_kg", "color", "assembly_required"],
    "grocery":     ["net_weight_g", "organic", "country_of_origin", "allergens", "expiry_days"],
    "automotive":  ["part_number", "compatible_makes", "material", "weight_kg", "oem"],
    "industrial":  ["unit_of_measure", "pack_size", "hazardous", "certification", "lead_time_days"],
    "digital":     ["platform", "license_type", "region_lock", "file_size_mb"],
    "service":     ["duration_minutes", "onsite", "sla_hours", "coverage_region"],
}

SIZES_CLOTHING = ["XS", "S", "M", "L", "XL", "XXL"]
SHOE_SIZES = [38, 39, 40, 41, 42, 43, 44, 45]
COLORS = ["Black", "White", "Red", "Blue", "Green", "Yellow", "Grey", "Navy", "Beige", "Pink", "Silver", "Gold"]
MATERIALS_CLOTHING = ["Cotton", "Polyester", "Wool", "Linen", "Silk", "Denim", "Leather", "Recycled"]
MATERIALS_HOME = ["Oak", "Pine", "Steel", "Aluminium", "Glass", "Ceramic", "Bamboo", "MDF"]
SEASONS = ["SS24", "AW24", "SS25", "AW25", "all-season"]
ALLERGENS = ["gluten", "nuts", "lactose", "soy", "egg", "none"]
PLATFORMS = ["Windows", "macOS", "Linux", "PS5", "Xbox", "Switch", "iOS", "Android"]
LICENSE_TYPES = ["perpetual", "annual", "monthly", "lifetime"]
UOM = ["each", "box", "pallet", "kg", "litre", "metre"]
GENDERS = ["women", "men", "unisex", "kids"]

# Brand pools per category root
BRANDS_BY_ROOT: dict[str, list[str]] = {
    "Electronics": ["Apple", "Samsung", "Sony", "LG", "Xiaomi", "Dell", "HP", "Lenovo", "Asus", "Bose", "Anker", "Garmin"],
    "Fashion": ["Zara", "Nike", "Adidas", "Gucci", "Levi's", "Uniqlo", "Patagonia", "Boss", "Veja"],
    "Home & Living": ["IKEA", "Bosch", "Philips", "De'Longhi", "Le Creuset", "Alessi", "Muji"],
    "Grocery": ["Lavazza", "Illy", "Barilla", "Mutti", "Antinori", "Parmareggio"],
    "Automotive": ["Bosch", "Brembo", "Michelin", "Pirelli", "Mann-Filter", "Castrol"],
    "Industrial & B2B": ["3M", "Honeywell", "DeWalt", "Festo", "Würth", "RS Pro"],
    "Digital Goods": ["Microsoft", "Adobe", "Valve", "Ubisoft", "JetBrains"],
    "Services": ["MangoCare", "FixIt", "InstallPro", "GuardPlus"],
}
DEFAULT_BRANDS = ["GenericCo", "ValueLine", "ProMax", "EcoChoice"]

# ---------------------------------------------------------------------------
# Listings (M:N merchant <-> catalog)
# ---------------------------------------------------------------------------

LISTING_CONDITIONS = ["new", "new", "new", "refurbished", "used_like_new", "used_good", "open_box"]
LISTING_STATUSES = ["live", "live", "live", "paused", "out_of_stock", "delisted"]

# ---------------------------------------------------------------------------
# Orders / payments / shipments / returns
# ---------------------------------------------------------------------------

# Order status changed taxonomy across schema versions (v1 vs v2) -> drift
ORDER_STATUSES_V2 = ["created", "paid", "partially_shipped", "shipped", "delivered",
                     "cancelled", "refunded", "partially_refunded", "disputed"]
ORDER_STATUS_WEIGHTS = [0.04, 0.06, 0.05, 0.08, 0.55, 0.07, 0.05, 0.06, 0.04]
# legacy v1 docs use a different vocabulary for the same concepts
ORDER_STATUS_LEGACY_MAP = {
    "created": "NEW", "paid": "PROCESSING", "shipped": "SENT",
    "delivered": "COMPLETE", "cancelled": "CANCELLED", "refunded": "REFUND",
}

CHANNELS = ["web", "ios_app", "android_app", "marketplace_api", "phone", "in_store"]
CHANNEL_WEIGHTS = [0.40, 0.22, 0.20, 0.10, 0.04, 0.04]

PAYMENT_METHODS = ["card", "paypal", "bank_transfer", "klarna", "apple_pay", "google_pay", "crypto", "gift_card"]
PAYMENT_METHOD_WEIGHTS = [0.42, 0.18, 0.10, 0.10, 0.08, 0.06, 0.03, 0.03]
PAYMENT_STATUSES = ["authorized", "captured", "captured", "captured", "failed", "voided", "refunded", "partially_refunded"]
CARD_NETWORKS = ["visa", "mastercard", "amex", "maestro"]
PSP_PROVIDERS = ["stripe", "adyen", "braintree", "mollie"]

CARRIERS = ["DHL", "FedEx", "UPS", "GLS", "BRT", "PostNL", "Correos", "DPD"]
CARRIER_WEIGHTS = [0.24, 0.16, 0.14, 0.14, 0.12, 0.08, 0.06, 0.06]
SHIPMENT_STATUSES = ["label_created", "in_transit", "out_for_delivery", "delivered", "exception", "returned_to_sender"]
SERVICE_LEVELS = ["standard", "express", "next_day", "economy"]

RETURN_REASONS = ["defective", "wrong_item", "not_as_described", "changed_mind", "too_small", "too_large", "late_delivery"]
RETURN_STATUSES = ["requested", "approved", "in_transit", "received", "refunded", "rejected"]
RETURN_RESOLUTIONS = ["refund", "replacement", "store_credit", "repair"]

# ---------------------------------------------------------------------------
# Promotions (polymorphic rule engine -> arrays of heterogeneous conditions)
# ---------------------------------------------------------------------------

PROMO_TYPES = ["percentage", "fixed_amount", "bogo", "free_shipping", "tiered", "bundle"]
PROMO_SCOPES = ["cart", "category", "merchant", "product", "first_order"]

# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------

SUB_PLANS = ["mango_plus", "mango_pro", "mango_business", "fresh_box", "coffee_club"]
SUB_INTERVALS = ["monthly", "monthly", "quarterly", "annual", "weekly"]
SUB_STATUSES = ["active", "active", "active", "trialing", "past_due", "paused", "cancelled"]

# ---------------------------------------------------------------------------
# Loyalty (points ledger)
# ---------------------------------------------------------------------------

LOYALTY_TIERS = ["green", "silver", "gold", "platinum"]
LOYALTY_TIER_WEIGHTS = [0.50, 0.28, 0.16, 0.06]
LOYALTY_TXN_TYPES = ["earn", "earn", "earn", "redeem", "expire", "adjust", "referral_bonus"]

# ---------------------------------------------------------------------------
# Support tickets (threaded messages, polymorphic channel)
# ---------------------------------------------------------------------------

TICKET_CHANNELS = ["email", "chat", "phone", "social", "in_app"]
TICKET_CATEGORIES = ["order_issue", "payment", "returns", "product_question", "account", "complaint", "other"]
TICKET_PRIORITIES = ["low", "medium", "high", "urgent"]
TICKET_STATUSES = ["open", "pending", "on_hold", "resolved", "closed"]
SENDER_ROLES = ["customer", "agent", "bot", "system"]

# ---------------------------------------------------------------------------
# Reviews (mixed rating scales: legacy 1-10, new 1-5)
# ---------------------------------------------------------------------------

REVIEW_SCALE_LEGACY = "ten"  # rating 1..10, field "score"
REVIEW_SCALE_NEW = "five"    # rating 1..5,  field "rating"

# ---------------------------------------------------------------------------
# Events (heterogeneous clickstream)
# ---------------------------------------------------------------------------

EVENT_TYPES = ["page_view", "search", "add_to_cart", "remove_from_cart", "checkout_start",
               "purchase", "wishlist_add", "review_submit", "support_open", "app_open"]
EVENT_WEIGHTS = [0.34, 0.18, 0.12, 0.05, 0.06, 0.07, 0.05, 0.03, 0.02, 0.08]
DEVICES = ["mobile", "desktop", "tablet", "smart_tv", "kiosk"]
OS_LIST = ["iOS", "Android", "Windows", "macOS", "Linux"]
REFERRERS = ["google.com", "instagram.com", "facebook.com", "tiktok.com", "direct", "email", "x.com", "affiliate"]
AB_VARIANTS = ["control", "variant_a", "variant_b"]

# ---------------------------------------------------------------------------
# Ledger (double-entry accounting)
# ---------------------------------------------------------------------------

LEDGER_ACCOUNTS = ["revenue", "cogs", "platform_fee", "payment_fee", "tax_payable",
                   "refunds", "shipping_revenue", "shipping_cost", "merchant_payable", "promotions"]

# ---------------------------------------------------------------------------
# Audit logs (polymorphic before/after diffs)
# ---------------------------------------------------------------------------

AUDIT_ENTITIES = ["order", "listing", "merchant", "user", "promotion", "payment", "catalog"]
AUDIT_ACTIONS = ["create", "update", "delete", "status_change", "price_change", "login", "permission_change"]
ACTOR_TYPES = ["user", "merchant", "admin", "system", "api_client"]

# ---------------------------------------------------------------------------
# Document counts per collection (scale = 1.0)  ~= 4M docs
# ---------------------------------------------------------------------------

BASE_COUNTS: dict[str, int] = {
    "currencies": 8,
    "fx_rates": 350,          # monthly-ish rates per currency over the window
    "categories": 250,        # actually derived from CATEGORY_TREE
    "merchants": 800,
    "warehouses": 30,
    "users": 60_000,
    "catalog": 12_000,
    "listings": 50_000,
    "inventory_snapshots": 80_000,
    "promotions": 2_000,
    "subscriptions": 15_000,
    "loyalty_accounts": 40_000,
    "loyalty_transactions": 200_000,
    "orders": 250_000,
    "payments": 260_000,
    "shipments": 200_000,
    "returns": 30_000,
    "reviews": 280_000,
    "support_tickets": 40_000,
    "ledger_entries": 400_000,
    "audit_logs": 150_000,
    "events": 1_000_000,
}
