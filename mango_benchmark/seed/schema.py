"""Domain constants for the mango_ecommerce synthetic benchmark database."""

from __future__ import annotations

from datetime import datetime

DB_NAME = "mango_ecommerce"

DATE_START = datetime(2023, 1, 1)
DATE_END = datetime(2025, 12, 31, 23, 59, 59)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

COUNTRIES = [
    "Italy", "Italy", "Italy", "Italy",
    "Germany", "Germany", "Germany",
    "France", "France",
    "Spain", "Spain",
    "United Kingdom", "United Kingdom",
    "United States", "United States",
    "Netherlands", "Belgium", "Switzerland", "Austria", "Poland",
    "Portugal", "Sweden", "Norway", "Denmark", "Japan", "Canada",
]

CUSTOMER_TIERS = ["bronze", "silver", "gold", "platinum"]
TIER_WEIGHTS = [0.45, 0.30, 0.18, 0.07]

PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"]
PAYMENT_WEIGHTS = [0.40, 0.25, 0.20, 0.12, 0.03]

CARRIERS = ["DHL", "FedEx", "UPS", "GLS", "BRT", "PostNL", "Correos"]
CARRIER_WEIGHTS = [0.30, 0.20, 0.15, 0.15, 0.10, 0.05, 0.05]

ORDER_STATUSES = [
    "pending", "confirmed", "processing", "shipped", "delivered", "cancelled", "returned"
]
ORDER_STATUS_WEIGHTS = [0.03, 0.04, 0.05, 0.08, 0.70, 0.07, 0.03]

SHIPMENT_STATUSES = ["created", "picked_up", "in_transit", "out_for_delivery", "delivered", "failed"]

REVIEW_SENTIMENTS = [
    ("positive", 0.55),
    ("neutral", 0.25),
    ("negative", 0.20),
]

EVENT_TYPES = ["view", "search", "cart_add", "cart_remove", "purchase", "wishlist"]
EVENT_WEIGHTS = [0.45, 0.25, 0.12, 0.05, 0.08, 0.05]

DEVICES = ["mobile", "desktop", "tablet"]
REFERRERS = ["google.com", "instagram.com", "facebook.com", "direct", "email", "twitter.com"]

CURRENCIES = ["EUR"]  # single currency for simplicity

# ---------------------------------------------------------------------------
# Category tree (hierarchical, 3 levels, ~200 total)
# ---------------------------------------------------------------------------

CATEGORY_TREE: list[dict] = [
    {
        "name": "Electronics",
        "slug": "electronics",
        "children": [
            {"name": "Mobile", "slug": "mobile", "children": [
                {"name": "Smartphones", "slug": "smartphones"},
                {"name": "Tablets", "slug": "tablets"},
                {"name": "Phone Accessories", "slug": "phone-accessories"},
            ]},
            {"name": "Computers", "slug": "computers", "children": [
                {"name": "Laptops", "slug": "laptops"},
                {"name": "Desktops", "slug": "desktops"},
                {"name": "Components", "slug": "components"},
            ]},
            {"name": "TV & Audio", "slug": "tv-audio", "children": [
                {"name": "Televisions", "slug": "televisions"},
                {"name": "Speakers", "slug": "speakers"},
                {"name": "Headphones", "slug": "headphones"},
            ]},
            {"name": "Cameras", "slug": "cameras", "children": [
                {"name": "Digital Cameras", "slug": "digital-cameras"},
                {"name": "Action Cameras", "slug": "action-cameras"},
            ]},
            {"name": "Gaming", "slug": "gaming", "children": [
                {"name": "Consoles", "slug": "consoles"},
                {"name": "Games", "slug": "games"},
                {"name": "Gaming Accessories", "slug": "gaming-accessories"},
            ]},
        ],
    },
    {
        "name": "Clothing & Fashion",
        "slug": "clothing",
        "children": [
            {"name": "Men's", "slug": "mens", "children": [
                {"name": "Men's T-Shirts", "slug": "mens-tshirts"},
                {"name": "Men's Jeans", "slug": "mens-jeans"},
                {"name": "Men's Jackets", "slug": "mens-jackets"},
            ]},
            {"name": "Women's", "slug": "womens", "children": [
                {"name": "Women's Dresses", "slug": "womens-dresses"},
                {"name": "Women's Tops", "slug": "womens-tops"},
                {"name": "Women's Jeans", "slug": "womens-jeans"},
            ]},
            {"name": "Kids", "slug": "kids", "children": [
                {"name": "Boys Clothing", "slug": "boys-clothing"},
                {"name": "Girls Clothing", "slug": "girls-clothing"},
            ]},
            {"name": "Shoes", "slug": "shoes", "children": [
                {"name": "Sneakers", "slug": "sneakers"},
                {"name": "Boots", "slug": "boots"},
                {"name": "Sandals", "slug": "sandals"},
            ]},
            {"name": "Accessories", "slug": "accessories", "children": [
                {"name": "Bags", "slug": "bags"},
                {"name": "Watches", "slug": "watches"},
                {"name": "Belts", "slug": "belts"},
            ]},
        ],
    },
    {
        "name": "Books & Media",
        "slug": "books",
        "children": [
            {"name": "Fiction", "slug": "fiction", "children": [
                {"name": "Thriller", "slug": "thriller"},
                {"name": "Romance", "slug": "romance"},
                {"name": "Science Fiction", "slug": "sci-fi"},
            ]},
            {"name": "Non-Fiction", "slug": "non-fiction", "children": [
                {"name": "Biography", "slug": "biography"},
                {"name": "History", "slug": "history"},
                {"name": "Self-Help", "slug": "self-help"},
            ]},
            {"name": "Textbooks", "slug": "textbooks", "children": [
                {"name": "University", "slug": "university"},
                {"name": "School", "slug": "school"},
            ]},
        ],
    },
    {
        "name": "Home & Garden",
        "slug": "home",
        "children": [
            {"name": "Furniture", "slug": "furniture", "children": [
                {"name": "Sofas", "slug": "sofas"},
                {"name": "Tables", "slug": "tables"},
                {"name": "Chairs", "slug": "chairs"},
            ]},
            {"name": "Kitchen", "slug": "kitchen", "children": [
                {"name": "Cookware", "slug": "cookware"},
                {"name": "Appliances", "slug": "appliances"},
                {"name": "Utensils", "slug": "utensils"},
            ]},
            {"name": "Bedding", "slug": "bedding", "children": [
                {"name": "Sheets", "slug": "sheets"},
                {"name": "Pillows", "slug": "pillows"},
            ]},
            {"name": "Garden Tools", "slug": "garden-tools", "children": [
                {"name": "Planters", "slug": "planters"},
                {"name": "Tools", "slug": "garden-tools-set"},
            ]},
        ],
    },
    {
        "name": "Sports & Outdoors",
        "slug": "sports",
        "children": [
            {"name": "Fitness", "slug": "fitness", "children": [
                {"name": "Gym Equipment", "slug": "gym-equipment"},
                {"name": "Yoga", "slug": "yoga"},
            ]},
            {"name": "Outdoor", "slug": "outdoor", "children": [
                {"name": "Camping", "slug": "camping"},
                {"name": "Hiking", "slug": "hiking"},
            ]},
            {"name": "Cycling", "slug": "cycling", "children": [
                {"name": "Bikes", "slug": "bikes"},
                {"name": "Bike Accessories", "slug": "bike-accessories"},
            ]},
        ],
    },
    {
        "name": "Beauty & Health",
        "slug": "beauty",
        "children": [
            {"name": "Skincare", "slug": "skincare", "children": [
                {"name": "Moisturizers", "slug": "moisturizers"},
                {"name": "Serums", "slug": "serums"},
                {"name": "Sunscreen", "slug": "sunscreen"},
            ]},
            {"name": "Haircare", "slug": "haircare", "children": [
                {"name": "Shampoo", "slug": "shampoo"},
                {"name": "Conditioner", "slug": "conditioner"},
            ]},
            {"name": "Makeup", "slug": "makeup", "children": [
                {"name": "Foundation", "slug": "foundation"},
                {"name": "Lipstick", "slug": "lipstick"},
                {"name": "Eyeshadow", "slug": "eyeshadow"},
            ]},
        ],
    },
]

# ---------------------------------------------------------------------------
# Brand pools per root category
# ---------------------------------------------------------------------------

BRANDS_BY_CATEGORY: dict[str, list[str]] = {
    "Electronics": ["Apple", "Samsung", "Sony", "LG", "Xiaomi", "Huawei", "Dell", "HP", "Lenovo", "Asus", "Bose", "Logitech"],
    "Clothing & Fashion": ["Zara", "H&M", "Nike", "Adidas", "Gucci", "Prada", "Levi's", "Calvin Klein", "Tommy Hilfiger", "Boss"],
    "Books & Media": ["Penguin", "HarperCollins", "Mondadori", "Feltrinelli", "Random House", "Oxford"],
    "Home & Garden": ["IKEA", "Bosch", "Philips", "De'Longhi", "Tefal", "Whirlpool", "Alessi"],
    "Sports & Outdoors": ["Nike", "Adidas", "Decathlon", "Columbia", "The North Face", "Garmin", "Fitbit"],
    "Beauty & Health": ["L'Oréal", "Nivea", "Clinique", "MAC", "Estée Lauder", "The Ordinary", "CeraVe"],
}
DEFAULT_BRANDS = ["GenericBrand", "ProBrand", "EcoBrand", "PremiumCo", "ValueMart"]

# ---------------------------------------------------------------------------
# Polymorphic specs templates per root category
# ---------------------------------------------------------------------------

SPECS_TEMPLATES: dict[str, list[str]] = {
    "Electronics": ["display_inches", "battery_mah", "ram_gb", "storage_gb", "connectivity"],
    "Clothing & Fashion": ["size", "color", "material", "gender"],
    "Books & Media": ["author", "pages", "publisher", "language"],
    "Home & Garden": ["material", "dimensions", "weight_kg", "color"],
    "Sports & Outdoors": ["sport", "material", "size", "weight_kg"],
    "Beauty & Health": ["volume_ml", "skin_type", "is_vegan"],
}

SIZES_CLOTHING = ["XS", "S", "M", "L", "XL", "XXL"]
COLORS = ["Black", "White", "Red", "Blue", "Green", "Yellow", "Grey", "Navy", "Beige", "Pink"]
MATERIALS_CLOTHING = ["Cotton", "Polyester", "Wool", "Linen", "Silk", "Denim", "Leather"]
MATERIALS_HOME = ["Wood", "Metal", "Plastic", "Glass", "Ceramic", "Bamboo", "Stainless Steel"]
LANGUAGES = ["Italian", "English", "German", "French", "Spanish"]
GENDERS = ["Men", "Women", "Unisex", "Kids"]
SKIN_TYPES = ["Normal", "Oily", "Dry", "Combination", "Sensitive"]
SPORTS = ["Running", "Cycling", "Swimming", "Yoga", "Hiking", "Football", "Tennis", "Gym"]

# Document counts per collection (scale=1.0)
BASE_COUNTS: dict[str, int] = {
    "categories": 200,
    "suppliers": 300,
    "warehouses": 25,
    "products": 8_000,
    "customers": 50_000,
    "inventory": 40_000,
    "orders": 200_000,
    "shipments": 180_000,
    "reviews": 300_000,
    "events": 800_000,
}
