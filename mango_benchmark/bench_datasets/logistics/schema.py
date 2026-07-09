"""Domain constants for the *bench_logistics* benchmark database.

Deliberately **clean** schema (DATASET_DESIGN.md §1): no drift, no boolean
dialects, no mixed encodings. Difficulty comes from structure — nested
``stops[]`` / ``items[]`` arrays, 3-level ``cost.surcharges[].amount_eur``
nesting, ObjectId references for `$lookup`/ObjectId-group-key cases, and geo
points (2dsphere) for the STRETCH geo cases.

Every sentinel the questions rely on is a **named constant here**, imported
by both ``generate.py`` and the question modules — ties, `$elemMatch` traps,
boundary dates and the >100-row cohort are shared constants, not
coincidences a regeneration could break (§7.1).
"""

from __future__ import annotations

from datetime import datetime

DB_NAME = "bench_logistics"
SEED = 74921

DATE_START = datetime(2024, 1, 1)
DATE_END = datetime(2025, 12, 31, 23, 59, 59)

# ---------------------------------------------------------------------------
# Depots — fully static so ranking questions have hand-controlled order.
# (code, city, country, lon, lat, capacity_m3, status, opened_on)
#
# SENTINEL — capacity ranking ties (CAP-SORT §6.2):
#   rank 1: DEP-14 (9800) unique
#   ranks 2-3: DEP-03 / DEP-21 tied at 8200   <- tie INSIDE a top-5 window
#   rank 4: DEP-08 (7900), rank 5: DEP-01 (7500), rank 6: DEP-19 (7100)
#   -> no tie across a limit-5 cut (gold determinism, §6.1)
# ---------------------------------------------------------------------------

DEPOT_SPECS: list[tuple[str, str, str, float, float, float, str, datetime]] = [
    ("DEP-01", "Milan", "Italy", 9.1900, 45.4640, 7500.0, "active", datetime(2019, 3, 12)),
    ("DEP-02", "Turin", "Italy", 7.6869, 45.0703, 4200.0, "active", datetime(2020, 6, 1)),
    ("DEP-03", "Lyon", "France", 4.8357, 45.7640, 8200.0, "active", datetime(2018, 9, 20)),
    ("DEP-04", "Paris", "France", 2.3522, 48.8566, 6900.0, "active", datetime(2017, 1, 15)),
    ("DEP-05", "Marseille", "France", 5.3698, 43.2965, 3800.0, "maintenance", datetime(2021, 4, 8)),
    ("DEP-06", "Munich", "Germany", 11.5820, 48.1351, 6400.0, "active", datetime(2018, 11, 2)),
    ("DEP-07", "Berlin", "Germany", 13.4050, 52.5200, 5900.0, "active", datetime(2019, 7, 23)),
    ("DEP-08", "Hamburg", "Germany", 9.9937, 53.5511, 7900.0, "active", datetime(2016, 5, 30)),
    ("DEP-09", "Frankfurt", "Germany", 8.6821, 50.1109, 5100.0, "active", datetime(2020, 2, 14)),
    ("DEP-10", "Madrid", "Spain", -3.7038, 40.4168, 6100.0, "active", datetime(2019, 10, 5)),
    ("DEP-11", "Barcelona", "Spain", 2.1734, 41.3851, 5600.0, "active", datetime(2020, 8, 19)),
    ("DEP-12", "Valencia", "Spain", -0.3763, 39.4699, 2900.0, "closed", datetime(2015, 12, 1)),
    ("DEP-13", "Amsterdam", "Netherlands", 4.9041, 52.3676, 4800.0, "active", datetime(2021, 1, 11)),
    ("DEP-14", "Rotterdam", "Netherlands", 4.4777, 51.9244, 9800.0, "active", datetime(2016, 3, 3)),
    ("DEP-15", "Antwerp", "Belgium", 4.4025, 51.2194, 5300.0, "active", datetime(2018, 6, 27)),
    ("DEP-16", "Vienna", "Austria", 16.3738, 48.2082, 4500.0, "active", datetime(2019, 5, 16)),
    ("DEP-17", "Zurich", "Switzerland", 8.5417, 47.3769, 3600.0, "active", datetime(2020, 9, 9)),
    ("DEP-18", "Geneva", "Switzerland", 6.1432, 46.2044, 2700.0, "maintenance", datetime(2021, 7, 4)),
    ("DEP-19", "Warsaw", "Poland", 21.0122, 52.2297, 7100.0, "active", datetime(2017, 8, 22)),
    ("DEP-20", "Krakow", "Poland", 19.9450, 50.0647, 4100.0, "active", datetime(2019, 12, 13)),
    ("DEP-21", "Prague", "Czechia", 14.4378, 50.0755, 8200.0, "active", datetime(2018, 2, 6)),
    ("DEP-22", "Lisbon", "Portugal", -9.1393, 38.7223, 3300.0, "active", datetime(2020, 11, 25)),
    ("DEP-23", "Porto", "Portugal", -8.6291, 41.1579, 2400.0, "active", datetime(2021, 10, 7)),
    ("DEP-24", "Stockholm", "Sweden", 18.0686, 59.3293, 4700.0, "active", datetime(2019, 4, 2)),
    ("DEP-25", "Gothenburg", "Sweden", 11.9746, 57.7089, 3500.0, "active", datetime(2020, 5, 21)),
    ("DEP-26", "Copenhagen", "Denmark", 12.5683, 55.6761, 5000.0, "active", datetime(2018, 10, 17)),
    ("DEP-27", "Oslo", "Norway", 10.7522, 59.9139, 3100.0, "active", datetime(2021, 2, 28)),
    ("DEP-28", "Helsinki", "Finland", 24.9384, 60.1699, 2800.0, "active", datetime(2021, 6, 15)),
    ("DEP-29", "Dublin", "Ireland", -6.2603, 53.3498, 3900.0, "active", datetime(2019, 9, 30)),
    ("DEP-30", "London", "United Kingdom", -0.1276, 51.5074, 6800.0, "active", datetime(2016, 7, 12)),
    ("DEP-31", "Manchester", "United Kingdom", -2.2426, 53.4808, 4400.0, "active", datetime(2018, 4, 25)),
    ("DEP-32", "Birmingham", "United Kingdom", -1.8904, 52.4862, 3700.0, "closed", datetime(2015, 8, 8)),
    ("DEP-33", "Rome", "Italy", 12.4964, 41.9028, 5700.0, "active", datetime(2017, 11, 19)),
    ("DEP-34", "Naples", "Italy", 14.2681, 40.8518, 3200.0, "active", datetime(2020, 3, 26)),
    ("DEP-35", "Bologna", "Italy", 11.3426, 44.4949, 5200.0, "active", datetime(2018, 12, 10)),
    ("DEP-36", "Budapest", "Hungary", 19.0402, 47.4979, 4600.0, "active", datetime(2019, 6, 18)),
]

DEPOT_CODES = [d[0] for d in DEPOT_SPECS]

# Sentinel facts derivable from DEPOT_SPECS (asserted by generate.py):
TIE_CAPACITY = 8200.0           # DEP-03 / DEP-21
TOP5_CAPACITY_CODES = ["DEP-14", "DEP-03", "DEP-21", "DEP-08", "DEP-01"]

# Geo reference point for STRETCH $near questions ("closest depot to Milan
# city center"): Duomo di Milano.
GEO_REF_MILAN = (9.1919, 45.4642)

# ---------------------------------------------------------------------------
# Vehicles / drivers
# ---------------------------------------------------------------------------

N_VEHICLES = 320
VEHICLE_TYPES = ["van", "truck", "trailer", "cargo_bike"]
VEHICLE_TYPE_WEIGHTS = [0.42, 0.34, 0.14, 0.10]

N_DRIVERS = 400
LICENSE_CLASSES = ["B", "C", "C1", "CE", "D1"]

FIRST_NAMES = [
    "Luca", "Marco", "Anna", "Julia", "Pierre", "Sofia", "Jan", "Marta",
    "Tomasz", "Ines", "Sven", "Elena", "David", "Clara", "Hugo", "Nadia",
    "Piotr", "Laura", "Felix", "Irene", "Oscar", "Petra", "Milan", "Greta",
]
LAST_NAMES = [
    "Bianchi", "Ferrari", "Weber", "Muller", "Dubois", "Lefevre", "Garcia",
    "Torres", "Jansen", "Visser", "Kowalski", "Nowak", "Novak", "Svensson",
    "Nielsen", "Andersen", "Silva", "Costa", "Murphy", "Walsh", "Smith",
    "Taylor", "Horvath", "Kiss",
]

# ---------------------------------------------------------------------------
# Shipments
# ---------------------------------------------------------------------------

N_SHIPMENTS = 24_000

SHIPMENT_STATUSES = ["created", "in_transit", "delivered", "cancelled", "lost"]
SHIPMENT_STATUS_WEIGHTS = [0.10, 0.16, 0.62, 0.10, 0.02]

PRIORITIES = ["standard", "express", "critical"]
PRIORITY_WEIGHTS = [0.70, 0.24, 0.06]

SURCHARGE_KINDS = ["toll", "customs", "cold_chain", "oversize", "weekend"]

STOP_CITIES = [
    "Milan", "Turin", "Lyon", "Paris", "Munich", "Berlin", "Hamburg",
    "Madrid", "Barcelona", "Amsterdam", "Rotterdam", "Antwerp", "Vienna",
    "Zurich", "Warsaw", "Prague", "Lisbon", "Stockholm", "Copenhagen",
    "London", "Rome", "Bologna", "Budapest", "Dublin",
]

SKUS = [f"SKU-{i:04d}" for i in range(1, 401)]

# --- SENTINEL: $elemMatch trap (CAP-NEST, §6.2) ----------------------------
# "a stop in Lyon with a dwell above 180 minutes":
#   correct  {stops: {$elemMatch: {city: "Lyon", dwell_min: {$gt: 180}}}}
#   naive    {"stops.city": "Lyon", "stops.dwell_min": {$gt: 180}}
# SAME-element cohort satisfies both; CROSS-element cohort (a Lyon stop with
# low dwell + a different-city stop with high dwell) satisfies only the naive
# query. The CORRECT count is exact (= len(ELEM_SAME_CODES)); the naive count
# is not a constant — bulk shipments cross-match too, which only widens the
# naive-vs-correct gap the §6.2 assertion needs.
ELEM_SAME_CODES = [f"SHP-EM-S{i:02d}" for i in range(1, 8)]     # 7 docs
ELEM_CROSS_CODES = [f"SHP-EM-X{i:02d}" for i in range(1, 12)]   # 11 docs
ELEM_DWELL_THRESHOLD = 180
ELEM_CITY = "Lyon"
# Bulk shipments never use Lyon stops with dwell > threshold (generator caps
# random dwell at 120 for Lyon stops) so the sentinel counts are exact.
BULK_LYON_DWELL_MAX = 120

# --- SENTINEL: $unwind empty/missing arrays (CAP-MULTI, §6.2) --------------
N_STOPS_EMPTY = 60      # stops: []
N_STOPS_MISSING = 45    # no stops field at all

# --- SENTINEL: boundary dates (CAP-TIME, §6.2) -----------------------------
# Shipments placed exactly at window endpoints of March 2025. A doc at
# 2025-04-01T00:00:00 exists so `$lt Apr-1` vs `$lte Apr-1` differ; docs at
# 2025-03-01T00:00:00 exist so `$gte Mar-1` vs `$gt Mar-1` differ.
BOUNDARY_MAR1_COUNT = 2
BOUNDARY_APR1_COUNT = 3
BOUNDARY_MAR1 = datetime(2025, 3, 1, 0, 0, 0)
BOUNDARY_APR1 = datetime(2025, 4, 1, 0, 0, 0)

# --- SENTINEL: >100-row cohort (EDGE-SCALE) --------------------------------
SCALE_VEHICLE_CODE = "VH-0007"
SCALE_VEHICLE_SHIPMENTS = 130   # exactly; bulk assignment skips VH-0007

# --- SENTINEL: EDGE-NUM must-FAIL (mirrors bench_meters CAL_DEVICE) --------
# Vehicle VH-0100 (reserved: bulk assignment skips it) carries exactly 10
# DELIVERED shipments (weights below, avg 503.007 — 3 decimals so the
# round-to-gold-precision path of DESIGN.md §4.6b cannot mask a near-miss)
# and 4 CANCELLED shipments tuned so that forgetting the status filter gives
#   combined avg = 7043.506 / 14 ≈ 503.1076  -> rel err ~2.0e-4 vs 503.007
# inside the (1e-6, 5e-3] must-FAIL zone the old 0.5% tolerance would have
# silently blessed.
NUM_VEHICLE_CODE = "VH-0100"
NUM_DELIVERED_WEIGHTS = [480.5, 512.25, 495.75, 530.0, 505.5,
                         498.25, 520.75, 489.0, 515.32, 482.75]  # sum 5030.07
NUM_CANCELLED_WEIGHTS = [503.359, 503.359, 503.359, 503.359]     # sum 2013.436
NUM_DELIVERED_AVG = 503.007
