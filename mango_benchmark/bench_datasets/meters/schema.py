"""Domain constants for the *bench_meters* benchmark database.

IoT time-series (DATASET_DESIGN.md §1): floats and Decimal128, high-volume
readings (EDGE-SCALE), engineered numeric-tolerance sentinels (EDGE-NUM),
empty windows by construction (EDGE-EMPTY), and the suite's single
mixed-encoding lever (``readings.logged_at``: ISODate / epoch-millis /
ISO-string — ``ts`` is always clean).

EDGE-NUM sentinel math (device MTR-CAL-0001, hour 2025-06-01T10):
    12 "good" readings, kwh sum = 60.096 -> avg 5.008     (the gold)
     4 "estimated" readings at 5.0120064 -> combined avg ~5.0090016
    Plausible-wrong query (forgetting the quality filter) is off by
    rel ~2e-4 — inside (1e-6, 5e-3]: the zone the old 0.5% tolerance would
    have silently blessed and the locked 1e-6 must FAIL (§2.2, §6.2).

    Why the gold avg must NOT be a low-precision number like 5.0: the locked
    scorer has a round-to-gold-precision escape hatch (DESIGN.md §4.6 path b)
    for hand-rounded golds — a gold of exactly 5.0 (one decimal) would let
    5.001 round back to 5.0 and PASS. Found live by validate.py's tol_fail
    gate on the first build. With a 3-decimal gold (5.008) the wrong value
    (≈5.009) no longer survives the rounding path.
"""

from __future__ import annotations

from datetime import datetime

DB_NAME = "bench_meters"
SEED = 90211

# ---------------------------------------------------------------------------
# Sites / devices
# ---------------------------------------------------------------------------

REGIONS = ["north", "south", "east", "west", "islands"]

# (code, name, region, country) — static, 24 sites.
SITE_SPECS: list[tuple[str, str, str, str]] = [
    ("SITE-01", "Brera Plant", "north", "Italy"),
    ("SITE-02", "Navigli Works", "north", "Italy"),
    ("SITE-03", "Dora Facility", "north", "Italy"),
    ("SITE-04", "Lambrate Yard", "north", "Italy"),
    ("SITE-05", "Testaccio Depot", "south", "Italy"),
    ("SITE-06", "Vomero Station", "south", "Italy"),
    ("SITE-07", "Libertà Hub", "south", "Italy"),
    ("SITE-08", "Kalsa Point", "south", "Italy"),
    ("SITE-09", "San Donato Grid", "east", "Italy"),
    ("SITE-10", "Murazzi Node", "east", "Italy"),
    ("SITE-11", "Arcella Field", "east", "Italy"),
    ("SITE-12", "Borgo Panigale Cell", "east", "Italy"),
    ("SITE-13", "Oltrarno Ring", "west", "Italy"),
    ("SITE-14", "Sampierdarena Dock", "west", "Italy"),
    ("SITE-15", "Chiaia Terrace", "west", "Italy"),
    ("SITE-16", "Prati Junction", "west", "Italy"),
    ("SITE-17", "Cagliari Shore", "islands", "Italy"),
    ("SITE-18", "Palermo Cross", "islands", "Italy"),
    ("SITE-19", "Messina Strait", "islands", "Italy"),
    ("SITE-20", "Olbia Cove", "islands", "Italy"),
    ("SITE-21", "Zagreb Annex", "east", "Croatia"),
    ("SITE-22", "Ljubljana Spur", "east", "Slovenia"),
    ("SITE-23", "Nice Corniche", "west", "France"),
    ("SITE-24", "Lugano Slope", "north", "Switzerland"),
]
SITE_CODES = [s[0] for s in SITE_SPECS]

N_BULK_DEVICES = 120
DEVICE_MODELS = ["EM-100", "EM-200", "EM-350", "HydroX", "ThermoPro"]
FIRMWARES = ["1.4.2", "1.5.0", "2.0.1", "2.1.3", "3.0.0"]

# ---------------------------------------------------------------------------
# Readings — bulk cadence hourly over the window below.
# ---------------------------------------------------------------------------

READINGS_START = datetime(2025, 4, 1)
READINGS_END = datetime(2025, 6, 30)  # exclusive; 90 days
QUALITIES = ["good", "estimated", "suspect"]
QUALITY_WEIGHTS = [0.86, 0.10, 0.04]

# Fraction of bulk readings whose *logged_at* uses a messy encoding
# (documented lever): epoch-millis int / ISO string; ``ts`` stays ISODate.
LOGGED_AT_EPOCH_FRACTION = 0.10
LOGGED_AT_STRING_FRACTION = 0.05

# --- SENTINEL: EDGE-NUM calibration device ---------------------------------
CAL_DEVICE = "MTR-CAL-0001"
CAL_HOUR = datetime(2025, 6, 1, 10, 0, 0)
CAL_GOOD_KWH = [4.831, 5.214, 4.902, 5.113, 5.007, 4.995,
                4.723, 5.301, 4.956, 5.049, 4.983, 5.022]  # sum 60.096
CAL_ESTIMATED_KWH = [5.0120064, 5.0120064, 5.0120064, 5.0120064]
CAL_GOOD_AVG = 5.008              # 60.096 / 12 (3-decimal gold, see above)
CAL_COMBINED_AVG = 5.0090016      # 80.1440256 / 16 -> rel err ~2e-4 vs gold

# --- SENTINEL: EDGE-NUM must-PASS repr-noise device ------------------------
# Two mathematically equivalent computations ($avg vs $sum/$count) whose
# float results differ only at ~1e-16: must PASS under 1e-6.
REPR_DEVICE = "MTR-CAL-0002"
REPR_HOUR = datetime(2025, 6, 1, 11, 0, 0)
REPR_KWH = [0.1, 0.2, 0.3]

# --- SENTINEL: EDGE-EMPTY gap window ---------------------------------------
GAP_DEVICE = "MTR-GAP-0001"
GAP_START = datetime(2025, 5, 12)
GAP_END = datetime(2025, 5, 19)  # exclusive: zero readings in [start, end)

# 'islands' region has ZERO critical alerts by construction (EDGE-EMPTY):
# the alert generator never emits severity=critical for islands sites.
EMPTY_ALERT_REGION = "islands"

# --- SENTINEL: EDGE-SCALE high-frequency day -------------------------------
HF_DEVICE = "MTR-HF-0001"
HF_DAY = datetime(2025, 6, 10)
HF_CADENCE_MIN = 5
HF_READINGS = 288  # 24h * 12/h

SENTINEL_DEVICES = [CAL_DEVICE, REPR_DEVICE, GAP_DEVICE, HF_DEVICE]

# ---------------------------------------------------------------------------
# Alerts / invoices
# ---------------------------------------------------------------------------

N_ALERTS = 5_000
ALERT_SEVERITIES = ["info", "warning", "critical"]
ALERT_SEVERITY_WEIGHTS = [0.55, 0.33, 0.12]
ALERT_CODES = ["OVERVOLT", "UNDERVOLT", "COMM_LOSS", "TAMPER", "OVERTEMP", "CLOCK_DRIFT"]

# invoices.amount_eur is Decimal128 (type-normalization lever, §4.5 of
# DESIGN.md). One invoice per site per month over the window.
INVOICE_MONTHS = [datetime(2025, 4, 1), datetime(2025, 5, 1), datetime(2025, 6, 1)]
