"""Domain constants for the *bench_workforce* benchmark database.

HR domain (DATASET_DESIGN.md §1): dates everywhere, engineered
**null-vs-missing** cohorts, explicit-timezone offices, and string collision
traps for regex questions. Mostly clean; the only labeled messiness lever is
the null/missing split on ``terminated_on``.

All sentinels are named constants shared with ``generate.py`` and the
question modules.
"""

from __future__ import annotations

from datetime import datetime

DB_NAME = "bench_workforce"
SEED = 31017

# ---------------------------------------------------------------------------
# Offices — static (explicit IANA tz is a CAP-TIME lever).
# (code, city, country, tz, opened_on)
# ---------------------------------------------------------------------------

OFFICE_SPECS: list[tuple[str, str, str, str, datetime]] = [
    ("OFF-MIL", "Milan", "Italy", "Europe/Rome", datetime(2012, 5, 2)),
    ("OFF-ROM", "Rome", "Italy", "Europe/Rome", datetime(2014, 9, 15)),
    ("OFF-PAR", "Paris", "France", "Europe/Paris", datetime(2013, 3, 20)),
    ("OFF-BER", "Berlin", "Germany", "Europe/Berlin", datetime(2015, 11, 9)),
    ("OFF-MAD", "Madrid", "Spain", "Europe/Madrid", datetime(2016, 6, 27)),
    ("OFF-AMS", "Amsterdam", "Netherlands", "Europe/Amsterdam", datetime(2017, 2, 13)),
    ("OFF-LON", "London", "United Kingdom", "Europe/London", datetime(2011, 8, 1)),
    ("OFF-DUB", "Dublin", "Ireland", "Europe/Dublin", datetime(2018, 4, 23)),
    ("OFF-WAW", "Warsaw", "Poland", "Europe/Warsaw", datetime(2019, 10, 7)),
    ("OFF-LIS", "Lisbon", "Portugal", "Europe/Lisbon", datetime(2020, 1, 20)),
    ("OFF-NYC", "New York", "United States", "America/New_York", datetime(2016, 3, 14)),
    ("OFF-AUS", "Austin", "United States", "America/Chicago", datetime(2021, 7, 6)),
    ("OFF-TOK", "Tokyo", "Japan", "Asia/Tokyo", datetime(2019, 5, 27)),
    ("OFF-SYD", "Sydney", "Australia", "Australia/Sydney", datetime(2020, 9, 3)),
]
OFFICE_CODES = [o[0] for o in OFFICE_SPECS]

# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------

N_EMPLOYEES = 2_600

DEPARTMENTS = [
    "engineering", "sales", "marketing", "finance", "hr",
    "operations", "legal", "support", "product", "data",
]
DEPARTMENT_WEIGHTS = [0.26, 0.15, 0.09, 0.08, 0.05, 0.14, 0.03, 0.10, 0.06, 0.04]

ROLES_BY_DEPT = {
    "engineering": ["software engineer", "senior engineer", "staff engineer", "engineering manager", "sre"],
    "sales": ["account executive", "sdr", "sales manager", "solutions engineer"],
    "marketing": ["content manager", "growth manager", "brand designer"],
    "finance": ["accountant", "controller", "fp&a analyst"],
    "hr": ["recruiter", "hr business partner", "people ops"],
    "operations": ["ops specialist", "ops manager", "logistics coordinator"],
    "legal": ["counsel", "paralegal"],
    "support": ["support agent", "support lead"],
    "product": ["product manager", "product designer"],
    "data": ["data analyst", "data engineer", "data scientist"],
}

SKILLS = [
    "python", "sql", "excel", "salesforce", "figma", "react", "kubernetes",
    "negotiation", "copywriting", "tableau", "airflow", "gdpr", "sap",
]

FIRST_NAMES = [
    "Alessandro", "Beatrice", "Chiara", "Davide", "Elisa", "Federico",
    "Giulia", "Hannah", "Ethan", "Noah", "Olivia", "Liam", "Emma", "Lucas",
    "Mia", "Leon", "Camille", "Louis", "Ana", "Miguel", "Sofia", "Jakub",
    "Zofia", "Yuki", "Haruto", "Charlotte", "Jack",
]

# --- SENTINEL: regex traps (CAP-FILTER, §6.2) ------------------------------
# "surname contains Rossi" vs "surname is Rossi": Grossi and Rossini are
# substring/superstring colliders with exact planted counts. The bulk surname
# pool excludes all three so counts stay exact.
SURNAME_ROSSI_COUNT = 18
SURNAME_GROSSI_COUNT = 7
SURNAME_ROSSINI_COUNT = 5

LAST_NAMES_BULK = [
    "Ferrari", "Esposito", "Ricci", "Marino", "Greco", "Bruno", "Gallo",
    "Conti", "Mancini", "Costa", "Giordano", "Rizzo", "Lombardi", "Moretti",
    "Dubois", "Bernard", "Petit", "Durand", "Leroy", "Moreau", "Fournier",
    "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker",
    "Garcia", "Martinez", "Lopez", "Sanchez", "Perez", "Gomez",
    "Smith", "Jones", "Williams", "Brown", "Davies", "Evans", "Wilson",
    "Kowalski", "Nowak", "Wojcik", "Kaminski",
    "Tanaka", "Suzuki", "Takahashi", "Watanabe",
    "de Jong", "van den Berg", "Bakker", "Visser",
    "O'Brien", "Murphy", "Kelly", "Ryan",
]

# --- SENTINEL: null vs missing on terminated_on (CAP-FILTER, §6.2) ---------
# Three cohorts among *inactive-looking* employees:
#   value    — terminated_on: <datetime>   (genuinely terminated)
#   null     — terminated_on: null         (imported records, unknown)
#   missing  — no terminated_on field      (active employees)
# {terminated_on: null} matches null AND missing (2 cohorts); only
# {$type: "date"}/{$ne: null} isolate the terminated. Counts are exact.
TERMINATED_VALUE_COUNT = 240
TERMINATED_NULL_COUNT = 37
# missing = everyone else (N_EMPLOYEES - value - null)

# --- SENTINEL: hire-date boundaries (CAP-TIME, §6.2) -----------------------
# Docs exactly on the endpoints of March 2024 so inclusivity flips change
# results: hired_on == Mar-1 00:00 (2 docs) and == Apr-1 00:00 (3 docs).
HIRED_MAR1 = datetime(2024, 3, 1, 0, 0, 0)
HIRED_APR1 = datetime(2024, 4, 1, 0, 0, 0)
HIRED_MAR1_COUNT = 2
HIRED_APR1_COUNT = 3

# --- SENTINEL: salary ranking ties (CAP-SORT, §6.2) ------------------------
# Top-10 salaries are static. Ranks 2-3 tie at 188_000 (inside a top-5
# window); ranks 5 and 6 differ (no tie across the top-5 cut). The bulk
# salary generator caps at BULK_SALARY_MAX so these stay the global top.
TOP_SALARIES = [195_000, 188_000, 188_000, 176_000, 171_000,
                168_000, 164_000, 159_000, 155_000, 151_000]
TOP_SALARY_CODES = [f"EMP-X{i:02d}" for i in range(1, 11)]
BULK_SALARY_MAX = 145_000
SALARY_MIN = 28_000

# ---------------------------------------------------------------------------
# Contracts / leave / appointments
# ---------------------------------------------------------------------------

CONTRACT_TYPES = ["permanent", "fixed_term", "contractor"]
CONTRACT_TYPE_WEIGHTS = [0.72, 0.18, 0.10]

N_LEAVE = 12_000
LEAVE_KINDS = ["vacation", "sick", "parental", "unpaid"]
LEAVE_KIND_WEIGHTS = [0.58, 0.28, 0.09, 0.05]
LEAVE_STATUSES = ["pending", "approved", "rejected"]
LEAVE_STATUS_WEIGHTS = [0.12, 0.78, 0.10]

N_APPOINTMENTS = 8_000
APPOINTMENT_KINDS = ["interview", "performance_review", "training", "onboarding"]

# --- SENTINEL: EDGE-NUM must-FAIL (mirrors bench_meters CAL_DEVICE) --------
# 13 employees with the reserved role below (bulk ROLES_BY_DEPT never emits
# it): 10 active (no terminated_on field), salaries summing 607_500 ->
# avg exactly 60_750; 3 terminated at 60_802 each so that forgetting to
# exclude terminated staff gives
#   combined avg = 789_906 / 13 = 60_762  -> rel err ~1.98e-4 vs 60_750
# inside the (1e-6, 5e-3] must-FAIL zone. Integer salaries are fine here:
# the 12-euro absolute gap survives any gold-precision rounding.
AUDITOR_ROLE = "compliance auditor"
AUDITOR_DEPARTMENT = "legal"
AUDITOR_ACTIVE_SALARIES = [58_500, 61_000, 59_500, 63_000, 60_500,
                           62_000, 57_500, 64_000, 61_500, 60_000]  # sum 607_500
AUDITOR_TERMINATED_SALARIES = [60_802, 60_802, 60_802]              # sum 182_406
AUDITOR_ACTIVE_AVG = 60_750.0

# --- SENTINEL: explicit-tz window (CAP-TIME) -------------------------------
# Appointments at the New York office scheduled 2025-01-13..17 (EST, UTC-5 —
# mid-January avoids any DST edge) with local start 09:00 == 14:00 UTC.
# Exactly NYC_NINE_AM_COUNT of that week's NYC appointments are at 09:00
# local; the generator never places other NYC appointments at 14:00 UTC that
# week.
NYC_WEEK_START = datetime(2025, 1, 13)
NYC_WEEK_END = datetime(2025, 1, 18)  # exclusive
NYC_NINE_AM_UTC_HOUR = 14
NYC_NINE_AM_COUNT = 6
