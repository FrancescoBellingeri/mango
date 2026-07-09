"""Deterministic data generator for *bench_workforce*.

Usage:
    python -m mango_benchmark.bench_datasets.workforce.generate --uri mongodb://localhost:27017

Idempotent (drop + recreate), seeded RNG, sentinel self-checks at the end.
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, MongoClient

from mango_benchmark.bench_datasets.workforce.schema import (
    APPOINTMENT_KINDS,
    AUDITOR_ACTIVE_AVG,
    AUDITOR_ACTIVE_SALARIES,
    AUDITOR_DEPARTMENT,
    AUDITOR_ROLE,
    AUDITOR_TERMINATED_SALARIES,
    BULK_SALARY_MAX,
    CONTRACT_TYPES,
    CONTRACT_TYPE_WEIGHTS,
    DB_NAME,
    DEPARTMENTS,
    DEPARTMENT_WEIGHTS,
    FIRST_NAMES,
    HIRED_APR1,
    HIRED_APR1_COUNT,
    HIRED_MAR1,
    HIRED_MAR1_COUNT,
    LAST_NAMES_BULK,
    LEAVE_KINDS,
    LEAVE_KIND_WEIGHTS,
    LEAVE_STATUSES,
    LEAVE_STATUS_WEIGHTS,
    N_APPOINTMENTS,
    N_EMPLOYEES,
    N_LEAVE,
    NYC_NINE_AM_COUNT,
    NYC_NINE_AM_UTC_HOUR,
    NYC_WEEK_END,
    NYC_WEEK_START,
    OFFICE_SPECS,
    ROLES_BY_DEPT,
    SALARY_MIN,
    SEED,
    SKILLS,
    SURNAME_GROSSI_COUNT,
    SURNAME_ROSSI_COUNT,
    SURNAME_ROSSINI_COUNT,
    TERMINATED_NULL_COUNT,
    TERMINATED_VALUE_COUNT,
    TOP_SALARIES,
    TOP_SALARY_CODES,
)

rng = random.Random(SEED)


def _dt_between(a: datetime, b: datetime) -> datetime:
    return a + timedelta(seconds=rng.randint(0, int((b - a).total_seconds())))


def make_offices() -> list[dict[str, Any]]:
    return [
        {
            "_id": ObjectId(),
            "code": code,
            "city": city,
            "country": country,
            "tz": tz,
            "opened_on": opened,
        }
        for code, city, country, tz, opened in OFFICE_SPECS
    ]


def make_employees(offices: list[dict]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    # Sentinel surnames with exact counts (bulk pool excludes all three).
    sentinel_surnames = (
        ["Rossi"] * SURNAME_ROSSI_COUNT
        + ["Grossi"] * SURNAME_GROSSI_COUNT
        + ["Rossini"] * SURNAME_ROSSINI_COUNT
    )

    for i in range(1, N_EMPLOYEES + 1):
        dept = rng.choices(DEPARTMENTS, weights=DEPARTMENT_WEIGHTS)[0]
        last = sentinel_surnames.pop() if sentinel_surnames else rng.choice(LAST_NAMES_BULK)
        first = rng.choice(FIRST_NAMES)
        docs.append(
            {
                "_id": ObjectId(),
                "employee_code": f"EMP-{i:05d}",
                "first_name": first,
                "last_name": last,
                "email": f"{first}.{last}".lower().replace(" ", "").replace("'", "") + "@benchcorp.example",
                "department": dept,
                "role": rng.choice(ROLES_BY_DEPT[dept]),
                "office_code": rng.choice(offices)["code"],
                "hired_on": _dt_between(datetime(2012, 1, 1), datetime(2025, 10, 31)),
                "salary_eur": rng.randrange(SALARY_MIN, BULK_SALARY_MAX, 500),
                "part_time_pct": rng.choice([100, 100, 100, 100, 100, 80, 60, 50]),
                "skills": sorted(rng.sample(SKILLS, rng.randint(1, 5))),
            }
        )
    rng.shuffle(docs)  # sentinel surnames must not cluster at the head

    # --- SENTINEL: top-salary executives (static ranking with ties) --------
    for code, salary in zip(TOP_SALARY_CODES, TOP_SALARIES):
        doc = rng.choice(docs)
        while doc["employee_code"].startswith("EMP-X"):
            doc = rng.choice(docs)
        doc["employee_code"] = code
        doc["salary_eur"] = salary
        doc["role"] = "executive"

    # --- SENTINEL: terminated_on value / null / missing cohorts ------------
    # (terminations only for hires old enough that hired+90d < window end)
    old_enough = [d for d in docs if d["hired_on"] < datetime(2025, 1, 1)]
    pool = rng.sample(old_enough, TERMINATED_VALUE_COUNT + TERMINATED_NULL_COUNT)
    for doc in pool[:TERMINATED_VALUE_COUNT]:
        doc["terminated_on"] = _dt_between(doc["hired_on"] + timedelta(days=90), datetime(2025, 11, 30))
    for doc in pool[TERMINATED_VALUE_COUNT:]:
        doc["terminated_on"] = None
    # everyone else: field absent.

    # --- SENTINEL: hire-date boundary docs ----------------------------------
    boundary = rng.sample([d for d in docs if "terminated_on" not in d],
                          HIRED_MAR1_COUNT + HIRED_APR1_COUNT)
    for doc in boundary[:HIRED_MAR1_COUNT]:
        doc["hired_on"] = HIRED_MAR1
    for doc in boundary[HIRED_MAR1_COUNT:]:
        doc["hired_on"] = HIRED_APR1

    # --- SENTINEL: EDGE-NUM auditor cohort (appended LAST so earlier -------
    # sentinel sampling can never clobber salaries/terminations) ------------
    def _auditor(i: int, salary: int) -> dict[str, Any]:
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES_BULK)
        return {
            "_id": ObjectId(),
            "employee_code": f"EMP-A{i:03d}",
            "first_name": first,
            "last_name": last,
            "email": f"{first}.{last}.aud{i}".lower().replace(" ", "").replace("'", "") + "@benchcorp.example",
            "department": AUDITOR_DEPARTMENT,
            "role": AUDITOR_ROLE,
            "office_code": rng.choice(offices)["code"],
            "hired_on": _dt_between(datetime(2015, 1, 1), datetime(2023, 12, 31)),
            "salary_eur": salary,
            "part_time_pct": 100,
            "skills": sorted(rng.sample(SKILLS, 3)),
        }

    for i, sal in enumerate(AUDITOR_ACTIVE_SALARIES, 1):
        docs.append(_auditor(i, sal))
    for j, sal in enumerate(AUDITOR_TERMINATED_SALARIES, 1):
        doc = _auditor(100 + j, sal)
        doc["terminated_on"] = _dt_between(datetime(2024, 6, 1), datetime(2025, 6, 30))
        docs.append(doc)

    return docs


def make_contracts(employees: list[dict]) -> list[dict[str, Any]]:
    docs = []
    for emp in employees:
        n = rng.choice([1, 1, 1, 2])
        start = emp["hired_on"]
        for k in range(n):
            ctype = rng.choices(CONTRACT_TYPES, weights=CONTRACT_TYPE_WEIGHTS)[0]
            end = None
            if ctype != "permanent" or k < n - 1:
                end = start + timedelta(days=rng.randint(180, 900))
            docs.append(
                {
                    "_id": ObjectId(),
                    "employee_id": emp["_id"],
                    "type": ctype,
                    "start_on": start,
                    "end_on": end,
                    "salary_eur": emp["salary_eur"] - rng.randrange(0, 6000, 500) * k,
                    "signed_at": start - timedelta(days=rng.randint(7, 45)),
                }
            )
            start = (end or start) + timedelta(days=1)
    return docs


def make_leave(employees: list[dict]) -> list[dict[str, Any]]:
    docs = []
    for _ in range(N_LEAVE):
        emp = rng.choice(employees)
        frm = _dt_between(datetime(2024, 1, 1), datetime(2025, 11, 15))
        days = rng.choice([0.5, 1, 1, 2, 2, 3, 5, 5, 10, 15])
        docs.append(
            {
                "_id": ObjectId(),
                "employee_id": emp["_id"],
                "kind": rng.choices(LEAVE_KINDS, weights=LEAVE_KIND_WEIGHTS)[0],
                "from_on": frm,
                "to_on": frm + timedelta(days=int(days) or 1),
                "days": days,
                "status": rng.choices(LEAVE_STATUSES, weights=LEAVE_STATUS_WEIGHTS)[0],
                "submitted_at": frm - timedelta(days=rng.randint(3, 60)),
            }
        )
    return docs


def make_appointments(employees: list[dict], offices: list[dict]) -> list[dict[str, Any]]:
    nyc = next(o for o in offices if o["code"] == "OFF-NYC")
    docs = []
    for _ in range(N_APPOINTMENTS):
        emp = rng.choice(employees)
        office = rng.choice(offices)
        at = _dt_between(datetime(2024, 6, 1), datetime(2025, 11, 30)).replace(
            minute=rng.choice([0, 15, 30, 45]), second=0, microsecond=0
        )
        # keep the sentinel NYC week clean of accidental 14:00-UTC collisions
        if (
            office["code"] == "OFF-NYC"
            and NYC_WEEK_START <= at < NYC_WEEK_END
            and at.hour == NYC_NINE_AM_UTC_HOUR
        ):
            at = at.replace(hour=16)
        docs.append(
            {
                "_id": ObjectId(),
                "employee_id": emp["_id"],
                "kind": rng.choice(APPOINTMENT_KINDS),
                "scheduled_at": at,  # UTC
                "duration_min": rng.choice([30, 45, 60, 90]),
                "office_code": office["code"],
            }
        )

    # --- SENTINEL: 09:00 America/New_York == 14:00 UTC (mid-January) -------
    for j in range(NYC_NINE_AM_COUNT):
        day = NYC_WEEK_START + timedelta(days=j % 5)
        docs.append(
            {
                "_id": ObjectId(),
                "employee_id": rng.choice(employees)["_id"],
                "kind": "interview",
                "scheduled_at": day.replace(hour=NYC_NINE_AM_UTC_HOUR),
                "duration_min": 60,
                "office_code": nyc["code"],
            }
        )
    return docs


# ---------------------------------------------------------------------------


def assert_sentinels(db: Any) -> None:
    emp = db.employees
    assert emp.count_documents({"last_name": "Rossi"}) == SURNAME_ROSSI_COUNT
    assert emp.count_documents({"last_name": {"$regex": "Rossi"}}) == (
        SURNAME_ROSSI_COUNT + SURNAME_ROSSINI_COUNT
    )
    assert emp.count_documents({"last_name": {"$regex": "rossi", "$options": "i"}}) == (
        SURNAME_ROSSI_COUNT + SURNAME_ROSSINI_COUNT + SURNAME_GROSSI_COUNT
    )

    n_value = emp.count_documents({"terminated_on": {"$type": "date"}})
    n_null_or_missing = emp.count_documents({"terminated_on": None})
    n_null_only = emp.count_documents({"terminated_on": {"$type": "null"}})
    n_total = emp.count_documents({})
    assert n_value == TERMINATED_VALUE_COUNT + len(AUDITOR_TERMINATED_SALARIES), n_value
    assert n_null_only == TERMINATED_NULL_COUNT, n_null_only
    assert n_null_or_missing == n_total - n_value

    # EDGE-NUM auditor cohort (§6.2 must-FAIL window)
    active = [d["salary_eur"] for d in emp.find({"role": AUDITOR_ROLE, "terminated_on": None})]
    everyone = [d["salary_eur"] for d in emp.find({"role": AUDITOR_ROLE})]
    assert sorted(active) == sorted(AUDITOR_ACTIVE_SALARIES), active
    assert len(everyone) == len(AUDITOR_ACTIVE_SALARIES) + len(AUDITOR_TERMINATED_SALARIES)
    good_avg = sum(active) / len(active)
    combined = sum(everyone) / len(everyone)
    rel = abs(combined - good_avg) / good_avg
    assert 1e-6 < rel <= 5e-3, rel
    assert abs(good_avg - AUDITOR_ACTIVE_AVG) < 1e-9, good_avg

    assert emp.count_documents({"hired_on": HIRED_MAR1}) == HIRED_MAR1_COUNT
    assert emp.count_documents({"hired_on": HIRED_APR1}) == HIRED_APR1_COUNT

    sal = [d["salary_eur"] for d in emp.find().sort("salary_eur", -1).limit(11)]
    assert sal[:10] == TOP_SALARIES, sal
    assert sal[10] < TOP_SALARIES[9], "bulk salary must stay below the top-10"

    n_nine = db.appointments.count_documents(
        {
            "office_code": "OFF-NYC",
            "scheduled_at": {"$gte": NYC_WEEK_START, "$lt": NYC_WEEK_END},
        }
    )
    n_nine_local = db.appointments.count_documents(
        {
            "office_code": "OFF-NYC",
            "scheduled_at": {"$gte": NYC_WEEK_START, "$lt": NYC_WEEK_END},
            "$expr": {"$eq": [{"$hour": "$scheduled_at"}, NYC_NINE_AM_UTC_HOUR]},
        }
    )
    assert n_nine_local == NYC_NINE_AM_COUNT, (n_nine, n_nine_local)

    print("sentinel self-checks OK")


def generate(uri: str) -> None:
    client: MongoClient = MongoClient(uri)
    client.drop_database(DB_NAME)
    db = client[DB_NAME]

    offices = make_offices()
    db.offices.insert_many(offices)

    employees = make_employees(offices)
    db.employees.insert_many(employees)
    db.employees.create_index([("employee_code", ASCENDING)], unique=True)

    db.contracts.insert_many(make_contracts(employees))
    db.leave_requests.insert_many(make_leave(employees))
    db.appointments.insert_many(make_appointments(employees, offices))

    for name in ("offices", "employees", "contracts", "leave_requests", "appointments"):
        print(f"  {name}: {db[name].estimated_document_count()}")
    assert_sentinels(db)
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate {DB_NAME}")
    parser.add_argument("--uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    args = parser.parse_args()
    generate(args.uri)


if __name__ == "__main__":
    main()
