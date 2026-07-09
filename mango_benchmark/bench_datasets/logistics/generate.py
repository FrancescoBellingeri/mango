"""Deterministic data generator for *bench_logistics*.

Usage:
    python -m mango_benchmark.bench_datasets.logistics.generate --uri mongodb://localhost:27017

Idempotent: drops and recreates the database. Seeded RNG; the sentinel
documents (§6.2 of DATASET_DESIGN.md) are planted from the shared constants
in ``schema.py`` and self-asserted at the end of the run.
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, GEOSPHERE, MongoClient

from mango_benchmark.bench_datasets.logistics.schema import (
    BOUNDARY_APR1,
    BOUNDARY_APR1_COUNT,
    BOUNDARY_MAR1,
    BOUNDARY_MAR1_COUNT,
    BULK_LYON_DWELL_MAX,
    DATE_END,
    DATE_START,
    DB_NAME,
    DEPOT_SPECS,
    ELEM_CITY,
    ELEM_CROSS_CODES,
    ELEM_DWELL_THRESHOLD,
    ELEM_SAME_CODES,
    FIRST_NAMES,
    LAST_NAMES,
    LICENSE_CLASSES,
    N_DRIVERS,
    N_SHIPMENTS,
    N_STOPS_EMPTY,
    N_STOPS_MISSING,
    N_VEHICLES,
    NUM_CANCELLED_WEIGHTS,
    NUM_DELIVERED_AVG,
    NUM_DELIVERED_WEIGHTS,
    NUM_VEHICLE_CODE,
    PRIORITIES,
    PRIORITY_WEIGHTS,
    SCALE_VEHICLE_CODE,
    SCALE_VEHICLE_SHIPMENTS,
    SEED,
    SHIPMENT_STATUSES,
    SHIPMENT_STATUS_WEIGHTS,
    SKUS,
    STOP_CITIES,
    SURCHARGE_KINDS,
    VEHICLE_TYPES,
    VEHICLE_TYPE_WEIGHTS,
)

rng = random.Random(SEED)


def _dt_between(a: datetime, b: datetime) -> datetime:
    delta = int((b - a).total_seconds())
    return a + timedelta(seconds=rng.randint(0, delta))


def _round2(x: float) -> float:
    return round(x, 2)


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


def make_depots() -> list[dict[str, Any]]:
    docs = []
    for code, city, country, lon, lat, cap, status, opened in DEPOT_SPECS:
        docs.append(
            {
                "_id": ObjectId(),
                "code": code,
                "name": f"{city} Hub",
                "city": city,
                "country": country,
                "location": {"type": "Point", "coordinates": [lon, lat]},
                "capacity_m3": cap,
                "status": status,
                "opened_on": opened,
            }
        )
    return docs


def make_vehicles(depots: list[dict]) -> list[dict[str, Any]]:
    docs = []
    for i in range(1, N_VEHICLES + 1):
        vtype = rng.choices(VEHICLE_TYPES, weights=VEHICLE_TYPE_WEIGHTS)[0]
        capacity = {
            "van": rng.randint(800, 1500),
            "truck": rng.randint(3000, 12000),
            "trailer": rng.randint(15000, 26000),
            "cargo_bike": rng.randint(80, 180),
        }[vtype]
        docs.append(
            {
                "_id": ObjectId(),
                "code": f"VH-{i:04d}",
                "type": vtype,
                "depot_id": rng.choice(depots)["_id"],
                "capacity_kg": capacity,
                "year": rng.randint(2012, 2025),
                "active": rng.random() < 0.88,
                "last_service_at": _dt_between(datetime(2024, 1, 1), DATE_END),
            }
        )
    return docs


def make_drivers(depots: list[dict]) -> list[dict[str, Any]]:
    docs = []
    for i in range(1, N_DRIVERS + 1):
        n_lic = rng.randint(1, 3)
        docs.append(
            {
                "_id": ObjectId(),
                "code": f"DRV-{i:04d}",
                "first_name": rng.choice(FIRST_NAMES),
                "last_name": rng.choice(LAST_NAMES),
                "depot_id": rng.choice(depots)["_id"],
                "license_classes": sorted(rng.sample(LICENSE_CLASSES, n_lic)),
                "hired_on": _dt_between(datetime(2015, 1, 1), datetime(2025, 6, 30)),
                "on_leave": rng.random() < 0.07,
            }
        )
    return docs


def _make_stops(dest_city: str, placed_at: datetime, n: int) -> list[dict[str, Any]]:
    stops = []
    t = placed_at
    for seq in range(1, n + 1):
        city = dest_city if seq == n else rng.choice(STOP_CITIES)
        t = t + timedelta(hours=rng.randint(4, 30))
        dwell_max = BULK_LYON_DWELL_MAX if city == ELEM_CITY else 360
        stops.append(
            {
                "seq": seq,
                "city": city,
                "eta": t,
                "dwell_min": rng.randint(10, dwell_max),
            }
        )
    return stops


def _make_items() -> list[dict[str, Any]]:
    return [
        {
            "sku": rng.choice(SKUS),
            "qty": rng.randint(1, 40),
            "unit_weight_kg": _round2(rng.uniform(0.2, 45.0)),
        }
        for _ in range(rng.randint(1, 5))
    ]


def _make_cost(weight_kg: float) -> dict[str, Any]:
    surcharges = [
        {"kind": kind, "amount_eur": _round2(rng.uniform(8, 220))}
        for kind in rng.sample(SURCHARGE_KINDS, rng.randint(0, 3))
    ]
    return {
        "base_eur": _round2(weight_kg * rng.uniform(0.08, 0.22) + 12),
        "fuel_eur": _round2(weight_kg * rng.uniform(0.01, 0.05)),
        "surcharges": surcharges,
    }


def _base_shipment(
    i: int, depots: list[dict], vehicles: list[dict], drivers: list[dict]
) -> dict[str, Any]:
    origin, dest = rng.sample(depots, 2)
    status = rng.choices(SHIPMENT_STATUSES, weights=SHIPMENT_STATUS_WEIGHTS)[0]
    placed_at = _dt_between(DATE_START, datetime(2025, 11, 30))
    weight = _round2(rng.uniform(3, 18000))
    doc: dict[str, Any] = {
        "_id": ObjectId(),
        "code": f"SHP-{i:06d}",
        "status": status,
        "priority": rng.choices(PRIORITIES, weights=PRIORITY_WEIGHTS)[0],
        "origin_depot_id": origin["_id"],
        "dest_depot_id": dest["_id"],
        "dest_city": dest["city"],
        "vehicle_id": rng.choice(vehicles)["_id"],
        "driver_id": rng.choice(drivers)["_id"],
        "weight_kg": weight,
        "volume_m3": _round2(rng.uniform(0.05, 90.0)),
        "placed_at": placed_at,
        "items": _make_items(),
        "cost": _make_cost(weight),
        "stops": _make_stops(dest["city"], placed_at, rng.randint(1, 5)),
    }
    if status == "delivered":
        doc["delivered_at"] = placed_at + timedelta(hours=rng.randint(12, 240))
    else:
        doc["delivered_at"] = None
    return doc


def make_shipments(
    depots: list[dict], vehicles: list[dict], drivers: list[dict]
) -> list[dict[str, Any]]:
    by_code = {v["code"]: v for v in vehicles}
    scale_vehicle = by_code[SCALE_VEHICLE_CODE]
    num_vehicle = by_code[NUM_VEHICLE_CODE]
    bulk_vehicles = [
        v for v in vehicles if v["code"] not in (SCALE_VEHICLE_CODE, NUM_VEHICLE_CODE)
    ]

    docs: list[dict[str, Any]] = []
    for i in range(1, N_SHIPMENTS + 1):
        doc = _base_shipment(i, depots, bulk_vehicles, drivers)
        docs.append(doc)

    # --- SENTINEL: EDGE-SCALE — exactly N shipments on VH-0007 -------------
    for doc in rng.sample(docs, SCALE_VEHICLE_SHIPMENTS):
        doc["vehicle_id"] = scale_vehicle["_id"]

    # --- SENTINEL: unwind empty/missing stops ------------------------------
    no_stop_docs = rng.sample(docs, N_STOPS_EMPTY + N_STOPS_MISSING)
    for doc in no_stop_docs[:N_STOPS_EMPTY]:
        doc["stops"] = []
    for doc in no_stop_docs[N_STOPS_EMPTY:]:
        del doc["stops"]

    # --- SENTINEL: boundary dates (placed_at exactly on window endpoints) --
    boundary_docs = rng.sample(
        [d for d in docs if "stops" in d and d["stops"]],
        BOUNDARY_MAR1_COUNT + BOUNDARY_APR1_COUNT,
    )
    for doc in boundary_docs[:BOUNDARY_MAR1_COUNT]:
        doc["placed_at"] = BOUNDARY_MAR1
    for doc in boundary_docs[BOUNDARY_MAR1_COUNT:]:
        doc["placed_at"] = BOUNDARY_APR1

    # --- SENTINEL: $elemMatch trap cohorts ---------------------------------
    # Same-element: one Lyon stop with dwell above the threshold.
    for j, code in enumerate(ELEM_SAME_CODES):
        doc = _base_shipment(N_SHIPMENTS + 1 + j, depots, bulk_vehicles, drivers)
        doc["code"] = code
        doc["stops"] = [
            {"seq": 1, "city": ELEM_CITY, "eta": doc["placed_at"] + timedelta(hours=8),
             "dwell_min": ELEM_DWELL_THRESHOLD + 20 + j},
            {"seq": 2, "city": doc["dest_city"], "eta": doc["placed_at"] + timedelta(hours=20),
             "dwell_min": rng.randint(10, 60)},
        ]
        docs.append(doc)
    # Cross-element: a low-dwell Lyon stop + a high-dwell stop elsewhere.
    for j, code in enumerate(ELEM_CROSS_CODES):
        doc = _base_shipment(N_SHIPMENTS + 100 + j, depots, bulk_vehicles, drivers)
        doc["code"] = code
        other_city = "Munich" if doc["dest_city"] == ELEM_CITY else doc["dest_city"]
        doc["stops"] = [
            {"seq": 1, "city": ELEM_CITY, "eta": doc["placed_at"] + timedelta(hours=8),
             "dwell_min": rng.randint(10, BULK_LYON_DWELL_MAX)},
            {"seq": 2, "city": other_city, "eta": doc["placed_at"] + timedelta(hours=20),
             "dwell_min": ELEM_DWELL_THRESHOLD + 40 + j},
        ]
        docs.append(doc)

    # --- SENTINEL: EDGE-NUM must-FAIL cohort on the reserved vehicle -------
    for j, weight in enumerate(NUM_DELIVERED_WEIGHTS):
        doc = _base_shipment(N_SHIPMENTS + 200 + j, depots, bulk_vehicles, drivers)
        doc["code"] = f"SHP-NM-D{j + 1:02d}"
        doc["vehicle_id"] = num_vehicle["_id"]
        doc["status"] = "delivered"
        doc["weight_kg"] = weight
        doc["delivered_at"] = doc["placed_at"] + timedelta(hours=48)
        docs.append(doc)
    for j, weight in enumerate(NUM_CANCELLED_WEIGHTS):
        doc = _base_shipment(N_SHIPMENTS + 300 + j, depots, bulk_vehicles, drivers)
        doc["code"] = f"SHP-NM-C{j + 1:02d}"
        doc["vehicle_id"] = num_vehicle["_id"]
        doc["status"] = "cancelled"
        doc["weight_kg"] = weight
        doc["delivered_at"] = None
        docs.append(doc)

    return docs


# ---------------------------------------------------------------------------
# Sentinel self-checks (§7.1 — constants and data must agree)
# ---------------------------------------------------------------------------


def assert_sentinels(db: Any) -> None:
    ship = db.shipments

    same = ship.count_documents(
        {"stops": {"$elemMatch": {"city": ELEM_CITY, "dwell_min": {"$gt": ELEM_DWELL_THRESHOLD}}}}
    )
    naive = ship.count_documents(
        {"stops.city": ELEM_CITY, "stops.dwell_min": {"$gt": ELEM_DWELL_THRESHOLD}}
    )
    # The CORRECT query's result is exact (only sentinel docs have a Lyon stop
    # above the threshold — bulk Lyon dwell is capped). The NAIVE query also
    # matches bulk shipments whose high-dwell stop is in another city; its
    # count is not a constant, only the gap matters (§6.2: differ by ≥2).
    assert same == len(ELEM_SAME_CODES), f"elemMatch same-element: {same}"
    assert naive >= same + len(ELEM_CROSS_CODES), f"naive: {naive}"

    assert ship.count_documents({"stops": []}) == N_STOPS_EMPTY
    assert ship.count_documents({"stops": {"$exists": False}}) == N_STOPS_MISSING

    assert ship.count_documents({"placed_at": BOUNDARY_MAR1}) == BOUNDARY_MAR1_COUNT
    assert ship.count_documents({"placed_at": BOUNDARY_APR1}) == BOUNDARY_APR1_COUNT

    scale_vehicle = db.vehicles.find_one({"code": SCALE_VEHICLE_CODE})
    n_scale = ship.count_documents({"vehicle_id": scale_vehicle["_id"]})
    assert n_scale == SCALE_VEHICLE_SHIPMENTS, f"scale cohort: {n_scale}"

    num_vehicle = db.vehicles.find_one({"code": NUM_VEHICLE_CODE})
    delivered = [
        d["weight_kg"]
        for d in ship.find({"vehicle_id": num_vehicle["_id"], "status": "delivered"})
    ]
    all_w = [d["weight_kg"] for d in ship.find({"vehicle_id": num_vehicle["_id"]})]
    assert sorted(delivered) == sorted(NUM_DELIVERED_WEIGHTS), delivered
    assert len(all_w) == len(NUM_DELIVERED_WEIGHTS) + len(NUM_CANCELLED_WEIGHTS)
    good_avg = sum(delivered) / len(delivered)
    combined = sum(all_w) / len(all_w)
    rel = abs(combined - good_avg) / good_avg
    assert 1e-6 < rel <= 5e-3, rel  # the must-FAIL zone (§6.2)
    assert round(combined, 3) != round(good_avg, 3), (combined, good_avg)
    assert abs(good_avg - NUM_DELIVERED_AVG) < 1e-9, good_avg

    caps = sorted((d["capacity_m3"] for d in db.depots.find()), reverse=True)
    assert caps[1] == caps[2] and caps[0] != caps[1] and caps[3] not in caps[1:3], (
        "depot capacity tie sentinel broken"
    )
    assert caps[4] != caps[5], "tie must not straddle the top-5 cut"

    print("sentinel self-checks OK")


# ---------------------------------------------------------------------------


def generate(uri: str) -> None:
    client: MongoClient = MongoClient(uri)
    client.drop_database(DB_NAME)
    db = client[DB_NAME]

    depots = make_depots()
    db.depots.insert_many(depots)
    db.depots.create_index([("location", GEOSPHERE)])
    db.depots.create_index([("code", ASCENDING)], unique=True)

    vehicles = make_vehicles(depots)
    db.vehicles.insert_many(vehicles)
    db.vehicles.create_index([("code", ASCENDING)], unique=True)

    drivers = make_drivers(depots)
    db.drivers.insert_many(drivers)

    shipments = make_shipments(depots, vehicles, drivers)
    for i in range(0, len(shipments), 5000):
        db.shipments.insert_many(shipments[i : i + 5000])
    db.shipments.create_index([("code", ASCENDING)], unique=True)
    db.shipments.create_index([("placed_at", ASCENDING)])

    for name in ("depots", "vehicles", "drivers", "shipments"):
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
