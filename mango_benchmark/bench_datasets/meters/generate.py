"""Deterministic data generator for *bench_meters*.

Usage:
    python -m mango_benchmark.bench_datasets.meters.generate --uri mongodb://localhost:27017

~260k readings (hourly, 124 devices, 90 days) plus sentinel devices. Seeded,
idempotent, sentinel self-checks at the end.
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from bson import Decimal128, ObjectId
from pymongo import ASCENDING, MongoClient

from mango_benchmark.bench_datasets.meters.schema import (
    ALERT_CODES,
    ALERT_SEVERITIES,
    ALERT_SEVERITY_WEIGHTS,
    CAL_DEVICE,
    CAL_ESTIMATED_KWH,
    CAL_GOOD_KWH,
    CAL_HOUR,
    DB_NAME,
    DEVICE_MODELS,
    EMPTY_ALERT_REGION,
    FIRMWARES,
    GAP_DEVICE,
    GAP_END,
    GAP_START,
    HF_CADENCE_MIN,
    HF_DAY,
    HF_DEVICE,
    HF_READINGS,
    INVOICE_MONTHS,
    LOGGED_AT_EPOCH_FRACTION,
    LOGGED_AT_STRING_FRACTION,
    N_ALERTS,
    N_BULK_DEVICES,
    QUALITIES,
    QUALITY_WEIGHTS,
    READINGS_END,
    READINGS_START,
    REPR_DEVICE,
    REPR_HOUR,
    REPR_KWH,
    SEED,
    SENTINEL_DEVICES,
    SITE_SPECS,
)

rng = random.Random(SEED)


def make_sites() -> list[dict[str, Any]]:
    return [
        {
            "_id": ObjectId(),
            "code": code,
            "name": name,
            "region": region,
            "country": country,
            "commissioned_on": datetime(2020 + i % 5, (i % 12) + 1, (i % 27) + 1),
        }
        for i, (code, name, region, country) in enumerate(SITE_SPECS)
    ]


def make_devices(sites: list[dict]) -> list[dict[str, Any]]:
    docs = []
    for i in range(1, N_BULK_DEVICES + 1):
        docs.append(
            {
                "_id": ObjectId(),
                "serial": f"MTR-{i:04d}",
                "model": rng.choice(DEVICE_MODELS),
                "site_code": rng.choice(sites)["code"],
                "installed_on": datetime(2023 + i % 3, (i % 12) + 1, (i % 27) + 1),
                "firmware": rng.choice(FIRMWARES),
                "active": rng.random() < 0.93,
                "calibration": {
                    "offset": round(rng.uniform(-0.05, 0.05), 4),
                    "scale": round(rng.uniform(0.98, 1.02), 4),
                },
            }
        )
    for serial in SENTINEL_DEVICES:
        docs.append(
            {
                "_id": ObjectId(),
                "serial": serial,
                "model": "EM-350",
                "site_code": rng.choice(sites)["code"],
                "installed_on": datetime(2024, 2, 1),
                "firmware": "2.1.3",
                "active": True,
                "calibration": {"offset": 0.0, "scale": 1.0},
            }
        )
    return docs


def _messy_logged_at(ts: datetime) -> Any:
    r = rng.random()
    logged = ts + timedelta(seconds=rng.randint(30, 240))
    if r < LOGGED_AT_EPOCH_FRACTION:
        return int(logged.timestamp() * 1000)
    if r < LOGGED_AT_EPOCH_FRACTION + LOGGED_AT_STRING_FRACTION:
        return logged.isoformat()
    return logged


def _reading(dev: dict, ts: datetime, kwh: float, quality: str) -> dict[str, Any]:
    return {
        "device_id": dev["_id"],
        "serial": dev["serial"],
        "ts": ts,
        "kwh": kwh,
        "voltage": round(rng.uniform(224.0, 236.0), 1),
        "temp_c": round(rng.uniform(8.0, 42.0), 1),
        "quality": quality,
        "logged_at": _messy_logged_at(ts),
    }


def make_readings(devices: list[dict]) -> list[dict[str, Any]]:
    by_serial = {d["serial"]: d for d in devices}
    bulk = [d for d in devices if d["serial"] not in SENTINEL_DEVICES]

    docs: list[dict[str, Any]] = []
    n_hours = int((READINGS_END - READINGS_START).total_seconds() // 3600)

    for dev in bulk:
        base = rng.uniform(0.8, 22.0)
        for h in range(n_hours):
            ts = READINGS_START + timedelta(hours=h)
            kwh = round(max(0.0, rng.gauss(base, base * 0.18)), 3)
            docs.append(
                _reading(dev, ts, kwh, rng.choices(QUALITIES, weights=QUALITY_WEIGHTS)[0])
            )

    # --- SENTINEL: gap device — hourly, but nothing in [GAP_START, GAP_END)
    gap_dev = by_serial[GAP_DEVICE]
    for h in range(n_hours):
        ts = READINGS_START + timedelta(hours=h)
        if GAP_START <= ts < GAP_END:
            continue
        docs.append(_reading(gap_dev, ts, round(rng.uniform(2, 9), 3), "good"))

    # --- SENTINEL: EDGE-NUM calibration hour (exact kwh values) ------------
    cal_dev = by_serial[CAL_DEVICE]
    for j, kwh in enumerate(CAL_GOOD_KWH):
        docs.append(_reading(cal_dev, CAL_HOUR + timedelta(minutes=3 * j), kwh, "good"))
    for j, kwh in enumerate(CAL_ESTIMATED_KWH):
        docs.append(
            _reading(cal_dev, CAL_HOUR + timedelta(minutes=40 + 3 * j), kwh, "estimated")
        )

    # --- SENTINEL: repr-noise device ---------------------------------------
    repr_dev = by_serial[REPR_DEVICE]
    for j, kwh in enumerate(REPR_KWH):
        docs.append(_reading(repr_dev, REPR_HOUR + timedelta(minutes=10 * j), kwh, "good"))

    # --- SENTINEL: high-frequency day (exactly HF_READINGS rows) -----------
    hf_dev = by_serial[HF_DEVICE]
    for j in range(HF_READINGS):
        ts = HF_DAY + timedelta(minutes=HF_CADENCE_MIN * j)
        docs.append(_reading(hf_dev, ts, round(rng.uniform(1, 4), 3), "good"))

    return docs


def make_alerts(devices: list[dict], sites: list[dict]) -> list[dict[str, Any]]:
    region_of = {s["code"]: s["region"] for s in sites}
    docs = []
    for _ in range(N_ALERTS):
        dev = rng.choice(devices)
        severity = rng.choices(ALERT_SEVERITIES, weights=ALERT_SEVERITY_WEIGHTS)[0]
        # EDGE-EMPTY sentinel: islands region never gets critical alerts.
        if severity == "critical" and region_of[dev["site_code"]] == EMPTY_ALERT_REGION:
            severity = "warning"
        ts = READINGS_START + timedelta(seconds=rng.randint(0, 90 * 86400 - 1))
        resolved = rng.random() < 0.7
        docs.append(
            {
                "device_id": dev["_id"],
                "serial": dev["serial"],
                "site_code": dev["site_code"],
                "ts": ts,
                "severity": severity,
                "code": rng.choice(ALERT_CODES),
                "resolved": resolved,
                "resolved_at": ts + timedelta(hours=rng.randint(1, 96)) if resolved else None,
            }
        )
    return docs


def make_invoices(sites: list[dict]) -> list[dict[str, Any]]:
    docs = []
    for site in sites:
        for month in INVOICE_MONTHS:
            amount = Decimal(rng.randrange(180_000, 2_400_000)) / Decimal(100)
            docs.append(
                {
                    "_id": ObjectId(),
                    "site_code": site["code"],
                    "month": month,
                    "kwh_total": round(rng.uniform(4_000, 90_000), 1),
                    "amount_eur": Decimal128(amount),  # Decimal128 lever (§4.5)
                }
            )
    return docs


# ---------------------------------------------------------------------------


def assert_sentinels(db: Any) -> None:
    rd = db.readings

    cal = list(rd.find({"serial": CAL_DEVICE}))
    good = [r["kwh"] for r in cal if r["quality"] == "good"]
    assert sorted(good) == sorted(CAL_GOOD_KWH)
    good_avg = sum(good) / len(good)
    all_kwh = [r["kwh"] for r in cal]
    combined = sum(all_kwh) / len(all_kwh)
    rel = abs(combined - good_avg) / good_avg
    assert 1e-6 < rel <= 5e-3, rel  # the must-FAIL zone (§6.2)
    # guard against the round-to-gold-precision escape hatch (§4.6 path b):
    assert round(combined, 3) != round(good_avg, 3), (combined, good_avg)

    assert rd.count_documents({"serial": GAP_DEVICE, "ts": {"$gte": GAP_START, "$lt": GAP_END}}) == 0
    assert rd.count_documents({"serial": GAP_DEVICE}) > 0

    n_hf = rd.count_documents(
        {"serial": HF_DEVICE, "ts": {"$gte": HF_DAY, "$lt": HF_DAY + timedelta(days=1)}}
    )
    assert n_hf == HF_READINGS, n_hf

    islands_sites = [s["code"] for s in db.sites.find({"region": EMPTY_ALERT_REGION})]
    assert islands_sites, "no islands sites"
    assert db.alerts.count_documents(
        {"site_code": {"$in": islands_sites}, "severity": "critical"}
    ) == 0
    assert db.alerts.count_documents({"severity": "critical"}) > 0

    from bson import Decimal128 as D128

    inv = db.invoices.find_one()
    assert isinstance(inv["amount_eur"], D128)

    print("sentinel self-checks OK")


def generate(uri: str) -> None:
    client: MongoClient = MongoClient(uri)
    client.drop_database(DB_NAME)
    db = client[DB_NAME]

    sites = make_sites()
    db.sites.insert_many(sites)

    devices = make_devices(sites)
    db.devices.insert_many(devices)
    db.devices.create_index([("serial", ASCENDING)], unique=True)

    readings = make_readings(devices)
    for i in range(0, len(readings), 20_000):
        db.readings.insert_many(readings[i : i + 20_000])
    db.readings.create_index([("serial", ASCENDING), ("ts", ASCENDING)])

    db.alerts.insert_many(make_alerts(devices, sites))
    db.invoices.insert_many(make_invoices(sites))

    for name in ("sites", "devices", "readings", "alerts", "invoices"):
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
