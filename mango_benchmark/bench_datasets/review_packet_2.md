# INDEPENDENT AUDIT — ROUND 2 (escalation): CAP-NEST + CAP-TIME at 100%

## Why this round exists

In round 1, a stratified 20% sample of these two categories was audited and
each produced one confirmed gold defect (an over-specified sort tie-break the
NL never stated; a time-window whose correctness rested on planted-data
properties rather than query semantics). The benchmark's review protocol
mandates: >5% sample failure -> the WHOLE category is re-reviewed. This
packet is the remaining 32 cases of CAP-NEST and CAP-TIME.

## Your role and focus

Same adversarial-auditor role as round 1: you audit another AI's authored
benchmark cases; approve is not the default; every verdict carries a
specific reason; the author's notes are claims to verify, not evidence.

**Single criterion for every case in this packet** — does the gold MQL answer
the NL question exactly as a careful human reads it? Hunt specifically for
the two defect classes round 1 caught:

1. **Over-specification** — the gold encodes constraints the NL never states
   (extra sort keys, extra filters, extra projected fields, a limit the NL
   doesn't imply).
2. **Latent data-dependence** — the gold is only correct because of how the
   synthetic data happens to be shaped (boundary encodings, absent edge
   values, minute/second blindness in date-part matching, $size vs missing
   fields), rather than because the query semantics match the question.

Also flag the inverse (under-specification: NL states a constraint the gold
lacks) and boundary-inclusivity mismatches ("between X and Y", "in March",
"after 2020" vs $gte/$gt).

## Verdict semantics and output format

Identical to round 1: `approve` / `rewrite` (say exactly what and propose the
fix) / `reject`. Reply with ONLY:

```json
{
  "verdicts": { "<CASE-ID>": {"verdict": "...", "notes": "<specific reason>"} },
  "systemic_observations": ["..."]
}
```

All 32 case IDs must appear.

## Schema and data-window reference

(Identical to round 1; repeated for self-containment.)

### bench_logistics
- `depots`: code, name, city, country, location (GeoJSON Point), capacity_m3, status (active|maintenance|closed), opened_on
- `vehicles`: code, type (van|truck|trailer|cargo_bike), depot_id (ObjectId), capacity_kg, year, active, last_service_at
- `drivers`: code, first_name, last_name, depot_id, license_classes (string array), hired_on, on_leave
- `shipments`: code, status (created|in_transit|delivered|cancelled|lost), priority (standard|express|critical), origin_depot_id, dest_depot_id, dest_city, vehicle_id, driver_id, weight_kg, volume_m3, placed_at, delivered_at (datetime|null), items [{sku, qty, unit_weight_kg}], cost {base_eur, fuel_eur, surcharges [{kind, amount_eur}]}, stops [{seq, city, eta, dwell_min}] — some docs have stops: [] (60 docs) and some have NO stops field (45 docs)
- placed_at spans 2024-01-01 .. 2025-11-30, plus sentinel docs exactly at 2025-03-01 00:00 and 2025-04-01 00:00.

### bench_workforce
- `offices`: code, city, country, tz, opened_on
- `employees`: employee_code, first_name, last_name, email, department (10-value enum), role, office_code, hired_on, salary_eur, part_time_pct, skills (string array), terminated_on (datetime | explicit null | field absent)
- `contracts`: employee_id, type, start_on, end_on (datetime|null), salary_eur, signed_at
- `leave_requests`: employee_id, kind, from_on, to_on, days (float), status, submitted_at
- `appointments`: employee_id, kind, scheduled_at (UTC), duration_min, office_code
- Hires 2012-01-01 .. 2025-10-31; sentinel hires exactly at 2024-03-01 00:00 (2 docs) and 2024-04-01 00:00 (3 docs). Appointment minutes are 0/15/30/45.

### bench_meters
- `sites`: code, name, region (north|south|east|west|islands), country, commissioned_on
- `devices`: serial, model, site_code, installed_on, firmware, active, calibration {offset, scale}
- `readings`: device_id, serial, ts (clean), kwh, voltage, temp_c, quality (good|estimated|suspect), logged_at (mixed encodings by design)
- `alerts`: device_id, serial, site_code, ts, severity, code, resolved, resolved_at
- `invoices`: site_code, month, kwh_total, amount_eur (Decimal128)
- Readings 2025-04-01 .. 2025-06-30. Device MTR-GAP-0001 has a planted gap 2025-05-12 .. 2025-05-19 (exclusive).

Scoring: multiset comparison unless the gold declares a sort (then tie-runs
on the gold's sort keys); float tolerance 1e-6 relative.

---


# SECTION: CAP-NEST (remaining cases)

### LOG-NEST-001  `CAP-NEST/elemmatch`  difficulty=hard

**NL question:** "How many shipments have a stop in Lyon where the dwell time at that same stop exceeded 180 minutes?"

**Author note (verify, do not trust):** naive implicit-array-match rewrite matches across elements (documented wrong_mql)

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "shipments",
  "filter": {
    "stops": {
      "$elemMatch": {
        "city": "Lyon",
        "dwell_min": {
          "$gt": 180
        }
      }
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `7`

### LOG-NEST-002  `CAP-NEST/array_size`  difficulty=easy

**NL question:** "How many shipments have exactly 3 stops recorded?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "shipments",
  "filter": {
    "stops": {
      "$size": 3
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `4778`

### LOG-NEST-003  `CAP-NEST/dot_notation`  difficulty=easy

**NL question:** "How many shipments have a fuel cost above 50 euros?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "shipments",
  "filter": {
    "cost.fuel_eur": {
      "$gt": 50
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `21354`

### LOG-NEST-004  `CAP-NEST/elemmatch`  difficulty=hard

**NL question:** "How many shipments contain an item line for SKU-0001 with a quantity greater than 20?"

**Author note (verify, do not trust):** naive dot-notation matches across different item lines in a multi-item basket

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "shipments",
  "filter": {
    "items": {
      "$elemMatch": {
        "sku": "SKU-0001",
        "qty": {
          "$gt": 20
        }
      }
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `75`

### LOG-NEST-005  `CAP-NEST/elemmatch`  difficulty=hard

**NL question:** "How many shipments have a stop in Rome with a dwell time under 30 minutes at that same stop?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "shipments",
  "filter": {
    "stops": {
      "$elemMatch": {
        "city": "Rome",
        "dwell_min": {
          "$lt": 30
        }
      }
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `186`

### LOG-NEST-006  `CAP-NEST/dot_notation`  difficulty=easy

**NL question:** "For shipments delivered to Lisbon, return the shipment code and the base cost."

**Gold MQL:**
```json
{
  "operation": "find",
  "collection": "shipments",
  "filter": {
    "status": "delivered",
    "dest_city": "Lisbon"
  },
  "pipeline": null,
  "projection": {
    "_id": 0,
    "code": 1,
    "cost.base_eur": 1
  },
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: 415 rows; first rows: `[{"code": "SHP-000054", "cost": {"base_eur": 1040.2}}, {"code": "SHP-000068", "cost": {"base_eur": 2471.46}}, {"code": "SHP-000093", "cost": {"base_eur": 3431.55}}]`

### LOG-NEST-007  `CAP-NEST/array_size`  difficulty=medium

**NL question:** "How many shipments recorded zero stops (an empty list, not a missing field)?"

**Author note (verify, do not trust):** isolates the empty-array cohort (60) from the field-missing cohort (45)

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "shipments",
  "filter": {
    "stops": {
      "$size": 0
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `60`

### LOG-NEST-009  `CAP-NEST/array_size`  difficulty=hard

**NL question:** "List the shipment code and stop count for shipments with more than 4 stops."

**Author note (verify, do not trust):** $ifNull guards $size against the field-missing cohort

**Gold MQL:**
```json
{
  "operation": "aggregate",
  "collection": "shipments",
  "filter": null,
  "pipeline": [
    {
      "$addFields": {
        "stop_count": {
          "$size": {
            "$ifNull": [
              "$stops",
              []
            ]
          }
        }
      }
    },
    {
      "$match": {
        "stop_count": {
          "$gt": 4
        }
      }
    },
    {
      "$project": {
        "_id": 0,
        "code": 1,
        "stop_count": 1
      }
    }
  ],
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: 4799 rows; first rows: `[{"code": "SHP-000003", "stop_count": 5}, {"code": "SHP-000010", "stop_count": 5}, {"code": "SHP-000012", "stop_count": 5}]`

### MET-NEST-001  `CAP-NEST/dot_notation`  difficulty=medium

**NL question:** "List the serials of devices whose calibration offset is greater than 0.03."

**Gold MQL:**
```json
{
  "operation": "find",
  "collection": "devices",
  "filter": {
    "calibration.offset": {
      "$gt": 0.03
    }
  },
  "pipeline": null,
  "projection": {
    "_id": 0,
    "serial": 1
  },
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: 38 rows; first rows: `[{"serial": "MTR-0002"}, {"serial": "MTR-0015"}, {"serial": "MTR-0016"}]`

### MET-NEST-002  `CAP-NEST/dot_notation`  difficulty=easy

**NL question:** "How many devices have a calibration scale between 0.99 and 1.01, inclusive?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "devices",
  "filter": {
    "calibration.scale": {
      "$gte": 0.99,
      "$lte": 1.01
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `54`

### MET-NEST-003  `CAP-NEST/dot_notation`  difficulty=easy

**NL question:** "For active devices, return the serial and the calibration offset."

**Gold MQL:**
```json
{
  "operation": "find",
  "collection": "devices",
  "filter": {
    "active": true
  },
  "pipeline": null,
  "projection": {
    "_id": 0,
    "serial": 1,
    "calibration.offset": 1
  },
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: 119 rows; first rows: `[{"serial": "MTR-0001", "calibration": {"offset": 0.0191}}, {"serial": "MTR-0002", "calibration": {"offset": 0.0429}}, {"serial": "MTR-0003", "calibration": {"offset": -0.01}}]`

### MET-NEST-004  `CAP-NEST/dot_notation_negation`  difficulty=medium

**NL question:** "How many devices have a calibration offset that is NOT between -0.01 and 0.01?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "devices",
  "filter": {
    "$or": [
      {
        "calibration.offset": {
          "$lt": -0.01
        }
      },
      {
        "calibration.offset": {
          "$gt": 0.01
        }
      }
    ]
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `101`

### WF-NEST-001  `CAP-NEST/array_size`  difficulty=easy

**NL question:** "How many employees have exactly 3 skills listed?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "employees",
  "filter": {
    "skills": {
      "$size": 3
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `540`

### WF-NEST-002  `CAP-NEST/all_vs_in`  difficulty=hard

**NL question:** "How many employees have BOTH python and sql among their skills?"

**Author note (verify, do not trust):** naive $in matches employees with EITHER skill, a much larger set than the $all requirement

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "employees",
  "filter": {
    "skills": {
      "$all": [
        "python",
        "sql"
      ]
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `128`

### WF-NEST-004  `CAP-NEST/array_size_expr`  difficulty=medium

**NL question:** "How many employees have more than 4 skills listed?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "employees",
  "filter": {
    "$expr": {
      "$gt": [
        {
          "$size": "$skills"
        },
        4
      ]
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `523`

### WF-NEST-005  `CAP-NEST/array_size_expr`  difficulty=medium

**NL question:** "How many employees have at most 2 skills listed?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "employees",
  "filter": {
    "$expr": {
      "$lte": [
        {
          "$size": "$skills"
        },
        2
      ]
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `1059`


# SECTION: CAP-TIME (remaining cases)

### LOG-TIME-002  `CAP-TIME/quarter_window`  difficulty=medium

**NL question:** "How many shipments were placed in Q1 2025 (January through March)?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "shipments",
  "filter": {
    "placed_at": {
      "$gte": "2025-01-01T00:00:00",
      "$lt": "2025-04-01T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `3123`

### LOG-TIME-003  `CAP-TIME/date_format_project`  difficulty=hard

**NL question:** "List the shipment code and placement date (formatted as YYYY-MM-DD) for shipments delivered to Rome in December 2024."

**Gold MQL:**
```json
{
  "operation": "aggregate",
  "collection": "shipments",
  "filter": null,
  "pipeline": [
    {
      "$match": {
        "dest_city": "Rome",
        "placed_at": {
          "$gte": "2024-12-01T00:00:00",
          "$lt": "2025-01-01T00:00:00"
        }
      }
    },
    {
      "$project": {
        "_id": 0,
        "code": 1,
        "placed_date": {
          "$dateToString": {
            "format": "%Y-%m-%d",
            "date": "$placed_at"
          }
        }
      }
    }
  ],
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: 31 rows; first rows: `[{"code": "SHP-008526", "placed_date": "2024-12-01"}, {"code": "SHP-002212", "placed_date": "2024-12-01"}, {"code": "SHP-023895", "placed_date": "2024-12-01"}]`

### LOG-TIME-004  `CAP-TIME/half_year_window`  difficulty=medium

**NL question:** "How many shipments were placed in the second half of 2025 (July through December)?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "shipments",
  "filter": {
    "placed_at": {
      "$gte": "2025-07-01T00:00:00",
      "$lt": "2026-01-01T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `5175`

### LOG-TIME-005  `CAP-TIME/cross_year_window`  difficulty=medium

**NL question:** "How many shipments were placed between November 2024 and February 2025, both months inclusive?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "shipments",
  "filter": {
    "placed_at": {
      "$gte": "2024-11-01T00:00:00",
      "$lt": "2025-03-01T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `4287`

### LOG-TIME-006  `CAP-TIME/single_day`  difficulty=easy

**NL question:** "How many shipments were placed on exactly 2025-06-15?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "shipments",
  "filter": {
    "placed_at": {
      "$gte": "2025-06-15T00:00:00",
      "$lt": "2025-06-16T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `34`

### MET-TIME-002  `CAP-TIME/quarter_window`  difficulty=medium

**NL question:** "How many suspect-quality readings were logged in Q2 2025 (April through June)?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "readings",
  "filter": {
    "quality": "suspect",
    "ts": {
      "$gte": "2025-04-01T00:00:00",
      "$lt": "2025-07-01T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `10332`

### MET-TIME-003  `CAP-TIME/date_format_project`  difficulty=hard

**NL question:** "List the timestamp (formatted as YYYY-MM-DD HH:MM) and kwh for readings of device MTR-CAL-0001 on 2025-06-01."

**Gold MQL:**
```json
{
  "operation": "aggregate",
  "collection": "readings",
  "filter": null,
  "pipeline": [
    {
      "$match": {
        "serial": "MTR-CAL-0001",
        "ts": {
          "$gte": "2025-06-01T00:00:00",
          "$lt": "2025-06-02T00:00:00"
        }
      }
    },
    {
      "$project": {
        "_id": 0,
        "ts_fmt": {
          "$dateToString": {
            "format": "%Y-%m-%d %H:%M",
            "date": "$ts"
          }
        },
        "kwh": 1
      }
    }
  ],
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: 16 rows; first rows: `[{"kwh": 4.831, "ts_fmt": "2025-06-01 10:00"}, {"kwh": 5.214, "ts_fmt": "2025-06-01 10:03"}, {"kwh": 4.902, "ts_fmt": "2025-06-01 10:06"}]`

### MET-TIME-004  `CAP-TIME/single_day`  difficulty=easy

**NL question:** "How many alerts occurred on exactly 2025-05-20?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "alerts",
  "filter": {
    "ts": {
      "$gte": "2025-05-20T00:00:00",
      "$lt": "2025-05-21T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `52`

### MET-TIME-005  `CAP-TIME/month_window`  difficulty=easy

**NL question:** "How many alerts occurred in May 2025?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "alerts",
  "filter": {
    "ts": {
      "$gte": "2025-05-01T00:00:00",
      "$lt": "2025-06-01T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `1765`

### MET-TIME-006  `CAP-TIME/open_ended`  difficulty=easy

**NL question:** "How many devices were installed before 2024?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "devices",
  "filter": {
    "installed_on": {
      "$lt": "2024-01-01T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `40`

### WF-TIME-001  `CAP-TIME/month_window`  difficulty=medium

**NL question:** "How many employees were hired in March 2024?"

**Author note (verify, do not trust):** sentinel hires exactly at Mar-1 00:00 (2) and Apr-1 00:00 (3)

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "employees",
  "filter": {
    "hired_on": {
      "$gte": "2024-03-01T00:00:00",
      "$lt": "2024-04-01T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `17`

### WF-TIME-002  `CAP-TIME/explicit_tz`  difficulty=hard

**NL question:** "Appointments are stored in UTC. How many appointments at the New York office (code OFF-NYC) between January 13 and January 17 2025 (UTC calendar days, inclusive) start at exactly 09:00 local New York time? New York is on EST in January, which is UTC-5."

**Author note (verify, do not trust):** explicit-tz in NL by design (§2.1); tz-implicit variants belong to AMB. Post-audit: window declared as UTC calendar days in the NL (the local-day reading would shift bounds by 5h) and gold made minute-aware ('exactly 09:00' vs any 09:xx) so correctness rests on query semantics, not on planted-data luck

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "appointments",
  "filter": {
    "office_code": "OFF-NYC",
    "scheduled_at": {
      "$gte": "2025-01-13T00:00:00",
      "$lt": "2025-01-18T00:00:00"
    },
    "$expr": {
      "$and": [
        {
          "$eq": [
            {
              "$hour": "$scheduled_at"
            },
            14
          ]
        },
        {
          "$eq": [
            {
              "$minute": "$scheduled_at"
            },
            0
          ]
        }
      ]
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `6`

### WF-TIME-003  `CAP-TIME/quarter_window`  difficulty=medium

**NL question:** "How many employees were hired in Q3 2024 (July through September)?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "employees",
  "filter": {
    "hired_on": {
      "$gte": "2024-07-01T00:00:00",
      "$lt": "2024-10-01T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `58`

### WF-TIME-004  `CAP-TIME/date_format_project`  difficulty=hard

**NL question:** "List the employee code and hire month (formatted as YYYY-MM) for employees in the finance department hired during 2025."

**Gold MQL:**
```json
{
  "operation": "aggregate",
  "collection": "employees",
  "filter": null,
  "pipeline": [
    {
      "$match": {
        "department": "finance",
        "hired_on": {
          "$gte": "2025-01-01T00:00:00",
          "$lt": "2026-01-01T00:00:00"
        }
      }
    },
    {
      "$project": {
        "_id": 0,
        "employee_code": 1,
        "hire_month": {
          "$dateToString": {
            "format": "%Y-%m",
            "date": "$hired_on"
          }
        }
      }
    }
  ],
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: 18 rows; first rows: `[{"employee_code": "EMP-02513", "hire_month": "2025-03"}, {"employee_code": "EMP-00897", "hire_month": "2025-07"}, {"employee_code": "EMP-01241", "hire_month": "2025-06"}]`

### WF-TIME-007  `CAP-TIME/month_window`  difficulty=easy

**NL question:** "How many employees were hired in the month of April 2024?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "employees",
  "filter": {
    "hired_on": {
      "$gte": "2024-04-01T00:00:00",
      "$lt": "2024-05-01T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `21`

### WF-TIME-008  `CAP-TIME/open_ended`  difficulty=easy

**NL question:** "How many employees were hired before 2015?"

**Gold MQL:**
```json
{
  "operation": "count",
  "collection": "employees",
  "filter": {
    "hired_on": {
      "$lt": "2015-01-01T00:00:00"
    }
  },
  "pipeline": null,
  "projection": null,
  "sort": null,
  "limit": null,
  "distinct_field": null
}
```
-> computed gold result: `579`


---

End of packet. 32 cases. Reply with the single JSON object only.