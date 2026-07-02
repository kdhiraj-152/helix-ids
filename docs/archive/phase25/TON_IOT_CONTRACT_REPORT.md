# TON-IoT Learnability Contract Report

**Generated:** 2026-06-21 IST
**Pipeline:** `load_ton_iot()` → `harmonize_ton_iot()` → contract validation

## Verdict

**PASS**

## Summary

| Check | Status |
|-------|--------|
| Feature count = 17 | PASS |
| Feature order matches canonical | PASS |
| Schema hash matches canonical | PASS |
| No NaN in features | PASS |
| No Inf in features | PASS |
| No constant features | PASS |
| No impossible values | PASS |
| Label range valid [0–6] | PASS |
| All labels resolved | PASS |

## Data Details

| Metric | Value |
|--------|-------|
| Raw rows loaded | 211,043 |
| Raw columns | 44 |
| After dedup | 190,474 |
| After harmonization | 190,474 |
| Feature count | 17 |
| Label classes | [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(6)] |
| Class distribution | {"0": 42040, "1": 38985, "2": 20000, "3": 54962, "6": 34487} |

## Schema

| Property | Value |
|----------|-------|
| Schema version | 2026-05-25 |
| Canonical hash | 00ca8cc663c655e7cd28aff4271f9b22e0868e107202aca38b73504f5b5a4646 |
| Computed hash | 00ca8cc663c655e7cd28aff4271f9b22e0868e107202aca38b73504f5b5a4646 |
| Hash match | PASS |

## Feature List

```text
[
  "protocol_type",
  "connection_state",
  "traffic_direction",
  "has_rst",
  "log_src_bytes",
  "log_dst_bytes",
  "src_dst_bytes_ratio",
  "dst_src_bytes_ratio",
  "same_host_rate_x_service",
  "diff_srv_rate_x_flag",
  "count_x_srv_count",
  "protocol_service_flag",
  "src_bytes",
  "dst_bytes",
  "service_tier",
  "duration",
  "flag"
]
```

## Integrity Details

| Check | Result |
|-------|--------|
| NaN features | None |
| Inf features | None |
| Constant features | None |
| Impossible values | None |

## Raw Label Distribution (Before Mapping)

The TON-IoT raw CSV contains both a `label` and a `type` column. The pipeline uses the `label` column directly (no rename needed).

## Feature Distributions

```text
{
  "protocol_type": {
    "dtype": "int64",
    "min": 0.0,
    "max": 2.0,
    "mean": 0.2156,
    "std": 0.4135,
    "n_unique": 3,
    "n_zero": 149596
  },
  "connection_state": {
    "dtype": "int64",
    "min": 0.0,
    "max": 8.0,
    "mean": 2.4352,
    "std": 2.4399,
    "n_unique": 6,
    "n_zero": 50002
  },
  "traffic_direction": {
    "dtype": "int64",
    "min": 0.0,
    "max": 2.0,
    "mean": 1.1477,
    "std": 0.5802,
    "n_unique": 3,
    "n_zero": 20074
  },
  "has_rst": {
    "dtype": "int64",
    "min": 0.0,
    "max": 1.0,
    "mean": 0.2499,
    "std": 0.433,
    "n_unique": 2,
    "n_zero": 142870
  },
  "log_src_bytes": {
    "dtype": "float64",
    "min": 0.0,
    "max": 22.0819,
    "mean": 1.9928,
    "std": 2.6366,
    "n_unique": 2199,
    "n_zero": 117612
  },
  "log_dst_bytes": {
    "dtype": "float64",
    "min": 0.0,
    "max": 22.0878,
    "mean": 2.1558,
    "std": 3.2761,
    "n_unique": 2338,
    "n_zero": 128324
  },
  "src_dst_bytes_ratio": {
    "dtype": "float64",
    "min": 0.0,
    "max": 909211709.0,
    "mean": 10471.7252,
    "std": 2640623.893,
    "n_unique": 5777,
    "n_zero": 117612
  },
  "dst_src_bytes_ratio": {
    "dtype": "float64",
    "min": 0.0,
    "max": 1000000000.0,
    "mean": 16154.618,
    "std": 3535642.806,
    "n_unique": 5625,
    "n_zero": 128324
  },
  "same_host_rate_x_service": {
    "dtype": "float64",
    "min": 0.0,
    "max": 1.0,
    "mean": 0.1572,
    "std": 0.2993,
    "n_unique": 5777,
    "n_zero": 117612
  },
  "diff_srv_rate_x_flag": {
    "dtype": "float64",
    "min": 0.0,
    "max": 2.0,
    "mean": 0.2893,
    "std": 0.4064,
    "n_unique": 6346,
    "n_zero": 117439
  },
  "count_x_srv_count": {
    "dtype": "float64",
    "min": 0.0,
    "max": 909212103.28,
    "mean": 10480.2577,
    "std": 2640632.7979,
    "n_unique": 85145,
    "n_zero": 41766
  },
  "protocol_service_flag": {
    "dtype": "float64",
    "min": 1.0,
    "max": 2.0,
    "mean": 1.2499,
    "std": 0.433
```
