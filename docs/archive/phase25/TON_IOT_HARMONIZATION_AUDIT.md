# TON-IoT Harmonization Audit Report

**Generated:** 2026-06-21 IST

## Canonical Feature Contract

| Property | Expected | Actual | Status |
|----------|----------|--------|--------|
| Feature count | 17 | 17 | PASS |
| Feature order matches canonical | Yes | Yes | PASS |
| Feature collapse | None | None | PASS |

## Feature Distributions (TON-IoT)

| Feature | dtype | n_unique | min | max | mean | std | zero_pct |
|---------|-------|----------|-----|-----|------|-----|----------|
| protocol_type | int64 | 3 | 0.0000 | 2.0000 | 0.2156 | 0.4135 | 78.5% |
| connection_state | int64 | 6 | 0.0000 | 8.0000 | 2.4352 | 2.4399 | 26.3% |
| traffic_direction | int64 | 3 | 0.0000 | 2.0000 | 1.1477 | 0.5802 | 10.5% |
| has_rst | int64 | 2 | 0.0000 | 1.0000 | 0.2499 | 0.4330 | 75.0% |
| log_src_bytes | float64 | 2199 | 0.0000 | 22.0819 | 1.9928 | 2.6366 | 61.7% |
| log_dst_bytes | float64 | 2338 | 0.0000 | 22.0878 | 2.1558 | 3.2761 | 67.4% |
| src_dst_bytes_ratio | float64 | 5777 | 0.0000 | 909211709.0000 | 10471.7252 | 2640623.8930 | 61.7% |
| dst_src_bytes_ratio | float64 | 5625 | 0.0000 | 1000000000.0000 | 16154.6180 | 3535642.8060 | 67.4% |
| same_host_rate_x_service | float64 | 5777 | 0.0000 | 1.0000 | 0.1572 | 0.2993 | 61.7% |
| diff_srv_rate_x_flag | float64 | 6346 | 0.0000 | 2.0000 | 0.2893 | 0.4064 | 61.7% |
| count_x_srv_count | float64 | 85145 | 0.0000 | 909212103.2800 | 10480.2577 | 2640632.7979 | 21.9% |
| protocol_service_flag | float64 | 2 | 1.0000 | 2.0000 | 1.2499 | 0.4330 | 0.0% |
| src_bytes | float64 | 2182 | 0.0000 | 1000000000.0000 | 252802.0673 | 14445330.6999 | 61.7% |
| dst_bytes | float64 | 2322 | 0.0000 | 1000000000.0000 | 244829.5470 | 14077458.1152 | 67.4% |
| service_tier | int64 | 5 | 0.0000 | 6.0000 | 0.6593 | 0.9443 | 59.1% |
| duration | float64 | 68570 | 0.0000 | 93516.9292 | 8.5325 | 593.8158 | 21.9% |
| flag | int64 | 11 | 0.0000 | 10.0000 | 3.0476 | 3.2010 | 26.4% |

## Feature Order Verification

```
Position | Canonical Feature | TON-IoT Feature | Match
---------|-------------------|-----------------|-------
       1 | protocol_type     | protocol_type   | ✓
       2 | connection_state  | connection_state | ✓
       3 | traffic_direction | traffic_direction | ✓
       4 | has_rst           | has_rst         | ✓
       5 | log_src_bytes     | log_src_bytes   | ✓
       6 | log_dst_bytes     | log_dst_bytes   | ✓
       7 | src_dst_bytes_ratio | src_dst_bytes_ratio | ✓
       8 | dst_src_bytes_ratio | dst_src_bytes_ratio | ✓
       9 | same_host_rate_x_service | same_host_rate_x_service | ✓
      10 | diff_srv_rate_x_flag | diff_srv_rate_x_flag | ✓
      11 | count_x_srv_count | count_x_srv_count | ✓
      12 | protocol_service_flag | protocol_service_flag | ✓
      13 | src_bytes         | src_bytes       | ✓
      14 | dst_bytes         | dst_bytes       | ✓
      15 | service_tier      | service_tier    | ✓
      16 | duration          | duration        | ✓
      17 | flag              | flag            | ✓
```

## Distribution Assessment

No distribution anomalies detected. All 17 features show reasonable variability.

## Verdict

**PASS** — All 17 canonical features present in correct order with no collapse.
