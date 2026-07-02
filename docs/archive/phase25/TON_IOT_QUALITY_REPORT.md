# TON-IoT Data Quality Report

Generated: 2026-06-21 20:21:07 (updated with deep inspection)

## Summary

- **Total rows:** 422,086 (211,043 train + 211,043 test)
- **Total columns:** 44
- **Duplicate rows:** 231,612 (54.87%) across combined dataset
- **Duplicate rows per split:** ~20,569 (9.75%) in train; ~20,569 (9.75%) in test
- **Duplicate flows (IP+port+proto):** 300,560 (71.21%)
- **Unique rows (after dedup):** ~190,474
- **Empty columns:** 0
- **Constant columns:** 0

## CRITICAL: Train/Test Split Identity

`train.csv` and `test.csv` are **identical files** — same shape (211,043 × 44), same content.

```
train.equals(test) → True
```

This means:
- There is **no train/test separation** — the same data appears in both files
- Any training run using both files will experience 100% data leakage
- The 54.87% duplicate rate is an artifact of combining two identical copies
- **Effective unique dataset size:** ~190,474 rows (not 422,086)

## NaN Rates

| Column | NaN Count | NaN Rate |
|--------|-----------|---------|
| All 44 columns | 0 | 0.00% |

**No missing values detected in any column.**

## Inf Rates

No infinite values detected.

## Empty Columns

None.

## Constant Columns

None.

## Corrupted Values

None detected.

## Numeric Stats (selected columns)

| Column | Min | Max | Mean | % Zero | % Negative |
|--------|-----|-----|------|--------|-----------|
| `src_bytes` | 0 | 3.89e9 | 2.58e5 | ~+50% | 0% |
| `dst_bytes` | 0 | 3.91e9 | 2.59e5 | ~+50% | 0% |
| `duration` | 0 | 9.35e4 | 7.70 | ~+75% | 0% |
| `src_pkts` | 0 | 2.09e6 | 3.96e2 | — | 0% |
| `dst_pkts` | 0 | 8.55e5 | 2.63e2 | — | 0% |
| `src_ip_bytes` | 40 | 1.70e9 | 3.42e5 | — | 0% |
| `dst_ip_bytes` | 40 | 1.03e9 | 1.97e5 | — | 0% |
| `missed_bytes` | 0 | 2.29e5 | 5.73 | 98.6% | 0% |

## Conn_State Values (categorical)

| State | Count | In NSL_STATE_MAP? |
|-------|-------|-------------------|
| `S0` | 51,937 | Yes |
| `SF` | 50,210 | Yes |
| `REJ` | 44,852 | Yes |
| `OTH` | 23,332 | Yes |
| `SH` | 12,014 | Yes |
| `S1` | 10,771 | Yes |
| `S3` | 6,557 | Yes |
| `SHR` | 5,629 | Yes |
| `RSTR` | 1,989 | Yes |
| `RSTRH` | 1,690 | **NO** |
| `RSTO` | 1,309 | Yes |
| `S2` | 627 | Yes |
| `RSTOS0` | 126 | Yes |

Only **RSTRH** (1,690 rows, 0.8%) has no mapping in the existing NSL_STATE_MAP. It should be mapped to `"RST"`.

## Protocol Values

| Protocol | Count | Canonical Mapping |
|----------|-------|------------------|
| `tcp` | 168,747 | protocol_type=0 (TCP) |
| `udp` | 42,015 | protocol_type=1 (UDP) |
| `icmp` | 281 | protocol_type=2 (ICMP) |

## Service Values

| Service | Count | Service Tier |
|---------|-------|-------------|
| `-` (none) | 132,032 | none |
| `dns` | 39,446 | dns |
| `http` | 37,029 | web |
| `ftp` | 1,065 | ftp |
| `ssl` | 1,025 | web |
| `gssapi` | 184 | auth |
| `dce_rpc` | 136 | auth |
| `smb` | 108 | auth |
| `smb;gssapi` | 18 | auth |

## Quality Rating: MODERATE

| Metric | Rating | Notes |
|--------|--------|-------|
| NaN rate | ✅ Excellent | 0% |
| Inf rate | ✅ Excellent | 0% |
| Duplicate rows | ⚠️ Moderate | 9.75% in split; 54.87% if combined |
| Empty columns | ✅ Excellent | None |
| Constant columns | ✅ Excellent | None |
| Corrupted values | ✅ Excellent | None |
| Train/test split | ❌ Critical | Identical files |
| Class imbalance | ⚠️ Moderate | 76.31% attack vs 23.69% normal; R2L heavily overrepresented |
