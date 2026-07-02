# TON-IoT Learnability Contract Test

Generated: 2026-06-21 20:21:07 (updated after deep inspection)

**Verdict: FAIL**

## Test Summary

- Pipeline test: simulation (no production code modified)
- Can harmonize: FALSE (without code changes)
- Core features available: YES (all 17 canonical features derivable)

## Failures

1. **No FeatureMapping for TON-IoT** — `harmonize_features()` falls back to `_derive_connection_state()` which rejects `dataset_name="ton_iot"` with `AssertionError: Unsupported dataset for connection_state derivation`.

2. **Dataset-specific derivation functions** — The following functions have hardcoded branches for only `nsl_kdd`, `unsw_nb15`, and `cicids`:
   - `_derive_connection_state()` — needs TON-IoT branch (conn_state values map directly to NSL-KDD flag values, plus RSTRH)
   - `_derive_traffic_direction()` — needs TON-IoT branch (src_bytes/dst_bytes available)
   - `_derive_has_rst()` — needs TON-IoT branch (conn_state has REJ/RST values)
   - `_derive_service_tier()` — needs TON-IoT branch (service column mirrors NSL-KDD)

3. **Conn_state value RSTRH** — not present in `NSL_STATE_MAP`. Requires one new entry: `"rstrh": "RST"`.

## Warnings

- All core raw features present for harmonization mapping:
  - `proto` → `protocol_type` (exact)
  - `conn_state` → `flag`/`connection_state` (via NSL-KDD state map + RSTRH)
  - `src_bytes`/`dst_bytes` → direct pass-through
  - `duration` → direct pass-through
  - `service` → `service_tier` derivation
- No NaN/Inf values in numeric columns
- No missing columns for canonical feature derivation

## Detailed Failure Path

The exact call chain that fails:

```
harmonize_features(df, mapping, label_col="label", mode="strict")
  → _derive_connection_state(working_df, "ton_iot")
    → raise AssertionError(f"Unsupported dataset for connection_state derivation: {dataset_name}")
```

## Assessment

**The TON-IoT data is structurally COMPATIBLE with the Helix canonical schema.** The barrier is engineering-only: no code path exists for TON-IoT in the dataset-specific derivation functions. The data itself has all the raw ingredients:

| Raw Feature | Usage | Compatibility |
|-------------|-------|--------------|
| `proto` | → `protocol_type` | Exact match (tcp=6, udp=17, icmp=1) |
| `conn_state` | → `flag` + `connection_state` | 12/13 values in NSL_STATE_MAP; RSTRH → RST |
| `src_bytes` | → `src_bytes` | Exact |
| `dst_bytes` | → `dst_bytes` | Exact |
| `duration` | → `duration` | Exact |
| `service` | → `service_tier` | Exact match to NSL-KDD pattern |
| `label` | → binary label | Exact (0=normal, 1=attack) |
| `type` | → family label | 9 attack types, 1 normal |

## Schema Hash

The expected schema hash for the canonical 17-feature order remains unchanged from NSL-KDD/UNSW/CICIDS pipelines.
