# Typecheck Report — Phase 25C

Generated: 2026-06-21

**Tool:** mypy src/

## Summary

Success: no issues found in 73 source files

## TON-IoT Related Fixes

The following type error was found and fixed:

1. **`feature_harmonization.py:538`** — `create_ton_iot_mapping()` passed `label_column="type"` which is not a valid field of the `FeatureMapping` dataclass, and omitted required fields `original_features` and `common_features`.

### Fix applied

- Removed invalid `label_column="type"` argument
- Added `original_features=[]` and `common_features=COMMON_FEATURES` to match the pattern used by `create_nslkdd_mapping()`, `create_unsw_mapping()`, and `create_cicids_mapping()`

## Verification

- mypy clean: Success
- pytest data+operations: 172 passed in 11.51s

## Verdict

**PASS** — All type checks now pass.
