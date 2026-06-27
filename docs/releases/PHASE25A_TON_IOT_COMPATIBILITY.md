---
title: PHASE 25A — TON-IoT Compatibility Verdict
status: final
date: 2026-06-21
tags: [ton-iot, ingestion, phase25a]
---

# PHASE 25A — TON-IoT Compatibility Verdict

## Recommendation

**NO-GO**

TON-IoT cannot enter the HelixIDS-Full pipeline today without engineering work. Two BLOCKER-level issues prevent automated ingestion.

## Dataset Overview

- **Files:** 2 (train.csv, test.csv)
- **Raw rows:** 422,086 (211,043 × 2 — identical files)
- **Unique rows after dedup:** ~190,474
- **Total size:** 56.78 MB
- **Columns:** 44
- **Attack categories:** 9 (backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss)
- **Normal categories:** 1 (normal)

## Schema Compatibility

| Metric | Value |
|--------|-------|
| Helix canonical features | 17 |
| TON-IoT columns | 44 |
| Exact matches to canonical | 4 (src_bytes, dst_bytes, duration, protocol_type via proto) |
| Derivable from raw | 13 (all geometry expansion + derived features) |
| Missing canonical features | 0 |

**All 17 canonical features are representable** from TON-IoT raw columns. No canonical feature is structurally missing.

## Label Compatibility

| Metric | Value |
|--------|-------|
| Unique attack types | 9 |
| Known to Helix (no changes) | 4 (backdoor, ddos, dos, normal) |
| Needs expansion | 5 (injection, password, ransomware, scanning, xss) |
| Impossible to map | 1 (mitm — 0.49%, 2,086 rows) |

**5 new label mappings** must be added to `attack_taxonomy.py`. MITM has no canonical family; proposed as R2L.

## Data Quality

| Metric | Value |
|--------|-------|
| Duplicate rows (per split) | 9.75% (manageable) |
| Duplicate rows (combined) | 54.87% (artifact of identical train/test) |
| Train/test split identity | ❌ **CRITICAL** — files are byte-identical |
| NaN/Inf | 0.00% ✅ |
| Empty/constant columns | None ✅ |
| Corrupted values | None ✅ |

## Pipeline Test

| Test | Result |
|------|--------|
| Learnability contract | **FAIL** — no FeatureMapping for TON-IoT |
| Feature harmonization (simulated) | **PARTIAL** — all 17 features derivable, but dataset-specific code rejects TON-IoT |

## Blockers

### [BLOCKER] No FeatureMapping for TON-IoT

The `harmonize_features()` function in `feature_harmonization.py` has no TON-IoT entry. The dataset-specific derivation functions (`_derive_connection_state`, `_derive_traffic_direction`, `_derive_has_rst`, `_derive_service_tier`) each have hardcoded branches for only `nsl_kdd`, `unsw_nb15`, and `cicids`. Calling with `dataset_name="ton_iot"` raises `AssertionError: Unsupported dataset`.

**Engineering needed:** Add TON-IoT branches to 4 derivation functions + add `FeatureMapping` for TON-IoT.

**Effort:** ~1 day.

### [BLOCKER] Missing Conn_State Value: RSTRH

The TON-IoT `conn_state` column contains the value `RSTRH` (1,690 rows, 0.8%) which is not present in the `NSL_STATE_MAP`. The existing map has `"rstoh"` → RST but does not handle `RSTRH`. The existing NSL-KDD state map would need one entry: `"rstrh": "RST"`.

**Engineering needed:** Add `"rstrh": "RST"` to `NSL_STATE_MAP`.

**Effort:** 5 minutes.

### [HIGH] Train/Test Split Identity

`train.csv` and `test.csv` are identical files. Any training run using both without fixing this split will experience 100% data leakage. The 54.87% duplicate rate is entirely an artifact of combining two identical copies.

**Mitigation:** Use `train_test_split()` on the concatenated deduplicated data instead of the raw files. The effective dataset size is ~190,474 unique rows.

**Effort:** 1 hour (data pipeline fix).

### [MEDIUM] Label Mappings Required

5 TON-IoT attack types require mapping expansion in `attack_taxonomy.py`:

| TON-IoT Label | Proposed Family | Rationale |
|--------------|----------------|-----------|
| `injection` | R2L | Similar to existing CICIDS SQL injection mapping |
| `password` | R2L | Similar to existing NSL-KDD guess_passwd mapping |
| `ransomware` | DoS | Disrupts availability; closest match |
| `scanning` | Probe | Exact semantic match to NSL-Probe |
| `xss` | R2L | Similar to existing CICIDS web attack mapping |

One label (`mitm`, 2,086 rows) has no exact family equivalent. Proposed as R2L (unauthorised access).

**Engineering needed:** Add `TON_IOT_TO_UNIFIED_5CLASS` mapping (6 entries) to `attack_taxonomy.py`.

**Effort:** 2 hours.

### [LOW] Dataset Download Source Not Verified

The current `data/ton_iot/raw/` contains pre-split train/test CSV files but there is no download script (`scripts/data/download_ton_iot.py`) to reproduce the dataset acquisition.

**Engineering needed:** Add download script referencing the official UNSW TON-IoT repository.

**Effort:** 2 hours.

## Phase 25B Effort Estimate

If GO was issued, the following work items would be required:

| Item | Effort | Description |
|------|--------|-------------|
| Feature mapping | 1 day | Add TON-IoT FeatureMapping to feature_harmonization.py + 4 derivation branches |
| Conn state fix | 5 min | Add RSTRH→RST mapping to NSL_STATE_MAP |
| Label mappings | 2 hours | Add TON_IOT_TO_UNIFIED_5CLASS to attack_taxonomy.py (6 entries) |
| Train/test fix | 1 hour | Replace raw split with proper train_test_split on deduplicated data |
| Data pipeline | 1 day | Add process_ton_iot.py entrypoint + schema hash |
| Download script | 2 hours | Add scripts/data/download_ton_iot.py |
| Integration test | 1 day | Validate multi-dataset contract with TON-IoT |
| **Total** | **4-5 days** | Not a full sprint, but requires deliberate engineering |

## Verdict Details

| Check | Severity | Status |
|-------|----------|--------|
| Learnability Contract | BLOCKER | ❌ No FeatureMapping |
| Conn State Mapping | BLOCKER | ❌ RSTRH missing |
| Train/Test Split | HIGH | ❌ Identical files |
| Label Expansion | MEDIUM | ⚠️ 5 new labels needed |
| Data Quality (NaN/Inf) | LOW | ✅ Clean |
| Duplicate Rows | LOW | ⚠️ 9.75% dedup needed |
| Protocol Mapping | LOW | ✅ Complete |
| Service Tier Derivation | LOW | ✅ Feasible |
| Download Reproducibility | LOW | ⚠️ No download script |

## Detailed Assessment

### Why this is NO-GO instead of GO

The barriers are all **engineering work**, not fundamental incompatibilities. However, the BLOCKER-level items (no FeatureMapping + missing conn_state value) mean TON-IoT **cannot be loaded by any existing pipeline command** without source code changes to `feature_harmonization.py` and `attack_taxonomy.py`. This violates the Phase 25A constraint of "no production code modified."

### If Phase 25B is approved

The TON-IoT dataset is structurally a strong candidate for HelixIDS-Full:

- All 17 canonical features are derivable from existing columns
- 0% NaN/Inf — no data quality issues
- Attack types are well-represented (40K rows each, except MITM)
- Protocol/state/service schemas mirror NSL-KDD closely
- Only 1 new conn_state value needed (RSTRH)
- Only 6 new label mappings needed

The data requires **4-5 days of deliberate engineering**, not a full sprint.

---

*Generated by Phase 25A audit script. No training performed. No production code modified.*
