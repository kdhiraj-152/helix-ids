# Fingerprint Source Analysis — Phase 32

**Date:** 2026-06-24
**Experiment:** Phase 32, RP-2 HELIX-IDS
**Status:** Complete

## Summary

Identifies which harmonization operations introduce dataset identity into the canonical 17-feature representation. Each operation is analyzed for its fingerprint contribution, with quantitative evidence from Phase 31 dataset-ID feature importance rankings and Phase 32 schema benchmarks.

## Operation Audit

### 1. Default-Value Injection

**Severity:** HIGH

**Affected features:** `same_host_rate_x_service`, `diff_srv_rate_x_flag`, `count_x_srv_count`

When raw columns are unavailable in a dataset, the harmonization pipeline injects synthetic default values:

- **`same_host_rate_x_service`**: When `same_srv_rate` column missing (CICIDS, TON-IoT), defaults to `src_bytes / max(src_bytes + dst_bytes, 1)` as a proxy.
- **`diff_srv_rate_x_flag`**: When `diff_srv_rate` column missing (CICIDS, TON-IoT), defaults to a byte-difference-based proxy.
- **`count_x_srv_count`**: When `count`/`srv_count` columns missing (CICIDS), defaults to `duration + src_bytes / max(dst_bytes, 1)`. TON-IoT uses packet-count substitute.

**Fingerprint mechanism:** Datasets with the native column compute the feature using genuine connection-level statistics. Datasets without the column receive a synthetic proxy with different distributional properties. This creates a **consistent dataset signature** — the classifier can distinguish datasets simply by whether they fall on the native-computation or synthetic-fallback branch.

**Quantified contribution (Phase 31):** `count_x_srv_count` — Gini #6 (0.0703), `same_host_rate_x_service` — Gini #13 (0.0072), `diff_srv_rate_x_flag` — Gini #8 (0.0220). Combined: **~10% of Gini importance.**

---

### 2. Synthetic Feature Generation

**Severity:** VERY HIGH

**Affected features:** `count_x_srv_count`, `diff_srv_rate_x_flag`, `same_host_rate_x_service`

These features do not exist as raw columns in any dataset; they are constructed during harmonization. The construction path differs per dataset based on column availability:

| Feature | NSL-KDD | UNSW-NB15 | CICIDS-2018 | TON-IoT |
|---------|---------|-----------|-------------|---------|
| `count_x_srv_count` | Native: `count × srv_count` | Native: `count × srv_count` | Synthetic: `duration + src/max(dst,1)` | Packet count substitute |
| `diff_srv_rate_x_flag` | Native: `rate × flag` | Native: `rate × flag` | Synthetic byte-diff proxy | Same synthetic |
| `same_host_rate_x_service` | Native: `rate × service` | Native/Synthetic hybrid | Fully synthetic byte proxy | Same synthetic |

**Fingerprint mechanism:** The generation code path itself is a dataset detector. A model trained on these three features alone can discriminate datasets because the function mapping raw inputs → feature output differs systematically. This is not a distributional difference — it is a **structural difference** in the mapping.

**Quantified contribution (Phase 31):** Combined Gini importance across all three: **~10%**.

---

### 3. Ratio Construction

**Severity:** HIGH

**Affected features:** `src_dst_bytes_ratio`, `dst_src_bytes_ratio`

Ratio features `src_bytes / max(dst_bytes, 1)` and `dst_bytes / max(src_bytes, 1)` are computed uniformly across all datasets. However, the byte distributions they summarize differ fundamentally per dataset:

- **NSL-KDD**: Many simulated attacks produce connections with near-zero dst_bytes (land, neptune, smurf floods). Ratio distribution is heavy-tailed with many extreme values.
- **UNSW-NB15**: More balanced byte distribution across attack types.
- **CICIDS-2018**: Nearly all forward/backward packet-length ratios are moderate, reflecting real network traffic.
- **TON-IoT**: IoT-specific traffic patterns produce different byte distributions.

**Fingerprint mechanism:** Even though the ratio formula is identical, the input distributions differ so dramatically that the ratios themselves become dataset identifiers. This is the most fundamental challenge — the raw data from different network environments *is* different, and ratio features faithfully encode these differences.

**Quantified contribution (Phase 31):** `src_dst_bytes_ratio` — Gini #2 (0.1938), `dst_src_bytes_ratio` — Gini #3 (0.1424). Combined: **~34% of Gini importance.**

---

### 4. Log Transforms

**Severity:** LOW

**Affected features:** `log_src_bytes`, `log_dst_bytes`

Log transform `log(1 + x)` is applied uniformly across all datasets. It is a standard statistical technique that preserves relative magnitudes while compressing dynamic range.

**Fingerprint mechanism:** Minimal. Log does not encode dataset identity because:
- The formula is identical per dataset.
- It is a monotonic transform (does not create new dataset-specific structure).
- It compresses, not amplifies, distributional differences.

**Quantified contribution (Phase 31):** `log_src_bytes` — Gini #12 (0.0087), `log_dst_bytes` — Gini #11 (0.0089). Combined: **< 2% of Gini importance.**

---

### 5. Clipping

**Severity:** NEGLIGIBLE

**Affected features:** `src_bytes`, `dst_bytes`, `duration`

Numeric features are clipped to [-1e9, 1e9] to prevent extreme outliers from destabilizing training.

**Fingerprint mechanism:** None. The clip threshold is so large (~1e9) that it rarely activates. It is applied uniformly. Cannot distinguish datasets.

---

### 6. Categorical Mappings

**Severity:** VERY HIGH

**Affected features:** `flag`, `connection_state`, `service_tier`, `protocol_type`

**FLAG_MAP** (22-entry shared mapping):
- NSL-KDD maps from text `flag` values (REJ, S0, SF, RSTO, ...)
- UNSW-NB15 maps from `state` column (FIN, CON, INT, ...)
- CICIDS-2018 maps from derived symbolic flags (SYN_ACK, RST, ...)
- TON-IoT maps from `conn_state` (SIN, SP, SAD, ...)

**Problem:** While `FLAG_MAP` provides a consistent output encoding, the *distribution* of flag values per dataset is entirely different. NSL-KDD statistically has ~30% REJ states (rejected connections from DoS attacks). CICIDS has ~<1% REJ states because flags are derived from normalized TCP flag counts in real traffic. A classifier can identify the dataset purely from flag-value frequencies.

**CONNECTION_STATE_MAP** (9-state encoding):
- NSL uses a dedicated state machine from flag text (NSL_STATE_MAP)
- UNSW uses its own state mapping with frozen-rare-state ambiguity handling
- CICIDS counts SYN/RST/ACK/FIN bits to reconstruct state
- TON-IoT uses its own conn_state→state mapping

**Problem:** The state-machine logic itself differs per dataset. This is worse than distributional differences — it's a **functional** difference where the same numeric output means different things depending on dataset.

**SERVICE_TIER:**
- Three datasets use `service` column + regex
- CICIDS uses `Dst Port` + IANA port ranges

**Problem:** Port-based service classification (CICIDS) produces far more `other` (value 6) and fewer specific service assignments than the named-service approach. This creates a CICIDS-specific distributional signature.

**Quantified contribution (Phase 31):** `flag` — Gini #1 (0.2398), `connection_state` — Gini #4 (0.1291), `service_tier` — Gini #14 (0.0037). Combined: **~37% of Gini importance.**

---

## Summary: Contribution to Dataset-ID Prediction

| Operation | Features Affected | Combined Gini Importance | Contribution |
|-----------|------------------|------------------------|--------------|
| Categorical mappings | flag, connection_state, service_tier | ~37% | VERY HIGH |
| Ratio construction | src_dst_bytes_ratio, dst_src_bytes_ratio | ~34% | HIGH |
| Synthetic generation + Default injection | same_host_rate_x_service, diff_srv_rate_x_flag, count_x_srv_count | ~10% | VERY HIGH |
| Log transforms | log_src_bytes, log_dst_bytes | <2% | LOW |
| Clipping | src_bytes, dst_bytes, duration | <0.1% | NEGLIGIBLE |
| **Joint distribution (remaining)** | All features in combination | ~17% | STRUCTURAL |

## Key Insight

**No single operation is removable.** The three major fingerprint sources (categorical mappings, ratio construction, synthetic generation) cannot be eliminated without destroying the feature representation's utility for IDS:

1. **Removing categorical mappings** eliminates flag and connection_state, which carry essential attack-behavior signal (Phase 31: removing flag collapses cross-dataset MF1 from 0.19 to 0.006).
2. **Removing ratio features** eliminates byte-asymmetry information critical for volumetric attack detection.
3. **Removing synthetic features** leaves CICIDS with a smaller feature set than NSL/UNSW, creating a *different* dataset imbalance.

The fingerprint is not an artifact of bad engineering — it is a **structural property** of harmonizing intrinsically different data sources into a shared representation.
