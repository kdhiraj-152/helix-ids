# Alternative Schemas — Phase 32

**Date:** 2026-06-24
**Experiment:** Phase 32, RP-2 HELIX-IDS
**Status:** Complete — Negative Result

## Summary

Four new canonical schemas were designed from subsets of the 17 existing features, targeting different representational strategies. Two additional projection-based representations (PCA, random projection) were tested. **None reduced dataset-ID accuracy below 100%.**

## Schema Definitions

### Schema-A: Conservative (8 features)

**Design rationale:** Only universally available raw-mapped features with minimal dataset-specific processing. Excludes all interaction terms, ratios, and features with synthetic fallbacks.

| # | Feature | Justification |
|---|---------|---------------|
| 1 | protocol_type | 3-value categorical, universal |
| 2 | duration | Universal, direct measurement |
| 3 | src_bytes | Universal (name variant only) |
| 4 | dst_bytes | Universal (name variant only) |
| 5 | has_rst | Simple binary, consistent |
| 6 | traffic_direction | Derived but from byte ratio only |
| 7 | service_tier | Port/service-based categorization |
| 8 | flag | Encoded from raw flag/state |

**Notes:** Includes `flag` despite it being the #1 dataset-ID feature (Phase 31 Gini: 0.24), because excluding both flag and its derived features leaves only 6 features. This tests whether a conservative subset retains enough attack signal.

---

### Schema-B: Statistical (8 features)

**Design rationale:** Only derived features (log transforms, ratios, interaction terms). Excludes raw mappable features (protocol, flag, src_bytes). Tests whether statistical features alone encode the joint-distribution fingerprint.

| # | Feature | Category |
|---|---------|----------|
| 1 | log_src_bytes | Log transform |
| 2 | log_dst_bytes | Log transform |
| 3 | src_dst_bytes_ratio | Ratio |
| 4 | dst_src_bytes_ratio | Ratio |
| 5 | same_host_rate_x_service | Interaction + synthetic |
| 6 | diff_srv_rate_x_flag | Interaction + synthetic |
| 7 | count_x_srv_count | Interaction + synthetic |
| 8 | protocol_service_flag | Triple interaction |

**Notes:** These are the GEOMETRIC_EXPANSION_FEATURES from the harmonization pipeline. Tests whether the fingerprint is preserved even without direct raw features.

---

### Schema-C: Network-behavior (9 features)

**Design rationale:** Only features describing protocol and connection behavior. Excludes statistical aggregates.

| # | Feature | Category |
|---|---------|----------|
| 1 | protocol_type | Protocol identity |
| 2 | connection_state | Connection state |
| 3 | traffic_direction | Flow direction |
| 4 | has_rst | Connection termination |
| 5 | flag | Flag/state |
| 6 | service_tier | Service type |
| 7 | duration | Connection duration |
| 8 | src_bytes | Source byte count |
| 9 | dst_bytes | Destination byte count |

**Notes:** Includes the connection-behavior features that Phase 31 identified as highest fingerprint value (flag, connection_state). Tests whether the fingerprint survives in behavior-only space.

---

### Schema-D: Minimal Transfer (9 features)

**Design rationale:** Minimal set selected to minimize dataset fingerprint while retaining cross-dataset utility. Deliberately excludes:
- `flag` (#1 dataset-ID feature)
- `connection_state` (per-dataset state machine)
- All ratio features (#2, #3 fingerprint)
- All interaction features (synthetic fallbacks)

| # | Feature | Reason for Inclusion |
|---|---------|---------------------|
| 1 | protocol_type | Low-risk, 3-value universal |
| 2 | duration | Low-risk, direct |
| 3 | log_src_bytes | Low-risk, log transform |
| 4 | log_dst_bytes | Low-risk, log transform |
| 5 | src_bytes | Universal raw (medium risk) |
| 6 | dst_bytes | Universal raw (medium risk) |
| 7 | has_rst | Low-risk binary |
| 8 | traffic_direction | Medium-risk, byte-derived |
| 9 | service_tier | Higher-risk but needed for attack signal |

**Notes:** This is the best-effort attempt to minimize dataset identity. It removes the top-5 Phase 31 fingerprint features (flag, both ratios, connection_state, protocol_service_flag).

---

### Additional Representations

| Name | Description |
|------|-------------|
| PCA-8 | 8 principal components of all 17 features (StandardScaler-normalized) |
| PCA-5 | 5 principal components of all 17 features |
| RP-8 | 8-dimensional Gaussian random projection of all 17 features |

**Rationale:** These test whether *transformations* of the feature space can break the joint-distribution fingerprint that Phase 31 identified.

---

## Results

### Dataset-ID Accuracy (Primary Criterion)

| Schema | Features | Dataset-ID Acc | Target (<80%) | Met? |
|--------|----------|---------------|---------------|------|
| Phase31 Baseline | 17 | 100.0% | <80% | ❌ |
| Schema-A (Conservative) | 8 | 100.0% | <80% | ❌ |
| Schema-B (Statistical) | 8 | 100.0% | <80% | ❌ |
| Schema-C (Network-behavior) | 9 | 100.0% | <80% | ❌ |
| Schema-D (Minimal transfer) | 9 | 99.97% | <80% | ❌ |
| PCA-8 | 8 | 100.0% | <80% | ❌ |
| PCA-5 | 5 | 100.0% | <80% | ❌ |
| RP-8 | 8 | 99.99% | <80% | ❌ |

### Cross-Dataset Macro F1 (Secondary Criterion)

| Schema | Features | CD-MF1 | Δ vs Baseline | Target (≥+25%) | Met? |
|--------|----------|--------|--------------|----------------|------|
| Phase31 Baseline | 17 | 0.3718 | — | — | — |
| PCA-8 | 8 | 0.3739 | +0.6% | ≥+25% | ❌ |
| Schema-A (Conservative) | 8 | 0.3681 | −1.0% | ≥+25% | ❌ |
| Schema-C (Network-behavior) | 9 | 0.3639 | −2.1% | ≥+25% | ❌ |
| Schema-B (Statistical) | 8 | 0.3092 | −16.8% | ≥+25% | ❌ |
| Schema-D (Minimal transfer) | 9 | 0.3098 | −16.7% | ≥+25% | ❌ |
| RP-8 | 8 | 0.2968 | −20.2% | ≥+25% | ❌ |

### Attack Silhouette (Tertiary Criterion)

| Schema | Attack Silhouette | Δ vs Baseline | Better? |
|--------|------------------|--------------|---------|
| Phase31 Baseline | −0.0090 | — | — |
| Schema-C (Network-behavior) | −0.0173 | −0.008 | ❌ |
| PCA-8 | −0.0287 | −0.020 | ❌ |
| Schema-A (Conservative) | −0.0254 | −0.016 | ❌ |
| RP-8 | −0.0375 | −0.029 | ❌ |
| Schema-D (Minimal transfer) | −0.0537 | −0.045 | ❌ |
| Schema-B (Statistical) | −0.1117 | −0.103 | ❌ |

## Conclusion

**None of the 7 alternative representations meets any success criterion:**

1. Every schema achieves ≥99.97% dataset-ID accuracy (target: <80%)
2. No schema improves cross-dataset MF1 by ≥25% (best: PCA-8 at +0.6%)
3. No schema improves attack-family silhouette (all remain negative)

This confirms Phase 31's finding: **dataset identity is encoded in the joint distribution of ALL features**, not in specific features or interactions that can be removed or transformed away. Alternative feature representations within the same harmonized space cannot eliminate the fingerprint.
