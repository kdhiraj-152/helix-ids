# Phase 32 Schema Certification

## Verdict: NEGATIVE — No Schema Passes

**Date:** 2026-06-24 (results reproduced 2026-06-24)
**Certifier:** Phase 32 Harmonization Redesign Pipeline
**Status:** ❌ ALL CRITERIA NOT MET

---

## Executive Summary

Phase 32 tested 7 alternative canonical feature representations designed to reduce dataset identity below the Phase 31 ceiling. **None of the 7 representations meets any of the 3 success criteria.**

The 17-feature canonical schema cannot be meaningfully redesigned to eliminate dataset fingerprints while preserving cross-dataset attack detection performance.

---

## Representations Tested

| ID | Name | Features | Strategy |
|----|------|----------|----------|
| Baseline | Phase31-Baseline | 17 | Current full schema |
| Schema-A | Conservative | 8 | Universally available raw features |
| Schema-B | Statistical | 8 | Log transforms, ratios, interactions only |
| Schema-C | Network-behavior | 9 | Protocol, flags, connection states |
| Schema-D | Minimal transfer | 9 | Excludes top-5 fingerprint features |
| PCA-5 | PCA 5-dim | 5 | Principal components of all 17 |
| PCA-8 | PCA 8-dim | 8 | Principal components of all 17 |
| RP-8 | Random projection | 8 | Gaussian random projection of all 17 |

---

## Criterion Results

### Primary: Dataset-ID Accuracy < 80%

| Schema | Dataset-ID Acc | Target | Result |
|--------|---------------|--------|--------|
| Phase31 Baseline | 100.0% | <80% | ❌ |
| Schema-A | 100.0% | <80% | ❌ |
| Schema-B | 100.0% | <80% | ❌ |
| Schema-C | 100.0% | <80% | ❌ |
| Schema-D | 99.97% | <80% | ❌ |
| PCA-8 | 100.0% | <80% | ❌ |
| PCA-5 | 100.0% | <80% | ❌ |
| RP-8 | 99.99% | <80% | ❌ |

**Verdict:** ❌ NOT MET. All schemas ≥99.97% accuracy.

### Secondary: Cross-Dataset MF1 ≥ +25% Improvement

| Schema | Avg MF1 | Δ vs Baseline | Target | Result |
|--------|---------|--------------|--------|--------|
| PCA-8 | 0.3739 | +0.6% | ≥+25% | ❌ |
| Schema-A | 0.3681 | −1.0% | ≥+25% | ❌ |
| Schema-C | 0.3639 | −2.1% | ≥+25% | ❌ |
| Schema-D | 0.3098 | −16.7% | ≥+25% | ❌ |
| Schema-B | 0.3092 | −16.8% | ≥+25% | ❌ |
| RP-8 | 0.2968 | −20.2% | ≥+25% | ❌ |

**Verdict:** ❌ NOT MET. Best performer (PCA-8) achieves +0.6%.

### Tertiary: Attack Silhouette Improvement

| Schema | Attack Silhouette | Δ vs Baseline | Result |
|--------|------------------|--------------|--------|
| Phase31 Baseline | −0.0090 | — | ❌ |
| All tested | −0.017 to −0.112 | Negative | ❌ |

**Verdict:** ❌ NOT MET. All schemas have equal or worse attack-family silhouette.

---

## Key Findings

### 1. Dataset identity persists in every representation

Dataset-ID classification achieves effectively 100% accuracy across all 7 alternative representations. Removing the top-5 fingerprint features (Schema-D) only drops accuracy from 100.0% to 99.97%. PCA and random projection (which should break joint-distribution structure) also retain 100% accuracy.

### 2. Cross-dataset performance degrades when removing fingerprint features

Every schema that reduces the feature set hurts at least one hold-out dataset's binary F1. The CICIDS holdout is particularly sensitive: removing flag, connection_state, or the ratio features collapses its binary F1 from 0.590 to near zero (0.000–0.005).

### 3. No zero-tradeoff exists

The fingerprint and the attack signal share the same feature subspace. Features that identify the dataset also carry legitimate attack-discriminating information. They cannot be separated by feature selection, ablation, or linear transformation.

---

## Root Cause Analysis

### Why schema redesign failed

The handoff from Phase 31 identified the problem correctly: "dataset fingerprint is encoded in the joint distribution." Phase 32 confirms that **any feature subset or transformation of the harmonized data preserves this fingerprint**.

The root cause is structural:
1. **Source datasets capture fundamentally different network environments**: simulated military traffic (NSL-KDD), synthetic enterprise traffic (UNSW), real production traffic (CICIDS), and IoT sensor data (TON-IoT).
2. **Dataset identity is not in feature engineering — it's in the data itself**. No amount of feature redesign within the harmonized space can eliminate the fact that NSL-KDD attacks look different from CICIDS attacks because they *are* different.
3. **Harmonization homogenizes the column schema, not the data distribution.**

### Why projection transforms failed

PCA and random projection were tested specifically to break the joint-distribution fingerprint identified in Phase 31. Both fail because:
- PCA preserves the variance structure that encodes dataset identity in its top components.
- Random projection preserves pairwise distances statistically, which is sufficient for a sufficiently complex classifier (RF) to decode dataset origin.

---

## Recommendation

**Do not proceed to Phase 33 (DANN + New Schema).**

The fundamental assumption — that a better feature schema would reduce dataset identity — has been falsified. The 17-feature canonical schema is not the bottleneck; the bottleneck is the **intrinsic incompatibility** of the source datasets themselves.

### Alternative Paths

1. **Publish final conclusion** that cross-dataset IDS transfer is fundamentally constrained by dataset incompatibility rather than adaptation architecture.
2. **Phase 33 as originally scoped (DANN on current schema)**: This remains a valid experiment — DANN may extract dataset-invariant representations from the current 17 features. But Phase 32 implies DANN's effectiveness will be limited because the fingerprint is structural, not distributional.
3. **Multi-source training only**: Accept that NSL-KDD and UNSW-NB15 have compatible structure and train only on those; treat CICIDS as a separate deployment domain.
4. **Domain-specific fine-tuning**: Train on combined NSL+UNSW, then fine-tune a lightweight adapter for each target deployment without requiring cross-dataset transfer.

---

## Dependencies

- Requires `results.json` in `docs/phase32/` (generated by `phase32_harmonization_redesign.py`)
- All references to Phase 31 results refer to `docs/phase31/` reports
- Analysis script: `scripts/analysis/phase32_harmonization_redesign.py`
- All results produced at SEED=42, reproducible via the script

---

## Certification Record

- **Primary criterion (DS-ID <80%):** ❌ NOT MET (best: 99.97%)
- **Secondary criterion (MF1 ≥+25%):** ❌ NOT MET (best: +0.6%)
- **Tertiary criterion (Attack silhouette):** ❌ NOT MET (all negative)
- **Decision:** Do not proceed to Phase 33. Publish fundamental constraint.
