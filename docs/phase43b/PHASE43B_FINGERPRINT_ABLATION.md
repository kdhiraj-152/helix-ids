# Phase 43B — Fingerprint Feature Ablation Report

**Date:** 2026-06-26
**Experiment:** Phase 43B, RP-2 HELIX-IDS
**Objective:** Determine whether the dataset fingerprint is primarily caused by the top-ranked fingerprint features or by deeper structural differences throughout the canonical feature space.

## Executive Summary

Sequential feature ablation of the top 5 fingerprint features (identified in Phase 43A) across all 6 dataset pairs. At each ablation level, DOS (Domain Overlap Score) is computed as 1 - Dataset-ID Accuracy, where Dataset-ID Accuracy is the 5-fold CV accuracy of a Logistic Regression classifier distinguishing dataset origin.
DOS = 0.0 means perfect dataset separability (worst fingerprint). DOS = 0.5 means chance-level separability (no fingerprint).

## Per-Pair Results

### Per-Pair Dataset-ID Accuracy

| Dataset Pair | Baseline | Ablate top-1 | Ablate top-2 | Ablate top-3 | Ablate top-4 | Ablate top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| CICIDS2018 vs TON-IoT | 0.9999 | 0.9999 | 0.9991 | 0.9992 | 0.9992 | 0.9834 |
| NSL-KDD vs CICIDS2018 | 1.0000 | 1.0000 | 0.9966 | 0.9723 | 0.7168 | 0.7213 |
| NSL-KDD vs TON-IoT | 1.0000 | 1.0000 | 0.9971 | 0.9972 | 0.9971 | 0.9925 |
| NSL-KDD vs UNSW-NB15 | 1.0000 | 0.9951 | 0.9952 | 0.9241 | 0.9176 | 0.8731 |
| UNSW-NB15 vs CICIDS2018 | 1.0000 | 1.0000 | 0.9964 | 0.9830 | 0.8325 | 0.8326 |
| UNSW-NB15 vs TON-IoT | 1.0000 | 1.0000 | 0.9819 | 0.9810 | 0.9777 | 0.9076 |

### Per-Pair DOS

| Dataset Pair | Baseline | Ablate top-1 | Ablate top-2 | Ablate top-3 | Ablate top-4 | Ablate top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| CICIDS2018 vs TON-IoT | 0.0001 | 0.0001 | 0.0009 | 0.0008 | 0.0008 | 0.0166 |
| NSL-KDD vs CICIDS2018 | 0.0000 | 0.0000 | 0.0034 | 0.0277 | 0.2832 | 0.2787 |
| NSL-KDD vs TON-IoT | 0.0000 | 0.0000 | 0.0029 | 0.0028 | 0.0029 | 0.0075 |
| NSL-KDD vs UNSW-NB15 | 0.0000 | 0.0049 | 0.0048 | 0.0759 | 0.0824 | 0.1269 |
| UNSW-NB15 vs CICIDS2018 | 0.0000 | 0.0000 | 0.0036 | 0.0170 | 0.1675 | 0.1674 |
| UNSW-NB15 vs TON-IoT | 0.0000 | 0.0000 | 0.0181 | 0.0190 | 0.0223 | 0.0924 |

## Aggregate Ablation Results

### Aggregate Table (averaged across 6 dataset pairs)

| Features Removed | Mean Dataset-ID Accuracy | Mean DOS |
| --- | --- | --- |
| Baseline (0 removed) | 1.0000 | 0.0000 |
| 1: flag | 0.9992 | 0.0008 |
| 2: flag, protocol_service_flag | 0.9944 | 0.0056 |
| 3: flag, protocol_service_flag, connection_state | 0.9761 | 0.0239 |
| 4: flag, protocol_service_flag, connection_state, duration | 0.9068 | 0.0932 |
| 5: flag, protocol_service_flag, connection_state, duration, same_host_rate_x_service | 0.8851 | 0.1149 |

## Interpretation

- **Baseline mean DOS:** 0.0000
- **At top-5 removal mean DOS:** 0.1149
- **Mean DOS change (Δ):** +0.1149
- **Mean DOS change (median):** +0.1096

### Key Observations

1. **Redundant encoding.** Removing `flag` alone (the #1 ranked feature from Phase 43A) barely changes DOS (Δ = +0.0008). This confirms the fingerprint is NOT dependent on any single feature — it is redundantly encoded across correlated features. Removing one discriminatory feature leaves others that carry the same dataset-identity information.

2. **`duration` is the most impactful single feature for specific pairs.** The largest DOS jump occurs at ablation step 4 (adding `duration`), where mean DOS rises from 0.0239 to 0.0932 (Δ = +0.0693). This is almost entirely driven by the NSL-KDD vs CICIDS2018 pair (DOS jumps from 0.0277 → 0.2832). `duration` distributions differ massively between these datasets.

3. **Strong pair-specific asymmetry.** The fingerprint is not uniform across pairs:

   - **NSL-KDD vs CICIDS2018**: Δ DOS = +0.2787 (final DOS = 0.2787)
   - **UNSW-NB15 vs CICIDS2018**: Δ DOS = +0.1674 (final DOS = 0.1674)
   - **NSL-KDD vs UNSW-NB15**: Δ DOS = +0.1268 (final DOS = 0.1269)
   - **UNSW-NB15 vs TON-IoT**: Δ DOS = +0.0924 (final DOS = 0.0924)
   - **CICIDS2018 vs TON-IoT**: Δ DOS = +0.0165 (final DOS = 0.0166)
   - **NSL-KDD vs TON-IoT**: Δ DOS = +0.0075 (final DOS = 0.0075)

4. **Structural pairs resist ablation entirely.** NSL-KDD vs TON-IoT and CICIDS2018 vs TON-IoT remain at DOS < 0.02 even after removing all 5 top fingerprint features. These pairs have dataset-specific artifacts encoded across the remaining 12 features that no amount of feature removal within the canonical space can eliminate.

### Feature-by-Feature Impact

| Ablation Step | Feature Added | Δ Mean DOS | Cumulative Mean DOS | Key Impact Pair |
|---|---|---|---|
| 1 | flag | +0.0008 | 0.0008 | Minimal |
| 2 | protocol_service_flag | +0.0048 | 0.0056 | UNSW-NB15 vs TON-IoT (Δ=+0.0181) |
| 3 | connection_state | +0.0183 | 0.0239 | NSL-KDD vs UNSW-NB15 (Δ=+0.0710) |
| 4 | duration | +0.0693 | 0.0932 | NSL-KDD vs CICIDS2018 (Δ=+0.2555) |
| 5 | same_host_rate_x_service | +0.0217 | 0.1149 | UNSW-NB15 vs TON-IoT (Δ=+0.0701) |

### Verdict

**MIXED: Fingerprint is CONCENTRATED for some pairs, STRUCTURAL for others.**

The mean Δ DOS of +0.1149 exceeds the 0.10 threshold for "concentrated," but this aggregate statistic masks a bimodal distribution. Two distinct regimes exist:

- **Pairs where ablation works** (NSL-KDD↔CICIDS2018, UNSW-NB15↔CICIDS2018, NSL-KDD↔UNSW-NB15, UNSW-NB15↔TON-IoT): DOS rises 0.09–0.28 after removing top-5 features. For these pairs, the fingerprint IS concentrated in the top features. `duration` is particularly impactful.
- **Pairs where ablation fails** (NSL-KDD↔TON-IoT, CICIDS2018↔TON-IoT): DOS < 0.02 even after top-5 removal. The fingerprint is STRUCTURAL — encoded in the joint distribution of the remaining 12 features. Feature removal alone cannot fix this.

**Note:** NSL-KDD vs CICIDS2018 approaches the Phase 36 DOS threshold (0.30) after removing `duration`, reaching DOS=0.2832. This single pair suggests that benchmark repair MAY be partially feasible for specific dataset combinations by suppressing `duration` differences. However, most pairs remain far below threshold.

### Pairwise DOS Trajectory

| Dataset Pair | Baseline DOS | Top-5 Ablated DOS | Δ DOS |
|---|---|---|---|
| CICIDS2018 vs TON-IoT | 0.0001 | 0.0166 | +0.0165 |
| NSL-KDD vs CICIDS2018 | 0.0000 | 0.2787 | +0.2787 |
| NSL-KDD vs TON-IoT | 0.0000 | 0.0075 | +0.0075 |
| NSL-KDD vs UNSW-NB15 | 0.0000 | 0.1269 | +0.1268 |
| UNSW-NB15 vs CICIDS2018 | 0.0000 | 0.1674 | +0.1674 |
| UNSW-NB15 vs TON-IoT | 0.0000 | 0.0924 | +0.0924 |

### DOS by Ablation Step

This table shows how DOS evolves as each top feature is removed, across all 6 dataset pairs.

- **Baseline (0 removed)**: mean DOS = 0.0000 
- **1: flag**: mean DOS = 0.0008 (Δ from prev: +0.0008)
- **2: flag, protocol_service_flag**: mean DOS = 0.0056 (Δ from prev: +0.0048)
- **3: flag, protocol_service_flag, connection_state**: mean DOS = 0.0239 (Δ from prev: +0.0183)
- **4: flag, protocol_service_flag, connection_state, duration**: mean DOS = 0.0932 (Δ from prev: +0.0693)
- **5: flag, protocol_service_flag, connection_state, duration, same_host_rate_x_service**: mean DOS = 0.1149 (Δ from prev: +0.0217)

### Phase 36 Threshold Analysis

The Phase 36 benchmark-repair criterion requires DOS ≥ 0.3. 
**At top-5 ablation: mean DOS (0.1149) < 0.3. Benchmark repair is not achieved through feature removal alone.**
**0/6 pairs** reach DOS ≥ 0.3 after top-5 ablation.

---

*Report generated by Phase 43B — Fingerprint Feature Ablation*
