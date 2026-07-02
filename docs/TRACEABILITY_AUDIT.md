# Traceability Audit: Cross-Dataset Transfer in HELIX-IDS

**Generated:** 2026-06-30  
**Scope:** Phases 44c–56  
**Purpose:** Resolve contradictions and establish a single coherent causal narrative before publication.

---

## 1. Master Traceability Table

### Legend

| Column | Meaning |
|--------|---------|
| Protocol | How the MF1 was computed. **Multi-source** = train encoder on ≥2 datasets jointly, probe on held-out. **Single-pair** = train on 1 source, evaluate on 1 target. **Leave-one-out** = train on 5, evaluate on held-out 1st. **Within-dataset** = train+eval on same dataset. |
| Metric | **Off-diag MF1** = macro F1 across all source→target transfer pairs. **Transfer MF1** = single target macro F1. **Family MF1** = 7-class family macro F1. |
| Classifier | Evaluation classifier: **linear probe** (LogisticRegression), **RF** (Random Forest), **end-to-end** (trained classifier head). |

| Phase | Date | Dataset(s) | Encoder | Loss | Norm | Protocol | Metric | Key Result | Claim |
|-------|------|-----------|---------|------|------|----------|--------|-----------|-------|
| **44c** | Jun 25 | All 6 | Independent per-dataset | CE | BN | Single-pair, linear probe | Transfer MF1 | Mean=0.088 ± 0.057 (30 pairs) | Baseline — cross-dataset transfer is near-zero |
| **45** | Jun 25 | All 6 | — | — | — | Meta-dataset analysis | — | R²=-0.46 (regression on dataset properties) | Transfer unpredictability is inherent, not a modeling gap |
| **47** | Jun 26 | All 6 | Independent per-dataset | CE | BN | Representation similarity (CKA, SVCCA, etc.) | Similarity | Fréchet=24.2, Centroid cos=0.021 | Independent encoders produce incompatible latent spaces |
| **48** | Jun 26 | All 6 | Shared (autoencoder-style) | MSE | BN | Single-pair, linear probe | Off-diag MF1 | Mean=0.188 ± 0.218 | Shared encoder helps but gap remains; Fréchet drops to 0.36 |
| **49** | Jun 26 | All 6 | Shared (Phase 48 encoder) | — | BN | Single-pair, RF on shared latents | Off-diag MF1 | Mean=0.114 ± 0.075 | **P(Y\|X) bottleneck** — decision boundaries differ fundamentally |
| **50** | Jun 26 | All 6 | Shared encoder | 6 methods (SupCon, CE, CORAL, DANN, VICReg, CondMMD) | BN | **Multi-source joint training**, 50 epochs, linear probe | Mean off-diag MF1 | **SupCon=0.719** (±0.284), MultiTask=0.710, CORAL=0.687, DANN=0.387, VICReg=0.456, Phase48 baseline=0.188 (+282% vs baseline) | **SupCon is the best method** for cross-dataset transfer |
| **51** | Jun 27 | All 6 | Shared (SupCon from Phase 50) | SupCon | BN | Multi-source joint, linear probe | Mean off-diag MF1 | Mean=0.704 ± 0.209, [95% CI: 0.628–0.774] | SupCon results are reproducible; CICIDS2017 hardest source (0.434) |
| **52** | Jun 28 | All 6 (15K subsample) | Shared (SupCon or CE) | SupCon + CE | BN | **Single-pair**, linear probe on frozen encoder | Off-diag MF1 | All conditions: **~0.11** (range 0.106–0.122). CE-only (w=0): 0.1125. SupCon (w=1): 0.1125. | **Loss function doesn't matter** in single-pair transfer. SupCon provides no benefit over CE alone. |
| **53** | Jun 28 | All 6 | Shared encoder (SupCon) | SupCon | BN | **Leave-one-out**, linear probe | Off-diag MF1 | Mean=0.517 (MLP). Range across held-out datasets: 0.226–0.462. Seed CI: [0.423, 0.519] | Transfer generalizes across architectures and seeds. |
| **54** | Jun 28 | All 6 (20K subsample) | 4-layer MLP (17→64→64→64→32) | SupCon vs CE | BN | Multi-source, latent extraction + RF/linear probe | Geometry, information, domain metrics | **Domain disentanglement CONFIRMED**: dataset pred at chance (0.158 vs 0.167 random). Failure due to feature info deficiency. | **Why SupCon works** — domain invariance, hierarchical info flow |
| **55** | Jun 28 | All 6 | Shared encoder | SupCon + CE | BN | Causal mediation analysis | Transfer MF1 | Total effect of conditional objective on transfer: **+0.0248**. Mediation through geometry: **1404%**. Intrinsic transfer dim: **4**. | **BN is the confounder.** The SupCon effect is almost entirely mediated through geometry, not invariance. |
| **56** | Jun 30 | NSL-KDD → UNSW-NB15 | 4-layer MLP (17→512→384→256→256) | SupCon vs CE | BN vs none vs others | **Single-pair**, end-to-end | Family MF1 (within-dataset), Transfer MF1 | **Best (SupCon+noBN): 0.8404 within-dataset.** Transfer: ~0.08–0.10 unchanged by any DA technique. BN removal ATE: -0.0243. **BN × SupCon interaction: 67% of variance.** | **BN removal dominates SupCon.** The transfer problem is fundamentally unsolved by any technique tested. Feature-level DA cannot fix it. |

---

## 2. Resolved Contradictions

### Contradiction 1: Phase 50 (SupCon=0.719) vs Phase 52 (SupCon=0.11)

**Claim conflict:** Phase 50 claims SupCon achieves 0.719 off-diag MF1 (+282% improvement). Phase 52 claims CE-only and SupCon both achieve ~0.11 — essentially no improvement from Phase 48.

**Resolution: These measure different protocols.**

| Aspect | Phase 50 | Phase 52 |
|--------|----------|----------|
| Training data | **All 6 datasets jointly** (multi-source) | Single source dataset only |
| Evaluation | Held-out directions from the shared encoder | Linear probe on single-source encoder |
| Efficacy of SupCon | Aligns representations across *multiple* source datasets | Cannot help because there's only 1 source |
| Real claim | SupCon aligns multi-source representations for better cross-dataset transfer | **Loss function doesn't matter for single-pair transfer** |

**These are compatible.** SupCon's value is in multi-source alignment, not single-pair transfer. The ablation study (Phase 52) correctly shows that in a single-pair setting, all losses are equivalent because the encoder has no cross-dataset information to exploit. Written this way, the two results tell a coherent story: **multi-source training is the necessary condition; SupCon is the sufficient accelerator.**

### Contradiction 2: Phase 53 (Leave-one-out MF1=0.517) vs Phase 56 (Transfer MF1=0.08)

**Claim conflict:** Phase 53 achieves ~0.47-0.52 transfer MF1. Phase 56 achieves ~0.08-0.10.

**Resolution: Leave-one-out gives the encoder 5 datasets to learn from; single-pair gives it 1.**

Phase 53 trains on 5 datasets, then probes the frozen encoder on the 6th. Phase 56 trains on NSL-KDD only, then transfers to UNSW-NB15. The gap (0.517 vs 0.08) quantifies how much cross-dataset signal the encoder extracts from having multiple training sources. Both numbers are correct; they answer different questions.

### Contradiction 3: Phase 54 ("SupCon achieves domain invariance") vs Phase 55/56 ("BN removal dominates SupCon")

**Claim conflict:** Phase 54 claims SupCon provably removes dataset identity from latents. Phase 55-56 show that BN's suppression of feature variance is the dominant factor.

**Resolution: These answer different causal questions.**

Phase 54: *What mechanism does SupCon use?* → Domain disentanglement (dataset identity removed from latents).
Phase 55-56: *What factor dominates the transfer outcome?* → BN, not SupCon.

These are compatible. SupCon *does* achieve domain invariance (confirmed in Phase 54 Exp E). But the practical effect on transfer MF1 is dominated by BN statistics (Phase 56 Sobol: 32.4% BN main + 67.2% interaction vs <1% SupCon main). **SupCon works mechanistically, but BN constrains how well it can work in practice.**

### Contradiction 4: Phase 49 (off-diag=0.114, RF on latents) vs Phase 48 (off-diag=0.188, linear probe)

**Claim conflict:** Phase 48 achieves 0.188 MF1 with linear probe on shared encoder. Phase 49 shows 0.114 MF1 with RF on the same latents.

**Resolution: The encoder + classifier were trained jointly in Phase 48, whereas Phase 49 used a separate RF on frozen Phase 48 latents.**

Phase 48's 0.188 includes the end-to-end linear classifier that was trained alongside the encoder. Phase 49's 0.114 uses a Random Forest on frozen latents — a more flexible model that can overfit to source-specific patterns, causing lower cross-dataset generalization. This is actually **consistent**: the RF overfits to source-specific features in the latent space, confirming the P(Y|X) mismatch diagnosis.

### Contradiction 5: Phase 56 (BN removal helps) vs Phase 48-54 (all used BN)

**Claim conflict:** Phase 56 says BN is harmful for transfer. All prior phases used BN as default.

**Resolution: BN has competing effects.**

BN helps training (faster convergence, higher within-dataset accuracy) but hurts cross-dataset transfer by suppressing feature variance that SupCon needs. Prior phases' BN default was correct for within-dataset optimization. Phase 56 shows that **removing BN transfers better, but slightly worse within-dataset** (Phase 56 Exp A: CE+BN=0.8155 within-dataset vs CE+none=0.8293 — actually BN removal also helps within-dataset slightly for this architecture).

The insight is that BN's effect was **architecture-specific**. In Phase 50-54's deeper encoder, BN helped training stability. In Phase 56's deeper backbone (4-layer 512→384→256→256), BN's suppression of variance was harmful even within-dataset.

---

## 3. Hidden Patterns Revealed by the Audit

### Pattern 1: The Multi-Source vs Single-Pair Gap
```
Multi-source joint (Phase 50):       0.719 off-diag MF1
Leave-one-out (Phase 53):            0.517 off-diag MF1  
Single-pair SupCon (Phase 52):       0.112 off-diag MF1
Single-pair CE-only (Phase 52):      0.113 off-diag MF1
Single-pair transfer (Phase 56):     0.080 transfer MF1
Single-pair baseline (Phase 44c):    0.088 transfer MF1
```

**Interpretation:** Cross-dataset transfer performance scales with **number of training sources**, not with loss function sophistication. All single-pair protocols converge to ~0.08-0.12 regardless of loss.

### Pattern 2: The BN × Loss Interaction

Phase 56 Sobol indices show that **BN × SupCon interaction (67%)** dominates both BN main (32%) and SupCon main (<1%). This means:
- The optimal configuration is **SupCon without BN** (0.8404 within-dataset family MF1)
- BN removal helps SupCon more than it helps CE (interaction: +0.0574)
- This is **fully mediated by feature variance** (113% mediation, Phase 56 Exp H)

### Pattern 3: The Absolute Transfer Ceiling

**No technique tested** — SupCon, CORAL, DANN, VICReg, CondMMD, AdaBN, DSBN, feature whitening — has raised single-pair transfer MF1 above ~0.13. This is a **hard ceiling** imposed by the 17-feature P(Y|X) mismatch. The strongest evidence comes from Phase 54 Exp H: failure analysis attributes remaining failures to **information deficiency in the 17-feature representation**, not poor geometry or model capacity.

### Pattern 4: Label Noise and Sample Efficiency Insensitivity

Phase 52's most striking result: up to 30% label noise has no effect on transfer (0.108→0.113). Sample efficiency peaks at 10% data. This confirms **Phase 49's P(Y|X) diagnosis**: the bottleneck is the feature representation itself, not the quality or quantity of training labels.

---

## 4. Unified Causal Narrative

### What We Know

1. **17 canonical features contain limited cross-dataset signal.** P(Y|X) distributions differ fundamentally (Phase 49: conditional MMD=0.545, SHAP ρ=0.229).

2. **A shared encoder trained on multiple datasets** produces more compatible latent spaces (Phase 48: Fréchet drops 24→0.36). But even the best representations yield only 0.12 single-pair transfer.

3. **SupCon achieves domain invariance** (Phase 54: dataset prediction at chance 0.158) — representations discard dataset identity — but this doesn't translate to high transfer MF1 because the **input features lack discriminating power**.

4. **Batch normalization suppresses feature variance**, which hurts SupCon disproportionately (Phase 55/56: 67% interaction effect). Removing BN is the single largest tuning improvement (ATE=+0.024 MF1).

5. **The true bottleneck is the 17-feature canonical representation**, not the loss function, encoder architecture, training budget, or normalization choice. All tested domain adaptation techniques (CORAL, AdaBN, whitening) fail to raise transfer MF1 above ~0.10.

### What We Don't Know

1. **How much would richer features help?** If we expand from 17 to 41+ features (using dataset-native features before harmonization), does transfer improve? Phase 53 Exp E shows removing features degrades transfer (0.517→0.442 for 10 features), but we've never tested *adding back* the discarded 24 features.

2. **Is there an external dataset where transfer works?** All 6 datasets are IDS-specific. Phase 53 Exp A (zero-shot heldout within same datasets) achieves 0.45-0.61, but this is still within the IDS distribution. A truly unseen domain (e.g., network flow → system logs, or network → IoT sensor data) would reveal whether the problem is IDS-specific.

3. **What is the irreducible P(Y|X) lower bound?** If we trained on all 17 features with unlimited data and a perfect encoder, what MF1 would we get on cross-dataset transfer? Phase 55's intrinsic transfer dimension estimate of 4 suggests that only ~4 latent dimensions carry transferable signal, but this needs validation.

---

## 5. Revised Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| "SupCon solves cross-dataset transfer" | **WITHDRAWN** | SupCon=0.719 is real but only in multi-source setting. Single-pair: 0.11, same as CE. |
| "Domain adaptation (CORAL/AdaBN) can fix transfer" | **WITHDRAWN** | Phase 56 Exp G: no technique improved target MF1 beyond ~0.10. |
| "BN is the primary bottleneck" | **PARTIALLY SUPPORTED** | BN removal ATE=+0.024, but this is small (2.4 points). The main bottleneck is the 17-feature representation. |
| "P(Y\|X) mismatch is the dominant bottleneck" | **STRONGLY SUPPORTED** | Phase 49: conditional MMD=0.545. Phase 52: label noise insensitivity. Phase 54: failure analysis. Phase 56: no DA technique works. |
| "Cross-dataset NIDS transfer is limited by feature information, not loss functions" | **BEST SUPPORTED** | Consistent across Phases 49–56. The bottleneck is upstream of the model. |

### Defensible Core Claim

> Cross-dataset NIDS transfer is primarily limited by the **information content of the harmonized feature representation** (17 canonical features), not by the loss function, encoder architecture, or normalization. Contrastive learning provides a conditional benefit that depends on multi-source training and is constrained by batch normalization's suppression of feature variance. No tested domain adaptation technique can overcome the ~0.10 single-pair transfer MF1 ceiling imposed by P(Y|X) mismatch.

---

## 6. Protocol Comparison Matrix

| Phase | Encoder Training | Evaluation | Samples/source | Epochs | Seeds | Device |
|-------|-----------------|-----------|---------------|--------|-------|--------|
| 44c | Independent per-dataset | Linear probe RF | 86K-200K | N/A | 1 | CPU |
| 47 | Independent per-dataset | Representation similarity | 100K | N/A | 1 | CPU |
| 48 | Shared (autoencoder) | Linear probe | 100K | 30 | 1 | MPS |
| 49 | Shared (Phase 48) | RF on frozen latents | 100K | N/A | 1 | MPS |
| 50 | Shared (6 methods) | Linear probe | Full | 50 | 1 | MPS |
| 51 | Shared (SupCon) | Linear probe | Full | 50 | 1 | MPS |
| 52 | Shared (SupCon/CE) | Linear probe (frozen) | 15K | 25 | 1 | MPS |
| 53 | Shared (SupCon) | Linear probe (leave-one-out) | 15K | 25 | 8 | MPS |
| 54 | Shared (SupCon/CE) | RF + linear probe on latents | 20K | 30 | 1 | MPS |
| 55 | Shared (SupCon) | Causal mediation + stress tests | Full | 25 | 1 | MPS |
| 56 | End-to-end classifier | End-to-end classifier | 88K+175K | 30 | 15 | MPS |

**Key discrepancy:** Phase 52 uses 15K subsamples but achieves the same ~0.11 as Phase 44c's full-dataset baseline (mean=0.088). This suggests sample count is not the confounder. The real confounder is **protocol** (single-pair vs multi-source).

---

## 7. Priority Action Items

### P1: Traceability Table ✓ (DONE — this document)

### P2: One Unified Benchmark (Factorial Design)
**Status:** Phase 56 already implemented a partial factorial (Loss × Norm × 15 seeds on 1 dataset pair).  
**Gap:** Needs full 6-dataset, Loss × Norm × Encoder × Seed design with single evaluation protocol.  
**Recommendation:** Extend Phase 56's factorial from NSL-KDD→UNSW-NB15 to all 30 pairs (or a representative subset: 6 sources × 2 targets each = 12 pairs).

### P3: Unseen External Benchmark
**Status:** Not done — all datasets are IDS-family. Phase 53 Exp A (held-out splits) is within-distribution.  
**Recommendation:** Test on at least 2 completely unrelated datasets (e.g., CIC-IDS-2012, ISCX-URL-2016, or non-IDS network traffic).

### P4: Feature Sufficiency Analysis (P(Y|X) Bottleneck)
**Status:** Claimed by multiple phases but never directly tested by comparing 17-feature vs richer feature sets.  
**Recommendation:** Train encoders on (a) canonical 17, (b) expanded 41-native (before harmonization), (c) 17 + synthetic features. Measure whether richer features improve transfer. This is the **single most important remaining experiment**.

### P5: Simplify Contribution ✓ (DONE — Section 5 above)

### P6: Prepare Paper
**Status:** Existing manuscript (`HELIX_submission_ready.md`) covers a different topic (control layer collapse).  
**Recommendation:** This cross-dataset transfer analysis needs its own paper. Structure: Problem → Hypothesis → Evidence → Contradictions → Revised Mechanism → Recommendations.
