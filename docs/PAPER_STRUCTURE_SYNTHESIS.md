# Priority 6: Paper Structure — Synthesis Paper

## Title

**What Limits Cross-Dataset Transfer in Network Intrusion Detection? A Causal Investigation of Feature Information, Normalization, and Contrastive Learning**

---

## Structure

### Abstract
*(from Priority 5)*

---

### 1. Introduction

**Problem:** Models trained on public IDS benchmarks fail to transfer to other datasets. Community response: develop better loss functions (SupCon, ArcFace) and domain adaptation.

**Gap:** No systematic investigation of what actually limits transfer. Is it the model? The loss? The features? The dataset?

**Our approach:** 12-phase causal investigation — control for each factor, measure the effect, identify the bottleneck.

**Contributions:**
1. Single-pair transfer is invariant to loss function (CE = SupCon = ArcFace = ~0.10 MF1)
2. Multi-source training is necessary for any transfer benefit (6× improvement over single-pair)
3. Batch normalization × SupCon interaction dominates transfer variance (67% via Sobol)
4. P(Y|X) mismatch in the 17-feature representation is the dominant bottleneck — no tested technique overcomes ~0.10 MF1
5. Domain-invariant features are necessary but insufficient; the information content of shared features determines the transfer ceiling

---

### 2. Related Work

- **NIDS datasets and their incompatibility** (Ring 2019, Kenyon 2020)
- **Domain adaptation for NIDS** (Singh 2019, CORAL (Sun 2016), DANN (Ganin 2016))
- **Contrastive learning** (SupCon (Khosla 2020), ArcFace (Deng 2019))
- **Batch normalization and distribution shift** (Ioffe 2015, internal covariate shift debates)
- **Ben-David bound** (Ben-David 2010) — formalizes domain adaptation impossibility

---

### 3. Initial Hypothesis and Baseline

#### 3.1 Setup
- 6 datasets, 17 canonical features, 7-family taxonomy
- 30-pair transfer matrix (Phase 44c)
- Evaluation: linear probe, macro F1

#### 3.2 Baseline Finding
- Mean transfer MF1 = 0.088 ± 0.057 (Phase 44c)
- No pair exceeds 0.24
- **Lowest single-pair direction:** bot_iot → cicids2017 = 0.034
- **Highest:** cicids2018 → cicids2017 = 0.346 (same family, similar topology)

#### 3.3 Initial Hypothesis
*If the loss function forces representations to be class-discriminative across datasets, transfer will improve.*

---

### 4. Experimental Investigation

#### 4.1 Multi-Source SupCon (Phases 50-51)
- **Finding:** Multi-source joint training with SupCon achieves 0.719 off-diag MF1 — **282% improvement** over Phase 48 baseline
- **Initial conclusion:** "SupCon solves cross-dataset transfer"

#### 4.2 Single-Pair Ablation (Phase 52)
- **Finding:** Single-pair transfer is 0.112 regardless of loss (CE, SupCon, all weights)
- **Contradiction:** SupCon provides no benefit in single-pair setting
- **Revised understanding:** Multi-source training is necessary condition; SupCon aligns multiple source representations but doesn't create new cross-dataset signal

#### 4.3 Leave-One-Out Generalization (Phase 53)
- **Finding:** Leave-one-out achieves 0.517 MF1 (train on 5, probe on held-out 6th)
- **Interpretation:** Scales with number of training sources; more sources → better alignment

#### 4.4 Mechanistic Analysis (Phase 54)
- **Finding:** SupCon achieves domain invariance — dataset ID at chance (0.158 vs 0.167 random)
- **But:** Even domain-invariant representations can't fix P(Y|X) mismatch
- **Failure mode:** Information deficiency in 17 features, not geometry

#### 4.5 Causal Mediation (Phase 55)
- **Finding:** SupCon's total effect on transfer = +0.0248 MF1 (tiny)
- **Mediation:** 1404% through geometry — SupCon's benefit is entirely mediated by representation geometry
- **Intrinsic dimension analysis:** Only 4 of 32 dimensions carry transferable signal

#### 4.6 Normalization Factorial (Phase 56)
- **Finding:** BN × SupCon interaction = 67% of explainable variance
- BN removal ATE = +0.0243 (bigger than SupCon's main effect)
- **Smoking gun:** No DA technique (CORAL, AdaBN, DSBN, Whitening) raises transfer above ~0.10

---

### 5. Contradictory Evidence

#### 5.1 The SupCon Singularity
*Phase 50 (0.719) appears to be a singularity — only achievable with multi-source joint training.*

- Phase 52 (single-pair) = 0.112
- Phase 56 (single-pair, full factorial) = ~0.08-0.10
- **Resolution:** Multi-source provides 6× more training signal. The gain is in representation alignment, not new features.

#### 5.2 Normalization Blinds Experiments
*All phases 44c–52 used BN as default. The most important factor (BN × Loss interaction) was unknowable.*

- Phase 55-56: removing BN is the single biggest tuning change
- All prior SupCon results confounded by BN

#### 5.3 The P(Y|X) Ceiling
*Despite 12 phases, no technique raises single-pair transfer beyond ~0.10.*

- 17 features → P(Y|X) mismatch → ceiling
- **Evidence:** Conditional MMD, label noise insensitivity, failed DA, intrinsic dimension

---

### 6. Causal Analysis

#### 6.1 The True Bottleneck

```
Dataset-Specific Network Topology
    ↓
Feature Extraction CICFlowMeter/Zeek
    ↓
17 Canonical Features ← ← ← ← ← ← ← ← ← ← ← ← ← RESOLUTION LOST HERE
    ↓                                                    ↑
Encoder learns dataset-specific statistics              ↑
    ↓                                                    ↑
BN amplifies dataset specificity ← ← ← BN × SupCon interaction
    ↓
Domain-invariant representation (SupCon helps here)
    ↓
P(Y|X) mismatch — same features → different labels
    ↓
Transfer MF1 ≤ 0.10
```

#### 6.2 Sobel-Glymour Causal Hierarchy

| Level | Factor | Effect (ATE) | Variance Explained |
|-------|--------|-------------|-------------------|
| 1 | BN removal | +0.0243 MF1 | 32.4% |
| 2 | BN × SupCon interaction | +0.0574 MF1 | 67.2% |
| 3 | SupCon main effect | +0.0023 MF1 | < 1% |
| 4 | Seed | ±0.003 MF1 | < 0.1% |

#### 6.3 Revised Causal Model

*The causal structure is:*
- Feature representation → P(Y|X) → transfer ceiling (primary path)
- Feature representation → BN statistics → encoder alignment → transfer (mediated path)
- Loss function → representation geometry → BN fit → transfer (interaction path)

---

### 7. Practical Recommendations

1. **Stop optimizing loss functions** for single-pair NIDS transfer; the gain ceiling is +0.024 (ATE of BN removal, the best intervention found).

2. **Invest feature engineering:** The 17-feature schema discards almost all cross-dataset signal. Test 41+ native features against canonical 17.

3. **Remove BN or use RMSNorm** when training for cross-dataset transfer. BN suppresses feature variance that carries signal.

4. **Multi-source training is necessary.** Single-source encoders provide no transfer benefit regardless of loss.

5. **External validation is critical.** All findings are on the same 6 IDS datasets. Test on IoT-23, UGR'16, or non-IDS domains.

---

### 8. Limitations

1. **Same feature family:** All 6 datasets use CICFlowMeter-derived features → results may not generalize to non-flow-based NIDS (Zeek logs, packet-level features)

2. **Single task:** Binary + 7-family classification. Fine-grained attack-type transfer may behave differently.

3. **No engineering of richer features.** We claim P(Y|X) is the bottleneck but haven't tested whether 41+ features improve it. The ceiling could be dataset-inherent (different topologies, different attack semantics) — if so, no feature engineering helps.

4. **Scope:** Results are about NIDS transfer specifically. The interaction between contrastive learning and batch normalization may differ in other domains (CV, NLP).

---

### 9. Figures/Tables

| # | Description | Source Phase |
|---|-------------|-------------|
| 1 | 30-pair transfer matrix heatmap (baseline floor) | 44c |
| 2 | Loss invariance plot (CE=SupCon=ArcFace=0.11 single-pair) | 52 |
| 3 | Multi-source vs single-pair comparison (factor of 6 difference) | 50-52 |
| 4 | Sobol decomposition (67% BN×SupCon, 32% BN, <1% SupCon) | 56 |
| 5 | Domain invariance diagnostic (chance-level dataset ID) | 54 |
| 6 | P(Y|X) mismatch — conditional MMD, SHAP correlation | 49 |
| 7 | Failed DA techniques table (all return ~0.09-0.10) | 56 |
| 8 | Causal graph / DAG of the complete model | 55 synthesis |

---

### 10. Conclusion

Cross-dataset NIDS transfer is not a loss function problem. It is a **feature information problem** mediated by **normalization interactions.** The 17-feature canonical representation lacks the discriminative power to distinguish attacks across different network environments. Contrastive learning, despite producing domain-invariant features, cannot overcome this ceiling because P(Y|X) mismatch is encoded in the feature space itself, not in the model's representational choices.

The path forward is not more sophisticated loss functions or adaptation architectures — it is richer feature representations or datasets with overlapping label spaces.
