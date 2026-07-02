# Priority 5: Simplified Contribution Narrative

## Core Claim (Defensible)

> **Cross-dataset NIDS transfer is primarily limited by feature-level information constraints, not by loss function choice or domain adaptation architecture. Contrastive learning provides a conditional benefit that depends on multi-source training and is mediated by batch normalization's suppression of feature variance. No tested technique overcomes the ~0.10 single-pair transfer MF1 ceiling.**

---

## What We Can Defend (with evidence)

| Claim | Evidence | Contradictory? |
|-------|----------|----------------|
| **Multi-source joint training is necessary for any transfer signal.** | Phase 50: 0.719 multi-source vs Phase 52: 0.112 single-pair. 6× improvement. | Not contradicted. |
| **SupCon improves multi-source transfer over CE** | Phase 50: SupCon=0.719 vs CE multi-task=0.710. Phase 51: reproduced at 0.704. | Phase 56 shows the improvement is small (ATE=+0.024, Sobol main effect <1%). The benefit is fragile. |
| **BN removal helps SupCon more than CE** | Phase 56: SupCon+noBN = 0.840 vs SupCon+BN = 0.769 (within-dataset). Interaction = 67% of variance. | Compatible with Phase 50-54's BN default. Those were correct for within-dataset optimization but their BN usage suppressed transfer. |
| **The P(Y\|X) bottleneck dominates** | Phase 49: Conditional MMD=0.545. Phase 52: noise-insensitive. Phase 56: no DA technique works. Single-pair ceiling ~0.10. | Not contradicted — this is the unifying finding. |
| **Domain adaptation fails for the right reason** | H-divergence <0.1% of target bound (Phase 36, earlier analysis). P(Y\|X) mismatch dominates source error. | Compatible with Ben-David bound analysis. |

## What We Must Qualify

| Claim | Problem | Revised Statement |
|-------|---------|------------------|
| "SupCon solves cross-dataset transfer" | Phase 52 contradicts; Phase 55-56 show mediation | "SupCon improves multi-source alignment but single-pair transfer is unchanged regardless of loss function" |
| "CORAL improves transfer" | Phase 56 Exp G: CORAL=0.087 (same as baseline) | "Feature-level domain alignment cannot compensate for P(Y\|X) mismatch" |
| "AdaBN fixes cross-dataset BN mismatch" | Phase 56 Exp G: AdaBN=0.092 | "BN statistics are not the primary bottleneck" |
| "Domain-invariant features enable transfer" | Phase 54: domain invariance achieved (chance-level dataset ID) but transfer still 0.11 | "Domain invariance is necessary but not sufficient; information content of shared features determines transfer ceiling" |

---

## Killer Figures (Top 3 Most Important)

1. **Phase 52's "Null Result" plot** (Loss function × Transfer MF1 — all losses produce flat ~0.11). This single figure overturns the assumption that better loss functions = better transfer.

2. **Phase 55-56 Sobol decomposition** (67% BN × Loss interaction, 32.4% BN main, <1% SupCon main). Shows the dominant factor is not what anyone expected.

3. **Phase 44c transfer matrix** (30-pair heatmap, mean=0.088). Establishes the irreducible floor. Every later method's improvement is measured from this baseline.

---

## Abstract (One Paragraph)

> Cross-dataset transfer learning for network intrusion detection is widely believed to be limited by model capacity and domain adaptation architecture. Through a systematic 12-phase investigation spanning contrastive learning (SupCon, ArcFace), normalization (batch, RMS, layer, instance), domain alignment (CORAL, DANN, AdaBN), and causal mediation analysis, we demonstrate that the dominant constraint is the feature representation itself — not the loss function, encoder architecture, or normalization strategy. Specifically: (1) All 30 single-pair transfer directions converge to ~0.10 MF1 regardless of loss function; (2) Batch normalization suppresses feature variance critical for contrastive learning, explaining 67% of variance through its interaction with SupCon; (3) Domain-invariant representations are achievable (dataset identity at chance) but insufficient because P(Y|X) mismatch across the 17-feature canonical representation imposes a fundamental ceiling; (4) No tested domain adaptation technique raises single-pair transfer beyond ~0.10. We conclude that future progress requires richer feature representations or datasets with overlapping label spaces, not more sophisticated adaptation methods.
