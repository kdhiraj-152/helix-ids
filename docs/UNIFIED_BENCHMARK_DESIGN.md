# Priority 2: Unified Factorial Benchmark — Design Specification

**Goal:** One benchmark where every factor is tested under the exact same protocol, producing a single clean analysis.

**From:** The existing Phase 56 framework already implements a 2×2 factorial (Loss × Norm) with 15 seeds on NSL-KDD→UNSW-NB15.  
**Extension needed:** Multi-source (all 6 datasets), multi-loss, multi-norm, multi-encoder, cross-dataset transfer.

---

## Design

### Factors

| Factor | Levels | Rationale |
|--------|--------|-----------|
| **Loss** | CE, SupCon, ArcFace | CE = baseline, SupCon = Phase 50's best, ArcFace = angular margin variant not yet tested |
| **Norm** | BN, RMSNorm, None | From Phase 56, these are the only three that reach ≥0.82 MF1; all others collapse |
| **Encoder** | Small (17→32→2), Large (17→512→384→256→256) | Small = Phase 52 optimal (681 params, highest transfer), Large = Phase 56 production backbone |
| **Source datasets** | All 6 (multi-source joint training) | The setting where SupCon actually shows benefit (Phase 50: 0.719) |
| **Target datasets** | All 6 (leave-one-out evaluation) | Full transfer matrix, not single-pair |
| **Seeds** | 5 (42, 123, 512, 4096, 9999) | Sufficient to estimate between-seed variance (Phase 53: stable across 8 seeds) |

### Total Design

3 Loss × 3 Norm × 2 Encoder × 5 Seeds = **90 trained models**

Each trained on all 6 datasets jointly (multi-source), evaluated on all 30 transfer pairs.

Total transfer evaluations: 90 × 30 = 2,700 MF1 measurements.

### Analysis

Mixed-effects model:

```
MF1 ~ Loss × Norm × Encoder + (1 | Source Dataset) + (1 | Target Dataset) + (1 | Seed)
```

Variance decomposition via:
- Sobol sensitivity (Phase 56 H): which factors drive transfer variance
- Mediation analysis (Phase 55 G): does feature variance mediate the Norm × Loss interaction?
- Bootstrap CIs on all pairwise comparisons

### Expected Outcomes

1. **Confirms** that Loss × Norm interaction dominates (Phase 56 Sobol: 67%)
2. **Quantifies** encoder size effect — does the large encoder extract more cross-dataset signal?
3. **Extends** from single-pair (Phase 56: NSL-KDD→UNSW-NB15) to full 30-pair matrix
4. **Tests** ArcFace — if it behaves like SupCon, the effect is contrastive-learning-class-wide; if not, it's SupCon-specific

### Implementation Note

Phase 56's `phase56_main.py` already has:
- Configurable model builder with 7 normalization types
- SupCon loss implementation
- Cross-dataset evaluation pipeline
- Cross-seed replication framework

The extension is:
1. Add ArcFace loss (angular margin)
2. Add small encoder variant
3. Change training from single-pair to multi-source (all 6 datasets)
4. Change evaluation from single-pair to full leave-one-out transfer matrix
5. Wrap in 5-seed loop
