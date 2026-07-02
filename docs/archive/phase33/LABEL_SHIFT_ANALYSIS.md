# Label Shift Analysis

**Phase 33 — Dataset Incompatibility Proof**
**Created:** 2026-06-24

---

## Overview

Label shift quantifies the degree to which class priors (attack family distributions) differ between datasets. High label shift makes direct transfer impossible because the target distribution of classes differs from what the source classifier learned.

## Methodology

For each dataset, we compute:
- **Class priors**: Raw counts and proportions of each label (0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R, 5=Generic, 6=Backdoor)
- **Total Variation Distance (TVD)**: [0, 1], the maximum difference between probability distributions. TVD = 1 means completely disjoint support
- **Jensen-Shannon Divergence (JSD)**: [0, 1], symmetric measure of distribution difference

## Class Priors

### Raw Counts

| Class | NSL-KDD | UNSW-NB15 | CICIDS2018 | TON-IoT |
|---|---|---|---|---|
| Normal (0) | 57,242 | 59,509 | 15,946,665 | 42,040 |
| DoS (1) | 39,038 | — | — | 38,985 |
| Probe (2) | 9,907 | 26,074 | — | 20,000 |
| R2L (3) | 846 | 28,384 | 87 | 54,962 |
| U2R (4) | 44 | 963 | — | — |
| Generic (5) | — | 34,000 | — | — |
| Backdoor (6) | — | 110 | 286,191 | 34,487 |

### Proportions

| Class | NSL-KDD | UNSW-NB15 | CICIDS2018 | TON-IoT |
|---|---|---|---|---|
| Normal (0) | 53.46% | 39.93% | 98.24% | 22.07% |
| DoS (1) | 36.46% | — | — | 20.47% |
| Probe (2) | 9.25% | 17.49% | — | 10.50% |
| R2L (3) | 0.79% | 19.04% | 0.0005% | 28.86% |
| U2R (4) | 0.04% | 0.65% | — | — |
| Generic (5) | — | 22.81% | — | — |
| Backdoor (6) | — | 0.07% | 1.76% | 18.11% |

### Key Imbalances

1. **CICIDS2018 is 98.24% normal traffic.** Only 1.76% of samples are attack traffic (mostly Backdoor). This makes it an extreme anomaly detection scenario rather than a classification scenario.
2. **NSL-KDD has no Generic or Backdoor classes.** UNSW-NB15 introduced these labels as distinct attack families.
3. **TON-IoT has the most balanced distribution** of any dataset, with Normal (22%), DoS (20%), Probe (11%), R2L (29%), and Backdoor (18%).
4. **UNSW-NB15 is the only dataset with a Generic class** (22.81%), which is a catch-all for attacks that don't fit other categories.
5. **Rare classes vary by 3+ orders of magnitude.** U2R ranges from 44 samples (NSL-KDD) to 963 (UNSW-NB15).

## Pairwise Label Divergence

| Dataset Pair | TVD | JSD | Interpretation |
|---|---|---|---|
| NSL-KDD vs UNSW-NB15 | 0.500 | 0.520 | **Moderate-high shift** — different class taxonomies |
| NSL-KDD vs CICIDS2018 | 0.465 | 0.448 | **Moderate shift** — CICIDS dominated by Normal |
| NSL-KDD vs TON-IoT | 0.474 | 0.439 | **Moderate shift** — different attack proportions |
| UNSW-NB15 vs CICIDS2018 | 0.600 | 0.526 | **High shift** — Generic missing from CICIDS |
| UNSW-NB15 vs TON-IoT | 0.483 | 0.485 | **Moderate-high shift** — class distribution difference |
| CICIDS2018 vs TON-IoT | 0.762 | 0.614 | **Extreme shift** — 98% vs 22% normal |

### CICIDS2018 vs TON-IoT (Worst Case)

TVD = 0.762 is among the largest possible. The root cause:
- CICIDS is 98.24% normal, 1.76% attack
- TON-IoT is 22.07% normal, 77.93% attack
- Any classifier trained on CICIDS would predict "normal" for everything, achieving 98% accuracy but learning nothing about attacks

## Cross-Dataset Label Compatibility

| Class | Present in | Compatible Pairs |
|---|---|---|
| Normal (0) | All 4 | All 6 pairs |
| DoS (1) | NSL-KDD, TON-IoT | 1 pair (NSL↔TON) |
| Probe (2) | NSL-KDD, UNSW, TON-IoT | 3 pairs (NSL↔UNSW, NSL↔TON, UNSW↔TON) |
| R2L (3) | All 4 | All 6 pairs |
| U2R (4) | NSL-KDD, UNSW | 1 pair (NSL↔UNSW) |
| Generic (5) | UNSW only | 0 pairs |
| Backdoor (6) | UNSW, CICIDS, TON-IoT | 3 pairs |

**Only Normal and R2L are present in all 4 datasets.** Every other class is missing from at least one dataset, making uniform 7-class classification impossible without remapping.

## Summary

1. **Label shift is severe across all pairs** (TVD range: 0.47–0.76).
2. The CICIDS2018 vs TON-IoT pair (TVD = 0.76) effectively represents two different problems — one extreme anomaly detection, the other balanced multi-class classification.
3. Label taxonomies are **structurally incompatible**: Generic exists only in UNSW, Backdoor exists in 3/4 datasets, and the base-rate of Normal traffic varies from 22% to 98%.
4. Harmonization to a common label space necessarily loses distinction (e.g., mapping Generic→DoS or Backdoor→R2L) and introduces label noise that upper-bounds achievable performance.

## Plots

- `plots/phase33/label_shift_heatmaps/class_priors.png`
- `plots/phase33/label_shift_heatmaps/label_divergence.png`
