# Shared-Class-Only Transfer Results

## Methodology

For each source-target pair, remove all attack classes not present in both datasets. Train on filtered source, evaluate on filtered target. This isolates whether non-overlapping classes cause the cross-dataset transfer failure.

## Setup

| Source → Target | Shared Classes | Num Shared | Cross MF1 (full) | Shared MF1 | Improvement |
|----------------|---------------|:----------:|:-----------------:|:----------:|:-----------:|
| NSL-KDD → UNSW-NB15 | ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'] | 5 | 0.0145 | 0.1885 | +0.1740 |
| NSL-KDD → CICIDS2018 | ['Normal', 'DoS', 'Probe', 'R2L'] | 4 | 0.0000 | 0.1527 | +0.1527 |
| NSL-KDD → TON-IoT | [] | 0 | 0.0000 | 0.0000 | +0.0000 |
| UNSW-NB15 → NSL-KDD | ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'] | 5 | 0.0000 | 0.1543 | +0.1543 |
| UNSW-NB15 → CICIDS2018 | ['Normal', 'DoS', 'Probe', 'R2L', 'Generic'] | 5 | 0.0189 | 0.1787 | +0.1598 |
| UNSW-NB15 → TON-IoT | [] | 0 | 0.0000 | 0.0000 | +0.0000 |
| CICIDS2018 → NSL-KDD | ['Normal', 'DoS', 'Probe', 'R2L'] | 4 | 0.0000 | 0.1711 | +0.1711 |
| CICIDS2018 → UNSW-NB15 | ['Normal', 'DoS', 'Probe', 'R2L', 'Generic'] | 5 | 0.0000 | 0.1392 | +0.1392 |
| CICIDS2018 → TON-IoT | [] | 0 | 0.0184 | 0.0000 | -0.0184 |
| TON-IoT → NSL-KDD | [] | 0 | 0.0272 | 0.0000 | -0.0272 |
| TON-IoT → UNSW-NB15 | [] | 0 | 0.0000 | 0.0000 | +0.0000 |
| TON-IoT → CICIDS2018 | [] | 0 | 0.0000 | 0.0000 | +0.0000 |

## Analysis

- **Average improvement**: +0.0755
- **Maximum improvement**: +0.1740
- **Number of pairs that improved**: 6 / 12
- **Number of pairs that worsened**: 2 / 12

## Interpretation

**Moderate improvement** — Removing non-overlapping classes helps somewhat, but transfer remains poor due to covariate shift even on shared classes.
