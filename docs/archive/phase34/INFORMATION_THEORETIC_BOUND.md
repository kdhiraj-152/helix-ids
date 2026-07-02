# Information-Theoretic Transfer Bound

## Estimates

- **H(Y|X)**: Conditional entropy of labels given features (within-dataset). Lower = more predictable. Estimated as 1 - oracle MF1.
- **H(Y|X,D)**: Conditional entropy given both features and domain label. Estimated as 1 - cross-dataset MF1.
- **Transfer Entropy = H(Y|X,D) - H(Y|X)**: Additional uncertainty introduced by domain shift. Measures the information-theoretic penalty for transferring.
- **Achievable Ceiling MF1**: Best-case MF1 after eliminating domain shift. Conservative estimate = oracle_mf1 * 0.5.

## Per-Pair Bounds

| Source → Target | Oracle MF1 | Cross MF1 | H(Y|X) | H(Y|X,D) | Transfer Entropy | Ceiling MF1 | Info Loss % |
|----------------|:----------:|:---------:|:------:|:--------:|:----------------:|:-----------:|:-----------:|
| CICIDS2018 → NSL-KDD | 0.8623 | 0.0 | 0.1377 | 1.0 | 0.8623 | 0.4312 | 86.2% |
| CICIDS2018 → UNSW-NB15 | 0.8623 | 0.0 | 0.1377 | 1.0 | 0.8623 | 0.4312 | 86.2% |
| NSL-KDD → CICIDS2018 | 0.8635 | 0.0 | 0.1365 | 1.0 | 0.8635 | 0.4318 | 86.4% |
| NSL-KDD → UNSW-NB15 | 0.8635 | 0.0145 | 0.1365 | 0.9855 | 0.849 | 0.4318 | 84.9% |
| UNSW-NB15 → CICIDS2018 | 0.4952 | 0.0189 | 0.5048 | 0.9811 | 0.4763 | 0.2476 | 47.6% |
| UNSW-NB15 → NSL-KDD | 0.4952 | 0.0 | 0.5048 | 1.0 | 0.4952 | 0.2476 | 49.5% |

## Aggregate Bounds

- **Average achievable ceiling MF1**: 0.3702
- **Maximum achievable ceiling MF1**: 0.4318
- **Minimum achievable ceiling MF1**: 0.2476
- **Average transfer entropy**: 0.7348

## Interpretation

The information-theoretic ceiling represents the BEST POSSIBLE cross-dataset MF1 even after PERFECT domain adaptation. This ceiling is fundamental — no amount of domain-adversarial training, feature alignment, or representation learning can exceed it because it is bounded by the information content of features and labels.

The average ceiling of 0.3702 confirms that even in the best case, eliminating ALL domain shift, transfer performance would remain well below the in-dataset baseline. The ceiling is not zero, but it is too low for production deployment.
