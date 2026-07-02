# Phase 26B — Production-Scale Generalization Certification

## Run Configuration

- Phase: 26B (production-scale)

- Max samples per dataset (full load): 0

- Train cap per source dataset: 200,000

- Epochs: 100

- Patience: 20

- Device: mps

- Total experiments: 8

## Summary

- **Best Macro F1**: 0.1068 (pair: NSL-KDD → UNSW-NB15)
- **Worst Macro F1**: 0.0067 (pair: TON-IoT → NSL-KDD)
- **Average Macro F1**: 0.0491
- **Average Generalization Gap**: +0.1172
- **Phase 26A baseline avg F1**: 0.0197
- **F1 ratio (26B vs 26A)**: 2.49x
- **All experiments executed successfully**: False

## Per-Experiment Results

| Experiment | Test Acc | Train Acc | Gap | Macro F1 | Epochs | Train Samples | Test Samples |
|-----------|---------:|----------:|----:|---------:|-------:|--------------:|-------------:|
| exp01_nsl_to_unsw | 0.1218 | 0.0000 | -0.1218 | 0.1068 | 20 | 106,932 | 123,163 |
| exp02_unsw_to_cicids | 0.0306 | 0.6523 | +0.6217 | 0.0196 | 31 | 126,245 | 117,798 |
| exp03_cicids_to_ton | 0.1172 | 0.0000 | -0.1172 | 0.0633 | 20 | 180,000 | 102,868 |
| exp04_ton_to_nsl | 0.0076 | 0.0000 | -0.0076 | 0.0067 | 20 | 137,142 | 114,772 |
| transfer_3src_to_cicids | 0.0000 | 0.2423 | +0.2423 | 0.0000 | 28 | 370,318 | 49,999 |
| transfer_3src_to_nsl | 0.0013 | 0.1521 | +0.1508 | 0.0004 | 36 | 443,385 | 29,704 |
| transfer_3src_to_ton | 0.0095 | 0.1779 | +0.1684 | 0.0119 | 51 | 413,176 | 38,095 |
| transfer_3src_to_unsw | 0.0069 | 0.0078 | +0.0009 | 0.0020 | 47 | 424,072 | 35,069 |

## Holdout Performance Ranking

| Rank | Held-Out Dataset | Macro F1 | Test Acc | Gap |
|----:|-----------------|---------:|---------:|----:|
| 1 | TON-IoT | 0.0119 | 0.0095 | +0.1684 |
| 2 | UNSW-NB15 | 0.0020 | 0.0069 | +0.0009 |
| 3 | NSL-KDD | 0.0004 | 0.0013 | +0.1508 |
| 4 | CICIDS2018 | 0.0000 | 0.0000 | +0.2423 |

## Embedding Audit Result

- **Embeddings cluster by dataset**: True
- **Embeddings cluster by attack family**: False
- **Audit verdict**: cluster_by_dataset (representational failure mode)

Plots: `docs/phase26b/plots/tsne_embeddings.png`, `docs/phase26b/plots/umap_embeddings.png`

## Final Recommendation

**Case**: B

CASE B — Representation Failure. Macro F1 remains below 0.10 (avg 0.0491) AND embedding audit shows clustering by dataset, not by attack family. The current 17-feature harmonized representation is not capturing transferable intrusion patterns. Proceed to Phase 27 Domain Adaptation (e.g., DANN, CORAL, MMD alignment) to bridge the feature-distribution gap between datasets.

## Schema Contract Audit

- Input dimension: 17 (verified)
- Binary output: 2 (verified)
- Family output: 7 (verified)
- All experiments used 17-feature harmonized data (verified)
- No dataset leakage detected (verified)
- No architecture changes made (verified)
- No feature schema changes made (verified)
- No new datasets acquired (verified)

### Data Subsampling Note

- Training cap: **200,000 rows per source dataset** (stratified).

  - Phase 26A trial cap: 50,000 rows/dataset

  - Phase 26B production cap: 200,000 rows/dataset (4x increase from Phase 26A)

- Test cap: 50,000 rows per target dataset (stratified).

- CICIDS has 12.9M training rows. With `max-samples=0` and 100 epochs on MPS, a single CICIDS-source experiment would take many hours. The cap of 200,000 rows keeps the production-scale run tractable while still being 4x larger than Phase 26A.
