# Production-Scale Validation

## Configuration

- **Seeds**: [42, 1337, 2026, 7777, 9999]
- **Experiments**: 8 (4 pairwise + 4 holdout)
- **Total runs**: 40
- **Data**: Full datasets (no cap)
- **Epochs**: 200 (patience 30)
- **Architecture**: Exact Phase 28A DANNHelixModel

## Global Results

| Metric | Value |
|-------|------:|
| Mean Macro F1 | 0.1349 |
| Std Macro F1 | 0.0531 |
| 95% CI | [0.1185, 0.1514] |
| Min | 0.0669 |
| Max | 0.2534 |
| Win rate vs Baseline | 90.0% |
| Win rate vs CORAL | 65.0% |

## Per-Experiment

| Experiment | μ F1 | σ F1 | CI95 | Min | Max | Wins/Baseline | Wins/CORAL |
|-----------|-----:|----:|-----:|----:|----:|-------------:|----------:|
| exp01_pairwise_nsl_to_unsw | 0.0943 | 0.0121 | [0.0837, 0.1050] | 0.0778 | 0.1099 | 1/5 | 5/5 |
| exp02_pairwise_unsw_to_cicids | 0.1395 | 0.0026 | [0.1372, 0.1418] | 0.1358 | 0.1430 | 5/5 | 5/5 |
| exp03_pairwise_cicids_to_ton | 0.1465 | 0.0587 | [0.0950, 0.1979] | 0.0876 | 0.2204 | 5/5 | 0/5 |
| exp04_pairwise_ton_to_nsl | 0.1425 | 0.0715 | [0.0798, 0.2052] | 0.0819 | 0.2534 | 5/5 | 2/5 |
| exp05_holdout_3src_to_ton | 0.1392 | 0.0445 | [0.1002, 0.1782] | 0.0871 | 0.2080 | 5/5 | 1/5 |
| exp06_holdout_3src_to_cicids | 0.1902 | 0.0680 | [0.1306, 0.2498] | 0.1164 | 0.2486 | 5/5 | 3/5 |
| exp07_holdout_3src_to_nsl | 0.1549 | 0.0226 | [0.1351, 0.1748] | 0.1210 | 0.1786 | 5/5 | 5/5 |
| exp08_holdout_3src_to_unsw | 0.0721 | 0.0069 | [0.0661, 0.0782] | 0.0669 | 0.0839 | 5/5 | 5/5 |

## Success Criteria

### C1: Average Macro F1 >= 0.12

**Mean**: 0.1349
**Threshold**: 0.12
**Result**: ✅ PASS

### C2: Std deviation <= 0.03

**Std**: 0.0531
**Threshold**: 0.03
**Result**: ❌ FAIL

### C3: DANN beats CORAL in >= 75% of runs

**Win rate vs CORAL**: 65.0%
**Threshold**: 75%
**Result**: ❌ FAIL

### Overall: ❌ SOME FAILURES
