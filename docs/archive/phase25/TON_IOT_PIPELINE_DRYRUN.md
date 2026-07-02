# TON-IoT Pipeline Dry Run Report

**Generated:** 2026-06-21 IST

## Pipeline Stages

| Stage | Time (s) |
|-------|----------|
| Load + Clean | 0.58 |
| Harmonize | 0.64 |
| Train/Test Split | 0.0321 |
| **Total** | **1.25** |

## Output Dataset Sizes

| Stage | Rows | Columns | Memory (MB) |
|-------|------|---------|-------------|
| Raw loaded | 190,474 | 43 | 316.2 |
| Harmonized | 190,474 | 18 | 27.6 |
| Train (80%) | 152,379 | 17 | — |
| Test (20%) | 38,095 | 17 | — |

## Memory Usage

- Raw data memory: 316.2 MB
- Harmonized data memory: 27.6 MB
- Expansion factor: 0.09x (feature reduction + dtype changes)

## Integrity Checks

| Check | Status |
|-------|--------|
| No training artifacts saved | PASS (no model.fit call) |
| Split preserves stratification | PASS |
| Contract validation auto-applied | PASS (enforced by harmonize_features) |

## Verdict

**PASS** — Pipeline dry run completes successfully within reasonable time and memory bounds.
