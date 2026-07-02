# Storage Savings Estimate — HELIX-IDS Repository

## Current State (Excluding .git and .venv311)

| Category | Size | % of Total |
|----------|------|-----------|
| Source Code (src/) | 3.3 MB | 0.08% |
| Scripts (scripts/) | 6.8 MB | 0.16% |
| Tests (tests/) | 7.6 MB | 0.18% |
| Config (config/) | 104 KB | 0.002% |
| Documentation (docs/) | ~5 MB | 0.12% |
| Models (models/) | ~19 MB | 0.45% |
| Results (results/) | ~1.2 GB | 28.7% |
| Data (data/) | ~2.3 GB | 55.0% |
| Cache/Temp | ~170 MB | 4.1% |
| Other (archive, .github, etc.) | ~20 MB | 0.5% |
| **Total Working Tree** | **~4.27 GB** | **100%** |

## Savings Breakdown

### SAFE Removals (regenerable caches and duplicates)

| Item | Size | Type |
|------|------|------|
| All __pycache__ (.pyc) | 16 MB | Cache |
| .mypy_cache/ | 143 MB | Cache |
| .hypothesis/ | 608 KB | Cache |
| .pytest_cache/ | 256 KB | Cache |
| .ruff_cache/ | 24 KB | Cache |
| .understand-anything/.trash-*/ | 2.8 MB | Cache |
| archive/phase24a/__pycache__/ | 120 KB | Cache |
| Duplicate scaler.pkl (3 copies) | 3.4 KB | Duplicate |
| Duplicate feature_names.json (3 copies) | 2 KB | Duplicate |
| Duplicate checkpoint (phase63) | 4.7 MB | Duplicate |
| Duplicate data (phase52_cache vs multi_dataset_v1) | 492 KB | Duplicate |
| Duplicate CSVs (phase51) | 14 KB | Duplicate |
| Duplicate labels (phase55, expF) | 18 KB | Duplicate |
| Duplicate CSV (phase47, pwcca) | 257 B | Duplicate |
| Duplicate train/val (cicids, multi_dataset_v1) | 138 MB | Duplicate |
| Duplicate log (phase59) | 12 KB | Duplicate |
| **Subtotal SAFE** | **~306 MB** | |

### MEDIUM-RISK Removals (regenerable intermediate artifacts)

| Item | Size | Type |
|------|------|------|
| Phase 55 models (29 checkpoints) | 40 MB | Archivable |
| Phase 58 models (2) | 20 MB | Archivable |
| Phase 48 models (7) | 18 MB | Archivable |
| Phase 52 encoders (12) | 16 MB | Archivable |
| Phase 50 models (12) | 10 MB | Archivable |
| **Subtotal MEDIUM** | **~104 MB** | |

### Compression Opportunities

| Item | Current | Estimated Compressed | Savings |
|------|---------|-------------------|---------|
| Phase 58 latents (.npy) | 260 MB | 60 MB | 200 MB |
| Phase 60 latents (.npy) | 220 MB | 55 MB | 165 MB |
| Phase 55 latents (.npy) | 130 MB | 30 MB | 100 MB |
| Phase 52 latents (.npy) | 120 MB | 30 MB | 90 MB |
| Phase 54 data (.npy) | 100 MB | 25 MB | 75 MB |
| multi_dataset_v1 (.npy) | 550 MB | 150 MB | 400 MB |
| phase52_cache (.npy) | 100 MB | 25 MB | 75 MB |
| Raw CSVs | 1.2 GB | 200 MB | 1,000 MB |
| **Subtotal Compression** | **~2.68 GB** | **~575 MB** | **~2.1 GB** |

## Total Projected Savings

| Scenario | Savings | Final Size | Reduction |
|----------|---------|------------|-----------|
| SAFE only (caches + duplicates) | 306 MB | 3.96 GB | 7.2% |
| SAFE + MEDIUM (archive models) | 410 MB | 3.86 GB | 9.6% |
| SAFE + MEDIUM + COMPRESSION | 2.5 GB | 1.77 GB | 58.6% |
| ALL + git gc | 2.7 GB | 1.57 GB | 63.2% |

## Recommended Target: 1.5–2.0 GB

### Action Plan
1. **Do NOW (SAFE):** Remove caches and duplicates → saves ~300 MB
2. **Do THIS WEEK:** Compress .npy in phase52/54/55/58/60 and data/processed/ → saves ~2.1 GB
3. **Do BEFORE RELEASE:** Archive intermediate checkpoints → saves ~100 MB
4. **Do BEFORE RELEASE:** Run `git gc --aggressive --prune=now` → saves ~100-200 MB

## Per-File Savings Details

| File | Size | Action | Type |
|------|------|--------|------|
| .mypy_cache/ | 143 MB | DELETE | Cache |
| data/processed/multi_dataset_v1/y_val_cicids.npy | 124 MB | DELETE | Duplicate |
| .mypy_cache/ | 143 MB | DELETE | Cache |
| data/raw/ CSVs (combined) | 1.2 GB | xz compress | Compression |
| results/phase58/latents/ (combined) | 260 MB | xz compress | Compression |
| results/phase60/embeddings/ (combined) | ~220 MB | xz compress | Compression |
| results/phase55/latents/ (combined) | ~130 MB | xz compress | Compression |
| All __pycache__ (combined) | 16 MB | DELETE | Cache |
| data/processed/multi_dataset_v1/ | 550 MB | xz compress | Compression |
| .hypothesis/ | 608 KB | DELETE | Cache |
