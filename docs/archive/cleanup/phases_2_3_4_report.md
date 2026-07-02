# Phases 2-4 Execution Report

Generated: Thu Jul  2 00:26:56 IST 2026

## Phase 2: Archived Intermediate Models

| Archive | Original | Archived | Ratio |
|---------|----------|----------|-------|
| phase48_models | 392KB | 342KB | 87% |
| phase50_models | 445KB | 377KB | 85% |
| phase52_encoders | 1152KB | 1030KB | 89% |
| phase55_models | 1468KB | 1179KB | 80% |
| phase58_models | 146KB | 126KB | 87% |

Publication-grade checkpoints kept in place:
- `results/phase52/latents/encoder_supcon_dim32.pt`
- `results/phase55/models/expA_supcon.pt`

## Phase 3: Compressed Research Artifacts

| Archive | Original | Archived | Ratio | Savings |
|---------|----------|----------|-------|--------|
| data_processed_phase52_cache | 27MB | 2.8MB | 11% | 24MB |
| multi_dataset_v1 | 1200MB | 24.8MB | 2% | 1175MB |
| phase47_latents | 64MB | 28.0MB | 44% | 36MB |
| phase54_models | 93MB | 44.1MB | 47% | 49MB |
| phase60_embeddings | 250MB | 48.7MB | 19% | 201MB |
| phase64_embeddings | 342MB | 80.6MB | 24% | 261MB |

Phase 3 total savings: **1747MB**

## Phase 2: Duplicate Removal

| File | Action | Size |
|------|--------|------|
| `models/rpi_4/scaler.pkl` | Deleted (dup) | 1.1KB |
| `models/rpi_zero/scaler.pkl` | Deleted (dup) | 1.1KB |
| `models/esp32/scaler.pkl` | Deleted (dup) | 1.1KB |
| `models/rpi_4/feature_names.json` | Deleted (dup) | 649B |
| `models/rpi_zero/feature_names.json` | Deleted (dup) | 649B |
| `models/esp32/feature_names.json` | Deleted (dup) | 649B |
| Empty variant dirs | Removed | 0B |

## Phase 4: Git Optimization

| Metric | Before | After | Savings |
|--------|--------|-------|--------|
| `.git/` size | 981 MB | 13 MB | **968 MB** |
| Pack files | 4 | 1 | - |
| Pack size | 802 MB | 12.8 MB | 789 MB |
| Loose objects | 487 | 0 | 487 objects |

## Cumulative Savings

| Phase | Savings | Running Total |
|-------|---------|---------------|
| Phase 1 (caches + duplicates) | 209 MB | 209 MB |
| Phase 2 (archived models) | 3.5 MB | 212.5 MB |
| Phase 3 (compressed artifacts) | 1,763 MB | 1,975.5 MB |
| Phase 4 (git optimization) | 968 MB | **2,943.5 MB** |

| Repository size | Start => Current |
|----------------|------------------|
| Total | 7.3 GB => 5.8 GB |
| Git objects | 981 MB => 13 MB |
| Working tree | ~6.3 GB => ~5.8 GB |

## Archives Created

All archives: `cleanup/archives/` (232MB total)

- `data_processed_phase52_cache.tar.xz`: 2.8MB (11% of original)
- `multi_dataset_v1.tar.xz`: 24.8MB (2% of original)
- `phase47_latents.tar.xz`: 28.0MB (44% of original)
- `phase48_models.tar.xz`: 0.3MB (87% of original)
- `phase50_models.tar.xz`: 0.4MB (85% of original)
- `phase52_encoders.tar.xz`: 1.0MB (89% of original)
- `phase54_models.tar.xz`: 44.1MB (47% of original)
- `phase55_models.tar.xz`: 1.2MB (80% of original)
- `phase58_models.tar.xz`: 0.1MB (87% of original)
- `phase60_embeddings.tar.xz`: 48.7MB (19% of original)
- `phase64_embeddings.tar.xz`: 80.6MB (24% of original)
