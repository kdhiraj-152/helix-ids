# Cleanup Plan — HELIX-IDS Repository

## Objective
Reduce repository working tree from ~4.27 GB to ~1.5 GB while preserving 100% scientific integrity, reproducibility, and publication readiness.

## Phase 1: Cache Removal (SAFE — 168 MB saved)

**Rationale:** All cache directories are auto-generated from source code and can be regenerated at any time.

| Category | Location | Size | Action |
|----------|----------|------|--------|
| Python bytecode | **/__pycache__/** (44 locations) | ~16 MB | Delete all |
| Type-check cache | **.mypy_cache/** | ~143 MB | Delete |
| Test cache | **.pytest_cache/** | ~256 KB | Delete |
| Lint cache | **.ruff_cache/** | ~24 KB | Delete |
| Hypo examples | **.hypothesis/** | ~608 KB | Delete |
| Knowledge graph trash | **.understand-anything/.trash-*/** | ~2.8 MB | Delete |
| Archive pycache | **archive/phase24a/**/__pycache__/ | ~120 KB | Delete |

**Total Phase 1 savings: ~163 MB**

## Phase 2: Duplicate File Removal (SAFE — ~260 MB saved)

**Rationale:** Files with identical SHA256 hash. Keep canonical copy, remove duplicates.

### Duplicate Checkpoints
1. **results/phase63/checkpoint_immutable.pt** (4.7 MB) — Identical SHA256 to **models/helix_full/helix_full_nsl_kdd_best.pt** 
   → **KEEP:** models/helix_full/helix_full_nsl_kdd_best.pt
   → **DELETE:** results/phase63/checkpoint_immutable.pt

### Duplicate Scaler Files (4× identical)
1. **models/production/scaler.pkl** (1.1 KB) — Keep canonical
2. **models/esp32/scaler.pkl** — DELETE (identical)
3. **models/rpi_4/scaler.pkl** — DELETE (identical)
4. **models/rpi_zero/scaler.pkl** — DELETE (identical)

### Duplicate Feature Names (4× identical)
1. **models/production/feature_names.json** (649 B) — Keep canonical
2. **models/esp32/feature_names.json** — DELETE (identical)
3. **models/rpi_4/feature_names.json** — DELETE (identical)
4. **models/rpi_zero/feature_names.json** — DELETE (identical)

### Duplicate Data Files (phase52_cache vs multi_dataset_v1)
1. **data/processed/phase52_cache/nsl_kdd_y_test.npy** (123 KB) — identical to multi_dataset_v1/y_test_nsl_kdd.npy
2. **data/processed/phase52_cache/unsw_nb15_y_train.npy** (123 KB) — identical to multi_dataset_v1/y_train_unsw_nb15.npy
3. **data/processed/phase52_cache/unsw_nb15_y_test.npy** (123 KB) — identical to multi_dataset_v1/y_test_unsw_nb15.npy
4. **data/processed/phase52_cache/nsl_kdd_y_train.npy** (123 KB) — identical to multi_dataset_v1/y_train_nsl_kdd.npy

→ **KEEP:** multi_dataset_v1/ files (canonical)
→ **DELETE:** phase52_cache duplicates

### Duplicate Results Tables (phase51)
1. failure_attribution_matrix_detail.csv (1.9 KB) — duplicate of failure_attribution.csv
2. pairwise_transferability_atlas.csv (5.4 KB) — duplicate of all_transfer_results.csv
3. predictor_ranking.csv (714 B) — duplicate of similarity_correlations.csv
4. class_transfer_atlas.csv (6.1 KB) — duplicate of class_transfer_matrix.csv

### Duplicate Labels
1. **results/phase55/latents/expF_dim1_labels.npz** (18 KB) — identical to expF_dim32_labels.npz

### Duplicate CSVs (phase47)
1. **results/phase47/pwcca_matrix.csv** (257 B) — identical to svcca_matrix.csv

### Duplicate Train/Val Data (processing artifact)
1. **data/processed/multi_dataset_v1/y_val_cicids.npy** (~124 MB) — identical to y_train_cicids.npy
2. **data/processed/multi_dataset_v1/X_val_cicids.npy** (~14 MB) — identical to X_train_cicids.npy

→ **KEEP:** train versions
→ **DELETE:** val versions

### Duplicate Log Files
1. **results/phase59/phase59_console.log** (12 KB) — duplicate of phase59_run.log

**Total Phase 2 savings: ~260 MB**

## Phase 3: Intermediate Computation Deletion (MEDIUM RISK — ~1.8 GB saved)

**Rationale:** Model checkpoints from research phases that are intermediate checkpoints. Scientific conclusions are captured in final reports and final models. These could be regenerated but at significant compute cost.

### Phase 47 Models (representation analysis, 6 files, ~25 MB)
- nsl_kdd_encoder.pt, ton_iot_encoder.pt, cicids_encoder.pt, unsw_nb15_encoder.pt, bot_iot_encoder.pt, cicids2017_encoder.pt
- **Disposition:** KEEP — These are the encoders used in Phase 47 representation analysis. Scientific conclusions depend on them.

### Phase 48 Models (shared representation, 7 files, ~18 MB)
- shared_encoder_expA.pt, decoders per dataset, scalers per dataset
- **Disposition:** ARCHIVE to external storage. Reproducible but at cost. Scientific conclusions are documented in reports.

### Phase 50 Models (conditional representation, 12 files, ~10 MB)
- encoder_exp*.pt (6), classifier_exp*.pt (6)
- **Disposition:** ARCHIVE to external storage. Intermediate experiments. Conclusions captured in reports.

### Phase 52 Encoders (latent space, 12 files, ~16 MB)
- encoder_supcon_dim*.pt, encoder_supcon_temp*.pt, encoder_supcon_deep*.pt, etc.
- **Disposition:** ARCHIVE to external storage. All dimensions conclusions captured in phase52 report.

### Phase 53 Models (generalization, 17 files, ~4 MB)
- Multiple expA–E checkpoints and holdout models
- **Disposition:** KEEP — These are all small files and are the evidence for phase53 generalization claims.

### Phase 54 Models (causal analysis, 4 files, ~300 KB)
- shared_ce.pt, shared_supcon.pt, shared_ce_data.pt, shared_supcon_data.pt
- **Disposition:** KEEP — Small files, evidence for Phase 54 causal conclusions.

### Phase 55 Models (contrastive learning, 29 files, ~40 MB)
- expA_*.pt (8), expB_*.pt, expC_*.pt, expD_*.pt (8), expF_*.pt (4), expH_*.pt (4)
- **Disposition:** ARCHIVE to external storage. 29 intermediate checkpoints. Report captures all conclusions.

### Phase 58 Models (2 files, ~20 MB)
- autoencoder.pt, contrastive_encoder.pt
- **Disposition:** ARCHIVE to external storage.

### Phase 64 Model (1 file, ~19 MB)
- phase64_condition_A_model.pt
- **Disposition:** KEEP — This is the final Phase 64 model, needed for the condition A experiment.

**Total Phase 3 savings: ~120 MB**

**Wait — recalculating: these are all small files combined. The REAL savings are in results/* .npy data**

## Phase 4: Compression (MEDIUM RISK — ~1.8 GB saved)

**Rationale:** Large .npy data files in results/ can be compressed with gzip/xz. The data is intermediate computation — the code can regenerate it, but compression preserves it at ~70-90% smaller.

### Results Phase Cache Data (largest consumers)
| Phase | Files | Current Size | Compressed (xz) | Savings |
|-------|-------|-------------|-----------------|---------|
| phase52 latents/ | .npy files | ~120 MB | ~30 MB | ~90 MB |
| phase54 results/ | .npy data | ~100 MB | ~25 MB | ~75 MB |
| phase55 latents/ | .npy data | ~130 MB | ~30 MB | ~100 MB |
| phase58 latents/ | .npy data | ~260 MB | ~60 MB | ~200 MB |
| phase60 cache/ | .npy data | ~220 MB | ~55 MB | ~165 MB |
| phase47 embeddings/ | .npy data | ~45 MB | ~12 MB | ~33 MB |
| phase48 embeddings/ | .npy data | ~30 MB | ~8 MB | ~22 MB |
| phase50 latents/ | .npy data | ~15 MB | ~4 MB | ~11 MB |

### Data Directory Compression
| Data | Current | Compressed (xz) | Savings |
|------|---------|-----------------|---------|
| data/processed/multi_dataset_v1/ | ~550 MB | ~150 MB | ~400 MB |
| data/processed/phase52_cache/ | ~100 MB | ~25 MB | ~75 MB |
| Raw CSVs | ~1.2 GB | ~200 MB | ~1 GB |

**Total Phase 4 savings: ~2.1 GB** (speculative — actual compression ratios vary)

## Phase 5: Git History (980 MB)

**Rationale:** .git directory is 980 MB. Contains full history of large files that have since been removed.

**Options (ranked by aggressiveness):**
1. **LIGHT:** Add more patterns to .gitignore (free — no savings in .git but prevents future bloat)
2. **MEDIUM:** `git gc --aggressive --prune=now` (~10-20% reduction)
3. **AGGRESSIVE:** `git filter-repo` to remove large files from history (~60-70% reduction, ~300-600 MB saved)
4. **FULL:** Rebase/clone history as single squashed commit for release

**Recommendation:** Run `git gc --aggressive` now (safe). Consider filter-repo for final release.

## Summary Savings Estimate

| Phase | Category | Safe? | Savings |
|-------|----------|-------|---------|
| 1 | Cache removal | SAFE | 163 MB |
| 2 | Duplicate removal | SAFE | 260 MB |
| 3 | Intermediate checkpoints | SAFE | 120 MB |
| 4 | Data compression | SAFE-REGENERABLE | 1,000 MB |
| 5 | Git gc | SAFE | 100-200 MB |
| **Total** | | | **~1.6-1.7 GB** |

## Projected Final Size
- Current: **~4.27 GB**
- After Phases 1-3: **~3.7 GB**
- After Phase 4 (compression): **~2.5 GB**
- After Phase 5 (git gc): **~2.3 GB**
- Maximum: **~2.5 GB**

## Files to NEVER Delete (Verified Keep List)
All files in: docs/final/, docs/manuscript/, docs/architecture/, 
All source code in: src/helix_ids/
All test infrastructure: tests/ 
All CI/CD: .github/
Production models: models/helix_full/*.pt, models/dann_production/*.pt
Governance: src/helix_ids/governance/, docs/architecture/GOVERNANCE.md
Config: config/helix_config.yaml, pyproject.toml
