# Repo Status: End of Phases 2–6

> Generated: 2026-07-02

## Phase 2 — Archive Intermediate Models ✅

Archived intermediate model checkpoints (63 files). Kept only publication-grade checkpoints.

| Archive | Files | Original | Archived | Ratio |
|---------|-------|----------|----------|-------|
| phase48_models | 13 .pt + .pkl | 392 KB | 342 KB | 87% |
| phase50_models | 12 .pt | 445 KB | 377 KB | 85% |
| phase52_encoders | 11 .pt | 1.15 MB | 1.03 MB | 89% |
| phase55_models | 27 .pt | 1.47 MB | 1.18 MB | 80% |
| phase58_models | 2 .pt | 146 KB | 126 KB | 87% |

**Kept in place:**
- `encoder_supcon_dim32.pt` (phase52)
- `expA_supcon.pt` (phase55)

Also removed 6 duplicate files (scaler.pkl, feature_names.json — 4 copies → 1).

## Phase 3 — Compress Research Artifacts ✅

Compressed large .npy, .pt, and preprocessing artifacts into tar.xz archives.

| Archive | Original | Archived | Savings |
|---------|----------|----------|---------|
| phase47_latents | 64 MB | 28 MB | 36 MB |
| phase54_models | 93 MB | 44 MB | 49 MB |
| phase60_embeddings | 250 MB | 49 MB | 201 MB |
| phase64_embeddings | 342 MB | 81 MB | 261 MB |
| phase52_cache | 27 MB | 2.8 MB | 24 MB |
| multi_dataset_v1 | 1,200 MB | 82 MB | 1,118 MB |

**Phase 3 subtotal: ~1,689 MB saved**

## Phase 4 — Git Optimization ✅

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| .git/ size | 981 MB | 13 MB | **-968 MB** |
| Pack files | 4 | 1 | -3 |
| Pack size | 802 MB | 12.8 MB | -789 MB |
| Loose objects | 487 | 0 | -487 |

`git gc --aggressive --prune=now` removed all unreachable objects accumulated from 60+ research phases of commits and deletions.

## Phase 5 — Repository Packaging ✅

Publication-ready snapshot created at `releases/helix-ids-publication/` (279 MB):

| Component | Size |
|-----------|------|
| Code (src + scripts + tests) | 5.6 MB |
| Documentation | 4.7 MB |
| Models (production checkpoints) | 19 MB |
| Config | 104 KB |
| Datasets (raw, canonical) | — (if available) |
| Artifacts (compressed research) | 248 MB |
| Paper | 36 KB |
| Reproducibility | 572 KB |

Packaging script: `cleanup/release_package.sh [--dest /path/to/output]`

## Phase 6 — Long-Term Separation Strategy ✅

Strategy documented at `cleanup/phase6_separation_strategy.md`.

**Recommendation:** Multi-repo separation:
1. **HELIX-IDS** (production) — code, tests, config, models
2. **HELIX-IDS-research** (LFS) — all results/ + archived artifacts
3. **HELIX-IDS-datasets** (LFS) — canonical raw datasets
4. **Release Archives** — versioned publication snapshots

## Cumulative Savings

| Phase | Description | Savings |
|-------|-------------|---------|
| Phase 1 | Caches + safe duplicates | 209 MB |
| Phase 2 | Archived intermediate models | 3.5 MB |
| Phase 3 | Compressed research artifacts | ~1,689 MB |
| Phase 4 | Git aggressive gc | 968 MB |
| **Total** | | **~2,870 MB** |

**Before:** 7.3 GB
**After:**  5.0 GB (includes ~2.2 GB canonical datasets, ~1.8 GB virtual environment)
**Live code footprint:** <50 MB (src + scripts + tests + config + docs)

## Remaining Storage

| Category | Size | Action |
|----------|------|--------|
| data/raw/ | ~2.2 GB | Keep (canonical datasets) |
| .venv311/ | ~1.8 GB | Ignore / recreate |
| results/ | ~1.2 GB | Compressed → archived |
| .git/ | 13 MB | Optimized |
| src/ + scripts/ + tests/ + config/ + docs/ | <20 MB | Untouched |
| models/ | ~19 MB | Untouched |
