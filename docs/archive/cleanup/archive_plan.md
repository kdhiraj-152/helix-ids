# Archive Plan — HELIX-IDS Repository

## Purpose
Archive intermediate research artifacts to external storage (external HDD or cloud) while retaining the ability to reproduce all scientific conclusions.

## Archive Target
- Location: External storage (e.g., `/Volumes/Backup/helix_ids_archive/`)
- Format: `.tar.xz` per phase directory
- Verification: SHA256 manifest for each archive

## Phase-by-Phase Archive Instructions

### Phase 47 — Representation Analysis (KEEP mostly, archive scalers)
- **KEEP in repo:** reports, encoder models, embedding plots
- **ARCHIVE:** scaler .pkl files (6 files, ~15 KB total — but can be regenerated)
- Rationale: Scaler files are trivially regenerated from data

### Phase 48 — Shared Representation (ARCHIVE checkpoints)
- **Archive path:** `/Volumes/Backup/helix_ids_archive/phase48/models.tar.xz`
- **Files to archive:** All .pt and .pkl files in results/phase48/models/
- **Keep in repo:** reports, CSV results, figures

### Phase 50 — Conditional Representation (ARCHIVE checkpoints)
- **Archive path:** `/Volumes/Backup/helix_ids_archive/phase50/models.tar.xz`
- **Files to archive:** All .pt files in results/phase50/models/
- **Keep in repo:** reports, CSV results

### Phase 52 — Latent Space Analysis (ARCHIVE encoders)
- **Archive path:** `/Volumes/Backup/helix_ids_archive/phase52/models.tar.xz`
- **Files to archive:** All encoder .pt files in results/phase52/latents/
- **Keep in repo:** reports, CSV results, canonical encoder (encoder_supcon_dim32.pt)

### Phase 53 — Generalization (KEEP — small files, directly support conclusions)
- **Archive:** Nothing — all files are under 5MB total and are direct evidence

### Phase 54 — Causal Analysis (KEEP — small files)
- **Archive:** Nothing — all files are under 300KB total

### Phase 55 — Contrastive Learning (ARCHIVE non-best checkpoints)
- **Archive path:** `/Volumes/Backup/helix_ids_archive/phase55/models.tar.xz`
- **Files to archive:** All .pt in results/phase55/models/ except expA_supcon.pt (best)
- **Keep in repo:** expA_supcon.pt (best checkpoint), all CSV reports

### Phase 58 — Representation Analysis (ARCHIVE models)
- **Archive path:** `/Volumes/Backup/helix_ids_archive/phase58/models.tar.xz`
- **Files to archive:** Both .pt files
- **Keep in repo:** Reports, CSV data, figures

### Phase 58 Latents (COMPRESS)
- **Archive path:** `/Volumes/Backup/helix_ids_archive/phase58/latents.tar.xz`
- **Action:** compress .npy files with xz, keep compressed versions in repo, archive uncompressed

### Phase 60 Latents (COMPRESS)
- **Archive path:** `/Volumes/Backup/helix_ids_archive/phase60/latents.tar.xz`
- **Action:** compress .npy files with xz, keep compressed versions in repo

## Data Directory

### data/processed/multi_dataset_v1/ (COMPRESS or ARCHIVE)
- This is a *regenerable* preprocessing artifact
- **Option A** (recommended): Keep compressed (xz) versions in repo
  - Run: `tar -cJf data/processed/multi_dataset_v1.tar.xz -C data/processed multi_dataset_v1/`
  - Then delete: `rm -rf data/processed/multi_dataset_v1/`
  - To restore: `tar -xJf data/processed/multi_dataset_v1.tar.xz -C data/processed/`
- **Option B** (more conservative): Keep as-is
- Archives to keep in repo for reproducibility

### data/processed/phase52_cache/ (DELETE after archiving)
- This data is duplicated in multi_dataset_v1/
- Archive to external storage, then delete from repo

## External Archive Manifest
Create `cleanup/external_archive_manifest.json` summarizing all archived files with their SHA256 hashes.

## Archive Verification
After archiving, run:
```bash
cd /Volumes/Backup/helix_ids_archive/
for f in *.tar.xz; do
  echo "Verifying $f..."
  sha256sum "$f" >> helix_ids_archive_sha256.txt
done
```

This ensures all archived artifacts are traceable and verifiable.
