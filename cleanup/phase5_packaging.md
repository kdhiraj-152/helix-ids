# Phase 5: Research Repository Packaging
> Date: 2026-07-02
> Script: `cleanup/release_package.sh`

## Publication-Ready Structure

The release packaging script creates a clean, self-contained snapshot suitable for archival, publication, or transfer.

### Target Layout

```
HELIX-IDS-publication/
├── src/                    # Production code (helix_ids package)
├── scripts/                # Operational scripts (training, serving, evaluation)
├── docs/                   # Full documentation + manuscript
├── models/                 # Production checkpoints only
│   ├── helix_full/         # HelixIDSFull production model
│   ├── dann_production/    # DANN domain-adapted model
│   └── production/         # Current deployment scaler + feature_names
├── config/                 # YAML configuration
├── tests/                  # Complete test suite (76+ tests)
├── datasets/raw/           # Canonical raw datasets only (no processed cache)
├── artifacts/compressed/   # All compressed research archives (.tar.xz)
├── paper/                  # IEEE manuscript
├── reproducibility/        # Reproducibility guide, lockfile, reports
└── cleanup/                # Cleanup execution reports
```

### Usage

```bash
# Default: creates release in releases/helix-ids-publication/
bash cleanup/release_package.sh

# Custom output location
bash cleanup/release_package.sh --dest /path/to/output
```

### What's Excluded

- `data/processed/` — Preprocessing cache (regenerable; archived as `multi_dataset_v1.tar.xz`)
- `results/` — Intermediate research outputs (archived in`cleanup/archives/`)
- `.git/` — History (not needed for publication)
- `.venv311/` — Virtual environment (recreated via requirements-lock.txt)
- `.understand-anything/` — Local code analysis cache
- `archive/phase24a/` — Pre-cleanup archival artifacts
- Build artifacts, cache files, logs
