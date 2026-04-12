# Data management

## Purpose

This document defines active dataset locations and compatibility behavior for HELIX-IDS workflows.

## Active data directories

- data/raw: canonical location for source snapshots.
- data/processed: transformed datasets for training and evaluation.
- data/nsl_kdd_5class: NSL-KDD focused assets used by current scripts.

## Legacy directories still used by runtime logic

- archive
  - currently stores UNSW-related CSV datasets.
- archive-2
  - currently stores CICFlowMeter CSV datasets (e.g., CICIDS-2018 day-wise CSVs).

These are active directories in this repository and are not symlink placeholders.

## Loader behavior

UnifiedDataLoader in src/helix_ids/data/unified_loader.py resolves datasets from multiple locations. Current behavior includes both data and archive directory families.

## Pre-deployment data policy

Before production release:

1. Define canonical source paths for each dataset family.
2. Update loader path resolution accordingly.
3. Keep compatibility shims only if they are tested and documented.
4. Remove ambiguous path fallbacks that can mask data drift.

## Operational guidance

- For immediate reproducibility, keep archive and archive-2 populated.
- For fresh environments, run scripts/download_datasets.py first.
- If external dataset download fails, synthetic fallback generation can keep training and validation pipelines functional.
