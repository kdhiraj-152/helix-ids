# SCHEMA CONTRACT

## Active canonical paths
- `src/helix_ids/data/feature_harmonization.py` defines the 17-feature canonical order and enforces exact feature ordering.
- `src/helix_ids/data/multi_dataset_loader.py` validates harmonized frames against the canonical order and writes `feature_columns.npy` plus `canonical_contract.json`.
- `scripts/training/prepare_canonical_artifacts.py` verifies the canonical artifact bundle before release.

## Legacy compatibility paths
- None retained in the canonical artifact flow. Legacy feature padding and silent reordering were removed from the loader path.

## Dead paths
- 19-feature and 41-feature schema assumptions in the canonical loader/runtime path.
- Padding/truncation logic for missing cross-dataset features.
- Implicit column sorting used as a substitute for canonical order.

## Contradictory paths
- Generic export helpers still exist in `src/helix_ids/utils/export.py`, but their defaults now point to the canonical 17/2/7 contract.
- Older artifacts outside the current release flow may still exist on disk; they are rejected by the runtime and release validation.
