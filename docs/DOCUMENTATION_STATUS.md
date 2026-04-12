# Documentation Status

## Canonical Docs

These documents are the current source of truth:

- `README.md`: high-level usage and quick-start commands
- `docs/ARCHITECTURE.md`: implementation architecture and component boundaries
- `docs/FEATURE_ENGINEERING.md`: active feature harmonization contract
- `docs/REPOSITORY_LAYOUT.md`: file placement and cleanup policy
- `docs/PHASE4_PHASE5_STATUS.md`: concise phase/status tracker
- `docs/PROJECT_REORGANIZATION_PLAN.md`: comprehensive reorganization strategy
- `docs/reports/`: archived or long-form implementation/recovery reports

## Validation Checklist

When updating docs, ensure consistency with:

- `src/helix_ids/data/feature_harmonization.py`
- `src/helix_ids/models/helix_ids_full.py`
- `scripts/train_helix_ids_full.py`
- `scripts/quantize_helix_lite.py`
- `scripts/quantize_helix_micro.py`

## Known Rules

- Avoid hardcoding legacy claims (for example binary-only architecture)
- Keep feature dimension references aligned to the active 31-feature contract
- Keep deployment and benchmark commands aligned with script names in `scripts/`
- Keep root-level policy aligned with `docs/REPOSITORY_LAYOUT.md`
- Keep report documents under `docs/reports/` unless actively drafted
