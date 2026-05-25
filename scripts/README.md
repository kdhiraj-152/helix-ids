# Scripts Organization

This repository uses domain-based script organization with thin root wrappers for backwards compatibility.

## Canonical Rule

- Canonical executable scripts live under domain folders inside `scripts/`.
- Root-level `scripts/*.py` files must remain thin wrappers only.
- Wrappers should delegate with `os.execv` to preserve CLI behavior and exit codes.

## Domains

- `scripts/training/`: model training pipelines
- `scripts/evaluation/`: validation and benchmark pipelines
- `scripts/quantization/`: quantization/export/latency benchmarking
- `scripts/governance/`: governance and promotion guard checks
- `scripts/analysis/`: diagnostics and deep-dive analyses
- `scripts/data/`: ingestion, preprocessing, and dataset preparation
- `scripts/deployment/`: runtime and deployment entrypoints
- `scripts/operations/`: live service and staging/production operations checks
- `scripts/maintenance/`: repository and environment maintenance tasks

## Naming Convention

- Use `snake_case.py` for all scripts.
- Prefer action-oriented prefixes:
  - `train_*`, `evaluate_*`, `benchmark_*`, `validate_*`, `check_*`, `parse_*`, `prepare_*`, `process_*`, `deploy_*`, `quantize_*`.
- Keep version suffixes explicit when needed (`*_v2`, `*_v3`) and avoid hidden semantic drift.

## Adding a New Script

1. Place the canonical script in the correct domain folder.
2. Add a root compatibility wrapper only if legacy command paths are already used.
3. Document the script in `README.md` when it affects standard workflows.
4. Add or update tests under mirrored domain paths in `tests/`.

## Wrapper Template

```python
#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/<domain>/<name>.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "<domain>", "<name>.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
```

## Wrapper Governance

- Run `python scripts/governance/check_root_wrapper_consistency.py` to verify all root wrappers:
  - use `os.execv` delegation,
  - include the canonical wrapper docstring convention,
  - and point to existing canonical scripts.
- See migration mapping in `docs/reports/SCRIPT_MIGRATION_MATRIX.md`.
