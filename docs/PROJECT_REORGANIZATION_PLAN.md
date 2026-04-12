# Project Reorganization Plan

## Objective

Make repository structure predictable and scalable without breaking existing CLI/script usage.

## Design Principles

- Keep production code canonical under `src/helix_ids/`
- Keep scripts grouped by domain responsibility
- Keep reports in a single documentation archive location
- Preserve backward compatibility for existing commands
- Avoid risky code moves that break imports without wrappers

## Implemented Structure

### Scripts

Scripts are grouped into domain folders:

- `scripts/training/`
- `scripts/evaluation/`
- `scripts/quantization/`
- `scripts/governance/`
- `scripts/analysis/`
- `scripts/data/`
- `scripts/deployment/`
- `scripts/maintenance/`

Top-level files in `scripts/` are now thin wrappers that forward execution to canonical domain paths.

### Docs

- Core docs remain in `docs/`
- Long-form reports moved to `docs/reports/`

## Compatibility Strategy

For each moved script:

- Keep the original top-level script path as a wrapper
- Wrapper delegates execution to canonical location under `scripts/<domain>/`
- Existing automation/CI command lines continue to work unchanged

## Follow-up Hardening

- Update CI and internal docs to gradually adopt canonical domain paths
- Keep wrappers for a deprecation window
- Remove wrappers only after all references migrate

## Verification Steps

Use lightweight checks after reorganizing:

```bash
python scripts/training/train_helix_ids_full.py --help
python scripts/quantization/benchmark_helix_quantization.py --help
bash scripts/maintenance/safe_repo_cleanup.sh
```

## Non-Goals

- No package namespace refactor in `src/helix_ids/`
- No data directory migration in this step
- No model artifact relocation in this step
