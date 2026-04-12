# Repository Layout

This document defines where files belong in this repository so cleanup and future additions stay consistent.

## Canonical Top-Level Placement

- `src/helix_ids/`: Production implementation (canonical package)
- `scripts/`: Operational entrypoints and compatibility wrappers
- `tests/`: Automated tests mirrored to source domains
- `docs/`: Architecture, runbooks, status docs, and reports
- `config/`: YAML and config assets consumed by scripts and package code
- `data/`: Dataset assets and processed artifacts (see `data/README.md`)
- `models/`: Generated model artifacts by target profile
- `results/`: Generated experiment and benchmark outputs
- `checkpoints/`: Training checkpoints and resumable state
- `.github/`: CI workflows and Copilot prompts/instructions

## Scripts Taxonomy

Canonical script locations are domain-based:

- `scripts/training/`: model training pipelines
- `scripts/evaluation/`: evaluation and benchmark pipelines
- `scripts/quantization/`: quantization and quantization benchmarks
- `scripts/governance/`: governance checks and parsing utilities
- `scripts/analysis/`: analysis and audit utilities
- `scripts/data/`: dataset ingestion/preprocessing utilities
- `scripts/deployment/`: deployment entrypoints
- `scripts/maintenance/`: repository maintenance utilities

Top-level files in `scripts/` are compatibility wrappers and should stay thin.

## Root Files Policy

Keep only high-signal project files in repository root:

- `README.md`, `pyproject.toml`, `requirements.txt`
- Core coordination docs: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`

Report-style documents should live under `docs/reports/`.

If a report becomes stable reference material, move it under `docs/`.

## Runtime Hygiene

Transient files should not stay in git history:

- OS/editor noise (`.DS_Store`, caches)
- Local run logs (`*.log`, `train_output.log`)
- Reproducible analysis JSON outputs under `results/`
- Temporary governance gate logs under `results/gates/`

Use cleanup script:

```bash
bash scripts/safe_repo_cleanup.sh        # dry-run
bash scripts/safe_repo_cleanup.sh --apply
```
