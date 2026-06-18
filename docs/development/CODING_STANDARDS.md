# Coding Standards

> Last updated: 2026-06-18

## Language and Runtime

- **Python 3.11** — Must match CI runner (ubuntu-24.04) and venv (`.venv311/`)
- **PyTorch ≥2.0.0** — CPU/CUDA deployment
- **All scripts use `python3`** (not `python`)

## Code Style

- **Formatter**: Ruff formatter (enforced in CI)
- **Line length**: Configured and enforced via ruff (E501 disabled in favor of formatter)
- **Naming**: `snake_case` for all functions, methods, variables
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Imports**: Sorted via ruff (I ruleset), stdlib first, then third-party, then local

## Type Annotations

- **Required** for all public functions in `src/helix_ids/`
- **Mypy** enforced as blocking CI gate for `src/` only
- **Scripts** (`scripts/`) have fewer annotations — acceptable for paper cycle
- Use `Protocol`/`ABC` for delegate interfaces where appropriate
- `__init__.py` should include `__all__` for explicit exports

## Linting

- **Ruff** — Active with rulesets: E, W, F, I, B, C4, UP
- **Expected violations**: 0 in CI
- **Pre-commit**: Not configured (CI gate replaces it)

## Testing

- **pytest** for all testing
- Tests mirror `src/` structure under `tests/`
- No feature implementation in test files
- Prefer `tmp_path` fixture over hardcoded paths
- Property-based testing with Hypothesis for invariants
- Coverage threshold: ≥65% on `src/helix_ids/`

## Package Structure

- Production code: `src/helix_ids/`
- Operational scripts: `scripts/` (domain-organized in subdirectories)
- Tests: `tests/`
- Configuration: `config/`

### Script Domains

| Domain | Purpose |
|--------|---------|
| `scripts/training/` | Model training pipelines |
| `scripts/evaluation/` | Validation and benchmark pipelines |
| `scripts/operations/` | Live service and staging/production |
| `scripts/data/` | Ingestion, preprocessing |
| `scripts/deployment/` | Deployment entrypoints |
| `scripts/benchmarks/` | Performance/load/soak testing |
| `scripts/ci/` | CI validators and checks |
| `scripts/maintenance/` | Repo and environment maintenance |

### Script Wrappers

Root-level `scripts/*.py` must remain thin wrappers that delegate with `os.execv`.

## Forbidden Patterns

- **No `eval()`/`exec()`/`__import__()`** in governed source files (enforced by `ast_validator.py`)
- **No imports with `src.` prefix** from within `src/` (enforced by architecture tests)
- **No reverse dependencies** from `src/` → `scripts/`
- **No silent schema changes** — schema registry must be updated

## Git Conventions

- **Branch strategy**: `main` (protected, PR required) + `dev` (light protection)
- **Merge**: Squash-merge from dev → main via PR
- **Stale branches**: Cleaned after merge
- **Commit messages**: Descriptive, not necessarily conventional-commits style

## Files to Never Commit

- `.venv*/` — Virtual environments
- `models/` — Trained model artifacts
- `data/` — Raw datasets
- `results/` — Experiment outputs
- `benchmarks/` — Performance artifacts
- `artifacts/` — Runtime capture logs
- `*.npy`, `*.pt`, `*.pth` — Binary artifacts
- `.code-review-graph/` — Code review graph cache
