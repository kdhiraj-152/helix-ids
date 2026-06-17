# Dependency Lockdown — HELIX-IDS RC2

> Phase 21 deliverable. Provides reproducible, verifiable dependency
> management for production deployments.

## Lockfile Inventory

| File | Scope | Generated From |
|------|-------|----------------|
| `requirements.lock` | Core runtime | `pyproject.toml` (core deps) |
| `requirements-dev.lock` | Core + dev | `pyproject.toml` (`--extra=dev`) |
| `requirements-all.lock` | Core + dev + training + deployment | `pyproject.toml` (`--extra=dev --extra=training --extra=deployment`) |

## Generation

All lockfiles are generated with `pip-compile` (from `pip-tools`):

```bash
# Core runtime
pip-compile --generate-hashes --output-file=requirements.lock pyproject.toml

# With dev extras
pip-compile --generate-hashes --output-file=requirements-dev.lock --extra=dev pyproject.toml

# Full stack (dev + training + deployment)
pip-compile --generate-hashes --output-file=requirements-all.lock \
    --extra=dev --extra=training --extra=deployment pyproject.toml
```

## Installation

```bash
pip install -r requirements.lock              # production
pip install -r requirements-dev.lock           # development
pip install -r requirements-all.lock           # full stack
```

## Hash Verification

All lockfiles include SHA256 hashes. pip verifies hashes at install time:

```bash
pip install --require-hashes -r requirements.lock
```

If a single hash is wrong, pip will refuse to install any package from that
lockfile, preventing supply-chain attacks.

## Drift Detection

The test `tests/architecture/test_dependency_lockdown.py` detects drift by:

1. Running `pip list --format=json` to snapshot the current environment.
2. Comparing installed versions against the pinned versions in the lockfile.
3. Failing if any installed package version differs from its pinned version.

Run drift detection manually:

```bash
pytest tests/architecture/test_dependency_lockdown.py -v
```

## CI Integration (Recommended)

Add a step to your CI workflow:

```yaml
- name: Verify dependency lock
  run: |
    pip install -r requirements.lock
    pytest tests/architecture/test_dependency_lockdown.py -v
```

## Regeneration

When adding or updating dependencies in `pyproject.toml`:

1. Update `pyproject.toml` with the new dependency and version constraint.
2. Regenerate lockfiles:
   ```bash
   pip-compile --generate-hashes --output-file=requirements.lock pyproject.toml
   pip-compile --generate-hashes --output-file=requirements-dev.lock --extra=dev pyproject.toml
   pip-compile --generate-hashes --output-file=requirements-all.lock \
       --extra=dev --extra=training --extra=deployment pyproject.toml
   ```
3. Commit all three lockfiles along with the `pyproject.toml` change.

## Security

- All packages pinned to exact versions (no ranges).
- Hashes provided for every package.
- `--require-hashes` mode prevents tampering.
- Compatible with air-gapped / offline deployment.
