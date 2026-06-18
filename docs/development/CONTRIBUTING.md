# Contributing

> Last updated: 2026-06-18

## Current State

This repository is in **formalization mode**. The pipeline is locked for paper reproducibility. No new features, no new scripts, no broad refactors.

**Before contributing, read:**
- [README.md](../../README.md) — Project status and quickstart
- [SYSTEM_ARCHITECTURE.md](../architecture/SYSTEM_ARCHITECTURE.md) — Package boundaries and model scope
- [CODING_STANDARDS.md](CODING_STANDARDS.md) — Code style and conventions
- [TESTING.md](TESTING.md) — Testing requirements and CI gates

## What to Work On

- **Bug fixes** — Prefer minimal, targeted fixes in existing modules
- **Documentation** — Fix stale references, improve clarity
- **Testing** — Increase coverage, add edge cases
- **Performance** — Optimize existing paths without changing behavior

## What to Avoid

- New features or new scripts
- Broad refactors or reorganizations
- Changing runtime pipeline behavior
- Renaming existing APIs

## Development Setup

```bash
# Clone
git clone <repository-url>
cd helix-ids

# Create venv (Python 3.11 required)
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python -c "import torch; import sklearn; import numpy; print('OK')"
```

## Before Submitting

1. Run lint: `ruff check src scripts tests`
2. Run types: `mypy src`
3. Run tests: `PYTHONPATH=src pytest -q`
4. Run architecture tests: `PYTHONPATH=src pytest tests/architecture -q`
5. Verify coverage: `PYTHONPATH=src pytest --cov=src/helix_ids --cov-fail-under=65`

All commands assume `.venv311` is activated and `PYTHONPATH=src`.

## PR Process

1. Branch from `dev` (not `main`)
2. Keep changes minimal and focused
3. Update tests for any behavior change
4. Ensure all CI gates pass
5. Open PR to `dev` with squash-merge
6. After dev review, PR to `main` for release

## Questions?

- Start with `docs/architecture/SYSTEM_ARCHITECTURE.md` for system design
- Check `docs/development/TESTING.md` for test organization
- See `docs/development/CODING_STANDARDS.md` for code conventions
- Review `AGENTS.md` at repository root for AI agent guidance
