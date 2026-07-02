# Dependency Audit

## Overview

Project dependencies defined in pyproject.toml and requirements files.

## Core Dependencies (pyproject.toml)

### Essential (used by production code):
- torch >= 2.0.0 — Core ML framework
- numpy — Array operations
- pandas — Data loading/preprocessing
- scikit-learn — Metrics, preprocessing
- pyyaml — Config loading
- pydantic — Data validation (config models)

### Core Production (heavier):
- onnx, onnxruntime — Model export/serving
- torchvision — Augmentation utilities
- fastapi, uvicorn — REST API serving
- prometheus-client — Metrics monitoring

### Research/Heavy Dependencies:
- matplotlib — Plotting (research artifacts, not production)
- seaborn — Statistical plots
- scipy — Statistical tests
- umap-learn — Dimensionality reduction
- shap — Model interpretation

## Development Dependencies

### Testing:
- pytest — Test runner
- pytest-cov — Coverage
- hypothesis — Property-based testing

### Code Quality:
- ruff — Linting
- mypy — Type checking
- black — Formatting

### CI/Release:
- cosmic-ray — Mutation testing
- slsa-provenance — Supply chain

## Heavy Unused Dependencies

Dependencies that consume significant disk space but have limited usage:

1. **onnx + onnxruntime** (~200MB combined) — Used only in export pipeline. Essential.
2. **umap-learn** (~50MB) — Used in research plotting scripts. Could be dev-only.
3. **shap** (~80MB) — Used in phase49 analysis only.
4. **pyarrow** (~50MB) — Used in data loading. Essential for large datasets.
5. **opencv-python (cv2)** (~50MB) — Minimal usage in the codebase.

## Recommendations

1. Move `shap`, `umap-learn`, `seaborn` to `[tool.poetry.group.dev.dependencies]` or optional.
2. Pin `umap-learn` to dev-only — it's only used in research visualization.
3. Consider `onnxruntime` as optional — not needed for training, only for deployment.

## Unused Packages in Requirements

Checking for declared packages with no imports in the codebase:
- Most packages are used somewhere in the codebase.
- No clearly orphaned packages found.

## Disk Usage

Total site-packages: ~1.8GB (in .venv311)
Primary cost drivers: torch (~500MB), onnxruntime (~200MB), pyarrow (~50MB), opencv (~50MB), grpcio (~40MB), scipy (~40MB), matplotlib (~30MB)
