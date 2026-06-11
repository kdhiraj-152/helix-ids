# syntax=docker/dockerfile:1
# HELIX-IDS — Reproducible build container
# Phase 10A — Container Reproducibility
#
# Pin to a specific base image digest for reproducibility.
# See docs/reproducibility/CONTAINER_REPRODUCIBILITY.md for verification.

FROM python:3.11.11-slim-bookworm@sha256:cb9ccee5faf02a1d2d90af52c4f6d532e10ecf4b01fbaf117e08a0f60e30d9a6 AS base

LABEL org.opencontainers.image.source="https://github.com/kdhiraj/helix-ids"
LABEL org.opencontainers.image.description="HELIX-IDS: Hierarchical Edge-optimized Lightweight Intrusion eXpert"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="1.0.0"

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy only the dependency files first for layer caching
COPY requirements-lock.txt pyproject.toml ./

# Install runtime dependencies from the hash-pinned lockfile
RUN set -eux; \
    pip install --no-cache-dir -r requirements-lock.txt && \
    echo "Dependencies installed successfully"

# --- Test layer (build target: test) ---
FROM base AS test

# Copy the entire source tree
COPY . .

# Install test-only dependencies
RUN set -eux; \
    pip install --no-cache-dir pytest pytest-cov pytest-mock ruff mypy bandit && \
    echo "Test dependencies installed"

# Run the test suite
RUN set -eux; \
    python -m pytest -q --cov=src/helix_ids --cov-report=term-missing --cov-fail-under=65 -x --tb=short && \
    echo "All tests passed"

# Run Ruff
RUN set -eux; \
    ruff check src scripts tests && \
    echo "Ruff: 0 violations"

# Run Mypy
RUN set -eux; \
    mypy src && \
    echo "Mypy: 0 errors"

# --- Production layer ---
FROM base AS production

# Copy only production source code
COPY src/ src/
COPY scripts/ ./scripts/

# Verify lockfile installation is intact
RUN set -eux; \
    python -c "import sys; sys.path.insert(0, '/app/src'); from helix_ids import HELIXIDS; print('HELIX-IDS import OK')" && \
    echo "Production verification passed"

CMD ["python", "-c", "print('HELIX-IDS container ready')"]
