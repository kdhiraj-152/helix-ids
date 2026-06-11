# Container Reproducibility & Trust

**Document:** CONTAINER_REPRODUCIBILITY.md
**Version:** 2.0
**Last Updated:** 2026-06-12
**Phase:** 10B — Artifact Provenance, Container Trust & Compliance Hardening

This document describes the container reproducibility controls and
container signing/trust verification for the HELIX-IDS project.

---

## 1. Base Image

The Dockerfile uses a pinned, digest-addressed base image:

```
python:3.11.11-slim-bookworm@sha256:cb9ccee5faf02a1d2d90af52c4f6d532e10ecf4b01fbaf117e08a0f60e30d9a6
```

| Property | Value |
|----------|-------|
| **Image** | `python:3.11.11-slim-bookworm` |
| **Digest** | Pinned by SHA-256 |
| **Rationale** | Eliminates floating tags (never use `latest`). |
| **Policy** | Update only via deliberate PR with fresh digest verification. |

### Why Pinned Digests?

- Prevents supply-chain risk from compromised tags
- Ensures identical base across all builds and times
- Follows OpenSSF Scorecard recommendation for Docker-based builds

## 2. Dependency Installation

All Python dependencies are installed from the hash-pinned lockfile:

```dockerfile
COPY requirements-lock.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements-lock.txt
```

- Every transitive dependency is pinned by exact version and SHA-256 hash
- `pip install` is deterministic: same lockfile → same environment
- `PIP_NO_CACHE_DIR=1` ensures a clean, minimal layer

## 3. Build Targets

The Dockerfile defines three build targets:

| Target | Purpose | Base |
|--------|---------|------|
| `base` | Runtime with lockfile-pinned dependencies | `python:3.11.11-slim-bookworm` |
| `test` | Full test suite execution | `base` |
| `production` | Minimal production image with source only | `base` |

### Build Commands

```bash
# Build the base runtime
docker build --target base -t helix-ids:base .

# Build and run tests
docker build --target test -t helix-ids:test .

# Build the production image
docker build --target production -t helix-ids:production .

# Run the production container
docker run --rm helix-ids:production
```

## 4. Test Execution in Container

The `test` target runs:

1. **pytest** with coverage gate (>=65%)
2. **Ruff** — 0 violations
3. **Mypy** — 0 errors

All three must pass or the build fails.

## 5. SBOM Generation

SBOM can be generated from the running container:

```bash
# From outside the container
docker run --rm helix-ids:base sh -c "pip install cyclonedx-bom && cyclonedx-py requirements -g --output-reproducible --output-file /tmp/sbom.json --output-format JSON"

# Or build with SBOM step
docker build --target base --iidfile /tmp/image-id.txt .
docker sbom $(cat /tmp/image-id.txt) > results/sbom/container-sbom.json
```

## 6. Container Signing (Cosign Keyless)

**Phase 10B:** Container images are signed using Cosign keyless signing with GitHub OIDC tokens.

### Signing Workflow

Container signing occurs in the `sign-release.yml` workflow on tag push:

```yaml
- name: Build and sign container
  run: |
    docker build --target production -t ghcr.io/${{ github.repository }}:${{ github.ref_name }} .
    cosign sign --yes ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

### Verification by Downstream Consumers

Downstream consumers verify the container signature:

```bash
# Verify the container image signature
cosign verify \
  --certificate-identity-regexp "https://github.com/kdhiraj/helix-ids/.github/workflows/sign-release.yml" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

### Signature Artifacts

- Container signatures are stored in the container registry alongside the image
- Signatures use keyless mode (ephemeral keys via Fulcio)
- Signing identity is bound to the GitHub Actions workflow run

## 7. Container Trust Verification

The release-integrity workflow verifies container trust:

1. Container image signature is verified using Cosign
2. SBOM attestation is verified for integrity
3. Checksums are validated against the signed manifest
4. All verification steps must pass before release approval

## 8. Reproducibility Verification

To verify that the container produces identical results:

```bash
# Build 1
docker build --target test -t helix-ids:test-v1 .

# Build 2 (same commit)
docker build --target test -t helix-ids:test-v2 .

# Compare
docker run --rm helix-ids:test-v1 python -m pytest -q --tb=short > /tmp/test-output-v1.txt
docker run --rm helix-ids:test-v2 python -m pytest -q --tb=short > /tmp/test-output-v2.txt
diff /tmp/test-output-v1.txt /tmp/test-output-v2.txt
```

If both builds are from the same commit and lockfile, the outputs should match.

## 9. Trust Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Container Trust Chain                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Dockerfile (digest-pinned base)                              │
│       │                                                       │
│       ▼                                                       │
│  docker build → production image                              │
│       │                                                       │
│       ├──→ Cosign sign (keyless, OIDC)                       │
│       │       │                                               │
│       │       └──→ Signature stored in OCI registry          │
│       │                                                       │
│       ├──→ SBOM attached as attestation                      │
│       │       │                                               │
│       │       └──→ CycloneDX SBOM → in-toto attestation     │
│       │                                                       │
│       └──→ Release integrity verification                    │
│               │                                               │
│               └──→ Cosign verify → sha256sum -c            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## 10. CI Integration

The container reproducibility check is integrated into the CI pipeline:

```yaml
- name: Build container
  run: docker build --target test -t helix-ids:ci-${{ github.sha }} .

- name: Verify test suite in container
  run: |
    docker run --rm helix-ids:ci-${{ github.sha }} \
      python -m pytest -q --cov=src/helix_ids --cov-fail-under=65 --tb=short
```

## 11. Security Considerations

- **No `latest` tags** — all images built with explicit tags based on commit SHA
- **Minimal base** — `slim-bookworm` variant reduces attack surface
- **No build-time secrets** — Docker build args should never contain credentials
- **Layer hygiene** — Lockfile copied before source for optimal layer caching
- **Cache control** — CI should use `--no-cache` periodically to catch base image updates
- **Keyless signing** — No long-lived signing keys; ephemeral keys via Fulcio
- **OIDC binding** — Signing identity bound to GitHub Actions workflow

## 12. Risk Register

| ID | Finding | Mitigation | Status |
|----|---------|------------|--------|
| CR-1 | Base image drift | Digest-pinned base image | CLOSED |
| CR-2 | Floating version tags | All tags are explicit commit references | CLOSED |
| CR-3 | Non-reproducible install | Hash-pinned lockfile installation | CLOSED |
| CR-4 | Test drift between host and container | Test target runs full suite inside container | CLOSED |
| CR-5 | Container image tampering | Cosign keyless signing + verification | CLOSED (10B) |
| CR-6 | Unverified container provenance | SLSA provenance attestation for container builds | CLOSED (10B) |
| CR-7 | Missing SBOM attestation | CycloneDX SBOM attached as in-toto attestation | CLOSED (10B) |
