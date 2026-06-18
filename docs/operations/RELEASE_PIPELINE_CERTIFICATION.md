# Phase 23 — Release Pipeline Certification

Date: 2026-06-18
Status: Certified ✓

## Pipeline Structure

Single workflow: `release.yml`
Trigger: tags v*, workflow_dispatch

### Job 1: verify (required by sign)

| Step | Check | Artifact |
|------|-------|----------|
| 1 | Lockfile synced | — |
| 2 | SBOM generate + validate | results/sbom/sbom.json |
| 3 | SBOM attestation | results/attestations/sbom-attestation.json |
| 4 | Coverage ≥65% | results/junit/release-report.xml |
| 5 | Ruff 0 violations | — |
| 6 | Mypy 0 errors | — |
| 7 | pip-audit 0 high vulns | — |
| 8 | Bandit 0 HIGH findings | — |
| 9 | Artifact checksums | results/checksums.sha256 |
| 10 | Checksums verify | — |
| 11 | SLSA provenance gen | results/provenance/slsa-attestation.json |
| 12 | SLSA provenance verify | — |
| 13 | License compliance | results/licenses/ |
| 14 | Trust report | results/trust-report/trust-report.json |
| 15 | Upload artifacts | All above |

### Job 2: sign (depends on verify)

| Step | Operation |
|------|-----------|
| 1 | Download verification artifacts |
| 2 | Generate SBOM |
| 3 | License inventory |
| 4 | SLSA provenance |
| 5 | Checksums |
| 6 | Cosign keyless sign (all artifacts) |
| 7 | Verify signatures |
| 8 | Login to GHCR |
| 9 | Build container (production target) |
| 10 | Push to GHCR |
| 11 | Sign container (Cosign keyless) |
| 12 | Verify container signature |
| 13 | Upload signing artifacts |

## Verification Checks

### SBOM Generation
- [x] cyclonedx-py produces reproducible JSON
- [x] Validation passes
- [x] Attestation contains correct subject digest
- [x] SBOM components countable

### SLSA Provenance
- [x] generate_slsa_provenance.py succeeds
- [x] verify_slsa_provenance.py succeeds
- [x] Attestation is valid in-toto v1 statement

### Cosign Signing
- [x] All artifacts sign with keyless mode
- [x] Signatures verify against OIDC identity
- [x] Container image signs and verifies
- [x] Certificate identity matches workflow path

### Lockfile Integrity
- [x] pip-compile produces identical output
- [x] Compare against requirements.in

### Security Gates
- [x] pip-audit: 0 vulnerabilities
- [x] Bandit: 0 HIGH findings
- [x] Ruff: 0 violations
- [x] Mypy: 0 errors

## Release Artifact Chain

```
requirements.in
  → requirements-lock.txt (verified hashes)
  → results/sbom/sbom.json (CycloneDX)
  → results/attestations/sbom-attestation.json (in-toto)
  → results/provenance/slsa-attestation.json (SLSA v1.0)
  → results/signatures/*.sig + *.bundle (Cosign keyless)
  → ghcr.io/<repo>:<tag> (OCI container, signed)
  → results/trust-report/trust-report.json (consolidated)
```

## Rollback Plan

| Scenario | Action |
|----------|--------|
| Lockfile stale | Pin requirements.in, re-run pip-compile |
| SBOM validation fails | Check cyclonedx-py version |
| Provenance fails | Check generate_slsa_provenance.py |
| Cosign fails | Check OIDC token availability |
| Container fails | Check Dockerfile build context |
