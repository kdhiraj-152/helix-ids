# Workflow Retention Policy — Phase 24

Generated: 2026-06-18 | Status: RC3-ready

## Artifact Retention Periods

| Artifact Type | Retention | Rationale |
|---------------|-----------|-----------|
| CI pytest reports | 14 days | Fast debug of per-commit failures |
| Coverage reports | 30 days | Trend tracking between releases |
| Quality reports (junit) | 30 days | Release-readiness audit trail |
| SBOM | 90 days | Supply-chain transparency window |
| Checksums | 90 days | Verification window for consumers |
| Attestations | 90 days | Signing + provenance verification |
| Mutation logs | 30 days | Trend tracking; regeneratable |
| License reports | 90 days | Compliance audit window |
| Trust reports | 90 days | Release artifact |
| Container images | Indefinite | Published to GHCR |
| Signatures | 90 days | Verification window for consumers |

## Lifecycle Rules

1. **CI artifacts** (ci, architecture, dependency-review): Auto-delete after 14 days via GitHub retention settings
2. **Quality artifacts** (quality, nightly): Delete after 30 days
3. **Release artifacts** (release): Retain 90 days minimum; critical for audit
4. **Generated metadata** (mutation logs, coverage XML): Safe to delete; regeneratable from workflow_dispatch

## Deletion Safe List

All artifacts in `ci.yml`, `architecture.yml`, `quality.yml`, `nightly.yml` can be auto-deleted after the specified retention period with no downstream impact. Only `release.yml` artifacts (SBOM, signatures, checksums) should be archived externally if needed beyond 90 days.

## Configuration

GitHub Actions artifact retention is set in repository Settings > Actions > General. Default: 90 days. Recommendation:
- CI workflows: 14 days
- Quality workflows: 30 days
- Release workflows: 90 days
