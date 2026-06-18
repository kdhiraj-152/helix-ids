# HELIX-IDS Governance

> Last updated: 2026-06-18  
> Authoritative reference for all governance decisions, schema contracts, and enforcement policies.

## Core Principles

HELIX-IDS operates in formalization-first mode. All pipeline changes must be governed, auditable, and reproducible.

1. **Schema immutability** — Runtime feature schema is immutable post-export; drift is a contract violation
2. **Hash authority** — All hashes use SHA-256 with canonical JSON encoding
3. **Lineage completeness** — Every result carries provenance (run_id, fingerprint, git_commit, dataset/config hashes)
4. **Non-bypassable enforcement** — Governance at the entrypoint level; pipelines cannot opt out
5. **AST safety** — No `__import__`, `eval`, `exec`, or dynamic code in governed files
6. **Promotion consensus** — 3-seed minimum with variance thresholds before checkpoint promotion
7. **CI gates** — Every schema change requires registry update; CI blocks non-conforming artifacts

## ADR Register

### ADR-001: Governance Philosophy (Accepted 2026-06-03)

Establishes the governance framework. Key guarantees:
- Schema immutability
- SHA-256 hash authority with canonical JSON
- Full provenance lineage on every result
- Entrypoint-level enforcement
- AST safety validation
- 3-seed promotion consensus
- CI gates for schema changes

**Known limitations:**
- Single-seed runs fail promotion (research-only)
- UNSW classes 5,6 have zero training samples (no learnability possible)
- Determinism not cryptographically notarized (medium severity)
- Frozen dependencies not in provenance (medium severity)
- Governance AST validator runs per-file only (low severity)

### ADR-002: Schema Lifecycle and Versioning (Accepted 2026-06-03)

Manages the runtime feature schema (immutable) and manifest/result schemas (versioned).

Key policies:
- Schema registry (`schema_registry.yaml`) tracks all versions with owner, 30d compatibility window, deprecation
- Version format is YYYY-MM-DD, enforced monotonically
- Breaking changes require documented migration
- Dual-read window during compatibility period
- CI enforces schema updates on structural changes

### ADR-003: Hash Authority (Accepted 2026-06-03)

Single SHA-256 for all content hashing. Changes require documented migration plan. Canonical JSON encoding (`sort_keys=True`).

### ADR-004: Enforcement Pipeline (Accepted 2026-06-03)

Governance enforced at three levels:
1. **Build time** — AST validation, schema registry checks
2. **Training time** — Entrypoint wrapping, determinism, contract validation
3. **Deployment time** — Checkpoint verification, provenance chain validation, promotion gates

## Config Governance

Configuration resolution priority (highest to lowest):
1. CLI arguments
2. Environment variables
3. YAML config files
4. Python dataclass defaults
5. In-code immutable constants

**Important:** The system does not use a unified override fabric. Each layer is read independently at different call sites.

### Config Files

| File | Purpose | Status |
|------|---------|--------|
| `config/helix_config.yaml` | Model architecture variants, hyperparameters | Legacy — `input_features: 41` vs canonical 17 |
| `config/training.yaml` | Unified Transformer training config | Aspirational — many fields not wired |
| `config/platform_configs.yaml` | Per-platform deployment constraints | Active — single source of truth |
| `config/schema_registry.yaml` | Schema lifecycle registry | **ORPHANED** — not consumed by code |

### Schema Contract

- **Canonical input dim**: 17 features
- **Binary classes**: 2 (Normal, Attack)
- **Family classes**: 7
- Enforced by `schema_contract.py` and validated by `lifecycle_verifier.py`
- See `src/helix_ids/data/feature_harmonization.py` for canonical feature order

## Security Controls

| Control | Status | Enforcement |
|---------|--------|-------------|
| Ruff (lint) | Active | Blocking CI gate |
| Mypy (static types) | Active | Blocking CI gate (src/) |
| Coverage ≥65% | Active | Blocking CI gate |
| pip-audit | Active | Blocking CI gate (strict) |
| Dependabot | Active | Weekly, 5 PRs max |
| Dependency review | Active | PR gate |
| CodeQL | Active | PR + scheduled |
| Bandit | Active | Blocking CI gate |
| SLSA provenance | Active | Release pipeline |
| SBOM generation | Active | Release pipeline |

## Known Limitations

| Limitation | Severity | Mitigation |
|-----------|----------|-----------|
| Single-seed runs fail promotion | High | Always use `min_seed_runs=3` |
| UNSW classes 5,6 zero training samples | High | No learnability possible |
| Determinism not cryptographically notarized | Medium | Provenance chain hashed |
| Frozen dependencies not in provenance | Medium | Seed + deterministic ops |
| Config YAML version mismatch (41 vs 17) | Medium | Schema contract overrides at runtime |
| `schema_registry.yaml` orphaned | Medium | Not consumed by code |
