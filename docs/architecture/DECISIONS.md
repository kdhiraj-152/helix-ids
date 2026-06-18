# Architecture Decision Records

> Last updated: 2026-06-18  
> Summary of key architectural decisions. Full text of each ADR is in the governance module source or archived governance docs.

## ADR-001: Governance Philosophy

- **Status:** Accepted (2026-06-03)
- **Decider:** HELIX-IDS governance lead
- **Decision:** Adopt formalization-first governance with schema immutability, SHA-256 hash authority, provenance lineage, entrypoint-level enforcement, AST safety, and 3-seed promotion consensus.
- **Tradeoff:** ~5% CI runtime overhead for governance checks in exchange for auditability and reproducibility.

## ADR-002: Schema Lifecycle and Versioning

- **Status:** Accepted (2026-06-03)
- **Decider:** HELIX-IDS governance lead
- **Decision:** Two-tier schema management — runtime feature schema is immutable (enforced by contract), manifest/result schemas are versioned via registry with 30d compatibility windows.
- **Tradeoff:** Breaking changes require formal migration cycle; registry ownership is a single point of failure.

## ADR-003: Hash Authority

- **Status:** Accepted (2026-06-03)
- **Decider:** HELIX-IDS governance lead
- **Decision:** Single SHA-256 for all content hashing with canonical JSON encoding (`sort_keys=True`). Algorithm changes require documented migration plan.
- **Tradeoff:** Simplicity over flexibility — single algorithm but straightforward to audit.

## ADR-004: Enforcement Pipeline

- **Status:** Accepted (2026-06-03)
- **Decider:** HELIX-IDS governance lead
- **Decision:** Three-layer enforcement at build time (AST validation, schema registry), training time (entrypoint wrapping, determinism, contracts), and deployment time (checkpoint verification, provenance chain, promotion gates).
- **Tradeoff:** Redundant but layered — no single skip point but O(n) validation overhead.

## Architecture Freeze (Phase 19, Maintained)

- **Decision:** Freeze all public API boundaries in `src/helix_ids/` for the paper cycle.
- **Enforcement:** 6 architecture test files (34 tests) enforcing DAG constraints, package boundaries, method count limits.
- **Key metrics:**
  - Dependency graph: 256 nodes, 590 edges (frozen)
  - Package-level cycles: 0 (cross-boundary and src-internal)
  - Reverse dependencies (src → scripts): 0
  - Forbidden imports (src. prefix): 0
  - `HelixFullTrainer` LOC: ≤2,000, methods: ≤100
  - `TrainerFacade` LOC: ≤180, methods: ≤20

## Post-Hoc Control Layer Design

- **Context:** Margin-based threshold decoupling for class-selective override in deployed inference.
- **Decision:** Use frozen adaptive threshold (calibration-time estimate) rather than online hybrid that causes control collapse. The `frozen_z_hybrid` configuration is numerically equivalent to `fixed_tau_only` under current parameters.
- **Result:** Control collapse (override_rate → 0, KL → 0) identified and fixed; bootstrapped CIs confirm stable non-zero override actuation.
- **Transparency:** z-score normalization channel adds architectural robustness but no additional discriminative signal at these settings.
