# ADR-001: Governance Philosophy

**Status:** Accepted
**Date:** 2026-06-03
**Deciders:** HELIX-IDS governance lead

---

## Context

HELIX-IDS operates in a formalization-first mode where pipeline changes must be governed, auditable, and reproducible. This ADR defines the core governance principles.

## Current Guarantees

1. **Schema immutability** — Runtime feature schema is immutable post-export. Any drift is a contract violation (`IMMUTABLE_SCHEMA_CONTRACT.md`).
2. **Hash authority** — All hashes use SHA-256 with canonical JSON encoding. Algorithm changes require a migration plan.
3. **Lineage completeness** — Every result payload carries full provenance lineage (run_id, fingerprint, git_commit, dataset hashes, config hashes).
4. **Non-bypassable enforcement** — Governance is enforced at the entrypoint level; pipelines cannot opt out without failing (`helix_ids.governance.entrypoint.governed_entrypoint`).
5. **AST safety** — No `__import__`, `eval`, `exec`, or dynamic code generation in governed source files (`ast_validator.py`).
6. **Promotion consensus** — Checkpoints require 3-seed consensus with variance thresholds before promotion.
7. **CI gates** — Every schema change requires a registry update; CI blocks non-conforming artifacts.

## Known Limitations

| Limitation | Severity | Mitigation |
|-----------|----------|-----------|
| Single-seed runs fail promotion (`E-T3-SINGLE-SEED-INVALID`) | High | Always run with `min_seed_runs=3`; single-seed mode is research-only |
| UNSW classes 5,6 have zero training samples | High | No learnability possible; will always predict zero |
| Determinism not cryptographically notarized | Medium | Provenance chain is hashed but not externally witnessed |
| Frozen dependencies not in provenance | Medium | Environment reproducibility via seed + deterministic ops; pip lock out of scope |
| Governance AST validator runs per-file only | Low | Whole-pipeline governance via entrypoint + lifecycle_verifier |

## Future Provenance-Locking Roadmap

1. **Near (Q3 2026):** Add `requirements.lock` hash to result lineage for full environment reproducibility.
2. **Mid (Q4 2026):** Cryptographic provenance notarization via external timestamping service.
3. **Long (2027):** Individual pipeline step hashes (raw→processed→split) tracked separately for pinpoint reproducibility.

## Consequences

- **Positive:** Governance enables auditability and reproducibility.
- **Negative:** Overhead of governance checks adds ~5% to CI runtime.
- **Neutral:** Formalization mode restricts exploratory changes; required for paper-stage stability.