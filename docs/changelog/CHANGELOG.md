# Changelog

> Last updated: 2026-06-18

All notable changes to HELIX-IDS during the active development cycle.

---

## 2026-06-18

### Phase 23 — Repository Rationalization & Hygiene
- Consolidated CI/CD workflows from 13 to 6
- Branch protection applied to main (PR required, status checks, squash-merge)
- dev branch created with light protection
- Stale branches cleaned (4 deleted)
- Stale references fixed (eval→json.loads, weights_only=True, asserts→exceptions)
- ENGINEERED_FEATURE_NAMES migrated to `scripts/_constants.py`

### Phase 22C — Load Testing & Certification
- Load testing completed (1x–100x concurrency, zero errors)
- Soak test infrastructure implemented (3 runners: training, inference, logging)
- Telemetry collector with hourly snapshots and trend analysis
- Performance regression CI workflow
- RC3 certification verdict: PASS

### Phase 22B — Chaos & Fault Injection
- Checkpoint chaos testing completed
- Fault injection testing implemented
- Memory leak detection tests
- All subsystem recovery mechanisms validated

### Phase 22A — Property Testing & Hardening
- Property-based testing with Hypothesis
- Dataset corruption testing completed
- Unused config fields cleaned (11 fields)
- Config version mismatch (41 vs 17) documented

## 2026-06-16

### Phase 21 — Production Blocker Elimination
- Full type annotation audit (WARNING)
- Full lint audit (PASS)
- Full test audit (WARNING — performance tests missing)
- Security audit (PASS)
- Production readiness checklist generated

### Phase 20 — Architecture Refresh
- Lockdown test suite (10 tests for RC1 gates: ≤2,000 LOC, ≤100 methods)
- Architecture freeze maintained
- ENGINEERED_FEATURE_NAMES duplication resolved
- Config governance documented

## 2026-06-12

### Phase 19 — Architecture Freeze
- Package boundaries frozen
- Dependency graph locked (256 nodes, 590 edges)
- All architecture test files passing (6 files, 34 tests)
- Technical debt register established
- Delegate extraction (TrainerFacade pattern)

### Phase 18 — Architecture Extraction
- Multi-part extraction of HelixFullTrainer into delegate objects
- TrainerFacade (180 LOC, 20 methods)
- BatchProcessor, EpochRunner, PhaseOrchestrator delegates
- Phase 18A/B/C incremental extraction

### Phase 17 — Multi-Phase Training
- SupCon loss (representation learning phase)
- Multi-task classification (joint fine-tuning)
- Curriculum schedule with configurable phase transitions
- Phase orchestration state machine

## 2026-06-10

### Phase 10B — Mutation Testing Hardening
- 8 new modules under cosmic-ray testing
- 100% kill rate across all 15 modules, 8,479 mutants
- Coverage gate set at 65%
- Pre-processing, determinism, inference runtime, feature harmonization hardened

### Phase 10A — Mutation Testing & Release Integrity
- Pilot mutation testing (metrics.py, loss.py)
- Cosmic-Ray configs for 8 modules
- SLSA provenance generation
- Sigstore/Cosign keyless signing
- Dockerfile with digest-pinned base
- License compliance (check_licenses.py, LICENSE_POLICY.md)

### Phase 9B — Mutation Testing Pilot
- Pilot testing with 2 modules (143 mutants, 100% killed)
- Test coverage gap analysis

## 2026-06-03

### Phase 8B — Reproducible Builds
- Reproducible build guide
- SBOM generation (CycloneDX)
- Release integrity pipeline
- Container build verification

### Governance ADRs
- ADR-001: Governance Philosophy (accepted)
- ADR-002: Schema Lifecycle and Versioning (accepted)
- ADR-003: Hash Authority (accepted)
- ADR-004: Enforcement Pipeline (accepted)

## 2026-06-01

### Phase 4 — Initial Governance & Schema
- Governance coverage audit
- Schema contract definition
- Hash authority established
- IMMUTABLE_SCHEMA_CONTRACT

### Phase 1–3 — Foundation
- Repository setup and configuration
- Data harmonization pipeline
- Multi-dataset loader architecture
- REST inference server
- Staging gate and traffic expansion guards
- Monitoring metrics infrastructure
