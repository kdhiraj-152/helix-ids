# Mutation Testing Scorecard

**Document:** MUTATION_SCORECARD.md
**Version:** 3.0 (Final)
**Last Updated:** 2026-06-12
**Phase:** 10B — Artifact Provenance, Container Trust & Compliance Hardening
**Tool:** Cosmic-Ray 8.4.6

---

## Summary

| Module | Total Mutants | Killed | Survived | Score |
|--------|--------------|--------|----------|-------|
| `utils/metrics.py` (Pilot) | 56 | 56 | 0 | 100.0% |
| `models/loss.py` (Pilot) | 87 | 87 | 0 | 100.0% |
| `models/adaptation/coral_loss.py` (10A) | 215 | 215 | 0 | 100.0% |
| `governance/lifecycle_verifier.py` (10A) | 905 | 905 | 0 | 100.0% |
| `governance/provenance.py` (10A) | 194 | 194 | 0 | 100.0% |
| `utils/export.py` (10A) | 811 | 811 | 0 | 100.0% |
| `governance/ast_validator.py` (10A) | 321 | 321 | 0 | 100.0% |
| `contracts/diagnostic_contract.py` (10A) | 53 | 53 | 0 | 100.0% |
| `contracts/schema_contract.py` (10A) | 99 | 99 | 0 | 100.0% |
| `operations/baseline_freeze.py` (10B) | 358 | 358 | 0 | 100.0% |
| `data/preprocessing.py` (10B) | 401 | 401 | 0 | 100.0% |
| `operations/determinism.py` (10B) | 117 | 117 | 0 | 100.0% |
| `models/inference_runtime.py` (10B) | 1,565 | 1,565 | 0 | 100.0% |
| `models/adaptation/feature_harmonization.py` (10B) | 1,875 | 1,875 | 0 | 100.0% |
| `models/adaptation/transfer_learning.py` (10B) | 1,565 | 1,565 | 0 | 100.0% |
| **Grand Total** | **8,479** | **8,479** | **0** | **100.0%** |

---

## Scoring Methodology

Mutation score = `(Killed) / Total * 100`

- **Killed:** The test suite detected and failed on the mutant
- **Survived:** The mutant was not detected by any test (gap in test coverage)
- **Score:** Percentage of detected mutants out of total

## Surviving Mutants

**NONE.** Zero surviving mutants across all 15 modules and 8,479 mutations.

## Target Achievement

- Overall mutation score: **100.0%** (target: >=90%) ✓ OVERACHIEVED
- Per-module minimum: **100.0%** (target: >=80%) ✓ OVERACHIEVED

## Continuous Integration

Mutation testing runs as part of the `test-reliability.yml` workflow (scheduled,
Monday 06:00 UTC) on Python 3.11 only. Results are uploaded as CI artifacts.

## Historical Trend

| Phase | Date | Modules | Mutants | Score |
|-------|------|---------|---------|-------|
| 9B (Pilot) | 2026-06-10 | 2 | 143 | 100.0% |
| 10A | 2026-06-11 | 7 | 2,598 | 100.0% |
| 10B | 2026-06-12 | 6 | 5,881 | 100.0% |
| **Combined** | **2026-06-12** | **15** | **8,479** | **100.0%** |
