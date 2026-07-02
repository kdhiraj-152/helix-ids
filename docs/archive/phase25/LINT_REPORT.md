# Lint Report — Phase 25C

Generated: 2026-06-21

**Tool:** ruff check .

## Summary

**Total errors:** 11 (all pre-existing, non-TON-IoT)

## Errors by location

| Location | Error | Description |
|----------|-------|-------------|
| `scripts/analysis/architecture_audit_part_a.py:77` | E741 | Ambiguous variable name: `l` |
| `scripts/analysis/architecture_audit_part_a.py:78` | E741 | Ambiguous variable name: `l` |
| `scripts/analysis/architecture_audit_part_a.py:215` | C401 | Unnecessary generator (set comprehension) |
| `scripts/analysis/architecture_audit_part_a.py:219` | C401 | Unnecessary generator (set comprehension) |
| `scripts/benchmarks/soak_monitor.py:179` | F841 | Unused variable `run_dir` |
| `scripts/benchmarks/soak_monitor.py:193` | F841 | Unused variable `first` |
| `scripts/data/download_new_datasets.py:14` | F401 | Unused import `sys` |
| `scripts/data/download_new_datasets.py:15` | F401 | Unused import `time` |
| `tests/test_feature_harmonization.py:23` | E402 | Module level import not at top of file |
| `tests/test_feature_harmonization.py:24` | E402 | Module level import not at top of file |
| `tests/test_feature_harmonization.py:29` | E402 | Module level import not at top of file |

## TON-IoT Related Issues

**None.** All 11 errors are pre-existing issues in `scripts/` or `tests/` directories. No TON-IoT related lint errors found.

## Verdict

**PASS** — No TON-IoT related lint failures.
