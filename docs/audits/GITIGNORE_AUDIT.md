# Phase 23 — Git Ignore Audit

> Generated: 2026-06-18
> Target: `.gitignore` at repository root

---

## Current `.gitignore` Coverage

### Python / Build Artifacts
| Pattern | Status |
|---------|--------|
| `__pycache__/`, `*.py[cod]`, `*.pyo`, `*.pyd`, `*.so` | ✅ |
| `*.egg-info/`, `.eggs/`, `build/`, `dist/` | ✅ |
| `.venv/`, `.venv*/`, `venv/` | ✅ |

### Tool Caches
| Pattern | Status |
|---------|--------|
| `.pytest_cache/` | ✅ |
| `.mypy_cache/` | ✅ |
| `.ruff_cache/` | ✅ |
| `.ipynb_checkpoints/` | ✅ |
| `.hypothesis/` | ✅ |

### Coverage
| Pattern | Status |
|---------|--------|
| `.coverage` | ✅ |
| `coverage.xml` | ✅ |
| `htmlcov/` | ✅ |
| `*.coverage` | ✅ |

### OS / Editor
| Pattern | Status |
|---------|--------|
| `.DS_Store` | ✅ |
| `.vscode/` | ✅ |
| `.idea/` | ✅ |

### Model / Training Artifacts
| Pattern | Status |
|---------|--------|
| `checkpoints/` | ✅ |
| `/models/` (top-level) | ✅ |
| `**/models/*.pt` | ✅ |
| `**/models/*.pth` | ✅ |
| `results/` | ✅ |
| `outputs/` | ✅ |
| `artifacts/` | ✅ |
| `exported_model.pt` | ✅ |
| `exported_model.pt.*` | ✅ |
| `*.pt`, `*.pth` (top-level/binary) | ✅ |

### Data / Dataset Files
| Pattern | Status |
|---------|--------|
| `data/processed/` | ✅ |
| `data/**/raw/`, `data/**/raw` | ✅ |
| `data/**/*.csv` | ✅ |
| `data/**/*.npy` | ✅ |
| `data/**/*.npz` | ✅ |
| `data/**/*.parquet` | ✅ |
| `data/**/*.pkl` | ✅ |
| `data/splits/` | ✅ |

### Logs / Sessions
| Pattern | Status |
|---------|--------|
| `*.log` | ✅ |
| `logs/` | ✅ |
| `session_logs/` | ✅ |

### Temporary / Generated
| Pattern | Status |
|---------|--------|
| `tmp_artifact/` | ✅ |
| `*.db` (cosmic-ray sessions) | ✅ |
| `*.pdf`, `*.docx`, `*.pptx` | ✅ |
| `docs/figures/*.png` | ✅ |
| `baseline-freeze-session` | ✅ |
| `.code-review-graph/` | ✅ |

### Benchmark Outputs
| Pattern | Status |
|---------|--------|
| `/benchmarks/baseline.json` | ✅ |
| `/benchmarks/load_test_results.json` | ✅ |
| `/benchmarks/_tmp/` | ✅ |

### One-Time Audit/Report Artifacts
| Pattern | Status |
|---------|--------|
| `docs/DUPLICATE_AND_SUPERSEDED_ANALYSIS.md` | ✅ |
| `naming_consistency_assessment.md` | ✅ |
| `CLEANUP_REPORT.md` | ✅ |
| `finding-structured.json` | ✅ |

### Agent / Local State
| Pattern | Status |
|---------|--------|
| `.opencode.json` | ✅ |
| `.mcp.json` | ✅ |
| `src/helix_ids/_unused/` | ✅ |

---

## Verified Untracked Files

The following files exist on disk and are confirmed gitignored (not tracked):

| File | Size | Pattern |
|------|------|---------|
| `artifacts/operations/live_events.jsonl` | 928 B | `artifacts/` |
| `artifacts/soak/smoke_test_20260618_032231/snapshot_2026-...json` | 508 B | `artifacts/` |
| `benchmarks/load_test_results.json` | ~N/A | `/benchmarks/load_test_results.json` |
| `.vscode/settings.json` | 2 B | `.vscode/` |
| `session_logs/session_*.json` | 84 KB | `session_logs/` |

---

## Verification of Tracked Files

Only **one CSV file** is tracked in the repository:

- `tests/fixtures/cicids_snapshot.csv` (1,084 B) — legitimate test fixture

No `.npy`, `.npz`, `.parquet`, `.pkl`, `.pt`, `.pth` files are tracked. The patterns are working correctly.

---

## Recommendations

### Recommended Additions (Low Priority)

| Pattern | Reason |
|---------|--------|
| `*.swp`, `*.swo`, `*~` | Vim/editor swap files (common across dev environments) |
| `.Python` | `pyenv` local version marker |
| `*.tmp` | Generic temporary files |
| `mlruns/` | MLflow run directory (if MLflow is used for tracking) |
| `wandb/` | Weights & Biases logging directory |

### Recommended Removals (None)

No `.gitignore` patterns should be removed. All are still relevant.

### Documentation Cleanup

The `docs/figures/*.png` pattern works correctly — the 6 committed figures remain tracked because they were added before this pattern was added or were force-added. To fully reconcile, either:
1. Remove the committed figures and regenerate them on demand
2. Or add them to `.gitattributes` as `-diff -merge` if they're true source artifacts

---

## Summary

**The `.gitignore` is in excellent shape.** Coverage is comprehensive (40+ patterns). The audit found no missing critical patterns, no leaked binary artifacts, and no tracked temp files. The recommendations above are optional quality-of-life improvements.
