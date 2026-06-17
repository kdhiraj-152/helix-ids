# Benchmarks

Performance baseline and load-testing infrastructure for HELIX-IDS.

## Purpose

Benchmarks measure pipeline-stage latency, concurrency tolerance, and
system robustness.  Results are tracked over time to detect regressions
before they reach production.

## Files

| File | Purpose |
|------|---------|
| `baseline.json` | Current benchmark output (written by `benchmark_pipeline.py`, gitignored) |
| `baseline.reference.json` | Blessed reference baseline (committed to git) |
| `load_test_results.json` | Concurrency/load test results (written by `load_test.py`, gitignored) |

## Scripts

All scripts live in `scripts/benchmarks/`.

| Script | Measures | Output |
|--------|----------|--------|
| `benchmark_pipeline.py` | 7 pipeline stages: data loading, training, inference, checkpoint I/O | `benchmarks/baseline.json` |
| `load_test.py` | Concurrency tolerance for 5 subsystems + 4 storm scenarios | `benchmarks/load_test_results.json` |
| `check_performance_regression.py` | CI gate: compares current output against reference baseline | exit code (0 = pass, 1 = fail) |

## Workflow

### 1. Generate a baseline

```bash
PYTHONPATH=src python scripts/benchmarks/benchmark_pipeline.py
```

This writes `benchmarks/baseline.json` with stage timings and environment metadata.

### 2. Bless a baseline (first time, or after intentional improvements)

After verifying the output is acceptable:

```bash
PYTHONPATH=src python scripts/benchmarks/check_performance_regression.py --bless
```

This copies `baseline.json` → `baseline.reference.json`.  Commit the reference:

```bash
git add benchmarks/baseline.reference.json
git commit -m "chore: bless baseline reference"
```

### 3. Check for regressions (CI or local)

```bash
PYTHONPATH=src python scripts/benchmarks/check_performance_regression.py
```

Compares `baseline.reference.json` vs `baseline.json`.  Exits non-zero if
any metric exceeds its regression threshold.

## Regression Thresholds

These thresholds are also embedded in `baseline.reference.json` under
`ci_gates` for traceability.

| Gate | Metric | Direction | Threshold |
|------|--------|-----------|-----------|
| Training step | mean latency | higher = worse | +5% |
| Inference | mean latency | higher = worse | +5% |
| Checkpoint save | throughput | lower = worse | -10% |
| Checkpoint load | throughput | lower = worse | -10% |

## CI Integration

A typical CI step follows this sequence:

1. Run `benchmark_pipeline.py` → produces `benchmarks/baseline.json`
2. Run `check_performance_regression.py` → compares against the committed
   `benchmarks/baseline.reference.json`
3. If the gate exits 0, the CI step succeeds
4. If it exits 1, the CI step fails and the pipeline is blocked

To update the reference after intentional performance improvements, run
with `--bless` and commit the new `baseline.reference.json`.

## Custom Paths

All three scripts accept `--help` for available CLI options.  For example:

```bash
python scripts/benchmarks/check_performance_regression.py \
    --baseline benchmarks/baseline.reference.json \
    --current benchmarks/baseline.json \
    --bless
```
