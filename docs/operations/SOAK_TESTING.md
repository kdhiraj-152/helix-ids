# Soak Testing

> Last updated: 2026-06-18  
> Purpose, metrics, pass criteria, and execution procedures for certification soak testing.

## Purpose

Soak testing verifies that HELIX-IDS subsystems remain stable under sustained load for 24 hours. Three subsystems are certified independently:
1. **Training** — Continuous training loop
2. **Inference** — Persistent inference service
3. **Logging** — Structured logging subsystem

## Telemetry Metrics

Collected hourly via `scripts/benchmarks/soak_telemetry.py` (320 lines):

| Metric | Source | Notes |
|--------|--------|-------|
| RSS memory | psutil | Process-level memory |
| GPU memory | CUDA/MPS | If GPU available |
| File handles | psutil | Process-level FD count |
| Thread count | threading | Active threads |
| GC stats | gc | Collected objects count |
| Tensor count | torch | Active tensor count |
| Checkpoint count | filesystem | Total checkpoints written |
| Log volume | filesystem | Total log bytes |
| Latency | custom | Via metric injection |
| Throughput | custom | Via metric injection |

Snapshots written to: `artifacts/soak/<run_id>/snapshot_<timestamp>.json`

Trend analysis via `summarize_run()` computes start/end/min/max/mean/trend direction for each metric.

## Certification Runners

| Script | Purpose |
|--------|---------|
| `scripts/benchmarks/soak_training.py` | 24h training loop with telemetry |
| `scripts/benchmarks/soak_inference.py` | 24h inference service with telemetry |
| `scripts/benchmarks/soak_logging.py` | 24h logging subsystem with telemetry |

Each runner: hourly snapshots, trend analysis, PASS/FAIL verdict generation.

## Pass Criteria

- **Zero deadlocks** over 24h
- **Zero corruption** in any subsystem
- **p99 latency < 2x baseline** at all measurement points
- **No monotonic memory growth** (RSS stable or mean-reverting)
- **No file handle leak** (count stable or mean-reverting)
- **All hourly trend verdicts = PASS**

## Failure Criteria

- Any deadlock → FAIL
- Data corruption in any subsystem → FAIL
- Monotonic memory increase over 6+ consecutive hours → FAIL
- File handle leak over 6+ consecutive hours → FAIL
- p99 latency > 2x baseline for sustained period → FAIL
- Process crash or unhandled exception → FAIL

## Execution

```bash
# Training certification (requires GPU for realistic workload)
python scripts/benchmarks/soak_training.py --duration 24

# Inference certification
python scripts/benchmarks/soak_inference.py --duration 24

# Logging certification
python scripts/benchmarks/soak_logging.py --duration 24
```

**Prerequisites:**
- `psutil>=5.8.0` installed (in `[monitoring]` optional group)
- Adequate disk space for 24h of telemetry snapshots
- GPU recommended for training certification

## Certification Process

1. Run each certification script for 24 hours
2. Collect hourly telemetry snapshots
3. Run `summarize_run()` for trend analysis
4. Generate PASS/FAIL verdict per subsystem
5. All three subsystems must pass for full RC3 certification

## Load Testing

`scripts/benchmarks/load_test.py` (576 lines) provides concurrency ladder testing:

| Concurrency | Inference Throughput | p50 | p95 | Errors |
|-------------|---------------------|-----|-----|--------|
| 1x | 1,519 req/s | 0.57ms | 0.75ms | 0 |
| 10x | 1,459 req/s | 6.66ms | 7.21ms | 0 |
| 25x | 1,487 req/s | 16.07ms | 16.96ms | 0 |
| 50x | 1,475 req/s | 32.40ms | 34.64ms | 0 |
| 100x | 1,425 req/s | 65.58ms | 71.42ms | 0 |

Additional scenarios: checkpoint save storm (1,000 ops), log flood (10,000 msgs), restart storm (10,000 ops), cascading failures — all zero errors.
