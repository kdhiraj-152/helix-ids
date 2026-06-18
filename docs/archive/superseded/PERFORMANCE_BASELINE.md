# Performance Baseline — HELIX-IDS Phase 22C

> **Environment**: Apple M4 (17.2 GB RAM) · MPS (Apple Silicon) · Python 3.11.15 · PyTorch 2.12.0  
> **Date**: 2026-06-18  
> **Baseline file**: `benchmarks/baseline.json`

## Pipeline Benchmarks

| Stage | Mean (s) | Median (s) | p95 (s) | p99 (s) | StdDev |
|---|---|---|---|---|---|
| **Data loading (single-thread)** | 0.0821 | 0.0806 | 0.0906 | 0.0958 | 0.0054 |
| **Data loading (multi-thread)** | 0.0441 | 0.0436 | 0.0463 | 0.0464 | 0.0013 |
| **Feature engineering** | 0.0089 | 0.0088 | 0.0097 | 0.0100 | 0.0004 |
| **Training step** | 0.0029 | 0.0029 | 0.0032 | 0.0033 | 0.0002 |
| **Inference** | 0.0007 | 0.0007 | 0.0008 | 0.0008 | 0.00004 |
| **Checkpoint save** | 0.0012 | 0.0011 | 0.0020 | 0.0021 | 0.0003 |
| **Checkpoint load** | 0.0017 | 0.0017 | 0.0018 | 0.0018 | 0.00004 |

## Checkpoint Throughput

| Metric | Value |
|---|---|
| Checkpoint size | 1.51 MB |
| Save throughput | 1,249.4 MB/s |
| Load throughput | 899.75 MB/s |

## Environment Metadata

| Attribute | Value |
|---|---|
| CPU | Apple M4 |
| RAM | 17.2 GB |
| GPU | MPS (Apple Silicon) |
| PyTorch | 2.12.0 |
| Python | 3.11.15 |
| Platform | macOS 27.0 arm64 |

## CI Regression Gates

Failures are raised when runtime performance degrades beyond these thresholds relative to the baseline:

| Gate | Threshold |
|---|---|
| Training step | >5% regression |
| Inference | >5% regression |
| Checkpoint throughput | >10% regression |

## What These Benchmarks Cover

- **Data loading**: `pd.read_csv` on 124,710-row NSL-KDD train set (41 features), single-thread and 4-thread concurrent.
- **Feature engineering**: `FeatureEngineer.engineer_all_features()` on synthetic 4096-row batch with all transformation stages.
- **Training step**: Full forward + backward + optimizer.step on `HelixIDSFull` (~500K params) with batch_size=256.
- **Inference**: `model.eval()` forward pass on `HelixIDSFull`, batch_size=256, no gradients.
- **Checkpoint save/load**: `torch.save / load` of `state_dict` (~1.51 MB). Throughput derived from mean latency + checkpoint size.

## Measurement Protocol

- Warmup: 2-3 iterations before sampling.
- Repeats: 10-30 iterations per stage (higher for low-variance operations).
- GC collected between GPU-dependent stages to isolate measurement.
- Results mirrored to `benchmarks/baseline.json` for automated regression detection.

---
*Baseline established by `scripts/benchmarks/benchmark_pipeline.py`*
