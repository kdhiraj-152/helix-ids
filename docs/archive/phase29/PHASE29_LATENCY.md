# Phase 29 — Inference Latency Benchmarks

**Generated**: 2026-06-23 23:31:45 IST
**Device**: mps
**Model**: DANNHelixModel (164773 test samples)
**Warmup**: 50 runs, **Trials**: 200 per batch size

## Batch Size vs Latency

| Batch Size | Avg Latency (ms) | Throughput (samples/s) | Per-Sample (μs) |
|-----------:|-----------------:|----------------------:|----------------:|
| 1 | 0.39 | 2555 | 391.4 |
| 8 | 0.40 | 19928 | 50.2 |
| 16 | 0.39 | 40928 | 24.4 |
| 32 | 0.38 | 84206 | 11.9 |
| 64 | 0.45 | 141313 | 7.1 |
| 128 | 0.39 | 328104 | 3.0 |
| 256 | 0.40 | 639964 | 1.6 |

## Observations

- **Best throughput**: batch size 256 (639964 samples/s)
- **Single-sample latency**: 0.39 ms
- **Device**: mps
