# Benchmark Protocol

Last updated: 2026-06-09

## Metrics

### Primary Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| Macro F1 | Unweighted mean of per-class F1 scores | Overall detection quality (class-balanced) |
| Binary F1 | F1 for Normal vs. Attack | Overall detection quality |
| Per-class F1 | F1 per attack family (7 classes) | Individual class detection quality |
| Threat-Weighted F1 | `sum(w_i * F1_i) / sum(w_i)` where w = threat weight | Rare-class-sensitive metric |

### Secondary Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| Accuracy | Correct predictions / total | General correctness (misleading with imbalance) |
| Precision | TP / (TP + FP) | False alarm rate control |
| Recall | TP / (TP + FN) | Detection rate |
| False Positive Rate | FP / (FP + TN) | Operational cost |
| AUC-ROC | Area under ROC curve | Ranking quality |

### Production Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| Latency (p50/p95/p99) | Inference time per request | User experience SLA |
| Throughput | Requests per second | Capacity planning |
| Memory (RSS) | Resident set size | Edge deployment viability |
| CPU utilization | % CPU used | Hardware sizing |
| Zero-prediction rate | % of requests where a class receives zero predictions | Drindicator |

### Adversarial Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| Adversarial Robustness | Accuracy under FGSM/PGD attack | Security evaluation |
| Perturbation tolerance | Max epsilon before prediction flips | Robustness boundary |

## Baselines

| Baseline | Method | Source |
|----------|--------|--------|
| Random Forest | sklearn RandomForestClassifier | Baseline ML |
| SVM (RBF) | sklearn SVC | Baseline ML |
| MLP (2-layer) | Simple neural network | Ablation baseline |
| HELIX (no attention) | HELIX without TAM | Ablation |
| HELIX (no threshold decoupling) | HELIX without per-class margins | Ablation |
| HELIX (full) | HELIX with all components | Proposed method |

(NOT YET RUN — baseline benchmark results are not archived)

## Hardware Targets

| Target | Expected Latency (p95) | Expected Throughput | Status |
|--------|----------------------|-------------------|--------|
| Server (GPU) | <5ms | >1000 req/s | Not benchmarked |
| Server (CPU) | <50ms | >100 req/s | Not benchmarked |
| RPi 4 | <100ms | >10 req/s | Config exists |
| RPi Zero | <500ms | >2 req/s | Config exists |
| ESP32 | — | — | Model not quantized yet |

## Latency Measurements

Procedure:
```bash
# Server latency benchmark
PYTHONPATH=src python scripts/operations/serve_rest.py \
    --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
    --port 8080 &
    
# From another terminal
python -c "
import requests, time, numpy as np
import numpy as np
features = np.random.randn(41).tolist()
times = []
for _ in range(1000):
    start = time.time()
    requests.post('http://localhost:8080/predict', json={'features': features})
    times.append((time.time() - start) * 1000)
print(f'p50: {np.percentile(times, 50):.1f}ms')
print(f'p95: {np.percentile(times, 95):.1f}ms')
print(f'p99: {np.percentile(times, 99):.1f}ms')
"
```

(NOT YET RUN — no archived latency results)

## Throughput Measurements

Procedure:
```bash
# Using Apache Bench or similar
ab -n 10000 -c 10 -p request.json \
    http://localhost:8080/predict
```

(NOT YET RUN — no archived throughput results)

## Memory Measurements

Procedure:
```python
import psutil, os
process = psutil.Process(os.getpid())
rss_mb = process.memory_info().rss / 1024 / 1024
print(f"RSS: {rss_mb:.1f} MB")
```

## Power Measurements

(NOT YET SET UP — requires hardware power monitoring equipment)

- RPi 4: USB power monitor
- ESP32: Current measurement with multimeter
- Server: IPMI/iDRAC power reporting

## Evaluation Procedure

### Holdout Evaluation
```bash
PYTHONPATH=src python scripts/evaluation/holdout_evaluation_v2.py \
    --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt
```

### Benchmark Orchestration
```bash
PYTHONPATH=src python scripts/evaluation/benchmarks.py \
    --manifest config/experiments/smoke.yaml \
    --experiments-dir config/experiments
```

### Adversarial Evaluation
```bash
PYTHONPATH=src python -m pytest tests/test_adversarial_robustness.py -v
```

## Reproducibility Rules

1. **Seed recorded**: Every benchmark run must record the seed used
2. **Environment snapshotted**: `requirements.txt` + `pip freeze` archive
3. **Hardware recorded**: `lscpu`, `nvidia-smi`, `uname -a`
4. **Dataset version recorded**: Use dataset fingerprint (SHA-256)
5. **Results archived**: JSON output saved alongside provenance manifest
6. **Multiple runs**: Minimum 3 runs per configuration for statistical significance
7. **Outlier handling**: Report median, p50/p95/p99 for latency; mean ± std for accuracy

### Results format:
```json
{
  "benchmark_id": "bench_20260609_abc123",
  "run_id": "run_42",
  "hardware": {"cpu": "...", "gpu": "...", "ram_gb": 64},
  "seed": 42,
  "metrics": {
    "macro_f1": 0.923,
    "binary_f1": 0.985,
    "threat_weighted_f1": 0.887,
    "per_class_f1": [0.99, 0.95, 0.91, 0.85, 0.72, 0.88, 0.90]
  },
  "latency_ms": {"p50": 2.1, "p95": 4.8, "p99": 8.3},
  "throughput_rps": 1200,
  "dataset_fingerprint": "abc123...",
  "model_hash": "def456..."
}
```

## Benchmarks Not Yet Run

This document defines the protocol. The benchmarks themselves have not been executed.

1. Server latency benchmarks (code exists, not run)
2. Edge latency benchmarks (config exists, not run)
3. Throughput benchmarks (code exists, not run)
4. Memory benchmarks (code exists, not run)
5. Power benchmarks (hardware setup required)
6. Baseline comparisons (Random Forest, SVM, MLP not benchmarked)
7. Ablation benchmarks (not yet written)
8. Results archive (no JSON output from prior runs preserved)
9. Statistical significance analysis (multiple seeds not run)