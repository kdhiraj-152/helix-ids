# CLI & API Reference

> Last updated: 2026-06-18

## REST API

The inference server (`serve_rest.py`) exposes three endpoints.

### GET /health

Liveness check.

**Response**: `200 OK` (plain text)

### POST /predict

Run inference on a feature vector.

**Request:**
```json
{
  "features": [0.0, 0.1, ..., 0.16]  // 17-element float32 array
}
```

**Response:**
```json
{
  "prediction": 0,          // Binary: 0=Normal, 1=Attack
  "confidence": 0.95,
  "family_class": 2,        // Attack family (0-6)
  "family_probs": [0.01, 0.02, 0.90, 0.03, 0.02, 0.01, 0.01],
  "override_applied": false,
  "timestamp": "2026-06-18T12:00:00Z"
}
```

### GET /metrics

Prometheus-format metrics snapshot.

**Response**: Plain text with Prometheus exposition format.

## CLI — Training

```bash
PYTHONPATH=src python scripts/training/train_helix_ids_full.py \
    --config config/experiments/smoke.yaml \
    --output /path/to/output \
    --device cpu \
    --epochs 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | Experiment config YAML path |
| `--output` | — | Output directory for checkpoints |
| `--device` | `cpu` | `cpu`, `cuda:0`, `mps` |
| `--epochs` | 50 | Max training epochs |
| `--seed` | 42 | Random seed |
| `--ab-baseline` | — | Ablation baseline checkpoint path |

## CLI — Inference Server

```bash
PYTHONPATH=src python scripts/operations/serve_rest.py \
    --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
    --host 127.0.0.1 --port 8080 --device cpu
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | — | Model checkpoint path (required) |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8080` | Bind port |
| `--device` | `cpu` | Inference device |
| `--global-coverage-quantile` | `1.0` | Coverage quantile for override layer |

## CLI — Staging Gate

```bash
python scripts/operations/staging_gate_check.py --metrics-endpoint http://host:port/metrics
```

Exit code 0 = PASS, exit code 1 = FAIL.

## CLI — Quantization

```bash
# Lite variant
PYTHONPATH=src python scripts/quantize_helix_lite.py \
    --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
    --output models/quantized/

# Micro variant (ESP32)
PYTHONPATH=src python scripts/quantize_helix_micro.py \
    --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
    --output models/quantized/
```

## CLI — Benchmark

```bash
# Full benchmark suite
PYTHONPATH=src python scripts/evaluation/benchmarks.py \
    --experiment config/experiments/smoke.yaml

# Load test
python scripts/benchmarks/load_test.py --endpoint http://127.0.0.1:8080

# Soak test
python scripts/benchmarks/soak_inference.py --duration 24
```

## CLI — One-Shot Reproduce

The following trains, serves, requests, and validates in a single command:

```bash
source .venv311/bin/activate && \
PYTHONPATH=src python3 scripts/training/train_helix_ids_full.py \
  --config config/helix_config.yaml \
  --output models/helix_full \
  --device cpu \
  --epochs 10 && \
(PYTHONPATH=src python3 scripts/operations/serve_rest.py \
    --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
    --host 127.0.0.1 --port 8080 --device cpu --global-coverage-quantile 1.0 \
    >/tmp/helix_serve.log 2>&1 & HELIX_PID=$!; \
# ... (see README.md for full inline script) \
kill $HELIX_PID)
```
