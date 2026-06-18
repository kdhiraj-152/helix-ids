# Monitoring

> Last updated: 2026-06-18

## Metrics Endpoint

The REST inference server exposes Prometheus-format metrics at `GET /metrics`.

### Core Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `helix_requests_total` | Counter | Total requests received |
| `helix_coverage_override_total` | Counter | Total override events |
| `helix_coverage_override_rate` | Gauge | Fraction of requests with override actuation |
| `helix_degraded_state` | Gauge | 1 if override_rate > 0.02, else 0 |
| `helix_class_predictions_total{class="..."}` | Counter | Per-class prediction count |
| `helix_class_entropy` | Gauge | Entropy of class distribution |

### Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| `helix_coverage_override_rate` | > 0.01 | > 0.02 |
| `helix_degraded_state` | — | == 1 |

## Monitored Components

### Training Monitoring
- Training step latency
- Loss convergence
- Logit saturation (warn when `|logit| > 10.0`)
- Gradient norm (clipped at `max_grad_norm` config)

### Inference Monitoring
- Request throughput and latency (p50/p95/p99)
- Coverage override rate
- Class distribution entropy
- Memory (RSS)
- Request event capture to JSONL

### System Health
- Liveness check (`GET /health`)
- Process-level metrics
- Checkpoint integrity
- Thread count

## Alerting

**Immediate incident:**
- `helix_degraded_state == 1`
- Coverage override rate sustained > 0.02

**Investigation workflow:**
1. Snapshot `/metrics`
2. Pull `artifacts/operations/live_events.jsonl`
3. Segment by traffic pattern
4. Diagnose data shift vs runtime behavior
5. If confirmed degraded: rollback traffic, freeze deployment

## Telemetry

### Per-Request Capture
Every prediction is logged to `artifacts/operations/live_events.jsonl`:
- Timestamp (UTC)
- Input feature vector
- Prediction (class, confidence)
- Override state (applied, class, threshold)
- Confidence calibration output

### Soak Testing Telemetry
Hourly snapshots during certification runs monitor:
- RSS memory (psutil)
- GPU memory (CUDA/MPS)
- File handles
- Thread count
- GC stats
- Tensor count (torch)
- Log volume
- Latency / throughput

Snapshots: `artifacts/soak/<run_id>/snapshot_<timestamp>.json`

Trend analysis: `summarize_run()` computes start/end/min/max/mean/trend for each metric.

## Dashboards

No permanent dashboard infrastructure in repo. Monitoring is done via:
- Direct `/metrics` scraping
- Prometheus (if configured in target environment)
- CI artifact collection for batch runs
