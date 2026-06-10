# HELIX-IDS

Network intrusion detection. Edge-to-cloud. Governed training and deployment.

This repo is in **formalization mode** — no new features, no new scripts, no refactors. The pipeline is locked for paper reproducibility.

## Layout

```
src/helix_ids/          Core package
config/                 Experiment configs
scripts/
  training/             Training pipelines
  operations/           Serving, gating, deployment
  evaluation/           Benchmark orchestration
  data/                 Data processing
  deployment/           Deployment tooling
  ci/                   CI validators
tests/                  Test suite
docs/
  architecture/         System design, models, schemas
  development/          Training methodology
  operations/           Runbooks, checkpoint audit
  reports/              Audits, reviews, benchmarks
  governance/           ADRs, hash authority, contracts
  manuscript/           Paper drafts
  results/              Staging validation artifacts
  archives/             Historical phase docs
```

## Staging Validation (Current)

Artifacts in `docs/results/` and `docs/figures/`:

| Metric | Value |
|---|---|
| Window 1 | 1500 requests |
| Window 2 | 1500 requests |
| Override rate | 0.0 |
| Degraded state | 0 |

## One-Shot Reproduce

This does train → deploy → validate in one command:

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
python3 - <<'PY'
import json
import time
from pathlib import Path
from urllib import request

def post_predict(sample):
    payload = json.dumps({'features': sample}).encode('utf-8')
    req = request.Request('http://127.0.0.1:8080/predict', data=payload, headers={'Content-Type':'application/json'}, method='POST')
    with request.urlopen(req, timeout=20):
        return

def get_metrics():
    with request.urlopen('http://127.0.0.1:8080/metrics', timeout=20) as r:
        return r.read().decode('utf-8', errors='replace')

def parse_metrics(text):
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or ' ' not in line:
            continue
        k, v = line.split(None, 1)
        try:
            out[k] = float(v.strip())
        except ValueError:
            pass
    return out

for _ in range(60):
    try:
        with request.urlopen('http://127.0.0.1:8080/health', timeout=5):
            break
    except Exception:
        time.sleep(0.5)

sample = [0.0] * 17
for _ in range(3000):
    post_predict(sample)

metrics_text = get_metrics()
metrics = parse_metrics(metrics_text)
Path('docs/results').mkdir(parents=True, exist_ok=True)
Path('docs/results/staging_validation.json').write_text(
    json.dumps(
        {
            'total_requests': int(metrics.get('helix_requests_total', 0.0)),
            'override_rate': float(metrics.get('helix_coverage_override_rate', 0.0)),
            'degraded_state': int(metrics.get('helix_degraded_state', 0.0)),
        },
        indent=2,
    ),
    encoding='utf-8',
)
print(metrics_text)
PY
python3 scripts/operations/staging_gate_check.py --metrics-endpoint http://127.0.0.1:8080/metrics; \
kill $HELIX_PID)
```

## Paper

- Manuscript: `docs/manuscript/HELIX_submission_ready.md`
- Figures: `docs/fig/` and `docs/fig_revamp/`

## Notes

- Set `PYTHONPATH=src` for all script invocations.
- Gating: `scripts/operations/serve_rest.py` metrics → `scripts/operations/staging_gate_check.py`.
- Benchmark orchestration: `scripts/evaluation/benchmarks.py` reads `config/experiments/*.yaml`.
- Doc index at `docs/README.md`.
