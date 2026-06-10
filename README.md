# HELIX-IDS

Production-aligned IDS runtime with invariant-based deployment gating and paper-grade staging validation artifacts.

## System Freeze

Formalization mode is active for this repository state:

- no new features
- no new scripts
- no refactors
- current runtime pipeline locked

## Project Layout

```
RP-2/
├── src/helix_ids/               # Core package
├── config/                      # Experiment configs
├── scripts/
│   ├── training/                # Training pipelines
│   ├── operations/              # Serving & deployment
│   ├── evaluation/              # Benchmark orchestration
│   ├── data/                    # Data processing
│   ├── deployment/              # Deployment tooling
│   └── ci/                      # CI validators
├── tests/                       # Test suite
├── docs/
│   ├── README.md                # Doc index
│   ├── architecture/            # System design, models, schemas
│   ├── development/             # Training methodology, data pipeline
│   ├── operations/              # Deployment runbooks, checkpoint audit
│   ├── reports/                 # Audits, reviews, benchmarks
│   ├── governance/              # ADRs, hash authority, contracts
│   ├── manuscript/              # Paper drafts
│   ├── results/                 # Staging validation artifacts
│   └── archives/                # Historical phase documentation
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Final Staging Validation Artifacts

- `docs/results/staging_validation.json`
- `docs/figures/override_rate_vs_requests.png`
- `docs/figures/degraded_state_timeline.png`
- `docs/figures/batch_vs_single_consistency.png`

Current recorded outcome in `docs/results/staging_validation.json`:

- window 1: 1500 requests
- window 2: 1500 requests
- final override rate: 0.0
- final degraded state: 0

## One-Command Reproducibility Path

The command below executes train -> deploy -> validate in one shell command.

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

## Paper Artifacts

- Manuscript: `docs/manuscript/HELIX_submission_ready.md`
- Figures: `docs/fig/` and `docs/fig_revamp/`

## Notes

- For reproducible paper runs, keep `PYTHONPATH=src` set for all script invocations.
- Runtime gating logic is implemented through `scripts/operations/serve_rest.py` metrics and `scripts/operations/staging_gate_check.py`.
- Governed benchmark orchestration lives in `scripts/evaluation/benchmarks.py` and reads manifests from `config/experiments/*.yaml`.
- Documentation index: `docs/README.md`
