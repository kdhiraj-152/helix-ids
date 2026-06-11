# HELIX-IDS

Some years ago I got fed up with intrusion detection systems that either
(a) worked great on paper but fell apart on real network traffic, or
(b) were too heavy to run anywhere but a beefy server in a data center.

That second one bothered me most. Good network security should not require
expensive hardware at every network tap. So I built HELIX-IDS — a detection
system designed from the ground up to run on a Raspberry Pi or even an ESP32
microcontroller, while still matching (and in some cases beating) what the
big server-based systems could do.

## What it does

HELIX-IDS ingests network flow features (NSL-KDD, UNSW-NB15, CICIDS-2018
formats) and classifies traffic into normal operation or one of several attack
families (DoS, Probe, R2L, U2R, and the specific sub-types in each dataset).
It uses a neural network architecture with temporal attention, domain
adaptation (so you can train on one dataset and deploy on another), and a
hierarchical classifier head that treats rare attack classes differently
from common ones — because the whole point is catching the thing that
almost never happens.

The system runs on three tiers:

- **Server / cloud**: Training, evaluation, heavy inference
- **Raspberry Pi (4 and Zero)**: Optimized inference, smaller model variants
- **ESP32**: Minimal quantized model, does the basics on a microcontroller

## What makes this different

**Edge-first design.** Most NIDS work starts with a server model and then
tries to shrink it. I started with the question "what can we fit on a Pi?"
and built up from there. The tradeoffs are explicit — the "Nano" variant for
ESP32, "Lite" for Pi Zero, and "Full" for server. No hidden assumptions about
available compute.

**Rare-class awareness.** In network intrusion, the dangerous attacks are the
ones that almost never show up in training data (R2L, U2R). Standard loss
functions wash them out. HELIX uses a threat-weighted focal loss that
amplifies the signal from rare classes without destroying overall accuracy.

**Provenance on every artifact.** Every model checkpoint, every training run,
every processed dataset gets a SHA-256 hash recorded in a manifest. Not
because I wanted to, but because I got tired of asking "wait, which model did
that number come from?" and having no answer. The provenance chain makes
every result in the paper independently verifiable.

**Safe deployment gates.** The runtime monitors itself — coverage override
rate, degraded state, request throughput. If the model starts guessing too
often (override rate climbs), the system flags itself before anyone has to
page. The staging gate check enforces this before any deployment is accepted.

## Current state

The repo is in formalization mode — the pipeline is locked for paper
reproducibility. No new features, no new scripts, no refactors. Everything
here exists to produce the numbers in the manuscript and let anyone else
reproduce them verbatim.

### Last validated staging results

| Metric | Value |
|---|---|
| Window 1 requests | 1500 |
| Window 2 requests | 1500 |
| Coverage override rate | 0.0 |
| Degraded state | 0 |

Artifacts live in `docs/reports/` and `docs/figures/`.

## One-shot reproduce

The following bash command trains the model, starts a REST server, fires
3000 requests at it, collects metrics, and runs the staging gate check.
If you have the venv set up (`.venv311/`), this is all you need:

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
Path('docs/reports/staging_validation.json').write_text(
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

## Project layout

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

## Manuscript and figures

- Manuscript: `docs/manuscript/HELIX_submission_ready.md`
- Figures: `docs/fig/` and `docs/figures/`

## A few notes if you're poking around

- Everything expects `PYTHONPATH=src` — I should probably make this a
  proper installable package but for the paper pipeline this works.
- The gating logic lives in two files: `serve_rest.py` emits the metrics,
  `staging_gate_check.py` reads them and decides pass/fail.
- Benchmarks are orchestrated from `scripts/evaluation/benchmarks.py` which
  reads experiment manifests from `config/experiments/*.yaml`.
- If you're lost, `docs/README.md` has the full doc index.
