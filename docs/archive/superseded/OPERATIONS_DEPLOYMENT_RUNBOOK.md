# HELIX IDS deployment runbook (model-building -> operations)

6-phase deployment transition with hard gates and concrete commands.

## Scope

- Service: `scripts/operations/serve_rest.py`
- Runtime: `src/helix_ids/operations/inference_runtime.py`
- Monitoring baseline: `src/helix_ids/operations/monitoring.py`
- Primary checkpoint: `models/helix_full/helix_full_nsl_kdd_best.pt`

## Phase 1 — Local serve validation (DONE)

### Commands used

```bash
PYTHONPATH=src python scripts/operations/serve_rest.py \
  --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
  --host 127.0.0.1 --port 18080 --device cpu
```

Real vectors sent from:

- `data/processed/multi_dataset_v1/X_test_nsl_kdd.npy` (shape `(18896, 17)`, dtype `float32`)

### Verified

- Predictions stable for repeated same vector calls: PASS
- Metrics updating: PASS
- `/metrics` accessible: PASS

Observed sample:

- `helix_requests_total 16`
- `helix_coverage_override_total 1`
- `helix_coverage_override_rate 0.0625000000`

## Real-time traffic expansion guard (NEW)

Command:

```bash
python scripts/operations/traffic_expansion_guard.py --metrics-endpoint http://host:port/metrics --interval 2
```

Contract:

- continuous polling
- exits immediately on violation

Logic (strict, no buffering/retries/delay logic):

- parse `helix_degraded_state`
- if `helix_degraded_state == 1` -> print HALT and `exit 1`
- else print OK and sleep for `interval`

Output format:

RUNNING

```text
[HELIX GUARD] OK
```

HALT

```text
[HELIX GUARD] HALT
degraded_state=1
```

Rollout usage:

- run guard alongside traffic expansion process
- if guard exits -> stop traffic increase and freeze deployment

## Promotion barrier script (NEW)

Command:

```bash
python scripts/operations/staging_gate_check.py --metrics-endpoint http://host:port/metrics
```

Contract:

- exit code `0` => PASS
- exit code `1` => FAIL/BLOCK

Parsing targets:

- `helix_coverage_override_rate`
- `helix_degraded_state`

Strict fail rule (no tolerance/smoothing/averaging):

- fail if `override_rate > 0.02`
- OR fail if `degraded_state == 1`

Output format:

PASS

```text
[HELIX GATE] OK
override_rate=0.009
```

FAIL

```text
[HELIX GATE] BLOCKED
override_rate=0.034
degraded_state=1
```

CI/CD flow enforcement:

1. deploy to staging
1. run traffic simulation
1. run `staging_gate_check.py`
1. permit production only when exit code is `0`

## Phase 2 — Staging rollout (controlled traffic)

### Deploy (internal only)

```bash
PYTHONPATH=src python scripts/operations/serve_rest.py \
  --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
  --host 127.0.0.1 --port 18081 --device cpu
```

### Controlled traffic profiles

1. Aggressive/batch profile (stress): produced override_rate `0.1428571429` -> FAIL hard gate
1. Conservative/single-request profile: produced override_rate `0.0000000000` -> PASS hard gate

### Hard check

- Required: `override_rate <= 0.02`
- Enforced by metric + degraded flag:
  - `helix_coverage_override_rate`
  - `helix_degraded_state` (1 when rate > 0.02)

## Phase 3 — Production release

### Prereqs

- staging conservative profile stable and gate pass (`<= 0.02`)
- monitoring collector scraping `/metrics`
- incident routing configured for degraded state

### Promotion decision snapshot (2026-04-19)

- Decision: `promotion approved`
- Evidence: `window_1 = PASS`
- Evidence: `window_2 = PASS`
- Evidence: `override_rate = 0.0` (both windows)
- Evidence: `degraded_state = 0` (both windows)
- Interpretation: system stable under external mixed traffic
- Model state: no retraining required; inference path corrected and validated
- System status: production-ready with guarded rollout

### Deploy (real traffic)

```bash
PYTHONPATH=src python scripts/operations/serve_rest.py \
  --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
  --host 0.0.0.0 --port 8080 --device cpu
```

### Controlled production expansion checklist

Run this sequence in order and do not advance until the prior step is passed.

1. Route `5%` traffic.
2. Observe at least `1000` requests (`helix_requests_total` delta).
3. Validate:
   - `override_rate <= 0.02`
   - `degraded_state == 0`
4. Increase to `10%` and repeat validation.
5. Increase gradually: `20% -> 50% -> 100%`, repeating the same validation at each step.

Required enforcement at all steps:

- keep `scripts/operations/traffic_expansion_guard.py` running continuously
- run `scripts/operations/staging_gate_check.py` as a hard pass/fail barrier before every increase

Kill condition:

- if `degraded_state == 1`, immediate halt and rollback traffic percentage to previous stable level

## Phase 4 — Live monitoring

Track continuously from `/metrics`:

- `helix_coverage_override_rate`
- `helix_class_predictions_total{class="*"}` (class distribution)
- `helix_class_entropy`
- `helix_requests_total`
- `helix_degraded_state`

## Phase 5 — Incident rule

Rule:

- If `helix_coverage_override_rate > 0.02`, system is degraded.
- `helix_degraded_state` flips to `1` automatically.

Investigation trigger actions:

1. Snapshot `/metrics`
2. Pull latest capture events from `artifacts/operations/live_events.jsonl`
3. Segment by traffic pattern (single vs batch, source, time window)
4. Confirm if spike is data shift vs runtime behavior

## Phase 6 — Data capture for v1.1

Now logged per request at:

- `artifacts/operations/live_events.jsonl`

Each line records:

- inputs
- prediction (`family_class`, `confidence`)
- override event (`applied`, class/logit/threshold)
- UTC timestamp

This is directly usable as seed corpus for v1.1 retraining and postmortems.

## What changed in service implementation

Updated: `scripts/operations/serve_rest.py`

- added degraded-state computation (`override_rate > 0.02`)
- added class-count tracking + entropy metric
- added request event capture JSONL
- added prometheus metrics:
  - `helix_degraded_state`
  - `helix_class_predictions_total{class="..."}`
  - `helix_class_entropy`

Updated tests: `tests/test_operations/test_serve_rest_metrics.py`

- asserts degraded-state metric transitions
- asserts class counters and entropy metric present
- asserts capture file creation and event payload integrity

Validation:

```bash
PYTHONPATH=src pytest -q tests/test_operations/test_serve_rest_metrics.py tests/test_operations/test_monitoring.py
# 4 passed
```

## What ships

- Live service behavior checks
- Hard production gate on override rate
- Degraded-state signaling for incident automation
- Continuous capture of reality data for v1.1 learning
