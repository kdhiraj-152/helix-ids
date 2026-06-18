# Deployment

> Last updated: 2026-06-18  
> Authoritative runbook for deploying HELIX-IDS inference service.

## Scope

- Service: `scripts/operations/serve_rest.py`
- Runtime: `src/helix_ids/operations/inference_runtime.py`
- Monitoring baseline: `src/helix_ids/operations/monitoring.py`
- Primary checkpoint: `models/helix_full/helix_full_nsl_kdd_best.pt`
- Gate checker: `scripts/operations/staging_gate_check.py`

## Deployment Stages

### Phase 1 — Local Validation

```bash
# Start server
PYTHONPATH=src python scripts/operations/serve_rest.py \
  --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
  --host 127.0.0.1 --port 18080 --device cpu

# Verify
# - Predictions stable for repeated same vectors
# - /metrics updates correctly
# - /health returns 200
```

**Pass criteria:**
- `/health` responds
- Predictions are deterministic for same input
- Metrics accessible at `/metrics`

### Phase 2 — Staging Rollout

```bash
PYTHONPATH=src python scripts/operations/serve_rest.py \
  --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
  --host 127.0.0.1 --port 18081 --device cpu
```

**Controlled traffic profiles:**
1. Conservative (single-request): override_rate → 0.0 → **PASS**
2. Aggressive (batch/stress): override_rate may exceed threshold → **FAIL**

**Hard gate:** `override_rate <= 0.02` enforced by `staging_gate_check.py`.

### Phase 3 — Production Release

**Prereqs:**
- Staging conservative profile stable with gate pass
- Monitoring collector scraping `/metrics`
- Incident routing configured for degraded state

```bash
PYTHONPATH=src python scripts/operations/serve_rest.py \
  --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
  --host 0.0.0.0 --port 8080 --device cpu
```

**Controlled production expansion:**
1. Route 5% traffic; observe ≥1000 requests
2. Validate: `override_rate <= 0.02` and `degraded_state == 0`
3. Increase to 10%, re-validate
4. Progress: 20% → 50% → 100%, validating at each step

**Required enforcement at all steps:**
- Keep `traffic_expansion_guard.py` running continuously
- Run `staging_gate_check.py` as hard pass/fail barrier before every increase

**Kill condition:** If `degraded_state == 1`, immediate halt and rollback.

### Phase 4 — Live Monitoring

Track continuously from `/metrics`:
- `helix_coverage_override_rate`
- `helix_class_predictions_total{class="*"}`
- `helix_class_entropy`
- `helix_requests_total`
- `helix_degraded_state`

### Phase 5 — Incident Rule

If `helix_coverage_override_rate > 0.02`, system is degraded.
`helix_degraded_state` flips to `1` automatically.

**Investigation trigger:**
1. Snapshot `/metrics`
2. Pull latest events from `artifacts/operations/live_events.jsonl`
3. Segment by traffic pattern
4. Confirm data shift vs runtime behavior

### Phase 6 — Data Capture

Every request is logged to `artifacts/operations/live_events.jsonl` with:
- Input features
- Prediction (family_class, confidence)
- Override event (applied, class/logit/threshold)
- UTC timestamp

This is directly usable as seed corpus for retraining and postmortems.

## Key Commands

```bash
# Start server
PYTHONPATH=src python scripts/operations/serve_rest.py \
  --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
  --host 127.0.0.1 --port 8080 --device cpu

# Run staging gate check
python scripts/operations/staging_gate_check.py --metrics-endpoint http://host:port/metrics

# Run traffic expansion guard
python scripts/operations/traffic_expansion_guard.py --metrics-endpoint http://host:port/metrics --interval 2
```

## Gate Logic

### staging_gate_check.py
- Parses `helix_coverage_override_rate` and `helix_degraded_state`
- **FAIL** if `override_rate > 0.02` OR `degraded_state == 1`
- Exit code 0 = PASS, exit code 1 = FAIL/BLOCK

### traffic_expansion_guard.py
- Continuous polling at configurable interval
- If `helix_degraded_state == 1`: prints HALT, exits 1
- Otherwise: prints OK, sleeps `interval` seconds

## CI/CD Flow

1. Deploy to staging
2. Run traffic simulation
3. Run `staging_gate_check.py`
4. Permit production only when exit code is 0

## Branch Governance

| Branch | Protection | Merge |
|--------|-----------|-------|
| `main` | PR required, status checks, squash-merge only | dev → main via PR |
| `dev` | Block deletion, allow force-push | Feature branches → dev |

**Main protection:** `required_status_checks = [CI, Quality, Architecture]`, 1 approval, `dismiss_stale_reviews=true`, `allow_force_pushes=false`, `allow_deletions=false`.
