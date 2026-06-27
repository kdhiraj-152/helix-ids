#!/usr/bin/env bash
# Run Phase 52 experiments sequentially as separate processes.
# Each experiment runs in its own Python interpreter so memory is fully
# reclaimed between them (prevents cumulative OOM kills on MPS/16GB).

set -euo pipefail

cd /Users/kdhiraj/Downloads/RP-2
source .venv311/bin/activate
export PYTHONPATH=src
export PYTORCH_ENABLE_MPS_FALLBACK=1

RESULTS_DIR="results/phase52"
TABLES_DIR="${RESULTS_DIR}/tables"
mkdir -p "$TABLES_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_exp() {
    local exp_id="$1"
    local name="$2"
    local logfile="/tmp/phase52_${exp_id}.log"
    
    log "=== Running Experiment ${exp_id}: ${name} ==="
    if python3 -u scripts/analysis/phase52_main.py \
        --experiments "${exp_id}" --skip-stats 2>&1 | tee "$logfile"; then
        log "=== Experiment ${exp_id}: ${name} SUCCESS ==="
    else
        local rc=$?
        log "=== Experiment ${exp_id}: ${name} FAILED (exit code ${rc}) ==="
        tail -20 "$logfile"
        return $rc
    fi
}

# A and B already done from the first run
# Check if they're complete
if ls "${TABLES_DIR}"/expA_*_metrics.json 1>/dev/null 2>&1; then
    log "Skipping Experiment A (cached results found)"
else
    log "WARNING: Experiment A results missing, running..."
    run_exp "A" "Latent Dimension Ablation"
fi

if ls "${TABLES_DIR}"/expB_*_metrics.json 1>/dev/null 2>&1; then
    log "Skipping Experiment B (cached results found)"
else
    log "WARNING: Experiment B results missing, running..."
    run_exp "B" "Encoder Depth Ablation"
fi

# Run C (temperature) first while MPS is fresh — it's most sensitive
run_exp "C" "Temperature Sweep"
run_exp "D" "Loss Weight Ablation"
run_exp "E" "Label Noise Robustness"
run_exp "F" "Sample Efficiency"

# Then A and B (latent dim and depth — less sensitive)
run_exp "A" "Latent Dimension Ablation"  
run_exp "B" "Encoder Depth Ablation"

log "=== All experiments complete. Running report generation. ==="

# Final pass: generate all deliverables from cached results
python3 -u scripts/analysis/phase52_main.py --experiments A,B,C,D,E,F 2>&1 | tee /tmp/phase52_report.log

log "=== Phase 52 complete ==="
echo ""
echo "Results: ${RESULTS_DIR}/"
echo "Logs: /tmp/phase52_*.log"
