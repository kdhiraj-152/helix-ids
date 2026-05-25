#!/usr/bin/env bash
set -euo pipefail

# HELIX-IDS demo script
# Usage: ./demo.sh [all|export|serve|validate|gate|help]
# Requirements: python3, PYTHONPATH=src, FastAPI/uvicorn installed for serve_rest

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="python3"
CHECKPOINT="models/helix_full/helix_full_nsl_kdd_best.pt"
HOST="127.0.0.1"
PORT=18080
LOG=/tmp/helix_serve.log

function die() { echo "$@" >&2; exit 1; }
function ensure_pythonpath() { export PYTHONPATH="$REPO_ROOT/src"; }

function do_export() {
  ensure_pythonpath
  echo "Exporting inference bundle..."
  $PYTHON -u scripts/operations/export_inference_bundle.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir artifacts/releases/helix_demo_packaging \
    --temperature 1.0
}

function start_server() {
  ensure_pythonpath
  echo "Starting server on $HOST:$PORT (logs -> $LOG)"
  PYTHONPATH="$REPO_ROOT/src" $PYTHON -u scripts/operations/serve_rest.py \
    --checkpoint "$CHECKPOINT" --host "$HOST" --port "$PORT" --device cpu > "$LOG" 2>&1 &
  HELIX_PID=$!
  echo "HELIX PID: $HELIX_PID"
  for i in $(seq 1 60); do
    if curl -sSf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      echo "Server healthy"
      return 0
    fi
    sleep 0.5
  done
  die "Server didn't start in time; see $LOG"
}

function stop_server() {
  if [ -n "${HELIX_PID:-}" ]; then
    echo "Stopping server (PID $HELIX_PID)"
    kill "$HELIX_PID" || true
    wait "$HELIX_PID" 2>/dev/null || true
  fi
}

trap 'stop_server' EXIT

function run_sample_loop() {
  ensure_pythonpath
  echo "Running sample predict loop (3000 requests)..."
  $PYTHON - <<'PY'
import json, time
from pathlib import Path
from urllib import request

HOST='127.0.0.1'
PORT=18080

def post_predict(sample):
    payload = json.dumps({'features': sample}).encode('utf-8')
    req = request.Request(f'http://{HOST}:{PORT}/predict', data=payload, headers={'Content-Type':'application/json'}, method='POST')
    with request.urlopen(req, timeout=20):
        return

def get_metrics():
    with request.urlopen(f'http://{HOST}:{PORT}/metrics', timeout=20) as r:
        return r.read().decode('utf-8', errors='replace')

for _ in range(60):
    try:
        with request.urlopen(f'http://{HOST}:{PORT}/health', timeout=5):
            break
    except Exception:
        time.sleep(0.5)

sample = [0.0] * 17
for _ in range(3000):
    post_predict(sample)

print(get_metrics())
PY
}

function run_gate_check() {
  echo "Running staging gate check..."
  PYTHONPATH="$REPO_ROOT/src" $PYTHON scripts/operations/staging_gate_check.py --metrics-endpoint "http://$HOST:$PORT/metrics"
  echo "Gate check exit code: $?"
}

case "${1:-demo}" in
  all)
    do_export
    start_server
    run_sample_loop
    run_gate_check
    ;;
  export) do_export ;;
  serve) start_server && wait "$HELIX_PID" ;;
  validate)
    start_server
    run_sample_loop
    run_gate_check
    ;;
  gate) run_gate_check ;;
  demo)
    echo "Demo: running visualization (no-HELIX vs HELIX). This will start servers, send requests, and generate a comparison plot."
    echo
    cat <<'NARR'
Narration - what you'll see in the generated image:
  - Left panel: cumulative override rate vs request index.
      Shows how often HELIX applied a class override over time.
  - Middle panel: confidence histogram.
      Distribution of model confidence scores across requests.
  - Right panel: class-margin histogram.
      Distribution of class-margin values when margin overrides are present
      (may show 'No margin values captured' if none were recorded).
NARR
    echo
    echo "Running visualization now..."
    PYTHONPATH="$REPO_ROOT/src" $PYTHON scripts/operations/visualize_helix_demo.py --requests-per-run 200 --output-dir artifacts/operations/visuals
    IMG="$REPO_ROOT/artifacts/operations/visuals/helix_vs_nohelix.png"
    echo
    echo "Demo finished. Visual: $IMG"
    echo "Quick tips:"
    echo "  - To increase signal, run the visualization with more requests:"
    echo "      PYTHONPATH=src python3 scripts/operations/visualize_helix_demo.py --requests-per-run 3000"
    echo "  - To view the image on macOS: open $IMG"
    if command -v open >/dev/null 2>&1; then
      echo "Opening $IMG"
      open "$IMG" || true
    else
      echo "Run: open $IMG (macOS) or view the file in your image viewer."
    fi
    ;;
  help|--help|-h)
    sed -n '1,160p' "$0"
    ;;
  *)
    echo "Unknown command: $1"
    exit 2
    ;;
esac

# Example alternate server command (frozen adaptive + z-score)
# PYTHONPATH=src python3 scripts/operations/serve_rest.py \
#   --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
#   --host 127.0.0.1 --port 18081 --device cpu \
#   --class-margin-override-freeze-adaptive-tau \
#   --class-margin-override-frozen-tau-adaptive 70000 \
#   --class-margin-override-use-margin-zscore
