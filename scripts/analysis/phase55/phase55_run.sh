#!/bin/bash
# Phase 55 Runner
# Usage: bash scripts/analysis/phase55/phase55_run.sh [experiments] [args]
#   experiments: comma-separated (A,B,C) or empty for all
#   args: --skip-train, etc.

cd "$(dirname "$0")/../../.."
export PYTHONPATH=src

EXPERIMENTS="${1:-A,B,C,D,E,F,G,H}"
shift 2>/dev/null || true

CMD=".venv311/bin/python -u scripts/analysis/phase55/phase55_main.py --experiments $EXPERIMENTS $@"

echo "Running: $CMD"
echo "Output: results/phase55/phase55_run.log"
echo ""

# Run in foreground
$CMD 2>&1 | tee /tmp/phase55_console.log

echo ""
echo "Done. Check results/phase55/ for outputs."
