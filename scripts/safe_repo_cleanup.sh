#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper. Canonical script moved to scripts/maintenance/safe_repo_cleanup.sh.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/maintenance/safe_repo_cleanup.sh" "$@"
