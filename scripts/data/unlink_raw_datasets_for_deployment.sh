#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

unlink_one() {
  local dst="$1"
  if [[ -L "$dst" ]]; then
    rm "$dst"
    echo "Removed symlink: $dst"
  else
    echo "Skipped (not a symlink): $dst"
  fi
}

main() {
  echo "Repo root: $ROOT_DIR"
  unlink_one "$ROOT_DIR/data/nsl_kdd/raw"
  unlink_one "$ROOT_DIR/data/unsw_nb15/raw"
  unlink_one "$ROOT_DIR/data/cicids2018/raw"
  echo "Deployment cleanup complete. Raw dataset symlinks detached."
}

main "$@"
