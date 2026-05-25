#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_SOURCE_ROOT="${RAW_SOURCE_ROOT:-/Users/kdhiraj/Datasets/RP-2-raw}"
FORCE="${FORCE:-0}"

require_source() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing source path: $path" >&2
    exit 1
  fi
}

link_one() {
  local src="$1"
  local dst="$2"
  local parent
  parent="$(dirname "$dst")"

  require_source "$src"
  mkdir -p "$parent"

  if [[ -L "$dst" ]]; then
    local current_target
    current_target="$(readlink "$dst")"
    if [[ "$current_target" == "$src" ]]; then
      echo "Already linked: $dst -> $src"
      return
    fi
    rm "$dst"
  elif [[ -e "$dst" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      rm -rf "$dst"
    else
      echo "Destination exists and is not a symlink: $dst" >&2
      echo "Set FORCE=1 to replace existing destination." >&2
      exit 1
    fi
  fi

  ln -s "$src" "$dst"
  echo "Linked: $dst -> $src"
}

check_test_readiness() {
  local nsl_train="$ROOT_DIR/data/nsl_kdd/raw/KDDTrain+.txt"
  local unsw_train="$ROOT_DIR/data/unsw_nb15/raw/UNSW_NB15_training-set.csv"
  local cicids_raw="$ROOT_DIR/data/cicids2018/raw"

  [[ -f "$nsl_train" ]] || {
    echo "Missing NSL-KDD file: $nsl_train" >&2
    exit 1
  }
  [[ -f "$unsw_train" ]] || {
    echo "Missing UNSW file: $unsw_train" >&2
    exit 1
  }
  compgen -G "$cicids_raw/*.csv" > /dev/null || {
    echo "Missing CICIDS CSV files in: $cicids_raw" >&2
    exit 1
  }

  echo "Test-time raw dataset links are ready."
}

main() {
  echo "Repo root: $ROOT_DIR"
  echo "Raw source root: $RAW_SOURCE_ROOT"

  link_one "$RAW_SOURCE_ROOT/nsl_kdd/raw" "$ROOT_DIR/data/nsl_kdd/raw"
  link_one "$RAW_SOURCE_ROOT/unsw_nb15/raw" "$ROOT_DIR/data/unsw_nb15/raw"
  link_one "$RAW_SOURCE_ROOT/cicids2018/raw" "$ROOT_DIR/data/cicids2018/raw"

  check_test_readiness
}

main "$@"
