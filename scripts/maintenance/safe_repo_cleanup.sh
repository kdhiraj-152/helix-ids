#!/usr/bin/env bash
set -euo pipefail

# Safe cleanup utility.
# Defaults to dry-run. Use --apply to delete.

usage() {
  cat <<'EOF'
Usage: safe_repo_cleanup.sh [--apply] [--dry-run] [--profile safe|ultra-lean] [--purge-venv] [--help]

Profiles:
  safe        Remove transient local artifacts only (default)
  ultra-lean  Includes safe profile plus bulky generated/data artifacts

Flags:
  --apply         Perform deletions (default is dry-run)
  --dry-run       Show what would be deleted
  --profile NAME  Cleanup profile: safe or ultra-lean
  --purge-venv    Also remove .venv and venv directories (optional)
  --help          Show this message
EOF
}

MODE="dry-run"
PROFILE="safe"
PURGE_VENV="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      MODE="apply"
      ;;
    --dry-run)
      MODE="dry-run"
      ;;
    --profile)
      shift
      PROFILE="${1:-}"
      ;;
    --purge-venv)
      PURGE_VENV="true"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if [[ "$PROFILE" != "safe" && "$PROFILE" != "ultra-lean" ]]; then
  echo "Invalid profile: $PROFILE (expected safe|ultra-lean)" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"

name_patterns=(
  ".DS_Store"
)

cache_dirs=(
  "__pycache__"
  ".pytest_cache"
  ".mypy_cache"
  ".ruff_cache"
)

root_files=(
  "train_output.log"
  ".coverage"
  "coverage.xml"
)

ultra_dirs=(
  "checkpoints"
  "CICDS2018"
  ".code-review-graph"
  "logs"
  "evaluate"
  "results/benchmarks"
  "results/figures"
  "results/helix_full"
  "results/unified_training"
  "results/unsw_anomaly_analysis"
  "results/unsw_only_cleaned"
  "results/v2_fixed"
)

echo "[safe-cleanup] repo: $repo_root"
echo "[safe-cleanup] mode: $MODE"
echo "[safe-cleanup] profile: $PROFILE"
echo "[safe-cleanup] purge_venv: $PURGE_VENV"

find_args=(
  "$repo_root"
  "("
  -path "$repo_root/.git"
  -o -path "$repo_root/.venv"
  -o -path "$repo_root/venv"
  ")"
  -prune
  -o
)

delete_file() {
  local path="$1"
  local rel="${path#$repo_root/}"
  if [[ "$MODE" == "apply" ]]; then
    rm -f "$path"
    echo "deleted file: $rel"
  else
    echo "would delete file: $rel"
  fi
}

delete_dir() {
  local path="$1"
  local rel="${path#$repo_root/}"
  if [[ "$MODE" == "apply" ]]; then
    rm -rf "$path"
    echo "deleted dir: $rel"
  else
    echo "would delete dir: $rel"
  fi
}

# Files matched by name (anywhere in repo)
for p in "${name_patterns[@]}"; do
  while IFS= read -r -d '' match; do
    delete_file "$match"
  done < <(find "${find_args[@]}" -type f -name "$p" -print0)
done

# Cache directories matched by name (anywhere in repo)
for d in "${cache_dirs[@]}"; do
  while IFS= read -r -d '' match; do
    delete_dir "$match"
  done < <(find "${find_args[@]}" -type d -name "$d" -print0)
done

# Specific root files
for f in "${root_files[@]}"; do
  target="$repo_root/$f"
  if [[ -e "$target" ]]; then
    delete_file "$target"
  fi
done

if [[ "$PROFILE" == "ultra-lean" ]]; then
  # Drop large generated/derived directories.
  for d in "${ultra_dirs[@]}"; do
    target="$repo_root/$d"
    if [[ -e "$target" ]]; then
      delete_dir "$target"
    fi
  done

  # Keep small JSON metadata in processed data, remove bulky tabular artifacts.
  while IFS= read -r -d '' artifact; do
    delete_file "$artifact"
  done < <(find "$repo_root/data/processed" -type f \( -name '*.csv' -o -name '*.txt' -o -name '*.parquet' -o -name '*.feather' -o -name '*.pkl' -o -name '*.joblib' -o -name '*.pt' \) -print0 2>/dev/null || true)

  # Keep README and gates metadata in results root.
  while IFS= read -r -d '' artifact; do
    rel="${artifact#$repo_root/}"
    case "$rel" in
      results/README.md|results/gates/*)
        ;;
      *)
        delete_file "$artifact"
        ;;
    esac
  done < <(find "$repo_root/results" -type f -print0 2>/dev/null || true)

  if [[ "$PURGE_VENV" == "true" ]]; then
    [[ -d "$repo_root/.venv" ]] && delete_dir "$repo_root/.venv"
    [[ -d "$repo_root/venv" ]] && delete_dir "$repo_root/venv"
  fi
fi

echo "[safe-cleanup] complete"
