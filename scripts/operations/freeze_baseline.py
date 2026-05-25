#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from helix_ids.operations.baseline_freeze import seal_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze HELIX baseline artifacts as immutable release")
    p.add_argument("--release-id", default="helix_ids_v1.0")
    p.add_argument("--model-checkpoint", default="models/helix_full/helix_full_nsl_kdd_best.pt")
    p.add_argument("--artifact-dir", default="data/processed/multi_dataset_v1")
    p.add_argument("--training-report", default="results/helix_full/training_results_seed42.json")
    p.add_argument("--eval-report", default="results/helix_full/eval_results_seed42.json")
    p.add_argument("--output-root", default="artifacts/releases")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = seal_baseline(
        release_id=args.release_id,
        model_checkpoint=Path(args.model_checkpoint),
        artifact_dir=Path(args.artifact_dir),
        training_report=Path(args.training_report),
        eval_report=Path(args.eval_report),
        output_root=Path(args.output_root),
    )
    print(f"Baseline frozen: {out}")


if __name__ == "__main__":
    main()
