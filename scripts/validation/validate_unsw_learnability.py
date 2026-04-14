#!/usr/bin/env python3
"""Validate UNSW processed split learnability and write dataset contract metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helix_ids.data.learnability_contract import (  # noqa: E402
    build_meta,
    compute_contract_metrics,
    compute_schema_hash,
    format_failure_message,
    freeze_snapshot_if_valid,
    load_reference_profile_bundle,
    write_meta,
    write_reference_profile,
)
from helix_ids.data.feature_harmonization import create_unsw_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate UNSW learnability contract")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "multi_dataset_v1",
        help="Directory containing X_/y_ split artifacts",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="unsw_nb15",
        choices=["unsw_nb15"],
        help="Dataset contract to validate",
    )
    parser.add_argument(
        "--transformations",
        type=str,
        default="split_then_nan_to_num",
        help="Comma-separated transformation identifiers used by preprocessing",
    )
    parser.add_argument(
        "--ci-output",
        action="store_true",
        help="Print only summary for CI; omit full diagnostics JSON",
    )
    return parser.parse_args()


def print_ci_summary(meta: dict, verbose: bool = False) -> None:
    """Print CI-optimized summary (no full JSON)."""
    summary = meta.get("summary", {})
    diagnosis = meta.get("diagnosis", {})
    
    status = summary.get("status", "UNKNOWN")
    primary = summary.get("primary_issue", "unknown")
    stage = summary.get("stage", "unknown")
    action = summary.get("action", "INVESTIGATE")
    confidence = summary.get("confidence", 0.0)
    
    print(f"LEARNABILITY: {status}")
    print(f"  Primary Issue: {primary}")
    print(f"  Stage: {stage}")
    print(f"  Action: {action}")
    print(f"  Confidence: {confidence:.2f}")
    
    if verbose:
        # Print kill list if present
        kill_list = summary.get("kill_list", [])
        if kill_list:
            print(f"  Target Features: {', '.join(kill_list)}")
        
        # Print secondary causes
        secondary = diagnosis.get("secondary", [])
        if secondary:
            print(f"  Secondary Issues: {', '.join(secondary)}")


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir

    x_path = artifact_dir / f"X_train_{args.dataset}.npy"
    y_path = artifact_dir / f"y_train_{args.dataset}.npy"
    feature_columns_path = artifact_dir / "feature_columns.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing required split files: {x_path} / {y_path}")
    x_train = np.load(x_path)
    y_train = np.load(y_path).astype(np.int64)

    if not feature_columns_path.exists():
        raise FileNotFoundError(
            f"Missing feature schema file: {feature_columns_path}. Regenerate processed artifacts."
        )
    feature_columns = np.load(feature_columns_path, allow_pickle=True).astype(str).tolist()

    transformations = [item.strip() for item in args.transformations.split(",") if item.strip()]

    schema_hash = compute_schema_hash(
        feature_columns=feature_columns,
        transformations=transformations,
    )

    unsw_mapping = create_unsw_mapping().feature_mapping
    feature_lineage = {
        f"f_{idx}": ",".join(unsw_mapping.get(feature_name, [feature_name]))
        for idx, feature_name in enumerate(feature_columns)
    }

    metrics = compute_contract_metrics(
        x_train=x_train,
        y_train=y_train,
        dataset=args.dataset,
        schema_hash=schema_hash,
        feature_names=feature_columns,
        feature_lineage=feature_lineage,
        stage_snapshots={"split_then_nan_to_num": x_train},
    )
    profile_bundle = load_reference_profile_bundle(
        artifact_dir=artifact_dir,
        dataset_signature="unsw",
    )
    metrics["reference_profile"] = profile_bundle["profile"]
    metrics["expected_reference_profile_version"] = profile_bundle["profile"].get("version")
    meta = build_meta(metrics)
    write_meta(meta, artifact_dir=artifact_dir)
    profile_bundle["payload"]["reference_profiles"][profile_bundle["profile_key"]] = meta["reference_profile"]
    write_reference_profile(profile_bundle["payload"], artifact_dir=artifact_dir)
    meta = freeze_snapshot_if_valid(artifact_dir=artifact_dir)
    out_path = artifact_dir / "meta.json"

    # Print output based on --ci-output flag
    if args.ci_output:
        print_ci_summary(meta, verbose=True)
    else:
        print(json.dumps(meta, indent=2, sort_keys=True))
    
    if not meta["validated"]:
        # Use deterministic failure message
        summary = meta.get("summary", {})
        if summary:
            failure_msg = format_failure_message(summary)
        else:
            failure_msg = f"UNSW learnability validation failed. meta={out_path}"
        raise RuntimeError(failure_msg)


if __name__ == "__main__":
    main()
