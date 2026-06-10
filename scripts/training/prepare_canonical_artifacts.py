#!/usr/bin/env python3
"""Prepare canonical processed artifacts for governed HELIX training.

This script performs four steps:
1) Verifies required raw datasets are present.
2) Builds canonical split artifacts via MultiDatasetLoader.
3) Runs UNSW learnability validation.
4) Verifies required output artifacts and meta validation state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helix_ids.contracts import (  # noqa: E402
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_INPUT_DIM,
    SCHEMA_HASH,
    SCHEMA_VERSION,
)
from helix_ids.data.feature_harmonization import FEATURE_ORDER  # noqa: E402
from helix_ids.data.multi_dataset_loader import MultiDatasetLoader  # noqa: E402

REQUIRED_ARTIFACTS = [
    "X_train.npy",
    "y_train.npy",
    "X_val.npy",
    "y_val.npy",
    "X_test_nsl_kdd.npy",
    "y_test_nsl_kdd.npy",
    "X_test_unsw_nb15.npy",
    "y_test_unsw_nb15.npy",
    "feature_columns.npy",
    "canonical_contract.json",
    "meta.json",
]

REQUIRED_CICIDS_ARTIFACTS = [
    "X_test_cicids.npy",
    "y_test_cicids.npy",
]


def _check_raw_inputs(project_root: Path, *, require_cicids: bool) -> None:
    errors: list[str] = []

    nsl_path = project_root / "data" / "nsl_kdd" / "raw" / "KDDTrain+.txt"
    unsw_path = project_root / "data" / "unsw_nb15" / "raw" / "UNSW_NB15_training-set.csv"
    cicids_dir = project_root / "data" / "cicids2018" / "raw"

    if not nsl_path.exists():
        errors.append(f"missing file: {nsl_path}")
    if not unsw_path.exists():
        errors.append(f"missing file: {unsw_path}")

    if require_cicids:
        if not cicids_dir.exists():
            errors.append(f"missing directory: {cicids_dir}")
        else:
            daywise_files = list(cicids_dir.glob("*.csv"))
            if not daywise_files:
                errors.append(f"no day-wise CSV files found in: {cicids_dir}")

    if errors:
        bullet_list = "\n".join(f"- {item}" for item in errors)
        raise FileNotFoundError(f"Raw dataset prerequisite check failed:\n{bullet_list}")


def _build_processed_artifacts(project_root: Path, artifact_dir: Path) -> None:
    loader = MultiDatasetLoader(project_root=project_root)
    loader.save_processed_data(output_dir=artifact_dir)


def _run_unsw_validation(
    project_root: Path,
    artifact_dir: Path,
    *,
    ci_output: bool,
    threshold_profile: str,
) -> None:
    validation_script = project_root / "scripts" / "validation" / "validate_unsw_learnability.py"
    cmd = [
        sys.executable,
        str(validation_script),
        "--artifact-dir",
        str(artifact_dir),
        "--threshold-profile",
        threshold_profile,
    ]
    if ci_output:
        cmd.append("--ci-output")
    subprocess.run(cmd, check=True)


def _verify_artifacts(artifact_dir: Path, *, require_cicids: bool) -> None:
    required = list(REQUIRED_ARTIFACTS)
    if require_cicids:
        required.extend(REQUIRED_CICIDS_ARTIFACTS)

    missing = [name for name in required if not (artifact_dir / name).exists()]
    if missing:
        missing_text = "\n".join(f"- {artifact_dir / name}" for name in sorted(missing))
        raise FileNotFoundError(
            "Processed artifact check failed; missing required files:\n"
            f"{missing_text}"
        )

    meta_path = artifact_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not bool(meta.get("validated", False)):
        raise RuntimeError(
            "Processed artifact check failed; learnability contract is not validated. "
            f"meta={meta_path}"
        )

    contract_path = artifact_dir / "canonical_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_order = [str(col) for col in FEATURE_ORDER]
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Processed artifact check failed; schema version mismatch")
    if contract.get("schema_hash") != SCHEMA_HASH:
        raise RuntimeError("Processed artifact check failed; schema hash mismatch")
    if contract.get("input_dim") != CANONICAL_INPUT_DIM:
        raise RuntimeError("Processed artifact check failed; input dim mismatch")
    if contract.get("binary_output_dim") != CANONICAL_BINARY_CLASSES:
        raise RuntimeError("Processed artifact check failed; binary class mismatch")
    if contract.get("family_output_dim") != CANONICAL_FAMILY_CLASSES:
        raise RuntimeError("Processed artifact check failed; family class mismatch")
    if contract.get("feature_order") != expected_order:
        raise RuntimeError("Processed artifact check failed; canonical feature order mismatch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare canonical processed artifacts for governed HELIX training"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root path",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "multi_dataset_v1",
        help="Output artifact directory",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip split materialization and only run validation/checks",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip UNSW validation script invocation",
    )
    parser.add_argument(
        "--require-cicids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require CICIDS raw inputs and CICIDS split artifacts",
    )
    parser.add_argument(
        "--ci-output",
        action="store_true",
        help="Use compact output mode when running validation",
    )
    parser.add_argument(
        "--threshold-profile",
        choices=["default", "preprocess"],
        default="preprocess",
        help="Threshold profile for UNSW learnability validation",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = args.project_root.resolve()
    artifact_dir = args.artifact_dir.resolve()

    print("[1/4] Checking raw dataset prerequisites...")
    _check_raw_inputs(project_root, require_cicids=bool(args.require_cicids))

    if not args.skip_build:
        print("[2/4] Building canonical processed artifacts...")
        _build_processed_artifacts(project_root, artifact_dir)
    else:
        print("[2/4] Skipped artifact build (--skip-build).")

    if not args.skip_validation:
        print("[3/4] Running UNSW learnability validation...")
        _run_unsw_validation(
            project_root,
            artifact_dir,
            ci_output=bool(args.ci_output),
            threshold_profile=str(args.threshold_profile),
        )
    else:
        print("[3/4] Skipped validation (--skip-validation).")

    print("[4/4] Verifying required artifacts and contract state...")
    _verify_artifacts(artifact_dir, require_cicids=bool(args.require_cicids))

    print("✅ Canonical artifact preparation complete.")
    print(f"Artifacts: {artifact_dir}")
    print(
        "Next: python scripts/training/train_helix_ids_full.py "
        "--output models/helix_full --device mps"
    )


if __name__ == "__main__":
    main()
