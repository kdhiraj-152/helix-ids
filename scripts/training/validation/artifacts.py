"""Calibration artifact persistence for HELIX-IDS.

Includes JSON atomic writes and paper-oriented calibration artifact generation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd

from scripts.training.governance import (  # noqa: F401
    coerce_finite_float as _coerce_finite_float,
)
from scripts.training.governance import (
    materialize_phase8_artifacts as _materialize_phase8_artifacts,
)
from scripts.training.governance import (
    normalize_calibration_block as _normalize_calibration_block,
)

__all__ = [
    "_atomic_write_json",
    "_emit_calibration_artifacts",
    "_materialize_phase8_artifacts",
    "_normalize_calibration_block",
]


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via temporary file and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp_path, path)


def _emit_calibration_artifacts(
    *,
    results_dir: Path,
    dataset_name: str,
    seed: int,
    calibration_payload: dict[str, Any],
    artifact_tag: str | None = None,
) -> dict[str, str]:
    """Persist paper-oriented calibration artifacts for one dataset/seed."""
    calibration_dir = results_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_seed{int(seed)}"
    if artifact_tag:
        suffix = f"_{str(artifact_tag)}{suffix}"

    calibration_json_path = calibration_dir / f"{dataset_name}_calibration{suffix}.json"
    _atomic_write_json(calibration_json_path, calibration_payload)

    uncal_test = cast(dict[str, Any], calibration_payload.get("uncalibrated", {}).get("test_argmax", {}))
    threshold_only = cast(
        dict[str, Any],
        calibration_payload.get("ablation", {}).get("without_temperature_scaling", {}),
    )
    calibrated = cast(dict[str, Any], calibration_payload.get("test", {}))

    before_after_rows = [
        {
            "phase": "baseline_collapse",
            "macro_f1": float(uncal_test.get("macro_f1", 0.0)),
            "class4_precision": float(uncal_test.get("class4_precision", 0.0)),
            "class4_recall": float(uncal_test.get("class4_recall", 0.0)),
            "zero_prediction_classes": int(uncal_test.get("zero_prediction_classes", 0)),
            "mean_entropy": float(uncal_test.get("mean_entropy", 0.0)),
        },
        {
            "phase": "enforcement_high_recall_low_precision",
            "macro_f1": float(threshold_only.get("macro_f1", 0.0)),
            "class4_precision": float(threshold_only.get("class4_precision", 0.0)),
            "class4_recall": float(threshold_only.get("class4_recall", 0.0)),
            "zero_prediction_classes": int(uncal_test.get("zero_prediction_classes", 0)),
            "mean_entropy": float(threshold_only.get("mean_entropy", 0.0)),
        },
        {
            "phase": "calibrated_balanced",
            "macro_f1": float(calibrated.get("macro_f1", 0.0)),
            "class4_precision": float(calibrated.get("class4_precision", 0.0)),
            "class4_recall": float(calibrated.get("class4_recall", 0.0)),
            "zero_prediction_classes": int(calibrated.get("zero_prediction_classes", 0)),
            "mean_entropy": float(calibrated.get("mean_entropy", 0.0)),
        },
    ]

    before_after_json_path = calibration_dir / f"{dataset_name}_before_after{suffix}.json"
    _atomic_write_json(
        before_after_json_path,
        {
            "dataset": dataset_name,
            "seed": int(seed),
            "temperature": float(calibration_payload.get("temperature", 1.0)),
            "tau_4": float(calibration_payload.get("tau_4", 0.5)),
            "rows": before_after_rows,
        },
    )
    before_after_csv_path = calibration_dir / f"{dataset_name}_before_after{suffix}.csv"
    pd.DataFrame(before_after_rows).to_csv(before_after_csv_path, index=False)

    pr_payload = cast(dict[str, Any], calibration_payload.get("pr_curve_class4", {}))
    pr_precision = [float(v) for v in cast(list[Any], pr_payload.get("precision", []))]
    pr_recall = [float(v) for v in cast(list[Any], pr_payload.get("recall", []))]
    pr_thresholds = [float(v) for v in cast(list[Any], pr_payload.get("thresholds", []))]
    max_rows = max(len(pr_precision), len(pr_recall), len(pr_thresholds))
    pr_rows: list[dict[str, Any]] = []
    for idx in range(max_rows):
        pr_rows.append(
            {
                "point_index": int(idx),
                "precision": pr_precision[idx] if idx < len(pr_precision) else None,
                "recall": pr_recall[idx] if idx < len(pr_recall) else None,
                "threshold": pr_thresholds[idx] if idx < len(pr_thresholds) else None,
            }
        )
    pr_csv_path = calibration_dir / f"{dataset_name}_pr_curve_class4{suffix}.csv"
    pd.DataFrame(pr_rows).to_csv(pr_csv_path, index=False)

    confusion_payload = {
        "dataset": dataset_name,
        "seed": int(seed),
        "uncalibrated_test_argmax": cast(dict[str, Any], calibration_payload.get("uncalibrated", {}).get("test_argmax", {})).get(
            "confusion_matrix", []
        ),
        "ablation_without_thresholding": cast(dict[str, Any], calibration_payload.get("ablation", {}).get("without_thresholding", {})).get(
            "confusion_matrix", []
        ),
        "ablation_without_temperature_scaling": cast(
            dict[str, Any],
            calibration_payload.get("ablation", {}).get("without_temperature_scaling", {}),
        ).get("confusion_matrix", []),
        "calibrated": cast(dict[str, Any], calibration_payload.get("test", {})).get("confusion_matrix", []),
    }
    confusion_json_path = calibration_dir / f"{dataset_name}_confusion_matrices{suffix}.json"
    _atomic_write_json(confusion_json_path, confusion_payload)

    ablation_json_path = calibration_dir / f"{dataset_name}_ablation{suffix}.json"
    _atomic_write_json(
        ablation_json_path,
        {
            "dataset": dataset_name,
            "seed": int(seed),
            "ablation": cast(dict[str, Any], calibration_payload.get("ablation", {})),
        },
    )

    return {
        "calibration_json": str(calibration_json_path),
        "before_after_json": str(before_after_json_path),
        "before_after_csv": str(before_after_csv_path),
        "pr_curve_csv": str(pr_csv_path),
        "confusion_matrices_json": str(confusion_json_path),
        "ablation_json": str(ablation_json_path),
    }
