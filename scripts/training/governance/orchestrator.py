"""Governance orchestration: seed-run artifact loading, calibration normalization.

Extracted from train_helix_ids_full.py functions:
  - _load_seed_run_artifacts
  - _load_json_dict
  - _coerce_finite_float
  - _normalize_metrics_payload
  - _materialize_phase8_artifacts
  - _normalize_calibration_block
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from helix_ids.models.full import (
    HelixFullConfig,
    HelixIDSFull,
    create_helix_full,
)

HELIX_FULL_RESULTS_DIR = Path("results/helix_full")
REQUIRED_GEOMETRY_FEATURE_DIM = 17


class CoerceFloatError(ValueError):
    """Raised when a metric value is non-finite."""


def load_json_dict(path: Path) -> dict[str, Any]:
    """Load JSON object from path and validate dictionary payload."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return cast(dict[str, Any], payload)


def coerce_finite_float(value: Any, *, field: str) -> float:
    """Coerce value to float and validate it is finite."""
    numeric = float(value)
    if not np.isfinite(numeric):
        raise CoerceFloatError(f"{field} must be finite, got {value!r}")
    return float(numeric)


def normalize_metrics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize metric aliases into strict external contract keys.

    Handles family_macro_f1 / macro_f1 / family_f1 aliasing across
    different evaluation backends.
    """
    normalized_metrics = {
        "macro_f1": coerce_finite_float(
            metrics.get("macro_f1", metrics.get("family_macro_f1", metrics.get("family_f1", 0.0))),
            field="macro_f1",
        ),
        "class4_precision": coerce_finite_float(
            metrics.get("class4_precision", metrics.get("family_class4_precision", 0.0)),
            field="class4_precision",
        ),
        "class4_recall": coerce_finite_float(
            metrics.get(
                "class4_recall",
                metrics.get("family_class4_recall", metrics.get("family_minority_recall_min", 0.0)),
            ),
            field="class4_recall",
        ),
        "entropy": coerce_finite_float(
            metrics.get("mean_entropy", metrics.get("family_entropy", 0.0)),
            field="entropy",
        ),
        "zero_prediction_classes": int(
            metrics.get(
                "zero_prediction_classes",
                metrics.get("family_zero_prediction_classes", 0),
            )
        ),
    }
    if normalized_metrics["zero_prediction_classes"] < 0:
        raise ValueError("zero_prediction_classes must be >= 0")
    return normalized_metrics


def materialize_phase8_artifacts(calibration_artifacts: dict[str, str]) -> dict[str, str]:
    """Create canonical artifact filenames required by strict completion contract.

    Copies source artifacts to canonical names so the downstream consumer
    finds them at predictable relative paths.
    """
    required_artifacts = {
        "before_after_csv": "before_after.csv",
        "before_after_json": "before_after.json",
        "pr_curve_csv": "pr_curve.csv",
        "confusion_matrices_json": "confusion_matrices.json",
        "ablation_json": "ablation.json",
    }
    canonical: dict[str, str] = {}
    for source_key, canonical_name in required_artifacts.items():
        source_value = calibration_artifacts.get(source_key)
        if not source_value:
            raise ValueError(f"Missing required calibration artifact key: {source_key}")
        source_path = Path(source_value)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing required calibration artifact: {source_path}")
        canonical_path = source_path.parent / canonical_name
        if source_path.resolve() != canonical_path.resolve():
            shutil.copyfile(source_path, canonical_path)
        canonical[source_key] = str(canonical_path)
    return canonical


def normalize_calibration_block(
    *,
    calibration_payload: dict[str, Any],
    calibration_artifacts: dict[str, str],
) -> dict[str, Any]:
    """Normalize calibration outputs into strict contract schema with required paths."""
    normalized_calibration = {
        "temperature": coerce_finite_float(calibration_payload.get("temperature", 1.0), field="temperature"),
        "tau_4": coerce_finite_float(calibration_payload.get("tau_4", 0.5), field="tau_4"),
        "pr_curve_path": str(calibration_artifacts["pr_curve_csv"]),
        "confusion_matrix_path": str(calibration_artifacts["confusion_matrices_json"]),
        "ablation_path": str(calibration_artifacts["ablation_json"]),
    }
    for key in ("pr_curve_path", "confusion_matrix_path", "ablation_path"):
        path = Path(str(normalized_calibration[key]))
        if not path.exists():
            raise FileNotFoundError(f"Required calibration artifact missing: {path}")
    return normalized_calibration


def load_seed_run_artifacts(
    *,
    seed: int,
    proc: subprocess.CompletedProcess[str],
) -> tuple[str, dict[str, Any], dict[str, Any], HelixIDSFull]:
    """Load evaluation and training artifacts for a completed seed run.

    Returns (dataset_name, train_payload, eval_results, model).

    Raises RuntimeError when expected artifacts are missing or the seed
    run exited with a non-zero code.
    """
    eval_path = HELIX_FULL_RESULTS_DIR / f"eval_results_seed{int(seed)}.json"
    train_path = HELIX_FULL_RESULTS_DIR / f"training_results_seed{int(seed)}.json"
    if not eval_path.exists() or not train_path.exists():
        raise RuntimeError(
            "Multi-seed run did not emit expected artifacts for seed "
            f"{seed}; exit={proc.returncode}"
        )

    train_payload = load_json_dict(train_path)
    train_exit_code = int(train_payload.get("run_exit_code", proc.returncode))
    train_guard_failure = str(train_payload.get("guard_failure", "") or "")
    if train_exit_code != 0:
        raise RuntimeError(
            "Seed run failed before calibration artifacts were materialized: "
            f"seed={seed} exit={train_exit_code} guard_failure={train_guard_failure}"
        )

    eval_payload = load_json_dict(eval_path)
    eval_results = cast(dict[str, Any], eval_payload.get("results", {}))
    if not eval_results:
        raise RuntimeError(f"Missing eval results for seed {seed}")

    dataset_name = min(eval_results.keys())
    model_path = Path("models/helix_full") / f"helix_full_{dataset_name}_best.pt"
    if not model_path.exists():
        raise RuntimeError(f"Missing best checkpoint for seed {seed}: {model_path}")

    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    model_state = cast(dict[str, Any], artifact.get("model_state_dict", artifact.get("model", {})))
    model: HelixIDSFull = create_helix_full(
        HelixFullConfig(
            input_dim=REQUIRED_GEOMETRY_FEATURE_DIM,
            hidden_dims=(512, 384, 256, 256),
            dropout_rates=(0.3, 0.3, 0.25, 0.2),
        )
    )
    model.load_state_dict(model_state)

    return dataset_name, train_payload, eval_results, model
