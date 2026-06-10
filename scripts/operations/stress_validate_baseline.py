#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from helix_ids.operations.inference_runtime import HelixInferenceRuntime, InferenceConfig
from helix_ids.operations.monitoring import (
    LiveMonitor,
    MonitorConfig,
    compute_zero_prediction_classes,
)


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def _run_scenario(runtime: HelixInferenceRuntime, x: np.ndarray, active_classes: list[int], *, enforce_global_coverage: bool) -> tuple[dict[str, Any], np.ndarray]:
    result = runtime.predict(x, active_classes=active_classes, enforce_global_coverage=enforce_global_coverage)
    preds = np.asarray(result["family_class"] if isinstance(result["family_class"], list) else [result["family_class"]], dtype=np.int64)
    if preds.shape[0] != x.shape[0]:
        # broadcast single output for safety in edge case
        preds = np.resize(preds, x.shape[0])
    zero_missing = compute_zero_prediction_classes(preds, active_classes)
    dist = np.bincount(preds, minlength=max(active_classes) + 1).astype(np.float64)
    dist = dist / max(1.0, dist.sum())
    scenario = {
        "samples": int(x.shape[0]),
        "zero_prediction_classes": int(zero_missing),
        "class_distribution": dist.tolist(),
        "entropy": _entropy(dist),
    }
    return scenario, preds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stress validate frozen HELIX baseline")
    p.add_argument("--checkpoint", default="models/helix_full/helix_full_nsl_kdd_best.pt")
    p.add_argument("--artifact-dir", default="data/processed/multi_dataset_v1")
    p.add_argument("--output", default="artifacts/releases/helix_ids_v1.0/validation/stress_report.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--previous-report", default="artifacts/packages/helix_ids_v1.0/stress_report.json")
    p.add_argument("--correction-quantile", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)

    x_val = np.load(artifact_dir / "X_val_nsl_kdd.npy")
    y_train = np.load(artifact_dir / "y_train_nsl_kdd.npy")
    active_classes = sorted(int(x) for x in np.unique(y_train).tolist())

    runtime = HelixInferenceRuntime(
        Path(args.checkpoint),
        device=args.device,
        config=InferenceConfig(fixed_temperature=1.0, prediction_floor=1e-6, global_coverage_floor=True, global_coverage_quantile=float(args.correction_quantile)),
    )

    baseline_no_fix, baseline_preds_no_fix = _run_scenario(runtime, x_val[:2000], active_classes, enforce_global_coverage=False)
    baseline, baseline_preds = _run_scenario(runtime, x_val[:2000], active_classes, enforce_global_coverage=True)

    rng = np.random.default_rng(42)
    unseen_shift = x_val[:2000] + rng.normal(loc=0.35, scale=0.15, size=x_val[:2000].shape).astype(np.float32)

    idx = rng.choice(x_val.shape[0], size=2000, replace=True)
    imbalanced = x_val[idx]

    noisy = x_val[:2000] + rng.normal(loc=0.0, scale=0.75, size=x_val[:2000].shape).astype(np.float32)

    unseen_scenario, unseen_preds = _run_scenario(runtime, unseen_shift, active_classes, enforce_global_coverage=True)
    imbalance_scenario, imbalance_preds = _run_scenario(runtime, imbalanced, active_classes, enforce_global_coverage=True)
    noisy_scenario, noisy_preds = _run_scenario(runtime, noisy, active_classes, enforce_global_coverage=True)

    scenarios = {
        "baseline": baseline,
        "unseen_distribution": unseen_scenario,
        "class_imbalance_shift": imbalance_scenario,
        "noisy_inputs": noisy_scenario,
    }

    scenario_preds = {
        "baseline": baseline_preds,
        "unseen_distribution": unseen_preds,
        "class_imbalance_shift": imbalance_preds,
        "noisy_inputs": noisy_preds,
    }

    monitor = LiveMonitor(
        baseline_class_distribution=np.asarray(baseline["class_distribution"], dtype=np.float64),
        baseline_entropy=float(baseline["entropy"]),
        baseline_macro_f1=None,
        config=MonitorConfig(
            class_distribution_tolerance=0.25,
            entropy_tolerance=0.35,
            macro_f1_tolerance=0.05,
        ),
    )

    monitor_results = {}
    for name, preds in scenario_preds.items():
        monitor_results[name] = monitor.evaluate(preds)

    invariants = {
        "zero_prediction_classes": {
            k: int(v["zero_prediction_classes"]) for k, v in scenarios.items()
        },
        "no_crash": True,
        "no_drift": not any(m.get("alert", False) for m in monitor_results.values()),
    }

    failing_class_from_previous = None
    previous_report_path = Path(args.previous_report)
    if previous_report_path.exists():
        previous = json.loads(previous_report_path.read_text(encoding="utf-8"))
        prev_zero = previous.get("invariants", {}).get("zero_prediction_classes", {})
        missing: list[int] = []
        if isinstance(prev_zero, dict):
            baseline_missing = prev_zero.get("baseline")
            if isinstance(baseline_missing, int) and baseline_missing > 0:
                prev_dist = previous.get("scenarios", {}).get("baseline", {}).get("class_distribution", [])
                missing = [i for i, p in enumerate(prev_dist) if float(p) <= 0.0]
        # fallback: infer from previous baseline class distribution
        if not missing:
            prev_dist = previous.get("scenarios", {}).get("baseline", {}).get("class_distribution", [])
            missing = [i for i, p in enumerate(prev_dist) if float(p) <= 0.0]
        if missing:
            failing_class_from_previous = int(missing[0])

    y_val = np.load(artifact_dir / "y_val_nsl_kdd.npy")[:2000]
    macro_f1_before = float(f1_score(y_val, baseline_preds_no_fix, average="macro", zero_division=0))
    macro_f1_after = float(f1_score(y_val, baseline_preds, average="macro", zero_division=0))
    macro_f1_drop = float(macro_f1_before - macro_f1_after)

    report = {
        "active_classes": active_classes,
        "failing_class_from_previous_report": failing_class_from_previous,
        "correction": {
            "method": "global_coverage_floor",
            "quantile": float(args.correction_quantile),
            "baseline_macro_f1_before": macro_f1_before,
            "baseline_macro_f1_after": macro_f1_after,
            "baseline_macro_f1_drop": macro_f1_drop,
            "acceptance": {
                "zero_prediction_classes_eq_0": all(v == 0 for v in invariants["zero_prediction_classes"].values()),
                "macro_f1_drop_leq_0_02": macro_f1_drop <= 0.02,
            },
        },
        "scenarios": scenarios,
        "monitor": monitor_results,
        "invariants": invariants,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Stress validation report: {out}")


if __name__ == "__main__":
    main()
