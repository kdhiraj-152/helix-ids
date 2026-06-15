"""Temperature scaling and class-4 calibration for HELIX-IDS family predictions."""
from __future__ import annotations

from typing import Any, cast

import numpy as np
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

from helix_ids.models.full import HelixIDSFull
from helix_ids.utils.metrics import compute_macro_f1
from scripts.training.data import (
    _compute_class4_metrics,
    _compute_multiclass_confusion,
    _summarize_prediction_coverage,
)

from .evaluator import (
    _collect_eval_family_outputs,
    _normalized_entropy_from_probs,
)


def _softmax_with_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    """Apply softmax with temperature scaling."""
    scaled = np.asarray(logits, dtype=np.float64) / max(1e-6, float(t))
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return cast(np.ndarray, exp_vals / np.clip(np.sum(exp_vals, axis=1, keepdims=True), 1e-12, None))


def _fit_temperature_nll(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_temperature: float,
) -> tuple[float, float]:
    """Fit global temperature by minimizing NLL over a deterministic grid."""
    if logits.ndim != 2 or logits.shape[0] == 0:
        return 1.0, 0.0

    y = np.asarray(labels, dtype=np.int64)
    y = np.clip(y, 0, int(logits.shape[1]) - 1)

    min_t = 1.0
    max_t = max(min_t, float(max_temperature))
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(min_t, min(2.0, max_t), 151),
                np.linspace(min_t, max_t, 301),
            ]
        )
    )

    best_t = 1.0
    best_nll = float("inf")
    for t in grid:
        scaled = np.asarray(logits, dtype=np.float64) / max(1e-6, float(t))
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.clip(np.sum(np.exp(shifted), axis=1, keepdims=True), 1e-12, None))
        nll = float(-np.mean(log_probs[np.arange(log_probs.shape[0]), y]))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return float(best_t), float(best_nll)


def _apply_class4_logit_shift(
    logits: np.ndarray,
    *,
    class4_id: int,
    delta: float,
) -> np.ndarray:
    """Subtract a fixed delta from class-4 logits (inference/eval only)."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return cast(np.ndarray, arr.copy())
    class_idx = int(class4_id)
    if class_idx < 0 or class_idx >= int(arr.shape[1]):
        return cast(np.ndarray, arr.copy())
    shift = float(delta)
    if abs(shift) <= 0.0:
        return cast(np.ndarray, arr.copy())
    out = arr.copy()
    out[:, class_idx] = out[:, class_idx] - shift
    return cast(np.ndarray, out)


def _predict_with_class4_threshold(
    probs: np.ndarray,
    *,
    class4_id: int,
    threshold: float,
) -> np.ndarray:
    """Apply class-4 gating: predict 4 when P4>=tau else argmax over other classes."""
    if probs.ndim != 2 or probs.shape[0] == 0:
        return np.array([], dtype=np.int64)

    pred = np.argmax(probs, axis=1).astype(np.int64, copy=False)
    if class4_id < 0 or class4_id >= int(probs.shape[1]):
        return cast(np.ndarray, pred)

    p4 = probs[:, class4_id]
    choose4 = p4 >= float(threshold)

    others = np.asarray(probs, dtype=np.float64).copy()
    others[:, class4_id] = -np.inf
    fallback = np.argmax(others, axis=1).astype(np.int64, copy=False)

    out = fallback.copy()
    out[choose4] = int(class4_id)
    return cast(np.ndarray, out)


def _calibrate_family_predictions(
    *,
    model: HelixIDSFull,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    class4_id: int = 4,
    max_temperature: float = 5.0,
    threshold_grid: np.ndarray | None = None,
    min_class4_recall: float = 0.80,
    class4_logit_shift: float = 0.0,
) -> dict[str, Any]:
    """Temperature-scale logits and tune class-4 threshold on validation set."""
    if threshold_grid is None:
        threshold_grid = np.linspace(0.3, 0.95, 66)

    y_val, val_logits, val_probs_uncal = _collect_eval_family_outputs(
        model=model,
        loader=val_loader,
        device=device,
    )
    y_test, test_logits, test_probs_uncal = _collect_eval_family_outputs(
        model=model,
        loader=test_loader,
        device=device,
    )

    if y_val.size == 0 or y_test.size == 0:
        return {
            "class4_logit_shift": float(class4_logit_shift),
            "temperature": 1.0,
            "tau_4": 0.5,
            "uncalibrated": {
                "val_argmax": {
                    "class4_precision": 0.0,
                    "class4_recall": 0.0,
                    "macro_f1": 0.0,
                    "zero_prediction_classes": 0,
                    "mean_entropy": 0.0,
                    "confusion_matrix": [],
                },
                "test_argmax": {
                    "class4_precision": 0.0,
                    "class4_recall": 0.0,
                    "macro_f1": 0.0,
                    "zero_prediction_classes": 0,
                    "mean_entropy": 0.0,
                    "confusion_matrix": [],
                },
            },
            "val": {
                "class4_precision": 0.0,
                "class4_recall": 0.0,
                "macro_f1": 0.0,
                "zero_prediction_classes": 0,
                "mean_entropy": 0.0,
            },
            "test": {
                "class4_precision": 0.0,
                "class4_recall": 0.0,
                "macro_f1": 0.0,
                "zero_prediction_classes": 0,
                "mean_entropy": 0.0,
                "confusion_matrix": [],
            },
            "ablation": {
                "without_thresholding": {},
                "without_temperature_scaling": {},
            },
            "pr_curve_class4": {
                "precision": [],
                "recall": [],
                "thresholds": [],
            },
            "threshold_sweep": {
                "tau_min": float(np.min(threshold_grid)),
                "tau_max": float(np.max(threshold_grid)),
                "num_points": int(np.asarray(threshold_grid).size),
                "points": [],
            },
        }

    val_logits = _apply_class4_logit_shift(
        val_logits,
        class4_id=int(class4_id),
        delta=float(class4_logit_shift),
    )
    test_logits = _apply_class4_logit_shift(
        test_logits,
        class4_id=int(class4_id),
        delta=float(class4_logit_shift),
    )

    class_count = int(test_logits.shape[1])

    best_t, _best_nll = _fit_temperature_nll(
        logits=val_logits,
        labels=y_val,
        max_temperature=float(max_temperature),
    )

    val_probs_uncal = _softmax_with_temperature(val_logits, 1.0)
    test_probs_uncal = _softmax_with_temperature(test_logits, 1.0)
    val_probs_cal = _softmax_with_temperature(val_logits, best_t)
    test_probs_cal = _softmax_with_temperature(test_logits, best_t)

    val_pred_uncal = np.argmax(val_probs_uncal, axis=1).astype(np.int64, copy=False)
    test_pred_uncal = np.argmax(test_probs_uncal, axis=1).astype(np.int64, copy=False)
    val_uncal_c4 = _compute_class4_metrics(y_val, val_pred_uncal, class4_id=int(class4_id))
    test_uncal_c4 = _compute_class4_metrics(y_test, test_pred_uncal, class4_id=int(class4_id))
    val_uncal_macro = float(compute_macro_f1(y_val, val_pred_uncal))
    test_uncal_macro = float(compute_macro_f1(y_test, test_pred_uncal))
    val_uncal_zero = _summarize_prediction_coverage(y_val, val_pred_uncal)
    test_uncal_zero = _summarize_prediction_coverage(y_test, test_pred_uncal)
    val_uncal_entropy = _normalized_entropy_from_probs(val_probs_uncal)
    test_uncal_entropy = _normalized_entropy_from_probs(test_probs_uncal)
    val_uncal_conf = _compute_multiclass_confusion(y_val, val_pred_uncal, class_count=class_count)
    test_uncal_conf = _compute_multiclass_confusion(y_test, test_pred_uncal, class_count=class_count)

    # Sweep tau_4 on validation: maximize class4 precision while preserving recall.
    best_tau = 0.5
    best_precision = -1.0
    best_macro = -1.0
    best_recall = -1.0
    feasible_tau_found = False
    sweep_points: list[dict[str, float]] = []
    for tau in np.asarray(threshold_grid, dtype=np.float64):
        val_pred_tau = _predict_with_class4_threshold(
            val_probs_cal,
            class4_id=int(class4_id),
            threshold=float(tau),
        )
        test_pred_tau = _predict_with_class4_threshold(
            test_probs_cal,
            class4_id=int(class4_id),
            threshold=float(tau),
        )
        val_c4 = _compute_class4_metrics(y_val, val_pred_tau, class4_id=int(class4_id))
        test_c4_tau = _compute_class4_metrics(y_test, test_pred_tau, class4_id=int(class4_id))
        val_macro = float(compute_macro_f1(y_val, val_pred_tau))
        test_macro_tau = float(compute_macro_f1(y_test, test_pred_tau))
        precision = float(val_c4["class4_precision"])
        recall = float(val_c4["class4_recall"])
        test_precision_tau = float(test_c4_tau["class4_precision"])
        test_recall_tau = float(test_c4_tau["class4_recall"])

        sweep_points.append(
            {
                "tau_4": float(tau),
                "val_class4_precision": float(precision),
                "val_class4_recall": float(recall),
                "val_macro_f1": float(val_macro),
                "test_class4_precision": float(test_precision_tau),
                "test_class4_recall": float(test_recall_tau),
                "test_macro_f1": float(test_macro_tau),
            }
        )

        recall_ok = recall >= float(min_class4_recall)
        if not recall_ok:
            continue
        feasible_tau_found = True
        candidate = (
            (precision, val_macro, recall, -float(tau)),
            float(tau),
        )
        incumbent = (
            (best_precision, best_macro, best_recall, -best_tau),
            best_tau,
        )
        if candidate[0] > incumbent[0]:
            best_tau = float(tau)
            best_precision = precision
            best_macro = val_macro
            best_recall = recall

    # Evaluate selected setting on val + test.
    if not feasible_tau_found:
        best_tau = float(np.min(np.asarray(threshold_grid, dtype=np.float64)))
    val_pred = _predict_with_class4_threshold(val_probs_cal, class4_id=int(class4_id), threshold=best_tau)
    test_pred = _predict_with_class4_threshold(test_probs_cal, class4_id=int(class4_id), threshold=best_tau)

    val_c4 = _compute_class4_metrics(y_val, val_pred, class4_id=int(class4_id))
    test_c4 = _compute_class4_metrics(y_test, test_pred, class4_id=int(class4_id))

    val_macro = float(compute_macro_f1(y_val, val_pred))
    test_macro = float(compute_macro_f1(y_test, test_pred))
    val_zero = _summarize_prediction_coverage(y_val, val_pred)
    test_zero = _summarize_prediction_coverage(y_test, test_pred)
    val_entropy = _normalized_entropy_from_probs(val_probs_cal)
    test_entropy = _normalized_entropy_from_probs(test_probs_cal)

    conf_test = _compute_multiclass_confusion(y_test, test_pred, class_count=class_count)

    # Ablations.
    test_pred_temp_only = np.argmax(test_probs_cal, axis=1).astype(np.int64, copy=False)
    test_pred_thresh_only = _predict_with_class4_threshold(
        test_probs_uncal,
        class4_id=int(class4_id),
        threshold=best_tau,
    )

    temp_only_c4 = _compute_class4_metrics(y_test, test_pred_temp_only, class4_id=int(class4_id))
    thresh_only_c4 = _compute_class4_metrics(y_test, test_pred_thresh_only, class4_id=int(class4_id))
    conf_temp_only = _compute_multiclass_confusion(y_test, test_pred_temp_only, class_count=class_count)
    conf_thresh_only = _compute_multiclass_confusion(y_test, test_pred_thresh_only, class_count=class_count)

    # PR curve on class-4 from calibrated test probs.
    y_true_bin = (y_test == int(class4_id)).astype(np.int64)
    if int(np.unique(y_true_bin).size) > 1:
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true_bin, test_probs_cal[:, int(class4_id)])
        pr_precision_list = [float(v) for v in pr_precision.tolist()]
        pr_recall_list = [float(v) for v in pr_recall.tolist()]
        pr_threshold_list = [float(v) for v in pr_thresholds.tolist()]
    else:
        pr_precision_list = []
        pr_recall_list = []
        pr_threshold_list = []

    return {
        "class4_logit_shift": float(class4_logit_shift),
        "temperature": float(best_t),
        "tau_4": float(best_tau),
        "uncalibrated": {
            "val_argmax": {
                "class4_precision": float(val_uncal_c4["class4_precision"]),
                "class4_recall": float(val_uncal_c4["class4_recall"]),
                "macro_f1": float(val_uncal_macro),
                "zero_prediction_classes": int(val_uncal_zero),
                "mean_entropy": float(val_uncal_entropy),
                "confusion_matrix": val_uncal_conf.tolist(),
            },
            "test_argmax": {
                "class4_precision": float(test_uncal_c4["class4_precision"]),
                "class4_recall": float(test_uncal_c4["class4_recall"]),
                "macro_f1": float(test_uncal_macro),
                "zero_prediction_classes": int(test_uncal_zero),
                "mean_entropy": float(test_uncal_entropy),
                "confusion_matrix": test_uncal_conf.tolist(),
            },
        },
        "val": {
            "class4_precision": float(val_c4["class4_precision"]),
            "class4_recall": float(val_c4["class4_recall"]),
            "macro_f1": float(val_macro),
            "zero_prediction_classes": int(val_zero),
            "mean_entropy": float(val_entropy),
        },
        "test": {
            "class4_precision": float(test_c4["class4_precision"]),
            "class4_recall": float(test_c4["class4_recall"]),
            "macro_f1": float(test_macro),
            "zero_prediction_classes": int(test_zero),
            "mean_entropy": float(test_entropy),
            "confusion_matrix": conf_test.tolist(),
        },
        "ablation": {
            "without_thresholding": {
                "class4_precision": float(temp_only_c4["class4_precision"]),
                "class4_recall": float(temp_only_c4["class4_recall"]),
                "macro_f1": float(compute_macro_f1(y_test, test_pred_temp_only)),
                "mean_entropy": float(_normalized_entropy_from_probs(test_probs_cal)),
                "confusion_matrix": conf_temp_only.tolist(),
            },
            "without_temperature_scaling": {
                "class4_precision": float(thresh_only_c4["class4_precision"]),
                "class4_recall": float(thresh_only_c4["class4_recall"]),
                "macro_f1": float(compute_macro_f1(y_test, test_pred_thresh_only)),
                "mean_entropy": float(_normalized_entropy_from_probs(test_probs_uncal)),
                "confusion_matrix": conf_thresh_only.tolist(),
            },
        },
        "pr_curve_class4": {
            "precision": pr_precision_list,
            "recall": pr_recall_list,
            "thresholds": pr_threshold_list,
        },
        "threshold_sweep": {
            "tau_min": float(np.min(threshold_grid)),
            "tau_max": float(np.max(threshold_grid)),
            "num_points": int(np.asarray(threshold_grid).size),
            "points": sweep_points,
        },
    }
