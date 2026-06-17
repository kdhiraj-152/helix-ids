"""Evaluation helpers for HELIX-IDS: family output collection, metrics, A/B evaluation."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from helix_ids.models.full import HelixIDSFull

# Canonical implementations live in scripts/training/governance/ (Phase 13A-5).
# Compatibility aliases preserve the private-name public API across tests.
from scripts.training.governance import (  # noqa: F401
    ab_rejection as _ab_rejection,
)
from scripts.training.governance import (
    build_ab_raw_metrics as _build_ab_raw_metrics,
)
from scripts.training.governance import (
    detect_cluster_mode_collapse as _detect_cluster_mode_collapse,
)
from scripts.training.governance import (
    detect_feature_and_objective_changes as _detect_feature_and_objective_changes,
)
from scripts.training.governance import (
    evaluate_ab_candidate,
)
from scripts.training.governance import (
    normalized_entropy_from_counts as _normalized_entropy_from_counts,
)
from scripts.training.governance import (
    validate_ab_contract as _validate_ab_contract,
)
from scripts.training.governance import (
    validate_track as _validate_track,
)

__all__ = [
    "_ab_rejection",
    "_build_ab_raw_metrics",
    "_collect_eval_family_outputs",
    "_detect_cluster_mode_collapse",
    "_detect_feature_and_objective_changes",
    "_normalized_entropy_from_counts",
    "_normalized_entropy_from_probs",
    "_validate_ab_contract",
    "_validate_track",
    "evaluate_ab_candidate",
]


def _collect_eval_family_outputs(
    *,
    model: HelixIDSFull,
    loader: DataLoader,
    device: str,
    active_class_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect family-label evaluation outputs (labels, logits, probs) for calibration."""
    model.eval()
    labels_chunks: list[np.ndarray] = []
    logits_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for x, _y_binary, y_family in loader:
            x_dev = x.to(device, non_blocking=True)
            _binary_logits, family_logits_dev = model(x_dev)
            family_logits = family_logits_dev.detach().to(device="cpu")

            if active_class_ids:
                allowed = [
                    int(cls)
                    for cls in sorted(active_class_ids)
                    if 0 <= int(cls) < int(family_logits.shape[1])
                ]
                if allowed:
                    mask = torch.full_like(family_logits, float("-inf"))
                    mask[:, allowed] = family_logits[:, allowed]
                    family_logits = mask

            logits_chunks.append(family_logits.numpy().astype(np.float64, copy=False))
            labels_chunks.append(
                y_family.to(device="cpu", dtype=torch.long, non_blocking=True)
                .numpy()
                .astype(np.int64, copy=False)
            )

    if not logits_chunks:
        return (
            np.array([], dtype=np.int64),
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        )

    labels = np.concatenate(labels_chunks, axis=0).astype(np.int64, copy=False)
    logits = np.concatenate(logits_chunks, axis=0).astype(np.float64, copy=False)
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

    logits_shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shift)
    probs = exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-12, None)
    return labels, logits, probs


def _normalized_entropy_from_probs(probs: np.ndarray) -> float:
    """Compute mean normalized entropy in [0, 1] from class probabilities."""
    if probs.ndim != 2 or probs.shape[0] == 0:
        return 0.0
    safe = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    ent = -np.sum(safe * np.log(safe), axis=1)
    class_count = int(safe.shape[1])
    if class_count <= 1:
        return 0.0
    return float(np.mean(ent / math.log(float(class_count))))

