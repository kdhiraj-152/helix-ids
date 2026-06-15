"""
Numpy-backed Dataset and indexing helpers for HELIX-IDS full training.

Extracted from scripts/training/train_helix_ids_full.py — Phase 12B-3.
No behavioral changes. All public symbols are backward-compatible.
"""

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiTaskNumpyDataset(Dataset):
    """Lazy dataset backed by numpy arrays (or memmaps) for multi-task labels.

    Extracted from train_helix_ids_full.py (lines 1901-1923).
    """

    def __init__(self, features: np.ndarray, family_labels: np.ndarray):
        if int(features.shape[0]) != int(family_labels.shape[0]):
            raise ValueError(
                f"Feature/label length mismatch: X={features.shape[0]}, y={family_labels.shape[0]}"
            )
        self.features = features
        self.family_labels = np.asarray(family_labels, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.family_labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_family = int(self.family_labels[idx])
        y_binary = 1 if y_family != 0 else 0
        x_row = np.asarray(self.features[idx], dtype=np.float32)
        return (
            torch.from_numpy(x_row),
            torch.tensor(y_binary, dtype=torch.long),
            torch.tensor(y_family, dtype=torch.long),
        )


def _chunk_finite_check(x: np.ndarray, chunk_rows: int = 250000) -> bool:
    """Check finite values in chunks to avoid large temporary allocations."""
    for start_idx in range(0, int(x.shape[0]), chunk_rows):
        chunk = np.asarray(x[start_idx : start_idx + chunk_rows], dtype=np.float32)
        if not np.isfinite(chunk).all():
            return False
    return True


def _sample_rows(x: np.ndarray, *, seed: int, max_rows: int = 50000) -> np.ndarray:
    """Sample rows for distribution checks without loading full arrays into memory."""
    n_rows = int(x.shape[0])
    if n_rows <= max_rows:
        return np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, size=max_rows, replace=False)
    return np.asarray(x[idx], dtype=np.float32)


def build_class_index(y: np.ndarray) -> dict[int, np.ndarray]:
    """Build per-class index lists for balanced batch sampling."""
    class_index: defaultdict[int, list[int]] = defaultdict(list)
    y_int = np.asarray(y, dtype=np.int64)
    for idx, label in enumerate(y_int.tolist()):
        class_index[int(label)].append(idx)
    return {label: np.asarray(idxs, dtype=np.int64) for label, idxs in class_index.items()}


def _build_stratified_val_subset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a stratified validation subset with at least target_per_class per active class."""
    y_int = np.asarray(y, dtype=np.int64)
    selected_idx = _build_stratified_subset_indices(
        y_int,
        target_per_class=target_per_class,
        seed=seed,
    )
    return (
        np.asarray(x[selected_idx], dtype=np.float32),
        np.asarray(y_int[selected_idx], dtype=np.int64),
    )


def _build_stratified_subset_indices(
    y: np.ndarray,
    *,
    target_per_class: int,
    seed: int,
) -> np.ndarray:
    """Return deterministic stratified indices for reproducible evaluation subsets."""
    y_int = np.asarray(y, dtype=np.int64)
    class_index = build_class_index(y_int)
    if not class_index:
        return np.arange(y_int.shape[0], dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(class_index):
        idxs = class_index[class_id]
        sampled = rng.choice(
            idxs,
            size=int(target_per_class),
            replace=bool(idxs.size < int(target_per_class)),
        )
        selected.extend(int(i) for i in sampled.tolist())

    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)
