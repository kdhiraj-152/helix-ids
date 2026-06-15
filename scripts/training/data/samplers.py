"""
Balanced and frozen index samplers for HELIX-IDS full training.

Extracted from scripts/training/train_helix_ids_full.py — Phase 12B-3.
No behavioral changes. All symbols preserve original behavior.
"""

import math
from typing import Optional

import numpy as np
from torch.utils.data import Sampler

from .dataset_builder import build_class_index


def _default_tail_multiplier(count: int) -> float:
    """Return aggressive oversampling multipliers for extreme tail classes."""
    if count <= 64:
        return 80.0
    if count <= 1000:
        return 15.0
    return 1.0


def _inverse_frequency_weights(y: np.ndarray, *, minlength: int) -> np.ndarray:
    """Return mean-normalized inverse-frequency weights with zeros for missing classes."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=minlength).astype(np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    present_mask = counts > 0
    if bool(np.any(present_mask)):
        present = counts[present_mask]
        weights[present_mask] = present.sum() / (present_mask.sum() * present)
        weights[present_mask] /= max(1e-8, float(np.mean(weights[present_mask])))
    return weights.astype(np.float32)


def _sqrt_inverse_frequency_weights(y: np.ndarray, *, minlength: int) -> np.ndarray:
    """Return mean-normalized sqrt-inverse-frequency weights with zeros for missing classes."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=minlength).astype(np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    present_mask = counts > 0
    if bool(np.any(present_mask)):
        present = counts[present_mask]
        # Equivalent to 1 / sqrt(freq_i) up to a shared constant scaling.
        weights[present_mask] = np.sqrt(present.sum() / (present_mask.sum() * present))
        weights[present_mask] /= max(1e-8, float(np.mean(weights[present_mask])))
    return weights.astype(np.float32)


class ClassBalancedIndexSampler(Sampler[int]):
    """Yield flattened indices where each batch contains all active classes and tail oversampling.

    Extracted from train_helix_ids_full.py (lines 2315-2400).
    """

    def __init__(
        self,
        y: np.ndarray,
        batch_size: int,
        *,
        seed: int = 42,
        min_per_class: int = 1,
        class_multipliers: Optional[dict[int, float]] = None,
    ) -> None:
        self.y = np.asarray(y, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.min_per_class = max(1, int(min_per_class))

        self.class_index = build_class_index(self.y)
        self.classes = sorted(self.class_index.keys())
        if not self.classes:
            raise ValueError("ClassBalancedBatchSampler requires at least one class")

        self.class_counts = {
            class_id: int(self.class_index[class_id].shape[0]) for class_id in self.classes
        }
        if class_multipliers is None:
            self.class_multipliers = {
                class_id: _default_tail_multiplier(self.class_counts[class_id])
                for class_id in self.classes
            }
        else:
            self.class_multipliers = {
                class_id: float(class_multipliers.get(class_id, 1.0)) for class_id in self.classes
            }

        self.steps_per_epoch = max(1, int(math.ceil(self.y.shape[0] / max(1, self.batch_size))))
        required_slots = len(self.classes) * self.min_per_class
        if required_slots > self.batch_size:
            raise ValueError(
                "batch_size too small for per-class presence constraint: "
                f"required={required_slots}, batch_size={self.batch_size}"
            )
        self.remainder = self.batch_size - required_slots
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch * self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        for _ in range(self.steps_per_epoch):
            batch_indices: list[int] = []

            # Ensure every active class appears in each batch.
            for class_id in self.classes:
                cls_indices = self.class_index[class_id]
                sampled = rng.choice(cls_indices, size=self.min_per_class, replace=True)
                batch_indices.extend(sampled.tolist())

            if self.remainder > 0:
                sampling_weights = np.asarray(
                    [
                        float(self.class_counts[class_id]) * float(self.class_multipliers[class_id])
                        for class_id in self.classes
                    ],
                    dtype=np.float64,
                )
                if float(sampling_weights.sum()) <= 0.0:
                    sampling_weights = np.ones_like(sampling_weights)
                sampling_weights = sampling_weights / float(sampling_weights.sum())
                extra_classes = rng.choice(
                    self.classes,
                    size=self.remainder,
                    replace=True,
                    p=sampling_weights,
                )
                for class_id in extra_classes.tolist():
                    cls_indices = self.class_index[int(class_id)]
                    sampled = rng.choice(cls_indices, size=1, replace=True)
                    batch_indices.append(int(sampled[0]))

            rng.shuffle(batch_indices)
            for sample_idx in batch_indices:
                yield int(sample_idx)


class FrozenIndexSampler(Sampler[int]):
    """Deterministic sampler backed by precomputed indices.

    Extracted from train_helix_ids_full.py (lines 2403-2417).
    """

    def __init__(self, indices: np.ndarray) -> None:
        indices_np = np.asarray(indices, dtype=np.int64)
        if indices_np.size == 0:
            raise ValueError("FrozenIndexSampler requires at least one index")
        self.indices = indices_np

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __iter__(self):
        for idx in self.indices.tolist():
            yield int(idx)


def _build_frozen_class_balanced_indices(
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    min_per_class: int = 1,
    class_multipliers: Optional[dict[int, float]] = None,
) -> np.ndarray:
    """Build a fixed class-balanced index schedule once per run."""
    y_int = np.asarray(y, dtype=np.int64)
    class_index = build_class_index(y_int)
    classes = sorted(class_index.keys())
    if not classes:
        raise ValueError("No classes available for frozen sampler")

    min_per_class = max(1, int(min_per_class))
    required_slots = len(classes) * min_per_class
    if required_slots > int(batch_size):
        raise ValueError(
            "batch_size too small for per-class presence constraint: "
            f"required={required_slots}, batch_size={int(batch_size)}"
        )

    class_counts = {class_id: int(class_index[class_id].shape[0]) for class_id in classes}
    if class_multipliers is None:
        multipliers = {
            class_id: _default_tail_multiplier(class_counts[class_id])
            for class_id in classes
        }
    else:
        multipliers = {
            class_id: float(class_multipliers.get(class_id, 1.0))
            for class_id in classes
        }

    steps_per_epoch = max(1, int(math.ceil(y_int.shape[0] / max(1, int(batch_size)))))
    remainder = int(batch_size) - required_slots
    rng = np.random.default_rng(int(seed))
    flat_indices: list[int] = []

    sampling_weights = np.asarray(
        [float(class_counts[class_id]) * float(multipliers[class_id]) for class_id in classes],
        dtype=np.float64,
    )
    if float(sampling_weights.sum()) <= 0.0:
        sampling_weights = np.ones_like(sampling_weights)
    sampling_weights = sampling_weights / float(sampling_weights.sum())

    for _ in range(steps_per_epoch):
        batch_indices: list[int] = []
        for class_id in classes:
            cls_indices = class_index[class_id]
            sampled = rng.choice(cls_indices, size=min_per_class, replace=True)
            batch_indices.extend(int(x) for x in sampled.tolist())

        if remainder > 0:
            extra_classes = rng.choice(classes, size=remainder, replace=True, p=sampling_weights)
            for class_id in extra_classes.tolist():
                cls_indices = class_index[int(class_id)]
                sampled = rng.choice(cls_indices, size=1, replace=True)
                batch_indices.append(int(sampled[0]))

        rng.shuffle(batch_indices)
        flat_indices.extend(batch_indices)

    return np.asarray(flat_indices, dtype=np.int64)


def _build_frozen_tempered_indices(
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    temperature_power: float = 0.5,
    class_multipliers: Optional[dict[int, float]] = None,
) -> tuple[np.ndarray, dict[int, float]]:
    """Build fixed sampler indices with tempered class-frequency sampling.

    This relaxes strict class-balanced batches while preserving deterministic
    per-epoch exposure for all active classes.
    """
    y_int = np.asarray(y, dtype=np.int64)
    class_index = build_class_index(y_int)
    classes = sorted(class_index.keys())
    if not classes:
        raise ValueError("No classes available for tempered sampler")

    class_counts = {class_id: int(class_index[class_id].shape[0]) for class_id in classes}
    if class_multipliers is None:
        multipliers = dict.fromkeys(classes, 1.0)
    else:
        multipliers = {class_id: float(class_multipliers.get(class_id, 1.0)) for class_id in classes}

    power = float(max(0.0, temperature_power))
    raw_weights = np.asarray(
        [
            max(1.0, float(class_counts[class_id])) ** power
            * max(1e-6, float(multipliers[class_id]))
            for class_id in classes
        ],
        dtype=np.float64,
    )
    if float(raw_weights.sum()) <= 0.0:
        raw_weights = np.ones_like(raw_weights)
    class_probs = raw_weights / float(raw_weights.sum())

    steps_per_epoch = max(1, int(math.ceil(y_int.shape[0] / max(1, int(batch_size)))))
    total_draws = steps_per_epoch * int(batch_size)
    rng = np.random.default_rng(int(seed))

    # Guarantee at least one sample per active class each epoch.
    seeded_classes = np.asarray(classes, dtype=np.int64)
    rng.shuffle(seeded_classes)
    seeded_draws = min(int(seeded_classes.shape[0]), total_draws)

    sampled_class_ids = np.empty((total_draws,), dtype=np.int64)
    if seeded_draws > 0:
        sampled_class_ids[:seeded_draws] = seeded_classes[:seeded_draws]
    remaining = total_draws - seeded_draws
    if remaining > 0:
        sampled_class_ids[seeded_draws:] = rng.choice(
            np.asarray(classes, dtype=np.int64),
            size=remaining,
            replace=True,
            p=class_probs,
        )

    flat_indices: list[int] = []
    for class_id in sampled_class_ids.tolist():
        cls_indices = class_index[int(class_id)]
        sample_idx = int(rng.choice(cls_indices, size=1, replace=True)[0])
        flat_indices.append(sample_idx)

    return np.asarray(flat_indices, dtype=np.int64), {
        int(class_id): float(prob) for class_id, prob in zip(classes, class_probs.tolist())
    }
