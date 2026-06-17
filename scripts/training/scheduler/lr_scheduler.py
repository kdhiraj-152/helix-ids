"""lr_scheduler: Learning-rate schedule computation.

Phase 13A-2 extraction from HelixFullTrainer.

LRScheduler provides:
    - Cosine-decay LR with linear warmup
    - Phase-specific LR scale application logic
    - Current LR inspection

No optimizer references — pure LR computation.  Trainer wrappers apply
computed LR to optimizer param groups.
"""

from __future__ import annotations

import math
from typing import Any


class LRScheduler:
    """Learning-rate schedule with warmup + cosine decay.

    Computes base LR based on epoch and config.  The trainer wrapper
    applies LR and scales to the optimizer param groups.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        warmup_epochs: int,
        warmup_init_lr: float,
        epochs: int,
        min_lr_ratio: float = 0.05,
    ) -> None:
        self._learning_rate = float(learning_rate)
        self._warmup_epochs = int(max(0, warmup_epochs))
        self._warmup_init_lr = float(warmup_init_lr)
        self._epochs = int(max(1, epochs))
        self._min_lr_ratio = float(max(0.0, min_lr_ratio))
        self._min_lr = self._learning_rate * self._min_lr_ratio

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def warmup_epochs(self) -> int:
        return self._warmup_epochs

    @property
    def epochs(self) -> int:
        return self._epochs

    # ------------------------------------------------------------------ #
    # LR computation
    # ------------------------------------------------------------------ #

    def get_learning_rate(self, epoch: int) -> float:
        """Compute learning rate with linear warmup and cosine decay.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Base learning rate for this epoch.
        """
        epoch_idx = int(epoch)

        # Linear warmup
        if epoch_idx < self._warmup_epochs:
            warmup_denom = max(1, self._warmup_epochs)
            return float(
                self._warmup_init_lr
                + (self._learning_rate - self._warmup_init_lr)
                * ((epoch_idx + 1) / warmup_denom)
            )

        # Cosine decay
        decay_epochs = max(1, self._epochs - self._warmup_epochs)
        decay_step = min(epoch_idx - self._warmup_epochs, decay_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_epochs))
        return float(self._min_lr + (self._learning_rate - self._min_lr) * cosine_factor)

    @staticmethod
    def apply_lr_scales(
        param_groups: list[dict[str, Any]],
        base_lr_scales: dict[str, float],
        backbone_multiplier: float,
        head_multiplier: float,
    ) -> dict[str, float]:
        """Compute LR scale updates for each param group.

        Args:
            param_groups: Optimizer param groups (for reading current state).
            base_lr_scales: Base per-group LR scale snapshot.
            backbone_multiplier: Phase-specific backbone multiplier.
            head_multiplier: Phase-specific head multiplier.

        Returns:
            Dict mapping group_name → new lr_scale value.
        """
        backbone_mult = max(1e-4, float(backbone_multiplier))
        head_mult = max(1e-4, float(head_multiplier))

        new_scales: dict[str, float] = {}
        for idx, param_group in enumerate(param_groups):
            group_name = str(param_group.get("group_name", f"group_{idx}"))
            base_scale = float(base_lr_scales.get(group_name, param_group.get("lr_scale", 1.0)))

            if group_name == "backbone":
                new_scales[group_name] = base_scale * backbone_mult
            elif group_name == "family_head":
                new_scales[group_name] = base_scale * head_mult
            else:
                new_scales[group_name] = base_scale

        return new_scales
