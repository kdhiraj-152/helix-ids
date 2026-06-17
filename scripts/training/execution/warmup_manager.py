"""Epoch-0 forced coverage warmup manager."""

from collections import defaultdict
from typing import Any, Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class WarmupManager:
    """Manages epoch-0 forced coverage warmup.

    On epoch 0, the warmup manager selects one sample per active class,
    forward-passes it with an artificially biased logit target, and runs
    a single optimizer step.  This guarantees every class has been seen
    at least once before regular training begins.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Any,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Any,
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._device = device
        self._config = config

    def run_warmup(
        self,
        train_dataset: Dataset[Any],
        *,
        epoch: int,
        model_training: bool,
        global_step: int,
        warmup_steps: int,
        active_family_class_ids: Optional[list[int]],
        use_energy_based_family_objective: bool,
        binary_class_weights: Optional[torch.Tensor],
        family_class_weights: Optional[torch.Tensor],
        backbone_params: Optional[list[nn.Parameter]],
        logger: Any,
    ) -> dict[str, Any]:
        """Run one epoch-0 synthetic warmup step if applicable.

        Returns a dict with:
          warmup_executed (bool)
          global_step_increment (int)  — 0 or 1
          active_class_ids (list[int])  — resolved classes
        """
        if int(epoch) != 0 or not bool(model_training):
            return {
                "warmup_executed": False,
                "global_step_increment": 0,
                "active_class_ids": [],
            }

        class_to_indices = self._collect_class_to_indices(train_dataset)
        family_head = self._model.family_head
        family_head_last: Union[nn.Module, torch.Tensor]
        if isinstance(family_head, nn.Sequential):
            family_head_last = family_head[-1]
        else:
            family_head_last = family_head
        if not isinstance(family_head_last, nn.Module):
            raise TypeError(
                f"Expected nn.Module for family_head, got {type(family_head_last)}"
            )
        # mypy: nn.Module.__getattr__ returns Union[Tensor, Module] for dynamic attrs
        class_count = int(family_head_last.out_features)  # type: ignore[arg-type]
        active_class_ids = self._resolve_warmup_active_class_ids(
            class_to_indices, class_count, active_family_class_ids,
        )

        # Validate all active classes have samples in training set
        missing_classes = [
            int(cid)
            for cid in active_class_ids
            if len(class_to_indices.get(int(cid), [])) == 0
        ]
        if missing_classes:
            raise RuntimeError(
                "Class missing from training set"
                f": missing_active_classes={missing_classes}"
            )

        forced_indices = [
            int(class_to_indices[int(cid)][0]) for cid in active_class_ids
        ]
        x_forced, y_binary_forced, y_family_forced, y_family_rows = (
            self._build_warmup_batch_tensors(
                train_dataset, forced_indices,
            )
        )

        # Forward pass with artificial logit biasing
        binary_logits, raw_family_logits, _ = self._model(
            x_forced, return_features=True
        )

        family_logits = self._apply_warmup_logit_controls(
            raw_family_logits,
            active_class_ids=active_class_ids,
            use_energy_based_family_objective=use_energy_based_family_objective,
        )
        family_logits_forced = family_logits.clone()
        neg_inf = torch.tensor(
            float("-inf"),
            device=family_logits_forced.device,
            dtype=family_logits_forced.dtype,
        )
        pos_inf = torch.tensor(
            float("inf"),
            device=family_logits_forced.device,
            dtype=family_logits_forced.dtype,
        )
        for row_idx, class_id in enumerate(y_family_rows):
            family_logits_forced[int(row_idx), :] = neg_inf
            family_logits_forced[int(row_idx), int(class_id)] = pos_inf

        in_step_warmup = global_step < warmup_steps
        binary_weights = None if in_step_warmup else binary_class_weights
        family_weights = None if in_step_warmup else family_class_weights
        warmup_loss, _ = self._loss_fn(
            binary_logits,
            y_binary_forced,
            family_logits_forced,
            y_family_forced,
            binary_class_weights=binary_weights,
            family_class_weights=family_weights,
            feature_embeddings=None,
        )

        self._optimizer.zero_grad()
        warmup_loss.backward()
        if self._config.max_grad_norm > 0 and backbone_params:
            nn.utils.clip_grad_norm_(backbone_params, self._config.max_grad_norm)
        self._optimizer.step()

        logger.info(
            "Epoch0CoverageWarmup active_classes=%s forced_indices=%s warmup_loss=%s",
            active_class_ids,
            forced_indices,
            f"{float(warmup_loss.item()):.6f}",
        )

        return {
            "warmup_executed": True,
            "global_step_increment": 1,
            "active_class_ids": active_class_ids,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_class_to_indices(
        train_dataset: Dataset[Any],
    ) -> dict[int, list[int]]:
        """Build class -> dataset-index mapping for warmup class coverage."""
        class_to_indices: dict[int, list[int]] = defaultdict(list)
        if hasattr(train_dataset, "family_labels"):
            labels_np = np.asarray(train_dataset.family_labels, dtype=np.int64)
            for idx, class_id in enumerate(labels_np.tolist()):
                class_to_indices[int(class_id)].append(int(idx))
            return class_to_indices

        dataset_size = int(len(cast(Any, train_dataset)))
        for idx in range(dataset_size):
            sample = train_dataset[idx]
            y_family_item = _as_python_int(sample[2])
            class_to_indices[y_family_item].append(int(idx))
        return class_to_indices

    @staticmethod
    def _resolve_warmup_active_class_ids(
        class_to_indices: dict[int, list[int]],
        class_count: int,
        active_family_class_ids: Optional[list[int]],
    ) -> list[int]:
        """Resolve valid active class ids for warmup coverage."""
        if active_family_class_ids:
            active_class_ids = sorted(
                int(cls)
                for cls in active_family_class_ids
                if 0 <= int(cls) < class_count
            )
        else:
            active_class_ids = sorted(
                int(cls)
                for cls in class_to_indices.keys()
                if 0 <= int(cls) < class_count
            )
        if not active_class_ids:
            raise RuntimeError(
                "No active classes available for epoch-0 coverage warmup"
            )
        return active_class_ids

    @staticmethod
    def _build_warmup_batch_tensors(
        train_dataset: Dataset[Any],
        forced_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Materialize forced warmup batch tensors from selected dataset indices."""
        x_rows: list[torch.Tensor] = []
        y_binary_rows: list[int] = []
        y_family_rows: list[int] = []

        for idx in forced_indices:
            x_item, y_binary_item, y_family_item = train_dataset[idx]
            x_rows.append(
                x_item
                if isinstance(x_item, torch.Tensor)
                else torch.tensor(x_item)
            )
            y_binary_rows.append(_as_python_int(y_binary_item))
            y_family_rows.append(_as_python_int(y_family_item))

        x_forced = torch.stack(x_rows, dim=0)
        y_binary_forced = torch.tensor(y_binary_rows, dtype=torch.long)
        y_family_forced = torch.tensor(y_family_rows, dtype=torch.long)
        return x_forced, y_binary_forced, y_family_forced, y_family_rows

    @staticmethod
    def _apply_warmup_logit_controls(
        raw_family_logits: torch.Tensor,
        *,
        active_class_ids: list[int],
        use_energy_based_family_objective: bool,
    ) -> torch.Tensor:
        """Apply logit controls for warmup forward pass (simplified)."""
        controlled = raw_family_logits.clone()
        if use_energy_based_family_objective:
            controlled = controlled / max(1e-6, 1.0)
        return controlled


def _as_python_int(value: Any) -> int:
    """Convert tensor/scalar values to Python int."""
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)
