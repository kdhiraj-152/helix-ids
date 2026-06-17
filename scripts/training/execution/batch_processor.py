"""Batch processor — single training batch forward, loss, backprop."""

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from scripts.training.losses import LossRegistry


class BatchProcessor:
    """Process a single training batch: forward, loss dispatch, backprop.

    Constructor receives stable model/registry references.
    Per-call data (batch tensors, flags, current state values) arrives
    as keyword arguments to ``process_batch``.
    Mutating side effects (optimizer step, state updates) are returned
    as structured result values the caller applies.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Any,
        loss_registry: LossRegistry,
        config: Any,
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._loss_registry = loss_registry
        self._config = config

        # Mutable temperature state
        self.logit_temp: float = 1.0
        self.temperature_calibration: float = 1.0

    # ------------------------------------------------------------------
    # Public API  —  one training batch
    # ------------------------------------------------------------------

    def process_batch(
        self,
        x: torch.Tensor,
        y_binary: torch.Tensor,
        y_family: torch.Tensor,
        *,
        in_step_warmup: bool,
        in_representation_phase: bool,
        optimizer: torch.optim.Optimizer,
        backbone_params: Optional[list[nn.Parameter]],
        global_step: int,
        warmup_steps: int,
        binary_class_weights: Optional[torch.Tensor],
        family_class_weights: Optional[torch.Tensor],
        active_family_class_ids: list[int],
        use_energy_based_family_objective: bool,
        disable_tail_focal_regularizer: bool = False,
        energy_logit_temperature: float = 1.0,
        family_log_prior: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Forward, loss, backprop, metrics for one training batch.

        Returns a dict with keys:
          loss, raw_family_logits, family_pred,
          binary_correct, family_correct, batch_size,
          energy_diag, global_step_increment (int),
          rep_feature_chunks_delta, rep_label_chunks_delta (lists).
        """
        # --- Forward pass ---
        binary_logits, raw_family_logits, backbone_features = self._model(
            x, return_features=True
        )

        active_ids = self._resolve_batch_active_family_class_ids(
            raw_family_logits, y_family, active_family_class_ids,
        )
        family_logits_train = self._stabilize_batch_family_logits(
            raw_family_logits,
            active_family_class_ids=active_ids,
            use_energy_based_family_objective=use_energy_based_family_objective,
            energy_logit_temperature=energy_logit_temperature,
            family_log_prior=family_log_prior,
        )

        family_pred = torch.argmax(family_logits_train, dim=1)

        # Update energy EMA if needed
        if use_energy_based_family_objective:
            self._loss_registry.update_energy_win_rate_ema(
                family_logits_train,
                active_class_ids=active_ids,
            )

        # --- Classification loss ---
        bw = None if in_step_warmup else binary_class_weights
        fw = None if in_step_warmup else family_class_weights
        classification_loss, _ = self._loss_fn(
            binary_logits,
            y_binary,
            family_logits_train,
            y_family,
            binary_class_weights=bw,
            family_class_weights=fw,
            feature_embeddings=backbone_features,
        )

        # --- Optional energy objective ---
        loss, energy_diag = self._loss_registry.compute_loss_with_optional_energy(
            classification_loss=classification_loss,
            family_logits_train=family_logits_train,
            y_family=y_family,
            y_binary=y_binary,
            binary_logits=binary_logits,
            in_representation_phase=in_representation_phase,
            active_family_class_ids=active_ids,
            use_energy_based_family_objective=use_energy_based_family_objective,
            disable_tail_focal_regularizer=disable_tail_focal_regularizer,
        )

        # --- Regularizations ---
        active_class_count = max(1, int(len(active_ids)))
        loss = LossRegistry.apply_entropy_floor_regularizer_to_loss(
            loss,
            family_logits_train=family_logits_train,
            active_class_count=active_class_count,
        )
        if not in_representation_phase:
            loss = self._loss_registry.compute_total_loss(
                loss,
                family_logits_train,
                raw_family_logits,
                y_family,
                epoch=0,
                global_step=0,
                in_step_warmup=in_step_warmup,
            )

        # --- Backprop ---
        self._backpropagate(
            loss,
            optimizer=optimizer,
            backbone_params=backbone_params,
            in_representation_phase=in_representation_phase,
        )

        # --- Metrics ---
        batch_size = int(y_binary.shape[0])
        binary_correct = int(
            (torch.argmax(binary_logits, dim=1) == y_binary).sum().item()
        )
        family_correct = int((family_pred == y_family).sum().item())

        return {
            "loss": loss,
            "raw_family_logits": raw_family_logits,
            "family_pred": family_pred,
            "binary_correct": binary_correct,
            "family_correct": family_correct,
            "batch_size": batch_size,
            "energy_diag": energy_diag,
            "global_step_increment": 1,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_batch_active_family_class_ids(
        raw_family_logits: torch.Tensor,
        y_family: torch.Tensor,
        active_family_class_ids: Optional[list[int]],
    ) -> list[int]:
        """Resolve active class ids for a train batch."""
        class_count = int(raw_family_logits.shape[1])
        if active_family_class_ids:
            return sorted(
                int(cls)
                for cls in active_family_class_ids
                if 0 <= int(cls) < class_count
            )
        return sorted(
            int(cls) for cls in torch.unique(y_family.detach(), dim=0).tolist()
        )

    def _stabilize_batch_family_logits(
        self,
        raw_family_logits: torch.Tensor,
        *,
        active_family_class_ids: list[int],
        use_energy_based_family_objective: bool,
        energy_logit_temperature: float = 1.0,
        family_log_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply logit controls and lightweight temperature stabilization."""
        controlled_logits = raw_family_logits.clone()

        if use_energy_based_family_objective:
            emergence_bias = self._loss_registry.compute_energy_emergence_bias(
                controlled_logits,
                active_class_ids=active_family_class_ids,
                active_family_class_ids=active_family_class_ids,
            )
            controlled_logits = controlled_logits + emergence_bias
            controlled_logits = controlled_logits / max(1e-6, energy_logit_temperature)

        if family_log_prior is not None:
            if int(family_log_prior.shape[-1]) != int(controlled_logits.shape[-1]):
                raise RuntimeError(
                    "family prior dimension mismatch: "
                    f"priors={int(family_log_prior.shape[-1])} logits={int(controlled_logits.shape[-1])}"
                )
            controlled_logits = controlled_logits - family_log_prior

        current_std = controlled_logits.detach().std().clamp(min=1e-6)
        self.logit_temp = (0.9 * float(self.logit_temp)) + (
            0.1 * float(current_std.item())
        )
        controlled_logits = controlled_logits / max(1e-6, float(self.logit_temp))

        calib_temp = max(1e-6, float(self.temperature_calibration))
        controlled_logits = controlled_logits / calib_temp
        with torch.no_grad():
            probs_calib = torch.softmax(controlled_logits, dim=1)
            safe_probs_calib = torch.clamp(probs_calib, min=1e-12, max=1.0)
            class_count = max(2, int(controlled_logits.shape[1]))
            uniform_logp = -math.log(float(class_count))
            temp_kl = torch.sum(
                probs_calib * (torch.log(safe_probs_calib) - uniform_logp),
                dim=1,
            ).mean()
            self.temperature_calibration = max(
                0.5,
                min(
                    5.0,
                    float(self.temperature_calibration)
                    + (1.0 * float(temp_kl.item())),
                ),
            )

        return torch.clamp(controlled_logits, -10.0, 10.0)

    def _backpropagate(
        self,
        loss: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
        backbone_params: Optional[list[nn.Parameter]],
        in_representation_phase: bool,
    ) -> None:
        """Run backward pass, gradient clipping, and optimizer step."""
        optimizer.zero_grad()
        loss.backward()

        if self._config.max_grad_norm > 0 and backbone_params:
            nn.utils.clip_grad_norm_(backbone_params, self._config.max_grad_norm)

        optimizer.step()
