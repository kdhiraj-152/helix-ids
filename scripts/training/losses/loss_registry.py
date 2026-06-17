"""Loss-registry module: single authoritative loss dispatcher for HelixFullTrainer.

Phase 14 consolidation: LossRegistry is now an instance-based class that
owns all loss composition, weighting, scaling, aggregation, and diagnostics
payload generation. Individual loss functions live in categorized sibling
modules (contrastive_losses, energy_losses, regularization_losses).

Public API:
    LossRegistry              — instance-based loss dispatch orchestrator

Dependency rules:
    loss_registry -> torch, contrastive_losses, energy_losses, regularization_losses
    loss_registry -> trainer internals (forbidden)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from scripts.training.losses.contrastive_losses import (
    centroid_repulsion_loss as _centroid_repulsion_loss,
)
from scripts.training.losses.contrastive_losses import (
    centroid_separation_barrier_loss as _centroid_separation_barrier_loss,
)
from scripts.training.losses.contrastive_losses import (
    compute_batch_class_centroids_for_loss as _compute_batch_class_centroids_for_loss,
)
from scripts.training.losses.contrastive_losses import (
    critical_pair_centroid_push_loss as _critical_pair_centroid_push_loss,
)
from scripts.training.losses.contrastive_losses import (
    global_centroid_guided_losses as _global_centroid_guided_losses,
)
from scripts.training.losses.contrastive_losses import (
    intra_class_variance_clamp_loss as _intra_class_variance_clamp_loss,
)
from scripts.training.losses.contrastive_losses import (
    pairwise_margin_repulsion_loss as _pairwise_margin_repulsion_loss,
)
from scripts.training.losses.contrastive_losses import (
    supcon_anchor_weights as _supcon_anchor_weights,
)
from scripts.training.losses.contrastive_losses import (
    supervised_contrastive_loss as _supervised_contrastive_loss,
)
from scripts.training.losses.energy_losses import (
    EnergyLossFunctions as _EnergyLossFunctions,
)
from scripts.training.losses.energy_losses import (
    EnergyStateManager as _EnergyStateManager,
)
from scripts.training.losses.regularization_losses import (
    apply_entropy_floor_regularizer as _apply_entropy_floor_regularizer,
)
from scripts.training.losses.regularization_losses import (
    apply_entropy_warmup as _apply_entropy_warmup,
)
from scripts.training.losses.regularization_losses import (
    apply_kl_uniform_regularization as _apply_kl_uniform_regularization,
)
from scripts.training.losses.regularization_losses import (
    apply_logit_floor_penalty as _apply_logit_floor_penalty,
)
from scripts.training.losses.regularization_losses import (
    apply_tail_ce_regularization as _apply_tail_ce_regularization,
)
from scripts.training.losses.regularization_losses import (
    compute_tail_focal_loss as _compute_tail_focal_loss,
)

__all__ = [
    "LossRegistry",
]


# --------------------------------------------------------------------------- #
# LossRegistry — instance-based dispatch orchestrator
# --------------------------------------------------------------------------- #


class LossRegistry:
    """Single authoritative loss dispatcher for HelixFullTrainer.

    Wraps all extracted loss functions from the categorized loss modules,
    manages energy win-rate EMA state, and provides ``compute_total_loss``
    as the main entry point for regularization composition.

    Parameters passed via keyword-args to ``__init__`` correspond to the
    trainer's config attributes.  See each method's docstring for details.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, **kwargs: Any) -> None:
        # --- Regularization config ---
        self._entropy_warmup_steps: int = int(kwargs.get("entropy_warmup_steps", 0))
        self._entropy_warmup_weight: float = float(kwargs.get("entropy_warmup_weight", 0.0))
        self._kl_uniform_weight: float = float(kwargs.get("kl_uniform_weight", 0.0))
        self._warmup_kl_uniform_weight: float = float(kwargs.get("warmup_kl_uniform_weight", 0.0))
        self._logit_floor: float = float(kwargs.get("logit_floor", 0.0))
        self._logit_floor_weight: float = float(kwargs.get("logit_floor_weight", 0.0))
        self._tail_ce_weight: float = float(kwargs.get("tail_ce_weight", 0.0))
        self._tail_class_mask: torch.Tensor | None = kwargs.get("tail_class_mask", None)
        self._loss_fn: nn.Module | None = kwargs.get("loss_fn", None)

        # --- Energy loss config ---
        self._energy_gap_weight: float = float(kwargs.get("energy_gap_weight", 0.0))
        self._energy_multi_negative_alpha: float = float(kwargs.get(
            "energy_multi_negative_alpha", 1.0,
        ))
        self._energy_balance_weight: float = float(kwargs.get("energy_balance_weight", 0.0))
        self._energy_winner_weight: float = float(kwargs.get("energy_winner_weight", 0.0))
        self._energy_winner_min_count: int = int(kwargs.get("energy_winner_min_count", 1))
        self._energy_logit_temperature: float = float(kwargs.get("energy_logit_temperature", 1.0))

        # --- Energy state manager ---
        self._energy_manager = _EnergyStateManager(
            energy_win_rate_ema_momentum=float(kwargs.get(
                "energy_win_rate_ema_momentum", 0.9,
            )),
            energy_emergence_bias_eps=float(kwargs.get(
                "energy_emergence_bias_eps", 1e-3,
            )),
            energy_emergence_bias_beta=float(kwargs.get(
                "energy_emergence_bias_beta", 0.5,
            )),
            energy_emergence_bias_ratio_max=float(kwargs.get(
                "energy_emergence_bias_ratio_max", 0.30,
            )),
        )

    # ------------------------------------------------------------------ #
    # Static delegates — backward-compatible pure loss functions
    # ------------------------------------------------------------------ #

    @staticmethod
    def supervised_contrastive_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        temperature: float,
        anchor_weights: torch.Tensor | None = None,
        negative_weight: float = 1.0,
        min_negatives: int = 1,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss (delegated to contrastive_losses)."""
        return _supervised_contrastive_loss(
            features, labels,
            temperature=temperature,
            anchor_weights=anchor_weights,
            negative_weight=negative_weight,
            min_negatives=min_negatives,
        )

    @staticmethod
    def supcon_anchor_weights(labels: torch.Tensor) -> torch.Tensor:
        """Build class-balanced anchor weights (delegated to contrastive_losses)."""
        return _supcon_anchor_weights(labels)

    @staticmethod
    def class_conditional_energy_gap_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        alpha: float,
    ) -> tuple[torch.Tensor, float, float, float, float]:
        """Compute class-conditional energy gap loss (delegated to energy_losses)."""
        return _EnergyLossFunctions.class_conditional_energy_gap_loss(
            logits, labels, alpha=alpha,
        )

    @staticmethod
    def energy_class_balance_loss(
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float]:
        """Compute energy class balance loss (delegated to energy_losses)."""
        return _EnergyLossFunctions.energy_class_balance_loss(logits)

    @staticmethod
    def energy_min_winner_loss(
        logits: torch.Tensor,
        active_class_ids: list[int] | None,
        *,
        min_winners: int,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute energy min-winner loss (delegated to energy_losses)."""
        return _EnergyLossFunctions.energy_min_winner_loss(
            logits, active_class_ids, min_winners=min_winners,
        )

    @staticmethod
    def pairwise_margin_repulsion_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        margin: float,
        hard_negative_weight: float = 1.0,
        top_k: int = 3,
    ) -> torch.Tensor:
        """Compute pairwise margin repulsion loss (delegated to contrastive_losses)."""
        return _pairwise_margin_repulsion_loss(
            features, labels,
            margin=margin,
            hard_negative_weight=hard_negative_weight,
            top_k=top_k,
        )

    @staticmethod
    def centroid_separation_barrier_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        min_distance: float,
    ) -> torch.Tensor:
        """Compute centroid separation barrier loss (delegated to contrastive_losses)."""
        return _centroid_separation_barrier_loss(
            features, labels, min_distance=min_distance,
        )

    @staticmethod
    def centroid_repulsion_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        margin: float,
    ) -> torch.Tensor:
        """Compute centroid repulsion loss (delegated to contrastive_losses)."""
        return _centroid_repulsion_loss(features, labels, margin=margin)

    @staticmethod
    def intra_class_variance_clamp_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        var_lower_bound: float,
        var_upper_bound: float,
    ) -> torch.Tensor:
        """Penalize per-class embedding variance (delegated to contrastive_losses)."""
        return _intra_class_variance_clamp_loss(
            features, labels,
            var_lower_bound=var_lower_bound,
            var_upper_bound=var_upper_bound,
        )

    @staticmethod
    def compute_batch_class_centroids_for_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        """Compute differentiable per-class centroids (delegated to contrastive_losses)."""
        return _compute_batch_class_centroids_for_loss(features, labels)

    @staticmethod
    def global_centroid_guided_losses(
        batch_centroids: torch.Tensor,
        class_ids: list[int],
        epoch_frozen_centroids: dict[int, torch.Tensor],
        *,
        rep_centroid_repulsion_margin: float,
        rep_centroid_barrier_min_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Compute centroid forces against running global centroids (delegated)."""
        return _global_centroid_guided_losses(
            batch_centroids, class_ids,
            epoch_frozen_centroids,
            rep_centroid_repulsion_margin=rep_centroid_repulsion_margin,
            rep_centroid_barrier_min_distance=rep_centroid_barrier_min_distance,
        )

    @staticmethod
    def critical_pair_centroid_push_loss(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        min_distance: float,
        critical_collision_pairs: set[tuple[int, int]],
    ) -> torch.Tensor:
        """Apply direct centroid push for known collision pairs (delegated)."""
        return _critical_pair_centroid_push_loss(
            features, labels,
            min_distance=min_distance,
            critical_collision_pairs=critical_collision_pairs,
        )

    @staticmethod
    def compute_representation_energy_objective(
        *,
        family_logits_train: torch.Tensor,
        y_family: torch.Tensor,
        y_binary: torch.Tensor,
        binary_logits: torch.Tensor,
        active_family_class_ids: list[int],
        loss_fn: torch.nn.Module,
        energy_gap_weight: float,
        energy_multi_negative_alpha: float,
        energy_balance_weight: float,
        energy_winner_weight: float,
        energy_winner_min_count: int,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute representation-phase energy objective (delegated to energy_losses)."""
        return _EnergyLossFunctions.compute_representation_energy_objective(
            family_logits_train=family_logits_train,
            y_family=y_family,
            y_binary=y_binary,
            binary_logits=binary_logits,
            active_family_class_ids=active_family_class_ids,
            loss_fn=loss_fn,
            energy_gap_weight=energy_gap_weight,
            energy_multi_negative_alpha=energy_multi_negative_alpha,
            energy_balance_weight=energy_balance_weight,
            energy_winner_weight=energy_winner_weight,
            energy_winner_min_count=energy_winner_min_count,
            epoch=epoch,
        )

    # ------------------------------------------------------------------ #
    # Tail focal loss
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_tail_focal_loss(
        family_logits_train: torch.Tensor,
        y_family: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal tail stabilization term for classes 3/4."""
        return _compute_tail_focal_loss(family_logits_train, y_family)

    # ------------------------------------------------------------------ #
    # Entropy floor regularizer
    # ------------------------------------------------------------------ #

    @staticmethod
    def apply_entropy_floor_regularizer_to_loss(
        loss: torch.Tensor,
        *,
        family_logits_train: torch.Tensor,
        active_class_count: int,
    ) -> torch.Tensor:
        """Apply entropy floor regularization on family logits."""
        return _apply_entropy_floor_regularizer(
            loss,
            family_logits_train=family_logits_train,
            active_class_count=active_class_count,
        )

    # ------------------------------------------------------------------ #
    # Energy-EMA state management  (delegated to EnergyStateManager)
    # ------------------------------------------------------------------ #

    @property
    def energy_win_rate_ema(self) -> torch.Tensor | None:
        """Current energy win-rate EMA tensor (read-only)."""
        return self._energy_manager.energy_win_rate_ema

    @property
    def energy_bias_last_std(self) -> float:
        """Last computed energy bias standard deviation."""
        return self._energy_manager.energy_bias_last_std

    @property
    def energy_bias_last_max_abs(self) -> float:
        """Last computed energy bias max absolute value."""
        return self._energy_manager.energy_bias_last_max_abs

    @property
    def energy_bias_last_logit_std(self) -> float:
        """Last logit standard deviation used for bias scaling."""
        return self._energy_manager.energy_bias_last_logit_std

    def ensure_energy_win_rate_ema(
        self,
        class_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Initialize or resize the EMA tracker for per-class argmax win rates."""
        return self._energy_manager.ensure_energy_win_rate_ema(
            class_count, device=device, dtype=dtype,
        )

    def update_energy_win_rate_ema(
        self,
        logits: torch.Tensor,
        *,
        active_class_ids: list[int] | None = None,
    ) -> None:
        """Update EMA win-rate estimate from hard argmax class wins."""
        self._energy_manager.update_energy_win_rate_ema(
            logits,
            active_class_ids=active_class_ids,
        )

    def compute_energy_emergence_bias(
        self,
        logits: torch.Tensor,
        *,
        active_class_ids: list[int] | None = None,
        active_family_class_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Compute per-class pre-argmax bias from inverse EMA win rates."""
        return self._energy_manager.compute_energy_emergence_bias(
            logits,
            active_class_ids=active_class_ids,
            active_family_class_ids=active_family_class_ids,
        )

    # ------------------------------------------------------------------ #
    # compute_loss_with_optional_energy  — composite orchestrator
    # ------------------------------------------------------------------ #

    def compute_loss_with_optional_energy(
        self,
        *,
        classification_loss: torch.Tensor,
        family_logits_train: torch.Tensor,
        y_family: torch.Tensor,
        y_binary: torch.Tensor,
        binary_logits: torch.Tensor,
        in_representation_phase: bool,
        active_family_class_ids: list[int],
        use_energy_based_family_objective: bool,
        disable_tail_focal_regularizer: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute base loss and optionally replace it with representation energy objective.

        This replaces ``HelixFullTrainer._compute_loss_with_optional_energy``.
        """
        tail_focal_loss = torch.zeros(
            (),
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        )
        if not disable_tail_focal_regularizer:
            tail_focal_loss = _compute_tail_focal_loss(family_logits_train, y_family)
        loss = classification_loss + (0.1 * tail_focal_loss)

        energy_diag: dict[str, float] = {
            "mean_e_y": 0.0,
            "mean_e_others": 0.0,
            "mean_gap": 0.0,
            "mean_energy_total": 0.0,
            "mean_balance_kl": 0.0,
            "mean_pred_entropy": 0.0,
            "min_pred_mass": 0.0,
            "mean_winner_deficit": 0.0,
            "min_winner_count": 0.0,
            "effective_energy_balance_weight": 0.0,
            "effective_energy_winner_weight": 0.0,
        }
        use_energy_objective = (
            in_representation_phase and use_energy_based_family_objective
        )
        if use_energy_objective:
            # loss_fn is guaranteed non-None when use_energy_based_family_objective is True
            from typing import cast
            loss, energy_diag = self.compute_representation_energy_objective(
                family_logits_train=family_logits_train,
                y_family=y_family,
                y_binary=y_binary,
                binary_logits=binary_logits,
                active_family_class_ids=active_family_class_ids,
                loss_fn=cast(nn.Module, self._loss_fn),
                energy_gap_weight=self._energy_gap_weight,
                energy_multi_negative_alpha=self._energy_multi_negative_alpha,
                energy_balance_weight=self._energy_balance_weight,
                energy_winner_weight=self._energy_winner_weight,
                energy_winner_min_count=self._energy_winner_min_count,
                epoch=0,  # callers pass epoch via compute_total_loss when needed
            )
        return loss, energy_diag

    # ------------------------------------------------------------------ #
    # compute_total_loss  — main regularization dispatch (replaces
    #                       _apply_loss_regularizations)
    # ------------------------------------------------------------------ #

    def compute_total_loss(
        self,
        loss: torch.Tensor,
        family_logits_train: torch.Tensor,
        raw_family_logits: torch.Tensor,
        y_family: torch.Tensor,
        *,
        epoch: int,
        global_step: int,
        in_step_warmup: bool,
        entropy_warmup_steps: int | None = None,
        entropy_warmup_weight: float | None = None,
        kl_uniform_weight: float | None = None,
        warmup_kl_uniform_weight: float | None = None,
        logit_floor: float | None = None,
        logit_floor_weight: float | None = None,
        tail_ce_weight: float | None = None,
        tail_class_mask: torch.Tensor | None = None,
        loss_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        """Apply all loss regularization terms.

        Single dispatch point replacing ``HelixFullTrainer._apply_loss_regularizations``.
        Most parameters have defaults drawn from the registry's config, but can be
        overridden per-call (supporting the trainer's dynamic overrides).
        """
        # Resolve effective params (caller override → registry default)
        _entropy_warmup_steps: int = (
            entropy_warmup_steps if entropy_warmup_steps is not None
            else self._entropy_warmup_steps
        )
        _entropy_warmup_weight: float = (
            entropy_warmup_weight if entropy_warmup_weight is not None
            else self._entropy_warmup_weight
        )
        _kl_uniform_weight: float = (
            kl_uniform_weight if kl_uniform_weight is not None
            else self._kl_uniform_weight
        )
        _warmup_kl_uniform_weight: float = (
            warmup_kl_uniform_weight if warmup_kl_uniform_weight is not None
            else self._warmup_kl_uniform_weight
        )
        _logit_floor: float = (
            logit_floor if logit_floor is not None
            else self._logit_floor
        )
        _logit_floor_weight: float = (
            logit_floor_weight if logit_floor_weight is not None
            else self._logit_floor_weight
        )
        _tail_ce_weight: float = (
            tail_ce_weight if tail_ce_weight is not None
            else self._tail_ce_weight
        )
        _tail_class_mask: torch.Tensor | None = (
            tail_class_mask if tail_class_mask is not None
            else self._tail_class_mask
        )
        _loss_fn: nn.Module | None = (
            loss_fn if loss_fn is not None
            else self._loss_fn
        )

        # --- 1. Entropy warmup regularization ---
        loss = _apply_entropy_warmup(
            loss,
            family_logits_train=family_logits_train,
            entropy_warmup_weight=_entropy_warmup_weight,
            global_step=global_step,
            entropy_warmup_steps=_entropy_warmup_steps,
            epoch=epoch,
        )

        # --- 2. KL divergence to uniform distribution ---
        loss = _apply_kl_uniform_regularization(
            loss,
            family_logits_train=family_logits_train,
            kl_uniform_weight=_kl_uniform_weight,
            warmup_kl_uniform_weight=_warmup_kl_uniform_weight,
            in_step_warmup=in_step_warmup,
            epoch=epoch,
        )

        # --- 3. Logit floor penalty ---
        loss = _apply_logit_floor_penalty(
            loss,
            raw_family_logits=raw_family_logits,
            logit_floor_weight=_logit_floor_weight,
            logit_floor=_logit_floor,
            in_step_warmup=in_step_warmup,
        )

        # --- 4. Tail class cross-entropy ---
        loss = _apply_tail_ce_regularization(
            loss,
            family_logits_train=family_logits_train,
            y_family=y_family,
            tail_ce_weight=_tail_ce_weight,
            tail_class_mask=_tail_class_mask,
            loss_fn=_loss_fn,
        )

        return loss

    # ------------------------------------------------------------------ #
    # Diagnostics generation
    # ------------------------------------------------------------------ #

    def log_energy_gap_diag_message(
        self,
        global_step: int,
        diagnostics: dict[str, float],
    ) -> str:
        """Build the formatted energy gap diagnostics log message.

        Returns the formatted log string (the trainer's logger.info call
        stays in the trainer; this builds the message content).
        """
        return (
            f"EnergyGapDiag step={int(global_step)} "
            f"weight={self._energy_gap_weight:.3f} "
            f"alpha={self._energy_multi_negative_alpha:.3f} "
            f"T={self._energy_logit_temperature:.3f} "
            f"balance_w={self._energy_balance_weight:.3f} "
            f"winner_w={self._energy_winner_weight:.3f} "
            f"winner_m={self._energy_winner_min_count} "
            f"emergence_beta={self._energy_manager._energy_emergence_bias_beta:.3f} "
            f"emergence_eps={self._energy_manager._energy_emergence_bias_eps:.1e} "
            f"winrate_m={self._energy_manager._energy_win_rate_ema_momentum:.3f} "
            f"bias_std={self._energy_manager.energy_bias_last_std:.6f} "
            f"bias_max={self._energy_manager.energy_bias_last_max_abs:.6f} "
            f"logit_std={self._energy_manager.energy_bias_last_logit_std:.6f} "
            f"E_y={diagnostics['mean_e_y']:.6f} "
            f"E_neg_lse={diagnostics['mean_e_others']:.6f} "
            f"gap_all={diagnostics['mean_gap']:.6f} "
            f"energy_total={diagnostics['mean_energy_total']:.6f} "
            f"balance_kl={diagnostics['mean_balance_kl']:.6f} "
            f"pred_entropy={diagnostics['mean_pred_entropy']:.6f} "
            f"min_pred_mass={diagnostics['min_pred_mass']:.6f} "
            f"winner_deficit={diagnostics['mean_winner_deficit']:.6f} "
            f"min_winner_count={diagnostics['min_winner_count']:.6f}"
        )

    def reset_energy_win_rate_ema(self) -> None:
        """Force re-initialization of the EMA tracker (delegated to EnergyStateManager)."""
        self._energy_manager.reset_energy_win_rate_ema()
