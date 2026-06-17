"""Energy-based loss functions and EMA state management extracted from HelixFullTrainer.

Provides two classes:

1. ``EnergyLossFunctions`` — pure static-method energy loss computations
   (copied from ``LossRegistry`` with no trainer dependency).

2. ``EnergyStateManager`` — instance-based manager for energy win-rate EMA
   and emergence bias computation, extracted from
   ``HelixFullTrainer._ensure_energy_win_rate_ema``,
   ``HelixFullTrainer._update_energy_win_rate_ema``, and
   ``HelixFullTrainer._compute_energy_emergence_bias``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class EnergyLossFunctions:
    """Pure-energy loss functions extracted from ``LossRegistry``.

    All methods are @staticmethod — they accept explicit tensor parameters
    and configuration values; no trainer state required.
    """

    # ------------------------------------------------------------------ #
    # Energy gap loss
    # ------------------------------------------------------------------ #

    @staticmethod
    def class_conditional_energy_gap_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        alpha: float,
    ) -> tuple[torch.Tensor, float, float, float, float]:
        """Compute class-conditional multi-negative energy ordering loss from family logits.

        Returns:
            (loss, mean_e_y, mean_e_others, mean_gap, mean_energy_total)
        """
        if int(logits.ndim) != 2 or int(logits.shape[0]) == 0:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return zero, 0.0, 0.0, 0.0, 0.0
        if int(logits.shape[1]) <= 1:
            ce_only = F.cross_entropy(logits, labels)
            ce_scalar = float(ce_only.detach().item())
            return ce_only, ce_scalar, 0.0, 0.0, 0.0

        true_class_mask = F.one_hot(labels, num_classes=int(logits.shape[1])).to(dtype=torch.bool)
        logit_y = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        logits_negatives = logits.masked_fill(true_class_mask, float("-inf"))
        logsumexp_neg = torch.logsumexp(logits_negatives, dim=1)

        energy_y = -logit_y
        energy_gap = logit_y - logsumexp_neg
        total = torch.mean(energy_y + (float(alpha) * logsumexp_neg))
        return (
            total,
            float(energy_y.detach().mean().item()),
            float(logsumexp_neg.detach().mean().item()),
            float(energy_gap.detach().mean().item()),
            float(total.detach().item()),
        )

    # ------------------------------------------------------------------ #
    # Energy class balance loss
    # ------------------------------------------------------------------ #

    @staticmethod
    def energy_class_balance_loss(
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float]:
        """Encourage non-collapsed class support via KL(pred || target).

        Returns:
            (kl_loss, mean_balance_kl, mean_pred_entropy, min_pred_mass)
        """
        if int(logits.ndim) != 2 or int(logits.shape[0]) == 0:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return zero, 0.0, 0.0, 0.0

        probs = F.softmax(logits, dim=1)
        p_pred = probs.mean(dim=0).clamp_min(1e-12)
        num_classes = max(1, int(logits.shape[1]))
        p_target = torch.full_like(p_pred, 1.0 / float(num_classes))
        kl_loss = torch.sum(p_pred * (torch.log(p_pred) - torch.log(p_target)))
        pred_entropy = -torch.sum(p_pred * torch.log(p_pred))
        return (
            kl_loss,
            float(kl_loss.detach().item()),
            float(pred_entropy.detach().item()),
            float(p_pred.detach().min().item()),
        )

    # ------------------------------------------------------------------ #
    # Energy min winner loss
    # ------------------------------------------------------------------ #

    @staticmethod
    def energy_min_winner_loss(
        logits: torch.Tensor,
        active_class_ids: list[int] | None,
        *,
        min_winners: int,
    ) -> tuple[torch.Tensor, float, float]:
        """Penalize per-batch argmax winner starvation over active classes.

        Returns:
            (loss, mean_winner_deficit, min_winner_count)
        """
        if int(logits.ndim) != 2 or int(logits.shape[0]) == 0:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return zero, 0.0, 0.0

        class_count = int(logits.shape[1])
        hard_pred = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        hard_counts = torch.bincount(hard_pred, minlength=class_count).to(dtype=logits.dtype)
        soft_counts = probs.sum(dim=0)
        counts = soft_counts + (hard_counts - soft_counts).detach()

        if active_class_ids is None:
            active_indices = torch.arange(class_count, device=logits.device, dtype=torch.int64)
        else:
            clean_ids = [cid for cid in active_class_ids if 0 <= int(cid) < class_count]
            if not clean_ids:
                active_indices = torch.arange(class_count, device=logits.device, dtype=torch.int64)
            else:
                active_indices = torch.tensor(clean_ids, device=logits.device, dtype=torch.int64)

        if int(active_indices.numel()) == 0:
            zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
            return zero, 0.0, 0.0

        active_counts = counts.index_select(0, active_indices)
        deficits = F.relu(float(max(0, min_winners)) - active_counts)
        loss = torch.sum(deficits)
        return (
            loss,
            float(deficits.detach().sum().item()),
            float(active_counts.detach().min().item()),
        )

    # ------------------------------------------------------------------ #
    # Representation energy objective (composite)
    # ------------------------------------------------------------------ #

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
        """Compute representation-phase energy objective and diagnostics.

        Combines energy gap, class balance, and winner-minimum losses.
        """
        (
            energy_loss,
            mean_e_y,
            mean_e_others,
            mean_gap,
            mean_energy_total,
        ) = EnergyLossFunctions.class_conditional_energy_gap_loss(
            family_logits_train,
            y_family,
            alpha=energy_multi_negative_alpha,
        )

        if int(epoch) == 0:
            energy_balance_loss = torch.zeros((), dtype=family_logits_train.dtype, device=family_logits_train.device)
            energy_min_winner_loss_val = torch.zeros((), dtype=family_logits_train.dtype, device=family_logits_train.device)
            mean_balance_kl = 0.0
            mean_pred_entropy = 0.0
            min_pred_mass = 0.0
            mean_winner_deficit = 0.0
            min_winner_count = 0.0
            effective_energy_balance_weight = 0.0
            effective_energy_winner_weight = 0.0
        else:
            (
                energy_balance_loss,
                mean_balance_kl,
                mean_pred_entropy,
                min_pred_mass,
            ) = EnergyLossFunctions.energy_class_balance_loss(family_logits_train)
            (
                energy_min_winner_loss_val,
                mean_winner_deficit,
                min_winner_count,
            ) = EnergyLossFunctions.energy_min_winner_loss(
                family_logits_train,
                active_family_class_ids,
                min_winners=energy_winner_min_count,
            )
            effective_energy_balance_weight = float(energy_balance_weight)
            effective_energy_winner_weight = float(energy_winner_weight)

        binary_only_loss = loss_fn._classification_loss(  # type: ignore[operator]
            binary_logits,
            y_binary,
            None,
        )
        loss = (
            (float(loss_fn.lambda_binary) * binary_only_loss)  # type: ignore[arg-type]
            + (float(energy_gap_weight) * energy_loss)
            + (effective_energy_balance_weight * energy_balance_loss)
            + (effective_energy_winner_weight * energy_min_winner_loss_val)
        )

        diagnostics = {
            "mean_e_y": float(mean_e_y),
            "mean_e_others": float(mean_e_others),
            "mean_gap": float(mean_gap),
            "mean_energy_total": float(mean_energy_total),
            "mean_balance_kl": float(mean_balance_kl),
            "mean_pred_entropy": float(mean_pred_entropy),
            "min_pred_mass": float(min_pred_mass),
            "mean_winner_deficit": float(mean_winner_deficit),
            "min_winner_count": float(min_winner_count),
            "effective_energy_balance_weight": float(effective_energy_balance_weight),
            "effective_energy_winner_weight": float(effective_energy_winner_weight),
        }
        return loss, diagnostics


class EnergyStateManager:
    """Instance-based manager for energy win-rate EMA and emergence bias.

    Extracted from ``HelixFullTrainer`` methods:

    * ``_ensure_energy_win_rate_ema``
    * ``_update_energy_win_rate_ema``
    * ``_compute_energy_emergence_bias``

    The EMA tensor is stored as ``self._energy_win_rate_ema`` (managed
    internally) together with the last-computed bias diagnostics.

    Args:
        energy_win_rate_ema_momentum: Momentum factor for win-rate EMA
            updates (typical: 0.9, clamped [0.80, 0.95]).
        energy_emergence_bias_eps: Small epsilon to avoid division by zero
            in log(1 / (ema + eps)).
        energy_emergence_bias_beta: Scaling coefficient for the emergence
            bias magnitude.
        energy_emergence_bias_ratio_max: Maximum ratio of bias magnitude
            relative to logit standard deviation.
    """

    def __init__(
        self,
        *,
        energy_win_rate_ema_momentum: float = 0.9,
        energy_emergence_bias_eps: float = 1e-3,
        energy_emergence_bias_beta: float = 0.5,
        energy_emergence_bias_ratio_max: float = 0.30,
    ) -> None:
        self._energy_win_rate_ema_momentum = float(
            max(0.80, min(float(energy_win_rate_ema_momentum), 0.95))
        )
        self._energy_emergence_bias_eps = float(max(1e-6, float(energy_emergence_bias_eps)))
        self._energy_emergence_bias_beta = float(max(0.0, float(energy_emergence_bias_beta)))
        self._energy_emergence_bias_ratio_max = float(energy_emergence_bias_ratio_max)

        # Internal state
        self._energy_win_rate_ema: torch.Tensor | None = None
        self._energy_bias_last_std: float = 0.0
        self._energy_bias_last_max_abs: float = 0.0
        self._energy_bias_last_logit_std: float = 0.0

    # ------------------------------------------------------------------ #
    # Properties (public read-only view of internal state)
    # ------------------------------------------------------------------ #

    @property
    def energy_win_rate_ema(self) -> torch.Tensor | None:
        """The current EMA win-rate tensor (None if not yet initialized)."""
        return self._energy_win_rate_ema

    @property
    def energy_bias_last_std(self) -> float:
        """Standard deviation of the last computed emergence bias."""
        return self._energy_bias_last_std

    @property
    def energy_bias_last_max_abs(self) -> float:
        """Maximum absolute value of the last computed emergence bias."""
        return self._energy_bias_last_max_abs

    @property
    def energy_bias_last_logit_std(self) -> float:
        """Standard deviation of the logits from the last bias computation."""
        return self._energy_bias_last_logit_std

    # ------------------------------------------------------------------ #
    # EMA ensure / resize
    # ------------------------------------------------------------------ #

    def ensure_energy_win_rate_ema(
        self,
        class_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Lazily initialise or resize the EMA tracker for per-class argmax win rates.

        Returns the (possibly re-allocated or re-casted) EMA tensor.
        """
        if (
            self._energy_win_rate_ema is None
            or int(self._energy_win_rate_ema.numel()) != int(class_count)
        ):
            init = torch.ones((class_count,), device=device, dtype=dtype)
            init = init / float(max(1, class_count))
            self._energy_win_rate_ema = init
        elif (
            self._energy_win_rate_ema.device != device
            or self._energy_win_rate_ema.dtype != dtype
        ):
            self._energy_win_rate_ema = self._energy_win_rate_ema.to(device=device, dtype=dtype)
        return self._energy_win_rate_ema

    # ------------------------------------------------------------------ #
    # EMA update
    # ------------------------------------------------------------------ #

    def update_energy_win_rate_ema(
        self,
        logits: torch.Tensor,
        *,
        active_class_ids: list[int] | None = None,
    ) -> None:
        """Update EMA win-rate estimate from hard argmax class wins in the current batch.

        Only the classes listed in *active_class_ids* participate in the
        EMA; inactive classes are reset to ``eps``.
        """
        if int(logits.ndim) != 2 or int(logits.shape[0]) <= 0:
            return
        class_count = int(logits.shape[1])
        clean_active_ids: list[int] = []
        if active_class_ids is not None:
            clean_active_ids = [
                int(cls) for cls in active_class_ids if 0 <= int(cls) < class_count
            ]
        if not clean_active_ids:
            return

        ema = self.ensure_energy_win_rate_ema(
            class_count,
            device=logits.device,
            dtype=logits.dtype,
        )
        hard_pred = torch.argmax(logits.detach(), dim=1)
        counts = torch.bincount(hard_pred, minlength=class_count).to(dtype=ema.dtype)
        batch_rate = counts / max(1, int(logits.shape[0]))

        active_idx = torch.tensor(clean_active_ids, device=logits.device, dtype=torch.int64)
        active_batch_rate = batch_rate.index_select(0, active_idx)
        active_batch_sum = active_batch_rate.sum().clamp_min(1e-12)
        active_batch_rate = active_batch_rate / active_batch_sum

        active_ema = ema.index_select(0, active_idx)
        momentum = float(self._energy_win_rate_ema_momentum)
        updated_active = (momentum * active_ema) + ((1.0 - momentum) * active_batch_rate)
        updated_active = updated_active / updated_active.sum().clamp_min(1e-12)

        next_ema = ema.clone()
        next_ema.index_copy_(0, active_idx, updated_active)
        inactive_mask = torch.ones(class_count, device=logits.device, dtype=torch.bool)
        inactive_mask.index_fill_(0, active_idx, False)
        if bool(torch.any(inactive_mask)):
            next_ema[inactive_mask] = max(float(self._energy_emergence_bias_eps), 1e-6)

        self._energy_win_rate_ema = next_ema

    # ------------------------------------------------------------------ #
    # Emergence bias computation
    # ------------------------------------------------------------------ #

    def compute_energy_emergence_bias(
        self,
        logits: torch.Tensor,
        *,
        active_class_ids: list[int] | None = None,
        active_family_class_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Compute per-class pre-argmax bias from inverse EMA win rates.

        The bias ``beta * log(1 / (ema + eps))`` is scaled so its standard
        deviation reaches 20% of the logit standard deviation, then
        clamped to ``[−ratio_max * logit_std, +ratio_max * logit_std]``.

        When *active_class_ids* is None and *active_family_class_ids* is
        not None, the latter is used as a fallback (mirroring the trainer's
        ``self.active_family_class_ids`` behaviour).

        Returns a zero-initialised bias tensor (shape ``(class_count,)``)
        with active-class entries filled.
        """
        class_count = int(logits.shape[1])
        eps = float(self._energy_emergence_bias_eps)
        beta = float(self._energy_emergence_bias_beta)

        bias = torch.zeros((class_count,), device=logits.device, dtype=logits.dtype)
        if active_class_ids is not None:
            active_ids = [
                int(cls)
                for cls in sorted(active_class_ids)
                if 0 <= int(cls) < class_count
            ]
        elif active_family_class_ids is not None:
            active_ids = [
                int(cls)
                for cls in sorted(active_family_class_ids)
                if 0 <= int(cls) < class_count
            ]
        else:
            active_ids = []

        if not active_ids:
            self._energy_bias_last_std = 0.0
            self._energy_bias_last_max_abs = 0.0
            self._energy_bias_last_logit_std = float(logits.detach().std().item())
            return bias

        ema = self.ensure_energy_win_rate_ema(
            class_count,
            device=logits.device,
            dtype=logits.dtype,
        )
        active_idx = torch.tensor(active_ids, device=logits.device, dtype=torch.int64)
        active_ema = ema.index_select(0, active_idx).to(dtype=logits.dtype)
        active_bias = beta * torch.log(1.0 / torch.clamp(active_ema + eps, min=eps))

        logit_std = float(logits.detach().std().item())
        if logit_std > eps:
            bias_std = active_bias.detach().std()
            scale = (0.2 * logit_std) / (float(bias_std.item()) + 1e-6)
            active_bias = active_bias * scale
            max_abs = max(eps, self._energy_emergence_bias_ratio_max * logit_std)
            active_bias = torch.clamp(active_bias, min=-max_abs, max=max_abs)

        self._energy_bias_last_std = float(active_bias.detach().std().item())
        self._energy_bias_last_max_abs = float(active_bias.detach().abs().max().item())
        self._energy_bias_last_logit_std = logit_std

        bias.index_copy_(0, active_idx, active_bias)
        return bias

    def reset_energy_win_rate_ema(self) -> None:
        """Force re-initialization of the EMA tracker."""
        self._energy_win_rate_ema = None
