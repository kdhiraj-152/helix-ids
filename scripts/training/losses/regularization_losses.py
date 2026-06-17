"""Regularization / penalty loss functions extracted from HelixFullTrainer.

Pure module-level functions — no classes, no state.
Each function is self-contained and operates on tensor inputs.

Dependency rules:
    losses -> torch (allowed)
    losses -> numpy (allowed)
    losses -> trainer internals (forbidden)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# ── Focal tail stabilization ──────────────────────────────────────────────────


def compute_tail_focal_loss(
    family_logits_train: torch.Tensor,
    y_family: torch.Tensor,
) -> torch.Tensor:
    """Compute focal tail stabilization term for classes 3/4.

    Extracted from ``HelixFullTrainer._compute_tail_focal_loss`` (static method,
    lines 4441-4461 of train_helix_ids_full.py).

    Applies a focal loss (gamma=2.0) only to samples whose ground-truth family
    is in {3, 4}.  Returns a zero scalar if no tail samples are present.

    Args:
        family_logits_train: Raw family logits ``(B, num_classes)``.
        y_family: Ground-truth family labels ``(B,)``.

    Returns:
        Scalar focal-loss tensor (mean over tail samples).
    """
    tail_focal_classes = {3, 4}
    tail_mask = torch.zeros_like(y_family, dtype=torch.bool)
    for cls_id in tail_focal_classes:
        tail_mask = tail_mask | (y_family == int(cls_id))

    if not bool(torch.any(tail_mask)):
        return torch.zeros((), dtype=family_logits_train.dtype, device=family_logits_train.device)

    focal_logits = family_logits_train[tail_mask]
    focal_labels = y_family[tail_mask]
    log_probs = torch.log_softmax(focal_logits, dim=1)
    gathered_logp = log_probs.gather(1, focal_labels.unsqueeze(1)).squeeze(1)
    p_t = gathered_logp.exp()
    gamma_tail = 2.0
    focal_term = ((1.0 - p_t).clamp(min=0.0) ** gamma_tail) * (-gathered_logp)
    return focal_term.mean()


# ── Entropy floor regularizer ─────────────────────────────────────────────────


def apply_entropy_floor_regularizer(
    loss: torch.Tensor,
    *,
    family_logits_train: torch.Tensor,
    active_class_count: int,
) -> torch.Tensor:
    """Apply entropy floor regularisation on family logits.

    Extracted from ``HelixFullTrainer._apply_entropy_floor_regularizer`` (static
    method, lines 4519-4541 of train_helix_ids_full.py).

    Computes a ReLU-based penalty when the mean entropy of the softmax
    distribution falls below ``0.6 * log(active_class_count)``.  The
    hardcoded weight of 0.01 is retained.

    Args:
        loss: Current loss scalar.
        family_logits_train: Raw family logits ``(B, num_classes)``.
        active_class_count: Number of classes currently active in the batch.

    Returns:
        ``loss + 0.01 * entropy_floor_loss``.
    """
    entropy_target = 0.6 * math.log(float(max(1, active_class_count)))
    family_prob_entropy = torch.softmax(family_logits_train, dim=1)
    family_prob_entropy = torch.clamp(family_prob_entropy, min=1e-10, max=1.0)
    mean_entropy = -torch.sum(
        family_prob_entropy * torch.log(family_prob_entropy),
        dim=1,
    ).mean()
    entropy_floor_loss = torch.relu(
        torch.tensor(
            entropy_target,
            dtype=family_logits_train.dtype,
            device=family_logits_train.device,
        )
        - mean_entropy
    )
    return loss + (0.01 * entropy_floor_loss)


# ── Entropy warmup regularizer ────────────────────────────────────────────────


def apply_entropy_warmup(
    loss: torch.Tensor,
    *,
    family_logits_train: torch.Tensor,
    entropy_warmup_weight: float,
    global_step: int,
    entropy_warmup_steps: int,
    epoch: int,
) -> torch.Tensor:
    """Subtract entropy as a warmup regularisation term.

    Extracted from ``HelixFullTrainer._apply_loss_regularizations`` (lines
    3952-3963 of train_helix_ids_full.py).

    Only active when **not** in the first epoch (``epoch > 0``) and
    ``global_step < entropy_warmup_steps``.  In that regime the mean entropy
    of the softmax distribution is subtracted from the loss, weighted by
    ``entropy_warmup_weight``, which encourages higher prediction entropy early
    in training.

    Args:
        loss: Current loss scalar.
        family_logits_train: Raw family logits ``(B, num_classes)``.
        entropy_warmup_weight: Scaling weight for the entropy term.
        global_step: Current training step (0-indexed).
        entropy_warmup_steps: Number of steps over which warmup is active.
        epoch: Current epoch number.

    Returns:
        ``loss - entropy_warmup_weight * family_entropy_warmup`` if condition
        is met, otherwise ``loss`` unchanged.
    """
    first_epoch_only = int(epoch) == 0

    if (
        (not first_epoch_only)
        and entropy_warmup_steps > 0
        and global_step < entropy_warmup_steps
    ):
        family_prob_warmup = torch.softmax(family_logits_train, dim=1)
        safe_prob_warmup = torch.clamp(family_prob_warmup, min=1e-12, max=1.0)
        family_entropy_warmup = -torch.sum(
            family_prob_warmup * torch.log(safe_prob_warmup),
            dim=1,
        ).mean()
        loss = loss - entropy_warmup_weight * family_entropy_warmup

    return loss


# ── KL divergence to uniform ──────────────────────────────────────────────────


def apply_kl_uniform_regularization(
    loss: torch.Tensor,
    *,
    family_logits_train: torch.Tensor,
    kl_uniform_weight: float,
    warmup_kl_uniform_weight: float,
    in_step_warmup: bool,
    epoch: int,
) -> torch.Tensor:
    """Add KL divergence to a uniform distribution as a regulariser.

    Extracted from ``HelixFullTrainer._apply_loss_regularizations`` (lines
    3965-3983 of train_helix_ids_full.py).

    Uses ``warmup_kl_uniform_weight`` during the warmup phase
    (``in_step_warmup=True``) and ``kl_uniform_weight`` otherwise.  The
    penalty is zero in the first epoch (``epoch == 0``).

    Args:
        loss: Current loss scalar.
        family_logits_train: Raw family logits ``(B, num_classes)``.
        kl_uniform_weight: Weight after warmup ends.
        warmup_kl_uniform_weight: Weight during warmup.
        in_step_warmup: ``True`` during the warmup phase.
        epoch: Current epoch number.

    Returns:
        ``loss + kl_weight * kl_to_uniform`` if ``kl_weight > 0``, otherwise
        ``loss`` unchanged.
    """
    if in_step_warmup:
        effective_kl_weight = float(warmup_kl_uniform_weight)
    else:
        effective_kl_weight = float(kl_uniform_weight)

    first_epoch_only = int(epoch) == 0

    if first_epoch_only:
        kl_weight = 0.0
    else:
        kl_weight = effective_kl_weight

    if kl_weight > 0.0:
        family_prob = torch.softmax(family_logits_train, dim=1)
        safe_family_prob = torch.clamp(family_prob, min=1e-12, max=1.0)
        uniform_log_prob = -math.log(float(family_prob.shape[1]))
        kl_to_uniform = torch.sum(
            family_prob * (torch.log(safe_family_prob) - uniform_log_prob),
            dim=1,
        ).mean()
        loss = loss + kl_weight * kl_to_uniform

    return loss


# ── Logit floor penalty ───────────────────────────────────────────────────────


def apply_logit_floor_penalty(
    loss: torch.Tensor,
    *,
    raw_family_logits: torch.Tensor,
    logit_floor_weight: float,
    logit_floor: float,
    in_step_warmup: bool,
) -> torch.Tensor:
    """Penalise logits that fall below a floor threshold.

    Extracted from ``HelixFullTrainer._apply_loss_regularizations`` (lines
    3985-3988 of train_helix_ids_full.py).

    Active only when ``logit_floor_weight > 0`` and the model is **not** in
    warmup (``in_step_warmup=False``).  The penalty is the mean ReLU distance
    from ``logit_floor``.

    Args:
        loss: Current loss scalar.
        raw_family_logits: Raw (unclamped) family logits ``(B, num_classes)``.
        logit_floor_weight: Scaling weight for the penalty.
        logit_floor: Floor threshold value.
        in_step_warmup: ``True`` during the warmup phase.

    Returns:
        ``loss + logit_floor_weight * logit_floor_penalty`` if active,
        otherwise ``loss`` unchanged.
    """
    if logit_floor_weight > 0.0 and not in_step_warmup:
        logit_floor_penalty = torch.relu(logit_floor - raw_family_logits).mean()
        loss = loss + logit_floor_weight * logit_floor_penalty

    return loss


# ── Tail class cross-entropy ──────────────────────────────────────────────────


def apply_tail_ce_regularization(
    loss: torch.Tensor,
    *,
    family_logits_train: torch.Tensor,
    y_family: torch.Tensor,
    tail_ce_weight: float,
    tail_class_mask: torch.Tensor | None,
    loss_fn: object,
) -> torch.Tensor:
    """Add cross-entropy only on tail-class samples.

    Extracted from ``HelixFullTrainer._apply_loss_regularizations`` (lines
    3991-4004 of train_helix_ids_full.py).

    When ``tail_ce_weight > 0`` and a valid ``tail_class_mask`` is provided,
    selects samples whose ground-truth class falls into a tail class and
    computes an additional cross-entropy term on those samples.  The
    label-smoothing value, if any, is read from ``loss_fn.label_smoothing``.

    Args:
        loss: Current loss scalar.
        family_logits_train: Raw family logits ``(B, num_classes)``.
        y_family: Ground-truth family labels ``(B,)``.
        tail_ce_weight: Scaling weight for the tail CE term.
        tail_class_mask: Boolean mask ``(num_classes,)`` indicating tail classes,
            or ``None``.
        loss_fn: Loss function object (may carry a ``label_smoothing``
            attribute).

    Returns:
        ``loss + tail_ce_weight * tail_ce`` if active, otherwise ``loss``
        unchanged.
    """
    if (
        tail_ce_weight > 0.0
        and tail_class_mask is not None
        and int(tail_class_mask.numel()) == int(family_logits_train.shape[1])
    ):
        tail_sample_mask = tail_class_mask[y_family]
        if bool(torch.any(tail_sample_mask)):
            tail_ce = F.cross_entropy(
                family_logits_train[tail_sample_mask],
                y_family[tail_sample_mask],
                reduction="mean",
                label_smoothing=float(getattr(loss_fn, "label_smoothing", 0.0)),
            )
            loss = loss + tail_ce_weight * tail_ce

    return loss
