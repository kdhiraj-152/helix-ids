"""Contrastive and repulsion loss functions extracted from HelixFullTrainer.

These are pure loss functions — they accept explicit tensor parameters
and configuration values; no trainer state required.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ #
# Supervised contrastive loss
# ------------------------------------------------------------------ #


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    anchor_weights: torch.Tensor | None = None,
    negative_weight: float = 1.0,
    min_negatives: int = 1,
) -> torch.Tensor:
    """Compute supervised contrastive loss over a batch of backbone features."""
    if int(features.shape[0]) <= 1:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    feat = F.normalize(features, p=2, dim=1)
    logits = torch.matmul(feat, feat.T) / max(1e-6, float(temperature))

    # Mask self-comparisons with a large finite negative value to avoid NaNs.
    self_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)

    labels_col = labels.view(-1, 1)
    positive_mask = (labels_col == labels_col.T) & (~self_mask)
    negative_mask = (labels_col != labels_col.T) & (~self_mask)

    positive_count = positive_mask.sum(dim=1)
    negative_count = negative_mask.sum(dim=1)
    valid_anchor = (positive_count > 0) & (negative_count >= int(max(1, min_negatives)))
    if not bool(valid_anchor.any()):
        return torch.zeros((), dtype=features.dtype, device=features.device)

    masked_logits = logits.masked_fill(self_mask, -1e9)
    row_max = masked_logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(masked_logits - row_max)
    exp_logits = exp_logits.masked_fill(self_mask, 0.0)

    neg_multiplier = torch.where(
        negative_mask,
        torch.full_like(exp_logits, float(max(1.0, negative_weight))),
        torch.ones_like(exp_logits),
    )
    weighted_exp = exp_logits * neg_multiplier

    denom = weighted_exp.sum(dim=1).clamp_min(1e-12)
    pos_sum = (exp_logits * positive_mask.to(exp_logits.dtype)).sum(dim=1).clamp_min(1e-12)
    loss_per_anchor = -torch.log(pos_sum / denom)
    valid_loss = loss_per_anchor[valid_anchor]
    if anchor_weights is None:
        return valid_loss.mean()

    valid_weights = anchor_weights.to(device=valid_loss.device, dtype=valid_loss.dtype)[valid_anchor]
    valid_weights = valid_weights / valid_weights.sum().clamp_min(1e-12)
    return torch.sum(valid_loss * valid_weights)


# ------------------------------------------------------------------ #
# SupCon anchor weights
# ------------------------------------------------------------------ #


def supcon_anchor_weights(labels: torch.Tensor) -> torch.Tensor:
    """Build class-balanced anchor weights for SupCon: 1 / log(1 + class_freq)."""
    labels_int = labels.to(dtype=torch.int64)
    max_label = int(torch.max(labels_int).item()) if int(labels_int.numel()) > 0 else 0
    class_counts = torch.bincount(labels_int, minlength=max_label + 1).to(dtype=torch.float32)
    class_counts = torch.clamp(class_counts, min=1.0)
    weights = 1.0 / torch.log1p(class_counts[labels_int])
    return weights / weights.mean().clamp_min(1e-12)


# ------------------------------------------------------------------ #
# Pairwise margin repulsion loss
# ------------------------------------------------------------------ #


def pairwise_margin_repulsion_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
    hard_negative_weight: float = 1.0,
    top_k: int = 3,
) -> torch.Tensor:
    """Penalize hardest/top-k nearest negative pairs with class-balanced weighting."""
    if int(features.shape[0]) <= 1:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    pair_dists = torch.cdist(features, features, p=2)
    labels_int = labels.to(dtype=torch.int64)
    labels_col = labels_int.view(-1, 1)
    neg_mask = labels_col != labels_col.T
    upper_mask = torch.triu(torch.ones_like(pair_dists, dtype=torch.bool), diagonal=1)
    valid_neg = neg_mask & upper_mask

    neg_indices = torch.where(valid_neg)
    neg_dists = pair_dists[neg_indices]
    if int(neg_dists.numel()) == 0:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    max_label = int(torch.max(labels_int).item()) if int(labels_int.numel()) > 0 else 0
    class_counts = torch.bincount(labels_int, minlength=max_label + 1).to(
        device=features.device,
        dtype=features.dtype,
    )
    class_counts = torch.clamp(class_counts, min=1.0)

    neg_label_i = labels_int[neg_indices[0]]
    neg_label_j = labels_int[neg_indices[1]]
    count_i = class_counts[neg_label_i]
    count_j = class_counts[neg_label_j]
    balance_weight = torch.sqrt(1.0 / torch.clamp(count_i * count_j, min=1.0))
    balance_weight = balance_weight / balance_weight.mean().clamp_min(1e-12)

    hard_negative_multiplier = torch.where(
        neg_dists < float(margin),
        torch.full_like(neg_dists, float(max(1.0, hard_negative_weight))),
        torch.ones_like(neg_dists),
    )
    margin_violation = F.relu(float(margin) - neg_dists)
    weighted_violation = margin_violation * balance_weight * hard_negative_multiplier

    k = int(max(1, top_k))
    k = min(k, int(weighted_violation.numel()))
    topk_values, _ = torch.topk(weighted_violation, k=k, largest=True, sorted=False)
    return topk_values.sum()


# ------------------------------------------------------------------ #
# Centroid separation barrier loss
# ------------------------------------------------------------------ #


def centroid_separation_barrier_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    min_distance: float,
) -> torch.Tensor:
    """Penalize class-centroid pairs that sit below a minimum separation distance."""
    unique_labels = torch.unique(labels, dim=0)
    if int(unique_labels.numel()) <= 1:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    centroids: list[torch.Tensor] = []
    for class_id in unique_labels.tolist():
        class_features = features[labels == int(class_id)]
        if int(class_features.shape[0]) == 0:
            continue
        centroids.append(class_features.mean(dim=0))

    if len(centroids) <= 1:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    center_tensor = F.normalize(torch.stack(centroids, dim=0), p=2, dim=1)
    center_dists = torch.cdist(center_tensor, center_tensor, p=2)
    upper_mask = torch.triu(torch.ones_like(center_dists, dtype=torch.bool), diagonal=1)
    pair_dists = center_dists[upper_mask]
    if int(pair_dists.numel()) == 0:
        return torch.zeros((), dtype=features.dtype, device=features.device)
    return (F.relu(float(min_distance) - pair_dists) * pair_dists).sum()


# ------------------------------------------------------------------ #
# Centroid repulsion loss
# ------------------------------------------------------------------ #


def centroid_repulsion_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Apply centroid-level margin force with constant gradient inside margin."""
    unique_labels = torch.unique(labels, dim=0)
    if int(unique_labels.numel()) <= 1:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    centroids: list[torch.Tensor] = []
    for class_id in unique_labels.tolist():
        class_features = features[labels == int(class_id)]
        if int(class_features.shape[0]) == 0:
            continue
        centroids.append(class_features.mean(dim=0))

    if len(centroids) <= 1:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    center_tensor = F.normalize(torch.stack(centroids, dim=0), p=2, dim=1)
    center_dists = torch.cdist(center_tensor, center_tensor, p=2)
    upper_mask = torch.triu(torch.ones_like(center_dists, dtype=torch.bool), diagonal=1)
    pair_dists = center_dists[upper_mask]
    if int(pair_dists.numel()) == 0:
        return torch.zeros((), dtype=features.dtype, device=features.device)
    return F.relu(float(margin) - pair_dists).sum()


# ------------------------------------------------------------------ #
# Intra-class variance clamp loss
# ------------------------------------------------------------------ #


def intra_class_variance_clamp_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    var_lower_bound: float,
    var_upper_bound: float,
) -> torch.Tensor:
    """Penalize exploding and collapsing per-class embedding variance."""
    unique_labels = torch.unique(labels, dim=0)
    if int(unique_labels.numel()) <= 1:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    penalties: list[torch.Tensor] = []
    for class_id in unique_labels.tolist():
        class_features = features[labels == int(class_id)]
        if int(class_features.shape[0]) <= 1:
            continue
        class_var = torch.var(class_features, dim=0, unbiased=False).mean()
        penalties.append(F.relu(class_var - var_upper_bound) + F.relu(var_lower_bound - class_var))

    if not penalties:
        return torch.zeros((), dtype=features.dtype, device=features.device)
    return torch.stack(penalties).mean()


# ------------------------------------------------------------------ #
# Batch class centroids (for loss computation, differentiable)
# ------------------------------------------------------------------ #


def compute_batch_class_centroids_for_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Compute differentiable per-class centroids from a batch."""
    if int(features.shape[0]) == 0 or int(labels.shape[0]) == 0:
        return torch.zeros((0, 0), dtype=features.dtype, device=features.device), []

    class_ids = sorted(int(v) for v in torch.unique(labels, dim=0).tolist())
    centroids: list[torch.Tensor] = []
    for class_id in class_ids:
        class_features = features[labels == int(class_id)]
        if int(class_features.shape[0]) == 0:
            continue
        centroids.append(class_features.mean(dim=0))

    if not centroids:
        return torch.zeros((0, 0), dtype=features.dtype, device=features.device), []
    return torch.stack(centroids, dim=0), class_ids


# ------------------------------------------------------------------ #
# Global centroid guided losses
# ------------------------------------------------------------------ #


def global_centroid_guided_losses(
    batch_centroids: torch.Tensor,
    class_ids: list[int],
    epoch_frozen_centroids: dict[int, torch.Tensor],
    *,
    rep_centroid_repulsion_margin: float,
    rep_centroid_barrier_min_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute centroid forces against running global centroids for topology-level separation.

    Modifies *epoch_frozen_centroids* in-place if empty (lazy init).
    """
    if int(batch_centroids.shape[0]) == 0 or len(class_ids) == 0:
        zero = torch.zeros((), dtype=batch_centroids.dtype, device=batch_centroids.device)
        return zero, zero, 0.0

    if not epoch_frozen_centroids:
        for idx, class_id in enumerate(class_ids):
            epoch_frozen_centroids[int(class_id)] = F.normalize(
                batch_centroids[idx].detach().to(device="cpu", dtype=torch.float32),
                p=2,
                dim=0,
            )

    global_ids = sorted(int(k) for k in epoch_frozen_centroids.keys())
    if len(global_ids) <= 1:
        zero = torch.zeros((), dtype=batch_centroids.dtype, device=batch_centroids.device)
        return zero, zero, 0.0

    global_centroids = torch.stack(
        [epoch_frozen_centroids[class_id] for class_id in global_ids],
        dim=0,
    ).to(device=batch_centroids.device, dtype=batch_centroids.dtype)
    global_centroids = F.normalize(global_centroids, p=2, dim=1)

    global_pair_dists = torch.cdist(global_centroids, global_centroids, p=2)
    global_upper_mask = torch.triu(
        torch.ones_like(global_pair_dists, dtype=torch.bool),
        diagonal=1,
    )
    global_pairs = global_pair_dists[global_upper_mask]
    global_min_inter = float(torch.min(global_pairs).item()) if int(global_pairs.numel()) > 0 else 0.0

    batch_centroids_norm = F.normalize(batch_centroids, p=2, dim=1)
    repulsion_terms: list[torch.Tensor] = []
    barrier_terms: list[torch.Tensor] = []
    margin = float(rep_centroid_repulsion_margin)
    barrier_min = float(rep_centroid_barrier_min_distance)

    for idx, class_id in enumerate(class_ids):
        other_indices = [j for j, gid in enumerate(global_ids) if int(gid) != int(class_id)]
        if not other_indices:
            continue
        others = global_centroids[other_indices]
        dists = torch.cdist(batch_centroids_norm[idx: idx + 1], others, p=2).squeeze(0)
        nearest = torch.min(dists)
        repulsion_terms.append(F.relu(margin - nearest))
        barrier_terms.append(F.relu(barrier_min - nearest) * nearest)

    if not repulsion_terms:
        zero = torch.zeros((), dtype=batch_centroids.dtype, device=batch_centroids.device)
        return zero, zero, global_min_inter

    repulsion_loss = torch.stack(repulsion_terms).sum()
    barrier_loss = torch.stack(barrier_terms).sum()
    return repulsion_loss, barrier_loss, global_min_inter


# ------------------------------------------------------------------ #
# Critical pair centroid push loss
# ------------------------------------------------------------------ #


def critical_pair_centroid_push_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    min_distance: float,
    critical_collision_pairs: set[tuple[int, int]],
) -> torch.Tensor:
    """Apply direct centroid push for known critically colliding class pairs."""
    if not critical_collision_pairs:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    penalties: list[torch.Tensor] = []
    for class_a, class_b in critical_collision_pairs:
        mask_a = labels == int(class_a)
        mask_b = labels == int(class_b)
        if int(mask_a.sum().item()) == 0 or int(mask_b.sum().item()) == 0:
            continue
        centroid_a = F.normalize(features[mask_a].mean(dim=0, keepdim=True), p=2, dim=1)
        centroid_b = F.normalize(features[mask_b].mean(dim=0, keepdim=True), p=2, dim=1)
        dist = torch.linalg.vector_norm(centroid_a - centroid_b, ord=2, dim=1).squeeze(0)
        penalties.append(F.relu(float(min_distance) - dist))

    if not penalties:
        return torch.zeros((), dtype=features.dtype, device=features.device)
    return torch.stack(penalties).sum()
