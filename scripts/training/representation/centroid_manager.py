"""Centroid manager: centroid state and lifecycle extracted from HelixFullTrainer.

Phase 13A-3 extraction from HelixFullTrainer.

Owns:
    _centroid_ema_state    — per-class EMA of centroid positions (CPU tensors)
    _epoch_frozen_centroids — snapshots taken at epoch boundaries

Methods:
    update_running_rep_centroids    — accumulate per-class EMA from batch centroids
    freeze_epoch_centroid_snapshot  — freeze current EMA state as epoch reference
    update_centroids_from_epoch_buffer — process accumulated epoch buffers
    stabilize_centroids             — apply EMA smoothing for stable output

Dependency rules:
    centroid_manager -> torch (allowed)
    centroid_manager -> numpy (allowed)
    centroid_manager -> trainer internals (forbidden)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class CentroidManager:
    """Manages per-class centroid state and lifecycle.

    Keeps two dictionaries:
        _centroid_ema_state    (int -> Tensor on CPU, float32)
        _epoch_frozen_centroids (int -> Tensor on CPU, float32)
    """

    def __init__(self, centroid_ema_momentum: float = 0.9) -> None:
        self._centroid_ema_state: dict[int, torch.Tensor] = {}
        self._epoch_frozen_centroids: dict[int, torch.Tensor] = {}
        self._momentum = float(np.clip(centroid_ema_momentum, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def centroid_ema_state(self) -> dict[int, torch.Tensor]:
        """Read-only access to current EMA state."""
        return self._centroid_ema_state

    @property
    def epoch_frozen_centroids(self) -> dict[int, torch.Tensor]:
        """Read-only access to frozen epoch centroids."""
        return self._epoch_frozen_centroids

    # ------------------------------------------------------------------ #
    # Running centroid update (per-batch EMA)
    # ------------------------------------------------------------------ #

    def update_running_rep_centroids(
        self,
        batch_centroids: torch.Tensor,
        class_ids: list[int],
    ) -> None:
        """Update running per-class centroids (EMA) during representation phase."""
        if int(batch_centroids.shape[0]) == 0:
            return

        m = self._momentum
        for idx, class_id in enumerate(class_ids):
            current = F.normalize(
                batch_centroids[idx].detach().to(device="cpu", dtype=torch.float32),
                p=2,
                dim=0,
            )
            prev = self._centroid_ema_state.get(int(class_id))
            ema = current if prev is None else ((m * prev) + ((1.0 - m) * current))
            self._centroid_ema_state[int(class_id)] = F.normalize(ema, p=2, dim=0).detach().clone()

    # ------------------------------------------------------------------ #
    # Epoch centroid snapshot
    # ------------------------------------------------------------------ #

    def freeze_epoch_centroid_snapshot(self) -> None:
        """Freeze centroid reference frame for the current epoch."""
        self._epoch_frozen_centroids = {
            int(class_id): centroid.detach().clone()
            for class_id, centroid in self._centroid_ema_state.items()
        }

    # ------------------------------------------------------------------ #
    # Update centroids from epoch buffer
    # ------------------------------------------------------------------ #

    def update_centroids_from_epoch_buffer(
        self,
        rep_epoch_feature_chunks: list[torch.Tensor],
        rep_epoch_label_chunks: list[torch.Tensor],
        cluster_analyzer: Any,
    ) -> None:
        """Update running centroid EMA once per epoch from accumulated representation buffers.

        Clears the provided chunk lists after processing.
        """
        if not rep_epoch_feature_chunks or not rep_epoch_label_chunks:
            return

        features = torch.cat(rep_epoch_feature_chunks, dim=0).to(dtype=torch.float32)
        labels = torch.cat(rep_epoch_label_chunks, dim=0).to(dtype=torch.int64)
        centroids, class_ids = cluster_analyzer.compute_class_centroids(features, labels)
        if int(centroids.shape[0]) == 0:
            rep_epoch_feature_chunks.clear()
            rep_epoch_label_chunks.clear()
            return

        self.update_running_rep_centroids(centroids, class_ids)
        rep_epoch_feature_chunks.clear()
        rep_epoch_label_chunks.clear()

    # ------------------------------------------------------------------ #
    # Stabilize centroids (EMA smoothing)
    # ------------------------------------------------------------------ #

    def stabilize_centroids(
        self,
        batch_centroids: torch.Tensor,
        class_ids: list[int],
    ) -> torch.Tensor:
        """Apply centroid EMA smoothing for stable margin/logging targets."""
        if int(batch_centroids.shape[0]) == 0:
            return batch_centroids.detach()

        stabilized: list[torch.Tensor] = []
        m = self._momentum
        for idx, class_id in enumerate(class_ids):
            current = batch_centroids[idx].detach().to(device="cpu", dtype=torch.float32)
            prev = self._centroid_ema_state.get(int(class_id))
            if prev is None:
                ema = current
            else:
                ema = (m * prev) + ((1.0 - m) * current)
            self._centroid_ema_state[int(class_id)] = ema.detach().clone()
            stabilized.append(ema.to(device=batch_centroids.device, dtype=batch_centroids.dtype))

        return torch.stack(stabilized, dim=0)

    def reset_epoch_frozen_centroids(self) -> None:
        """Clear the frozen epoch centroids dict."""
        self._epoch_frozen_centroids.clear()
