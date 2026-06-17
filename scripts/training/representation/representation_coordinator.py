"""Representation coordinator: orchestration helpers extracted from HelixFullTrainer.

Phase 13A-3 extraction from HelixFullTrainer.

Provides:
    Rebalance batch helpers
    Chunk storage helpers
    Curriculum checking utilities

Dependency rules:
    coordinator -> torch (allowed)
    coordinator -> numpy (allowed)
    coordinator -> trainer internals (forbidden)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from scripts.training.data.dataset_builder import build_class_index


class RepresentationCoordinator:
    """Coordination helpers for representation-phase logic.

    Stateless utility methods — all state (buffers, flags) lives in
    the trainer and is passed explicitly.
    """

    # ------------------------------------------------------------------ #
    # Batch rebalancing
    # ------------------------------------------------------------------ #

    @staticmethod
    def rebalance_representation_batch(
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        target_per_class: int,
        run_seed: int,
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebalance representation-phase batches to avoid majority-class manifold domination."""
        if int(features.shape[0]) <= 1:
            return features, labels

        labels_np = np.asarray(
            labels.detach().to(device="cpu", dtype=torch.int64).numpy(), dtype=np.int64
        )
        class_index = build_class_index(labels_np)
        if len(class_index) <= 1:
            return features, labels

        counts = [int(idxs.shape[0]) for idxs in class_index.values()]
        target = max(1, min(int(target_per_class), max(counts)))
        rng = np.random.default_rng(run_seed + int(global_step))
        selected: list[int] = []

        for class_id in sorted(class_index.keys()):
            idxs = class_index[class_id]
            sampled = rng.choice(idxs, size=target, replace=bool(idxs.shape[0] < target))
            selected.extend(int(i) for i in sampled.tolist())

        if not selected:
            return features, labels

        rng.shuffle(selected)
        index_tensor = torch.tensor(selected, device=features.device, dtype=torch.long)
        return features.index_select(0, index_tensor), labels.index_select(0, index_tensor)

    # ------------------------------------------------------------------ #
    # Representation chunk storage
    # ------------------------------------------------------------------ #

    @staticmethod
    def store_representation_chunks(
        rep_phase_feature_chunks: list[torch.Tensor],
        rep_phase_label_chunks: list[torch.Tensor],
        *,
        backbone_features: torch.Tensor,
        y_family: torch.Tensor,
        cluster_analyzer: Any,
    ) -> None:
        """Store detached representation features for representation-phase diagnostics.

        Appends normalized and detached chunks to the provided lists.
        """
        rep_features = cluster_analyzer.prepare_representation_features(backbone_features)
        rep_phase_feature_chunks.append(rep_features.detach().to(device="cpu"))
        rep_phase_label_chunks.append(y_family.detach().to(device="cpu", dtype=torch.int64))

    # ------------------------------------------------------------------ #
    # Epoch buffer chunk storage
    # ------------------------------------------------------------------ #

    @staticmethod
    def store_epoch_chunks(
        rep_epoch_feature_chunks: list[torch.Tensor],
        rep_epoch_label_chunks: list[torch.Tensor],
        *,
        backbone_features: torch.Tensor,
        y_family: torch.Tensor,
        cluster_analyzer: Any,
    ) -> None:
        """Store detached epoch-buffer chunks for centroid computation.

        Appends detached chunks to the provided lists.
        """
        rep_features = cluster_analyzer.prepare_representation_features(backbone_features)
        rep_epoch_feature_chunks.append(rep_features.detach().to(device="cpu"))
        rep_epoch_label_chunks.append(y_family.detach().to(device="cpu", dtype=torch.int64))

    # ------------------------------------------------------------------ #
    # Buffer concat helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def concat_chunks(
        chunks: list[torch.Tensor],
        *,
        default: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate chunk tensors or return default if empty."""
        if not chunks:
            if default is not None:
                return default
            return torch.zeros((0, 0), dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def concat_label_chunks(
        chunks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Concatenate label chunk tensors or return empty int64 tensor."""
        if not chunks:
            return torch.zeros((0,), dtype=torch.int64)
        return torch.cat(chunks, dim=0)
