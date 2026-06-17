"""RepresentationDiagnostics — representation metric computation, snapshot generation.

Pure computation — no trainer state imported. Model and device injected via
constructor for embedding collection.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from helix_ids.data.geometric_representation_fixes import GeometricRepresentationFixer
from scripts.training.diagnostics.cluster_analyzer import ClusterAnalyzer
from scripts.training.diagnostics.geometry_analyzer import GeometryAnalyzer


class RepresentationDiagnostics:
    """Representation metric computation and embedding diagnostics.

    Owns:
      - full representation diagnostics pipeline (compute_representation_diagnostics)
      - embedding collection from loaders (collect_normalized_embeddings)
      - snapshot ID generation (build_representation_snapshot_id)
      - center pair diagnostics (compute_center_pair_diagnostics)
      - nearest-center accuracy (nearest_center_accuracy)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        logger: Any,
        *,
        representation_only_steps: int,
        head_only_steps: int,
        sampler_mode: str,
    ) -> None:
        self._model = model
        self._device = device
        self._logger = logger
        self._representation_only_steps = int(representation_only_steps)
        self._head_only_steps = int(head_only_steps)
        self._sampler_mode = str(sampler_mode)

    # ------------------------------------------------------------------
    # Embedding collection
    # ------------------------------------------------------------------

    def collect_normalized_embeddings(
        self,
        loader: DataLoader,
        *,
        max_batches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect L2-normalized backbone embeddings and family labels from a loader.

        Returns:
            (features_tensor, labels_tensor) on CPU.
        """
        was_training = self._model.training
        self._model.eval()

        feature_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []

        with torch.no_grad():
            for batch_idx, (x, _, y_family) in enumerate(loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                x = x.to(self._device, non_blocking=True)
                _, _, features = self._model(x, return_features=True)
                feature_chunks.append(
                    ClusterAnalyzer.prepare_representation_features(features)
                    .detach()
                    .to(device="cpu")
                )
                label_chunks.append(y_family.detach().to(device="cpu", dtype=torch.int64))

        if was_training:
            self._model.train()

        if not feature_chunks:
            return (
                torch.zeros((0, 0), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)

    # ------------------------------------------------------------------
    # Center pair diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_center_pair_diagnostics(
        dist_mat: torch.Tensor,
        available_class_ids: list[int],
    ) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
        """Compute pairwise center diagnostics and percentile-based threshold.

        Returns:
            (center_pairs, collision_threshold_p05, nearest_cluster_pairs_top5)
        """
        center_pairs: list[dict[str, Any]] = []
        for i, cls_i in enumerate(available_class_ids):
            for j in range(i + 1, len(available_class_ids)):
                center_pairs.append(
                    {
                        "class_i": int(cls_i),
                        "class_j": int(available_class_ids[j]),
                        "distance": float(dist_mat[i, j].item()),
                    }
                )

        inter_pair_distances = [float(item["distance"]) for item in center_pairs]
        collision_threshold_p05 = (
            float(
                np.percentile(
                    np.asarray(inter_pair_distances, dtype=np.float32), 5.0
                )
            )
            if inter_pair_distances
            else 0.0
        )
        nearest_cluster_pairs_top5 = sorted(
            center_pairs,
            key=lambda item: float(item["distance"]),
        )[:5]
        return center_pairs, collision_threshold_p05, nearest_cluster_pairs_top5

    # ------------------------------------------------------------------
    # Nearest-center accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def nearest_center_accuracy(
        features: torch.Tensor,
        labels: torch.Tensor,
        centers: dict[int, torch.Tensor],
        class_ids: list[int],
    ) -> float:
        """Compute nearest-center classification accuracy.

        Returns:
            Fraction of examples whose nearest center matches true label.
        """
        if int(features.shape[0]) == 0 or not class_ids:
            return 0.0

        center_tensor = torch.stack([centers[c] for c in class_ids], dim=0)
        dists = torch.cdist(features, center_tensor, p=2)
        pred_idx = torch.argmin(dists, dim=1)
        pred_labels = torch.tensor(
            [class_ids[int(i)] for i in pred_idx.tolist()], dtype=torch.int64
        )
        return float(
            (pred_labels == labels.to(dtype=torch.int64)).float().mean().item()
        )

    # ------------------------------------------------------------------
    # Full representation diagnostics pipeline
    # ------------------------------------------------------------------

    def compute_representation_diagnostics(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        *,
        class_ids: list[int],
        geometry_analyzer: GeometryAnalyzer,
        run_seed: int,
    ) -> dict[str, Any]:
        """Compute center distances, nearest-center accuracy, and separability.

        Includes geometric representation Fixes 4-6:
        - Fix 4: Secondary collision detection
        - Fix 5: Nearest-center confusion matrix
        - Fix 6: Embedding capacity assessment
        """
        empty_result = {
            "available_class_ids": [],
            "center_distance_matrix": {},
            "nearest_center_accuracy_train": 0.0,
            "nearest_center_accuracy_val": 0.0,
            "nearest_center_acc_val": 0.0,
            "intra_class_distance_mean": 0.0,
            "inter_center_distance_mean": 0.0,
            "intra_inter_ratio": 0.0,
            "min_inter_center_distance": 0.0,
            "cluster_size_counts": [],
            "cluster_sizes": [],
            "collision_threshold_p05": 0.0,
            "nearest_cluster_pairs_top5": [],
            "density_variance": 0.0,
            "density_feature_dead": True,
            "collision_pairs": [],
            "secondary_collision_pairs": [],
            "nearest_center_confusion_matrix": {},
            "embedding_capacity_assessment": {},
        }

        if int(train_features.shape[0]) == 0 or not class_ids:
            return empty_result

        centers, available_class_ids = ClusterAnalyzer.build_class_centers(
            train_features, train_labels, class_ids
        )
        if not available_class_ids:
            return empty_result

        centers_tensor = torch.stack([centers[c] for c in available_class_ids], dim=0)
        dist_mat = torch.cdist(centers_tensor, centers_tensor, p=2)
        _, collision_threshold_p05, nearest_cluster_pairs_top5 = (
            self.compute_center_pair_diagnostics(dist_mat, available_class_ids)
        )

        center_distance_matrix: dict[str, dict[str, float]] = {}
        for i, cls_i in enumerate(available_class_ids):
            row: dict[str, float] = {}
            for j, cls_j in enumerate(available_class_ids):
                row[str(cls_j)] = float(dist_mat[i, j].item())
            center_distance_matrix[str(cls_i)] = row

        inter_distances, collision_pairs, intra_distances = (
            GeometryAnalyzer.compute_inter_and_intra_distances(
                train_features,
                train_labels,
                centers,
                dist_mat,
                available_class_ids,
                collision_threshold_p05,
            )
        )

        intra_mean = float(np.mean(intra_distances)) if intra_distances else 0.0
        inter_mean = float(np.mean(inter_distances)) if inter_distances else 0.0
        intra_inter_ratio = (
            float(intra_mean / max(1e-8, inter_mean)) if inter_mean > 0.0 else 0.0
        )
        min_inter = float(min(inter_distances)) if inter_distances else 0.0

        val_mask = torch.zeros_like(val_labels, dtype=torch.bool)
        for cls in available_class_ids:
            val_mask = val_mask | (val_labels == int(cls))
        val_features_eval = (
            val_features[val_mask] if int(val_features.shape[0]) > 0 else val_features
        )
        val_labels_eval = (
            val_labels[val_mask] if int(val_labels.shape[0]) > 0 else val_labels
        )

        # Fix 4: Secondary collision detection
        fixer = GeometricRepresentationFixer()
        centers_np = {cid: centers[cid].cpu().numpy() for cid in available_class_ids}
        secondary_collisions, _ = fixer.detect_secondary_collisions(
            centers_np,
            collision_threshold=collision_threshold_p05,
        )

        # Fix 5: Nearest-center confusion matrix
        confusion_matrix = fixer.build_nearest_center_confusion_matrix(
            train_features.cpu().numpy(),
            train_labels.cpu().numpy(),
            centers_np,
            top_k=3,
        )

        # Fix 6: Embedding capacity assessment
        capacity_assessment = fixer.assess_embedding_capacity(
            intra_inter_ratio,
            embedding_dim=int(train_features.shape[1]),
            dropout_rate=0.2,
            target_ratio=0.8,
        )

        density_diagnostics = geometry_analyzer.estimate_local_density_diagnostics(
            train_features, run_seed=run_seed
        )
        cluster_sizes = [
            int((train_labels == int(cls)).sum().item()) for cls in available_class_ids
        ]
        nearest_center_acc_train = self.nearest_center_accuracy(
            train_features,
            train_labels,
            centers,
            available_class_ids,
        )
        nearest_center_acc_val = self.nearest_center_accuracy(
            val_features_eval,
            val_labels_eval,
            centers,
            available_class_ids,
        )

        return {
            "available_class_ids": available_class_ids,
            "center_distance_matrix": center_distance_matrix,
            "nearest_center_accuracy_train": nearest_center_acc_train,
            "nearest_center_accuracy_val": nearest_center_acc_val,
            "nearest_center_acc_val": nearest_center_acc_val,
            "intra_class_distance_mean": intra_mean,
            "inter_center_distance_mean": inter_mean,
            "intra_inter_ratio": intra_inter_ratio,
            "min_inter_center_distance": min_inter,
            "cluster_size_counts": cluster_sizes,
            "cluster_sizes": cluster_sizes,
            "collision_threshold_p05": collision_threshold_p05,
            "nearest_cluster_pairs_top5": nearest_cluster_pairs_top5,
            "density_variance": float(
                density_diagnostics.get("density_variance", 0.0)
            ),
            "density_feature_dead": bool(
                density_diagnostics.get("density_feature_dead", False)
            ),
            "collision_pairs": collision_pairs,
            "secondary_collision_pairs": secondary_collisions,
            "nearest_center_confusion_matrix": confusion_matrix,
            "embedding_capacity_assessment": capacity_assessment,
        }

    # ------------------------------------------------------------------
    # Runner — orchestrates embedding collection + diagnostics
    # ------------------------------------------------------------------

    def run_representation_diagnostics(
        self,
        *,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        label_space: str,
        active_family_class_ids: set[int],
        val_loaders: dict[str, DataLoader],
        geometry_analyzer: GeometryAnalyzer,
        run_seed: int,
    ) -> dict[str, Any]:
        """Run mandatory embedding diagnostics.

        Collects validation embeddings, computes diagnostics, and logs results.
        Does NOT mutate trainer state — returns the diagnostics dict.

        Returns:
            diagnostics dict.
        """
        if not active_family_class_ids:
            return {}

        class_ids = sorted(int(c) for c in active_family_class_ids)

        val_loader = next(iter(val_loaders.values()), None)
        if val_loader is None:
            val_features = torch.zeros(
                (0, train_features.shape[1]), dtype=torch.float32
            )
            val_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            val_features, val_labels = self.collect_normalized_embeddings(val_loader)

        diagnostics = self.compute_representation_diagnostics(
            train_features,
            train_labels,
            val_features,
            val_labels,
            class_ids=class_ids,
            geometry_analyzer=geometry_analyzer,
            run_seed=run_seed,
        )

        self._logger.info(
            "RepDiag[%s] nearest_center_acc(train)=%.4f nearest_center_acc(val)=%.4f "
            "intra=%.4f inter=%.4f ratio=%.4f min_inter=%.4f collisions=%s",
            str(label_space),
            float(diagnostics.get("nearest_center_accuracy_train", 0.0)),
            float(diagnostics.get("nearest_center_accuracy_val", 0.0)),
            float(diagnostics.get("intra_class_distance_mean", 0.0)),
            float(diagnostics.get("inter_center_distance_mean", 0.0)),
            float(diagnostics.get("intra_inter_ratio", 0.0)),
            float(diagnostics.get("min_inter_center_distance", 0.0)),
            diagnostics.get("collision_pairs", []),
        )
        self._logger.info(
            "RepDiag[%s] center_distance_matrix=%s",
            str(label_space),
            diagnostics.get("center_distance_matrix", {}),
        )
        self._logger.info(
            "RepDiag[%s] collision_threshold_p05=%.4f top5_nearest_cluster_pairs=%s",
            str(label_space),
            float(diagnostics.get("collision_threshold_p05", 0.0)),
            diagnostics.get("nearest_cluster_pairs_top5", []),
        )
        self._logger.info(
            "RepDiag[%s] cluster_sizes=%s nearest_center_acc_val=%.4f "
            "density_variance=%.8f",
            str(label_space),
            diagnostics.get("cluster_sizes", []),
            float(
                diagnostics.get(
                    "nearest_center_acc_val",
                    diagnostics.get("nearest_center_accuracy_val", 0.0),
                )
            ),
            float(diagnostics.get("density_variance", 0.0)),
        )
        if bool(diagnostics.get("density_feature_dead", False)):
            self._logger.warning(
                "RepDiag[%s] density_feature_dead=true (near-zero variance)",
                str(label_space),
            )

        # Log Fixes 4-6 diagnostics
        secondary_collisions = diagnostics.get("secondary_collision_pairs", [])
        if secondary_collisions:
            self._logger.info(
                "RepDiag[%s] Fix4_secondary_collisions: %d pairs detected: %s",
                str(label_space),
                len(secondary_collisions),
                secondary_collisions,
            )

        confusion_matrix = diagnostics.get("nearest_center_confusion_matrix", {})
        if confusion_matrix:
            misclassified_counts = {
                cid: len(cm.get("confusion_with", {}))
                for cid, cm in confusion_matrix.items()
            }
            self._logger.info(
                "RepDiag[%s] Fix5_confusion_matrix: classes confused with other classes: %s",
                str(label_space),
                misclassified_counts,
            )

        capacity = diagnostics.get("embedding_capacity_assessment", {})
        if capacity and capacity.get("under_capacity"):
            recs = capacity.get("recommendations", [])
            self._logger.warning(
                "RepDiag[%s] Fix6_capacity_warning: under-capacity detected. "
                "Recommendations: %s",
                str(label_space),
                [r.get("action") for r in recs],
            )

        return diagnostics

    # ------------------------------------------------------------------
    # Snapshot ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def build_representation_snapshot_id(
        diagnostics: dict[str, Any],
        *,
        label_space: str,
        representation_only_steps: int,
        head_only_steps: int,
        sampler_mode: str,
    ) -> str:
        """Build a stable snapshot ID for post-representation geometry state."""
        payload = {
            "label_space": str(label_space),
            "ratio": float(
                diagnostics.get(
                    "intra_inter_ratio", diagnostics.get("ratio", 0.0)
                )
            ),
            "min_inter": float(
                diagnostics.get("min_inter_center_distance", 0.0)
            ),
            "nearest_center_acc": float(
                diagnostics.get(
                    "nearest_center_acc_val",
                    diagnostics.get("nearest_center_accuracy_val", 0.0),
                )
            ),
            "cluster_sizes": [
                int(v) for v in diagnostics.get("cluster_sizes", [])
            ],
            "density_variance": float(
                diagnostics.get("density_variance", 0.0)
            ),
            "representation_only_steps": int(representation_only_steps),
            "head_only_steps": int(head_only_steps),
            "sampler_mode": str(sampler_mode),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"rep_phase_v1_{digest[:16]}"
