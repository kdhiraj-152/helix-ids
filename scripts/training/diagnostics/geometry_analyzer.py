"""GeometryAnalyzer — geometry integrity, collision detection, density estimation.

Pure computation — no trainer state imported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


class GeometryAnalyzer:
    """Geometry integrity and collision analysis for embedding spaces.

    Owns:
      - geometry integrity validation (enforce_geometry_integrity)
      - collision detection (has_critical_collision_pairs, critical_pair_key)
      - density estimation (estimate_local_density_diagnostics)
      - inter/intra distance computation (compute_inter_and_intra_distances)
      - stage-aware ratio threshold (current_geometry_ratio_threshold)
    """

    def __init__(
        self,
        *,
        use_energy_based_family_objective: bool,
        geometry_min_cluster_size: int,
        critical_collision_pairs: set[tuple[int, int]],
        geometry_min_inter_threshold: float,
        geometry_max_intra_inter_ratio_warmup: float,
        geometry_max_intra_inter_ratio_post_phase: float,
        logger: Any,
    ) -> None:
        self._use_energy_based_family_objective = bool(use_energy_based_family_objective)
        self._geometry_min_cluster_size = int(geometry_min_cluster_size)
        self._critical_collision_pairs: set[tuple[int, int]] = critical_collision_pairs
        self._geometry_min_inter_threshold = float(geometry_min_inter_threshold)
        self._geometry_max_intra_inter_ratio_warmup = float(geometry_max_intra_inter_ratio_warmup)
        self._geometry_max_intra_inter_ratio_post_phase = float(geometry_max_intra_inter_ratio_post_phase)
        self._logger = logger

    # ------------------------------------------------------------------
    # Inter/intra distance computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_inter_and_intra_distances(
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        centers: dict[int, torch.Tensor],
        dist_mat: torch.Tensor,
        available_class_ids: list[int],
        collision_threshold: float,
    ) -> tuple[list[float], list[dict[str, Any]], list[float]]:
        """Compute inter-class and intra-class distances.

        Returns:
            (inter_distances, collision_pairs, intra_distances)
        """
        inter_distances: list[float] = []
        collision_pairs: list[dict[str, Any]] = []
        for i, cls_i in enumerate(available_class_ids):
            for j in range(i + 1, len(available_class_ids)):
                cls_j = available_class_ids[j]
                dist_val = float(dist_mat[i, j].item())
                inter_distances.append(dist_val)
                if dist_val <= collision_threshold:
                    collision_pairs.append(
                        {
                            "class_i": int(cls_i),
                            "class_j": int(cls_j),
                            "distance": dist_val,
                        }
                    )

        intra_distances: list[float] = []
        for cls in available_class_ids:
            mask = train_labels == int(cls)
            if bool(mask.any()):
                class_points = train_features[mask]
                class_center = centers[int(cls)].unsqueeze(0)
                intra_distances.extend(
                    torch.norm(class_points - class_center, dim=1).to(device="cpu").tolist()
                )

        return inter_distances, collision_pairs, intra_distances

    # ------------------------------------------------------------------
    # Density estimation
    # ------------------------------------------------------------------

    def estimate_local_density_diagnostics(
        self,
        train_features: torch.Tensor,
        *,
        k: int = 20,
        max_samples: int = 4096,
        run_seed: int,
    ) -> dict[str, Any]:
        """Estimate k-NN density stability on embedding space.

        Args:
            train_features: L2-normalized embedding tensor.
            k: Number of nearest neighbours for density estimation.
            max_samples: Maximum samples to use (random subset if larger).
            run_seed: PRNG seed for deterministic subsampling.

        Returns:
            dict with density_variance, density_mean, density_k, density_feature_dead.
        """
        if int(train_features.shape[0]) <= 2:
            return {
                "density_variance": 0.0,
                "density_mean": 0.0,
                "density_sample_count": int(train_features.shape[0]),
                "density_k": 0,
                "density_feature_dead": True,
            }

        features_np = np.asarray(
            train_features.detach().to(device="cpu", dtype=torch.float32).numpy(),
            dtype=np.float32,
        )
        if int(features_np.shape[0]) > int(max_samples):
            rng = np.random.default_rng(run_seed)
            idx = rng.choice(features_np.shape[0], size=int(max_samples), replace=False)
            features_np = features_np[idx]

        k_eff = int(min(k, max(2, int(features_np.shape[0]) - 1)))
        if k_eff < 2:
            return {
                "density_variance": 0.0,
                "density_mean": 0.0,
                "density_sample_count": int(features_np.shape[0]),
                "density_k": int(k_eff),
                "density_feature_dead": True,
            }

        nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1)
        nn.fit(features_np)
        distances, _ = nn.kneighbors(features_np)
        avg_distances = distances[:, 1:].mean(axis=1)
        log_density = np.log1p(1.0 / (avg_distances + 1e-8))
        density_variance = float(np.var(log_density))

        return {
            "density_variance": density_variance,
            "density_mean": float(np.mean(log_density)),
            "density_sample_count": int(features_np.shape[0]),
            "density_k": int(k_eff),
            "density_feature_dead": bool(density_variance <= 1e-8),
        }

    # ------------------------------------------------------------------
    # Geometry integrity
    # ------------------------------------------------------------------

    def enforce_geometry_integrity(
        self,
        diagnostics: dict[str, Any],
        *,
        label_space: str,
    ) -> None:
        """Fail fast when embedding geometry is not classifier-ready.

        Raises:
            RuntimeError if geometry fails validation gates.
        """
        if self._use_energy_based_family_objective:
            self._logger.info(
                "Geometry integrity gate bypassed in energy mode [label_space=%s]",
                str(label_space),
            )
            return

        ratio = float(diagnostics.get("intra_inter_ratio", diagnostics.get("ratio", 0.0)))
        min_inter = float(diagnostics.get("min_inter_center_distance", 0.0))
        nearest_center_acc = float(
            diagnostics.get(
                "nearest_center_acc_val",
                diagnostics.get("nearest_center_accuracy_val", 0.0),
            )
        )
        cluster_sizes = [int(v) for v in diagnostics.get("cluster_sizes", [])]
        ratio_threshold = 1.8
        min_inter_threshold = 0.4
        nearest_center_threshold = 0.85

        if ratio > ratio_threshold:
            raise RuntimeError(
                "Geometry invalid: intra/inter ratio above threshold "
                f"[{label_space}] ratio={ratio:.4f} "
                f"threshold={ratio_threshold:.4f}"
            )

        if min_inter < min_inter_threshold:
            raise RuntimeError(
                "Geometry invalid: unresolved cluster collisions "
                f"[{label_space}] min_inter={min_inter:.4f} "
                f"threshold={min_inter_threshold:.4f}"
            )

        enforce_cluster_size_gate = str(label_space).strip().lower() in {
            "cluster_relabel",
            "joint_finetune",
        }
        if (
            enforce_cluster_size_gate
            and cluster_sizes
            and min(cluster_sizes) < int(self._geometry_min_cluster_size)
        ):
            raise RuntimeError(
                "Dead cluster detected "
                f"[{label_space}] min_cluster_size={min(cluster_sizes)} "
                f"threshold={int(self._geometry_min_cluster_size)}"
            )

        if nearest_center_acc < nearest_center_threshold:
            raise RuntimeError(
                "Geometry invalid: nearest_center_acc below threshold "
                f"[{label_space}] nearest_center_acc={nearest_center_acc:.4f} "
                f"threshold={nearest_center_threshold:.4f}"
            )

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    @staticmethod
    def critical_pair_key(class_i: int, class_j: int) -> tuple[int, int]:
        """Return normalized class-pair key (always (min, max))."""
        a, b = int(class_i), int(class_j)
        return (a, b) if a <= b else (b, a)

    def has_critical_collision_pairs(self, diagnostics: dict[str, Any]) -> bool:
        """Check whether critical collision pairs remain unresolved."""
        top_pairs: list[dict[str, Any]] = diagnostics.get("nearest_cluster_pairs_top5", [])
        for item in top_pairs:
            pair_key = self.critical_pair_key(
                int(item.get("class_i", -1)), int(item.get("class_j", -1))
            )
            if pair_key in self._critical_collision_pairs and float(
                item.get("distance", 1.0)
            ) < self._geometry_min_inter_threshold:
                return True
        return False

    # ------------------------------------------------------------------
    # Stage-aware threshold
    # ------------------------------------------------------------------

    def current_geometry_ratio_threshold(
        self,
        representation_phase_active: bool,
        head_phase_start_step: int,
    ) -> float:
        """Return stage-aware geometry ratio threshold."""
        if representation_phase_active or head_phase_start_step < 0:
            return float(self._geometry_max_intra_inter_ratio_warmup)
        return float(self._geometry_max_intra_inter_ratio_post_phase)
