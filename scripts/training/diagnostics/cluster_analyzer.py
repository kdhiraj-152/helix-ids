"""ClusterAnalyzer — embedding clustering, relabel proposal generation.

Pure computation — no trainer state imported. Model and device are injected
via constructor for inference; loaders are passed as method arguments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, TensorDataset


class ClusterAnalyzer:
    """Embedding clustering and relabel proposal generation.

    Owns:
      - cluster fitting (fit_embedding_clusters)
      - cluster-label bridge metadata (build_cluster_label_bridge)
      - relabel propagation (apply_cluster_relabels_to_datasets)
      - nearest-center assignment (assign_labels_from_centers)
      - feature embedding through model (embed_feature_matrix)
      - class centroid computation (compute_class_centroids, build_class_centers)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        logger: Any,
        *,
        cluster_relabel_objective: str,
        cluster_relabel_seed: int,
        cluster_relabel_spectral_affinity: str,
    ) -> None:
        self._model = model
        self._device = device
        self._logger = logger
        self._cluster_relabel_objective = str(cluster_relabel_objective)
        self._cluster_relabel_seed = int(cluster_relabel_seed)
        self._cluster_relabel_spectral_affinity = str(cluster_relabel_spectral_affinity)

    # ------------------------------------------------------------------
    # Feature embedding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_representation_features(features: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings for geometry stages."""
        return F.normalize(features, p=2, dim=1)

    def embed_feature_matrix(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """Project feature matrix through backbone and return normalized embeddings.

        Args:
            features: Raw input features as numpy array.
            batch_size: Batch size for chunked forward pass.

        Returns:
            L2-normalized embedding tensor (on CPU).
        """
        x_np = np.asarray(features, dtype=np.float32)
        if int(x_np.shape[0]) == 0:
            return torch.zeros((0, 0), dtype=torch.float32)

        was_training = self._model.training
        self._model.eval()
        embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for start_idx in range(0, int(x_np.shape[0]), int(batch_size)):
                chunk = torch.from_numpy(
                    x_np[start_idx : start_idx + int(batch_size)]
                ).to(self._device, non_blocking=True)
                _, _, features_chunk = self._model(chunk, return_features=True)
                embeddings.append(
                    self.prepare_representation_features(features_chunk)
                    .detach()
                    .to(device="cpu")
                )

        if was_training:
            self._model.train()

        return torch.cat(embeddings, dim=0)

    # ------------------------------------------------------------------
    # Centroid computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_class_centroids(
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        """Compute detached class centroids from normalized embeddings.

        Returns:
            (centroid_tensor [n_classes, dim], class_ids [list[int]])
        """
        if int(features.shape[0]) == 0 or int(labels.shape[0]) == 0:
            return torch.zeros((0, 0), dtype=torch.float32), []

        class_ids = sorted(int(v) for v in torch.unique(labels, dim=0).tolist())
        centroids: list[torch.Tensor] = []
        for class_id in class_ids:
            class_features = features[labels == int(class_id)]
            if int(class_features.shape[0]) == 0:
                continue
            centroids.append(class_features.mean(dim=0))

        if not centroids:
            return torch.zeros((0, 0), dtype=torch.float32), []

        centroid_tensor = (
            torch.stack(centroids, dim=0).detach().to(device="cpu", dtype=torch.float32)
        )
        centroid_tensor = F.normalize(centroid_tensor, p=2, dim=1)
        return centroid_tensor, class_ids

    # ------------------------------------------------------------------
    # Nearest-center assignment
    # ------------------------------------------------------------------

    @staticmethod
    def assign_labels_from_centers(
        embeddings: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """Assign nearest-center cluster labels for embeddings.

        Returns:
            Tensor of integer labels.
        """
        if int(embeddings.shape[0]) == 0:
            return torch.zeros((0,), dtype=torch.int64)
        dists = torch.cdist(embeddings, centers.to(dtype=embeddings.dtype), p=2)
        return torch.argmin(dists, dim=1).to(dtype=torch.int64)

    # ------------------------------------------------------------------
    # Cluster fitting
    # ------------------------------------------------------------------

    def fit_embedding_clusters(
        self,
        embeddings: torch.Tensor,
        *,
        n_clusters: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit clusters on normalized embeddings and return labels/centers.

        Supports kmeans, gmm, and spectral clustering objectives.

        Returns:
            (cluster_labels, cluster_centers)
        """
        if int(embeddings.shape[0]) == 0:
            raise RuntimeError("Cannot fit clusters on empty embedding set")

        k = max(2, min(int(n_clusters), int(embeddings.shape[0])))
        emb_np = embeddings.numpy()

        if self._cluster_relabel_objective == "kmeans":
            kmeans = KMeans(
                n_clusters=k,
                random_state=self._cluster_relabel_seed,
                n_init=10,
            )
            cluster_labels_np = kmeans.fit_predict(emb_np).astype(np.int64)
            centers_np = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
        elif self._cluster_relabel_objective == "gmm":
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=self._cluster_relabel_seed,
            )
            cluster_labels_np = gmm.fit_predict(emb_np).astype(np.int64)
            centers_np = np.asarray(gmm.means_, dtype=np.float32)
        elif self._cluster_relabel_objective == "spectral":
            spectral = SpectralClustering(
                n_clusters=k,
                affinity=self._cluster_relabel_spectral_affinity,
                assign_labels="kmeans",
                random_state=self._cluster_relabel_seed,
            )
            cluster_labels_np = spectral.fit_predict(emb_np).astype(np.int64)
            unique_labels = sorted(int(v) for v in np.unique(cluster_labels_np).tolist())
            if len(unique_labels) != k:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: "
                    "spectral_objective_empty_cluster_detected"
                )
            centers_np = np.stack(
                [
                    emb_np[cluster_labels_np == cluster_id].mean(axis=0)
                    for cluster_id in range(k)
                ],
                axis=0,
            ).astype(np.float32)
        else:
            raise RuntimeError(
                "Hard-stop integrity guard triggered: unsupported_cluster_objective"
            )

        centers = torch.from_numpy(centers_np)
        centers = F.normalize(centers, p=2, dim=1)
        return torch.from_numpy(cluster_labels_np), centers

    # ------------------------------------------------------------------
    # Cluster-label bridge
    # ------------------------------------------------------------------

    @staticmethod
    def build_cluster_label_bridge(
        old_labels: torch.Tensor,
        cluster_labels: torch.Tensor,
        *,
        n_clusters: int,
    ) -> dict[str, Any]:
        """Build stable bridge metadata from legacy labels to cluster labels.

        Returns:
            dict with n_clusters, old_labels, old_to_cluster_counts,
            old_to_cluster_dominant, old_to_cluster_purity, cluster_to_old_counts.
        """
        old_np = np.asarray(old_labels.to(device="cpu", dtype=torch.int64).numpy(), dtype=np.int64)
        cluster_np = np.asarray(
            cluster_labels.to(device="cpu", dtype=torch.int64).numpy(),
            dtype=np.int64,
        )
        unique_old = sorted(int(v) for v in np.unique(old_np).tolist())
        n_clusters = max(2, int(n_clusters))

        old_to_cluster_counts: dict[str, dict[str, int]] = {}
        old_to_cluster_dominant: dict[str, int] = {}
        old_to_cluster_purity: dict[str, float] = {}
        cluster_to_old_counts: dict[str, dict[str, int]] = {}

        for old in unique_old:
            mask = old_np == int(old)
            counts = np.bincount(cluster_np[mask], minlength=n_clusters).astype(np.int64)
            old_to_cluster_counts[str(old)] = {
                str(cluster_id): int(count)
                for cluster_id, count in enumerate(counts.tolist())
            }
            dominant = int(np.argmax(counts))
            old_to_cluster_dominant[str(old)] = dominant
            old_to_cluster_purity[str(old)] = float(
                counts[dominant] / max(1, int(counts.sum()))
            )

        for cluster_id in range(n_clusters):
            mask = cluster_np == int(cluster_id)
            old_counts = np.bincount(
                old_np[mask],
                minlength=max(unique_old) + 1 if unique_old else 1,
            )
            cluster_to_old_counts[str(cluster_id)] = {
                str(old_label): int(old_counts[int(old_label)])
                for old_label in unique_old
            }

        return {
            "n_clusters": n_clusters,
            "old_labels": unique_old,
            "old_to_cluster_counts": old_to_cluster_counts,
            "old_to_cluster_dominant": old_to_cluster_dominant,
            "old_to_cluster_purity": old_to_cluster_purity,
            "cluster_to_old_counts": cluster_to_old_counts,
        }

    # ------------------------------------------------------------------
    # Dataset relabeling
    # ------------------------------------------------------------------

    def apply_cluster_relabels_to_datasets(
        self,
        centers: torch.Tensor,
        *,
        train_loader: DataLoader,
        val_loaders: dict[str, DataLoader],
        test_loaders: dict[str, DataLoader],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Relabel train/val/test datasets by nearest cluster centers.

        Mutates dataset labels in-place on train, val, and test loaders.

        Returns:
            (train_emb, train_cluster_labels, val_emb_all, val_cluster_labels_all)
        """
        train_dataset = train_loader.dataset
        if not isinstance(train_dataset, TensorDataset):
            raise RuntimeError(
                "Cluster relabeling currently requires TensorDataset train loader"
            )

        train_x = train_dataset.tensors[0].detach().to(device="cpu").numpy()
        train_emb = self.embed_feature_matrix(train_x)
        train_cluster_labels = self.assign_labels_from_centers(train_emb, centers)

        train_y_binary = train_dataset.tensors[1]
        train_y_family = train_dataset.tensors[2]
        train_y_family.copy_(
            train_cluster_labels.to(
                device=train_y_family.device, dtype=train_y_family.dtype
            )
        )
        train_y_binary.copy_(
            (train_cluster_labels != 0).to(
                device=train_y_binary.device, dtype=train_y_binary.dtype
            )
        )

        val_emb_all = torch.zeros((0, train_emb.shape[1]), dtype=torch.float32)
        val_labels_all = torch.zeros((0,), dtype=torch.int64)

        # Import MultiTaskNumpyDataset locally to avoid circular imports
        from helix_ids.data.datasets import MultiTaskNumpyDataset

        for loader in val_loaders.values():
            val_dataset = loader.dataset
            if not isinstance(val_dataset, MultiTaskNumpyDataset):
                raise RuntimeError(
                    "Cluster relabeling expects MultiTaskNumpyDataset for validation"
                )
            val_emb = self.embed_feature_matrix(
                np.asarray(val_dataset.features, dtype=np.float32)
            )
            val_cluster_labels = self.assign_labels_from_centers(val_emb, centers)
            val_dataset.family_labels = np.asarray(
                val_cluster_labels.numpy(), dtype=np.int64
            ).copy()
            if int(val_dataset.family_labels.shape[0]) > 0:
                self._logger.info(
                    "ClusterRelabel[val] rows=%d label_min=%d label_max=%d unique=%s",
                    int(val_dataset.family_labels.shape[0]),
                    int(np.min(val_dataset.family_labels)),
                    int(np.max(val_dataset.family_labels)),
                    [int(x) for x in np.unique(val_dataset.family_labels).tolist()],
                )
            val_emb_all = (
                torch.cat([val_emb_all, val_emb], dim=0)
                if int(val_emb_all.shape[0]) > 0
                else val_emb
            )
            val_labels_all = (
                torch.cat([val_labels_all, val_cluster_labels], dim=0)
                if int(val_labels_all.shape[0]) > 0
                else val_cluster_labels
            )

        for loader in test_loaders.values():
            test_dataset = loader.dataset
            if not isinstance(test_dataset, MultiTaskNumpyDataset):
                raise RuntimeError(
                    "Cluster relabeling expects MultiTaskNumpyDataset for test"
                )
            test_emb = self.embed_feature_matrix(
                np.asarray(test_dataset.features, dtype=np.float32)
            )
            test_cluster_labels = self.assign_labels_from_centers(test_emb, centers)
            test_dataset.family_labels = np.asarray(
                test_cluster_labels.numpy(), dtype=np.int64
            ).copy()
            if int(test_dataset.family_labels.shape[0]) > 0:
                self._logger.info(
                    "ClusterRelabel[test] rows=%d label_min=%d label_max=%d unique=%s",
                    int(test_dataset.family_labels.shape[0]),
                    int(np.min(test_dataset.family_labels)),
                    int(np.max(test_dataset.family_labels)),
                    [int(x) for x in np.unique(test_dataset.family_labels).tolist()],
                )

        return train_emb, train_cluster_labels, val_emb_all, val_labels_all

    # ------------------------------------------------------------------
    # Class center building
    # ------------------------------------------------------------------

    @staticmethod
    def build_class_centers(
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        class_ids: list[int],
    ) -> tuple[dict[int, torch.Tensor], list[int]]:
        """Build centers for available classes.

        Returns:
            (centers dict, available_class_ids list)
        """
        centers: dict[int, torch.Tensor] = {}
        available_class_ids: list[int] = []
        for cls in class_ids:
            mask = train_labels == int(cls)
            if bool(mask.any()):
                centers[int(cls)] = train_features[mask].mean(dim=0)
                available_class_ids.append(int(cls))
        return centers, available_class_ids
