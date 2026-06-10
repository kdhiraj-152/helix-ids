"""Geometric representation fixes for embedding space quality.

Addresses the geometric non-separability issue post-label-merge by:
1. Adding interaction and distributional features
2. Splitting destructive normalization
3. Adding local density awareness
4. Computing collision detection metrics
5. Building confusion matrices

This module is mandatory before clustering optimization phase.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GeometricRepresentationFixer:
    """Fixes geometric non-separability in embedding space."""

    def __init__(self, *, k_nn: int = 20, density_percentile: float = 95.0):
        """Initialize with k-NN parameters.
        
        Args:
            k_nn: Number of neighbors for density computation (default 20)
            density_percentile: Percentile for outlier density bounds (default 95.0)
        """
        self.k_nn = k_nn
        self.density_percentile = density_percentile

    def add_interaction_features(self, df: pd.DataFrame, *, minimal_set: bool = True) -> pd.DataFrame:
        """Add interaction and distributional features per user specification.
        
        Minimal set (Fix 1):
        1. log(src_bytes + 1), log(dst_bytes + 1)
        2. src_bytes / (dst_bytes + 1)
        3. Connection count features (same src/dst/service window)
        4. Service × protocol one-hot cross
        5. Flag entropy per connection window
        
        Args:
            df: Input dataframe with raw features
            minimal_set: If True, use minimal fix set; if False, extended set
            
        Returns:
            DataFrame with new interaction features appended
        """
        df = df.copy()

        # Apply feature engineering in stages
        self._add_byte_features(df)
        self._add_count_features(df)
        self._add_rate_flag_service_interactions(df)
        self._add_categorical_cross_features(df)
        self._add_flag_entropy_features(df)

        if not minimal_set:
            self._add_extended_features(df)

        # Count new features for logging
        new_feature_pattern = (
            "log_",
            "src_dst",
            "dst_src",
            "srv_count_",
            "count_x_srv_count",
            "same_host_rate_x_service",
            "diff_srv_rate_x_flag",
            "service_protocol",
            "protocol_service_flag",
            "flag_entropy",
        )
        new_count = len([c for c in df.columns if c.startswith(new_feature_pattern)])

        logger.info("Added %d interaction features: new shape %s", new_count, df.shape)
        return df

    def _add_byte_features(self, df: pd.DataFrame) -> None:
        """Add log-transformed and ratio-based byte features."""
        if "src_bytes" in df.columns:
            df["log_src_bytes"] = np.log1p(np.abs(df["src_bytes"].fillna(0)))
        if "dst_bytes" in df.columns:
            df["log_dst_bytes"] = np.log1p(np.abs(df["dst_bytes"].fillna(0)))

        if "src_bytes" in df.columns and "dst_bytes" in df.columns:
            src_b = np.abs(df["src_bytes"].fillna(0)).astype(np.float32)
            dst_b = np.abs(df["dst_bytes"].fillna(0)).astype(np.float32)
            df["src_dst_bytes_ratio"] = src_b / (dst_b + 1.0)
            df["dst_src_bytes_ratio"] = dst_b / (src_b + 1.0)

    def _add_count_features(self, df: pd.DataFrame) -> None:
        """Add log-transformed connection count features and ratios."""
        if "count" in df.columns:
            df["log_count"] = np.log1p(np.abs(df["count"].fillna(0)))
        if "srv_count" in df.columns:
            df["log_srv_count"] = np.log1p(np.abs(df["srv_count"].fillna(0)))
        if "count" in df.columns and "srv_count" in df.columns:
            count_val = np.abs(df["count"].fillna(0)).astype(np.float32)
            srv_count_val = np.abs(df["srv_count"].fillna(0)).astype(np.float32)
            df["srv_count_ratio"] = srv_count_val / (count_val + 1.0)
            df["count_x_srv_count"] = count_val * srv_count_val

    def _add_rate_flag_service_interactions(self, df: pd.DataFrame) -> None:
        """Add high-signal rate × categorical interactions for minority-class separation."""
        same_host_rate_col = next(
            (col for col in ("same_host_rate", "same_srv_rate", "dst_host_same_srv_rate") if col in df.columns),
            None,
        )
        if same_host_rate_col is not None and "service" in df.columns:
            same_host_rate = np.abs(df[same_host_rate_col].fillna(0)).astype(np.float32)
            service_codes = pd.factorize(df["service"].fillna("__MISSING__").astype(str), sort=True)[0]
            df["same_host_rate_x_service"] = same_host_rate * (service_codes.astype(np.float32) + 1.0)

        if "diff_srv_rate" in df.columns and "flag" in df.columns:
            diff_srv_rate = np.abs(df["diff_srv_rate"].fillna(0)).astype(np.float32)
            flag_codes = pd.factorize(df["flag"].fillna("__MISSING__").astype(str), sort=True)[0]
            df["diff_srv_rate_x_flag"] = diff_srv_rate * (flag_codes.astype(np.float32) + 1.0)

    def _add_categorical_cross_features(self, df: pd.DataFrame) -> None:
        """Add service × protocol categorical interaction."""
        if "service" in df.columns and "protocol_type" in df.columns:
            df["service_protocol"] = (
                df["service"].astype(str) + "_" + df["protocol_type"].astype(str)
            )

        if all(col in df.columns for col in ("protocol_type", "service", "flag")):
            df["protocol_service_flag"] = (
                df["protocol_type"].astype(str)
                + "_"
                + df["service"].astype(str)
                + "_"
                + df["flag"].astype(str)
            )

    def _add_flag_entropy_features(self, df: pd.DataFrame) -> None:
        """Add flag entropy per row for behavioral diversity."""
        if "flag" in df.columns:
            flag_vals = df["flag"].fillna(0).astype(str)
            df["flag_entropy"] = flag_vals.apply(
                lambda x: float(entropy([ord(c) for c in str(x)]) if len(str(x)) > 0 else 0.0)
            )

    def _add_extended_features(self, df: pd.DataFrame) -> None:
        """Add extended set of discriminative features."""
        for col in ["serror_rate", "rerror_rate"]:
            if col in df.columns:
                df[f"{col}_logged"] = -np.log10(np.abs(df[col].fillna(0)) + 1e-9)

        if "same_srv_rate" in df.columns:
            df["same_srv_rate_inv"] = 1.0 - (np.abs(df["same_srv_rate"].fillna(0)))

    def split_normalization(
        self,
        X: np.ndarray,
        *,
        feature_names: Optional[list[str]] = None,
        categorical_cols: Optional[set[str]] = None,
        fit: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply selective normalization (Fix 2).
        
        Splits continuous and categorical features:
        - Continuous: StandardScaler (z-score normalization)
        - Categorical/one-hot: Untouched (preserves sparsity & rarity signals)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names (inferred if None)
            categorical_cols: Set of categorical column names to skip normalization
            fit: If True, fit scaler; if False, use stored stats
            
        Returns:
            (normalized_X, normalization_stats) tuple
        """
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        if categorical_cols is None:
            categorical_cols = set()

        x_out = X.copy().astype(np.float32)
        stats: dict[str, Any] = {
            "scaler": None,
            "categorical_cols": list(categorical_cols),
            "continuous_indices": [],
            "categorical_indices": [],
        }

        # Identify continuous vs categorical
        continuous_idx = []
        categorical_idx = []
        for i, name in enumerate(feature_names):
            if name in categorical_cols or name.startswith(("service_", "flag", "protocol")):
                categorical_idx.append(i)
            else:
                continuous_idx.append(i)

        stats["continuous_indices"] = continuous_idx
        stats["categorical_indices"] = categorical_idx

        # Normalize only continuous features
        if continuous_idx:
            x_continuous = x_out[:, continuous_idx]
            if fit:
                scaler = StandardScaler()
                x_continuous = scaler.fit_transform(x_continuous)
                stats["scaler"] = scaler
            else:
                if stats["scaler"]:
                    x_continuous = stats["scaler"].transform(x_continuous)

            x_out[:, continuous_idx] = x_continuous

        logger.info(
            "Split normalization: %d continuous (scaled), %d categorical (preserved)",
            len(continuous_idx),
            len(categorical_idx),
        )
        return x_out, stats

    def add_local_density_features(
        self,
        X: np.ndarray,
        *,
        k: Optional[int] = None,
        fit: bool = True,
        nn_model: Optional[NearestNeighbors] = None,
    ) -> tuple[np.ndarray, Optional[NearestNeighbors]]:
        """Add local k-NN density as feature (Fix 3).
        
        Separates sparse anomalies from dense normal traffic by appending
        1 / avg_distance_to_k_neighbors as a feature.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            k: Number of neighbors (defaults to self.k_nn)
            fit: If True, fit neighbors model; if False, use provided model
            nn_model: Pre-fitted NearestNeighbors model (used if fit=False)
            
        Returns:
            (X_with_density, nn_model) tuple
        """
        k = k or self.k_nn

        if fit:
            nn_model = NearestNeighbors(n_neighbors=k, n_jobs=-1)
            nn_model.fit(X)

        if nn_model is None:
            raise ValueError("nn_model required when fit=False")

        # Compute k-NN distances (exclude self)
        distances, _ = nn_model.kneighbors(X)

        # Average distance to k neighbors (density inverse)
        avg_distances = distances[:, 1:].mean(axis=1)  # Exclude self
        density = 1.0 / (avg_distances + 1e-8)

        # Append as feature (log scale for better dynamic range)
        log_density = np.log1p(density).reshape(-1, 1).astype(np.float32)
        x_out = np.hstack([X, log_density]).astype(np.float32)

        logger.info(
            "Added density feature: density range [%.4f, %.4f], "
            "shape %s -> %s",
            float(np.min(log_density)),
            float(np.max(log_density)),
            X.shape,
            x_out.shape,
        )
        return x_out, nn_model

    def detect_secondary_collisions(
        self,
        centers: dict[int, np.ndarray],
        *,
        collision_threshold: float = 0.20,
    ) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
        """Detect secondary collisions at centroid level (Fix 4).
        
        Identifies class pairs where centroids are too close, indicating
        geometric indistinguishability that must be resolved via merge or
        feature augmentation.
        
        Args:
            centers: Dict mapping class_id -> centroid vector
            collision_threshold: Maximum allowed inter-centroid distance (default 0.20)
            
        Returns:
            (collision_pairs, class_collision_map) tuple
        """
        class_ids = sorted(centers.keys())
        collision_pairs: list[dict[str, Any]] = []
        class_collision_map: dict[int, list[int]] = {cid: [] for cid in class_ids}

        if len(class_ids) < 2:
            return collision_pairs, class_collision_map

        center_matrix = np.array([centers[cid] for cid in class_ids], dtype=np.float32)
        dist_mat = cdist(center_matrix, center_matrix, metric="euclidean")

        for i, cls_i in enumerate(class_ids):
            for j in range(i + 1, len(class_ids)):
                cls_j = class_ids[j]
                dist_val = float(dist_mat[i, j])

                if dist_val < collision_threshold:
                    collision_pairs.append({
                        "class_i": int(cls_i),
                        "class_j": int(cls_j),
                        "distance": dist_val,
                        "severity": "critical" if dist_val < 0.10 else "warning",
                    })
                    class_collision_map[cls_i].append(cls_j)
                    class_collision_map[cls_j].append(cls_i)

        logger.info(
            "Detected %d secondary collisions (threshold=%.3f)",
            len(collision_pairs),
            collision_threshold,
        )
        return collision_pairs, class_collision_map

    def build_nearest_center_confusion_matrix(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centers: dict[int, np.ndarray],
        *,
        top_k: int = 3,
    ) -> dict[int, dict[str, Any]]:
        """Build per-class confusion matrix showing geometric indistinguishability (Fix 5).
        
        For each class, computes which other classes are geometrically nearby (nearest centers).
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Class labels (n_samples,)
            centers: Dict mapping class_id -> centroid vector
            top_k: Number of nearest centers to track per sample
            
        Returns:
            Dict mapping class_id -> {
                'n_samples': int,
                'correct_nearest_center_ratio': float,
                'nearest_classes': list of (class_id, count) pairs,
                'confusion_with': {other_class_id: misclassified_count}
            }
        """
        class_ids = sorted(centers.keys())
        confusion_matrix: dict[int, dict[str, Any]] = {}

        for cls in class_ids:
            confusion_matrix[cls] = {
                "n_samples": 0,
                "correct_nearest_center_count": 0,
                "nearest_classes": {},
                "confusion_with": {},
            }

        if len(features) == 0 or len(class_ids) < 1:
            return confusion_matrix

        # Compute distances to all centers for all samples
        center_matrix = np.array([centers[cid] for cid in class_ids], dtype=np.float32)
        dists = cdist(features, center_matrix, metric="euclidean")  # (n_samples, n_classes)
        nearest_class_idx = np.argsort(dists, axis=1)

        # Build confusion matrix - split into helper to reduce complexity
        self._populate_confusion_matrix(
            nearest_class_idx,
            labels,
            class_ids,
            confusion_matrix,
            top_k,
        )

        # Compute ratios
        self._compute_confusion_ratios(class_ids, confusion_matrix)

        logger.info(
            "Built nearest-center confusion matrix for %d classes",
            len(class_ids),
        )
        return confusion_matrix

    def _populate_confusion_matrix(
        self,
        nearest_class_idx: np.ndarray,
        labels: np.ndarray,
        class_ids: list[int],
        confusion_matrix: dict[int, dict[str, Any]],
        top_k: int,
    ) -> None:
        """Helper to populate confusion matrix entries."""
        for sample_idx, true_class_idx in enumerate(labels):
            true_class = int(true_class_idx)
            if true_class not in confusion_matrix:
                continue

            confusion_matrix[true_class]["n_samples"] += 1
            nearest_idx = nearest_class_idx[sample_idx, 0]
            nearest_class = class_ids[nearest_idx]

            if nearest_class == true_class:
                confusion_matrix[true_class]["correct_nearest_center_count"] += 1
            else:
                # Track confusion
                confusion_with: dict[int, int] = confusion_matrix[true_class]["confusion_with"]
                if nearest_class not in confusion_with:
                    confusion_with[nearest_class] = 0
                confusion_with[nearest_class] += 1

            # Track all nearest classes
            nearest_classes: dict[int, int] = confusion_matrix[true_class]["nearest_classes"]
            for rank in range(min(top_k, len(class_ids))):
                nc = class_ids[nearest_class_idx[sample_idx, rank]]
                if nc not in nearest_classes:
                    nearest_classes[nc] = 0
                nearest_classes[nc] += 1

    def _compute_confusion_ratios(
        self,
        class_ids: list[int],
        confusion_matrix: dict[int, dict[str, Any]],
    ) -> None:
        """Helper to compute confusion matrix ratios."""
        for cls in class_ids:
            n_samples: int = confusion_matrix[cls]["n_samples"]
            if n_samples > 0:
                correct_count: int = confusion_matrix[cls]["correct_nearest_center_count"]
                correct_ratio = correct_count / float(n_samples)
                confusion_matrix[cls]["correct_nearest_center_ratio"] = float(correct_ratio)

    def assess_embedding_capacity(
        self,
        intra_inter_ratio: float,
        embedding_dim: int,
        dropout_rate: float,
        *,
        target_ratio: float = 0.8,
    ) -> dict[str, Any]:
        """Assess if embedding has sufficient capacity (Fix 6).
        
        If ratio still > 1 after feature fixes, the model may be under-capacity.
        
        Args:
            intra_inter_ratio: Current intra/inter ratio
            embedding_dim: Current embedding dimension
            dropout_rate: Current dropout rate
            target_ratio: Target ratio to aim for (default 0.8)
            
        Returns:
            Dict with capacity assessment and recommendations
        """
        recommendations = []

        if intra_inter_ratio >= target_ratio:
            # Model is under-capacity
            if embedding_dim < 256:
                recommendations.append({
                    "action": "increase_embedding_dim",
                    "current": embedding_dim,
                    "suggested": int(embedding_dim * 1.5),
                    "rationale": "Embedding dimension too small for manifold complexity",
                })

            if dropout_rate > 0.3:
                recommendations.append({
                    "action": "reduce_dropout",
                    "current": dropout_rate,
                    "suggested": max(0.1, dropout_rate - 0.1),
                    "rationale": "Dropout may be over-regularizing, destroying discriminative structure",
                })

            recommendations.append({
                "action": "remove_bottleneck_layers",
                "rationale": "Remove any intermediate layers that compress to < 50% of embedding_dim",
            })

        return {
            "current_ratio": float(intra_inter_ratio),
            "target_ratio": float(target_ratio),
            "under_capacity": bool(intra_inter_ratio >= target_ratio),
            "recommendations": recommendations,
        }


def apply_geometric_fixes(
    X: np.ndarray,
    labels: np.ndarray,
    centers: dict[int, np.ndarray],
    *,
    feature_names: Optional[list[str]] = None,
    categorical_cols: Optional[set[str]] = None,
    k_nn: int = 20,
    collision_threshold: float = 0.20,
) -> dict[str, Any]:
    """Apply all geometric representation fixes atomically.
    
    This is the entry point for fixing geometric non-separability.
    
    Args:
        X: Original feature matrix
        labels: Class labels
        centers: Dict mapping class_id -> centroid vector
        feature_names: List of feature column names
        categorical_cols: Set of categorical column names
        k_nn: Number of neighbors for density computation
        collision_threshold: Threshold for secondary collision detection
        
    Returns:
        Dict containing:
        - X_enhanced: Enhanced feature matrix with all fixes applied
        - diagnostics: Collision/confusion diagnostics
        - capacity_assessment: Embedding capacity recommendations
        - normalization_stats: Normalization metadata
        - nn_model: Fitted k-NN model for density
    """
    fixer = GeometricRepresentationFixer(k_nn=k_nn)

    # Fix 2: Apply split normalization first
    x_normalized, norm_stats = fixer.split_normalization(
        X,
        feature_names=feature_names,
        categorical_cols=categorical_cols,
        fit=True,
    )

    # Fix 3: Add local density features
    x_with_density, nn_model = fixer.add_local_density_features(
        x_normalized,
        k=k_nn,
        fit=True,
    )

    # Fix 4 & 5: Collision detection & confusion matrix
    collision_pairs, class_collision_map = fixer.detect_secondary_collisions(
        centers,
        collision_threshold=collision_threshold,
    )
    confusion_matrix = fixer.build_nearest_center_confusion_matrix(
        x_normalized,  # Use pre-density-augmented features for clarity
        labels,
        centers,
        top_k=3,
    )

    return {
        "X_enhanced": x_with_density,
        "collision_pairs": collision_pairs,
        "class_collision_map": class_collision_map,
        "confusion_matrix": confusion_matrix,
        "normalization_stats": norm_stats,
        "nn_model": nn_model,
    }
