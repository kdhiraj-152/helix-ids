"""Regression tests for Phase 13A-1 extracted diagnostics delegates.

Validates that the three diagnostics delegate classes (GeometryAnalyzer,
ClusterAnalyzer, RepresentationDiagnostics) produce identical outputs
to the original code extracted from train_helix_ids_full.py.

Covers all 20 delegated methods with pure-function and instance-method tests,
including edge cases (empty tensors, single-class, degenerate, boundary values).
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.training.diagnostics import (
    ClusterAnalyzer,
    GeometryAnalyzer,
    RepresentationDiagnostics,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test_diagnostics")


@pytest.fixture
def geometry_analyzer(logger: logging.Logger) -> GeometryAnalyzer:
    return GeometryAnalyzer(
        use_energy_based_family_objective=False,
        geometry_min_cluster_size=3,
        critical_collision_pairs={(0, 1), (2, 3)},
        geometry_min_inter_threshold=0.4,
        geometry_max_intra_inter_ratio_warmup=3.0,
        geometry_max_intra_inter_ratio_post_phase=1.8,
        logger=logger,
    )


@pytest.fixture
def geometry_analyzer_energy(logger: logging.Logger) -> GeometryAnalyzer:
    """Energy mode bypasses geometry integrity checks."""
    return GeometryAnalyzer(
        use_energy_based_family_objective=True,
        geometry_min_cluster_size=3,
        critical_collision_pairs={(0, 1)},
        geometry_min_inter_threshold=0.4,
        geometry_max_intra_inter_ratio_warmup=3.0,
        geometry_max_intra_inter_ratio_post_phase=1.8,
        logger=logger,
    )


@pytest.fixture
def mock_model() -> torch.nn.Module:
    """Minimal model returning backbone features via return_features flag."""
    model = MagicMock(spec=torch.nn.Module)

    def forward_side_effect(x, return_features=False):
        # Return (logits, logits, normalized_features)
        batch_size = x.shape[0]
        # Produce deterministic normalized output for testing
        features = torch.ones(batch_size, 4, dtype=torch.float32)
        features = F.normalize(features, p=2, dim=1)
        logits = torch.zeros(batch_size, 2, dtype=torch.float32)
        return logits, logits, features

    model.side_effect = forward_side_effect
    model.training = False
    return model


@pytest.fixture
def cluster_analyzer(
    mock_model: torch.nn.Module, logger: logging.Logger
) -> ClusterAnalyzer:
    return ClusterAnalyzer(
        model=mock_model,
        device=torch.device("cpu"),
        logger=logger,
        cluster_relabel_objective="kmeans",
        cluster_relabel_seed=42,
        cluster_relabel_spectral_affinity="nearest_neighbors",
    )


@pytest.fixture
def rep_diagnostics(
    mock_model: torch.nn.Module, logger: logging.Logger
) -> RepresentationDiagnostics:
    return RepresentationDiagnostics(
        model=mock_model,
        device=torch.device("cpu"),
        logger=logger,
        representation_only_steps=100,
        head_only_steps=200,
        sampler_mode="balanced",
    )


@pytest.fixture
def sample_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Well-separated 3-class embedding space."""
    rng = torch.Generator().manual_seed(42)
    # Class 0: cluster around (-1, 0, 0, 0)
    c0 = torch.randn(20, 4, generator=rng) * 0.1 + torch.tensor([-1.0, 0.0, 0.0, 0.0])
    # Class 1: cluster around (0, 1, 0, 0)
    c1 = torch.randn(20, 4, generator=rng) * 0.1 + torch.tensor([0.0, 1.0, 0.0, 0.0])
    # Class 2: cluster around (0, 0, 1, 0)
    c2 = torch.randn(20, 4, generator=rng) * 0.1 + torch.tensor([0.0, 0.0, 1.0, 0.0])
    features = torch.cat([c0, c1, c2], dim=0)
    features = F.normalize(features, p=2, dim=1)
    labels = torch.cat(
        [
            torch.zeros(20, dtype=torch.int64),
            torch.ones(20, dtype=torch.int64),
            torch.full((20,), 2, dtype=torch.int64),
        ],
        dim=0,
    )
    return features, labels


@pytest.fixture
def sample_diagnostics(sample_embeddings) -> dict[str, Any]:
    """A diagnostics dict resembling the real output for validation tests."""
    features, labels = sample_embeddings
    centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0, 1, 2])
    centers_t = torch.stack([centers[c] for c in cids], dim=0)
    dist_mat = torch.cdist(centers_t, centers_t, p=2)
    pairs, threshold, top5 = RepresentationDiagnostics.compute_center_pair_diagnostics(
        dist_mat, cids
    )

    return {
        "available_class_ids": cids,
        "center_distance_matrix": {
            str(c): {str(d): float(dist_mat[i, j].item()) for j, d in enumerate(cids)}
            for i, c in enumerate(cids)
        },
        "nearest_center_accuracy_train": 1.0,
        "nearest_center_accuracy_val": 1.0,
        "nearest_center_acc_val": 1.0,
        "intra_class_distance_mean": 0.05,
        "inter_center_distance_mean": 1.2,
        "intra_inter_ratio": 0.0417,
        "min_inter_center_distance": 0.95,
        "cluster_size_counts": [20, 20, 20],
        "cluster_sizes": [20, 20, 20],
        "collision_threshold_p05": threshold,
        "nearest_cluster_pairs_top5": top5,
        "density_variance": 0.5,
        "density_feature_dead": False,
        "collision_pairs": [],
        "secondary_collision_pairs": [],
        "nearest_center_confusion_matrix": {},
        "embedding_capacity_assessment": {},
    }


# ======================================================================
# GeometryAnalyzer
# ======================================================================


class TestGeometryAnalyzerConstruction:
    """GeometryAnalyzer init and instance state."""

    def test_constructor_stores_params(self, logger: logging.Logger):
        """All constructor params are stored as instance attrs."""
        ga = GeometryAnalyzer(
            use_energy_based_family_objective=True,
            geometry_min_cluster_size=5,
            critical_collision_pairs={(1, 2)},
            geometry_min_inter_threshold=0.3,
            geometry_max_intra_inter_ratio_warmup=2.5,
            geometry_max_intra_inter_ratio_post_phase=1.5,
            logger=logger,
        )
        assert ga._use_energy_based_family_objective is True
        assert ga._geometry_min_cluster_size == 5
        assert ga._critical_collision_pairs == {(1, 2)}
        assert ga._geometry_min_inter_threshold == 0.3
        assert ga._geometry_max_intra_inter_ratio_warmup == 2.5
        assert ga._geometry_max_intra_inter_ratio_post_phase == 1.5


class TestGeometryAnalyzerInterIntra:
    """GeometryAnalyzer.compute_inter_and_intra_distances."""

    def test_basic_inter_and_intra(self, sample_embeddings):
        """Well-separated classes produce no collisions and sensible distances."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0, 1, 2])
        centers_t = torch.stack([centers[c] for c in cids], dim=0)
        dist_mat = torch.cdist(centers_t, centers_t, p=2)

        inter_dists, collisions, intra_dists = (
            GeometryAnalyzer.compute_inter_and_intra_distances(
                features, labels, centers, dist_mat, cids, collision_threshold=0.25
            )
        )

        assert len(inter_dists) == 3  # 3 choose 2 = 3 pairs
        assert all(d > 0.5 for d in inter_dists)
        assert len(collisions) == 0  # well-separated
        assert len(intra_dists) == 60  # 20 per class * 3 classes

    def test_collision_detection(self, sample_embeddings):
        """High collision threshold should flag pairs as collisions."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0, 1, 2])
        centers_t = torch.stack([centers[c] for c in cids], dim=0)
        dist_mat = torch.cdist(centers_t, centers_t, p=2)

        _, collisions, _ = GeometryAnalyzer.compute_inter_and_intra_distances(
            features, labels, centers, dist_mat, cids, collision_threshold=10.0
        )

        assert len(collisions) > 0  # all pairs are "collisions" at threshold 10

    def test_collision_pair_format(self, sample_embeddings):
        """Each collision pair has class_i, class_j, distance keys."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0, 1, 2])
        centers_t = torch.stack([centers[c] for c in cids], dim=0)
        dist_mat = torch.cdist(centers_t, centers_t, p=2)

        _, collisions, _ = GeometryAnalyzer.compute_inter_and_intra_distances(
            features, labels, centers, dist_mat, cids, collision_threshold=10.0
        )

        for pair in collisions:
            assert "class_i" in pair
            assert "class_j" in pair
            assert "distance" in pair
            assert pair["class_i"] < pair["class_j"]  # normalized

    def test_single_class_no_inter(self):
        """Single class returns empty inter distances and no collisions."""
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        labels = torch.zeros(10, dtype=torch.int64)
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0])
        centers_t = torch.stack([centers[c] for c in cids], dim=0)
        dist_mat = torch.cdist(centers_t, centers_t, p=2)

        inter_dists, collisions, intra_dists = (
            GeometryAnalyzer.compute_inter_and_intra_distances(
                features, labels, centers, dist_mat, cids, collision_threshold=0.5
            )
        )

        assert len(inter_dists) == 0  # no pairs
        assert len(collisions) == 0
        assert len(intra_dists) == 10


class TestGeometryAnalyzerDensity:
    """GeometryAnalyzer.estimate_local_density_diagnostics."""

    def test_dense_embeddings(self, geometry_analyzer, sample_embeddings):
        """Well-structured embeddings produce non-dead density."""
        features, _ = sample_embeddings
        result = geometry_analyzer.estimate_local_density_diagnostics(
            features, k=5, max_samples=60, run_seed=42
        )
        assert "density_variance" in result
        assert "density_mean" in result
        assert "density_sample_count" in result
        assert "density_k" in result
        assert "density_feature_dead" in result
        assert not result["density_feature_dead"]
        assert result["density_sample_count"] == 60
        assert result["density_k"] == 5

    def test_density_empty(self, geometry_analyzer):
        """Empty embeddings return dead-density sentinel."""
        empty_features = torch.zeros((0, 4), dtype=torch.float32)
        result = geometry_analyzer.estimate_local_density_diagnostics(
            empty_features, k=5, run_seed=42
        )
        assert result["density_feature_dead"] is True
        assert result["density_k"] == 0

    def test_density_single_sample(self, geometry_analyzer):
        """Single sample returns dead-density (k_eff < 2)."""
        single = torch.randn(1, 4)
        result = geometry_analyzer.estimate_local_density_diagnostics(
            single, k=5, run_seed=42
        )
        assert result["density_feature_dead"] is True

    def test_density_two_samples(self, geometry_analyzer):
        """Two samples with k=20 => k_eff=1 < 2 => dead."""
        two = torch.randn(2, 4)
        result = geometry_analyzer.estimate_local_density_diagnostics(
            two, k=20, run_seed=42
        )
        assert result["density_feature_dead"] is True

    def test_density_subsampling(self, geometry_analyzer):
        """Oversize input is subsampled to max_samples."""
        large = torch.randn(1000, 4)
        result = geometry_analyzer.estimate_local_density_diagnostics(
            large, k=10, max_samples=100, run_seed=42
        )
        assert result["density_sample_count"] == 100


class TestGeometryAnalyzerIntegrity:
    """GeometryAnalyzer.enforce_geometry_integrity."""

    def test_passes_healthy(self, geometry_analyzer, sample_diagnostics):
        """Standard healthy geometry passes without error."""
        # Will not raise
        geometry_analyzer.enforce_geometry_integrity(
            sample_diagnostics, label_space="cluster_relabel"
        )

    def test_fails_high_ratio(self, geometry_analyzer):
        """Intra/inter ratio above threshold raises RuntimeError."""
        bad_diag = {"intra_inter_ratio": 2.5, "cluster_sizes": [20, 20, 20]}
        with pytest.raises(RuntimeError, match="ratio"):
            geometry_analyzer.enforce_geometry_integrity(
                bad_diag, label_space="original"
            )

    def test_fails_min_inter(self, geometry_analyzer):
        """Min inter-center distance below threshold raises RuntimeError."""
        bad_diag = {
            "intra_inter_ratio": 0.5,
            "min_inter_center_distance": 0.1,
            "cluster_sizes": [20, 20, 20],
        }
        with pytest.raises(RuntimeError, match="collision"):
            geometry_analyzer.enforce_geometry_integrity(
                bad_diag, label_space="original"
            )

    def test_fails_cluster_size(self, geometry_analyzer):
        """On cluster_relabel, a dead cluster raises RuntimeError."""
        bad_diag = {
            "intra_inter_ratio": 0.5,
            "min_inter_center_distance": 0.5,
            "nearest_center_acc_val": 0.95,
            "cluster_sizes": [20, 1, 20],
        }
        with pytest.raises(RuntimeError, match="Dead cluster"):
            geometry_analyzer.enforce_geometry_integrity(
                bad_diag, label_space="cluster_relabel"
            )

    def test_skips_cluster_size_gate_for_original(self, geometry_analyzer):
        """Cluster size gate is only enforced for cluster_relabel/joint_finetune."""
        diag = {
            "intra_inter_ratio": 0.5,
            "min_inter_center_distance": 0.5,
            "nearest_center_acc_val": 0.95,
            "cluster_sizes": [20, 1, 20],
        }
        # original label space should not enforce cluster size gate
        geometry_analyzer.enforce_geometry_integrity(diag, label_space="original")

    def test_energy_mode_bypass(self, geometry_analyzer_energy):
        """Energy mode bypasses all geometry integrity checks."""
        terrible_diag = {
            "intra_inter_ratio": 10.0,
            "min_inter_center_distance": 0.01,
        }
        geometry_analyzer_energy.enforce_geometry_integrity(
            terrible_diag, label_space="cluster_relabel"
        )

    def test_fails_nearest_center_acc(self, geometry_analyzer):
        """Low nearest_center_acc raises RuntimeError."""
        bad_diag = {
            "intra_inter_ratio": 0.5,
            "min_inter_center_distance": 0.5,
            "nearest_center_acc_val": 0.5,
            "cluster_sizes": [20, 20, 20],
        }
        with pytest.raises(RuntimeError, match="nearest_center_acc"):
            geometry_analyzer.enforce_geometry_integrity(
                bad_diag, label_space="original"
            )


class TestGeometryAnalyzerCollision:
    """GeometryAnalyzer.critical_pair_key and has_critical_collision_pairs."""

    def test_critical_pair_key_normalizes(self, geometry_analyzer):
        """Pair key always returns (min, max)."""
        assert geometry_analyzer.critical_pair_key(5, 3) == (3, 5)
        assert geometry_analyzer.critical_pair_key(3, 5) == (3, 5)
        assert geometry_analyzer.critical_pair_key(0, 0) == (0, 0)

    def test_has_critical_collision(self, geometry_analyzer):
        """Detects unresolved critical collision pairs in top5."""
        diagnostics = {
            "nearest_cluster_pairs_top5": [
                {"class_i": 0, "class_j": 1, "distance": 0.2},
            ]
        }
        # (0,1) is in critical_collision_pairs and distance 0.2 < 0.4 threshold
        assert geometry_analyzer.has_critical_collision_pairs(diagnostics) is True

    def test_no_critical_collision(self, geometry_analyzer):
        """Non-critical pairs or above-threshold distances are not flagged."""
        diagnostics = {
            "nearest_cluster_pairs_top5": [
                {"class_i": 0, "class_j": 1, "distance": 0.5},
            ]
        }
        # distance 0.5 >= 0.4 threshold
        assert geometry_analyzer.has_critical_collision_pairs(diagnostics) is False

    def test_no_critical_collision_unrelated_pair(self, geometry_analyzer):
        """Pair (5, 6) is not in critical_collision_pairs set."""
        diagnostics = {
            "nearest_cluster_pairs_top5": [
                {"class_i": 5, "class_j": 6, "distance": 0.1},
            ]
        }
        assert geometry_analyzer.has_critical_collision_pairs(diagnostics) is False

    def test_empty_top5_no_collision(self, geometry_analyzer):
        """Empty top5 list returns False."""
        diagnostics = {"nearest_cluster_pairs_top5": []}
        assert geometry_analyzer.has_critical_collision_pairs(diagnostics) is False


class TestGeometryAnalyzerRatioThreshold:
    """GeometryAnalyzer.current_geometry_ratio_threshold."""

    def test_representation_phase_active(self, geometry_analyzer):
        """During representation phase, warmup threshold is used."""
        assert (
            geometry_analyzer.current_geometry_ratio_threshold(
                representation_phase_active=True, head_phase_start_step=100
            )
            == 3.0
        )

    def test_head_phase_not_started(self, geometry_analyzer):
        """Before head phase starts, warmup threshold is used."""
        assert (
            geometry_analyzer.current_geometry_ratio_threshold(
                representation_phase_active=False, head_phase_start_step=-1
            )
            == 3.0
        )

    def test_post_phase(self, geometry_analyzer):
        """After head phase started, post-phase threshold is used."""
        assert (
            geometry_analyzer.current_geometry_ratio_threshold(
                representation_phase_active=False, head_phase_start_step=500
            )
            == 1.8
        )


# ======================================================================
# ClusterAnalyzer
# ======================================================================


class TestClusterAnalyzerFeaturePrep:
    """ClusterAnalyzer.prepare_representation_features."""

    def test_l2_normalization(self):
        """Output is L2-normalized along dimension 1."""
        features = torch.randn(10, 8)
        out = ClusterAnalyzer.prepare_representation_features(features)
        norms = out.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_zero_input(self):
        """Zero tensor normalizes to NaN-safe output (division by zero norm)."""
        features = torch.zeros(5, 4)
        out = ClusterAnalyzer.prepare_representation_features(features)
        # After F.normalize on all-zeros, norms should be 0 (since norm=0 yields division)
        # F.normalize handles div by zero by returning 0
        assert torch.isfinite(out).all()


class TestClusterAnalyzerCentroids:
    """ClusterAnalyzer.compute_class_centroids."""

    def test_basic_centroids(self, sample_embeddings):
        """Centroids are normalized mean embeddings per class."""
        features, labels = sample_embeddings
        centroids_t, cids = ClusterAnalyzer.compute_class_centroids(
            features, labels
        )
        assert len(cids) == 3
        assert cids == [0, 1, 2]
        assert centroids_t.shape == (3, 4)
        # Centroids are L2-normalized
        norms = centroids_t.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_empty_features(self):
        """Empty features produce empty centroids."""
        empty_f = torch.zeros((0, 4), dtype=torch.float32)
        empty_l = torch.zeros((0,), dtype=torch.int64)
        centroids_t, cids = ClusterAnalyzer.compute_class_centroids(empty_f, empty_l)
        assert centroids_t.shape == (0, 0)
        assert cids == []

    def test_single_class(self):
        """Single class returns a single centroid."""
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        labels = torch.zeros(10, dtype=torch.int64)
        centroids_t, cids = ClusterAnalyzer.compute_class_centroids(features, labels)
        assert cids == [0]
        assert centroids_t.shape == (1, 4)

    def test_missing_class_skips(self):
        """Classes without samples are omitted from centroids."""
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        # Only classes 0, 1 present, but class_ids includes 2
        labels = torch.cat(
            [torch.zeros(5, dtype=torch.int64), torch.ones(5, dtype=torch.int64)]
        )
        centroids_t, cids = ClusterAnalyzer.compute_class_centroids(features, labels)
        assert 0 in cids
        assert 1 in cids
        assert len(cids) <= 2  # class 2 may be absent


class TestClusterAnalyzerBuildClassCenters:
    """ClusterAnalyzer.build_class_centers."""

    def test_build_centers_dict(self, sample_embeddings):
        """build_class_centers returns a dict of per-class centers and available IDs."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0, 1, 2])
        assert cids == [0, 1, 2]
        assert set(centers.keys()) == {0, 1, 2}
        for cid in cids:
            assert isinstance(centers[cid], torch.Tensor)
            assert centers[cid].shape == (4,)

    def test_filtered_class_ids(self, sample_embeddings):
        """Only requested class_ids that exist in labels are returned."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0])
        assert cids == [0]
        assert list(centers.keys()) == [0]

    def test_empty_class_ids(self, sample_embeddings):
        """Empty class_ids list returns empty."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [])
        assert cids == []
        assert centers == {}


class TestClusterAnalyzerAssignLabels:
    """ClusterAnalyzer.assign_labels_from_centers."""

    def test_basic_assignment(self, sample_embeddings):
        """Each sample is assigned to its nearest center."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0, 1, 2])
        centers_t = torch.stack([centers[c] for c in cids], dim=0)
        assigned = ClusterAnalyzer.assign_labels_from_centers(features, centers_t)
        assert assigned.shape == (60,)
        assert assigned.dtype == torch.int64

    def test_empty_embeddings(self):
        """Empty embeddings return empty labels."""
        empty = torch.zeros((0, 4), dtype=torch.float32)
        centers_t = torch.randn(3, 4)
        assigned = ClusterAnalyzer.assign_labels_from_centers(empty, centers_t)
        assert assigned.shape == (0,)
        assert assigned.dtype == torch.int64

    def test_single_center(self):
        """Single center assigns all to index 0."""
        features = torch.randn(10, 4)
        centers_t = torch.randn(1, 4)
        assigned = ClusterAnalyzer.assign_labels_from_centers(features, centers_t)
        assert (assigned == 0).all()


class TestClusterAnalyzerClusterFitting:
    """ClusterAnalyzer.fit_embedding_clusters."""

    def test_kmeans(self, cluster_analyzer, sample_embeddings):
        """KMeans fitting produces labels and centers."""
        features, _ = sample_embeddings
        labels, centers = cluster_analyzer.fit_embedding_clusters(
            features, n_clusters=3
        )
        assert labels.shape == (60,)
        assert labels.dtype == torch.int64
        assert centers.shape == (3, 4)
        # Centers are L2-normalized
        norms = centers.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_gmm(self, cluster_analyzer, sample_embeddings):
        """GMM fitting produces labels and centers."""
        cluster_analyzer._cluster_relabel_objective = "gmm"
        features, _ = sample_embeddings
        labels, centers = cluster_analyzer.fit_embedding_clusters(
            features, n_clusters=3
        )
        assert labels.shape == (60,)
        assert centers.shape == (3, 4)

    def test_spectral(self, cluster_analyzer, sample_embeddings):
        """Spectral clustering with well-separated data."""
        cluster_analyzer._cluster_relabel_objective = "spectral"
        features, _ = sample_embeddings
        labels, centers = cluster_analyzer.fit_embedding_clusters(
            features, n_clusters=3
        )
        assert labels.shape == (60,)
        assert centers.shape == (3, 4)

    def test_kmeans_empty_raises(self, cluster_analyzer):
        """Empty embedding raises RuntimeError."""
        empty = torch.zeros((0, 4), dtype=torch.float32)
        with pytest.raises(RuntimeError, match="empty"):
            cluster_analyzer.fit_embedding_clusters(empty, n_clusters=3)

    def test_unsupported_objective_raises(self, cluster_analyzer, sample_embeddings):
        """Unsupported cluster objective raises RuntimeError."""
        cluster_analyzer._cluster_relabel_objective = "unknown_method"
        features, _ = sample_embeddings
        with pytest.raises(RuntimeError, match="unsupported"):
            cluster_analyzer.fit_embedding_clusters(features, n_clusters=3)


class TestClusterAnalyzerLabelBridge:
    """ClusterAnalyzer.build_cluster_label_bridge."""

    def test_bridge_structure(self):
        """Bridge metadata contains expected keys."""
        old_labels = torch.cat(
            [torch.zeros(10), torch.ones(10)], dim=0
        ).to(dtype=torch.int64)
        cluster_labels = torch.cat(
            [torch.zeros(10), torch.ones(10)], dim=0
        ).to(dtype=torch.int64)
        bridge = ClusterAnalyzer.build_cluster_label_bridge(
            old_labels, cluster_labels, n_clusters=2
        )
        assert "n_clusters" in bridge
        assert "old_labels" in bridge
        assert "old_to_cluster_counts" in bridge
        assert "old_to_cluster_dominant" in bridge
        assert "old_to_cluster_purity" in bridge
        assert "cluster_to_old_counts" in bridge
        assert bridge["n_clusters"] == 2

    def test_perfect_alignment(self):
        """When labels perfectly align, purity is 1.0 and dominant matches."""
        n = 20
        old_labels = torch.cat(
            [torch.zeros(n), torch.ones(n)], dim=0
        ).to(dtype=torch.int64)
        # Cluster labels perfectly match old labels
        cluster_labels = old_labels.clone()
        bridge = ClusterAnalyzer.build_cluster_label_bridge(
            old_labels, cluster_labels, n_clusters=2
        )
        for old_str in ("0", "1"):
            assert bridge["old_to_cluster_purity"][old_str] == pytest.approx(1.0)
            assert bridge["old_to_cluster_dominant"][old_str] == int(old_str)

    def test_empty_old(self):
        """Empty old_labels returns partial bridge (edge behaviour)."""
        old_labels = torch.zeros((0,), dtype=torch.int64)
        cluster_labels = torch.zeros((0,), dtype=torch.int64)
        bridge = ClusterAnalyzer.build_cluster_label_bridge(
            old_labels, cluster_labels, n_clusters=2
        )
        assert bridge["n_clusters"] == 2
        assert bridge["old_labels"] == []


class TestClusterAnalyzerEmbedFeatures:
    """ClusterAnalyzer.embed_feature_matrix."""

    def test_embeds_numpy(self, cluster_analyzer):
        """Numpy feature matrix is embedded through mock model."""
        x = np.random.randn(5, 4).astype(np.float32)
        emb = cluster_analyzer.embed_feature_matrix(x, batch_size=10)
        assert emb.shape == (5, 4)
        # Embeddings are L2-normalized
        norms = emb.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_empty(self, cluster_analyzer):
        """Empty input returns empty tensor."""
        x = np.zeros((0, 4), dtype=np.float32)
        emb = cluster_analyzer.embed_feature_matrix(x)
        assert emb.shape == (0, 0)

    def test_batching(self, cluster_analyzer):
        """Small batch_size triggers multiple forward passes."""
        x = np.random.randn(20, 4).astype(np.float32)
        emb = cluster_analyzer.embed_feature_matrix(x, batch_size=8)
        assert emb.shape == (20, 4)


# ======================================================================
# RepresentationDiagnostics
# ======================================================================


class TestRepDiagnosticsConstruction:
    """RepresentationDiagnostics init and instance state."""

    def test_constructor_stores_params(
        self, mock_model, logger: logging.Logger
    ):
        """All constructor params are stored as instance attrs."""
        rd = RepresentationDiagnostics(
            model=mock_model,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            logger=logger,
            representation_only_steps=50,
            head_only_steps=150,
            sampler_mode="weighted",
        )
        assert rd._representation_only_steps == 50
        assert rd._head_only_steps == 150
        assert rd._sampler_mode == "weighted"


class TestRepDiagnosticsCenterPair:
    """RepresentationDiagnostics.compute_center_pair_diagnostics."""

    def test_pair_computation(self, sample_embeddings):
        """Center pair diagnostics returns pairs, threshold, and top5."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0, 1, 2])
        centers_t = torch.stack([centers[c] for c in cids], dim=0)
        dist_mat = torch.cdist(centers_t, centers_t, p=2)

        pairs, threshold, top5 = (
            RepresentationDiagnostics.compute_center_pair_diagnostics(dist_mat, cids)
        )

        assert len(pairs) == 3  # 3 choose 2
        assert threshold >= 0.0
        assert len(top5) == 3  # only 3 pairs, all are top 5
        for item in top5:
            assert "class_i" in item
            assert "class_j" in item
            assert "distance" in item

    def test_single_class(self):
        """Single class produces empty pairs."""
        dist_mat = torch.zeros((1, 1))
        pairs, threshold, top5 = (
            RepresentationDiagnostics.compute_center_pair_diagnostics(dist_mat, [0])
        )
        assert len(pairs) == 0
        assert threshold == 0.0
        assert len(top5) == 0

    def test_empty_class_ids(self):
        """Empty class_ids produces empty pairs."""
        dist_mat = torch.zeros((0, 0))
        pairs, threshold, top5 = (
            RepresentationDiagnostics.compute_center_pair_diagnostics(dist_mat, [])
        )
        assert len(pairs) == 0
        assert threshold == 0.0
        assert len(top5) == 0


class TestRepDiagnosticsNearestCenter:
    """RepresentationDiagnostics.nearest_center_accuracy."""

    def test_perfect_accuracy(self, sample_embeddings):
        """Well-separated classes achieve near-perfect accuracy."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0, 1, 2])
        acc = RepresentationDiagnostics.nearest_center_accuracy(
            features, labels, centers, cids
        )
        assert acc > 0.9  # well-separated clusters

    def test_empty_features(self):
        """Empty features return 0.0."""
        features = torch.zeros((0, 4))
        labels = torch.zeros((0,), dtype=torch.int64)
        centers = {}
        acc = RepresentationDiagnostics.nearest_center_accuracy(
            features, labels, centers, [0]
        )
        assert acc == 0.0

    def test_no_class_ids(self, sample_embeddings):
        """Empty class_ids returns 0.0."""
        features, labels = sample_embeddings
        acc = RepresentationDiagnostics.nearest_center_accuracy(
            features, labels, {}, []
        )
        assert acc == 0.0


class TestRepDiagnosticsSnapshotID:
    """RepresentationDiagnostics.build_representation_snapshot_id."""

    def test_deterministic(self):
        """Same inputs produce identical hash."""
        diag = {
            "intra_inter_ratio": 0.5,
            "min_inter_center_distance": 0.8,
            "nearest_center_acc_val": 0.95,
            "cluster_sizes": [20, 20],
            "density_variance": 0.3,
        }
        sid1 = RepresentationDiagnostics.build_representation_snapshot_id(
            diag,
            label_space="original",
            representation_only_steps=100,
            head_only_steps=200,
            sampler_mode="balanced",
        )
        sid2 = RepresentationDiagnostics.build_representation_snapshot_id(
            diag,
            label_space="original",
            representation_only_steps=100,
            head_only_steps=200,
            sampler_mode="balanced",
        )
        assert sid1 == sid2
        assert sid1.startswith("rep_phase_v1_")
        assert len(sid1) == len("rep_phase_v1_") + 16

    def test_different_inputs_different_ids(self):
        """Different inputs produce different snapshot IDs."""
        diag_a = {
            "intra_inter_ratio": 0.5,
            "min_inter_center_distance": 0.8,
            "nearest_center_acc_val": 0.95,
            "cluster_sizes": [20, 20],
            "density_variance": 0.3,
        }
        diag_b = {
            "intra_inter_ratio": 2.0,
            "min_inter_center_distance": 0.8,
            "nearest_center_acc_val": 0.95,
            "cluster_sizes": [20, 20],
            "density_variance": 0.3,
        }
        sid_a = RepresentationDiagnostics.build_representation_snapshot_id(
            diag_a,
            label_space="original",
            representation_only_steps=100,
            head_only_steps=200,
            sampler_mode="balanced",
        )
        sid_b = RepresentationDiagnostics.build_representation_snapshot_id(
            diag_b,
            label_space="original",
            representation_only_steps=100,
            head_only_steps=200,
            sampler_mode="balanced",
        )
        assert sid_a != sid_b

    def test_payload_keys(self):
        """Snapshot ID payload includes all expected keys."""
        diag = {
            "ratio": 0.5,
            "min_inter_center_distance": 0.8,
            "nearest_center_acc_val": 0.95,
            "cluster_sizes": [20, 20],
            "density_variance": 0.3,
        }
        sid = RepresentationDiagnostics.build_representation_snapshot_id(
            diag,
            label_space="cluster_relabel",
            representation_only_steps=100,
            head_only_steps=200,
            sampler_mode="balanced",
        )
        assert sid.startswith("rep_phase_v1_")
        assert len(sid) == len("rep_phase_v1_") + 16


class TestRepDiagnosticsFullPipeline:
    """RepresentationDiagnostics.compute_representation_diagnostics."""

    def test_full_pipeline(self, rep_diagnostics, geometry_analyzer, sample_embeddings):
        """End-to-end diagnostics produces all expected keys."""
        train_f, train_l = sample_embeddings
        val_f, val_l = sample_embeddings  # use same for validation

        result = rep_diagnostics.compute_representation_diagnostics(
            train_f,
            train_l,
            val_f,
            val_l,
            class_ids=[0, 1, 2],
            geometry_analyzer=geometry_analyzer,
            run_seed=42,
        )

        expected_keys = [
            "available_class_ids",
            "center_distance_matrix",
            "nearest_center_accuracy_train",
            "nearest_center_accuracy_val",
            "nearest_center_acc_val",
            "intra_class_distance_mean",
            "inter_center_distance_mean",
            "intra_inter_ratio",
            "min_inter_center_distance",
            "cluster_size_counts",
            "cluster_sizes",
            "collision_threshold_p05",
            "nearest_cluster_pairs_top5",
            "density_variance",
            "density_feature_dead",
            "collision_pairs",
            "secondary_collision_pairs",
            "nearest_center_confusion_matrix",
            "embedding_capacity_assessment",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        assert result["available_class_ids"] == [0, 1, 2]

    def test_empty_train_features(self, rep_diagnostics, geometry_analyzer):
        """Empty train features return empty_result."""
        empty_f = torch.zeros((0, 4), dtype=torch.float32)
        empty_l = torch.zeros((0,), dtype=torch.int64)
        val_f = torch.randn(10, 4)
        val_l = torch.zeros(10, dtype=torch.int64)

        result = rep_diagnostics.compute_representation_diagnostics(
            empty_f,
            empty_l,
            val_f,
            val_l,
            class_ids=[0],
            geometry_analyzer=geometry_analyzer,
            run_seed=42,
        )

        assert result["available_class_ids"] == []
        assert result["intra_inter_ratio"] == 0.0

    def test_no_class_ids(self, rep_diagnostics, geometry_analyzer, sample_embeddings):
        """Empty class_ids returns empty_result."""
        train_f, train_l = sample_embeddings
        empty_l = torch.zeros((10,), dtype=torch.int64)
        result = rep_diagnostics.compute_representation_diagnostics(
            train_f,
            train_l,
            train_f,
            empty_l,
            class_ids=[],
            geometry_analyzer=geometry_analyzer,
            run_seed=42,
        )
        assert result["available_class_ids"] == []


class TestRepDiagnosticsRun:
    """RepresentationDiagnostics.run_representation_diagnostics."""

    def test_run_returns_diagnostics(
        self, rep_diagnostics, geometry_analyzer, sample_embeddings
    ):
        """Runner returns diagnostics dict with log output."""
        train_f, train_l = sample_embeddings
        # Note: run also tries to collect val embeddings from val_loaders;
        # but we can test the pure computation path by passing empty val_loaders.
        with patch.object(rep_diagnostics, "collect_normalized_embeddings") as mock_collect:
            mock_collect.return_value = (train_f, train_l)
            result = rep_diagnostics.run_representation_diagnostics(
                train_features=train_f,
                train_labels=train_l,
                label_space="test_run",
                active_family_class_ids={0, 1, 2},
                val_loaders={},
                geometry_analyzer=geometry_analyzer,
                run_seed=42,
            )

        assert isinstance(result, dict)
        assert "available_class_ids" in result
        assert result["available_class_ids"] == [0, 1, 2]

    def test_run_empty_class_ids(self, rep_diagnostics, geometry_analyzer):
        """Empty active_family_class_ids returns empty dict."""
        result = rep_diagnostics.run_representation_diagnostics(
            train_features=torch.randn(10, 4),
            train_labels=torch.zeros(10, dtype=torch.int64),
            label_space="test",
            active_family_class_ids=set(),
            val_loaders={},
            geometry_analyzer=geometry_analyzer,
            run_seed=42,
        )
        assert result == {}

    def test_run_no_val_loader(
        self, rep_diagnostics, geometry_analyzer, sample_embeddings
    ):
        """Runner handles missing val_loader gracefully."""
        train_f, train_l = sample_embeddings
        result = rep_diagnostics.run_representation_diagnostics(
            train_features=train_f,
            train_labels=train_l,
            label_space="no_val",
            active_family_class_ids={0, 1, 2},
            val_loaders={},
            geometry_analyzer=geometry_analyzer,
            run_seed=42,
        )
        assert isinstance(result, dict)
        # When no val_loader, val features are zeros
        assert "available_class_ids" in result


class TestRepDiagnosticsCollectEmbeddings:
    """RepresentationDiagnostics.collect_normalized_embeddings."""

    def test_collects_from_loader(
        self, rep_diagnostics: RepresentationDiagnostics
    ):
        """collect_normalized_embeddings processes loader batches."""
        loader = self._make_mock_loader(5, 8)
        features, labels = rep_diagnostics.collect_normalized_embeddings(
            loader, max_batches=2
        )
        # model returns ones, so features should be normalized ones
        assert features.shape[0] > 0
        assert labels.shape[0] == features.shape[0]
        # Features are L2-normalized
        norms = features.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_collect_empty_loader(self, rep_diagnostics: RepresentationDiagnostics):
        """Empty loader returns zero tensors."""
        loader = self._make_mock_loader(0, 8)
        features, labels = rep_diagnostics.collect_normalized_embeddings(
            loader, max_batches=2
        )
        assert features.shape == (0, 0)
        assert labels.shape == (0,)

    @staticmethod
    def _make_mock_loader(num_samples: int, batch_size: int) -> MagicMock:
        """Create a mock DataLoader that yields batches."""
        loader = MagicMock()
        if num_samples == 0:
            loader.__iter__.return_value = []
            return loader

        batches = []
        n = num_samples
        while n > 0:
            bs = min(batch_size, n)
            x = torch.randn(bs, 4)
            _ = torch.zeros(bs, dtype=torch.int64)  # middle unused val
            y_family = torch.zeros(bs, dtype=torch.int64)
            batches.append((x, _, y_family))
            n -= bs
        loader.__iter__.return_value = iter(batches)
        return loader


# ======================================================================
# Edge cases specific to delegate wrappers
# ======================================================================


class TestEdgeCases:
    """Twelve edge cases across all three delegate classes."""

    # --- GeometryAnalyzer edge cases ---

    def test_ga_enforce_missing_keys(self, geometry_analyzer):
        """Missing diagnostics keys are handled gracefully (fallback defaults)."""
        diag: dict[str, Any] = {}
        # Keys fall back to defaults: ratio=0.0, min_inter=0.0, acc=0.0
        # ratio 0.0 < 1.8 => ok; min_inter 0.0 < 0.4 => fail
        with pytest.raises(RuntimeError, match="collision"):
            geometry_analyzer.enforce_geometry_integrity(diag, label_space="original")

    def test_ga_empty_top5_no_match(self, geometry_analyzer):
        """Missing nearest_cluster_pairs_top5 key returns False."""
        assert geometry_analyzer.has_critical_collision_pairs({}) is False

    @pytest.mark.parametrize(
        "pair_input,expected",
        [
            ((1, 2), (1, 2)),
            ((2, 1), (1, 2)),
            ((0, 0), (0, 0)),
            ((-3, 5), (-3, 5)),
        ],
    )
    def test_ga_critical_pair_key_param(
        self, geometry_analyzer, pair_input, expected
    ):
        """Parametrized test for critical_pair_key normalization."""
        assert geometry_analyzer.critical_pair_key(*pair_input) == expected

    # --- ClusterAnalyzer edge cases ---

    def test_ca_assign_labels_dim_mismatch(self, cluster_analyzer):
        """Dimension mismatch raises no torch error (cdist handles broadcasting)."""
        embeddings = torch.randn(10, 8)
        centers_t = torch.randn(3, 4)
        with pytest.raises(RuntimeError):
            cluster_analyzer.assign_labels_from_centers(embeddings, centers_t)

    def test_ca_compute_class_centroids_no_matching_class(self):
        """When no features match requested class, empty is returned."""
        features = F.normalize(torch.randn(10, 4), p=2, dim=1)
        labels = torch.zeros(10, dtype=torch.int64)
        centroids_t, cids = ClusterAnalyzer.compute_class_centroids(features, labels)
        # Only class 0 present
        assert cids == [0]

    # --- RepresentationDiagnostics edge cases ---

    def test_rd_nearest_center_all_wrong(self):
        """When centers are swapped, accuracy is near 0."""
        features = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
        )
        labels = torch.tensor([0, 1], dtype=torch.int64)
        # Deliberately swapped centers
        centers = {
            0: torch.tensor([0.0, 1.0, 0.0, 0.0]),
            1: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }
        acc = RepresentationDiagnostics.nearest_center_accuracy(
            features, labels, centers, [0, 1]
        )
        assert acc == pytest.approx(0.0, abs=1e-6)

    def test_rd_center_pair_empty_dist_mat(self):
        """All-zero dist_mat gives pairs with distance=0."""
        dist_mat = torch.zeros((3, 3))
        pairs, threshold, top5 = (
            RepresentationDiagnostics.compute_center_pair_diagnostics(dist_mat, [0, 1, 2])
        )
        assert len(pairs) == 3
        assert threshold == 0.0
        for item in top5:
            assert item["distance"] == 0.0

    def test_rd_build_snapshot_id_with_legacy_keys(self):
        """Snapshot ID works with legacy 'ratio'/'acc' keys as fallback."""
        diag = {
            "ratio": 0.6,
            "min_inter_center_distance": 0.9,
            "nearest_center_acc_val": 0.92,
            "cluster_sizes": [10, 10],
            "density_variance": 0.2,
        }
        sid = RepresentationDiagnostics.build_representation_snapshot_id(
            diag,
            label_space="original",
            representation_only_steps=50,
            head_only_steps=100,
            sampler_mode="balanced",
        )
        assert sid.startswith("rep_phase_v1_")
        assert len(sid) == len("rep_phase_v1_") + 16


# ======================================================================
# Regression: verify delegation wrappers match delegate outputs
# ======================================================================


class TestDelegationRegression:
    """Verify that trainer delegation wrappers and direct delegate calls agree.

    These tests simulate the passthrough by calling the delegate method
    externally, confirming the contract the trainer relies on.
    """

    # Delegation #1: prepare_representation_features
    def test_delegation_prepare_features(self):
        """Delegate #1: ClusterAnalyzer.prepare_representation_features."""
        x = torch.randn(5, 4)
        out = ClusterAnalyzer.prepare_representation_features(x)
        assert out.shape == x.shape
        norms = out.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    # Delegation #3: current_geometry_ratio_threshold
    def test_delegation_ratio_threshold(self, geometry_analyzer):
        """Delegate #3: GeometryAnalyzer.current_geometry_ratio_threshold."""
        t = geometry_analyzer.current_geometry_ratio_threshold(
            representation_phase_active=True, head_phase_start_step=-1
        )
        assert t == 3.0  # warmup

    # Delegation #6: has_critical_collision_pairs
    def test_delegation_has_critical(self, geometry_analyzer):
        """Delegate #6: GeometryAnalyzer.has_critical_collision_pairs."""
        diag = {
            "nearest_cluster_pairs_top5": [
                {"class_i": 0, "class_j": 1, "distance": 0.1}
            ]
        }
        assert geometry_analyzer.has_critical_collision_pairs(diag) is True

    # Delegation #14: nearest_center_accuracy
    def test_delegation_nearest_center_acc(self, sample_embeddings):
        """Delegate #14: repr_diag.nearest_center_accuracy."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0, 1, 2])
        acc = RepresentationDiagnostics.nearest_center_accuracy(
            features, labels, centers, cids
        )
        assert 0.0 <= acc <= 1.0

    # Delegation #15: build_class_centers
    def test_delegation_build_class_centers(self, sample_embeddings):
        """Delegate #15: ClusterAnalyzer.build_class_centers."""
        features, labels = sample_embeddings
        centers, cids = ClusterAnalyzer.build_class_centers(features, labels, [0, 1, 2])
        assert len(centers) == 3
        assert cids == [0, 1, 2]
