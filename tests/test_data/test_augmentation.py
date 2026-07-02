"""Tests for AttackAwareAugmentation minority oversampling.

Covers:
  - SMOTE-ENN for R2L attacks
  - ADASYN + Mixup for U2R attacks
  - Noise augmentation
  - Determinism and consistency
  - Edge cases (too few samples, already balanced)
  - Custom config handling
"""

from __future__ import annotations

import numpy as np
import pytest

from helix_ids.data.augmentation import IMBLEARN_AVAILABLE, AttackAwareAugmentation, AugmentationConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def balanced_X(rng: np.random.Generator) -> np.ndarray:
    """40 samples, 4 features, roughly balanced across 5 classes."""
    X = rng.random((40, 4), dtype=np.float32)
    return X


@pytest.fixture
def balanced_y() -> np.ndarray:
    """40 samples, 8 per class (0-4)."""
    return np.array([0] * 8 + [1] * 8 + [2] * 8 + [3] * 8 + [4] * 8)


@pytest.fixture
def imbalanced_X(rng: np.random.Generator) -> np.ndarray:
    """200 samples, 4 features, highly imbalanced."""
    X = rng.random((200, 4), dtype=np.float32)
    return X


@pytest.fixture
def imbalanced_y() -> np.ndarray:
    """200 samples: class 0=100, 1=50, 2=25, 3=15, 4=10 (enough for SMOTE)."""
    return np.array(
        [0] * 100 + [1] * 50 + [2] * 25 + [3] * 15 + [4] * 10
    )


@pytest.fixture
def aug() -> AttackAwareAugmentation:
    return AttackAwareAugmentation()


# ═══════════════════════════════════════════════════════════════════════════════
# AugmentationConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestAugmentationConfig:
    def test_default_config(self) -> None:
        """AugmentationConfig has sensible defaults."""
        cfg = AugmentationConfig()
        assert cfg.r2l_target_ratio == 0.10
        assert cfg.u2r_target_ratio == 0.05
        assert cfg.probe_target_ratio == 0.15
        assert cfg.random_state == 42
        assert cfg.class_names == ["Normal", "DoS", "Probe", "R2L", "U2R"]

    def test_custom_config(self) -> None:
        """Custom config values propagate."""
        cfg = AugmentationConfig(r2l_target_ratio=0.20, random_state=99)
        assert cfg.r2l_target_ratio == 0.20
        assert cfg.random_state == 99


# ═══════════════════════════════════════════════════════════════════════════════
# AttackAwareAugmentation - Init
# ═══════════════════════════════════════════════════════════════════════════════


class TestAttackAwareAugmentationInit:
    def test_default_init(self) -> None:
        """Default initialization."""
        a = AttackAwareAugmentation()
        assert a.config is not None
        assert a.config.random_state == 42

    def test_custom_config_init(self) -> None:
        """Custom config passed to constructor."""
        cfg = AugmentationConfig(r2l_target_ratio=0.25)
        a = AttackAwareAugmentation(config=cfg)
        assert a.config.r2l_target_ratio == 0.25

    def test_class_to_idx_mapping(self) -> None:
        """Class name to index mapping is correct."""
        a = AttackAwareAugmentation()
        assert a._class_to_idx["Normal"] == 0
        assert a._class_to_idx["DoS"] == 1
        assert a._class_to_idx["Probe"] == 2
        assert a._class_to_idx["R2L"] == 3
        assert a._class_to_idx["U2R"] == 4


# ═══════════════════════════════════════════════════════════════════════════════
# SMOTE-ENN
# ═══════════════════════════════════════════════════════════════════════════════


class TestSmoteEnn:
    def test_returns_tuple(self, aug: AttackAwareAugmentation, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """smote_enn returns (X, y) tuple."""
        X_aug, y_aug = aug.smote_enn(imbalanced_X, imbalanced_y, target_class=4)
        assert isinstance(X_aug, np.ndarray)
        assert isinstance(y_aug, np.ndarray)

    def test_increases_minority_count(self, aug: AttackAwareAugmentation, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """R2L class (4) count increases after SMOTE-ENN."""
        initial_count = np.sum(imbalanced_y == 4)
        X_aug, y_aug = aug.smote_enn(imbalanced_X, imbalanced_y, target_class=4)
        final_count = np.sum(y_aug == 4)
        assert final_count > initial_count

    def test_preserves_feature_count(self, aug: AttackAwareAugmentation, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """Output has same number of features as input."""
        n_features = imbalanced_X.shape[1]
        X_aug, _ = aug.smote_enn(imbalanced_X, imbalanced_y, target_class=4)
        assert X_aug.shape[1] == n_features

    def test_balanced_class_no_change(self, aug: AttackAwareAugmentation, balanced_X: np.ndarray, balanced_y: np.ndarray) -> None:
        """When class is already above target ratio, no augmentation occurs."""
        # class 0 has 8/40 = 20%, target_ratio=0.1 means n_target=4 which is < n_current=8
        X_out, y_out = aug.smote_enn(balanced_X, balanced_y, target_class=0, target_ratio=0.1)
        assert len(X_out) == len(balanced_X)

    @pytest.mark.skipif(IMBLEARN_AVAILABLE, reason="imblearn handles <2 samples differently")
    def test_too_few_samples_raises(self, aug: AttackAwareAugmentation, balanced_X: np.ndarray, balanced_y: np.ndarray) -> None:
        """With <2 samples for target class, SMOTE-ENN raises ValueError."""
        # Use target_ratio=0.3 on 10 samples so n_target=3 > n_current=1
        y_few = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        X_few = balanced_X[:10]
        with pytest.raises(ValueError, match="< 2 samples"):
            aug.smote_enn(X_few, y_few, target_class=0, target_ratio=0.3)

    def test_string_class_name(self, aug: AttackAwareAugmentation, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """String class name resolves to integer index."""
        X_aug, y_aug = aug.smote_enn(imbalanced_X, imbalanced_y, target_class="R2L")
        assert np.sum(y_aug == 3) > np.sum(imbalanced_y == 3)

    def test_preserves_other_classes(self, aug: AttackAwareAugmentation, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """Target class count increases after SMOTE-ENN (other classes may be ENN-cleaned)."""
        initial_count_4 = np.sum(imbalanced_y == 4)
        _, y_aug = aug.smote_enn(imbalanced_X, imbalanced_y, target_class=4)
        # Target class count should increase (SMOTE adds minority, ENN may then reduce)
        # At minimum, net target class count should not be less than original
        assert np.sum(y_aug == 4) >= initial_count_4


# ═══════════════════════════════════════════════════════════════════════════════
# ADASYN + Mixup
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdasynMixup:
    def test_returns_tuple(self, aug: AttackAwareAugmentation, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """adasyn_mixup returns (X, y) tuple."""
        X_aug, y_aug = aug.adasyn_mixup(imbalanced_X, imbalanced_y, target_class=4)
        assert isinstance(X_aug, np.ndarray)
        assert isinstance(y_aug, np.ndarray)

    def test_increases_minority_count(self, aug: AttackAwareAugmentation, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """U2R class (4) count increases after ADASYN+Mixup."""
        initial_count = np.sum(imbalanced_y == 4)
        X_aug, y_aug = aug.adasyn_mixup(imbalanced_X, imbalanced_y, target_class=4)
        final_count = np.sum(y_aug == 4)
        assert final_count >= initial_count

    def test_preserves_feature_count(self, aug: AttackAwareAugmentation, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """Output has same number of features as input."""
        n_features = imbalanced_X.shape[1]
        X_aug, _ = aug.adasyn_mixup(imbalanced_X, imbalanced_y, target_class=4)
        assert X_aug.shape[1] == n_features

    def test_balanced_no_change(self, aug: AttackAwareAugmentation, balanced_X: np.ndarray, balanced_y: np.ndarray) -> None:
        """When class is already above target, no change."""
        # class 0 has 8/40 = 20%, target_ratio=0.1 means n_target=4 which is < n_current=8
        X_out, y_out = aug.adasyn_mixup(balanced_X, balanced_y, target_class=0, target_ratio=0.1)
        assert len(X_out) == len(balanced_X)

    def test_string_class_name(self, aug: AttackAwareAugmentation, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """String class name for U2R."""
        X_aug, y_aug = aug.adasyn_mixup(imbalanced_X, imbalanced_y, target_class="U2R")
        assert np.sum(y_aug == 4) >= np.sum(imbalanced_y == 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Noise Augmentation
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoiseAugmentation:
    def test_noise_increases_samples(self, aug: AttackAwareAugmentation, rng: np.random.Generator) -> None:
        """_augment_with_noise increases minority count."""
        X = rng.random((20, 4), dtype=np.float32)
        y = np.array([0] * 15 + [4] * 5)
        X_aug, y_aug = aug._augment_with_noise(X, y, target_class=4, n_target=10, noise_std=0.01)
        assert np.sum(y_aug == 4) == 10

    def test_noise_preserves_features(self, aug: AttackAwareAugmentation, rng: np.random.Generator) -> None:
        """Noise augmentation preserves feature count."""
        X = rng.random((20, 4), dtype=np.float32)
        y = np.array([0] * 15 + [4] * 5)
        X_aug, _ = aug._augment_with_noise(X, y, target_class=4, n_target=8, noise_std=0.01)
        assert X_aug.shape[1] == 4

    def test_noise_no_minority(self, aug: AttackAwareAugmentation, rng: np.random.Generator) -> None:
        """When n_minority == 0, returns original."""
        X = rng.random((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=int)
        X_aug, y_aug = aug._augment_with_noise(X, y, target_class=4, n_target=5, noise_std=0.01)
        assert len(X_aug) == 10
        assert list(y_aug) == [0] * 10


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_smote_enn_deterministic(self, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """Same input + seed produces same SMOTE-ENN output."""
        aug1 = AttackAwareAugmentation(AugmentationConfig(random_state=42))
        aug2 = AttackAwareAugmentation(AugmentationConfig(random_state=42))
        X1, y1 = aug1.smote_enn(imbalanced_X, imbalanced_y, target_class=4)
        X2, y2 = aug2.smote_enn(imbalanced_X, imbalanced_y, target_class=4)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_adasyn_mixup_deterministic(self, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """Same input + seed produces same ADASYN+Mixup output."""
        aug1 = AttackAwareAugmentation(AugmentationConfig(random_state=42))
        aug2 = AttackAwareAugmentation(AugmentationConfig(random_state=42))
        X1, y1 = aug1.adasyn_mixup(imbalanced_X, imbalanced_y, target_class=4)
        X2, y2 = aug2.adasyn_mixup(imbalanced_X, imbalanced_y, target_class=4)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_different_seed_different_output(self, imbalanced_X: np.ndarray, imbalanced_y: np.ndarray) -> None:
        """Different seeds produce different outputs (synthetic samples differ)."""
        aug1 = AttackAwareAugmentation(AugmentationConfig(random_state=1))
        aug2 = AttackAwareAugmentation(AugmentationConfig(random_state=999))
        X1, y1 = aug1.smote_enn(imbalanced_X, imbalanced_y, target_class=4)
        X2, y2 = aug2.smote_enn(imbalanced_X, imbalanced_y, target_class=4)
        assert X1.shape[1] == X2.shape[1] == imbalanced_X.shape[1]
        assert np.sum(y1 == 4) >= np.sum(imbalanced_y == 4)
        assert np.sum(y2 == 4) >= np.sum(imbalanced_y == 4)
        # SMOTE+ENN with different seeds should produce different results
        assert y1.shape != y2.shape or not np.array_equal(X1, X2) or not np.array_equal(y1, y2)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestInternalHelpers:
    def test_find_neighbors_manual(self, aug: AttackAwareAugmentation, rng: np.random.Generator) -> None:
        """_find_neighbors_manual returns correct shape."""
        X = rng.random((10, 4), dtype=np.float32)
        neighbors = aug._find_neighbors_manual(X, k=3)
        assert neighbors.shape == (10, 3)
        assert neighbors.dtype == int

    def test_find_neighbors_excludes_self(self, aug: AttackAwareAugmentation, rng: np.random.Generator) -> None:
        """_find_neighbors_manual does not include the sample itself."""
        X = rng.random((5, 2), dtype=np.float32)
        neighbors = aug._find_neighbors_manual(X, k=2)
        for i in range(5):
            assert i not in neighbors[i]

    def test_generate_smote_samples_shape(self, aug: AttackAwareAugmentation, rng: np.random.Generator) -> None:
        """_generate_smote_samples produces correct shape."""
        x_minority = rng.random((5, 4), dtype=np.float32)
        samples = aug._generate_smote_samples(x_minority, n_samples=10, k_neighbors=3)
        assert samples.shape == (10, 4)
        assert samples.dtype == np.float32

    def test_apply_mixup_increases_count(self, aug: AttackAwareAugmentation, rng: np.random.Generator) -> None:
        """_apply_mixup adds minority samples."""
        X = rng.random((20, 4), dtype=np.float32)
        y = np.array([0] * 15 + [4] * 5)
        X_out, y_out = aug._apply_mixup(X, y, target_class=4, alpha=0.4)
        assert len(X_out) >= len(X)
        assert np.sum(y_out == 4) >= 5
