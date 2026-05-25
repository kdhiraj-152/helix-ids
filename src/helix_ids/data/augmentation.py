"""
Attack-Aware Data Augmentation for Minority Class Suppression

This module implements specialized augmentation strategies to address the critical
minority class suppression problem identified in our research.
"""

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportPossiblyUnboundVariable=false, reportConstantRedefinition=false, reportAssignmentType=false, reportOptionalMemberAccess=false

import logging
from dataclasses import dataclass, field
from typing import Optional, Union, cast

import numpy as np

logger = logging.getLogger(__name__)

# Try to import imblearn, fall back to custom implementation
try:
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning(
        "imbalanced-learn not available. Using custom implementations. Install with: pip install imbalanced-learn"
    )

try:
    from sklearn.neighbors import NearestNeighbors

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class AugmentationConfig:
    """Configuration for attack-aware augmentation."""

    # R2L augmentation settings (SMOTE-ENN)
    r2l_target_ratio: float = 0.10
    r2l_k_neighbors: int = 5

    # U2R augmentation settings (ADASYN + Mixup)
    u2r_target_ratio: float = 0.05
    u2r_mixup_alpha: float = 0.4
    u2r_noise_std: float = 0.01

    # Probe augmentation settings (Light SMOTE)
    probe_target_ratio: float = 0.15
    probe_k_neighbors: int = 5

    # Class-balanced sampling
    min_samples_per_batch: dict[str, int] = field(
        default_factory=lambda: {
            "Normal": 8,
            "DoS": 8,
            "Probe": 4,
            "R2L": 4,
            "U2R": 4,  # Special handling for extreme minority
        }
    )

    # Class name to index mapping
    class_names: list[str] = field(default_factory=lambda: ["Normal", "DoS", "Probe", "R2L", "U2R"])

    random_state: int = 42


class AttackAwareAugmentation:
    """
    Attack-aware data augmentation to solve minority class suppression.

    Implements specialized augmentation strategies for each attack type:
    - SMOTE-ENN for R2L attacks
    - ADASYN + Mixup for U2R attacks
    - Light SMOTE for Probe attacks
    - Class-balanced batch sampling

    Attributes:
        config: AugmentationConfig with all settings
        random_state: numpy Generator for reproducibility
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize attack-aware augmentation.

        Args:
            config: AugmentationConfig instance. Uses defaults if None.
        """
        self.config = config or AugmentationConfig()
        self.random_state = np.random.default_rng(self.config.random_state)
        self._class_to_idx = {name: idx for idx, name in enumerate(self.config.class_names)}

    def smote_enn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_class: Union[int, str],
        target_ratio: float = 0.10,
        k_neighbors: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE-ENN augmentation for a target class (designed for R2L).

        SMOTE generates synthetic samples via interpolation between nearest neighbors.
        ENN (Edited Nearest Neighbors) removes noisy samples after oversampling.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,)
            target_class: Class index or name to augment
            target_ratio: Target ratio of this class in final dataset
            k_neighbors: Number of neighbors for SMOTE

        Returns:
            Tuple of (X_augmented, y_augmented)
        """
        if isinstance(target_class, str):
            target_class = self._class_to_idx[target_class]

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        n_total = len(y)
        n_target = int(n_total * target_ratio)
        n_current = np.sum(y == target_class)

        if n_current >= n_target:
            return X, y

        if n_current < 2:
            raise ValueError(f"Class {target_class} has < 2 samples. Cannot apply SMOTE.")

        if IMBLEARN_AVAILABLE:
            return self._smote_enn_imblearn(X, y, target_class, n_target, k_neighbors)
        return self._smote_enn_custom(X, y, target_class, n_target, k_neighbors)

    def _smote_enn_imblearn(
        self, X: np.ndarray, y: np.ndarray, target_class: int, n_target: int, k_neighbors: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """SMOTE-ENN using imbalanced-learn library."""
        # Calculate sampling strategy
        class_counts = np.bincount(y.astype(int), minlength=len(self.config.class_names))
        sampling_strategy = {target_class: n_target}

        # Adjust k_neighbors based on available samples
        n_minority = class_counts[target_class]
        k_actual = min(k_neighbors, n_minority - 1)

        if k_actual < 1:
            k_actual = 1

        try:
            smote_enn = SMOTEENN(
                sampling_strategy=sampling_strategy,
                smote=SMOTE(k_neighbors=k_actual, random_state=self.config.random_state),
                enn=EditedNearestNeighbours(n_neighbors=3),
                random_state=self.config.random_state,
            )
            x_res, y_res = cast(tuple[np.ndarray, np.ndarray], smote_enn.fit_resample(X, y))
            return x_res, y_res
        except Exception as e:
            raise RuntimeError(f"SMOTE-ENN failed: {e}") from e

    def _smote_enn_custom(
        self, X: np.ndarray, y: np.ndarray, target_class: int, n_target: int, k_neighbors: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Custom SMOTE-ENN implementation."""
        # Get minority class samples
        minority_mask = y == target_class
        x_minority = X[minority_mask]
        n_minority = len(x_minority)

        # Number of synthetic samples needed
        n_synthetic = n_target - n_minority
        if n_synthetic <= 0:
            return X, y

        # Adjust k_neighbors
        k_actual = min(k_neighbors, n_minority - 1)
        if k_actual < 1:
            k_actual = 1

        # SMOTE: Generate synthetic samples
        synthetic_samples = self._generate_smote_samples(x_minority, n_synthetic, k_actual)

        # Combine original and synthetic
        x_combined = np.vstack([X, synthetic_samples])
        y_combined = np.concatenate([y, np.full(n_synthetic, target_class)])

        # ENN: Clean up noisy samples
        x_cleaned, y_cleaned = self._edited_nearest_neighbors(x_combined, y_combined, n_neighbors=3)

        return x_cleaned, y_cleaned

    def _generate_smote_samples(
        self, x_minority: np.ndarray, n_samples: int, k_neighbors: int
    ) -> np.ndarray:
        """Generate synthetic samples using SMOTE algorithm."""
        n_minority = len(x_minority)
        n_features = x_minority.shape[1]

        # Find k nearest neighbors
        if SKLEARN_AVAILABLE:
            nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
            nn.fit(x_minority)
            _, neighbors_indices = nn.kneighbors(x_minority)
            neighbors_indices = neighbors_indices[:, 1:]  # Exclude self
        else:
            neighbors_indices = self._find_neighbors_manual(x_minority, k_neighbors)

        # Generate synthetic samples
        synthetic = np.zeros((n_samples, n_features), dtype=np.float32)

        for i in range(n_samples):
            # Randomly select a minority sample
            idx = self.random_state.integers(0, n_minority)

            # Randomly select one of its k neighbors
            neighbor_idx = neighbors_indices[idx, self.random_state.integers(0, k_neighbors)]

            # Interpolate
            alpha = self.random_state.random()
            synthetic[i] = x_minority[idx] + alpha * (x_minority[neighbor_idx] - x_minority[idx])

        return synthetic

    def _find_neighbors_manual(self, X: np.ndarray, k: int) -> np.ndarray:
        """Find k nearest neighbors without sklearn."""
        n_samples = len(X)
        neighbors = np.zeros((n_samples, k), dtype=int)

        for i in range(n_samples):
            distances = np.sum((X - X[i]) ** 2, axis=1)
            distances[i] = np.inf  # Exclude self
            neighbors[i] = np.argsort(distances)[:k]

        return neighbors

    def _edited_nearest_neighbors(
        self, X: np.ndarray, y: np.ndarray, n_neighbors: int = 3, protect_minority: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove samples whose class differs from majority of neighbors.

        Args:
            X: Feature matrix
            y: Label array
            n_neighbors: Number of neighbors to consider
            protect_minority: If True, never remove minority class samples
        """
        if SKLEARN_AVAILABLE:
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
            nn.fit(X)
            _, neighbors_indices = nn.kneighbors(X)
            neighbors_indices = neighbors_indices[:, 1:]  # Exclude self
        else:
            neighbors_indices = self._find_neighbors_manual(X, n_neighbors)

        # Identify minority classes (less than 10% of data)
        class_counts = np.bincount(y.astype(int))
        minority_classes = set(np.nonzero(class_counts < len(y) * 0.1)[0])

        # Keep samples where class matches majority of neighbors
        # But protect minority class samples from removal
        keep_mask = np.ones(len(y), dtype=bool)

        for i in range(len(y)):
            # Protect minority class samples
            if protect_minority and int(y[i]) in minority_classes:
                continue

            neighbor_labels = y[neighbors_indices[i]]
            majority_class = np.bincount(neighbor_labels.astype(int)).argmax()
            if y[i] != majority_class:
                keep_mask[i] = False

        return X[keep_mask], y[keep_mask]

    def adasyn_mixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_class: Union[int, str],
        target_ratio: float = 0.05,
        mixup_alpha: float = 0.4,
        noise_std: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply ADASYN + Mixup augmentation for extreme minority (designed for U2R).

        ADASYN adaptively generates more synthetic samples in regions where
        the minority class is harder to learn. Mixup adds diversity through
        convex combinations. Small Gaussian noise further increases diversity.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,)
            target_class: Class index or name to augment
            target_ratio: Target ratio of this class in final dataset
            mixup_alpha: Beta distribution parameter for mixup
            noise_std: Standard deviation of Gaussian noise

        Returns:
            Tuple of (X_augmented, y_augmented)
        """
        if isinstance(target_class, str):
            target_class = self._class_to_idx[target_class]

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        n_total = len(y)
        n_target = int(n_total * target_ratio)
        n_current = np.sum(y == target_class)

        if n_current >= n_target:
            return X, y

        if n_current < 2:
            raise ValueError(f"Class {target_class} has < 2 samples. Cannot apply ADASYN+Mixup.")

        # Step 1: ADASYN
        x_adasyn, y_adasyn = self._apply_adasyn(X, y, target_class, n_target)

        # Step 2: Mixup on the augmented minority class
        x_mixup, y_mixup = self._apply_mixup(x_adasyn, y_adasyn, target_class, mixup_alpha)

        # Step 3: Add small Gaussian noise
        minority_mask = y_mixup == target_class
        x_mixup[minority_mask] += self.random_state.normal(
            0, noise_std, x_mixup[minority_mask].shape
        ).astype(np.float32)

        return x_mixup, y_mixup

    def _apply_adasyn(
        self, X: np.ndarray, y: np.ndarray, target_class: int, n_target: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply ADASYN oversampling."""
        if IMBLEARN_AVAILABLE:
            try:
                # Calculate k_neighbors based on minority samples
                n_minority = np.sum(y == target_class)
                k_neighbors = min(5, n_minority - 1)
                if k_neighbors < 1:
                    k_neighbors = 1

                adasyn = ADASYN(
                    sampling_strategy={target_class: n_target},
                    n_neighbors=k_neighbors,
                    random_state=self.config.random_state,
                )
                return cast(tuple[np.ndarray, np.ndarray], adasyn.fit_resample(X, y))
            except Exception as e:
                raise RuntimeError(f"ADASYN failed: {e}") from e

        return self._adasyn_custom(X, y, target_class, n_target)

    def _adasyn_custom(
        self, X: np.ndarray, y: np.ndarray, target_class: int, n_target: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Custom ADASYN implementation."""
        minority_mask = y == target_class
        x_minority = X[minority_mask]
        n_minority = len(x_minority)

        n_synthetic = n_target - n_minority
        if n_synthetic <= 0:
            return X, y

        k = min(5, n_minority - 1)
        if k < 1:
            k = 1

        all_neighbors = self._find_adasyn_neighbors(x_minority, X, k)
        ratios = self._calculate_difficulty_ratios(y, all_neighbors, target_class, k)
        weights = self._normalize_weights(ratios, n_minority)
        synthetic = self._generate_adasyn_samples(
            x_minority, X, all_neighbors, weights, n_synthetic, k
        )

        x_combined = np.vstack([X, synthetic])
        y_combined = np.concatenate([y, np.full(n_synthetic, target_class)])

        return x_combined, y_combined

    def _find_adasyn_neighbors(self, x_minority: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
        """Find nearest neighbors for ADASYN weighting."""
        if SKLEARN_AVAILABLE:
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(X)
            _, all_neighbors = nn.kneighbors(x_minority)
            return np.asarray(all_neighbors[:, 1:])

        return self._find_neighbors_manual(x_minority, k)

    def _calculate_difficulty_ratios(
        self,
        y: np.ndarray,
        neighbors: np.ndarray,
        target_class: int,
        k: int,
    ) -> np.ndarray:
        """Calculate per-sample difficulty ratios."""
        ratios = np.zeros(len(neighbors))
        for i in range(len(neighbors)):
            neighbor_labels = y[neighbors[i]]
            ratios[i] = np.sum(neighbor_labels != target_class) / k
        return ratios

    def _normalize_weights(self, ratios: np.ndarray, n_samples: int) -> np.ndarray:
        """Normalize difficulty ratios into sampling weights."""
        if ratios.sum() > 0:
            return np.asarray(ratios / ratios.sum())
        return np.asarray(np.ones(n_samples) / n_samples)

    def _generate_adasyn_samples(
        self,
        x_minority: np.ndarray,
        X: np.ndarray,
        neighbors: np.ndarray,
        weights: np.ndarray,
        n_synthetic: int,
        k: int,
    ) -> np.ndarray:
        """Generate ADASYN synthetic samples."""
        samples_per_minority = (weights * n_synthetic).astype(int)

        while samples_per_minority.sum() < n_synthetic:
            idx = self.random_state.choice(len(weights), p=weights)
            samples_per_minority[idx] += 1

        synthetic = np.zeros((n_synthetic, X.shape[1]), dtype=np.float32)
        idx_synthetic = 0

        for i in range(len(x_minority)):
            for _ in range(samples_per_minority[i]):
                if idx_synthetic >= n_synthetic:
                    break

                neighbor_idx = neighbors[i, self.random_state.integers(0, k)]
                neighbor = X[neighbor_idx] if SKLEARN_AVAILABLE else x_minority[neighbor_idx]
                alpha = self.random_state.random()
                synthetic[idx_synthetic] = x_minority[i] + alpha * (neighbor - x_minority[i])
                idx_synthetic += 1

        return synthetic[:idx_synthetic]

    def _apply_mixup(
        self, X: np.ndarray, y: np.ndarray, target_class: int, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Mixup data augmentation to minority class samples."""
        minority_mask = y == target_class
        x_minority = X[minority_mask]
        n_minority = len(x_minority)

        if n_minority < 2:
            return X, y

        # Generate mixup samples (same count as current minority)
        n_mixup = n_minority // 2  # Generate half as many mixup samples

        if n_mixup < 1:
            return X, y

        mixup_samples = np.zeros((n_mixup, X.shape[1]), dtype=np.float32)

        for i in range(n_mixup):
            # Sample lambda from Beta distribution
            lam = self.random_state.beta(alpha, alpha)

            # Select two random minority samples
            idx1, idx2 = self.random_state.choice(n_minority, 2, replace=False)

            # Mixup
            mixup_samples[i] = lam * x_minority[idx1] + (1 - lam) * x_minority[idx2]

        x_combined = np.vstack([X, mixup_samples])
        y_combined = np.concatenate([y, np.full(n_mixup, target_class)])

        return x_combined, y_combined

    def _augment_with_noise(
        self, X: np.ndarray, y: np.ndarray, target_class: int, n_target: int, noise_std: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple noise-based augmentation for very small classes."""
        minority_mask = y == target_class
        x_minority = X[minority_mask]
        n_minority = len(x_minority)

        if n_minority == 0:
            return X, y

        n_synthetic = n_target - n_minority
        if n_synthetic <= 0:
            return X, y

        # Replicate and add noise
        synthetic = np.zeros((n_synthetic, X.shape[1]), dtype=np.float32)
        for i in range(n_synthetic):
            base_idx = i % n_minority
            noise = np.asarray(self.random_state.normal(0, noise_std, X.shape[1]), dtype=np.float32)
            synthetic[i] = x_minority[base_idx] + noise

        x_combined = np.vstack([X, synthetic])
        y_combined = np.concatenate([y, np.full(n_synthetic, target_class)])

        return x_combined, y_combined

    def light_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_class: Union[int, str],
        target_ratio: float = 0.15,
        k_neighbors: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply light SMOTE augmentation (designed for Probe class).

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,)
            target_class: Class index or name to augment
            target_ratio: Target ratio of this class in final dataset
            k_neighbors: Number of neighbors for SMOTE

        Returns:
            Tuple of (X_augmented, y_augmented)
        """
        if isinstance(target_class, str):
            target_class = self._class_to_idx[target_class]

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        n_total = len(y)
        n_target = int(n_total * target_ratio)
        n_current = np.sum(y == target_class)

        if n_current >= n_target:
            return X, y

        if n_current < 2:
            raise ValueError(f"Class {target_class} has < 2 samples. Cannot apply SMOTE.")

        if IMBLEARN_AVAILABLE:
            try:
                k_actual = min(k_neighbors, n_current - 1)
                if k_actual < 1:
                    k_actual = 1

                smote = SMOTE(
                    sampling_strategy={target_class: n_target},
                    k_neighbors=k_actual,
                    random_state=self.config.random_state,
                )
                return cast(tuple[np.ndarray, np.ndarray], smote.fit_resample(X, y))
            except Exception as e:
                raise RuntimeError(f"SMOTE failed: {e}") from e

        # Custom light SMOTE
        minority_mask = y == target_class
        x_minority = X[minority_mask]
        n_synthetic = n_target - n_current

        k_actual = min(k_neighbors, n_current - 1)
        if k_actual < 1:
            k_actual = 1

        synthetic = self._generate_smote_samples(x_minority, n_synthetic, k_actual)

        x_combined = np.vstack([X, synthetic])
        y_combined = np.concatenate([y, np.full(n_synthetic, target_class)])

        return x_combined, y_combined

    def get_balanced_sampler(
        self,
        y: np.ndarray,
        min_samples_per_class: Optional[dict[str, int]] = None,
        batch_size: int = 32,
    ) -> "ClassBalancedSampler":
        """
        Create a class-balanced batch sampler.

        Ensures minimum representation of each class in every batch,
        with special handling for extreme minority classes like U2R.

        Args:
            y: Label array (n_samples,)
            min_samples_per_class: Dict mapping class to minimum samples per batch.
                                   Uses config defaults if None.
            batch_size: Total batch size

        Returns:
            ClassBalancedSampler instance
        """
        if min_samples_per_class is None:
            min_samples_per_class = self.config.min_samples_per_batch

        # Convert string keys to int
        min_samples: dict[int, int] = {}
        for key, val in min_samples_per_class.items():
            mapped_key = self._class_to_idx.get(key)
            if mapped_key is not None:
                min_samples[mapped_key] = val

        return ClassBalancedSampler(
            y=y,
            min_samples_per_class=min_samples,
            batch_size=batch_size,
            random_state=self.config.random_state,
        )

    def augment_dataset(
        self, X: np.ndarray, y: np.ndarray, config: Optional[AugmentationConfig] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply all attack-aware augmentations to a dataset.

        Applies the appropriate augmentation strategy for each attack type:
        - R2L: SMOTE-ENN (target 10%)
        - U2R: ADASYN + Mixup (target 5%)
        - Probe: Light SMOTE (target 15%)

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,)
            config: AugmentationConfig to use. Uses self.config if None.

        Returns:
            Tuple of (X_augmented, y_augmented)
        """
        if config is not None:
            self.config = config
            self.random_state = np.random.default_rng(config.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        original_counts = np.bincount(y.astype(int), minlength=len(self.config.class_names))
        print(f"Original class distribution: {dict(zip(self.config.class_names, original_counts))}")

        # Step 1: SMOTE-ENN for R2L
        r2l_idx = self._class_to_idx.get("R2L", 3)
        if r2l_idx < len(original_counts) and original_counts[r2l_idx] > 0:
            print(
                f"\nApplying SMOTE-ENN for R2L (target: {self.config.r2l_target_ratio * 100:.1f}%)..."
            )
            X, y = self.smote_enn(
                X,
                y,
                target_class=r2l_idx,
                target_ratio=self.config.r2l_target_ratio,
                k_neighbors=self.config.r2l_k_neighbors,
            )
            new_count = np.sum(y == r2l_idx)
            print(f"  R2L: {original_counts[r2l_idx]} → {new_count} samples")

        # Step 2: ADASYN + Mixup for U2R
        u2r_idx = self._class_to_idx.get("U2R", 4)
        if u2r_idx < len(original_counts) and original_counts[u2r_idx] > 0:
            print(
                f"\nApplying ADASYN+Mixup for U2R (target: {self.config.u2r_target_ratio * 100:.1f}%)..."
            )
            X, y = self.adasyn_mixup(
                X,
                y,
                target_class=u2r_idx,
                target_ratio=self.config.u2r_target_ratio,
                mixup_alpha=self.config.u2r_mixup_alpha,
                noise_std=self.config.u2r_noise_std,
            )
            new_count = np.sum(y == u2r_idx)
            print(f"  U2R: {original_counts[u2r_idx]} → {new_count} samples")

        # Step 3: Light SMOTE for Probe
        probe_idx = self._class_to_idx.get("Probe", 2)
        if probe_idx < len(original_counts) and original_counts[probe_idx] > 0:
            print(
                f"\nApplying Light SMOTE for Probe (target: {self.config.probe_target_ratio * 100:.1f}%)..."
            )
            X, y = self.light_smote(
                X,
                y,
                target_class=probe_idx,
                target_ratio=self.config.probe_target_ratio,
                k_neighbors=self.config.probe_k_neighbors,
            )
            new_count = np.sum(y == probe_idx)
            print(f"  Probe: {original_counts[probe_idx]} → {new_count} samples")

        final_counts = np.bincount(y.astype(int), minlength=len(self.config.class_names))
        print(f"\nFinal class distribution: {dict(zip(self.config.class_names, final_counts))}")
        print(f"Total samples: {sum(original_counts)} → {len(y)}")

        return X, y


class ClassBalancedSampler:
    """
    Batch sampler ensuring minimum samples per class in each batch.

    Implements class-balanced sampling with special handling for
    extreme minority classes like U2R (0.89% of data).
    """

    def __init__(
        self,
        y: np.ndarray,
        min_samples_per_class: dict[int, int],
        batch_size: int = 32,
        random_state: int = 42,
    ):
        """
        Initialize class-balanced sampler.

        Args:
            y: Label array (n_samples,)
            min_samples_per_class: Dict mapping class index to minimum samples per batch
            batch_size: Total batch size
            random_state: Random seed for reproducibility
        """
        self.y = np.asarray(y)
        self.min_samples = min_samples_per_class
        self.batch_size = batch_size
        self.random_state = np.random.default_rng(random_state)

        # Pre-compute indices for each class
        self.class_indices = {}
        for cls in np.unique(self.y):
            self.class_indices[cls] = np.nonzero(self.y == cls)[0]

        self.n_samples = len(y)
        self.n_batches = max(1, self.n_samples // batch_size)

    def __iter__(self):
        """Generate batches with balanced class representation."""
        for _ in range(self.n_batches):
            yield self._sample_batch()

    def __len__(self):
        """Return number of batches per epoch."""
        return self.n_batches

    def _sample_batch(self) -> np.ndarray:
        """Sample a single balanced batch."""
        batch_indices: list[int] = []
        remaining_size = self.batch_size

        # First, ensure minimum samples for each class
        for cls, min_count in self.min_samples.items():
            if cls not in self.class_indices:
                continue

            class_idx = self.class_indices[cls]
            n_available = len(class_idx)

            if n_available == 0:
                continue

            # Sample with replacement if not enough samples
            n_to_sample = min(min_count, remaining_size)
            if n_available >= n_to_sample:
                sampled = self.random_state.choice(class_idx, n_to_sample, replace=False)
            else:
                sampled = self.random_state.choice(class_idx, n_to_sample, replace=True)

            batch_indices.extend(sampled)
            remaining_size -= n_to_sample

        # Fill remaining with random samples
        if remaining_size > 0:
            all_indices = np.arange(self.n_samples)
            # Exclude already sampled
            available = np.setdiff1d(all_indices, batch_indices)
            if len(available) >= remaining_size:
                extra = self.random_state.choice(available, remaining_size, replace=False)
            else:
                extra = self.random_state.choice(all_indices, remaining_size, replace=True)
            batch_indices.extend(extra)

        return np.array(batch_indices[: self.batch_size])

    def get_sample_weights(self) -> np.ndarray:
        """
        Compute sample weights for weighted random sampling.

        Returns inverse class frequency weights for each sample,
        allowing use with PyTorch WeightedRandomSampler.

        Returns:
            Array of weights (n_samples,)
        """
        class_counts = np.bincount(self.y.astype(int))
        class_weights = 1.0 / np.maximum(class_counts, 1)

        # Normalize
        class_weights = class_weights / class_weights.sum()

        # Assign weight to each sample
        sample_weights = class_weights[self.y.astype(int)]

        return np.asarray(sample_weights)


def create_augmentation_config(
    r2l_ratio: float = 0.10,
    u2r_ratio: float = 0.05,
    probe_ratio: float = 0.15,
    u2r_batch_min: int = 4,
    random_state: int = 42,
) -> AugmentationConfig:
    """
    Factory function to create AugmentationConfig with custom settings.

    Args:
        r2l_ratio: Target ratio for R2L class
        u2r_ratio: Target ratio for U2R class
        probe_ratio: Target ratio for Probe class
        u2r_batch_min: Minimum U2R samples per batch
        random_state: Random seed

    Returns:
        AugmentationConfig instance
    """
    return AugmentationConfig(
        r2l_target_ratio=r2l_ratio,
        u2r_target_ratio=u2r_ratio,
        probe_target_ratio=probe_ratio,
        min_samples_per_batch={"Normal": 8, "DoS": 8, "Probe": 4, "R2L": 4, "U2R": u2r_batch_min},
        random_state=random_state,
    )


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    r2l_target_ratio: float = 0.10,
    u2r_target_ratio: float = 0.05,
    probe_target_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to apply minority class augmentation to a dataset.

    Balances severely underrepresented classes (U2R: 0.04%, R2L: 3.7%) to
    10-15% representation using attack-aware augmentation strategies:
    - R2L: SMOTE-ENN (combines SMOTE with Edited Nearest Neighbors)
    - U2R: ADASYN + Mixup (adaptive sampling with data mixing)
    - Probe: Light SMOTE (limited oversampling)

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Label array with class indices (n_samples,)
        r2l_target_ratio: Target ratio for R2L class (default: 10%)
        u2r_target_ratio: Target ratio for U2R class (default: 5%)
        probe_target_ratio: Target ratio for Probe class (default: 15%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_augmented, y_augmented) with balanced class representation

    Example:
        >>> X_train, y_train = load_data()
        >>> X_bal, y_bal = balance_dataset(X_train, y_train)
        >>> print(f"Original: {np.bincount(y_train)}")
        >>> print(f"Balanced: {np.bincount(y_bal)}")
    """
    config = create_augmentation_config(
        r2l_ratio=r2l_target_ratio,
        u2r_ratio=u2r_target_ratio,
        probe_ratio=probe_target_ratio,
        random_state=random_state,
    )

    augmenter = AttackAwareAugmentation(config)
    return augmenter.augment_dataset(X, y)
