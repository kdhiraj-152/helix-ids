"""Transfer learning module for HELIX-IDS Phase 4 multi-dataset pre-training.

Implements multi-dataset pre-training on CIC-IDS2017 and UNSW-NB15 with
domain adaptation (DANN, MMD, CORAL) to transfer learned representations
to NSL-KDD.

Reference:
    - Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
    - Long et al., "Deep Transfer Learning with Joint Adaptation Networks", ICML 2017
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from helix_ids.data.unified_loader import UnifiedDataLoader
from helix_ids.governance import verify_artifact_provenance
from helix_ids.governance.provenance import checkpoint_manifest_payload
from helix_ids.models.adaptation.combined_da import CombinedDomainAdaptation
from helix_ids.models.adaptation.coral_loss import CORALLoss
from helix_ids.models.adaptation.dann import DANN, DANNConfig, DANNLoss
from helix_ids.models.adaptation.label_aware_da import ClassConditionalMMDLoss
from helix_ids.models.adaptation.mmd_loss import MMDLoss
from helix_ids.utils.export import (
    build_export_manifest,
    finalize_export_artifact,
    verify_export_artifact,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TransferLearningConfig:
    """Configuration for multi-dataset transfer learning."""

    # Datasets
    source_datasets: list[str] = field(default_factory=lambda: ["cicids-2017", "unsw-nb15"])
    target_dataset: str = "nsl-kdd"

    # Training parameters
    pretrain_epochs: int = 50
    finetune_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-4

    # Domain adaptation
    adaptation_lambda: float = 0.1
    mmd_weight: float = 0.5
    coral_weight: float = 0.5
    use_dann: bool = True
    use_mmd: bool = True
    use_class_conditional_mmd: bool = True
    use_coral: bool = True
    class_mmd_weight: float = 0.5
    use_class_weights: bool = True
    class_weight_power: float = 1.0
    max_class_weight: float = 10.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    use_balanced_sampler: bool = False
    monitor_domain_collapse: bool = True

    # Staged DA schedule
    use_staged_schedule: bool = True
    cls_only_last_epoch: int = 3
    full_da_start_epoch: int = 10

    # Model architecture
    input_dim: int = 41  # Will be updated based on aligned features
    encoder_dims: list[int] | None = None
    num_classes: int = 5  # NSL-KDD 5-class

    # Feature alignment
    common_feature_dim: int = 64  # Dimension for aligned feature space
    projection_hidden: int = 128

    # Checkpointing
    checkpoint_dir: Path | None = None
    save_every_n_epochs: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seed
    seed: int = 42

    def __post_init__(self) -> None:
        if self.encoder_dims is None:
            self.encoder_dims = [256, 128, 64]
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.full_da_start_epoch <= self.cls_only_last_epoch:
            raise ValueError(
                "full_da_start_epoch must be greater than cls_only_last_epoch"
            )


# ============================================================================
# Feature Alignment Module
# ============================================================================


class FeatureAligner(nn.Module):
    """Aligns features from different datasets to a common representation space.

    Uses learnable projections to map heterogeneous feature schemas to a
    unified representation, enabling transfer between datasets with different
    feature sets (e.g., CIC-IDS2017 with 78 features vs NSL-KDD with 41).
    """

    def __init__(
        self,
        source_dims: dict[str, int],
        target_dim: int,
        projection_hidden: int = 128,
    ):
        """Initialize feature aligner.

        Args:
            source_dims: Dictionary mapping dataset names to their feature dimensions
            target_dim: Common target feature dimension
            projection_hidden: Hidden layer size for projection networks
        """
        super().__init__()
        self.source_dims = source_dims
        self.target_dim = target_dim

        # Create projection networks for each source dataset
        self.projectors = nn.ModuleDict()
        for name, dim in source_dims.items():
            self.projectors[name] = nn.Sequential(
                nn.Linear(dim, projection_hidden),
                nn.BatchNorm1d(projection_hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(projection_hidden, target_dim),
                nn.BatchNorm1d(target_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        dataset_name: str,
    ) -> torch.Tensor:
        """Project features to common space.

        Args:
            x: Input features [batch, source_dim]
            dataset_name: Name of the source dataset

        Returns:
            Aligned features [batch, target_dim]
        """
        if dataset_name not in self.projectors:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        projected = self.projectors[dataset_name](x)
        return cast(torch.Tensor, projected)


# ============================================================================
# Multi-Dataset Pre-trainer
# ============================================================================


class MultiDatasetPretrainer:
    """Pre-trains on multiple source datasets and transfers to target.

    This class implements the Phase 4 multi-dataset pre-training strategy:
    1. Pre-train feature extractor on CIC-IDS2017 and UNSW-NB15
    2. Use domain adaptation (DANN, MMD, CORAL) to learn domain-invariant features
    3. Fine-tune on NSL-KDD target dataset

    Usage:
        config = TransferLearningConfig(
            source_datasets=["cicids-2017", "unsw-nb15"],
            target_dataset="nsl-kdd",
        )
        pretrainer = MultiDatasetPretrainer(config)

        # Pre-train on source datasets
        pretrainer.pretrain(source_datasets=["cicids-2017", "unsw-nb15"], epochs=50)

        # Fine-tune on target
        pretrainer.finetune(target_dataset="nsl-kdd", epochs=100)
    """

    def __init__(self, config: TransferLearningConfig | None = None):
        """Initialize multi-dataset pretrainer.

        Args:
            config: Transfer learning configuration. Uses defaults if None.
        """
        self.config = config or TransferLearningConfig()
        self.device = torch.device(self.config.device)

        # Set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Data loader
        self._data_loader = UnifiedDataLoader(
            scale_features=True,
            handle_missing=True,
            handle_outliers=True,
            verbose=False,
            label_mode="unified_5class",
        )

        # Models (initialized during training)
        self._feature_aligner: FeatureAligner | None = None
        self._dann_model: DANN | None = None
        self._class_mmd_loss: ClassConditionalMMDLoss | None = None
        self._combined_da: CombinedDomainAdaptation | None = None
        self._domain_monitor: nn.Linear | None = None
        self._domain_monitor_optimizer: Adam | None = None

        # Loss functions
        self._mmd_loss = MMDLoss(kernel="multi")
        self._coral_loss = CORALLoss(normalize=True)
        self._dann_loss: DANNLoss | None = None

        # Training state
        self._pretrain_history: list[dict[str, float]] = []
        self._finetune_history: list[dict[str, float]] = []
        self._is_pretrained = False

        # Dataset info (populated during align_features)
        self._source_dims: dict[str, int] = {}
        self._aligned_data: dict[str, dict[str, Any]] = {}

    def _get_da_weight(self, epoch: int) -> float:
        """Return DA component weight for current epoch under staged schedule."""
        if not self.config.use_staged_schedule:
            return 1.0

        if epoch <= self.config.cls_only_last_epoch:
            return 0.0

        if epoch >= self.config.full_da_start_epoch:
            return 1.0

        ramp_denominator = max(1, self.config.full_da_start_epoch - self.config.cls_only_last_epoch - 1)
        return float((epoch - self.config.cls_only_last_epoch) / ramp_denominator)

    @staticmethod
    def _compute_class_weights(
        y: np.ndarray,
        num_classes: int,
        power: float = 1.0,
        max_weight: float | None = 10.0,
    ) -> torch.Tensor:
        """Compute inverse-frequency class weights with optional amplification."""
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes).astype(np.float32)
        weights = np.zeros(num_classes, dtype=np.float32)
        nonzero = counts > 0

        if not np.any(nonzero):
            return torch.ones(num_classes, dtype=torch.float32)

        weights[nonzero] = float(counts[nonzero].sum()) / (counts[nonzero] * float(nonzero.sum()))
        if power > 0.0 and abs(power - 1.0) > 1e-8:
            weights[nonzero] = np.power(weights[nonzero], power)
        if max_weight is not None and max_weight > 0.0:
            weights[nonzero] = np.clip(weights[nonzero], 1e-6, max_weight)
        mean_weight = float(weights[nonzero].mean())
        if mean_weight > 0:
            weights[nonzero] /= mean_weight

        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def _make_balanced_sampler(y: np.ndarray, num_classes: int) -> WeightedRandomSampler:
        """Create a replacement sampler that oversamples minority classes."""
        labels = np.asarray(y, dtype=np.int64)
        counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
        inv = np.zeros(num_classes, dtype=np.float64)
        nonzero = counts > 0
        inv[nonzero] = 1.0 / counts[nonzero]
        sample_weights = inv[labels]
        mean_weight = float(sample_weights.mean())
        if mean_weight > 0:
            sample_weights /= mean_weight

        return WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(labels),
            replacement=True,
        )

    def _classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute CE or weighted focal loss based on config."""
        if not self.config.use_focal_loss or self.config.focal_gamma <= 0.0:
            return F.cross_entropy(logits, targets, weight=class_weight)

        ce = F.cross_entropy(logits, targets, weight=class_weight, reduction="none")
        pt = torch.exp(-ce)
        focal = torch.pow(1.0 - pt, self.config.focal_gamma) * ce
        return focal.mean()

    @staticmethod
    def _expected_domain_chance_acc(n_domains: int) -> float:
        return 1.0 / max(1, n_domains)

    def _initialize_domain_monitor(self, n_domains: int) -> None:
        """Initialize a detached tri-domain monitor used only for collapse telemetry."""
        if not self.config.monitor_domain_collapse or n_domains < 3:
            self._domain_monitor = None
            self._domain_monitor_optimizer = None
            return

        self._domain_monitor = nn.Linear(self.config.common_feature_dim, n_domains).to(self.device)
        self._domain_monitor_optimizer = Adam(
            self._domain_monitor.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _update_domain_monitor(
        self,
        aligned_features: dict[str, torch.Tensor],
        dataset_names: list[str],
    ) -> float:
        """Train and evaluate auxiliary domain monitor on detached aligned features."""
        if self._domain_monitor is None or self._domain_monitor_optimizer is None:
            return float("nan")

        features = []
        labels = []
        for idx, name in enumerate(dataset_names):
            feat = aligned_features[name].detach()
            features.append(feat)
            labels.append(torch.full((feat.shape[0],), idx, dtype=torch.long, device=self.device))

        monitor_x = torch.cat(features, dim=0)
        monitor_y = torch.cat(labels, dim=0)

        self._domain_monitor.train()
        self._domain_monitor_optimizer.zero_grad()
        monitor_logits = self._domain_monitor(monitor_x)
        monitor_loss = F.cross_entropy(monitor_logits, monitor_y)
        monitor_loss.backward()
        self._domain_monitor_optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(monitor_logits, dim=1)
            acc = (pred == monitor_y).float().mean().item()
        return float(acc)

    def align_features(
        self,
        source_df: pd.DataFrame | None = None,
        target_df: pd.DataFrame | None = None,
        source_name: str | None = None,
        target_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align features between source and target datasets.

        For datasets with different schemas (e.g., CIC-IDS2017 with 78 features
        vs NSL-KDD with 41), this method finds common feature semantics and
        creates a unified representation.

        Args:
            source_df: Source dataset DataFrame (optional if source_name provided)
            target_df: Target dataset DataFrame (optional if target_name provided)
            source_name: Source dataset name for auto-loading
            target_name: Target dataset name for auto-loading

        Returns:
            Tuple of (aligned_source, aligned_target) numpy arrays
        """
        # Load data if not provided
        if source_df is None and source_name is not None:
            source_data, _, _ = cast(
                tuple[np.ndarray, np.ndarray, list[str]],
                self._data_loader.load(source_name, return_class_names=True),
            )
            source_df = pd.DataFrame(source_data)
        if target_df is None and target_name is not None:
            target_data, _, _ = cast(
                tuple[np.ndarray, np.ndarray, list[str]],
                self._data_loader.load(target_name, return_class_names=True),
            )
            target_df = pd.DataFrame(target_data)

        if source_df is None or target_df is None:
            raise ValueError("Must provide either DataFrame or dataset name")

        # Get numeric columns only
        source_numeric = source_df.select_dtypes(include=[np.number])
        target_numeric = target_df.select_dtypes(include=[np.number])

        # Find common columns by name
        common_cols = list(set(source_numeric.columns) & set(target_numeric.columns))

        if common_cols:
            # Use common columns
            logger.info(f"Found {len(common_cols)} common features between datasets")
            aligned_source = source_numeric[common_cols].values
            aligned_target = target_numeric[common_cols].values
        else:
            # No common columns - use statistical alignment
            logger.info("No common features found, using projection alignment")
            # Pad/truncate to match dimensions
            min_dim = min(source_numeric.shape[1], target_numeric.shape[1])
            aligned_source = source_numeric.iloc[:, :min_dim].values
            aligned_target = target_numeric.iloc[:, :min_dim].values

        return aligned_source.astype(np.float32), aligned_target.astype(np.float32)

    def _load_dataset(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Load and preprocess a dataset.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Tuple of (X, y, domain_ids, num_features)
        """
        X, y, class_names, domain_ids = cast(
            tuple[np.ndarray, np.ndarray, list[str], np.ndarray],
            self._data_loader.load_with_domain_ids(
                dataset_name,
                return_class_names=True,
            ),
        )
        logger.info(
            f"Loaded {dataset_name}: {X.shape[0]} samples, "
            f"{X.shape[1]} features, {len(class_names)} classes"
        )
        return (
            X.astype(np.float32),
            y.astype(np.int64),
            domain_ids.astype(np.int64),
            X.shape[1],
        )

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        domain_ids: np.ndarray | None = None,
        batch_size: int | None = None,
        sampler: WeightedRandomSampler | None = None,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays.

        Args:
            X: Feature array
            y: Label array
            batch_size: Batch size (uses config default if None)
            shuffle: Whether to shuffle data

        Returns:
            PyTorch DataLoader
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()

        if domain_ids is not None:
            d_tensor = torch.from_numpy(domain_ids).long()
            dataset = TensorDataset(X_tensor, y_tensor, d_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            drop_last=True,
        )

    def _initialize_models(self, input_dims: dict[str, int]) -> None:
        """Initialize models with correct dimensions.

        Args:
            input_dims: Dictionary mapping dataset names to feature dimensions
        """
        self._source_dims = input_dims

        # Initialize feature aligner
        self._feature_aligner = FeatureAligner(
            source_dims=input_dims,
            target_dim=self.config.common_feature_dim,
            projection_hidden=self.config.projection_hidden,
        ).to(self.device)

        # Initialize DANN with aligned feature dimension
        dann_config = DANNConfig(
            input_dim=self.config.common_feature_dim,
            num_classes=self.config.num_classes,
            encoder_dims=self.config.encoder_dims,
            lambda_max=self.config.adaptation_lambda,
        )
        self._dann_model = DANN(dann_config).to(self.device)
        self._dann_loss = DANNLoss(adversarial_weight=1.0)
        self._class_mmd_loss = (
            ClassConditionalMMDLoss(num_classes=self.config.num_classes, kernel="multi")
            if self.config.use_class_conditional_mmd
            else None
        )
        active_da_total = (
            (1.0 if self.config.use_dann else 0.0)
            + (self.config.mmd_weight if self.config.use_mmd else 0.0)
            + (self.config.coral_weight if self.config.use_coral else 0.0)
        )
        if active_da_total > 0:
            self._combined_da = CombinedDomainAdaptation(
                dann_weight=1.0 if self.config.use_dann else 0.0,
                mmd_weight=self.config.mmd_weight if self.config.use_mmd else 0.0,
                coral_weight=self.config.coral_weight if self.config.use_coral else 0.0,
                lambda_da=1.0,
            ).to(self.device)
        else:
            self._combined_da = None

    def pretrain(
        self,
        source_datasets: list[str] | None = None,
        epochs: int | None = None,
    ) -> dict[str, list[float]]:
        """Pre-train on source datasets with domain adaptation.

        Uses DANN, MMD, and CORAL losses to learn domain-invariant features
        across multiple source datasets.

        Args:
            source_datasets: List of source dataset names.
                             Uses config defaults if None.
            epochs: Number of pre-training epochs. Uses config default if None.

        Returns:
            Dictionary of training history with loss curves
        """
        if source_datasets is None:
            source_datasets = self.config.source_datasets
        if epochs is None:
            epochs = self.config.pretrain_epochs

        logger.info(f"Starting pre-training on {source_datasets} for {epochs} epochs")

        # Load all source datasets
        source_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        input_dims: dict[str, int] = {}
        class_weights_by_dataset: dict[str, torch.Tensor] = {}

        for dataset_name in source_datasets:
            X, y, domain_ids, n_features = self._load_dataset(dataset_name)
            source_data[dataset_name] = (X, y, domain_ids)
            input_dims[dataset_name] = n_features
            if self.config.use_class_weights:
                class_weights_by_dataset[dataset_name] = self._compute_class_weights(
                    y,
                    self.config.num_classes,
                    power=self.config.class_weight_power,
                    max_weight=self.config.max_class_weight,
                ).to(self.device)

        # Initialize models
        self._initialize_models(input_dims)
        if self._feature_aligner is None or self._dann_model is None or self._dann_loss is None:
            raise RuntimeError("Failed to initialize pretraining models.")
        feature_aligner = self._feature_aligner
        dann_model = self._dann_model

        # Create data loaders
        source_loaders: dict[str, DataLoader] = {}
        for name, (X, y, domain_ids) in source_data.items():
            sampler = (
                self._make_balanced_sampler(y, self.config.num_classes)
                if self.config.use_balanced_sampler
                else None
            )
            source_loaders[name] = self._create_dataloader(
                X,
                y,
                domain_ids=domain_ids,
                sampler=sampler,
                shuffle=sampler is None,
            )

        # Optimizer for all parameters
        all_params = list(feature_aligner.parameters()) + list(dann_model.parameters())
        optimizer = Adam(
            all_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # Training loop
        history: dict[str, list[float]] = {
            "total_loss": [],
            "task_loss": [],
            "domain_loss": [],
            "mmd_loss": [],
            "class_mmd_loss": [],
            "coral_loss": [],
            "domain_acc": [],
            "tri_domain_acc": [],
            "domain_chance_acc": [],
            "da_weight": [],
        }

        dataset_names = list(source_loaders.keys())
        n_datasets = len(dataset_names)
        self._initialize_domain_monitor(n_datasets)
        domain_chance_acc = self._expected_domain_chance_acc(n_datasets)

        for epoch in range(epochs):
            feature_aligner.train()
            dann_model.train()
            da_weight = self._get_da_weight(epoch)

            epoch_losses = {
                "total_loss": 0.0,
                "task_loss": 0.0,
                "domain_loss": 0.0,
                "mmd_loss": 0.0,
                "class_mmd_loss": 0.0,
                "coral_loss": 0.0,
                "da_weight": 0.0,
            }
            n_batches = 0
            domain_acc_sum = 0.0
            domain_acc_count = 0
            tri_domain_acc_sum = 0.0
            tri_domain_acc_count = 0

            # Get iterators for all datasets
            iterators = {name: iter(loader) for name, loader in source_loaders.items()}

            # Train using pairs of datasets for domain adaptation
            while True:
                try:
                    batches = {}
                    for name in dataset_names:
                        batch = next(iterators[name])
                        if len(batch) == 3:
                            x_batch, y_batch, d_batch = batch
                            d_tensor = cast(torch.Tensor, d_batch).to(self.device)
                        else:
                            x_batch, y_batch = batch
                            d_tensor = torch.full(
                                (x_batch.shape[0],),
                                fill_value=-1,
                                dtype=torch.long,
                                device=self.device,
                            )
                        x_tensor = cast(torch.Tensor, x_batch).to(self.device)
                        y_tensor = cast(torch.Tensor, y_batch).to(self.device)
                        batches[name] = (x_tensor, y_tensor, d_tensor)
                except StopIteration:
                    break

                optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=self.device)

                # Process each dataset pair for domain adaptation
                aligned_features = {}
                for name, (x, _y, _d) in batches.items():
                    aligned_features[name] = feature_aligner(x, name)

                labels_by_dataset = {
                    name: cast(torch.Tensor, batches[name][1])
                    for name in dataset_names
                }

                # Task classification loss remains active even in cls-only schedule stage.
                task_losses = []
                for name in dataset_names:
                    logits = dann_model(aligned_features[name])
                    class_weight = (
                        class_weights_by_dataset.get(name)
                        if self.config.use_class_weights
                        else None
                    )
                    task_losses.append(
                        self._classification_loss(
                            logits,
                            labels_by_dataset[name],
                            class_weight=class_weight,
                        )
                    )
                task_loss = torch.stack(task_losses).mean()
                total_loss = total_loss + task_loss

                # DANN training between dataset pairs
                task_loss_sum = float(task_loss.item())
                domain_loss_sum = 0.0
                mmd_loss_sum = 0.0
                class_mmd_loss_sum = 0.0
                coral_loss_sum = 0.0

                progress = epoch / max(1, epochs - 1)
                if self.config.use_dann:
                    lambda_ = dann_model.update_lambda(progress)
                    dann_model.grl.set_lambda(lambda_ * da_weight)
                else:
                    lambda_ = 0.0

                if self._combined_da is not None:
                    self._combined_da.update_lambda(progress)

                for i in range(n_datasets):
                    for j in range(i + 1, n_datasets):
                        name_i, name_j = dataset_names[i], dataset_names[j]
                        feat_i = aligned_features[name_i]
                        feat_j = aligned_features[name_j]
                        y_i = labels_by_dataset[name_i]
                        y_j = labels_by_dataset[name_j]

                        domain_src = None
                        domain_tgt = None
                        if self.config.use_dann:
                            _, domain_src, domain_tgt = dann_model.forward_dann(feat_i, feat_j)

                        if self._combined_da is not None:
                            da_losses = self._combined_da(
                                feat_i,
                                feat_j,
                                source_domain_logits=domain_src,
                                target_domain_logits=domain_tgt,
                            )
                            total_loss = (
                                total_loss
                                + da_weight
                                * self.config.adaptation_lambda
                                * da_losses["combined_da_loss"]
                            )
                            if "dann_loss" in da_losses:
                                domain_loss_sum += float(da_losses["dann_loss"].item())
                            if "mmd_loss" in da_losses:
                                mmd_loss_sum += float(da_losses["mmd_loss"].item())
                            if "coral_loss" in da_losses:
                                coral_loss_sum += float(da_losses["coral_loss"].item())

                        if self.config.use_class_conditional_mmd and self._class_mmd_loss is not None:
                            class_mmd, _ = self._class_mmd_loss(feat_i, feat_j, y_i, y_j)
                            class_mmd_loss_sum += float(class_mmd.item())
                            total_loss = total_loss + da_weight * self.config.class_mmd_weight * class_mmd

                        if domain_src is not None and domain_tgt is not None:
                            with torch.no_grad():
                                src_pred = (domain_src >= 0).float()
                                tgt_pred = (domain_tgt >= 0).float()
                                src_acc = (src_pred == 0.0).float().mean()
                                tgt_acc = (tgt_pred == 1.0).float().mean()
                                domain_acc_sum += float(((src_acc + tgt_acc) / 2).item())
                                domain_acc_count += 1

                tri_domain_acc = self._update_domain_monitor(aligned_features, dataset_names)
                if not np.isnan(tri_domain_acc):
                    tri_domain_acc_sum += tri_domain_acc
                    tri_domain_acc_count += 1

                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

                # Accumulate losses
                n_pairs = n_datasets * (n_datasets - 1) / 2
                epoch_losses["total_loss"] += total_loss.item()
                epoch_losses["task_loss"] += task_loss_sum
                epoch_losses["domain_loss"] += domain_loss_sum / max(n_pairs, 1)
                epoch_losses["mmd_loss"] += mmd_loss_sum / max(n_pairs, 1)
                epoch_losses["class_mmd_loss"] += class_mmd_loss_sum / max(n_pairs, 1)
                epoch_losses["coral_loss"] += coral_loss_sum / max(n_pairs, 1)
                epoch_losses["da_weight"] += da_weight
                n_batches += 1

            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)
                history[k].append(epoch_losses[k])

            epoch_domain_acc = domain_acc_sum / max(1, domain_acc_count)
            history["domain_acc"].append(epoch_domain_acc)
            epoch_losses["domain_acc"] = epoch_domain_acc

            epoch_tri_domain_acc = tri_domain_acc_sum / max(1, tri_domain_acc_count)
            history["tri_domain_acc"].append(epoch_tri_domain_acc)
            history["domain_chance_acc"].append(domain_chance_acc)
            epoch_losses["tri_domain_acc"] = epoch_tri_domain_acc
            epoch_losses["domain_chance_acc"] = domain_chance_acc

            scheduler.step()

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Total: {epoch_losses['total_loss']:.4f}, "
                    f"Task: {epoch_losses['task_loss']:.4f}, "
                    f"Domain: {epoch_losses['domain_loss']:.4f}, "
                    f"ClassMMD: {epoch_losses['class_mmd_loss']:.4f}, "
                    f"DomainAcc(pair): {epoch_losses['domain_acc']:.4f}, "
                    f"DomainAcc(tri): {epoch_losses['tri_domain_acc']:.4f} "
                    f"(chance={epoch_losses['domain_chance_acc']:.4f}), "
                    f"DAWeight: {epoch_losses['da_weight']:.3f}"
                )

            # Checkpointing
            if (
                self.config.checkpoint_dir is not None
                and (epoch + 1) % self.config.save_every_n_epochs == 0
            ):
                self.save_checkpoint(self.config.checkpoint_dir / f"pretrain_epoch_{epoch + 1}.pt")

        self._pretrain_history = [{k: history[k][i] for k in history} for i in range(epochs)]
        self._is_pretrained = True

        logger.info("Pre-training complete")
        return history

    def finetune(
        self,
        target_dataset: str | None = None,
        epochs: int | None = None,
        freeze_encoder_epochs: int = 10,
    ) -> dict[str, list[float]]:
        """Fine-tune on target dataset (NSL-KDD).

        Transfers learned representations from pre-training to the target
        dataset, optionally freezing encoder layers initially.

        Args:
            target_dataset: Target dataset name. Uses config default if None.
            epochs: Number of fine-tuning epochs. Uses config default if None.
            freeze_encoder_epochs: Number of epochs to freeze encoder layers.

        Returns:
            Dictionary of training history with loss and accuracy curves
        """
        if target_dataset is None:
            target_dataset = self.config.target_dataset
        if epochs is None:
            epochs = self.config.finetune_epochs

        logger.info(f"Starting fine-tuning on {target_dataset} for {epochs} epochs")

        # Load target dataset
        X, y, _domain_ids, n_features = self._load_dataset(target_dataset)

        # Split into train/val
        n_samples = len(X)
        n_val = int(0.15 * n_samples)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Create data loaders
        train_sampler = (
            self._make_balanced_sampler(y_train, self.config.num_classes)
            if self.config.use_balanced_sampler
            else None
        )
        train_loader = self._create_dataloader(
            X_train,
            y_train,
            sampler=train_sampler,
            shuffle=train_sampler is None,
        )
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        # Initialize models if not pre-trained
        if not self._is_pretrained or self._dann_model is None:
            logger.warning("Models not pre-trained, initializing from scratch")
            self._initialize_models({target_dataset: n_features})
        if self._feature_aligner is None or self._dann_model is None:
            raise RuntimeError("Models are not available for fine-tuning.")
        feature_aligner = self._feature_aligner
        dann_model = self._dann_model

        # Add target dataset projector if needed
        if target_dataset not in feature_aligner.projectors:
            feature_aligner.projectors[target_dataset] = nn.Sequential(
                nn.Linear(n_features, self.config.projection_hidden),
                nn.BatchNorm1d(self.config.projection_hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.config.projection_hidden, self.config.common_feature_dim),
                nn.BatchNorm1d(self.config.common_feature_dim),
            ).to(self.device)

        # Optimizer
        all_params = list(feature_aligner.parameters()) + list(dann_model.parameters())
        optimizer = Adam(
            all_params,
            lr=self.config.learning_rate * 0.1,  # Lower LR for fine-tuning
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # Loss function
        if self.config.use_class_weights:
            class_weights = self._compute_class_weights(
                y_train,
                self.config.num_classes,
                power=self.config.class_weight_power,
                max_weight=self.config.max_class_weight,
            ).to(self.device)
        else:
            class_weights = None

        # Training history
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_acc = 0.0

        for epoch in range(epochs):
            # Freeze encoder for initial epochs
            if epoch < freeze_encoder_epochs:
                for param in dann_model.feature_extractor.parameters():
                    param.requires_grad = False
            else:
                for param in dann_model.feature_extractor.parameters():
                    param.requires_grad = True

            # Training
            feature_aligner.train()
            dann_model.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for x, y_batch in train_loader:
                x = cast(torch.Tensor, x).to(self.device)
                y_batch = cast(torch.Tensor, y_batch).to(self.device)

                optimizer.zero_grad()

                # Forward pass
                aligned = feature_aligner(x, target_dataset)
                logits = dann_model(aligned)

                loss = self._classification_loss(logits, y_batch, class_weight=class_weights)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += y_batch.size(0)
                train_correct += predicted.eq(y_batch).sum().item()

            # Validation
            feature_aligner.eval()
            dann_model.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for x, y_batch in val_loader:
                    x = cast(torch.Tensor, x).to(self.device)
                    y_batch = cast(torch.Tensor, y_batch).to(self.device)

                    aligned = feature_aligner(x, target_dataset)
                    logits = dann_model(aligned)

                    loss = self._classification_loss(logits, y_batch, class_weight=class_weights)
                    val_loss += loss.item()

                    _, predicted = logits.max(1)
                    val_total += y_batch.size(0)
                    val_correct += predicted.eq(y_batch).sum().item()

            # Compute averages
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            scheduler.step()

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.config.checkpoint_dir is not None:
                    self.save_checkpoint(self.config.checkpoint_dir / "best_finetune.pt")

        self._finetune_history = [{k: history[k][i] for k in history} for i in range(epochs)]

        logger.info(f"Fine-tuning complete. Best val accuracy: {best_val_acc:.2f}%")
        return history

    def predict(
        self,
        X: np.ndarray,
        dataset_name: str | None = None,
    ) -> np.ndarray:
        """Predict class labels for input samples.

        Args:
            X: Input features [n_samples, n_features]
            dataset_name: Dataset name for feature alignment.
                          Uses target dataset if None.

        Returns:
            Predicted class labels [n_samples]
        """
        if dataset_name is None:
            dataset_name = self.config.target_dataset

        if self._dann_model is None or self._feature_aligner is None:
            raise RuntimeError("Models not initialized. Call pretrain() or finetune().")

        self._feature_aligner.eval()
        self._dann_model.eval()

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        with torch.no_grad():
            aligned = self._feature_aligner(X_tensor, dataset_name)
            logits = self._dann_model(aligned)
            _, predictions = logits.max(1)

        return np.asarray(predictions.cpu().numpy(), dtype=np.int64)

    def predict_proba(
        self,
        X: np.ndarray,
        dataset_name: str | None = None,
    ) -> np.ndarray:
        """Predict class probabilities for input samples.

        Args:
            X: Input features [n_samples, n_features]
            dataset_name: Dataset name for feature alignment.

        Returns:
            Class probabilities [n_samples, n_classes]
        """
        if dataset_name is None:
            dataset_name = self.config.target_dataset

        if self._dann_model is None or self._feature_aligner is None:
            raise RuntimeError("Models not initialized. Call pretrain() or finetune().")

        self._feature_aligner.eval()
        self._dann_model.eval()

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        with torch.no_grad():
            aligned = self._feature_aligner(X_tensor, dataset_name)
            logits = self._dann_model(aligned)
            probs = F.softmax(logits, dim=1)

        return np.asarray(probs.cpu().numpy(), dtype=np.float32)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config,
            "feature_aligner_state": (
                self._feature_aligner.state_dict() if self._feature_aligner is not None else None
            ),
            "dann_model_state": (
                self._dann_model.state_dict() if self._dann_model is not None else None
            ),
            "source_dims": self._source_dims,
            "pretrain_history": self._pretrain_history,
            "finetune_history": self._finetune_history,
            "is_pretrained": self._is_pretrained,
        }

        # Embed immutable runtime contract metadata and write canonical sidecars
        import json

        from helix_ids.contracts.schema_contract import runtime_contract_payload

        payload = dict(checkpoint)
        payload.update(runtime_contract_payload())

        torch.save(payload, path)
        contract = runtime_contract_payload()
        manifest_base = build_export_manifest(
            contract=contract,
            model_architecture=type(self).__name__,
            export_config={"format": "transfer-learning-checkpoint"},
        )
        sidecars = {
            "contract": path.with_suffix(path.suffix + ".contract.json"),
            "feature_order": path.with_suffix(path.suffix + ".feature_order.json"),
            "schema_hash": path.with_suffix(path.suffix + ".schema_hash.txt"),
        }
        for sidecar_path in sidecars.values():
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecars["contract"].write_text(json.dumps(contract, indent=2), encoding="utf-8")
        sidecars["feature_order"].write_text(json.dumps(contract["feature_order"], indent=2), encoding="utf-8")
        sidecars["schema_hash"].write_text(str(contract["schema_hash"]) + "\n", encoding="utf-8")
        finalize_export_artifact(path, manifest_base, sidecars=sidecars)

        verify_export_artifact(path, kind="checkpoint", contract=contract, embedded_manifest=checkpoint_manifest_payload(manifest_base))

        logger.info(f"Checkpoint saved to {path} (with canonical runtime contract sidecars)")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint_preview = torch.load(path, map_location="cpu", weights_only=True)
        deploy_path = path.parent / "deployment.manifest.json"
        deployment_manifest = deploy_path if deploy_path.exists() else None
        verify_artifact_provenance(
            path,
            kind="checkpoint",
            contract=checkpoint_preview,
            embedded_manifest=checkpoint_preview.get("artifact_manifest"),
            sidecars={
                "contract": path.with_suffix(path.suffix + ".contract.json"),
                "feature_order": path.with_suffix(path.suffix + ".feature_order.json"),
                "schema_hash": path.with_suffix(path.suffix + ".schema_hash.txt"),
            },
            deployment_manifest=deployment_manifest,
        )

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.config = checkpoint["config"]
        self._source_dims = checkpoint["source_dims"]
        self._pretrain_history = checkpoint["pretrain_history"]
        self._finetune_history = checkpoint["finetune_history"]
        self._is_pretrained = checkpoint["is_pretrained"]

        # Reinitialize models
        if checkpoint["feature_aligner_state"] is not None:
            self._initialize_models(self._source_dims)
            if self._feature_aligner is None or self._dann_model is None:
                raise RuntimeError("Failed to initialize models while loading checkpoint.")
            self._feature_aligner.load_state_dict(checkpoint["feature_aligner_state"])
            self._dann_model.load_state_dict(checkpoint["dann_model_state"])

        logger.info(f"Checkpoint loaded from {path}")

    def get_features(
        self,
        X: np.ndarray,
        dataset_name: str | None = None,
    ) -> np.ndarray:
        """Extract domain-invariant features.

        Args:
            X: Input features [n_samples, n_features]
            dataset_name: Dataset name for feature alignment.

        Returns:
            Extracted features [n_samples, feature_dim]
        """
        if dataset_name is None:
            dataset_name = self.config.target_dataset

        if self._dann_model is None or self._feature_aligner is None:
            raise RuntimeError("Models not initialized.")

        self._feature_aligner.eval()
        self._dann_model.eval()

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        with torch.no_grad():
            aligned = self._feature_aligner(X_tensor, dataset_name)
            features = self._dann_model.get_features(aligned)

        return np.asarray(features.cpu().numpy(), dtype=np.float32)


# ============================================================================
# Factory Functions
# ============================================================================


def create_pretrainer(
    source_datasets: list[str] | None = None,
    target_dataset: str = "nsl-kdd",
    pretrain_epochs: int = 50,
    finetune_epochs: int = 100,
    **kwargs: Any,
) -> MultiDatasetPretrainer:
    """Factory function to create a MultiDatasetPretrainer.

    Args:
        source_datasets: List of source dataset names for pre-training.
        target_dataset: Target dataset name for fine-tuning.
        pretrain_epochs: Number of pre-training epochs.
        finetune_epochs: Number of fine-tuning epochs.
        **kwargs: Additional config parameters.

    Returns:
        Configured MultiDatasetPretrainer instance
    """
    if source_datasets is None:
        source_datasets = ["cicids-2017", "unsw-nb15"]

    config = TransferLearningConfig(
        source_datasets=source_datasets,
        target_dataset=target_dataset,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        **kwargs,
    )

    return MultiDatasetPretrainer(config)
