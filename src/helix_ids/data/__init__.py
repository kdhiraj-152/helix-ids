"""
HELIX-IDS Data Module

Provides unified data loading, feature engineering, and augmentation
for NSL-KDD, UNSW-NB15, and CICIDS-2018 datasets.
"""

from .augmentation import AttackAwareAugmentation, balance_dataset, create_augmentation_config
from .feature_engineering import (
    THROUGHPUT_FEATURES,
    FeatureEngineer,
    compute_throughput_features,
    get_throughput_features,
)
from .preprocessing import DataPreprocessor
from .unified_loader import UnifiedDataLoader

__all__ = [
    "UnifiedDataLoader",
    "FeatureEngineer",
    "AttackAwareAugmentation",
    "balance_dataset",
    "create_augmentation_config",
    "DataPreprocessor",
    "THROUGHPUT_FEATURES",
    "compute_throughput_features",
    "get_throughput_features",
]
