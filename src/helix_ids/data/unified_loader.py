"""
Unified Dataset Loader for HELIX-IDS.

This module has been refactored into focused sub-modules:
- dataset_config.py: Dataset configurations
- feature_io.py: File loading
- label_mapping.py: Label mapping and encoding
- loader_core.py: The main UnifiedDataLoader class

This file remains for backwards compatibility.
"""

from .dataset_config import DATASET_CONFIGS, DatasetConfig
from .feature_io import find_data_files, load_file
from .label_mapping import encode_labels, get_class_distribution, log_class_distribution, map_labels
from .loader_core import (
    UnifiedDataLoader,
    get_dataset_splits,
    list_available_datasets,
    load_dataset,
)

__all__ = [
    "UnifiedDataLoader",
    "DatasetConfig",
    "DATASET_CONFIGS",
    "load_dataset",
    "get_dataset_splits",
    "list_available_datasets",
    "load_file",
    "find_data_files",
    "map_labels",
    "encode_labels",
    "get_class_distribution",
    "log_class_distribution",
]
