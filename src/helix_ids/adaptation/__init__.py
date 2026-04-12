"""Domain adaptation modules for HELIX-IDS."""

from .feature_harmonization import (
    FeatureHarmonizer,
    create_cross_dataset_pipeline,
    harmonize_dataset_pair,
)
from .online_finetune import OnlineFineTuner, quick_calibrate

__all__ = [
    "FeatureHarmonizer",
    "create_cross_dataset_pipeline",
    "harmonize_dataset_pair",
    "OnlineFineTuner",
    "quick_calibrate",
]
