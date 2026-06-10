"""
HELIX-IDS Models Module

Contains the core model components:
- Temporal Attention Module (TAM)
- Hierarchical Classification Head
- Threat-Aware Focal Loss
- Main HELIX-IDS model with variants
"""

from .attention import TemporalAttentionModule
from .classifier import HierarchicalClassifier
from .core import HELIXIDS, HELIXFull, HELIXLite, HELIXNano
from .full import HelixFullConfig, HelixIDSFull, create_helix_full
from .loss import MultiTaskLoss, ThreatAwareFocalLoss

__all__ = [
    # Core components
    "TemporalAttentionModule",
    "HierarchicalClassifier",
    "ThreatAwareFocalLoss",
    "MultiTaskLoss",
    # Model variants
    "HELIXIDS",
    "HELIXNano",
    "HELIXLite",
    "HELIXFull",
    "HelixFullConfig",
    "HelixIDSFull",
    "create_helix_full",
]
