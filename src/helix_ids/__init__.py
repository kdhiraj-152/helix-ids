"""
HELIX-IDS: Hierarchical Edge-optimized Lightweight Intrusion eXpert

A production-ready intrusion detection system designed to:
1. Solve minority class suppression (R2L/U2R F1: 0.00 → 0.25+)
2. Maintain edge deployability (<30KB for ESP32)
3. Handle concept drift with adaptive retraining
4. Achieve cross-dataset generalization

Model Variants:
- HELIX-Nano: <30KB for ESP32 (520KB SRAM)
- HELIX-Lite: <200KB for RPi Zero (512MB RAM)
- HELIX-Full: <2MB for RPi 4 (4GB RAM)
"""

__version__ = "1.0.0"
__author__ = "K. Dhiraj"

from .data.augmentation import AttackAwareAugmentation
from .data.unified_loader import UnifiedDataLoader
from .models.core import HELIXIDS, HELIXFull, HELIXLite, HELIXNano

__all__ = [
    "HELIXIDS",
    "HELIXNano",
    "HELIXLite",
    "HELIXFull",
    "UnifiedDataLoader",
    "AttackAwareAugmentation",
]
