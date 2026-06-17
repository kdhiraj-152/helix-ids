"""representation: centroid management and coordination for HelixFullTrainer.

Phase 13A-3 extraction from HelixFullTrainer.

Public API:
    CentroidManager           — per-class centroid EMA state and lifecycle
    RepresentationCoordinator — phase-orchestration utilities

Dependency rules:
    representation -> torch (allowed)
    representation -> numpy (allowed)
    representation -> trainer internals (forbidden)
    representation -> governance (forbidden)
"""

from __future__ import annotations

from scripts.training.representation.centroid_manager import CentroidManager
from scripts.training.representation.representation_coordinator import (
    RepresentationCoordinator,
)

__all__ = [
    "CentroidManager",
    "RepresentationCoordinator",
]
