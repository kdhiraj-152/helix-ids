"""diagnostics: geometry analysis, clustering analysis, representation diagnostics.

Phase 13A-1 extraction from HelixFullTrainer.

Public API:
    GeometryAnalyzer         — geometry integrity, collision detection, density estimation
    ClusterAnalyzer          — embedding clustering, relabel proposal generation
    RepresentationDiagnostics — representation metric computation, snapshot generation
"""

from scripts.training.diagnostics.cluster_analyzer import ClusterAnalyzer
from scripts.training.diagnostics.geometry_analyzer import GeometryAnalyzer
from scripts.training.diagnostics.rep_diagnostics import RepresentationDiagnostics

__all__ = [
    "GeometryAnalyzer",
    "ClusterAnalyzer",
    "RepresentationDiagnostics",
]
