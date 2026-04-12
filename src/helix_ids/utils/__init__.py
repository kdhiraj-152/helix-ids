"""
HELIX-IDS Utilities Module

Provides metrics, logging, callbacks, ONNX export, and helper functions.
"""

from .callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    TrainingLogger,
    create_helix_callbacks,
)
from .export import (
    DEFAULT_THREAT_WEIGHTS,
    HELIX_CLASSES,
    HELIX_VARIANT_SIZES,
    ONNX_AVAILABLE,
    ONNXRUNTIME_AVAILABLE,
    ExportMetadata,
    ONNXExporter,
    benchmark_onnx,
    check_onnx_dependencies,
    export_for_edge,
    get_onnx_info,
    quick_export,
    validate_onnx,
)
from .metrics import (
    ModelMetrics,
    calculate_per_class_f1,
    calculate_pri_score,
    calculate_threat_weighted_f1,
    evaluate_model,
    print_evaluation_report,
)

__all__ = [
    # Metrics
    "calculate_per_class_f1",
    "calculate_threat_weighted_f1",
    "calculate_pri_score",
    "ModelMetrics",
    "evaluate_model",
    "print_evaluation_report",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "TrainingLogger",
    "CallbackList",
    "create_helix_callbacks",
    # ONNX Export
    "ONNXExporter",
    "ExportMetadata",
    "validate_onnx",
    "benchmark_onnx",
    "export_for_edge",
    "quick_export",
    "get_onnx_info",
    "check_onnx_dependencies",
    "HELIX_CLASSES",
    "HELIX_VARIANT_SIZES",
    "DEFAULT_THREAT_WEIGHTS",
    "ONNX_AVAILABLE",
    "ONNXRUNTIME_AVAILABLE",
]
