"""
Label mapping and encoding utilities for HELIX-IDS datasets.

Extracted from unified_loader.py (was >1230 lines).
Responsible solely for converting raw string labels → integer class indices.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .dataset_config import (
    CICIDS_2018_LABEL_MAPPING,
    CICIDS_LABEL_MAPPING,
    CICIDS_TO_UNIFIED_5CLASS,
    NSL_KDD_ATTACK_MAPPING,
    UNIFIED_5CLASS,
    UNSW_TO_UNIFIED_5CLASS,
    WEBATTACK_BRUTEFORCE,
    WEBATTACK_SQLINJECTION,
    WEBATTACK_XSS,
    DatasetConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_labels(
    y: np.ndarray,
    config: DatasetConfig,
    label_mode: str = "native",
) -> np.ndarray:
    """
    Map raw string labels to canonical form.

    Args:
        y: Raw label array.
        config: Dataset configuration.
        label_mode: ``"native"`` keeps dataset-specific labels;
                    ``"unified_5class"`` maps everything to
                    [Normal, DoS, Probe, R2L, U2R].

    Returns:
        Mapped label array (still strings, not yet integer-encoded).
    """
    if config.name == "NSL-KDD":
        mapped = _map_nsl_kdd_labels(y)
    elif config.name.startswith("CICIDS-"):
        mapped = _map_cicids_labels(y)
    elif config.name == "UNSW-NB15":
        mapped = _map_unsw_labels(y)
    else:
        mapped = y

    if label_mode == "unified_5class":
        return _map_to_unified_5class(mapped, config)
    return mapped


def encode_labels(
    y: np.ndarray,
    _config: DatasetConfig,
    label_mode: str = "native",
    encoder: Optional[LabelEncoder] = None,
    fit_labels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[str], LabelEncoder]:
    """
    Encode string labels to integer indices.

    Args:
        y: Mapped label array (output of :func:`map_labels`).
        _config: Dataset configuration retained for API compatibility.
        label_mode: ``"native"`` or ``"unified_5class"``.
        encoder: Existing :class:`LabelEncoder` to reuse (``transform`` only).
        fit_labels: If provided, fit a fresh encoder on these combined labels
                    (for ensuring consistent class ordering across splits).

    Returns:
        ``(y_encoded, class_names, encoder)`` — the encoder is returned so it
        can be reused across train/val/test splits.
    """
    if label_mode == "unified_5class":
        class_names = list(UNIFIED_5CLASS)
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        encoded = np.array([class_to_idx.get(str(label), 0) for label in y], dtype=np.int64)
        if encoder is None:
            encoder = LabelEncoder().fit(class_names)
        return encoded, class_names, encoder

    if encoder is None:
        encoder = LabelEncoder()
        labels_to_fit = fit_labels if fit_labels is not None else y
        unique = sorted(np.unique(labels_to_fit))
        encoder.fit(unique)

    known = set(encoder.classes_)
    fallback = _fallback_label(known)
    y_clean = [lbl if lbl in known else fallback for lbl in y]
    y_encoded = encoder.transform(y_clean)
    return y_encoded.astype(np.int64), list(encoder.classes_), encoder


def get_class_distribution(
    y: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Return a dict with per-class count, proportion, and imbalance ratio."""
    y = np.asarray(y)
    total = len(y)
    if total == 0:
        return {
            "total_samples": 0,
            "n_classes": 0,
            "classes": {},
            "minority_classes": [],
            "majority_class": None,
            "imbalance_ratio": None,
        }

    unique, counts = np.unique(y, return_counts=True)
    info: dict = {
        "total_samples": total,
        "n_classes": len(unique),
        "classes": {},
        "minority_classes": [],
        "majority_class": None,
        "imbalance_ratio": None,
    }
    max_count, min_count = 0, float("inf")

    for cls_idx, count in zip(unique, counts):
        cls_name = class_names[cls_idx] if class_names else str(cls_idx)
        proportion = count / total
        percentage = proportion * 100
        info["classes"][cls_name] = {
            "index": int(cls_idx),
            "count": int(count),
            "proportion": round(float(proportion), 4),
            "percentage": round(float(percentage), 2),
        }
        if percentage < 1.0:
            info["minority_classes"].append(cls_name)
        if count > max_count:
            max_count = int(count)
            info["majority_class"] = cls_name
        if count < min_count:
            min_count = int(count)

    info["imbalance_ratio"] = round(max_count / min_count, 2) if min_count > 0 else float("inf")
    return info


def log_class_distribution(
    y: np.ndarray,
    class_names: list[str],
    prefix: str = " ",
) -> None:
    """Log class distribution with minority-class markers."""
    dist = get_class_distribution(y, class_names)
    logger.info(f"{prefix} Class distribution ({dist['n_classes']} classes):")
    for cls_name, info in dist["classes"].items():
        marker = " ← MINORITY" if cls_name in dist["minority_classes"] else ""
        logger.info(
            f"{prefix}   {cls_name}: {info['count']:>6} ({info['percentage']:>5.2f}%){marker}"
        )
    if dist["imbalance_ratio"] and dist["imbalance_ratio"] > 10:
        logger.warning(f"{prefix} High class imbalance (ratio: {dist['imbalance_ratio']}:1)")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _map_nsl_kdd_labels(y: np.ndarray) -> np.ndarray:
    mapped = []
    for label in y:
        label_str = str(label).lower().strip()
        if label_str in NSL_KDD_ATTACK_MAPPING:
            mapped.append(NSL_KDD_ATTACK_MAPPING[label_str])
        elif label_str in ("normal", "anomaly"):
            mapped.append("Normal" if label_str == "normal" else "Anomaly")
        else:
            mapped.append("Unknown")
    return np.array(mapped)


def _map_cicids_labels(y: np.ndarray) -> np.ndarray:
    mapped = []
    for label in y:
        label_str = str(label).strip()
        label_lower = (
            label_str.lower()
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\x96", "-")
            .replace("\x97", "-")
        )
        if label_lower in CICIDS_2018_LABEL_MAPPING:
            mapped.append(CICIDS_2018_LABEL_MAPPING[label_lower])
        elif label_lower in CICIDS_LABEL_MAPPING:
            mapped.append(CICIDS_LABEL_MAPPING[label_lower])
        elif "web attack" in label_lower:
            if "brute" in label_lower:
                mapped.append(WEBATTACK_BRUTEFORCE)
            elif "xss" in label_lower:
                mapped.append(WEBATTACK_XSS)
            elif "sql" in label_lower:
                mapped.append(WEBATTACK_SQLINJECTION)
            else:
                mapped.append(label_str)
        else:
            mapped.append(label_str)
    return np.array(mapped)


def _map_unsw_labels(y: np.ndarray) -> np.ndarray:
    mapped = []
    for label in y:
        label_str = str(label).strip()
        if label_str.lower() == "normal":
            mapped.append("Normal")
        elif label_str.lower() in ("backdoors", "backdoor"):
            mapped.append("Backdoor")
        else:
            mapped.append(label_str.capitalize() if label_str else "Unknown")
    return np.array(mapped)


def _map_to_unified_5class(y: np.ndarray, config: DatasetConfig) -> np.ndarray:
    mapped: list[str] = []
    for label in y:
        label_key = str(label).strip().lower()
        if config.name == "NSL-KDD":
            unified = label if label in UNIFIED_5CLASS else "Normal"
        elif config.name == "UNSW-NB15":
            unified = UNSW_TO_UNIFIED_5CLASS.get(label_key, "Normal")
        elif config.name.startswith("CICIDS-"):
            unified = CICIDS_TO_UNIFIED_5CLASS.get(label_key, "Normal")
        else:
            unified = "Normal"
        mapped.append(unified)
    return np.asarray(mapped)


def _fallback_label(known: set[str]) -> str:
    if "Unknown" in known:
        return "Unknown"
    if "Normal" in known:
        return "Normal"
    return next(iter(sorted(known)))
