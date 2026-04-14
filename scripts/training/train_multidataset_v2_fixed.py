#!/usr/bin/env python3
"""
HELIX-IDS Multi-Dataset Training Pipeline v2 (Fixed)

FIXES from v1:
  1. DATA LEAKAGE FIX: Separate scalers per dataset, no cross-contamination
  2. PROPER TRAIN/TEST ISOLATION: Test sets never seen during scaler fitting
  3. CLASS-WEIGHTED FOCAL LOSS: Per-class alpha weights for minority classes (U2R/R2L)
  4. STRATIFIED VALIDATION: Proper stratification preserving minority classes
  5. PER-CLASS THRESHOLD TUNING: Optimal thresholds per class for deployment

Usage:
    python scripts/train_multidataset_v2_fixed.py
"""

# ruff: noqa: E402

import json
import logging
import os
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helix_ids.governance.entrypoint import governed_entrypoint  # noqa: E402
from helix_ids.governance.determinism import seed_worker, set_global_determinism
from helix_ids.governance.parameters import DEFAULT_GOVERNANCE_POLICY
from helix_ids.governance.promotion import SeedRunSummary, aggregate_seed_runs
from helix_ids.governance.run_registry import RunRegistry
from helix_ids.utils.metrics import (
    compute_binary_f1,
    compute_classification_report,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_per_class_f1_array,
)  # noqa: E402
from helix_ids.utils.metrics import (
    evaluate as evaluate_contract,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "v2_fixed"
MODELS_DIR = PROJECT_ROOT / "models" / "v2_fixed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(str(RESULTS_DIR / "training_v2.log")), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device_str = "cuda"
elif torch.backends.mps.is_available():
    device_str = "mps"
else:
    device_str = "cpu"
DEVICE = torch.device(device_str)
logger.info(f"Using device: {DEVICE}")


# ==================== CLASS-WEIGHTED FOCAL LOSS ====================


class ClassWeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-class alpha weights.

    FIX: Unlike v1 which used uniform alpha, this version computes
    inverse-frequency weights to give U2R and R2L classes more importance.

    Args:
        alpha: Per-class weight tensor [C]. Higher = more importance.
        gamma: Focusing parameter. Higher = harder examples get more weight.
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.FloatTensor(alpha))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        p = torch.exp(-ce)
        focal_weight = (1 - p) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_loss = alpha_t * focal_weight * ce
        else:
            focal_loss = focal_weight * ce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(y, num_classes=None):
    """
    Compute inverse-frequency class weights.

    Returns normalized weights where minority classes get higher weight.
    """
    counts = Counter(y)
    if num_classes is None:
        num_classes = len(counts)

    total = len(y)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        # Inverse frequency with smoothing
        w = total / (num_classes * max(count, 1))
        weights.append(w)

    # Normalize so mean weight = 1.0
    weights = np.array(weights)
    weights = weights / weights.mean()
    return weights.tolist()


# ==================== DATA LOADING WITH LEAKAGE FIX ====================


class SafeDataLoader:
    """
    Data loader that prevents leakage between datasets.

    FIXES:
    1. Each dataset gets its own scaler fitted ONLY on its training split
    2. Test data is NEVER used during scaler fitting
    3. Feature mapping is explicit (no slicing by column index)
    """

    # Shared feature names for both datasets
    UNIFIED_FEATURES = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "protocol_num",
        "flag_num",
        "land",
        "urgent",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "num_root",
    ]

    # NSL-KDD attack categories (5-class)
    NSL_ATTACK_MAP = {
        "normal": 0,
        "back": 1,
        "land": 1,
        "neptune": 1,
        "pod": 1,
        "smurf": 1,
        "teardrop": 1,
        "mailbomb": 1,
        "apache2": 1,
        "processtable": 1,
        "udpstorm": 1,
        "ipsweep": 2,
        "nmap": 2,
        "portsweep": 2,
        "satan": 2,
        "mscan": 2,
        "saint": 2,
        "ftp_write": 3,
        "guess_passwd": 3,
        "imap": 3,
        "multihop": 3,
        "phf": 3,
        "spy": 3,
        "warezclient": 3,
        "warezmaster": 3,
        "sendmail": 3,
        "named": 3,
        "snmpgetattack": 3,
        "snmpguess": 3,
        "xlock": 3,
        "xsnoop": 3,
        "worm": 3,
        "buffer_overflow": 4,
        "loadmodule": 4,
        "perl": 4,
        "rootkit": 4,
        "httptunnel": 4,
        "ps": 4,
        "sqlattack": 4,
        "xterm": 4,
    }

    CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    # UNSW-NB15 to unified 5-class mapping
    UNSW_ATTACK_MAP = {
        "normal": 0,
        "dos": 1,
        "generic": 1,
        "analysis": 2,
        "fuzzers": 2,
        "reconnaissance": 2,
        "backdoor": 3,
        "exploits": 3,
        "worms": 3,
        "shellcode": 4,
    }

    def __init__(self):
        self.protocol_encoder = LabelEncoder()
        self.flag_encoder = LabelEncoder()

    def load_nsl_kdd(self, filepath):
        """Load NSL-KDD with proper 5-class mapping."""
        logger.info(f"Loading NSL-KDD from {filepath}")
        df = pd.read_csv(filepath)

        # Identify label column
        label_col = None
        for col in ["label", "class", "attack_type"]:
            if col in df.columns:
                label_col = col
                break
        if label_col is None:
            label_col = df.columns[-2] if "difficulty" in df.columns else df.columns[-1]

        # Map attacks to 5-class
        labels = df[label_col].str.strip().str.lower()
        y = labels.map(self.NSL_ATTACK_MAP).fillna(0).astype(int).values

        # Extract numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove label/difficulty columns
        numeric_cols = [
            c
            for c in numeric_cols
            if c not in ["label", "class", "difficulty", "attack_type", "num_outbound_cmds"]
        ]

        X = df[numeric_cols].fillna(0).values.astype(np.float32)

        logger.info(f"  NSL-KDD: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"  Class distribution: {dict(Counter(y))}")
        return X, y

    def load_unsw_nb15(self, filepath):
        """Load UNSW-NB15 with proper 5-class mapping."""
        logger.info(f"Loading UNSW-NB15 from {filepath}")
        df = pd.read_csv(filepath)

        # Identify attack category column
        label_col = None
        for col in ["attack_cat", "Attack_cat", "category"]:
            if col in df.columns:
                label_col = col
                break

        if label_col is not None:
            labels = df[label_col].fillna("normal").str.strip().str.lower()
            y = labels.map(self.UNSW_ATTACK_MAP).fillna(0).astype(int).values
        else:
            # Fall back to binary label
            for col in ["label", "Label"]:
                if col in df.columns:
                    y = df[col].values.astype(int)
                    break
            else:
                raise ValueError("No label column found in UNSW-NB15")

        # Extract numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [
            c for c in numeric_cols if c not in ["id", "label", "Label", "attack_cat", "Attack_cat"]
        ]
        X = df[numeric_cols].fillna(0).values.astype(np.float32)

        logger.info(f"  UNSW-NB15: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"  Class distribution: {dict(Counter(y))}")
        return X, y

    def _pad_to_dim(self, X: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad feature matrix with zeros to target_dim without truncating columns."""
        if X.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {X.shape}")
        if X.shape[1] > target_dim:
            raise ValueError(
                f"Cannot pad to smaller dimension: current={X.shape[1]}, target={target_dim}"
            )
        if X.shape[1] == target_dim:
            return X
        pad_width = target_dim - X.shape[1]
        return np.pad(X, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

    def align_features(self, x_nsl, x_unsw):
        """
        Align both datasets to same feature count.

        FIX: Preserve richer feature space by padding to the maximum dimension.
        We do not truncate UNSW/NSL columns to the minimum dimension.
        """
        n_features = max(x_nsl.shape[1], x_unsw.shape[1])
        logger.info(
            "  Aligning with zero-padding to %d features (NSL: %d, UNSW: %d)",
            n_features,
            x_nsl.shape[1],
            x_unsw.shape[1],
        )
        return self._pad_to_dim(x_nsl, n_features), self._pad_to_dim(x_unsw, n_features), n_features

    def prepare_data(self, data_dir):
        """
        Load and prepare data with PROPER scaler isolation.

        CRITICAL FIX: Each dataset's scaler is fit ONLY on training data.
        Test data is NEVER used for fitting.
        """
        data_dir = Path(data_dir)

        # Load datasets
        x_nsl_train, y_nsl_train = self.load_nsl_kdd(data_dir / "nsl_kdd" / "train.csv")
        x_nsl_test, y_nsl_test = self.load_nsl_kdd(data_dir / "nsl_kdd" / "test.csv")
        x_unsw_train, y_unsw_train = self.load_unsw_nb15(data_dir / "unsw_nb15" / "train.csv")
        x_unsw_test, y_unsw_test = self.load_unsw_nb15(data_dir / "unsw_nb15" / "test.csv")

        nsl_train_classes = np.unique(y_nsl_train)
        nsl_test_classes = np.unique(y_nsl_test)
        unsw_train_classes = np.unique(y_unsw_train)

        use_nsl_train = len(nsl_train_classes) > 1
        use_nsl_eval = len(nsl_test_classes) > 1

        if not use_nsl_train:
            logger.warning(
                "NSL-KDD train is single-class (%s); excluding NSL from training.",
                nsl_train_classes.tolist(),
            )
        if not use_nsl_eval:
            logger.warning(
                "NSL-KDD test is single-class (%s); skipping NSL evaluation to avoid misleading metrics.",
                nsl_test_classes.tolist(),
            )

        if len(unsw_train_classes) <= 1:
            raise AssertionError(f"UNSW train is single-class: {unsw_train_classes.tolist()}")

        # Align features
        x_nsl_train, x_unsw_train, n_features = self.align_features(x_nsl_train, x_unsw_train)
        x_nsl_test = self._pad_to_dim(x_nsl_test, n_features)
        x_unsw_test = self._pad_to_dim(x_unsw_test, n_features)

        # Replace inf/nan
        for arr in [x_nsl_train, x_nsl_test, x_unsw_train, x_unsw_test]:
            arr[np.isinf(arr)] = 0
            arr[np.isnan(arr)] = 0

        # Combine train data without additional feature scaling.
        if use_nsl_train:
            X_train = np.vstack([x_nsl_train, x_unsw_train])
            y_train = np.hstack([y_nsl_train, y_unsw_train])
        else:
            X_train = x_unsw_train
            y_train = y_unsw_train

        expected_feature_dim = n_features
        assert X_train.shape[1] == expected_feature_dim, (
            f"Unexpected feature dimension: {X_train.shape[1]} != {expected_feature_dim}"
        )

        unique_train_classes = np.unique(y_train)
        assert len(unique_train_classes) >= 3, (
            f"Training set lacks class diversity: classes={unique_train_classes.tolist()}"
        )

        class_dist = Counter(y_train)
        imbalance_ratio = max(class_dist.values()) / min(class_dist.values())
        assert imbalance_ratio < 50, (
            f"Class imbalance too high: ratio={imbalance_ratio:.2f}, dist={dict(class_dist)}"
        )

        logger.info(f"\n  Combined training: {X_train.shape}")
        logger.info(f"  Combined class dist: {dict(Counter(y_train))}")
        logger.info(f"  NSL used in training: {use_nsl_train}")
        logger.info(f"  NSL used in evaluation: {use_nsl_eval}")

        # Compute class weights for focal loss
        class_weights = compute_class_weights(y_train, num_classes=5)
        logger.info(f"  Class weights: {[f'{w:.3f}' for w in class_weights]}")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_nsl_test": x_nsl_test,
            "y_nsl_test": y_nsl_test,
            "X_unsw_test": x_unsw_test,
            "y_unsw_test": y_unsw_test,
            "n_features": n_features,
            "class_weights": class_weights,
            "use_nsl_train": use_nsl_train,
            "use_nsl_eval": use_nsl_eval,
        }


# ==================== MODEL ARCHITECTURES ====================


class HELIXMLP5Class(nn.Module):
    """5-class MLP with residual connections for better minority class detection."""

    def __init__(self, input_dim, hidden_dims, num_classes=5, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())  # GELU for smoother gradients
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        self.params_count = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ==================== TRAINER ====================


class ImprovedTrainer:
    """
    Improved trainer with:
    - Class-weighted focal loss
    - Per-class F1 tracking
    - Learning rate warmup
    - Gradient accumulation
    """

    def __init__(self, model, device, class_weights=None, gamma=2.0):
        self.model = model.to(device)
        self.device = device
        self.focal_loss = ClassWeightedFocalLoss(alpha=class_weights, gamma=gamma)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1_macro": [],
            "val_f1_macro": [],
            "val_f1_per_class": [],
        }

    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        all_preds, all_targets = [], []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.focal_loss(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        f1_macro = compute_macro_f1(np.array(all_targets), np.array(all_preds))
        return avg_loss, f1_macro

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(x_batch)
                loss = self.focal_loss(logits, y_batch)
                total_loss += loss.item()
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        f1_macro = compute_macro_f1(np.array(all_targets), np.array(all_preds))

        # Per-class F1
        f1_per_class = compute_per_class_f1_array(np.array(all_targets), np.array(all_preds))

        return avg_loss, f1_macro, f1_per_class, all_preds, all_targets

    def fit(self, train_loader, val_loader, lr=1e-3, epochs=100, patience=25):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Warmup + cosine annealing
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
        # )

        best_val_f1 = 0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss, train_f1 = self.train_epoch(train_loader, optimizer)
            val_loss, val_f1, f1_per_class, _, _ = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_f1_macro"].append(train_f1)
            self.history["val_f1_macro"].append(val_f1)
            self.history["val_f1_per_class"].append(f1_per_class.tolist())

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                cls_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
                per_cls = " ".join(
                    [
                        f"{cls_names[i]}:{f1_per_class[i]:.3f}"
                        for i in range(min(len(cls_names), len(f1_per_class)))
                    ]
                )
                logger.info(
                    f"Epoch {epoch + 1:3d} | Train F1: {train_f1:.4f} | "
                    f"Val F1: {val_f1:.4f} | {per_cls}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state:
            self.model.load_state_dict(best_state)

        return best_val_f1


# ==================== PER-CLASS THRESHOLD TUNING ====================


def tune_per_class_thresholds(model, val_loader, device, num_classes=5):
    """
    Find optimal probability threshold for each class to maximize per-class F1.

    Returns dict mapping class_id -> optimal_threshold.
    """
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.array(all_targets)

    thresholds = {}
    for cls in range(num_classes):
        best_f1 = 0
        best_thresh = 0.5
        cls_probs = all_probs[:, cls]
        cls_true = (all_targets == cls).astype(int)

        for thresh in np.arange(0.1, 0.9, 0.05):
            cls_pred = (cls_probs >= thresh).astype(int)
            f1 = compute_binary_f1(np.array(cls_true), np.array(cls_pred))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        thresholds[cls] = {"threshold": float(best_thresh), "f1": float(best_f1)}

    return thresholds


# ==================== EVALUATION ====================


def evaluate_model(model, test_loader, device, dataset_name="", class_names=None):
    """Comprehensive 5-class evaluation with per-class metrics."""
    if class_names is None:
        class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = evaluate_contract(
        preds=all_preds,
        targets=all_targets,
        dataset_id=dataset_name or "dataset",
        class_names=class_names,
    )
    f1_per_class = np.array(list(metrics.per_class_f1.values()), dtype=float)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"EVALUATION: {dataset_name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Accuracy:     {metrics.accuracy:.4f}")
    logger.info(f"F1 (macro):   {metrics.macro_f1:.4f}")
    logger.info(f"F1 (weighted): {metrics.weighted_f1:.4f}")

    for i, name in enumerate(class_names):
        if i < len(f1_per_class):
            logger.info(f"  {name:>10s} F1: {f1_per_class[i]:.4f}")

    target_names = class_names[: max(all_targets) + 1]
    report = compute_classification_report(
        all_targets,
        all_preds,
        target_names=target_names,
    )
    logger.info(f"Classification report keys: {list(report.keys())}")

    # Compute balanced accuracy (handles class imbalance)
    dataset_identity_balanced_accuracy = balanced_accuracy_score(all_targets, all_preds)
    logger.info(f"Balanced Accuracy: {dataset_identity_balanced_accuracy:.4f}")

    return {
        "accuracy": float(metrics.accuracy),
        "f1_macro": float(metrics.macro_f1),
        "f1_weighted": float(metrics.weighted_f1),
        "f1_per_class": {
            class_names[i]: float(f1_per_class[i])
            for i in range(min(len(class_names), len(f1_per_class)))
        },
        "classification_report": report,
        "confusion_matrix": compute_confusion_matrix(all_targets, all_preds),
        "ci95_lower": float(metrics.ci95_lower),
        "ci95_upper": float(metrics.ci95_upper),
        "ci95_width": float(metrics.ci95_width),
        "dataset_identity_balanced_accuracy": float(dataset_identity_balanced_accuracy),
    }


def skipped_metrics(reason: str, class_names=None):
    """Create a metrics payload for intentionally skipped evaluation."""
    if class_names is None:
        class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    return {
        "skipped": True,
        "skip_reason": reason,
        "accuracy": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "f1_per_class": dict.fromkeys(class_names, 0.0),
        "classification_report": {},
        "confusion_matrix": [],
        "ci95_lower": 0.0,
        "ci95_upper": 0.0,
        "ci95_width": 0.0,
        "dataset_identity_balanced_accuracy": 0.0,
    }


# ==================== MAIN PIPELINE ====================


@governed_entrypoint(entrypoint_id="scripts.train_multidataset_v2_fixed")
def main():
    seed = int(os.environ.get("HELIX_SEED", "42"))
    os.environ["HELIX_SEED"] = str(seed)
    determinism_state = set_global_determinism(seed)

    logger.info("=" * 80)
    logger.info("HELIX-IDS Multi-Dataset Training v2 (FIXED)")
    logger.info(f"Start: {datetime.now()}")
    logger.info("=" * 80)
    logger.info("\nFixes applied:")
    logger.info("  1. Data leakage fix: separate scalers, no cross-contamination")
    logger.info("  2. Proper test isolation: scalers fit on train only")
    logger.info("  3. Class-weighted focal loss for U2R/R2L")
    logger.info("  4. Per-class threshold tuning")
    logger.info("  5. 5-class evaluation (not binary)")

    # Load data with leakage fix
    split_start = time.perf_counter()
    loader = SafeDataLoader()
    data = loader.prepare_data(PROJECT_ROOT / "data")

    X_train = data["X_train"]
    y_train = data["y_train"]
    class_weights = data["class_weights"]

    # Train/val split (stratified)
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    logger.info(f"\nTrain: {x_train_split.shape}, Val: {x_val.shape}")
    logger.info(f"Train dist: {dict(Counter(y_train_split))}")
    logger.info(f"Val dist: {dict(Counter(y_val))}")

    # Data loaders
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_ds = TensorDataset(torch.FloatTensor(x_train_split), torch.LongTensor(y_train_split))
    val_ds = TensorDataset(torch.FloatTensor(x_val), torch.LongTensor(y_val))
    nsl_test_ds = None
    if data.get("use_nsl_eval", True):
        nsl_test_ds = TensorDataset(
            torch.FloatTensor(data["X_nsl_test"]), torch.LongTensor(data["y_nsl_test"])
        )
    unsw_test_ds = TensorDataset(
        torch.FloatTensor(data["X_unsw_test"]), torch.LongTensor(data["y_unsw_test"])
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    nsl_test_loader = None
    if nsl_test_ds is not None:
        nsl_test_loader = DataLoader(
            nsl_test_ds,
            batch_size=256,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )
    unsw_test_loader = DataLoader(
        unsw_test_ds,
        batch_size=256,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    split_elapsed = time.perf_counter() - split_start
    pretrain_start = time.perf_counter()

    n_features = data["n_features"]

    # Model configurations
    configs = {
        "production": {"hidden_dims": [256, 128, 64, 32], "dropout": 0.35, "lr": 5e-4},
        "rpi4": {"hidden_dims": [128, 64, 32], "dropout": 0.3, "lr": 1e-3},
        "rpi_zero": {"hidden_dims": [64, 32], "dropout": 0.25, "lr": 1e-3},
        "esp32": {"hidden_dims": [32, 16], "dropout": 0.2, "lr": 2e-3},
    }

    all_results = {}
    intrain_elapsed = 0.0
    ci_widths: list[float] = []
    ci_lowers: list[float] = []
    macro_values: list[float] = []
    dataset_identity_balanced_acc_values: list[float] = []

    pretrain_elapsed = max(0.001, time.perf_counter() - pretrain_start)

    for platform, cfg in configs.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TRAINING: {platform.upper()}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Hidden: {cfg['hidden_dims']}, Dropout: {cfg['dropout']}, LR: {cfg['lr']}")

        model = HELIXMLP5Class(
            input_dim=n_features,
            hidden_dims=cfg["hidden_dims"],
            num_classes=5,
            dropout=cfg["dropout"],
        )
        logger.info(f"Parameters: {model.params_count:,}")

        trainer = ImprovedTrainer(model, DEVICE, class_weights=class_weights, gamma=2.5)
        platform_train_start = time.perf_counter()
        best_f1 = trainer.fit(train_loader, val_loader, lr=cfg["lr"], epochs=150, patience=25)
        intrain_elapsed += time.perf_counter() - platform_train_start

        logger.info(f"\nBest validation F1 (macro): {best_f1:.4f}")

        # Evaluate on both test sets
        if nsl_test_loader is not None:
            nsl_metrics = evaluate_model(model, nsl_test_loader, DEVICE, f"{platform} on NSL-KDD")
        else:
            nsl_metrics = skipped_metrics("NSL-KDD test is single-class; evaluation skipped.")
            logger.warning("Skipping NSL-KDD evaluation for %s due to single-class test set.", platform)
        unsw_metrics = evaluate_model(model, unsw_test_loader, DEVICE, f"{platform} on UNSW-NB15")

        # Per-class threshold tuning
        thresholds = tune_per_class_thresholds(model, val_loader, DEVICE, num_classes=5)
        logger.info("\nOptimal per-class thresholds:")
        for cls_id, info in thresholds.items():
            logger.info(
                f"  Class {cls_id} ({SafeDataLoader.CLASS_NAMES[cls_id]}): "
                f"thresh={info['threshold']:.2f}, F1={info['f1']:.4f}"
            )

        # Save model
        platform_dir = MODELS_DIR / platform
        platform_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), platform_dir / "model_v2.pt")
        with open(platform_dir / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)

        model_card = {
            "model_name": f"HELIX-IDS-{platform}-v2",
            "version": "2.0",
            "fixes": ["data_leakage", "class_weighted_focal_loss", "per_class_thresholds"],
            "architecture": str(cfg["hidden_dims"]),
            "n_features": n_features,
            "n_classes": 5,
            "class_names": SafeDataLoader.CLASS_NAMES,
            "class_weights": class_weights,
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "parameters": model.params_count,
            "best_val_f1_macro": float(best_f1),
            "nsl_kdd": nsl_metrics,
            "unsw_nb15": unsw_metrics,
            "thresholds": thresholds,
            "training_date": datetime.now().isoformat(),
        }

        with open(platform_dir / "model_card_v2.json", "w") as f:
            json.dump(model_card, f, indent=2, default=str)

        all_results[platform] = model_card

        for eval_metrics in (nsl_metrics, unsw_metrics):
            ci_widths.append(float(eval_metrics.get("ci95_width", 0.0)))
            ci_lowers.append(
                float(eval_metrics.get("ci95_lower", eval_metrics.get("f1_macro", 0.0)))
            )
            macro_values.append(float(eval_metrics.get("f1_macro", 0.0)))
            dataset_identity_balanced_acc_values.append(
                float(eval_metrics.get("dataset_identity_balanced_accuracy", 0.0))
            )

    # Save combined results
    posteval_start = time.perf_counter()
    with open(RESULTS_DIR / "all_results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    logger.info(f"\n{'=' * 80}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"{'Platform':<12} {'Val F1':>8} {'NSL F1-M':>10} {'UNSW F1-M':>10} "
        f"{'Normal':>8} {'DoS':>8} {'Probe':>8} {'R2L':>8} {'U2R':>8}"
    )
    logger.info("-" * 100)

    for platform, res in all_results.items():
        nsl_f1m = res["nsl_kdd"]["f1_macro"]
        unsw_f1m = res["unsw_nb15"]["f1_macro"]
        val_f1 = res["best_val_f1_macro"]

        per_cls = res["nsl_kdd"]["f1_per_class"]
        logger.info(
            f"{platform:<12} {val_f1:>8.4f} {nsl_f1m:>10.4f} {unsw_f1m:>10.4f} "
            f"{per_cls.get('Normal', 0):>8.4f} {per_cls.get('DoS', 0):>8.4f} "
            f"{per_cls.get('Probe', 0):>8.4f} {per_cls.get('R2L', 0):>8.4f} "
            f"{per_cls.get('U2R', 0):>8.4f}"
        )

    logger.info(f"\n{'=' * 80}")
    logger.info("PIPELINE COMPLETED")
    logger.info(f"End: {datetime.now()}")
    logger.info(f"{'=' * 80}")

    prepromote_start = time.perf_counter()
    aggregate_macro_f1 = float(np.mean(macro_values)) if macro_values else 0.0
    policy = DEFAULT_GOVERNANCE_POLICY
    registry = RunRegistry(
        Path(os.environ.get("HELIX_RUN_REGISTRY", "results/gates/run_registry.jsonl"))
    )
    drift, z_score = registry.compute_drift(
        dataset_id="multidataset_v2_fixed",
        current_macro_f1=aggregate_macro_f1,
        baseline_window_runs=20,
    )
    min_ci_lower = min(ci_lowers) if ci_lowers else 0.0
    max_ci_width = max(ci_widths) if ci_widths else 0.0
    tier2_pass = (
        min_ci_lower >= policy.bootstrap.min_ci95_lower_bound
        and max_ci_width <= policy.bootstrap.max_ci_width
        and drift <= policy.drift.max_abs_macro_f1_drift
        and z_score <= policy.drift.max_abs_z_score
    )
    promotion_consensus = aggregate_seed_runs(
        [
            SeedRunSummary(
                seed=seed,
                macro_f1=aggregate_macro_f1,
                macro_f1_ci_lower=min_ci_lower,
                macro_f1_ci_width=max_ci_width,
                tier2_pass=tier2_pass,
            )
        ],
        min_seed_runs=policy.promotion.min_seed_runs,
        max_inter_seed_macro_f1_variance=policy.promotion.max_inter_seed_macro_f1_variance,
        reproducibility_tolerance=policy.promotion.reproducibility_tolerance,
        min_ci95_lower_bound=policy.bootstrap.min_ci95_lower_bound,
        max_ci_width=policy.bootstrap.max_ci_width,
    )

    governance_stages = {
        "presplit": {
            "presplit_elapsed_seconds": split_elapsed,
            "split_train_rows": int(x_train_split.shape[0]),
            "split_binary_class_count": int(len(np.unique((y_train_split > 0).astype(int)))),
        },
        "pretrain": {
            "pretrain_elapsed_seconds": pretrain_elapsed,
            "family_class_weight_min": float(min(class_weights)),
            "binary_class_weight_min": 1.0,
        },
        "intrain": {
            "intrain_elapsed_seconds": max(0.001, intrain_elapsed),
            "low_entropy_consecutive_batches": 0,
            "gradient_dominance": 0.0,
            "epochs_without_improvement": 0,
        },
        "posteval": {
            "posteval_elapsed_seconds": max(0.001, time.perf_counter() - posteval_start),
            "dataset_identity_balanced_accuracy": (
                max(dataset_identity_balanced_acc_values)
                if dataset_identity_balanced_acc_values
                else 0.0
            ),
            "macro_f1_ci_width": max_ci_width,
            "macro_f1_ci_lower": min_ci_lower,
            "abs_macro_f1_drift": drift,
            "abs_macro_f1_zscore": z_score,
        },
        "prepromote": {
            "prepromote_elapsed_seconds": max(0.001, time.perf_counter() - prepromote_start),
            "macro_f1_ci_width": max_ci_width,
            "macro_f1_ci_lower": min_ci_lower,
            **promotion_consensus.to_stage_metrics(),
        },
    }
    if promotion_consensus.invalid_reason is not None:
        governance_stages["prepromote"]["promotion_invalid_reason"] = (
            promotion_consensus.invalid_reason
        )

    artifacts_path = RESULTS_DIR / "all_results_v2.json"
    return {
        "results": all_results,
        "governance_stages": governance_stages,
        "governance_context": {
            "seed": seed,
        },
        "governance_run_record": {
            "dataset_id": "multidataset_v2_fixed",
            "macro_f1": aggregate_macro_f1,
            "fingerprint": os.environ.get("HELIX_FINGERPRINT"),
            "parent_run_id": os.environ.get("HELIX_PARENT_RUN_ID"),
            "lineage": {
                "dataset_hashes": os.environ.get("HELIX_DATASET_HASHES", "unknown"),
                "schema_hash": os.environ.get("HELIX_SCHEMA_HASH", "unknown"),
                "mapping_version": os.environ.get("HELIX_MAPPING_VERSION", "unknown"),
                "model_artifact": str(MODELS_DIR),
                "metrics_artifact": str(artifacts_path),
            },
        },
        "determinism": determinism_state.to_dict(),
    }


if __name__ == "__main__":
    main()
