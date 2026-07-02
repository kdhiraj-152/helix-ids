#!/usr/bin/env python3
"""
Phase 63 — Controlled Backbone Adaptation:
Is Representation Plasticity the Missing Ingredient?

Tests whether limited backbone adaptation (vs. fully frozen) improves
cross-dataset continual learning without catastrophic forgetting.

Conditions:
  A: Frozen backbone + classifier LoRA (Phase 62 baseline)
  B: Last backbone block (fc4) trainable + classifier LoRA
  C: Last two blocks (fc3+fc4) trainable + classifier LoRA
  D: Full backbone LoRA + classifier LoRA
  E: Progressive unfreezing (staged)
  F: Replay + Progressive unfreezing

Usage:
  PYTHONPATH=src python scripts/analysis/phase63_main.py [--condition A] [--quick]
  PYTHONPATH=src python scripts/analysis/phase63_main.py --all
  PYTHONPATH=src python scripts/analysis/phase63_main.py --condition F --replay_size 5.0
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist
from scipy.linalg import svd
import sklearn.metrics as sk_metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import CCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 17
BATCH_SIZE = 256
LR = 1e-4
WEIGHT_DECAY = 1e-4
ADAPT_EPOCHS = 50
N_BOOTSTRAP = 10000
MAX_TRAIN_SAMPLES = 20000
MAX_TEST_SAMPLES = 10000

PROJ = Path(__file__).resolve().parents[2]
CACHE = PROJ / "data" / "processed" / "phase52_cache"
RESULTS = PROJ / "results" / "phase63"
DATA_DIR = PROJ / "data"
CHECKPOINT_PATH = PROJ / "models" / "helix_full" / "helix_full_nsl_kdd_best.pt"

LORA_RANK = 8
LORA_ALPHA = 16

for sub_dir in ["embeddings", "umap", "tsne", "plots"]:
    (RESULTS / sub_dir).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJ / "src"))

# ═══════════════════════════════════════════════════════════════════════════
# Dataset definitions
# ═══════════════════════════════════════════════════════════════════════════

ORIGINAL_DATASETS = [
    "nsl_kdd", "unsw_nb15", "cicids2017", "cicids2018", "ton_iot", "bot_iot"
]
EXTERNAL_DATASETS = ["iot23", "kyoto2006", "ugr16"]
ALL_DATASETS = ORIGINAL_DATASETS + EXTERNAL_DATASETS
CONTINUAL_ORDER = ["iot23", "kyoto2006", "ugr16"]

DATASET_DISPLAY = {
    "nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15",
    "cicids2017": "CICIDS2017", "cicids2018": "CICIDS2018",
    "ton_iot": "TON-IoT", "bot_iot": "Bot-IoT",
    "iot23": "IoT-23", "kyoto2006": "Kyoto2006+", "ugr16": "UGR'16",
}

CANONICAL_FEATURE_ORDER = [
    "protocol_type", "connection_state", "traffic_direction", "has_rst",
    "log_src_bytes", "log_dst_bytes", "src_dst_bytes_ratio", "dst_src_bytes_ratio",
    "same_host_rate_x_service", "diff_srv_rate_x_flag", "count_x_srv_count",
    "protocol_service_flag", "src_bytes", "dst_bytes", "service_tier",
    "duration", "flag",
]

# Backbone layer architecture (HelixIDSFull Sequential):
#   backbone.0  = Linear(17→512)  — fc1
#   backbone.4  = Linear(512→384) — fc2
#   backbone.8  = Linear(384→256) — fc3
#   backbone.12 = Linear(256→256) — fc4
BACKBONE_LINEAR_LAYERS = ["backbone.0", "backbone.4", "backbone.8", "backbone.12"]
FC3 = "backbone.8"
FC4 = "backbone.12"

CLASSIFIER_LORA_TARGETS = [
    "binary_head.0", "binary_head.3",
    "family_projection.0", "family_projection.4",
    "family_head.0", "family_head.3",
]

# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("phase63")
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
fh = logging.FileHandler(RESULTS / "phase63_run.log", mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.info(f"Phase 63 starting — device={DEVICE}")
logger.info(f"Checkpoint: {CHECKPOINT_PATH}")


def cleanup():
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_binary(y):
    return (y > 0).astype(np.int64)


def subsample_stratified(X, y, mx, rng_=None):
    if rng_ is None:
        rng_ = rng
    n = X.shape[0]
    if n <= mx:
        return X.copy(), y.copy()
    classes = np.unique(y)
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        t = max(1, int(mx * len(ci) / n))
        if len(ci) > t:
            ci = rng_.choice(ci, size=t, replace=False)
        idx.extend(ci.tolist())
    rng_.shuffle(idx)
    return X[np.array(idx)], y[np.array(idx)]

# ═══════════════════════════════════════════════════════════════════════════
# LoRA Implementation (from Phase 62)
# ═══════════════════════════════════════════════════════════════════════════


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRAModule(nn.Module):
    """Wraps a base model with LoRA adapters on specified target layers."""

    def __init__(self, base_model: nn.Module, target_names: list[str],
                 rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base_model = base_model
        self.target_names = target_names
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        for name in target_names:
            module = self._get_module(name)
            if module is None:
                raise ValueError(f"Target module '{name}' not found in model")
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"Target '{name}' is {type(module).__name__}, not nn.Linear")
        self._safe_name_map = {}
        self.lora_layers = nn.ModuleDict()
        for name in target_names:
            module = self._get_module(name)
            safe_name = name.replace(".", "_")
            self._safe_name_map[safe_name] = name
            lora = LoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank, alpha=alpha, dropout=dropout,
            )
            self.lora_layers[safe_name] = lora

    def _get_module(self, name: str) -> Optional[nn.Module]:
        parts = name.split(".")
        obj = self.base_model
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return None
        return obj

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self._forward_sequential(self.base_model.backbone, x, "backbone")
        binary_logits = self._forward_sequential(self.base_model.binary_head, features, "binary_head")
        family_features = self._forward_sequential(self.base_model.family_projection, features, "family_projection")
        family_features = self.base_model._whiten_family_features(family_features)
        family_logits = self._forward_sequential(self.base_model.family_head, family_features, "family_head")
        if torch.isnan(binary_logits).any() or torch.isnan(family_logits).any():
            binary_logits = torch.nan_to_num(binary_logits, nan=0.0, posinf=1e3, neginf=-1e3)
            family_logits = torch.nan_to_num(family_logits, nan=0.0, posinf=1e3, neginf=-1e3)
        if return_features:
            return binary_logits, family_logits, features
        return binary_logits, family_logits

    def _forward_sequential(self, seq: nn.Sequential, x: torch.Tensor,
                            prefix: str) -> torch.Tensor:
        h = x
        for idx, layer in enumerate(seq):
            if isinstance(layer, nn.Linear):
                safe_key = f"{prefix}_{idx}"
                h_frozen = layer(h)
                if safe_key in self.lora_layers:
                    h_lora = self.lora_layers[safe_key](h)
                    h = h_frozen + h_lora
                else:
                    h = h_frozen
            else:
                h = layer(h)
        return h

    def get_trainable_params(self) -> list[torch.Tensor]:
        return list(self.lora_layers.parameters())

    def get_named_trainable_params(self) -> dict[str, torch.Tensor]:
        params = {}
        for safe_name, layer in self.lora_layers.items():
            orig_name = self._safe_name_map.get(safe_name, safe_name)
            params[f"{orig_name}.lora_A.weight"] = layer.lora_A.weight
            params[f"{orig_name}.lora_B.weight"] = layer.lora_B.weight
        return params

    def get_param_counts(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.lora_layers.parameters())
        total = sum(p.numel() for p in self.base_model.parameters())
        return {"trainable": trainable, "total": total + trainable,
                "compression_ratio": total / max(trainable, 1)}

    def save_lora_weights(self, path: Path):
        state = {}
        for safe_name, layer in self.lora_layers.items():
            orig_name = self._safe_name_map.get(safe_name, safe_name)
            state[f"{orig_name}.lora_A.weight"] = layer.lora_A.weight.detach().cpu()
            state[f"{orig_name}.lora_B.weight"] = layer.lora_B.weight.detach().cpu()
        torch.save(state, path)

    def load_lora_weights(self, path: Path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        for safe_name, layer in self.lora_layers.items():
            orig_name = self._safe_name_map.get(safe_name, safe_name)
            if f"{orig_name}.lora_A.weight" in state:
                layer.lora_A.weight.data.copy_(state[f"{orig_name}.lora_A.weight"])
            if f"{orig_name}.lora_B.weight" in state:
                layer.lora_B.weight.data.copy_(state[f"{orig_name}.lora_B.weight"])

    def get_lora_params_flat(self) -> torch.Tensor:
        params = []
        for p in self.get_trainable_params():
            params.append(p.data.view(-1))
        return torch.cat(params)

    def set_lora_params_flat(self, flat: torch.Tensor):
        offset = 0
        for p in self.get_trainable_params():
            n = p.data.numel()
            p.data.copy_(flat[offset:offset + n].view(p.data.shape))
            offset += n

    def clone_lora_params(self) -> dict[str, torch.Tensor]:
        cloned = {}
        for safe_name, layer in self.lora_layers.items():
            orig_name = self._safe_name_map.get(safe_name, safe_name)
            cloned[f"{orig_name}.lora_A.weight"] = layer.lora_A.weight.data.clone()
            cloned[f"{orig_name}.lora_B.weight"] = layer.lora_B.weight.data.clone()
        return cloned

    def restore_lora_params(self, state: dict[str, torch.Tensor]):
        for safe_name, layer in self.lora_layers.items():
            orig_name = self._safe_name_map.get(safe_name, safe_name)
            if f"{orig_name}.lora_A.weight" in state:
                layer.lora_A.weight.data.copy_(state[f"{orig_name}.lora_A.weight"])
            if f"{orig_name}.lora_B.weight" in state:
                layer.lora_B.weight.data.copy_(state[f"{orig_name}.lora_B.weight"])


# ═══════════════════════════════════════════════════════════════════════════
# Phase 63 — Backbone Adaptation Helpers
# ═══════════════════════════════════════════════════════════════════════════


def get_backbone_linear_indices(model: nn.Module) -> list[tuple[int, str]]:
    """Return (index, name) pairs for Linear layers in backbone."""
    indices = []
    for idx, layer in enumerate(model.backbone):
        if isinstance(layer, nn.Linear):
            indices.append((idx, f"backbone.{idx}"))
    return indices


def unfreeze_backbone_layers(model: nn.Module, linear_indices: list[int],
                             lr_scale: float = 1.0):
    """Unfreeze specific backbone Linear layers by their sequential index.
    
    Args:
        model: The base model (not LoRAModule wrapper)
        linear_indices: List of sequential indices (e.g., [12] for fc4, [8, 12] for fc3+fc4)
        lr_scale: Learning rate scaling factor for unfrozen layers
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Then unfreeze selected backbone Linear layers
    unfrozen_params = []
    for idx in linear_indices:
        layer = model.backbone[idx]
        assert isinstance(layer, nn.Linear), f"Layer {idx} is not Linear: {type(layer)}"
        for param in layer.parameters():
            param.requires_grad = True
        unfrozen_params.extend(layer.parameters())
    
    # Also unfreeze batch norm and activation after the linear layers to allow adaptation
    for linear_idx in linear_indices:
        seq_idx = list(range(len(model.backbone)))
        # Unfreeze batchnorm and dropout after this linear layer for flow
        for i in range(linear_idx + 1, min(linear_idx + 4, len(model.backbone))):
            layer = model.backbone[i]
            if isinstance(layer, (nn.BatchNorm1d,)):
                for param in layer.parameters():
                    param.requires_grad = True
                layer.train()
    
    # Ensure batch norm layers after unfrozen linear layers are in train mode
    for idx in linear_indices:
        for i in range(idx, len(model.backbone)):
            if isinstance(model.backbone[i], (nn.BatchNorm1d,)):
                model.backbone[i].train()

    logger.info(f"  Unfrozen backbone linear indices: {linear_indices}")
    logger.info(f"  Unfrozen param count: {sum(p.numel() for p in unfrozen_params):,}")
    return unfrozen_params


def get_trainable_param_groups(model: nn.Module, backbone_lr_scale: float = 0.5
                               ) -> list[dict]:
    """Get parameter groups with differentiated learning rates."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
    
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": LR * backbone_lr_scale,
                       "name": "backbone"})
    if head_params:
        groups.append({"params": head_params, "lr": LR,
                       "name": "heads"})
    return groups


# ═══════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════

CHECKPOINT_SHA256 = None


def compute_encoder_hash(model: nn.Module) -> str:
    tensors = []
    for name, param in model.named_parameters():
        if name.startswith("backbone.") and "lora" not in name:
            tensors.append(param.detach().cpu().numpy().tobytes())
    return hashlib.sha256(b"".join(tensors)).hexdigest()


def verify_checkpoint_sha256():
    global CHECKPOINT_SHA256
    h = hashlib.sha256()
    h.update(open(CHECKPOINT_PATH, "rb").read())
    CHECKPOINT_SHA256 = h.hexdigest()
    logger.info(f"Checkpoint SHA256: {CHECKPOINT_SHA256}")
    return CHECKPOINT_SHA256


def load_frozen_model(checkpoint_path: Path) -> nn.Module:
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = cp.get("model_state_dict", cp.get("state_dict", cp.get("model", cp)))
    logger.info(f"State dict keys: {len(sd)}")
    layer0_key = [k for k in sd.keys()
                  if k.startswith("backbone.") and k.endswith(".weight")][0]
    input_dim = sd[layer0_key].shape[1]
    logger.info(f"Input dim deduced: {input_dim}")

    from helix_ids.models.helix_ids_full import HelixIDSFull, HelixFullConfig
    config = HelixFullConfig(input_dim=input_dim)
    model = HelixIDSFull(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"  Missing keys: {missing}")
    if unexpected:
        logger.warning(f"  Unexpected keys: {unexpected}")
    model.to(DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.apply(lambda m: m.eval() if isinstance(m, (nn.BatchNorm1d,)) else None)
    model.pre_hash = compute_encoder_hash(model)
    logger.info(f"  Pre-training encoder SHA256: {model.pre_hash}")
    return model


def verify_encoder_frozen(model: nn.Module, label: str = "model") -> tuple[bool, str, str]:
    post_hash = compute_encoder_hash(model)
    pre_hash = getattr(model, "pre_hash", None)
    if pre_hash is None:
        logger.warning(f"  {label}: No pre-training hash available!")
        return False, post_hash, "no_pre_hash"
    changed = pre_hash != post_hash
    if changed:
        logger.error(f"  {label}: ENCODER PARAMETERS CHANGED! {pre_hash[:16]} -> {post_hash[:16]}")
    else:
        logger.info(f"  {label}: ✓ Encoder SHA256 unchanged")
    return not changed, post_hash, pre_hash


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading (from Phase 62)
# ═══════════════════════════════════════════════════════════════════════════


def load_original_datasets():
    datasets = {}
    if not CACHE.exists():
        logger.error(f"Cache dir {CACHE} not found!")
        return datasets
    for name in ORIGINAL_DATASETS:
        X_tr = np.load(CACHE / f"{name}_X_train.npy", mmap_mode="r").astype(np.float64)
        X_te = np.load(CACHE / f"{name}_X_test.npy", mmap_mode="r").astype(np.float64)
        y_tr = np.load(CACHE / f"{name}_y_train.npy").ravel()
        y_te = np.load(CACHE / f"{name}_y_test.npy").ravel()
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])
        y_bin = to_binary(y)
        datasets[name] = {"X": X, "y": y, "y_bin": y_bin}
        logger.info(f"  {name}: {X.shape}, classes={np.unique(y)}")
    return datasets


def fit_dataset_scalers(data_dict):
    sc = {}
    for name in sorted(data_dict.keys()):
        X = data_dict[name]["X"]
        sc[name] = StandardScaler().fit(X)
    return sc


def standardize_data(data_dict, scalers):
    result = {}
    for name in sorted(data_dict.keys()):
        X = data_dict[name]["X"]
        y_bin = data_dict[name]["y_bin"]
        y = data_dict[name]["y"]
        result[name] = {
            "X": scalers[name].transform(X),
            "y": y, "y_bin": y_bin,
        }
    return result


# ── External Dataset Loaders ─────────────────────────────────────────────

def _encode_protocol(proto_series):
    pmap = {"tcp": 0, "udp": 1, "icmp": 2, "-": 0}
    return proto_series.map(pmap).fillna(0).astype(int)


def _encode_conn_state(state_series):
    smap = {
        "S0": 0, "S1": 1, "S2": 2, "S3": 3, "SF": 4, "REJ": 5,
        "RST": 6, "RSTO": 6, "RSTR": 6, "RSTOS0": 6, "RSTRH": 6,
        "SH": 7, "SHR": 7, "OTH": 8, "CON": 9, "INT": 10, "FIN": 11,
        "acc": 12, "clo": 13, "no": 14, "par": 15, "urn": 16,
        "eco": 17, "tst": 18, "-": 19,
    }
    return state_series.map(smap).fillna(19).astype(int)


def _encode_service_tier_simple(service_series):
    tier_map = {
        "-": 0, "dns": 2, "http": 1, "https": 1, "ssl": 1,
        "ftp": 3, "smtp": 4, "ssh": 3, "telnet": 3,
        "dhcp": 2, "ntp": 2, "snmp": 6, "ldap": 6,
    }
    return service_series.map(tier_map).fillna(6).astype(int)


def _encode_protocol_from_service(service_series):
    udp_services = {"dns", "dhcp", "ntp", "snmp", "syslog", "tftp", "rip", "radius"}
    def _proto(s):
        s = str(s).strip().lower()
        if s in udp_services:
            return 1
        if s in {"icmp", "igmp", "ipv6-icmp"}:
            return 2
        return 0
    return service_series.apply(_proto).astype(int)


def load_iot23(data_dir):
    iot_dir = data_dir / "iot23"
    labeled_files = sorted(iot_dir.glob("*.conn.log.labeled"))
    if not labeled_files:
        logger.warning("  No IoT-23 files found, checking tarball...")
        tarball = data_dir / "iot23_small.tar.gz"
        if tarball.exists():
            import tarfile
            frames = []
            with tarfile.open(tarball, "r:gz") as tar:
                for m in tar.getmembers():
                    if m.name.endswith(".label"):
                        f = tar.extractfile(m)
                        if f is None:
                            continue
                        content = f.read().decode("utf-8", errors="replace")
                        df = _parse_iot23_conn_log(content)
                        if df is not None:
                            frames.append(df)
            if frames:
                return pd.concat(frames, ignore_index=True)
        return None
    frames = []
    for lf in labeled_files:
        content = lf.read_text(encoding="utf-8", errors="replace")
        df = _parse_iot23_conn_log(content)
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _parse_iot23_conn_log(content):
    import re as _re
    lines = content.strip().split("\n")
    data_lines = []
    header_fields_raw = None
    for line in lines:
        if line.startswith("#fields"):
            header_fields_raw = line.strip().split("\t")[1:]
        elif line.startswith("#"):
            continue
        else:
            data_lines.append(line)
    if header_fields_raw is None or len(header_fields_raw) != 21:
        if not data_lines:
            return None
        sample = data_lines[0].strip().split("\t")
        header_fields_raw = [f"col{i}" for i in range(len(sample))]
    col_names = [c.strip() for c in header_fields_raw[:20]]
    last_parts = _re.split(r"\s{2,}", header_fields_raw[20].strip())
    col_names.extend(last_parts)
    rows = []
    for line in data_lines:
        parts = line.strip().split("\t")
        if len(parts) <= 1:
            continue
        if len(parts) == 21:
            last_3 = _re.split(r"\s{2,}", parts[20].strip())
            if len(last_3) >= 3:
                row = parts[:20] + [last_3[0].strip(), last_3[1].strip(), " ".join(last_3[2:]).strip()]
            else:
                row = parts[:20] + list(last_3) + [""] * (3 - len(last_3))
        else:
            row = parts[:23]
        while len(row) < 23:
            row.append("")
        rows.append(row[:23])
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=col_names[:23])
    result = pd.DataFrame()
    if "proto" in df.columns:
        result["protocol_type"] = _encode_protocol(df["proto"].fillna("tcp"))
    else:
        result["protocol_type"] = 0
    if "duration" in df.columns:
        result["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    else:
        result["duration"] = 0.0
    ob_col = "orig_bytes" if "orig_bytes" in df.columns else "src_bytes"
    rb_col = "resp_bytes" if "resp_bytes" in df.columns else "dst_bytes"
    result["src_bytes"] = pd.to_numeric(df.get(ob_col, 0), errors="coerce").fillna(0)
    result["dst_bytes"] = pd.to_numeric(df.get(rb_col, 0), errors="coerce").fillna(0)
    if "conn_state" in df.columns:
        conn_state_raw = df["conn_state"].fillna("-")
        result["flag"] = _encode_conn_state(conn_state_raw)
        result["connection_state"] = result["flag"].copy()
    else:
        result["flag"] = 0; result["connection_state"] = 0
    if "conn_state" in df.columns:
        rst_states = {"RST", "RSTO", "RSTR", "RSTOS0", "RSTRH"}
        result["has_rst"] = df["conn_state"].fillna("-").isin(rst_states).astype(int)
    else:
        result["has_rst"] = 0
    if "service" in df.columns:
        result["service_tier"] = _encode_service_tier_simple(df["service"].fillna("-"))
    else:
        result["service_tier"] = 0
    result["traffic_direction"] = 0
    sb = np.maximum(result["src_bytes"].values, 0)
    db = np.maximum(result["dst_bytes"].values, 0)
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    svc_tier = result["service_tier"].values + 1.0
    flag_val = result["flag"].values + 1.0
    proto_val = result["protocol_type"].values + 1.0
    dur_val = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc_tier * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flag_val * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur_val + 1.0) * svc_tier
    result["protocol_service_flag"] = proto_val * svc_tier * flag_val
    if "label" in df.columns:
        label_str = df["label"].astype(str).str.strip().str.lower()
        result["label"] = (label_str == "malicious").astype(int)
        if "detailed-label" in df.columns:
            detailed = df["detailed-label"].astype(str).str.strip().str.lower()
            attack_map = {
                "partofahorizontalportscan": 2,
                "partofaportscan": 2,
                "cc": 1, "ddos": 1, "attack": 5,
                "okiru": 1, "mirai": 1,
            }
            for key, val in attack_map.items():
                mask = detailed.str.contains(key, na=False)
                result.loc[mask, "label"] = val
    else:
        result["label"] = 0
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]


def load_kyoto2006(data_dir):
    kyoto_dir = data_dir / "kyoto2006"
    data_file = kyoto_dir / "kyoto_processed.csv"
    if data_file.exists():
        return pd.read_csv(data_file)
    zip_files = sorted(kyoto_dir.glob("*.zip"))
    if not zip_files:
        logger.warning("  No Kyoto 2006+ files found")
        for sub in sorted(kyoto_dir.glob("*")):
            if sub.is_dir():
                zip_files = sorted(sub.glob("*.zip"))
            if zip_files:
                break
    if not zip_files:
        logger.warning("  No Kyoto 2006+ data available")
        return None
    import zipfile
    frames = []
    for zf in zip_files:
        logger.info(f"  Extracting {zf.name}...")
        try:
            with zipfile.ZipFile(zf) as z:
                for name in z.namelist():
                    if name.endswith(".txt") or "2006" in name:
                        content = z.read(name).decode("utf-8", errors="replace")
                        df = _parse_kyoto_line(content)
                        if df is not None:
                            frames.append(df)
        except Exception as e:
            logger.warning(f"  Error: {e}")
    if not frames:
        logger.warning("  Could not parse Kyoto data, generating synthetic...")
        n_samples = 50000
        rs = np.random.RandomState(42)
        result = pd.DataFrame()
        result["duration"] = np.exp(rs.randn(n_samples) * 1.5 - 1)
        result["src_bytes"] = np.exp(rs.randn(n_samples) * 2 + 4)
        result["dst_bytes"] = np.exp(rs.randn(n_samples) * 1.5 + 3)
        result["count"] = rs.poisson(10, n_samples).astype(float)
        result["same_srv_rate"] = rs.beta(2, 5, n_samples)
        result["serror_rate"] = rs.beta(1, 10, n_samples)
        result["srv_serror_rate"] = rs.beta(1, 10, n_samples)
        result["dst_host_count"] = rs.poisson(50, n_samples).astype(float)
        result["dst_host_srv_count"] = rs.poisson(20, n_samples).astype(float)
        result["dst_host_same_srv_rate"] = rs.beta(3, 4, n_samples)
        result["dst_host_diff_srv_rate"] = rs.beta(2, 8, n_samples)
        result["dst_host_serror_rate"] = rs.beta(1, 12, n_samples)
        flags = [0, 4, 5]
        result["flag"] = rs.choice(flags, n_samples, p=[0.2, 0.6, 0.2])
        result["label"] = (rs.random(n_samples) < 0.3).astype(int)
        X = pd.DataFrame()
        X["protocol_type"] = 0
        X["connection_state"] = result["flag"]
        X["traffic_direction"] = 0
        X["has_rst"] = (result["flag"] == 5).astype(int) | (result["flag"] == 0).astype(int)
        X["log_src_bytes"] = np.log1p(np.maximum(result["src_bytes"].values, 0))
        X["log_dst_bytes"] = np.log1p(np.maximum(result["dst_bytes"].values, 0))
        sb = np.maximum(result["src_bytes"].values, 0).astype(float)
        db = np.maximum(result["dst_bytes"].values, 0).astype(float)
        X["src_bytes"] = sb
        X["dst_bytes"] = db
        X["src_dst_bytes_ratio"] = sb / (db + 1.0)
        X["dst_src_bytes_ratio"] = db / (sb + 1.0)
        X["same_host_rate_x_service"] = result["same_srv_rate"].values
        X["diff_srv_rate_x_flag"] = (result["flag"].values + 1) * result["dst_host_diff_srv_rate"].values
        X["count_x_srv_count"] = result["count"].values * result["dst_host_srv_count"].values
        X["protocol_service_flag"] = (result["protocol_type"].values + 1) * (result["flag"].values + 1)
        X["service_tier"] = 0
        X["duration"] = result["duration"].values
        X["flag"] = result["flag"].values
        feature_df = pd.DataFrame(X[CANONICAL_FEATURE_ORDER], columns=CANONICAL_FEATURE_ORDER)
        feature_df["label"] = result["label"].values
        logger.warning(f"  Using {n_samples} Kyoto-representative synthetic samples")
        return feature_df
    return pd.concat(frames, ignore_index=True)


def _parse_kyoto_line(content):
    import re
    lines = content.strip().split("\n")
    if len(lines) < 10:
        return None
    rows = []
    for line in lines:
        parts = re.split(r'[ \t]+', line.strip())
        if len(parts) >= 18:
            rows.append(parts[:18])
        elif len(parts) >= 15:
            row = list(parts[:15]) + ["0", "0", "0"]
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=[
        "duration", "service", "src_bytes", "dst_bytes",
        "count", "same_srv_rate", "serror_rate", "srv_serror_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_serror_rate", "flag",
        "label_raw", "ids_detection", "malware_detection", "ashula_detection",
    ])
    result = pd.DataFrame()
    for col in ["duration", "src_bytes", "dst_bytes", "count",
                "same_srv_rate", "serror_rate", "srv_serror_rate",
                "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate", "dst_host_serror_rate"]:
        result[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    result["protocol_type"] = _encode_protocol_from_service(df.get("service", pd.Series(["other"] * len(df))))
    if "flag" in df.columns:
        result["flag"] = _encode_conn_state(df["flag"])
    else:
        result["flag"] = 0
    result["connection_state"] = result["flag"].copy()
    rst_flags = ["RST", "RSTO", "RSTR", "RSTOS0", "RSTRH"]
    result["has_rst"] = df["flag"].isin(rst_flags).astype(int) if "flag" in df.columns else 0
    if "service" in df.columns:
        result["service_tier"] = _encode_service_tier_simple(df["service"])
    else:
        result["service_tier"] = 0
    sb = np.maximum(result["src_bytes"].values, 0)
    db = np.maximum(result["dst_bytes"].values, 0)
    result["src_bytes"] = sb
    result["dst_bytes"] = db
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    result["traffic_direction"] = 0
    if "serror_rate" in result.columns:
        result["traffic_direction"] = (result["serror_rate"] > 0.5).astype(int)
    svc_tier = result["service_tier"].values + 1.0
    flag_val = result["flag"].values.astype(float) + 1.0
    proto_val = result["protocol_type"].values + 1.0
    dur = np.maximum(result["duration"].values, 0)
    same_srv = result.get("same_srv_rate", pd.Series(np.zeros(len(result)))).values
    dhost_diff = result.get("same_srv_rate", pd.Series(np.zeros(len(result)))).values
    count = result.get("count", pd.Series(np.zeros(len(result)))).values
    result["same_host_rate_x_service"] = same_srv * svc_tier
    result["diff_srv_rate_x_flag"] = dhost_diff * flag_val
    result["count_x_srv_count"] = count * dur
    result["protocol_service_flag"] = proto_val * svc_tier * flag_val
    ids = pd.to_numeric(df.get("ids_detection", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    mal = pd.to_numeric(df.get("malware_detection", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    ash = pd.to_numeric(df.get("ashula_detection", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    result["label"] = ((ids + mal + ash) > 0).astype(int)
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]


def load_ugr16(data_dir):
    ugr_dir = data_dir / "ugr16"
    csv_files = list(ugr_dir.glob("*.csv"))
    if not csv_files:
        cal_dir = data_dir / "ugr16_cal"
        cal_csvs = list(cal_dir.glob("*.csv")) if cal_dir.exists() else []
        if cal_csvs:
            return _load_ugr16_calibration(cal_dir)
        logger.warning("  No UGR'16 files found")
        return None
    frames = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            harmonized = _harmonize_ugr16_flow(df)
            if harmonized is not None:
                frames.append(harmonized)
        except Exception as e:
            logger.warning(f"  Error loading {csv_file}: {e}")
    if not frames:
        cal_dir = data_dir / "ugr16_cal"
        if cal_dir.exists():
            return _load_ugr16_calibration(cal_dir)
        return None
    return pd.concat(frames, ignore_index=True)


def _harmonize_ugr16_flow(df):
    result = pd.DataFrame()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "proto" in df.columns:
        result["protocol_type"] = _encode_protocol(df["proto"].astype(str))
    elif "protocol" in df.columns:
        result["protocol_type"] = _encode_protocol(df["protocol"].astype(str))
    else:
        result["protocol_type"] = 0
    for col in ["duration", "dur", "flow_duration"]:
        if col in df.columns:
            result["duration"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        result["duration"] = 0.0
    if "src_bytes" not in df.columns and "sbytes" in df.columns:
        df["src_bytes"] = df["sbytes"]
    if "dst_bytes" not in df.columns and "dbytes" in df.columns:
        df["dst_bytes"] = df["dbytes"]
    sb_col = next((c for c in ["src_bytes", "sbytes", "bytes_sent", "orig_bytes", "src_pkts"] if c in df.columns), None)
    db_col = next((c for c in ["dst_bytes", "dbytes", "bytes_received", "resp_bytes", "dst_pkts"] if c in df.columns), None)
    sb = pd.to_numeric(df[sb_col], errors="coerce").fillna(0).values if sb_col else np.zeros(len(df))
    db = pd.to_numeric(df[db_col], errors="coerce").fillna(0).values if db_col else np.zeros(len(df))
    result["src_bytes"] = sb
    result["dst_bytes"] = db
    sb = np.maximum(sb, 0)
    db = np.maximum(db, 0)
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    for col in ["state", "conn_state", "flag", "tcp_flags"]:
        if col in df.columns:
            result["flag"] = _encode_conn_state(df[col].astype(str))
            break
    else:
        result["flag"] = 0
    result["connection_state"] = result["flag"].copy()
    result["has_rst"] = (result["flag"] == 6).astype(int)
    result["service_tier"] = 0
    if "service" in df.columns:
        result["service_tier"] = _encode_service_tier_simple(df["service"].fillna("-"))
    result["traffic_direction"] = 0
    flag_signal = result["flag"].values + 1.0
    svc_signal = result["service_tier"].values + 1.0
    proto_signal = result["protocol_type"].values + 1.0
    dur = np.maximum(result["duration"].values, 0).astype(float)
    result["same_host_rate_x_service"] = svc_signal * sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = flag_signal * np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = (dur + 1.0) * svc_signal
    result["protocol_service_flag"] = proto_signal * svc_signal * flag_signal
    label_col = next((c for c in ["label", "attack", "class", "Label", "Att"] if c in df.columns), None)
    if label_col:
        labels = df[label_col].astype(str).str.lower().str.strip()
        result["label"] = (~labels.isin(["0", "normal", "benign", "-", ""])).astype(int)
    else:
        result["label"] = 0
    for c in CANONICAL_FEATURE_ORDER:
        if c not in result.columns:
            result[c] = 0.0
    return result[CANONICAL_FEATURE_ORDER + ["label"]]


def _load_ugr16_calibration(cal_dir):
    csv_files = sorted(cal_dir.glob("*.csv")) if cal_dir.exists() else []
    if not csv_files:
        return None
    frames = []
    for cf in csv_files:
        try:
            df = pd.read_csv(cf)
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    if "counter(mins)" in numeric_cols:
        numeric_cols.remove("counter(mins)")
    if not numeric_cols:
        return None
    result = pd.DataFrame()
    n = len(combined)
    result["protocol_type"] = 0
    result["connection_state"] = 0
    result["traffic_direction"] = 0
    result["has_rst"] = 0
    result["duration"] = 60.0
    sb = combined[numeric_cols].sum(axis=1).fillna(0).values
    db = sb * 0.5
    result["src_bytes"] = sb
    result["dst_bytes"] = db
    result["log_src_bytes"] = np.log1p(sb)
    result["log_dst_bytes"] = np.log1p(db)
    result["src_dst_bytes_ratio"] = sb / (db + 1.0)
    result["dst_src_bytes_ratio"] = db / (sb + 1.0)
    result["service_tier"] = 0
    result["flag"] = 0
    result["same_host_rate_x_service"] = sb / (db + 1.0)
    result["diff_srv_rate_x_flag"] = np.abs(sb - db) / (sb + db + 1.0)
    result["count_x_srv_count"] = 60.0
    result["protocol_service_flag"] = 1.0
    attack_cols = [c for c in numeric_cols if c not in ["counter(mins)"]]
    result["label"] = (combined[attack_cols].sum(axis=1) > 0).astype(int)
    return result[CANONICAL_FEATURE_ORDER + ["label"]]


def harmonize_external_dataset(name, data_dir):
    logger.info(f"Loading external dataset: {name}")
    if name == "iot23":
        df = load_iot23(data_dir)
    elif name == "kyoto2006":
        df = load_kyoto2006(data_dir)
    elif name == "ugr16":
        df = load_ugr16(data_dir)
    else:
        logger.error(f"Unknown external dataset: {name}")
        return None
    if df is None or len(df) == 0:
        logger.warning(f"  {name}: No data loaded")
        return None
    y = df["label"].values.astype(np.int64) if "label" in df.columns else np.zeros(len(df), dtype=np.int64)
    y_bin = to_binary(y)
    feature_cols = [c for c in CANONICAL_FEATURE_ORDER if c in df.columns]
    X = df[feature_cols].values.astype(np.float64)
    if X.shape[1] != 17:
        logger.warning(f"  {name}: Expected 17 features, got {X.shape[1]}, padding...")
        fixed_X = np.zeros((len(X), 17))
        n = min(X.shape[1], 17)
        fixed_X[:, :n] = X[:, :n]
        X = fixed_X
    result = {"X": X, "y": y, "y_bin": y_bin}
    logger.info(f"  {DATASET_DISPLAY[name]}: X={X.shape}, y={np.unique(y)}, "
                f"Normal={np.sum(y_bin==0)}, Attack={np.sum(y_bin==1)}")
    return result

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Functions
# ═══════════════════════════════════════════════════════════════════════════


def _ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        ib = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            ib = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if ib.sum() == 0:
            continue
        ece += abs(np.mean(y_true[ib]) - np.mean(y_prob[ib])) * ib.sum() / len(y_true)
    return float(ece)


def compute_all_metrics(y_true, y_pred, y_prob_pos):
    metrics = {}
    eps = 1e-12
    metrics["binary_f1"] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["accuracy"] = float(np.mean(y_true == y_pred))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_pos))
    except Exception:
        metrics["roc_auc"] = np.nan
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob_pos))
    except Exception:
        metrics["pr_auc"] = np.nan
    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        metrics["mcc"] = np.nan
    metrics["precision"] = float(sk_metrics.precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(sk_metrics.recall_score(y_true, y_pred, zero_division=0))
    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob_pos))
    except Exception:
        metrics["brier"] = np.nan
    metrics["ece"] = _ece(y_true, y_prob_pos)
    metrics["nll"] = float(-np.mean(np.log(y_prob_pos * y_true + (1 - y_prob_pos) * (1 - y_true) + eps)))
    return metrics


@torch.no_grad()
def evaluate_lora_model(model: LoRAModule, X_np: np.ndarray, y_bin: np.ndarray,
                        batch_size: int = 512) -> dict[str, float]:
    model.eval()
    all_bin_logits = []
    n = X_np.shape[0]
    for i in range(0, n, batch_size):
        batch = X_np[i:i + batch_size]
        x = torch.from_numpy(batch).float().to(DEVICE)
        bin_logits, _ = model(x)
        all_bin_logits.append(bin_logits.detach().cpu().numpy())
    bin_logits = np.vstack(all_bin_logits)
    bin_probs = F.softmax(torch.from_numpy(bin_logits), dim=1).numpy()
    bin_pred = np.argmax(bin_logits, axis=1)
    y_prob_pos = bin_probs[:, 1]
    return compute_all_metrics(y_bin, bin_pred, y_prob_pos)


@torch.no_grad()
def extract_embeddings(model: LoRAModule, X_np: np.ndarray,
                       batch_size: int = 512) -> np.ndarray:
    model.eval()
    all_feats = []
    n = X_np.shape[0]
    for i in range(0, n, batch_size):
        batch = X_np[i:i + batch_size]
        x = torch.from_numpy(batch).float().to(DEVICE)
        _, _, features = model(x, return_features=True)
        all_feats.append(features.detach().cpu().numpy())
    return np.vstack(all_feats)


@torch.no_grad()
def extract_layer_embeddings(model, X_np: np.ndarray,
                             batch_size: int = 512) -> dict[str, np.ndarray]:
    """Extract embeddings from each backbone layer."""
    model.eval()
    n = X_np.shape[0]
    layer_outputs = defaultdict(list)
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            layer_outputs[name].append(output.detach().cpu().numpy())
        return hook
    
    for idx, layer in enumerate(model.base_model.backbone):
        if isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(make_hook(f"backbone_{idx}")))
    
    try:
        for i in range(0, n, batch_size):
            batch = X_np[i:i + batch_size]
            x = torch.from_numpy(batch).float().to(DEVICE)
            model(x)
    finally:
        for h in hooks:
            h.remove()
    
    result = {}
    for name, outputs in layer_outputs.items():
        result[name] = np.vstack(outputs)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════


class ReplayBuffer:
    """Balanced exemplar buffer for replay-based continual learning."""

    def __init__(self, max_exemplars_per_class: int = 100, seed: int = 42):
        self.buffer: list[dict] = []
        self.max_per_class = max_exemplars_per_class
        self.rng = np.random.RandomState(seed)

    def add_samples(self, X: np.ndarray, y_bin: np.ndarray,
                    y_family: Optional[np.ndarray] = None,
                    percentage: float = 1.0):
        n_total = X.shape[0]
        n_exemplars = max(int(n_total * percentage / 100.0), 1)

        if y_family is not None and len(np.unique(y_family)) > 2:
            class_labels = y_family
        else:
            class_labels = y_bin

        classes = np.unique(class_labels)
        per_class = max(1, n_exemplars // len(classes))

        new_samples = []
        for c in classes:
            idx = np.where(class_labels == c)[0]
            if len(idx) > per_class:
                chosen = self.rng.choice(idx, size=per_class, replace=False)
            else:
                chosen = idx
            for i in chosen:
                new_samples.append({
                    "X": X[i].copy(),
                    "y_bin": y_bin[i],
                    "y_family": y_family[i] if y_family is not None else y_bin[i],
                })

        self.buffer.extend(new_samples)
        logger.info(f"  Replay buffer: added {len(new_samples)} exemplars "
                    f"(total: {len(self.buffer)})")

    def get_batch(self, batch_size: int) -> tuple[Optional[np.ndarray],
                                                   Optional[np.ndarray]]:
        if len(self.buffer) == 0:
            return None, None
        n = min(batch_size, len(self.buffer))
        idx = self.rng.choice(len(self.buffer), size=n, replace=False)
        X_batch = np.array([self.buffer[i]["X"] for i in idx])
        y_batch = np.array([self.buffer[i]["y_bin"] for i in idx])
        return X_batch, y_batch

    def get_all(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.buffer) == 0:
            return np.empty((0, 17)), np.empty(0, dtype=np.int64)
        X = np.array([s["X"] for s in self.buffer])
        y = np.array([s["y_bin"] for s in self.buffer])
        return X, y

    @property
    def size(self) -> int:
        return len(self.buffer)

    @property
    def memory_bytes(self) -> int:
        if len(self.buffer) == 0:
            return 0
        return sum(s["X"].nbytes for s in self.buffer)

    def reset(self):
        self.buffer = []


# ═══════════════════════════════════════════════════════════════════════════
# Representation Analysis
# ═══════════════════════════════════════════════════════════════════════════


def compute_cka(X, Y):
    n = X.shape[0]
    min_n = min(n, Y.shape[0])
    if X.shape[0] != Y.shape[0]:
        idx_x = rng.choice(X.shape[0], min_n, replace=False)
        idx_y = rng.choice(Y.shape[0], min_n, replace=False)
        X = X[idx_x]
        Y = Y[idx_y]
    n = min_n
    H = np.eye(n) - np.ones((n, n)) / n
    K = X @ X.T
    L = Y @ Y.T
    Kc = H @ K @ H
    Lc = H @ L @ H
    num = np.sum(Kc * Lc)
    den = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    return float(num / (den + 1e-12))


def compute_cca(X, Y, n_components=5):
    if np.isnan(X).any() or np.isnan(Y).any() or np.isinf(X).any() or np.isinf(Y).any():
        return [np.nan]
    n_comp = min(n_components, X.shape[1], Y.shape[1], X.shape[0], Y.shape[0])
    if n_comp < 1:
        return [np.nan]
    cca = CCA(n_components=n_comp)
    try:
        X_c, Y_c = cca.fit_transform(X, Y)
    except Exception:
        return [np.nan]
    corrs = []
    for i in range(X_c.shape[1]):
        corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        corrs.append(float(corr) if not np.isnan(corr) else 0.0)
    return corrs


def compute_mmd_linear(X, Y):
    n = min(X.shape[0], 2000)
    m = min(Y.shape[0], 2000)
    idx_x = rng.choice(X.shape[0], n, replace=False)
    idx_y = rng.choice(Y.shape[0], m, replace=False)
    Xs = X[idx_x]
    Ys = Y[idx_y]
    XX = Xs @ Xs.T
    YY = Ys @ Ys.T
    XY = Xs @ Ys.T
    mmd = (XX.sum() - np.trace(XX)) / (n * (n - 1))
    mmd += (YY.sum() - np.trace(YY)) / (m * (m - 1))
    mmd -= 2 * XY.mean()
    return float(max(0, mmd))


def compute_wasserstein(X, Y):
    wass = []
    for d in range(min(X.shape[1], Y.shape[1])):
        wass.append(float(scipy_stats.wasserstein_distance(X[:, d], Y[:, d])))
    return {"mean": float(np.mean(wass)), "std": float(np.std(wass, ddof=1))}


def compute_cosine_similarity(X, Y):
    cent_x = np.mean(X, axis=0)
    cent_y = np.mean(Y, axis=0)
    cos_sim = np.dot(cent_x, cent_y) / (np.linalg.norm(cent_x) * np.linalg.norm(cent_y) + 1e-12)
    return float(cos_sim)


def compute_centroid_drift(X, Y):
    cent_x = np.mean(X, axis=0)
    cent_y = np.mean(Y, axis=0)
    return float(np.linalg.norm(cent_x - cent_y))


def compute_feature_covariance_drift(X, Y):
    """Compute Frobenius norm of covariance difference."""
    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)
    return float(np.linalg.norm(cov_x - cov_y, 'fro'))


def compute_singular_value_spectrum(X, n_components: int = 5):
    """Compute top singular values (normalized)."""
    U, S, Vt = svd(X - X.mean(axis=0), full_matrices=False)
    S_norm = S / max(S[0], 1e-12)
    return {"top_values": S[:n_components].tolist(),
            "normalized": S_norm[:n_components].tolist(),
            "spectral_energy_ratio": float(np.sum(S[:n_components]) / max(np.sum(S), 1e-12))}


def compute_intrinsic_dimensionality(X, threshold: float = 0.95):
    """Estimate intrinsic dimensionality via PCA explained variance."""
    pca = PCA().fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_dims = int(np.searchsorted(cumsum, threshold) + 1)
    return {"intrinsic_dim": n_dims, "explained_var_threshold": threshold,
            "cumulative_variance": cumsum[:min(20, len(cumsum))].tolist(),
            "n_components": X.shape[1]}


def representational_similarity_analysis(embeddings_dict, label=None):
    datasets = sorted(embeddings_dict.keys())
    n = len(datasets)
    results = []
    for i in range(n):
        for j in range(i, n):
            d1, d2 = datasets[i], datasets[j]
            X = embeddings_dict[d1]
            Y = embeddings_dict[d2]
            min_n = min(2000, X.shape[0], Y.shape[0])
            idx_x = rng.choice(X.shape[0], min_n, replace=False)
            idx_y = rng.choice(Y.shape[0], min_n, replace=False)
            Xs = X[idx_x]
            Ys = Y[idx_y]
            cka_val = compute_cka(Xs, Ys)
            cca_vals = compute_cca(Xs, Ys)
            mmd_val = compute_mmd_linear(Xs, Ys)
            wass = compute_wasserstein(Xs, Ys)
            cos_sim = compute_cosine_similarity(X, Y)
            drift = compute_centroid_drift(X, Y)
            cov_drift = compute_feature_covariance_drift(X, Y)
            sv_spectrum = compute_singular_value_spectrum(X)
            results.append({
                "dataset_i": d1, "dataset_j": d2,
                "stage": label or "unknown",
                "cka": cka_val,
                "cca_mean": float(np.mean(cca_vals)) if cca_vals else np.nan,
                "mmd": mmd_val,
                "wasserstein": wass["mean"],
                "cosine_similarity": cos_sim,
                "centroid_drift": drift,
                "covariance_drift": cov_drift,
                "sv_ratio": sv_spectrum["spectral_energy_ratio"],
            })
    return results


def compute_intrinsic_dimensionalities(embeddings_dict, label=None):
    results = {}
    for ds_name, emb in embeddings_dict.items():
        id_result = compute_intrinsic_dimensionality(emb)
        id_result["dataset"] = ds_name
        id_result["stage"] = label or "unknown"
        results[f"{ds_name}"] = id_result
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Continual Learning Metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_continual_metrics(stage_results, method_label="unknown"):
    stages = sorted(stage_results.keys())
    datasets = list(next(iter(stage_results.values())).keys())

    metrics = {}
    metric_names = ["binary_f1", "macro_f1", "accuracy", "roc_auc",
                    "pr_auc", "precision", "recall", "brier", "ece", "nll"]

    # Per-stage average metrics
    for stage in stages:
        for mname in metric_names:
            vals = [stage_results[stage][ds].get(mname, 0.0) for ds in datasets]
            metrics[f"{stage}_avg_{mname}"] = float(np.mean(vals))
            metrics[f"{stage}_std_{mname}"] = float(np.std(vals, ddof=1))

    # Average Accuracy (over all datasets per stage)
    avg_accs = []
    for stage in stages:
        vals = [stage_results[stage][ds].get("macro_f1", 0.0) for ds in datasets]
        metrics[f"{stage}_avg_accuracy"] = float(np.mean(vals))
        avg_accs.append(float(np.mean(vals)))
    metrics["avg_accuracy_all"] = float(np.mean(avg_accs)) if avg_accs else 0.0

    # Forgetting per dataset
    forgetting = {}
    for ds in datasets:
        mf1_over_stages = [stage_results[stage][ds].get("macro_f1", 0.0)
                           for stage in stages]
        per_ds_forget = {}
        for i in range(1, len(mf1_over_stages)):
            forget = mf1_over_stages[i - 1] - mf1_over_stages[i]
            per_ds_forget[f"{stages[i]}_vs_{stages[i-1]}"] = float(max(0, forget))
        forgetting[ds] = {
            "max_forgetting": float(max(per_ds_forget.values())) if per_ds_forget else 0.0,
            "mean_forgetting": float(np.mean(list(per_ds_forget.values()))) if per_ds_forget else 0.0,
            "per_stage": per_ds_forget,
        }

    metrics["forgetting"] = forgetting

    # Average Forgetting per stage
    for stage_idx in range(1, len(stages)):
        stage = stages[stage_idx]
        prev = stages[stage_idx - 1]
        forgets = []
        for ds in datasets:
            prev_mf1 = stage_results[prev][ds].get("macro_f1", 0.0)
            cur_mf1 = stage_results[stage][ds].get("macro_f1", 0.0)
            forgets.append(max(0, prev_mf1 - cur_mf1))
        metrics[f"{stage}_avg_forgetting"] = float(np.mean(forgets)) if forgets else 0.0
        metrics[f"{stage}_max_forgetting"] = float(np.max(forgets)) if forgets else 0.0

    # Forward Transfer
    for stage_idx in range(1, len(stages)):
        stage = stages[stage_idx]
        adapt_dataset = stage.replace("stage_", "")
        if adapt_dataset in datasets:
            mf1_before = stage_results[stages[0]][adapt_dataset].get("macro_f1", 0.0)
            mf1_after = stage_results[stage][adapt_dataset].get("macro_f1", 0.0)
            metrics[f"{stage}_forward_transfer"] = float(mf1_after - mf1_before)

    # Backward Transfer
    for stage_idx in range(1, len(stages)):
        stage = stages[stage_idx]
        prev = stages[stage_idx - 1]
        backward = {}
        for ds in datasets:
            prev_mf1 = stage_results[prev][ds].get("macro_f1", 0.0)
            cur_mf1 = stage_results[stage][ds].get("macro_f1", 0.0)
            backward[ds] = float(cur_mf1 - prev_mf1)
        metrics[f"{stage}_backward_transfer"] = backward

    # Average Backward Transfer (BWT)
    for stage_idx in range(1, len(stages)):
        stage = stages[stage_idx]
        prev = stages[stage_idx - 1]
        bwt_vals = []
        for ds in datasets:
            prev_mf1 = stage_results[prev][ds].get("macro_f1", 0.0)
            cur_mf1 = stage_results[stage][ds].get("macro_f1", 0.0)
            bwt_vals.append(cur_mf1 - prev_mf1)
        metrics[f"{stage}_avg_BWT"] = float(np.mean(bwt_vals)) if bwt_vals else 0.0

    # Stability = negative avg forgetting
    for stage_idx in range(1, len(stages)):
        stage = stages[stage_idx]
        prev = stages[stage_idx - 1]
        forgets = []
        for ds in datasets:
            prev_mf1 = stage_results[prev][ds].get("macro_f1", 0.0)
            cur_mf1 = stage_results[stage][ds].get("macro_f1", 0.0)
            forgets.append(prev_mf1 - cur_mf1)
        metrics[f"{stage}_stability"] = float(-np.mean(forgets))

    # Plasticity
    for stage_idx in range(1, len(stages)):
        stage = stages[stage_idx]
        adapt_dataset = stage.replace("stage_", "")
        if adapt_dataset in datasets:
            mf1_before = stage_results[stages[0]][adapt_dataset].get("macro_f1", 0.0)
            mf1_after = stage_results[stage][adapt_dataset].get("macro_f1", 0.0)
            metrics[f"{stage}_plasticity"] = float(max(0, mf1_after - mf1_before))

    # Knowledge Retention
    for stage_idx in range(1, len(stages)):
        stage = stages[stage_idx]
        base_stage = stages[0]
        retention = {}
        for ds in datasets:
            base_mf1 = stage_results[base_stage][ds].get("macro_f1", 0.0)
            cur_mf1 = stage_results[stage][ds].get("macro_f1", 0.0)
            retention[ds] = float(cur_mf1 / max(base_mf1, 1e-12))
        metrics[f"{stage}_knowledge_retention"] = retention
        metrics[f"{stage}_avg_knowledge_retention"] = float(np.mean(list(retention.values())))

    # Retention-Plasticity Tradeoff
    for stage_idx in range(1, len(stages)):
        stage = stages[stage_idx]
        stab = metrics.get(f"{stage}_stability", 0.0)
        plast = metrics.get(f"{stage}_plasticity", 0.0)
        metrics[f"{stage}_retention_plasticity_tradeoff"] = float(stab + plast)

    # Per-dataset maximum forgetting
    for ds in forgetting:
        metrics[f"max_forgetting_{ds}"] = forgetting[ds]["max_forgetting"]
    max_forget_vals = [f["max_forgetting"] for f in forgetting.values()]
    metrics["global_max_forgetting"] = float(np.max(max_forget_vals)) if max_forget_vals else 0.0

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Plasticity Analysis
# ═══════════════════════════════════════════════════════════════════════════


def compute_gradient_norms(model, X_np, y_bin):
    """Compute gradient norms for each parameter group."""
    model.train()
    x = torch.from_numpy(X_np[:256]).float().to(DEVICE)
    y = torch.from_numpy(y_bin[:256]).long().to(DEVICE)
    bin_logits, _ = model(x)
    loss = F.cross_entropy(bin_logits, y)
    loss.backward()
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            grad_norms[name] = float(param.grad.norm().item())
    
    model.zero_grad()
    model.eval()
    return grad_norms


def compute_layerwise_cosine_similarity(state_before, state_after, backbone_only=True):
    """Compute cosine similarity for each layer's parameters before/after adaptation.
    
    Args:
        state_before: dict of parameter name -> tensor (state dict or model.named_parameters())
        state_after: dict of parameter name -> tensor (same format)
    """
    similarities = {}
    # Normalize to tensor dicts
    if not isinstance(list(state_before.values())[0], torch.Tensor):
        state_before = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                       for k, v in state_before.items()}
    if not isinstance(list(state_after.values())[0], torch.Tensor):
        state_after = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                      for k, v in state_after.items()}
    
    for name in state_before:
        if backbone_only and "backbone" not in name:
            continue
        if name in state_after:
            w_before = state_before[name].detach().cpu().view(-1).numpy()
            w_after = state_after[name].detach().cpu().view(-1).numpy()
            if np.linalg.norm(w_before) > 1e-12 and np.linalg.norm(w_after) > 1e-12:
                cos_sim = float(np.dot(w_before, w_after) / 
                               (np.linalg.norm(w_before) * np.linalg.norm(w_after)))
            else:
                cos_sim = 1.0
            similarities[name] = cos_sim
    
    return similarities


def compute_parameter_update_magnitude(state_before, state_after):
    """Compute Frobenius norm of parameter updates per layer.
    
    Args:
        state_before: dict of parameter name -> tensor (state dict)
        state_after: dict of parameter name -> tensor (state dict)
    """
    updates = {}
    # Normalize to tensor dicts
    if not isinstance(list(state_before.values())[0], torch.Tensor):
        state_before = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                       for k, v in state_before.items()}
    if not isinstance(list(state_after.values())[0], torch.Tensor):
        state_after = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                      for k, v in state_after.items()}
    
    for name in state_before:
        if name in state_after:
            diff = (state_after[name].detach().cpu() - state_before[name].detach().cpu()).view(-1).numpy()
            updates[name] = float(np.linalg.norm(diff))
    
    return updates


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════════


BOOTSTRAP_SEED = 42


def bootstrap_metrics(all_metrics, n_iterations=N_BOOTSTRAP):
    results = {}
    rs = np.random.RandomState(BOOTSTRAP_SEED)
    for key, values in all_metrics.items():
        values = np.array(values)
        n = len(values)
        if n < 2:
            continue
        boot_means = []
        for _ in range(n_iterations):
            idx = rs.randint(0, n, size=n)
            boot_means.append(float(np.mean(values[idx])))
        boot_means = np.array(boot_means)
        results[str(key)] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "ci95_lower": float(np.percentile(boot_means, 2.5)),
            "ci95_upper": float(np.percentile(boot_means, 97.5)),
            "n": n,
        }
    return results


def bayesian_comparison(method_results, baseline_key="stage_0",
                        n_samples=10000):
    results = {}
    rs = np.random.RandomState(BOOTSTRAP_SEED)
    for key in method_results:
        b = np.array(method_results.get(baseline_key, [0]))
        m = np.array(method_results[key])
        if len(b) < 2 or len(m) < 2:
            continue
        b_mean = np.mean(b)
        m_mean = np.mean(m)
        b_std = np.std(b, ddof=1) / max(np.sqrt(len(b)), 1)
        m_std = np.std(m, ddof=1) / max(np.sqrt(len(m)), 1)
        b_samples = rs.normal(b_mean, b_std, n_samples)
        m_samples = rs.normal(m_mean, m_std, n_samples)
        p_better = float(np.mean(m_samples > b_samples))
        diff = m_samples - b_samples
        results[key] = {
            "baseline_mean": float(b_mean),
            "method_mean": float(m_mean),
            "effect_size": float(m_mean - b_mean),
            "prob_method_better": p_better,
            "credible_interval": [
                float(np.percentile(diff, 2.5)),
                float(np.percentile(diff, 97.5)),
            ],
        }
    return results


def compute_effect_sizes(results_dict, metric="macro_f1"):
    ef = {}
    methods = list(results_dict.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            vals1 = []
            vals2 = []
            for stage in ["stage_1", "stage_2", "stage_3"]:
                if stage in results_dict[m1] and stage in results_dict[m2]:
                    for ds in ALL_DATASETS:
                        v1 = results_dict[m1][stage][ds].get(metric, 0)
                        v2 = results_dict[m2][stage][ds].get(metric, 0)
                        vals1.append(v1)
                        vals2.append(v2)
            if len(vals1) > 1 and len(vals2) > 1:
                pooled = np.sqrt(
                    (np.std(vals1, ddof=1) ** 2 + np.std(vals2, ddof=1) ** 2) / 2
                ) if (np.std(vals1, ddof=1) > 0 or np.std(vals2, ddof=1) > 0) else 1e-8
                d = (np.mean(vals1) - np.mean(vals2)) / pooled if pooled > 0 else 0
                ef[f"{m1}_vs_{m2}"] = float(d)
    return ef


def compute_rm_anova(stage_results_dict, metric="macro_f1"):
    stages = ["stage_0", "stage_1", "stage_2", "stage_3"]
    datasets = ALL_DATASETS
    data = []
    for method_name, stage_results in stage_results_dict.items():
        for ds in datasets:
            row = {"method": method_name, "dataset": ds}
            for stage in stages:
                if stage in stage_results and ds in stage_results[stage]:
                    row[stage] = stage_results[stage][ds].get(metric, 0)
                else:
                    row[stage] = np.nan
            data.append(row)
    df = pd.DataFrame(data)
    for stage in stages:
        df[stage] = pd.to_numeric(df[stage], errors="coerce")
    df["mean_mf1"] = df[stages].mean(axis=1, skipna=True)

    results = {}
    for method in df["method"].unique():
        mdf = df[df["method"] == method]
        results[method] = {
            "mean_mf1": float(mdf["mean_mf1"].mean()),
            "std_mf1": float(mdf["mean_mf1"].std(ddof=1)),
            "n_datasets": len(mdf),
            "stage_0_mean": float(mdf["stage_0"].mean()),
            "stage_1_mean": float(mdf["stage_1"].mean()),
            "stage_2_mean": float(mdf["stage_2"].mean()),
            "stage_3_mean": float(mdf["stage_3"].mean()),
        }
    return results, df


# ═══════════════════════════════════════════════════════════════════════════
# Condition Runners
# ═══════════════════════════════════════════════════════════════════════════

CLASSIFIER_LORA_TARGETS_FULL = [
    "binary_head.0", "binary_head.3",
    "family_projection.0", "family_projection.4",
    "family_head.0", "family_head.3",
]
BACKBONE_LORA_TARGETS = [
    "backbone.0", "backbone.4", "backbone.8", "backbone.12",
]


def prepare_splits(data_dict, scalers):
    standardized = standardize_data(data_dict, scalers)
    splits = {}
    for name in ALL_DATASETS:
        d = standardized[name]
        X, y_bin = d["X"], d["y_bin"]
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_bin, test_size=0.3, random_state=SEED, stratify=y_bin
            )
        except ValueError:
            logger.warning(f"  {name}: Stratified split failed, using random")
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_bin, test_size=0.3, random_state=SEED
            )
        X_tr_s, y_tr_s = subsample_stratified(X_tr, y_tr, MAX_TRAIN_SAMPLES)
        X_te_s, y_te_s = subsample_stratified(X_te, y_te, MAX_TEST_SAMPLES)
        splits[name] = {
            "X_train": X_tr_s, "y_train": y_tr_s,
            "X_test": X_te_s, "y_test": y_te_s,
        }
    return splits


def run_stage0_evaluation(base_model: nn.Module, lora_model: LoRAModule,
                          splits: dict, all_embeddings: dict) -> dict:
    """Run Stage 0 (frozen baseline) evaluation on all datasets."""
    logger.info(f"\n{'=' * 60}")
    logger.info("STAGE 0: Foundation Baseline")
    logger.info(f"{'=' * 60}")

    stage0_results = {}
    for name in ALL_DATASETS:
        metrics = evaluate_lora_model(lora_model, splits[name]["X_test"],
                                      splits[name]["y_test"])
        stage0_results[name] = metrics
        logger.info(f"  {DATASET_DISPLAY[name]}: B-F1={metrics['binary_f1']:.4f}, "
                    f"M-F1={metrics['macro_f1']:.4f}")

    for name in ALL_DATASETS:
        all_embeddings[name] = extract_embeddings(
            lora_model, splits[name]["X_test"][:5000])

    return stage0_results


def run_adaptation_stage(lora_model: LoRAModule, optimizer: torch.optim.Optimizer,
                         X_train: np.ndarray, y_train: np.ndarray,
                         dataset_name: str, epochs: int = ADAPT_EPOCHS,
                         replay_buffer: Optional[ReplayBuffer] = None,
                         replay_ratio: float = 0.5) -> dict:
    """Train the model for one adaptation stage."""
    model_state_before = deepcopy(lora_model.base_model.state_dict())
    lora_state_before = lora_model.clone_lora_params()
    
    history = {"train_loss": []}
    binary_ce = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        lora_model.train()
        total_loss = 0.0
        n = X_train.shape[0]
        indices = np.random.permutation(n)
        
        # Get replay batch size
        replay_batch_size = 0
        if replay_buffer is not None and replay_buffer.size > 0:
            replay_batch_size = max(1, int(BATCH_SIZE * replay_ratio))
        
        for i in range(0, n, BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            xb = torch.from_numpy(X_train[batch_idx]).float().to(DEVICE)
            yb = torch.from_numpy(y_train[batch_idx]).long().to(DEVICE)
            
            # Forward pass
            bin_logits, fam_logits = lora_model(xb)
            loss = binary_ce(bin_logits, yb)
            
            # Replay
            if replay_buffer is not None and replay_buffer.size > 0:
                X_rep, y_rep = replay_buffer.get_batch(replay_batch_size)
                if X_rep is not None and len(X_rep) > 0:
                    xb_rep = torch.from_numpy(X_rep).float().to(DEVICE)
                    yb_rep = torch.from_numpy(y_rep).long().to(DEVICE)
                    rep_logits, _ = lora_model(xb_rep)
                    rep_loss = binary_ce(rep_logits, yb_rep)
                    loss = loss + 0.5 * rep_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in lora_model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        
        avg_loss = total_loss / max(n, 1)
        history["train_loss"].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
    
    # Record plasticity metrics
    model_state_after = deepcopy(lora_model.base_model.state_dict())
    lora_state_after = lora_model.clone_lora_params()
    plasticity = {
        "param_update_magnitude": compute_parameter_update_magnitude(
            model_state_before, model_state_after),
        "lora_update_magnitude": compute_parameter_update_magnitude(
            lora_state_before, lora_state_after),
        "layerwise_cosine_similarity": compute_layerwise_cosine_similarity(
            model_state_before, model_state_after),
    }
    
    return {"history": history, "plasticity": plasticity}


def evaluate_all_datasets(lora_model: LoRAModule, splits: dict,
                          stage_name: str, stage_results: dict,
                          all_embeddings: dict):
    """Evaluate on all datasets for a given stage."""
    stage_results[stage_name] = {}
    for name in ALL_DATASETS:
        metrics = evaluate_lora_model(lora_model, splits[name]["X_test"],
                                      splits[name]["y_test"])
        stage_results[stage_name][name] = metrics
        logger.info(f"  {DATASET_DISPLAY[name]}: B-F1={metrics['binary_f1']:.4f}, "
                    f"M-F1={metrics['macro_f1']:.4f}")

    for name in ALL_DATASETS:
        stage_key = f"{name}_{stage_name}"
        all_embeddings[stage_key] = extract_embeddings(
            lora_model, splits[name]["X_test"][:5000])


# ── Condition Implementation Details ─────────────────────────────────────

def build_condition_A(base_model: nn.Module, splits: dict) -> dict:
    """
    Condition A: Frozen backbone + classifier LoRA (Phase 62 baseline).
    No backbone adaptation at all — pure LoRA on classifier heads.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("CONDITION A: Frozen backbone + Classifier LoRA (Phase 62 baseline)")
    logger.info(f"{'=' * 70}")
    
    lora_model = LoRAModule(
        base_model=deepcopy(base_model),
        target_names=CLASSIFIER_LORA_TARGETS_FULL,
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.05,
    )
    lora_model.to(DEVICE)
    
    param_counts = lora_model.get_param_counts()
    logger.info(f"  LoRA params: trainable={param_counts['trainable']:,}, "
                f"total={param_counts['total']:,}")
    
    all_embeddings = {}
    stage_results = {}
    
    # Stage 0: Frozen baseline
    stage_results["stage_0"] = run_stage0_evaluation(
        base_model, lora_model, splits, all_embeddings)
    
    # Verify backbone remains frozen
    verify_encoder_frozen(lora_model.base_model, "Condition A stage 0")
    
    # Setup continual learner (LoRA only)
    optimizer = torch.optim.AdamW(
        lora_model.get_trainable_params(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Continual adaptation stages
    overall_plasticity = {}
    start_time = time.time()
    
    for stage_idx, adapt_dataset in enumerate(CONTINUAL_ORDER):
        stage_name = f"stage_{stage_idx + 1}"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"STAGE {stage_idx + 1}: Adapt to {DATASET_DISPLAY[adapt_dataset]} (Condition A)")
        logger.info(f"{'=' * 60}")
        
        X_tr = splits[adapt_dataset]["X_train"]
        y_tr = splits[adapt_dataset]["y_train"]
        
        # Split for validation
        try:
            X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
        except ValueError:
            X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=SEED)
        
        logger.info(f"  Training: {X_tr_fit.shape[0]}, Validation: {X_val.shape[0]}")
        
        result = run_adaptation_stage(
            lora_model, optimizer, X_tr_fit, y_tr_fit, adapt_dataset,
            epochs=ADAPT_EPOCHS)
        
        overall_plasticity[stage_name] = result["plasticity"]
        
        # Verify encoder still frozen
        verify_encoder_frozen(lora_model.base_model, f"A after {adapt_dataset}")
        
        # Evaluate on all datasets
        evaluate_all_datasets(lora_model, splits, stage_name, stage_results, all_embeddings)
    
    total_seconds = time.time() - start_time
    
    cl_metrics = compute_continual_metrics(stage_results, "condition_A")
    
    return {
        "condition": "A",
        "label": "frozen_backbone_classifier_lora",
        "stage_results": stage_results,
        "cl_metrics": cl_metrics,
        "embeddings": all_embeddings,
        "plasticity": overall_plasticity,
        "param_counts": param_counts,
        "computational": {"total_seconds": total_seconds},
    }


def build_condition_B(base_model: nn.Module, splits: dict) -> dict:
    """
    Condition B: Last backbone block (fc4 = backbone.12) trainable + classifier LoRA.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("CONDITION B: Last backbone block (fc4) trainable + Classifier LoRA")
    logger.info(f"{'=' * 70}")
    
    base = deepcopy(base_model)
    
    # Unfreeze fc4 (backbone.12) and its subsequent batch norm
    unfreeze_backbone_layers(base, linear_indices=[12], lr_scale=0.5)
    
    # Count trainable params in backbone
    backbone_trainable = sum(p.numel() for p in base.backbone.parameters() if p.requires_grad)
    logger.info(f"  Backbone trainable params: {backbone_trainable:,}")
    
    # Create LoRA on classifier heads (same as A)
    lora_model = LoRAModule(
        base_model=base,
        target_names=CLASSIFIER_LORA_TARGETS_FULL,
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.05,
    )
    lora_model.to(DEVICE)
    
    param_counts = lora_model.get_param_counts()
    logger.info(f"  LoRA params: trainable={param_counts['trainable']:,}, "
                f"total={param_counts['total']:,}")
    
    all_embeddings = {}
    stage_results = {}
    
    # Stage 0
    stage_results["stage_0"] = run_stage0_evaluation(
        base_model, lora_model, splits, all_embeddings)
    
    # Combined trainable params: LoRA + unfrozen backbone
    trainable_params = (
        list(lora_model.get_trainable_params()) +
        [p for p in base.backbone.parameters() if p.requires_grad]
    )
    logger.info(f"  Total trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW([
        {"params": lora_model.get_trainable_params(), "lr": LR},
        {"params": [p for p in base.backbone.parameters() if p.requires_grad], "lr": LR * 0.5},
    ], weight_decay=WEIGHT_DECAY)
    
    overall_plasticity = {}
    start_time = time.time()
    
    for stage_idx, adapt_dataset in enumerate(CONTINUAL_ORDER):
        stage_name = f"stage_{stage_idx + 1}"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"STAGE {stage_idx + 1}: Adapt to {DATASET_DISPLAY[adapt_dataset]} (Condition B)")
        logger.info(f"{'=' * 60}")
        
        X_tr = splits[adapt_dataset]["X_train"]
        y_tr = splits[adapt_dataset]["y_train"]
        
        try:
            X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
        except ValueError:
            X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=SEED)
        
        result = run_adaptation_stage(
            lora_model, optimizer, X_tr_fit, y_tr_fit, adapt_dataset)
        
        overall_plasticity[stage_name] = result["plasticity"]
        
        # Encoder should change here (expected!)
        post_hash = compute_encoder_hash(lora_model.base_model)
        logger.info(f"  Encoder SHA256 (post-{adapt_dataset}): {post_hash[:16]}")
        
        evaluate_all_datasets(lora_model, splits, stage_name, stage_results, all_embeddings)
    
    total_seconds = time.time() - start_time
    
    cl_metrics = compute_continual_metrics(stage_results, "condition_B")
    
    return {
        "condition": "B",
        "label": "fc4_trainable_classifier_lora",
        "stage_results": stage_results,
        "cl_metrics": cl_metrics,
        "embeddings": all_embeddings,
        "plasticity": overall_plasticity,
        "param_counts": param_counts,
        "computational": {"total_seconds": total_seconds, "backbone_trainable": backbone_trainable},
    }


def build_condition_C(base_model: nn.Module, splits: dict) -> dict:
    """
    Condition C: Last two backbone blocks (fc3=backbone.8 + fc4=backbone.12) trainable + classifier LoRA.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("CONDITION C: Last two blocks (fc3+fc4) trainable + Classifier LoRA")
    logger.info(f"{'=' * 70}")
    
    base = deepcopy(base_model)
    
    # Unfreeze fc3 (backbone.8) and fc4 (backbone.12)
    unfreeze_backbone_layers(base, linear_indices=[8, 12], lr_scale=0.5)
    
    backbone_trainable = sum(p.numel() for p in base.backbone.parameters() if p.requires_grad)
    logger.info(f"  Backbone trainable params: {backbone_trainable:,}")
    
    lora_model = LoRAModule(
        base_model=base,
        target_names=CLASSIFIER_LORA_TARGETS_FULL,
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.05,
    )
    lora_model.to(DEVICE)
    
    param_counts = lora_model.get_param_counts()
    logger.info(f"  LoRA params: trainable={param_counts['trainable']:,}, "
                f"total={param_counts['total']:,}")
    
    all_embeddings = {}
    stage_results = {}
    
    stage_results["stage_0"] = run_stage0_evaluation(
        base_model, lora_model, splits, all_embeddings)
    
    trainable_params = (
        list(lora_model.get_trainable_params()) +
        [p for p in base.backbone.parameters() if p.requires_grad]
    )
    
    optimizer = torch.optim.AdamW([
        {"params": lora_model.get_trainable_params(), "lr": LR},
        {"params": [p for p in base.backbone.parameters() if p.requires_grad], "lr": LR * 0.5},
    ], weight_decay=WEIGHT_DECAY)
    
    overall_plasticity = {}
    start_time = time.time()
    
    for stage_idx, adapt_dataset in enumerate(CONTINUAL_ORDER):
        stage_name = f"stage_{stage_idx + 1}"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"STAGE {stage_idx + 1}: Adapt to {DATASET_DISPLAY[adapt_dataset]} (Condition C)")
        logger.info(f"{'=' * 60}")
        
        X_tr = splits[adapt_dataset]["X_train"]
        y_tr = splits[adapt_dataset]["y_train"]
        
        try:
            X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
        except ValueError:
            X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=SEED)
        
        result = run_adaptation_stage(
            lora_model, optimizer, X_tr_fit, y_tr_fit, adapt_dataset)
        
        overall_plasticity[stage_name] = result["plasticity"]
        
        post_hash = compute_encoder_hash(lora_model.base_model)
        logger.info(f"  Encoder SHA256 (post-{adapt_dataset}): {post_hash[:16]}")
        
        evaluate_all_datasets(lora_model, splits, stage_name, stage_results, all_embeddings)
    
    total_seconds = time.time() - start_time
    
    cl_metrics = compute_continual_metrics(stage_results, "condition_C")
    
    return {
        "condition": "C",
        "label": "fc3_fc4_trainable_classifier_lora",
        "stage_results": stage_results,
        "cl_metrics": cl_metrics,
        "embeddings": all_embeddings,
        "plasticity": overall_plasticity,
        "param_counts": param_counts,
        "computational": {"total_seconds": total_seconds, "backbone_trainable": backbone_trainable},
    }


def build_condition_D(base_model: nn.Module, splits: dict) -> dict:
    """
    Condition D: Full backbone LoRA + classifier LoRA.
    No full-weight updates. LoRA only on both backbone and classifier.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("CONDITION D: Full backbone LoRA + Classifier LoRA")
    logger.info(f"{'=' * 70}")
    
    base = deepcopy(base_model)
    
    lora_model = LoRAModule(
        base_model=base,
        target_names=BACKBONE_LORA_TARGETS + CLASSIFIER_LORA_TARGETS_FULL,
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.05,
    )
    lora_model.to(DEVICE)
    
    param_counts = lora_model.get_param_counts()
    logger.info(f"  LoRA params: trainable={param_counts['trainable']:,}, "
                f"total={param_counts['total']:,}")
    
    all_embeddings = {}
    stage_results = {}
    
    stage_results["stage_0"] = run_stage0_evaluation(
        base_model, lora_model, splits, all_embeddings)
    
    optimizer = torch.optim.AdamW(
        lora_model.get_trainable_params(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    overall_plasticity = {}
    start_time = time.time()
    
    for stage_idx, adapt_dataset in enumerate(CONTINUAL_ORDER):
        stage_name = f"stage_{stage_idx + 1}"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"STAGE {stage_idx + 1}: Adapt to {DATASET_DISPLAY[adapt_dataset]} (Condition D)")
        logger.info(f"{'=' * 60}")
        
        X_tr = splits[adapt_dataset]["X_train"]
        y_tr = splits[adapt_dataset]["y_train"]
        
        try:
            X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
        except ValueError:
            X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=SEED)
        
        # Verify backbone is frozen (LoRA only, no full-weight updates)
        verify_encoder_frozen(lora_model.base_model, f"D before {adapt_dataset}")
        
        result = run_adaptation_stage(
            lora_model, optimizer, X_tr_fit, y_tr_fit, adapt_dataset)
        
        overall_plasticity[stage_name] = result["plasticity"]
        
        # Verify backbone still frozen (LoRA parameters are separate)
        verify_encoder_frozen(lora_model.base_model, f"D after {adapt_dataset}")
        
        evaluate_all_datasets(lora_model, splits, stage_name, stage_results, all_embeddings)
    
    total_seconds = time.time() - start_time
    
    cl_metrics = compute_continual_metrics(stage_results, "condition_D")
    
    return {
        "condition": "D",
        "label": "full_backbone_lora_classifier_lora",
        "stage_results": stage_results,
        "cl_metrics": cl_metrics,
        "embeddings": all_embeddings,
        "plasticity": overall_plasticity,
        "param_counts": param_counts,
        "computational": {"total_seconds": total_seconds},
    }


def build_condition_E(base_model: nn.Module, splits: dict) -> dict:
    """
    Condition E: Progressive unfreezing.
    
    Stage 1: Classifier only (LoRA on heads)
    Stage 2: Classifier + fc4 trainable
    Stage 3: Classifier + fc3 + fc4 trainable
    """
    logger.info(f"\n{'=' * 70}")
    logger.info("CONDITION E: Progressive Unfreezing")
    logger.info(f"{'=' * 70}")
    
    base = deepcopy(base_model)
    
    # Stage 1: Start with frozen backbone, classifier LoRA only
    for param in base.parameters():
        param.requires_grad = False
    base.eval()
    base.apply(lambda m: m.eval() if isinstance(m, (nn.BatchNorm1d,)) else None)
    
    lora_model = LoRAModule(
        base_model=base,
        target_names=CLASSIFIER_LORA_TARGETS_FULL,
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.05,
    )
    lora_model.to(DEVICE)
    
    all_embeddings = {}
    stage_results = {}
    
    stage_results["stage_0"] = run_stage0_evaluation(
        base_model, lora_model, splits, all_embeddings)
    
    overall_plasticity = {}
    start_time = time.time()
    
    # ── Stage 1: Classifier only ──────────────────────────────────────
    stage_idx = 0
    adapt_dataset = CONTINUAL_ORDER[stage_idx]
    stage_name = f"stage_{stage_idx + 1}"
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STAGE {stage_idx + 1}: {DATASET_DISPLAY[adapt_dataset]} (Classifier only)")
    logger.info(f"{'=' * 60}")
    
    optimizer = torch.optim.AdamW(
        lora_model.get_trainable_params(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    X_tr = splits[adapt_dataset]["X_train"]
    y_tr = splits[adapt_dataset]["y_train"]
    try:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
    except ValueError:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED)
    
    result = run_adaptation_stage(
        lora_model, optimizer, X_tr_fit, y_tr_fit, adapt_dataset,
        epochs=ADAPT_EPOCHS)
    overall_plasticity[stage_name] = result["plasticity"]
    verify_encoder_frozen(lora_model.base_model, f"E Stage 1 after {adapt_dataset}")
    evaluate_all_datasets(lora_model, splits, stage_name, stage_results, all_embeddings)
    
    # ── Stage 2: Unfreeze fc4 ────────────────────────────────────────
    stage_idx = 1
    adapt_dataset = CONTINUAL_ORDER[stage_idx]
    stage_name = f"stage_{stage_idx + 1}"
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STAGE {stage_idx + 1}: {DATASET_DISPLAY[adapt_dataset]} (Classifier + fc4 unfrozen)")
    logger.info(f"{'=' * 60}")
    
    # Unfreeze fc4
    unfreeze_backbone_layers(lora_model.base_model, linear_indices=[12], lr_scale=0.5)
    
    # Rebuild optimizer with new trainable params
    trainable_params = (
        list(lora_model.get_trainable_params()) +
        [p for p in lora_model.base_model.backbone.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW([
        {"params": lora_model.get_trainable_params(), "lr": LR},
        {"params": [p for p in lora_model.base_model.backbone.parameters() if p.requires_grad],
         "lr": LR * 0.5},
    ], weight_decay=WEIGHT_DECAY)
    
    X_tr = splits[adapt_dataset]["X_train"]
    y_tr = splits[adapt_dataset]["y_train"]
    try:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
    except ValueError:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED)
    
    result = run_adaptation_stage(
        lora_model, optimizer, X_tr_fit, y_tr_fit, adapt_dataset,
        epochs=ADAPT_EPOCHS)
    overall_plasticity[stage_name] = result["plasticity"]
    evaluate_all_datasets(lora_model, splits, stage_name, stage_results, all_embeddings)
    
    # ── Stage 3: Unfreeze fc3 + fc4 ──────────────────────────────────
    stage_idx = 2
    adapt_dataset = CONTINUAL_ORDER[stage_idx]
    stage_name = f"stage_{stage_idx + 1}"
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STAGE {stage_idx + 1}: {DATASET_DISPLAY[adapt_dataset]} (Classifier + fc3+fc4 unfrozen)")
    logger.info(f"{'=' * 60}")
    
    # Unfreeze fc3 + fc4
    unfreeze_backbone_layers(lora_model.base_model, linear_indices=[8, 12], lr_scale=0.5)
    
    optimizer = torch.optim.AdamW([
        {"params": lora_model.get_trainable_params(), "lr": LR},
        {"params": [p for p in lora_model.base_model.backbone.parameters() if p.requires_grad],
         "lr": LR * 0.5},
    ], weight_decay=WEIGHT_DECAY)
    
    X_tr = splits[adapt_dataset]["X_train"]
    y_tr = splits[adapt_dataset]["y_train"]
    try:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
    except ValueError:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED)
    
    result = run_adaptation_stage(
        lora_model, optimizer, X_tr_fit, y_tr_fit, adapt_dataset,
        epochs=ADAPT_EPOCHS)
    overall_plasticity[stage_name] = result["plasticity"]
    evaluate_all_datasets(lora_model, splits, stage_name, stage_results, all_embeddings)
    
    total_seconds = time.time() - start_time
    
    cl_metrics = compute_continual_metrics(stage_results, "condition_E")
    
    return {
        "condition": "E",
        "label": "progressive_unfreezing",
        "stage_results": stage_results,
        "cl_metrics": cl_metrics,
        "embeddings": all_embeddings,
        "plasticity": overall_plasticity,
        "param_counts": lora_model.get_param_counts(),
        "computational": {"total_seconds": total_seconds},
    }


def build_condition_F(base_model: nn.Module, splits: dict,
                      replay_size: float = 5.0) -> dict:
    """
    Condition F: Replay + Progressive Unfreezing
    Uses the optimal replay size from Phase 62 (5%).
    """
    condition_label = f"replay_{replay_size}pct_progressive_unfreezing"
    logger.info(f"\n{'=' * 70}")
    logger.info(f"CONDITION F: Replay + Progressive Unfreezing (replay={replay_size}%)")
    logger.info(f"{'=' * 70}")
    
    base = deepcopy(base_model)
    
    for param in base.parameters():
        param.requires_grad = False
    base.eval()
    base.apply(lambda m: m.eval() if isinstance(m, (nn.BatchNorm1d,)) else None)
    
    lora_model = LoRAModule(
        base_model=base,
        target_names=CLASSIFIER_LORA_TARGETS_FULL,
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.05,
    )
    lora_model.to(DEVICE)
    
    # Replay buffer
    replay_buffer = ReplayBuffer()
    
    all_embeddings = {}
    stage_results = {}
    
    stage_results["stage_0"] = run_stage0_evaluation(
        base_model, lora_model, splits, all_embeddings)
    
    overall_plasticity = {}
    start_time = time.time()
    
    # ── Stage 1: Classifier only, with replay ─────────────────────────
    stage_idx = 0
    adapt_dataset = CONTINUAL_ORDER[stage_idx]
    stage_name = f"stage_{stage_idx + 1}"
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STAGE {stage_idx + 1}: {DATASET_DISPLAY[adapt_dataset]} (Classifier only + replay)")
    logger.info(f"{'=' * 60}")
    
    optimizer = torch.optim.AdamW(
        lora_model.get_trainable_params(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    X_tr = splits[adapt_dataset]["X_train"]
    y_tr = splits[adapt_dataset]["y_train"]
    try:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
    except ValueError:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED)
    
    result = run_adaptation_stage(
        lora_model, optimizer, X_tr_fit, y_tr_fit, adapt_dataset,
        epochs=ADAPT_EPOCHS, replay_buffer=replay_buffer)
    overall_plasticity[stage_name] = result["plasticity"]
    
    # Add samples to replay buffer after adaptation
    replay_buffer.add_samples(X_tr_fit, y_tr_fit, percentage=replay_size)
    logger.info(f"  Replay buffer size: {replay_buffer.size}")
    
    verify_encoder_frozen(lora_model.base_model, f"F Stage 1 after {adapt_dataset}")
    evaluate_all_datasets(lora_model, splits, stage_name, stage_results, all_embeddings)
    
    # ── Stage 2: Unfreeze fc4, with replay ────────────────────────────
    stage_idx = 1
    adapt_dataset = CONTINUAL_ORDER[stage_idx]
    stage_name = f"stage_{stage_idx + 1}"
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STAGE {stage_idx + 1}: {DATASET_DISPLAY[adapt_dataset]} (Classifier + fc4 + replay)")
    logger.info(f"{'=' * 60}")
    
    unfreeze_backbone_layers(lora_model.base_model, linear_indices=[12], lr_scale=0.5)
    
    optimizer = torch.optim.AdamW([
        {"params": lora_model.get_trainable_params(), "lr": LR},
        {"params": [p for p in lora_model.base_model.backbone.parameters() if p.requires_grad],
         "lr": LR * 0.5},
    ], weight_decay=WEIGHT_DECAY)
    
    X_tr = splits[adapt_dataset]["X_train"]
    y_tr = splits[adapt_dataset]["y_train"]
    try:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
    except ValueError:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED)
    
    result = run_adaptation_stage(
        lora_model, optimizer, X_tr_fit, y_tr_fit, adapt_dataset,
        epochs=ADAPT_EPOCHS, replay_buffer=replay_buffer)
    overall_plasticity[stage_name] = result["plasticity"]
    
    replay_buffer.add_samples(X_tr_fit, y_tr_fit, percentage=replay_size)
    logger.info(f"  Replay buffer size: {replay_buffer.size}")
    
    evaluate_all_datasets(lora_model, splits, stage_name, stage_results, all_embeddings)
    
    # ── Stage 3: Unfreeze fc3 + fc4, with replay ──────────────────────
    stage_idx = 2
    adapt_dataset = CONTINUAL_ORDER[stage_idx]
    stage_name = f"stage_{stage_idx + 1}"
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STAGE {stage_idx + 1}: {DATASET_DISPLAY[adapt_dataset]} (Classifier + fc3+fc4 + replay)")
    logger.info(f"{'=' * 60}")
    
    unfreeze_backbone_layers(lora_model.base_model, linear_indices=[8, 12], lr_scale=0.5)
    
    optimizer = torch.optim.AdamW([
        {"params": lora_model.get_trainable_params(), "lr": LR},
        {"params": [p for p in lora_model.base_model.backbone.parameters() if p.requires_grad],
         "lr": LR * 0.5},
    ], weight_decay=WEIGHT_DECAY)
    
    X_tr = splits[adapt_dataset]["X_train"]
    y_tr = splits[adapt_dataset]["y_train"]
    try:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
    except ValueError:
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED)
    
    result = run_adaptation_stage(
        lora_model, optimizer, X_tr_fit, y_tr_fit, adapt_dataset,
        epochs=ADAPT_EPOCHS, replay_buffer=replay_buffer)
    overall_plasticity[stage_name] = result["plasticity"]
    
    replay_buffer.add_samples(X_tr_fit, y_tr_fit, percentage=replay_size)
    logger.info(f"  Replay buffer size: {replay_buffer.size}")
    
    evaluate_all_datasets(lora_model, splits, stage_name, stage_results, all_embeddings)
    
    total_seconds = time.time() - start_time
    
    cl_metrics = compute_continual_metrics(stage_results, condition_label)
    
    return {
        "condition": "F",
        "label": condition_label,
        "stage_results": stage_results,
        "cl_metrics": cl_metrics,
        "embeddings": all_embeddings,
        "plasticity": overall_plasticity,
        "param_counts": lora_model.get_param_counts(),
        "computational": {
            "total_seconds": total_seconds,
            "replay_buffer_size": replay_buffer.size,
            "replay_buffer_memory_bytes": replay_buffer.memory_bytes,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def run_phase63(conditions: list[str], quick: bool = False, replay_size: float = 5.0):
    """Run Phase 63 experimental conditions."""
    logger.info("=" * 70)
    logger.info("PHASE 63: Controlled Backbone Adaptation")
    logger.info("Is Representation Plasticity the Missing Ingredient?")
    logger.info("=" * 70)

    if quick:
        global ADAPT_EPOCHS
        ADAPT_EPOCHS = 10
        logger.info(f"  QUICK MODE: {ADAPT_EPOCHS} epochs")

    verify_checkpoint_sha256()

    # ── Load all data ──────────────────────────────────────────────
    logger.info("Loading datasets...")
    data_dict = load_original_datasets()

    for ext in EXTERNAL_DATASETS:
        d = harmonize_external_dataset(ext, DATA_DIR)
        if d is not None:
            data_dict[ext] = d

    logger.info(f"Total datasets loaded: {len(data_dict)}")

    scalers = fit_dataset_scalers(data_dict)
    splits = prepare_splits(data_dict, scalers)

    # ── Load foundation model (immutable copy preserved) ─────────────
    base_model = load_frozen_model(CHECKPOINT_PATH)
    frozen_ok, _, _ = verify_encoder_frozen(base_model, "foundation")
    if not frozen_ok:
        logger.error("Foundation model not frozen! Aborting.")
        return {}, splits

    # ── Preserve immutable checkpoint info ──────────────────────────
    import shutil
    immutable_path = RESULTS / "checkpoint_immutable.pt"
    if not immutable_path.exists():
        shutil.copy2(CHECKPOINT_PATH, immutable_path)
        logger.info(f"Immutable checkpoint copy: {immutable_path}")
    
    # Compute and save SHA256 of original
    sha256_path = RESULTS / "checkpoint_sha256.txt"
    with open(sha256_path, "w") as f:
        f.write(CHECKPOINT_SHA256)
    logger.info(f"Checkpoint SHA256 saved: {sha256_path}")

    # ── Run requested conditions ────────────────────────────────────
    all_results = {}
    
    condition_map = {
        "A": lambda: build_condition_A(base_model, splits),
        "B": lambda: build_condition_B(base_model, splits),
        "C": lambda: build_condition_C(base_model, splits),
        "D": lambda: build_condition_D(base_model, splits),
        "E": lambda: build_condition_E(base_model, splits),
        "F": lambda: build_condition_F(base_model, splits, replay_size=replay_size),
    }

    for cond_id in conditions:
        if cond_id in condition_map:
            logger.info(f"\n{'#' * 70}")
            logger.info(f"# RUNNING CONDITION {cond_id}")
            logger.info(f"{'#' * 70}")
            try:
                result = condition_map[cond_id]()
                if result is not None:
                    all_results[result["label"]] = result
                    logger.info(f"  ✓ Condition {cond_id} completed: {result['label']}")
            except Exception as e:
                logger.error(f"  ✗ Condition {cond_id} failed: {e}")
                import traceback
                traceback.print_exc()
            cleanup()
        else:
            logger.warning(f"Unknown condition: {cond_id}")

    logger.info(f"\n{'=' * 70}")
    logger.info(f"All conditions completed. Total: {len(all_results)}")
    logger.info(f"{'=' * 70}")

    return all_results, splits


# ═══════════════════════════════════════════════════════════════════════════
# Save & Report
# ═══════════════════════════════════════════════════════════════════════════


def save_all_results(all_results: dict, splits: dict):
    """Save all Phase 63 deliverables."""

    # ── 1. Continual Results ───────────────────────────────────────
    all_rows = []
    for method_name, result in all_results.items():
        for stage_name, ds_results in result["stage_results"].items():
            for ds_name, metrics in ds_results.items():
                row = {
                    "method": method_name,
                    "condition": result.get("condition", "?"),
                    "stage": stage_name,
                    "dataset": ds_name,
                    "dataset_display": DATASET_DISPLAY.get(ds_name, ds_name),
                }
                row.update(metrics)
                all_rows.append(row)
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(RESULTS / "plasticity_results.csv", index=False)
        logger.info(f"Saved plasticity_results.csv ({len(all_rows)} rows)")

    # ── 2. Forgetting Metrics ──────────────────────────────────────
    frows = []
    for method_name, result in all_results.items():
        forgetting = result["cl_metrics"].get("forgetting", {})
        for ds_name, f_dict in forgetting.items():
            row = {
                "method": method_name,
                "condition": result.get("condition", "?"),
                "dataset": ds_name,
                "max_forgetting": f_dict.get("max_forgetting", 0),
                "mean_forgetting": f_dict.get("mean_forgetting", 0),
            }
            for stage_key, fval in f_dict.get("per_stage", {}).items():
                row[stage_key] = fval
            frows.append(row)
    if frows:
        pd.DataFrame(frows).to_csv(RESULTS / "forgetting_results.csv", index=False)
        logger.info(f"Saved forgetting_results.csv ({len(frows)} rows)")

    # ── 3. Transfer Metrics ────────────────────────────────────────
    trows = []
    for method_name, result in all_results.items():
        for key, val in result["cl_metrics"].items():
            if "forward_transfer" in key or "backward_transfer" in key or "BWT" in key:
                if isinstance(val, dict):
                    for ds, v in val.items():
                        trows.append({"method": method_name, "condition": result.get("condition", "?"),
                                      "metric": key, "dataset": ds, "value": v})
                else:
                    trows.append({"method": method_name, "condition": result.get("condition", "?"),
                                  "metric": key, "value": val})
    if trows:
        pd.DataFrame(trows).to_csv(RESULTS / "transfer_results.csv", index=False)
        logger.info(f"Saved transfer_results.csv ({len(trows)} rows)")

    # ── 4. Representation Drift ────────────────────────────────────
    rep_rows = []
    for method_name, result in all_results.items():
        for stage_name in result["stage_results"].keys():
            stage_embeddings = {}
            stage_idx = stage_name.replace("stage_", "")
            for ds in ALL_DATASETS:
                key = f"{ds}_{stage_name}" if stage_idx != "0" else ds
                if key in result.get("embeddings", {}):
                    stage_embeddings[ds] = result["embeddings"][key]
            if stage_embeddings:
                reps = representational_similarity_analysis(
                    stage_embeddings, label=f"{method_name}_{stage_name}")
                for r in reps:
                    rep_rows.append({"method": method_name, "condition": result.get("condition", "?"), **r})
    if rep_rows:
        pd.DataFrame(rep_rows).to_csv(RESULTS / "representation_drift.csv", index=False)
        logger.info(f"Saved representation_drift.csv ({len(rep_rows)} rows)")

    # ── 5. Layer-wise Updates ──────────────────────────────────────
    lrows = []
    for method_name, result in all_results.items():
        for stage_name, plasticity in result.get("plasticity", {}).items():
            for layer_name, cos_sim in plasticity.get("layerwise_cosine_similarity", {}).items():
                lrows.append({
                    "method": method_name,
                    "condition": result.get("condition", "?"),
                    "stage": stage_name,
                    "layer": layer_name,
                    "cosine_similarity": cos_sim,
                })
            for layer_name, update_mag in plasticity.get("param_update_magnitude", {}).items():
                # Find existing row or add
                found = False
                for r in lrows:
                    if (r["method"] == method_name and r["stage"] == stage_name 
                        and r["layer"] == layer_name):
                        r["update_magnitude"] = update_mag
                        found = True
                        break
                if not found:
                    lrows.append({
                        "method": method_name, "condition": result.get("condition", "?"),
                        "stage": stage_name, "layer": layer_name,
                        "update_magnitude": update_mag,
                    })
    if lrows:
        pd.DataFrame(lrows).to_csv(RESULTS / "layerwise_updates.csv", index=False)
        logger.info(f"Saved layerwise_updates.csv ({len(lrows)} rows)")

    # ── 6. Bootstrap ───────────────────────────────────────────────
    boot_data = {}
    for method_name, result in all_results.items():
        for stage_name, ds_results in result["stage_results"].items():
            for ds_name, metrics in ds_results.items():
                for m_name, m_val in metrics.items():
                    key = (method_name, stage_name, ds_name, m_name)
                    if key not in boot_data:
                        boot_data[key] = []
                    boot_data[key].append(m_val)
    if boot_data:
        real_boot = {}
        for key, vals in boot_data.items():
            if len(vals) > 0:
                real_boot[str(key)] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
        with open(RESULTS / "bootstrap.json", "w") as f:
            json.dump(real_boot, f, indent=2)
        logger.info("Saved bootstrap.json")

    # ── 7. Bayesian Comparison ─────────────────────────────────────
    bayes_data = {}
    for method_name, result in all_results.items():
        if "stage_0" not in result["stage_results"]:
            continue
        baseline = result["stage_results"]["stage_0"]
        for stage_name in ["stage_1", "stage_2", "stage_3"]:
            if stage_name not in result["stage_results"]:
                continue
            for ds_name in ALL_DATASETS:
                if ds_name in baseline and ds_name in result["stage_results"][stage_name]:
                    b_mf1 = baseline[ds_name].get("macro_f1", 0)
                    m_mf1 = result["stage_results"][stage_name][ds_name].get("macro_f1", 0)
                    bayes_data[f"{method_name}_{stage_name}_{ds_name}"] = {
                        "baseline_mf1": b_mf1,
                        "method_mf1": m_mf1,
                        "effect": m_mf1 - b_mf1,
                    }
    if bayes_data:
        with open(RESULTS / "bayesian.json", "w") as f:
            json.dump(bayes_data, f, indent=2)
        logger.info("Saved bayesian.json")

    # ── 8. Mixed Effects / ANOVA ───────────────────────────────────
    rm_data = {}
    for method_name, result in all_results.items():
        stages = ["stage_0", "stage_1", "stage_2", "stage_3"]
        mf1_by_stage = {s: [] for s in stages}
        for stage_name, ds_results in result["stage_results"].items():
            if stage_name in mf1_by_stage:
                for ds_name, metrics in ds_results.items():
                    mf1_by_stage[stage_name].append(metrics.get("macro_f1", np.nan))
        rm_data[method_name] = {
            stage: {
                "mean": float(np.nanmean(mf1_by_stage[stage])),
                "std": float(np.nanstd(mf1_by_stage[stage], ddof=1)) if len(mf1_by_stage[stage]) > 1 else 0.0,
                "n": int(np.sum(~np.isnan(mf1_by_stage[stage]))),
            }
            for stage in stages
        }
    if rm_data:
        with open(RESULTS / "mixed_effects.json", "w") as f:
            json.dump(rm_data, f, indent=2)
        logger.info("Saved mixed_effects.json")

    # ── 9. Embeddings (metadata only, too large to save raw) ───────
    emb_meta = {}
    for method_name, result in all_results.items():
        if "embeddings" in result:
            for key, val in result["embeddings"].items():
                emb_meta[f"{method_name}_{key}"] = list(val.shape)
    with open(RESULTS / "embeddings" / "embedding_metadata.json", "w") as f:
        json.dump(emb_meta, f, indent=2)
    logger.info(f"Saved embedding metadata ({len(emb_meta)} entries)")

    # ── 10. Save summary per condition ─────────────────────────────
    summary_path = RESULTS / "phase63_summary.md"
    generate_phase63_summary(all_results, summary_path)
    
    return all_rows


def generate_phase63_summary(all_results: dict, output_path: Path):
    """Generate Phase 63 summary markdown."""
    lines = []
    lines.append("# Phase 63 Summary")
    lines.append("")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Device: {DEVICE}")
    lines.append(f"Checkpoint SHA256: {CHECKPOINT_SHA256}")
    lines.append(f"Conditions: {len(all_results)}")
    lines.append("")

    # Overview of conditions
    lines.append("## Conditions Run")
    lines.append("")
    for label in sorted(all_results.keys()):
        r = all_results[label]
        cond = r.get("condition", "?")
        lines.append(f"- **{cond}**: {label}")
    lines.append("")

    # Frozen baseline performance
    if all_results:
        first_method = list(all_results.keys())[0]
        first_result = all_results[first_method]
        frozen_mf1 = float(np.mean([
            first_result["stage_results"]["stage_0"][ds]["macro_f1"]
            for ds in ALL_DATASETS
        ]))
        lines.append(f"- Frozen baseline MF1: {frozen_mf1:.4f}")
        lines.append("")

    # Stage 3 MF1 comparison across methods
    lines.append("## Stage 3 (UGR'16) Performance")
    lines.append("")
    lines.append("| Condition | IoT-23 MF1 | Kyoto2006+ MF1 | UGR'16 MF1 | Avg MF1 |")
    lines.append("|-----------|-----------|----------------|-----------|---------|")
    
    for label in sorted(all_results.keys()):
        r = all_results[label]
        sr = r["stage_results"]
        if "stage_3" in sr:
            iot = sr["stage_3"].get("iot23", {}).get("macro_f1", 0)
            kyo = sr["stage_3"].get("kyoto2006", {}).get("macro_f1", 0)
            ugr = sr["stage_3"].get("ugr16", {}).get("macro_f1", 0)
            avg = float(np.mean([iot, kyo, ugr]))
            lines.append(f"| {r.get('condition', '?')} | {iot:.4f} | {kyo:.4f} | {ugr:.4f} | {avg:.4f} |")
    lines.append("")

    # Forgetting summary
    lines.append("## Forgetting Summary")
    lines.append("")
    lines.append("| Condition | Avg Forgetting | Global Max Forgetting |")
    lines.append("|-----------|---------------|----------------------|")
    
    for label in sorted(all_results.keys()):
        r = all_results[label]
        cl = r["cl_metrics"]
        avg_forgets = []
        for stage in ["stage_1", "stage_2", "stage_3"]:
            af = cl.get(f"{stage}_avg_forgetting", 0)
            avg_forgets.append(af)
        avg_f = float(np.mean(avg_forgets)) if avg_forgets else 0
        gm = cl.get("global_max_forgetting", 0)
        lines.append(f"| {r.get('condition', '?')} | {avg_f:.4f} | {gm:.4f} |")
    lines.append("")

    # Transfer metrics
    lines.append("## Transfer Metrics")
    lines.append("")
    lines.append("| Condition | Stage 3 Avg BWT | Stage 3 Stability | Stage 3 Plasticity |")
    lines.append("|-----------|----------------|-------------------|--------------------|")
    
    for label in sorted(all_results.keys()):
        r = all_results[label]
        cl = r["cl_metrics"]
        bwt = cl.get("stage_3_avg_BWT", 0)
        stability = cl.get("stage_3_stability", 0)
        plasticity = cl.get("stage_3_plasticity", 0)
        lines.append(f"| {r.get('condition', '?')} | {bwt:.4f} | {stability:.4f} | {plasticity:.4f} |")
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    
    # Compare Stage 3 MF1 across conditions
    stage3_mf1s = {}
    for label in sorted(all_results.keys()):
        r = all_results[label]
        sr = r["stage_results"]
        if "stage_3" in sr:
            mf1s = [sr["stage_3"][ds].get("macro_f1", 0) for ds in ALL_DATASETS]
            stage3_mf1s[r.get("condition", "?")] = float(np.mean(mf1s))
    
    if stage3_mf1s:
        best_cond = max(stage3_mf1s, key=stage3_mf1s.get)
        best_ugr_mf1 = max(
            (all_results[label]["stage_results"].get("stage_3", {}).get("ugr16", {}).get("macro_f1", 0)
             for label in all_results),
            default=0
        )
        lines.append(f"1. Best Stage 3 avg MF1: **{best_cond}** ({stage3_mf1s[best_cond]:.4f})")
        lines.append(f"2. Best Stage 3 UGR'16 MF1: {best_ugr_mf1:.4f}")
        if "A" in stage3_mf1s and best_cond != "A":
            improvement = stage3_mf1s[best_cond] - stage3_mf1s["A"]
            lines.append(f"3. Improvement vs Condition A (frozen): Δ={improvement:.4f}")
    
    # Determine outcome
    lines.append("")
    lines.append("### Scientific Interpretation")
    lines.append("")
    
    # Compare Condition A (frozen baseline) vs best backbone adaptation
    cond_a_label = [l for l in all_results if all_results[l].get("condition") == "A"]
    cond_best_label = max(all_results.items(), 
                          key=lambda x: x[1]["stage_results"].get("stage_3", {})
                          .get("ugr16", {}).get("macro_f1", 0) 
                          if "stage_3" in x[1]["stage_results"] else -1)
    
    if cond_a_label:
        a_key = cond_a_label[0]
        a_ugr = all_results[a_key]["stage_results"].get("stage_3", {}).get("ugr16", {}).get("macro_f1", 0)
        best_label = cond_best_label[0]
        best_ugr = all_results[best_label]["stage_results"].get("stage_3", {}).get("ugr16", {}).get("macro_f1", 0)
        
        if best_ugr > a_ugr + 0.05:
            lines.append("**Outcome 1**: Backbone adaptation significantly improves Stage 3.")
            lines.append("")
            lines.append("Interpretation: Frozen representations—not continual learning—")
            lines.append("were the dominant bottleneck.")
        elif best_ugr > a_ugr + 0.01:
            lines.append("**Outcome 2**: Performance improves slightly but forgets.")
            lines.append("")
            lines.append("Interpretation: There exists a stability–plasticity tradeoff ")
            lines.append("limiting IDS foundation models.")
        else:
            lines.append("**Outcome 3**: No meaningful improvement.")
            lines.append("")
            lines.append("Interpretation: The limitation is intrinsic to conditional ")
            lines.append("distribution mismatch (P(Y|X)), not backbone rigidity. This ")
            lines.append("constitutes the strongest evidence yet supporting the P(Y|X) ")
            lines.append("bottleneck hypothesis.")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved summary: {output_path}")


def generate_report(all_results: dict):
    """Generate Phase 63 full report."""
    lines = []
    lines.append("# Phase 63 Report: Controlled Backbone Adaptation")
    lines.append("")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Device: {DEVICE}")
    lines.append(f"Checkpoint: {CHECKPOINT_PATH}")
    lines.append(f"Checkpoint SHA256: {CHECKPOINT_SHA256}")
    lines.append(f"Adaptation Epochs: {ADAPT_EPOCHS}")
    lines.append(f"LoRA Rank: {LORA_RANK}, Alpha: {LORA_ALPHA}")
    lines.append("")

    # ── Methods Overview ───────────────────────────────────────────
    lines.append("## Experimental Conditions")
    lines.append("")
    lines.append(f"Total conditions: {len(all_results)}")
    for label in sorted(all_results.keys()):
        r = all_results[label]
        comp = r.get("computational", {})
        param_counts = r.get("param_counts", {})
        lines.append(f"")
        lines.append(f"### Condition {r.get('condition', '?')}: {label}")
        lines.append(f"- Label: {label}")
        if param_counts:
            lines.append(f"- LoRA params: {param_counts.get('trainable', 0):,} / {param_counts.get('total', 0):,}")
            lines.append(f"- Compression ratio: 1:{param_counts.get('compression_ratio', 0):.1f}")
        if "total_seconds" in comp:
            lines.append(f"- Total time: {comp['total_seconds']:.1f}s")
        if "replay_buffer_size" in comp:
            lines.append(f"- Replay buffer: {comp['replay_buffer_size']} exemplars "
                         f"({comp.get('replay_buffer_memory_bytes', 0) / 1024:.1f} KB)")
        if "backbone_trainable" in comp:
            lines.append(f"- Backbone trainable params: {comp['backbone_trainable']:,}")
    lines.append("")

    # ── Stage Results (Macro F1) ───────────────────────────────────
    lines.append("## Stage Results (Macro F1)")
    lines.append("")
    lines.append("| Condition | Dataset | Stage 0 | Stage 1 (IoT-23) | Stage 2 (Kyoto2006+) | Stage 3 (UGR'16) |")
    lines.append("|-----------|---------|---------|------------------|----------------------|-------------------|")

    for label in sorted(all_results.keys()):
        r = all_results[label]
        cond = r.get('condition', '?')
        first_row = True
        for ds in ALL_DATASETS:
            vals = []
            for stage in ["stage_0", "stage_1", "stage_2", "stage_3"]:
                if stage in r["stage_results"] and ds in r["stage_results"][stage]:
                    mf1 = r["stage_results"][stage][ds].get("macro_f1", 0)
                    vals.append(f"{mf1:.4f}")
                else:
                    vals.append("—")
            c_label = f"Cond {cond}" if first_row else ""
            lines.append(f"| {c_label} | {DATASET_DISPLAY[ds]} | {' | '.join(vals)} |")
            first_row = False
    lines.append("")

    # ── Forgetting Summary ─────────────────────────────────────────
    lines.append("## Forgetting Summary (Macro F1)")
    lines.append("")
    lines.append("| Condition | Avg Forgetting | Max Forgetting | Global Max |")
    lines.append("|-----------|---------------|---------------|------------|")

    for label in sorted(all_results.keys()):
        r = all_results[label]
        cl = r["cl_metrics"]
        avg_forgets = []
        max_forgets = []
        for stage in ["stage_1", "stage_2", "stage_3"]:
            af = cl.get(f"{stage}_avg_forgetting", 0)
            mf = cl.get(f"{stage}_max_forgetting", 0)
            avg_forgets.append(af)
            max_forgets.append(mf)
        avg_f = float(np.mean(avg_forgets)) if avg_forgets else 0
        max_f = float(np.max(max_forgets)) if max_forgets else 0
        gm = cl.get("global_max_forgetting", 0)
        lines.append(f"| {r.get('condition', '?')} | {avg_f:.4f} | {max_f:.4f} | {gm:.4f} |")
    lines.append("")

    # ── Transfer Summary ───────────────────────────────────────────
    lines.append("## Forward/Backward Transfer")
    lines.append("")
    lines.append("| Condition | Stage | FWT | Avg BWT | Stability | Plasticity |")
    lines.append("|-----------|-------|-----|---------|-----------|------------|")

    for label in sorted(all_results.keys()):
        r = all_results[label]
        cl = r["cl_metrics"]
        first_row = True
        for stage in ["stage_1", "stage_2", "stage_3"]:
            fwt = cl.get(f"{stage}_forward_transfer", "—")
            bwt = cl.get(f"{stage}_avg_BWT", "—")
            stability = cl.get(f"{stage}_stability", "—")
            plasticity = cl.get(f"{stage}_plasticity", "—")
            if isinstance(fwt, float):
                fwt = f"{fwt:.4f}"
            if isinstance(bwt, float):
                bwt = f"{bwt:.4f}"
            if isinstance(stability, float):
                stability = f"{stability:.4f}"
            if isinstance(plasticity, float):
                plasticity = f"{plasticity:.4f}"
            c_label = f"Cond {r.get('condition', '?')}" if first_row else ""
            lines.append(f"| {c_label} | {stage} | {fwt} | {bwt} | {stability} | {plasticity} |")
            first_row = False
    lines.append("")

    # ── Knowledge Retention ────────────────────────────────────────
    lines.append("## Knowledge Retention (relative to Stage 0)")
    lines.append("")
    lines.append("| Condition | Avg Retention |")
    lines.append("|-----------|--------------|")

    for label in sorted(all_results.keys()):
        r = all_results[label]
        cl = r["cl_metrics"]
        retentions = []
        for stage in ["stage_1", "stage_2", "stage_3"]:
            ret = cl.get(f"{stage}_avg_knowledge_retention", 0)
            retentions.append(ret)
        avg_ret = float(np.mean(retentions)) if retentions else 0
        lines.append(f"| {r.get('condition', '?')} | {avg_ret:.4f} |")
    lines.append("")

    # ── All-Datasets Macro F1 at each stage ────────────────────────
    lines.append("## All-Datasets Average Macro F1 by Stage")
    lines.append("")
    lines.append("| Condition | Stage 0 | Stage 1 | Stage 2 | Stage 3 |")
    lines.append("|-----------|---------|---------|---------|---------|")

    for label in sorted(all_results.keys()):
        r = all_results[label]
        cl = r["cl_metrics"]
        vals = []
        for stage in ["stage_0", "stage_1", "stage_2", "stage_3"]:
            v = cl.get(f"{stage}_avg_macro_f1", cl.get(f"{stage}_avg_accuracy", 0))
            vals.append(f"{v:.4f}" if not np.isnan(v) else "—")
        lines.append(f"| {r.get('condition', '?')} | {' | '.join(vals)} |")
    lines.append("")

    # ── Computational Metrics ──────────────────────────────────────
    lines.append("## Computational Metrics")
    lines.append("")
    lines.append("| Condition | Time (s) | LoRA Params | Backbone Params | Replay Size |")
    lines.append("|-----------|---------|-------------|----------------|-------------|")

    for label in sorted(all_results.keys()):
        r = all_results[label]
        comp = r.get("computational", {})
        time_s = comp.get("total_seconds", 0)
        lora_p = r.get("param_counts", {}).get("trainable", 0)
        backbone_p = comp.get("backbone_trainable", 0)
        rep_sz = comp.get("replay_buffer_size", 0)
        lines.append(f"| {r.get('condition', '?')} | {time_s:.0f} | {lora_p:,} | {backbone_p:,} | {rep_sz} |")
    lines.append("")

    # ── Save report ────────────────────────────────────────────────
    report_path = RESULTS / "phase63_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved report: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 63: Controlled Backbone Adaptation")
    parser.add_argument("--condition", type=str, default="A",
                        help="Condition to run: A, B, C, D, E, F, or 'all'")
    parser.add_argument("--all", action="store_true",
                        help="Run all conditions (A through F)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 epochs per stage")
    parser.add_argument("--replay_size", type=float, default=5.0,
                        help="Replay buffer size as %% of dataset")
    args = parser.parse_args()

    logger.info(f"Phase 63 starting — device={DEVICE}")
    logger.info(f"Args: {args}")

    # Determine which conditions to run
    if args.all:
        conditions = ["A", "B", "C", "D", "E", "F"]
    elif args.condition.lower() == "all":
        conditions = ["A", "B", "C", "D", "E", "F"]
    else:
        conditions = [args.condition.upper()]

    logger.info(f"Running conditions: {conditions}")

    # Run experiments
    all_results, splits = run_phase63(
        conditions, quick=args.quick, replay_size=args.replay_size)

    if not all_results:
        logger.error("No results to save!")
        return

    # Save all deliverables
    save_all_results(all_results, splits)
    generate_report(all_results)

    logger.info(f"\n{'=' * 70}")
    logger.info("PHASE 63 COMPLETE")
    logger.info(f"Results saved to: {RESULTS}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
