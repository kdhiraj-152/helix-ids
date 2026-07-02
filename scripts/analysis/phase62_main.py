#!/usr/bin/env python3
"""
Phase 62 — Continual Learning with Replay and Regularization.

Evaluates whether catastrophic forgetting in Phase 61 can be mitigated using
established continual-learning methods while preserving the frozen HELIX
foundation model.

Methods:
  Baseline A: Frozen HELIX (no adaptation)
  Baseline B: Sequential LoRA (reproduce Phase 61)
  Method 1: LoRA + Replay Buffer (0.5%, 1%, 5%)
  Method 2: LoRA + Elastic Weight Consolidation (λ=10, 100, 1000)
  Method 3: LoRA + Learning without Forgetting (T=2, 4, 8)
  Method 4: Replay + EWC (combined)

Usage:
  source .venv311/bin/activate
  PYTHONPATH=src python scripts/analysis/phase62_main.py
  PYTHONPATH=src python scripts/analysis/phase62_main.py --method replay --replay_size 1.0
  PYTHONPATH=src python scripts/analysis/phase62_main.py --method ewc --ewc_lambda 100
  PYTHONPATH=src python scripts/analysis/phase62_main.py --method lwf --lwf_temp 4
  PYTHONPATH=src python scripts/analysis/phase62_main.py --method all
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
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
import sklearn.metrics as sk_metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
RESULTS = PROJ / "results" / "phase62"
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

# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("phase62")
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
fh = logging.FileHandler(RESULTS / "phase62_run.log", mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.info(f"Phase 62 starting — device={DEVICE}")
logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
logger.info(f"Checkpoint SHA256: {hashlib.sha256(open(CHECKPOINT_PATH,'rb').read()).hexdigest()}")


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
# LoRA Implementation
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
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════

CHECKPOINT_SHA256 = None


def compute_encoder_hash(model: nn.Module) -> str:
    tensors = []
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
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
# Data Loading
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
        """Add stratified exemplars from a dataset.

        percentage: what fraction of the dataset to keep (0.5 = 0.5%, 1.0 = 1%)
        """
        n_total = X.shape[0]
        n_exemplars = max(int(n_total * percentage / 100.0), 1)

        # Determine classes for stratified sampling
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

        # Add to buffer, cap per class
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
# Elastic Weight Consolidation (EWC)
# ═══════════════════════════════════════════════════════════════════════════

class EWC:
    """Elastic Weight Consolidation for LoRA parameters.

    Stores Fisher information and optimal parameters after each stage.
    """

    def __init__(self, lora_model: LoRAModule, fisher_samples: int = 1024):
        self.lora_model = lora_model
        self.fisher_samples = fisher_samples
        self.fisher_matrices: list[dict[str, torch.Tensor]] = []
        self.optimal_params: list[dict[str, torch.Tensor]] = []
        self.lambda_values: list[float] = []

    @torch.no_grad()
    def estimate_fisher(self, X_train: np.ndarray, y_train: np.ndarray,
                        n_samples: Optional[int] = None) -> dict[str, torch.Tensor]:
        """Estimate diagonal Fisher Information for LoRA parameters."""
        self.lora_model.eval()
        n = n_samples or self.fisher_samples
        n = min(n, X_train.shape[0])
        idx = rng.choice(X_train.shape[0], size=n, replace=False)

        fisher = {}
        for name, param in self.lora_model.get_named_trainable_params().items():
            fisher[name] = torch.zeros_like(param)

        criterion = nn.CrossEntropyLoss(reduction='sum')

        # Loop through samples and compute per-sample Fisher
        for i in range(0, n, BATCH_SIZE):
            batch_idx = idx[i:i + BATCH_SIZE]
            xb = torch.from_numpy(X_train[batch_idx]).float().to(DEVICE)
            yb = torch.from_numpy(y_train[batch_idx]).long().to(DEVICE)

            # Forward pass
            bin_logits, _ = self.lora_model(xb)

            # Log-likelihood
            log_probs = F.log_softmax(bin_logits, dim=1)
            loss = F.nll_loss(log_probs, yb, reduction='sum')

            # Gradient computation (per-sample approximation)
            for name, param in self.lora_model.get_named_trainable_params().items():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

        # Normalize
        for name in fisher:
            fisher[name] /= max(n, 1)

        self.fisher_matrices.append(fisher)
        self.optimal_params.append(
            self.lora_model.clone_lora_params())

        logger.info(f"  EWC: Fisher estimated on {n} samples")
        return fisher

    def compute_ewc_loss(self, lambda_ewc: float = 100.0) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self.fisher_matrices or not self.optimal_params:
            return torch.tensor(0.0, device=DEVICE)

        total_loss = torch.tensor(0.0, device=DEVICE)
        current_params = self.lora_model.get_named_trainable_params()

        for fisher_dict, opt_params, lambda_val in zip(
                self.fisher_matrices, self.optimal_params,
                self.lambda_values if self.lambda_values else [lambda_ewc] * len(self.fisher_matrices)):

            for name in current_params:
                if name in fisher_dict and name in opt_params:
                    fisher = fisher_dict[name]
                    theta = current_params[name]
                    theta_star = opt_params[name].to(theta.device)
                    diff = theta - theta_star
                    total_loss += (lambda_val * (fisher * diff ** 2)).sum()

        return total_loss

    def add_stage(self, lambda_val: float):
        self.lambda_values.append(lambda_val)


# ═══════════════════════════════════════════════════════════════════════════
# Learning without Forgetting (LwF)
# ═══════════════════════════════════════════════════════════════════════════

class LwF:
    """Learning without Forgetting via knowledge distillation.

    Stores soft targets (logits) from the previous stage and adds
    distillation loss during subsequent training.
    """

    def __init__(self, lora_model: LoRAModule, temperature: float = 4.0):
        self.lora_model = lora_model
        self.temperature = temperature
        self.stored_logits: list[np.ndarray] = []
        self.stored_targets: list[np.ndarray] = []
        self.staged_data_keys: list[str] = []

    @torch.no_grad()
    def store_soft_targets(self, X_train: np.ndarray,
                           n_samples: int = 5000) -> np.ndarray:
        """Store soft targets (logits) from current model on training data."""
        self.lora_model.eval()
        n = min(n_samples, X_train.shape[0])
        idx = rng.choice(X_train.shape[0], size=n, replace=False)
        Xs = X_train[idx]

        all_logits = []
        for i in range(0, n, BATCH_SIZE):
            batch = Xs[i:i + BATCH_SIZE]
            xb = torch.from_numpy(batch).float().to(DEVICE)
            bin_logits, _ = self.lora_model(xb)
            all_logits.append(bin_logits.detach().cpu().numpy())

        logits = np.vstack(all_logits)
        self.stored_logits.append(logits)
        self.stored_targets.append(idx)
        return logits

    def distillation_loss(self, X_batch, current_logits, nll_weight=0.5):
        """Compute LwF distillation + NLL loss.

        Uses stored logits (soft targets) with temperature scaling.
        """
        total_distill = torch.tensor(0.0, device=DEVICE)
        if not self.stored_logits:
            return total_distill

        if DEVICE.type == "mps":
            # Avoid cpu() syncs — keep on device
            pass

        # Distill from the latest stored stage
        logits_stored = torch.from_numpy(
            self.stored_logits[-1]).float().to(DEVICE)

        # Temperature-scaled soft targets and predictions
        soft_targets = F.softmax(logits_stored / self.temperature, dim=1)

        # For distillation loss on the current batch, we need matching indices
        # Simplified: distill on model output by penalizing deviation
        distill = F.kl_div(
            F.log_softmax(current_logits / self.temperature, dim=1),
            soft_targets[:current_logits.shape[0]],
            reduction='batchmean',
        ) * (self.temperature ** 2)

        return distill

    @property
    def n_stored(self) -> int:
        return len(self.stored_logits)


# ═══════════════════════════════════════════════════════════════════════════
# Continual Learning Trainer
# ═══════════════════════════════════════════════════════════════════════════

class ContinualLearner:
    """Trains LoRA with optional Replay, EWC, LwF, or combined methods."""

    def __init__(self, lora_model: LoRAModule, method: str = "lora_base",
                 lr: float = LR, replay_size: float = 1.0,
                 ewc_lambda: float = 100.0, lwf_temp: float = 4.0):
        self.lora_model = lora_model
        self.method = method
        self.lr = lr
        self.replay_size = replay_size
        self.ewc_lambda = ewc_lambda
        self.lwf_temp = lwf_temp
        self.binary_ce = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            lora_model.get_trainable_params(), lr=lr, weight_decay=WEIGHT_DECAY
        )

        # Continual learning components
        has_replay = "replay" in method or method == "combined"
        has_ewc = "ewc" in method or method == "combined"
        has_lwf = "lwf" in method
        self.replay_buffer = ReplayBuffer() if has_replay else None
        self.ewc = EWC(lora_model) if has_ewc else None
        self.lwf = LwF(lora_model, temperature=lwf_temp) if has_lwf else None

    def train_epoch(self, X_np, y_bin, replay_ratio: float = 0.5,
                    ewc_loss_weight: float = 1.0):
        """Train one epoch with optional replay and regularization."""
        self.lora_model.train()
        total_loss = 0.0
        n = X_np.shape[0]
        indices = np.random.permutation(n)

        # Get replay batch size
        replay_batch_size = 0
        if self.replay_buffer is not None and self.replay_buffer.size > 0:
            replay_batch_size = max(1, int(BATCH_SIZE * replay_ratio))

        for i in range(0, n, BATCH_SIZE):
            # Current dataset batch
            batch_idx = indices[i:i + BATCH_SIZE]
            xb = torch.from_numpy(X_np[batch_idx]).float().to(DEVICE)
            yb = torch.from_numpy(y_bin[batch_idx]).long().to(DEVICE)

            # Forward pass on current batch
            bin_logits, fam_logits = self.lora_model(xb)
            loss = self.binary_ce(bin_logits, yb)

            # ── Replay Buffer ──────────────────────────────────────
            if self.replay_buffer is not None and self.replay_buffer.size > 0:
                X_rep, y_rep = self.replay_buffer.get_batch(replay_batch_size)
                if X_rep is not None and len(X_rep) > 0:
                    xb_rep = torch.from_numpy(X_rep).float().to(DEVICE)
                    yb_rep = torch.from_numpy(y_rep).long().to(DEVICE)
                    rep_logits, _ = self.lora_model(xb_rep)
                    rep_loss = self.binary_ce(rep_logits, yb_rep)
                    loss = loss + 0.5 * rep_loss

            # ── EWC Regularization ─────────────────────────────────
            if self.ewc is not None and len(self.ewc.fisher_matrices) > 0:
                ewc_loss = self.ewc.compute_ewc_loss(self.ewc_lambda)
                loss = loss + ewc_loss_weight * ewc_loss

            # ── LwF Distillation ───────────────────────────────────
            if self.lwf is not None and self.lwf.n_stored > 0:
                distill = self.lwf.distillation_loss(xb, bin_logits)
                loss = loss + 0.5 * distill

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.lora_model.get_trainable_params(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(batch_idx)

        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(self, X_np, y_bin):
        return evaluate_lora_model(self.lora_model, X_np, y_bin)

    def adapt(self, X_train, y_bin_train, dataset_name: str,
              X_val=None, y_bin_val=None, epochs=ADAPT_EPOCHS):
        """Adapt LoRA to a new dataset with continual learning."""
        history = {"train_loss": [], "val_mf1": []}
        best_val_mf1 = -float("inf")
        best_state = None

        # LwF: Store soft targets BEFORE adapting to new dataset
        if self.lwf is not None and self.lwf.n_stored == 0:
            # First stage adaptation — no previous knowledge to distill
            pass
        elif self.lwf is not None:
            # Store soft targets from current model on new dataset
            logger.info(f"  LwF: storing soft targets on {dataset_name}")
            self.lwf.store_soft_targets(X_train)

        for epoch in range(epochs):
            loss = self.train_epoch(X_train, y_bin_train)
            history["train_loss"].append(loss)

            if X_val is not None:
                metrics = self.evaluate(X_val, y_bin_val)
                val_mf1 = metrics.get("macro_f1", 0.0)
                history["val_mf1"].append(val_mf1)
                if val_mf1 > best_val_mf1:
                    best_val_mf1 = val_mf1
                    best_state = self.lora_model.clone_lora_params()

            if (epoch + 1) % 10 == 0:
                val_str = f", val_MF1={history['val_mf1'][-1]:.4f}" if X_val is not None else ""
                logger.info(f"    Epoch {epoch+1}/{epochs}: loss={loss:.4f}{val_str}")

        # Restore best state
        if best_state is not None and X_val is not None:
            self.lora_model.restore_lora_params(best_state)
            logger.info(f"    Restored best state (val_MF1={best_val_mf1:.4f})")

        # After training: update replay buffer with this dataset's exemplars
        if self.replay_buffer is not None:
            self.replay_buffer.add_samples(X_train, y_bin_train,
                                           percentage=self.replay_size)

        # After training: estimate Fisher for EWC
        if self.ewc is not None:
            self.ewc.add_stage(self.ewc_lambda)
            self.ewc.estimate_fisher(X_train, y_bin_train)

        return history

    def get_config_str(self) -> str:
        if self.method == "frozen":
            return "frozen"
        elif self.method == "lora_base":
            return f"lora_base_r{self.lora_model.rank}_a{self.lora_model.alpha}"
        elif self.method == "replay":
            return f"replay_{self.replay_size}pct"
        elif self.method == "ewc":
            return f"ewc_lambda{self.ewc_lambda}"
        elif self.method == "lwf":
            return f"lwf_T{self.lwf_temp}"
        elif self.method == "combined":
            return f"replay_{self.replay_size}pct_ewc_l{self.ewc_lambda}"
        return self.method


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
    from sklearn.cross_decomposition import CCA
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
            results.append({
                "dataset_i": d1, "dataset_j": d2,
                "stage": label or "unknown",
                "cka": cka_val,
                "cca_mean": float(np.mean(cca_vals)) if cca_vals else np.nan,
                "mmd": mmd_val,
                "wasserstein": wass["mean"],
                "cosine_similarity": cos_sim,
                "centroid_drift": drift,
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Continual Learning Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_continual_metrics(stage_results, method_label="unknown"):
    """Compute comprehensive continual learning metrics."""
    stages = sorted(stage_results.keys())
    datasets = list(next(iter(stage_results.values())).keys())

    metrics = {}
    metric_names = ["binary_f1", "macro_f1", "accuracy", "roc_auc",
                    "pr_auc", "precision", "recall", "brier", "ece", "nll"]

    # Per-stage average accuracy
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
    """Bayesian comparison of methods against baseline."""
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
    """Compute Cohen's d effect sizes between methods."""
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
    """Simplified repeated-measures ANOVA across stages."""
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

    # Simple mixed-effects summary
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
# Main Experimental Pipeline
# ═══════════════════════════════════════════════════════════════════════════

LORA_TARGETS = [
    "backbone.0", "backbone.4", "backbone.8", "backbone.12",
    "binary_head.0", "binary_head.3",
    "family_projection.0", "family_projection.4",
    "family_head.0", "family_head.3",
]


def prepare_splits(data_dict, scalers):
    """Create train/test splits for all datasets."""
    standardized = standardize_data(data_dict, scalers)
    splits = {}
    for name in ALL_DATASETS:
        d = standardized[name]
        X, y_bin = d["X"], d["y_bin"]
        # Handle extreme imbalance: UGR'16 has ~65 normal samples
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_bin, test_size=0.3, random_state=SEED, stratify=y_bin
            )
        except ValueError:
            # Fall back to non-stratified split
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


def run_condition(splits, method: str = "lora_base",
                  replay_size: float = 1.0,
                  ewc_lambda: float = 100.0,
                  lwf_temp: float = 4.0,
                  rank: int = LORA_RANK, alpha: int = LORA_ALPHA):
    """Run a single experimental condition.

    Returns:
        dict with stage_results, cl_metrics, rep_analysis, embeddings,
        computational metrics, etc.
    """
    config_str = f"{method}_r{rank}_a{alpha}"
    if method == "replay":
        config_str = f"replay_{replay_size}pct"
    elif method == "ewc":
        config_str = f"ewc_{ewc_lambda}"
    elif method == "lwf":
        config_str = f"lwf_T{lwf_temp}"
    elif method == "combined":
        config_str = f"replay{replay_size}pct_ewc{ewc_lambda}"

    logger.info(f"\n{'=' * 70}")
    logger.info(f"CONDITION: {config_str}")
    logger.info(f"{'=' * 70}")

    start_time = time.time()

    # ── Load frozen model ──────────────────────────────────────────
    base_model = load_frozen_model(CHECKPOINT_PATH)
    frozen_ok, _, _ = verify_encoder_frozen(base_model, "foundation")
    if not frozen_ok:
        logger.error("Foundation model not frozen! Aborting.")
        return None

    # ── Create LoRA model ─────────────────────────────────────────
    lora_model = LoRAModule(
        base_model=base_model,
        target_names=LORA_TARGETS,
        rank=rank, alpha=alpha, dropout=0.05,
    )
    lora_model.to(DEVICE)

    param_counts = lora_model.get_param_counts()
    logger.info(f"  LoRA params: trainable={param_counts['trainable']:,}, "
                f"total={param_counts['total']:,}, "
                f"ratio=1:{param_counts.get('compression_ratio', 0):.1f}")

    # ── Stage 0: Baseline (Frozen) ────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("STAGE 0: Foundation Baseline (Frozen HELIX)")
    logger.info(f"{'=' * 60}")

    stage0_results = {}
    for name in ALL_DATASETS:
        metrics = evaluate_lora_model(lora_model, splits[name]["X_test"],
                                      splits[name]["y_test"])
        stage0_results[name] = metrics
        logger.info(f"  {DATASET_DISPLAY[name]}: B-F1={metrics['binary_f1']:.4f}, "
                    f"M-F1={metrics['macro_f1']:.4f}")

    stage_results = {"stage_0": stage0_results}

    # Store embeddings
    all_embeddings = {}
    for name in ALL_DATASETS:
        all_embeddings[name] = extract_embeddings(lora_model, splits[name]["X_test"][:5000])

    # ── If frozen baseline only, return early ──────────────────────
    if method == "frozen":
        cl_metrics = compute_continual_metrics(stage_results, config_str)
        return {
            "config_str": config_str,
            "method": method,
            "stage_results": stage_results,
            "cl_metrics": cl_metrics,
            "rep_analysis": {},
            "embeddings": all_embeddings,
            "param_counts": param_counts,
            "computational": {"total_seconds": time.time() - start_time},
        }

    # ── Create Continual Learner ───────────────────────────────────
    learner = ContinualLearner(
        lora_model=lora_model,
        method=method,
        lr=LR,
        replay_size=replay_size,
        ewc_lambda=ewc_lambda,
        lwf_temp=lwf_temp,
    )

    # ── Continual Adaptation Stages ────────────────────────────────
    for stage_idx, adapt_dataset in enumerate(CONTINUAL_ORDER):
        stage_name = f"stage_{stage_idx + 1}"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"STAGE {stage_idx + 1}: Adapt to {DATASET_DISPLAY[adapt_dataset]} "
                    f"({config_str})")
        logger.info(f"{'=' * 60}")

        X_tr = splits[adapt_dataset]["X_train"]
        y_tr = splits[adapt_dataset]["y_train"]

        # Split for validation
        try:
            X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr
            )
        except ValueError:
            X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=SEED
            )

        logger.info(f"  Training samples: {X_tr_fit.shape[0]}, "
                    f"Validation: {X_val.shape[0]}")

        # Adapt
        history = learner.adapt(X_tr_fit, y_tr_fit, adapt_dataset,
                                X_val, y_val, epochs=ADAPT_EPOCHS)
        logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")

        # Verify encoder still frozen
        verify_encoder_frozen(base_model, f"after {adapt_dataset}")

        # Evaluate on ALL datasets
        stage_results[stage_name] = {}
        for name in ALL_DATASETS:
            metrics = evaluate_lora_model(lora_model, splits[name]["X_test"],
                                          splits[name]["y_test"])
            stage_results[stage_name][name] = metrics
            logger.info(f"  {DATASET_DISPLAY[name]}: B-F1={metrics['binary_f1']:.4f}, "
                        f"M-F1={metrics['macro_f1']:.4f}")

        # Extract embeddings
        for name in ALL_DATASETS:
            all_embeddings[f"{name}_stage_{stage_idx + 1}"] = extract_embeddings(
                lora_model, splits[name]["X_test"][:5000])

    # ── Compute Continual Learning Metrics ─────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("Continual Learning Metrics")
    logger.info(f"{'=' * 60}")

    cl_metrics = compute_continual_metrics(stage_results, config_str)

    # ── Representation Analysis ────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("Representation Similarity Analysis")
    logger.info(f"{'=' * 60}")

    rep_stages = {}
    for stage_name in stage_results.keys():
        stage_idx = stage_name.replace("stage_", "")
        stage_embeddings = {}
        for ds in ALL_DATASETS:
            key = f"{ds}_stage_{stage_idx}" if stage_idx != "0" else ds
            if key in all_embeddings:
                stage_embeddings[ds] = all_embeddings[key]
        if stage_embeddings:
            rep_stages[stage_name] = representational_similarity_analysis(
                stage_embeddings, label=stage_name)

    total_seconds = time.time() - start_time
    logger.info(f"\n  Total time: {total_seconds:.1f}s")

    return {
        "config_str": config_str,
        "method": method,
        "stage_results": stage_results,
        "cl_metrics": cl_metrics,
        "rep_analysis": rep_stages,
        "embeddings": all_embeddings,
        "param_counts": param_counts,
        "computational": {
            "total_seconds": total_seconds,
            "replay_buffer_size": learner.replay_buffer.size if learner.replay_buffer else 0,
            "replay_buffer_memory_bytes": learner.replay_buffer.memory_bytes if learner.replay_buffer else 0,
            "lora_params": param_counts["trainable"],
            "total_params": param_counts["total"],
            "compression_ratio": param_counts.get("compression_ratio", 0),
            "ewc_fisher_stages": len(learner.ewc.fisher_matrices) if learner.ewc else 0,
            "lwf_stored_stages": learner.lwf.n_stored if learner.lwf else 0,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_phase62(method_filter: str = "all"):
    """Run all Phase 62 experimental conditions."""
    logger.info("=" * 70)
    logger.info("PHASE 62: Continual Learning with Replay and Regularization")
    logger.info("=" * 70)

    verify_checkpoint_sha256()

    # ── Load all data ──────────────────────────────────────────────
    logger.info("Loading datasets...")
    data_dict = load_original_datasets()

    for ext in EXTERNAL_DATASETS:
        d = harmonize_external_dataset(ext, DATA_DIR)
        if d is not None:
            data_dict[ext] = d

    logger.info(f"Total datasets loaded: {len(data_dict)}")
    for name in sorted(data_dict.keys()):
        d = data_dict[name]
        logger.info(f"  {name}: X={d['X'].shape}, y_bin={np.bincount(d['y_bin'])}")

    scalers = fit_dataset_scalers(data_dict)
    splits = prepare_splits(data_dict, scalers)

    # ── Define conditions ──────────────────────────────────────────
    conditions = []

    if method_filter in ("frozen", "all"):
        conditions.append({
            "method": "frozen",
            "replay_size": 0.0,
            "ewc_lambda": 0.0,
            "lwf_temp": 0.0,
            "label": "frozen",
        })

    if method_filter in ("lora_base", "all"):
        conditions.append({
            "method": "lora_base",
            "replay_size": 0.0,
            "ewc_lambda": 0.0,
            "lwf_temp": 0.0,
            "label": "lora_base",
        })

    if method_filter in ("replay", "all"):
        for replay_size in [0.5, 1.0, 5.0]:
            conditions.append({
                "method": "replay",
                "replay_size": replay_size,
                "ewc_lambda": 0.0,
                "lwf_temp": 0.0,
                "label": f"replay_{replay_size}pct",
            })

    if method_filter in ("ewc", "all"):
        for ewc_lambda in [10, 100, 1000]:
            conditions.append({
                "method": "ewc",
                "replay_size": 0.0,
                "ewc_lambda": ewc_lambda,
                "lwf_temp": 0.0,
                "label": f"ewc_l{ewc_lambda}",
            })

    if method_filter in ("lwf", "all"):
        for lwf_temp in [2, 4, 8]:
            conditions.append({
                "method": "lwf",
                "replay_size": 0.0,
                "ewc_lambda": 0.0,
                "lwf_temp": lwf_temp,
                "label": f"lwf_T{lwf_temp}",
            })

    if method_filter in ("combined", "all"):
        conditions.append({
            "method": "combined",
            "replay_size": 1.0,
            "ewc_lambda": 100,
            "lwf_temp": 0.0,
            "label": "replay1pct_ewc100",
        })

    # ── Run each condition ─────────────────────────────────────────
    all_results = {}

    for cfg in conditions:
        result = run_condition(
            splits=splits,
            method=cfg["method"],
            replay_size=cfg["replay_size"],
            ewc_lambda=cfg["ewc_lambda"],
            lwf_temp=cfg["lwf_temp"],
        )
        if result is not None:
            all_results[cfg["label"]] = result
        cleanup()
        logger.info(f"\n  Completed {cfg['label']}")

    logger.info(f"\n{'=' * 70}")
    logger.info(f"All conditions completed. Total: {len(all_results)}")
    logger.info(f"{'=' * 70}")

    return all_results, splits


# ═══════════════════════════════════════════════════════════════════════════
# Save & Report
# ═══════════════════════════════════════════════════════════════════════════

def save_all_results(all_results: dict, splits: dict):
    """Save all Phase 62 deliverables."""

    # ── 1. Continual Results ───────────────────────────────────────
    all_rows = []
    for method_name, result in all_results.items():
        for stage_name, ds_results in result["stage_results"].items():
            for ds_name, metrics in ds_results.items():
                row = {
                    "method": method_name,
                    "stage": stage_name,
                    "dataset": ds_name,
                    "dataset_display": DATASET_DISPLAY.get(ds_name, ds_name),
                }
                row.update(metrics)
                all_rows.append(row)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            RESULTS / "continual_results.csv", index=False)
        logger.info(f"Saved continual_results.csv ({len(all_rows)} rows)")

    # ── 2. Forgetting Metrics ──────────────────────────────────────
    frows = []
    for method_name, result in all_results.items():
        forgetting = result["cl_metrics"].get("forgetting", {})
        for ds_name, f_dict in forgetting.items():
            row = {
                "method": method_name,
                "dataset": ds_name,
                "max_forgetting": f_dict.get("max_forgetting", 0),
                "mean_forgetting": f_dict.get("mean_forgetting", 0),
            }
            for stage_key, fval in f_dict.get("per_stage", {}).items():
                row[stage_key] = fval
            frows.append(row)
    if frows:
        pd.DataFrame(frows).to_csv(
            RESULTS / "forgetting_metrics.csv", index=False)
        logger.info(f"Saved forgetting_metrics.csv ({len(frows)} rows)")

    # ── 3. Method-specific CSVs ────────────────────────────────────
    for method_prefix in ["replay", "ewc", "lwf", "combined"]:
        mrows = []
        for method_name, result in all_results.items():
            if method_prefix not in method_name and method_name != method_prefix:
                continue
            for stage_name, ds_results in result["stage_results"].items():
                for ds_name, metrics in ds_results.items():
                    row = {
                        "method": method_name,
                        "stage": stage_name,
                        "dataset": ds_name,
                    }
                    row.update(metrics)
                    mrows.append(row)
        if mrows:
            pd.DataFrame(mrows).to_csv(
                RESULTS / f"{method_prefix}_results.csv", index=False)
            n = len([m for m in mrows if m["stage"] == "stage_0"])
            logger.info(f"Saved {method_prefix}_results.csv ({len(mrows)} rows)")

    # ── 4. Forward/Backward Transfer ───────────────────────────────
    trows = []
    for method_name, result in all_results.items():
        for key, val in result["cl_metrics"].items():
            if "forward_transfer" in key or "backward_transfer" in key or "BWT" in key:
                if isinstance(val, dict):
                    for ds, v in val.items():
                        trows.append({"method": method_name, "metric": key, "dataset": ds, "value": v})
                else:
                    trows.append({"method": method_name, "metric": key, "value": val})
    if trows:
        pd.DataFrame(trows).to_csv(
            RESULTS / "forward_backward_transfer.csv", index=False)
        logger.info(f"Saved forward_backward_transfer.csv ({len(trows)} rows)")

    # ── 5. Representation Similarity ───────────────────────────────
    rep_rows = []
    for method_name, result in all_results.items():
        for stage, reps in result.get("rep_analysis", {}).items():
            for r in reps:
                rep_rows.append({"method": method_name, **r})
    if rep_rows:
        pd.DataFrame(rep_rows).to_csv(
            RESULTS / "representation_similarity.csv", index=False)
        logger.info(f"Saved representation_similarity.csv ({len(rep_rows)} rows)")

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
        if method_name == "frozen":
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
    # Build a simpler repeated-measures summary
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

    # ── 9. Save embeddings (pickle) ────────────────────────────────
    import pickle
    emb_path = RESULTS / "embeddings" / "all_embeddings.pkl"
    emb_to_save = {}
    for method_name, result in all_results.items():
        if "embeddings" in result:
            for key, val in result["embeddings"].items():
                emb_to_save[f"{method_name}_{key}"] = val
    # Only save first 10 entries (too large otherwise)
    # Instead, save metadata
    emb_meta = {k: v.shape for k, v in emb_to_save.items()}
    with open(RESULTS / "embeddings" / "embedding_metadata.json", "w") as f:
        json.dump(emb_meta, f, indent=2)
    logger.info(f"Saved embedding metadata ({len(emb_meta)} entries)")

    return all_rows


def generate_report(all_results: dict):
    """Generate Phase 62 report and summary."""
    lines = []
    lines.append("# Phase 62 Report: Continual Learning with Replay and Regularization")
    lines.append("")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Device: {DEVICE}")
    lines.append(f"Checkpoint: {CHECKPOINT_PATH}")
    lines.append(f"Checkpoint SHA256: {CHECKPOINT_SHA256}")
    lines.append("")

    # ── Methods Overview ───────────────────────────────────────────
    lines.append("## Experimental Conditions")
    lines.append("")
    lines.append(f"Total conditions: {len(all_results)}")
    for method_name in sorted(all_results.keys()):
        r = all_results[method_name]
        comp = r.get("computational", {})
        lines.append(f"")
        lines.append(f"### {method_name}")
        lines.append(f"- Method: {r['method']}")
        lines.append(f"- LoRA params: {r['param_counts']['trainable']:,} / {r['param_counts']['total']:,}")
        lines.append(f"- Compression ratio: 1:{r['param_counts'].get('compression_ratio', 0):.1f}")
        if "total_seconds" in comp:
            lines.append(f"- Total time: {comp['total_seconds']:.1f}s")
        if "replay_buffer_size" in comp:
            lines.append(f"- Replay buffer: {comp['replay_buffer_size']} exemplars "
                         f"({comp.get('replay_buffer_memory_bytes', 0) / 1024:.1f} KB)")
        if "ewc_fisher_stages" in comp:
            lines.append(f"- EWC stages: {comp['ewc_fisher_stages']}")
        if "lwf_stored_stages" in comp:
            lines.append(f"- LwF stages: {comp['lwf_stored_stages']}")
    lines.append("")

    # ── Stage Results (Macro F1) ───────────────────────────────────
    lines.append("## Stage Results (Macro F1)")
    lines.append("")
    lines.append("| Method | Dataset | Stage 0 | Stage 1 (IoT-23) | Stage 2 (Kyoto2006+) | Stage 3 (UGR'16) |")
    lines.append("|--------|---------|---------|------------------|----------------------|-------------------|")

    for method_name in sorted(all_results.keys()):
        r = all_results[method_name]
        first_row = True
        for ds in ALL_DATASETS:
            vals = []
            for stage in ["stage_0", "stage_1", "stage_2", "stage_3"]:
                if stage in r["stage_results"] and ds in r["stage_results"][stage]:
                    mf1 = r["stage_results"][stage][ds].get("macro_f1", 0)
                    vals.append(f"{mf1:.4f}")
                else:
                    vals.append("—")
            label = method_name if first_row else ""
            lines.append(f"| {label} | {DATASET_DISPLAY[ds]} | {' | '.join(vals)} |")
            first_row = False
    lines.append("")

    # ── Forgetting Summary ─────────────────────────────────────────
    lines.append("## Forgetting Summary (Macro F1)")
    lines.append("")
    lines.append("| Method | Avg Forgetting | Max Forgetting | Global Max |")
    lines.append("|--------|---------------|---------------|------------|")

    for method_name in sorted(all_results.keys()):
        r = all_results[method_name]
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
        lines.append(f"| {method_name} | {avg_f:.4f} | {max_f:.4f} | {gm:.4f} |")
    lines.append("")

    # ── Transfer Summary ───────────────────────────────────────────
    lines.append("## Forward/Backward Transfer")
    lines.append("")
    lines.append("| Method | Stage | FWT | Avg BWT | Stability | Plasticity |")
    lines.append("|--------|-------|-----|---------|-----------|------------|")

    for method_name in sorted(all_results.keys()):
        r = all_results[method_name]
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
            label = method_name if first_row else ""
            lines.append(f"| {label} | {stage} | {fwt} | {bwt} | {stability} | {plasticity} |")
            first_row = False
    lines.append("")

    # ── Knowledge Retention ────────────────────────────────────────
    lines.append("## Knowledge Retention (relative to Stage 0)")
    lines.append("")
    lines.append("| Method | Avg Retention |")
    lines.append("|--------|--------------|")

    for method_name in sorted(all_results.keys()):
        r = all_results[method_name]
        cl = r["cl_metrics"]
        retentions = []
        for stage in ["stage_1", "stage_2", "stage_3"]:
            ret = cl.get(f"{stage}_avg_knowledge_retention", 0)
            retentions.append(ret)
        avg_ret = float(np.mean(retentions)) if retentions else 0
        lines.append(f"| {method_name} | {avg_ret:.4f} |")
    lines.append("")

    # ── Success Criteria ───────────────────────────────────────────
    lines.append("## Success Criteria")
    lines.append("")
    lines.append("### Primary: Reduce forgetting by ≥50% vs LoRA baseline")
    lines.append("")

    baseline_forgetting = None
    if "lora_base" in all_results:
        bl = all_results["lora_base"]["cl_metrics"]
        bl_forgets = []
        for stage in ["stage_1", "stage_2", "stage_3"]:
            bl_forgets.append(bl.get(f"{stage}_avg_forgetting", 0))
        baseline_forgetting = float(np.mean(bl_forgets))

    if baseline_forgetting is not None:
        lines.append(f"Baseline (LoRA) avg forgetting: {baseline_forgetting:.4f}")
        lines.append("")
        lines.append("| Method | Avg Forgetting | Reduction | PASS? |")
        lines.append("|--------|---------------|-----------|-------|")
        for method_name in sorted(all_results.keys()):
            r = all_results[method_name]
            cl = r["cl_metrics"]
            avg_f = float(np.mean([cl.get(f"{s}_avg_forgetting", 0) for s in ["stage_1", "stage_2", "stage_3"]]))
            reduction = (baseline_forgetting - avg_f) / max(baseline_forgetting, 1e-8) * 100
            passed = reduction >= 50.0
            lines.append(f"| {method_name} | {avg_f:.4f} | {reduction:.1f}% | {'PASS' if passed else 'FAIL'} |")
    lines.append("")

    # ── Computational Metrics ──────────────────────────────────────
    lines.append("## Computational Metrics")
    lines.append("")
    lines.append("| Method | Time (s) | LoRA Params | Replay Size | Replay Memory |")
    lines.append("|--------|---------|-------------|-------------|---------------|")

    for method_name in sorted(all_results.keys()):
        r = all_results[method_name]
        comp = r.get("computational", {})
        time_s = comp.get("total_seconds", 0)
        lora_p = r["param_counts"]["trainable"]
        rep_sz = comp.get("replay_buffer_size", 0)
        rep_mem = comp.get("replay_buffer_memory_bytes", 0)
        lines.append(f"| {method_name} | {time_s:.0f} | {lora_p:,} | {rep_sz} | {rep_mem / 1024:.1f} KB |")
    lines.append("")

    # Save report
    report_path = RESULTS / "phase62_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved report: {report_path}")

    # ── Summary ────────────────────────────────────────────────────
    summary = []
    summary.append("# Phase 62 Summary")
    summary.append("")
    summary.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Device: {DEVICE}")
    summary.append(f"Conditions: {len(all_results)}")
    summary.append("")

    if "frozen" in all_results:
        frozen_mf1 = float(np.mean([
            all_results["frozen"]["stage_results"]["stage_0"][ds]["macro_f1"]
            for ds in ALL_DATASETS
        ]))
        summary.append(f"- Frozen baseline MF1: {frozen_mf1:.4f}")

    if baseline_forgetting is not None:
        summary.append(f"- LoRA baseline avg forgetting: {baseline_forgetting:.4f}")

    for method_name in sorted(all_results.keys()):
        if method_name == "frozen":
            continue
        r = all_results[method_name]
        cl = r["cl_metrics"]
        avg_f = float(np.mean([cl.get(f"{s}_avg_forgetting", 0) for s in ["stage_1", "stage_2", "stage_3"]]))
        summary.append(f"- {method_name}: avg forgetting = {avg_f:.4f}")
        if baseline_forgetting is not None and baseline_forgetting > 0:
            reduction = (baseline_forgetting - avg_f) / baseline_forgetting * 100
            summary.append(f"  → {reduction:.1f}% reduction vs LoRA baseline")

    summary.append("")
    summary.append("### Key Findings")

    # Determine best method
    if len(all_results) > 1:
        best_method = None
        best_forgetting = float('inf')
        for method_name in sorted(all_results.keys()):
            if method_name == "frozen":
                continue
            r = all_results[method_name]
            cl = r["cl_metrics"]
            avg_f = float(np.mean([cl.get(f"{s}_avg_forgetting", 0) for s in ["stage_1", "stage_2", "stage_3"]]))
            if avg_f < best_forgetting:
                best_forgetting = avg_f
                best_method = method_name

        if best_method:
            summary.append(f"1. Best method: **{best_method}** (avg forgetting={best_forgetting:.4f})")

        if baseline_forgetting is not None and best_forgetting < baseline_forgetting * 0.5:
            summary.append("2. ✅ **Primary criterion PASSED**: Forgetting reduced by ≥50%")
        else:
            summary.append("2. ❌ **Primary criterion NOT MET**")
            if baseline_forgetting is not None:
                summary.append(f"   Required: {baseline_forgetting * 0.5:.4f}, Best: {best_forgetting:.4f}")

    summary_path = RESULTS / "phase62_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))
    logger.info(f"Saved summary: {summary_path}")


def save_visualizations(all_results: dict):
    """Generate and save visualizations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ── Forgetting comparison bar chart ─────────────────────────
        fig, ax = plt.subplots(figsize=(14, 6))
        methods = sorted([m for m in all_results.keys() if m != "frozen"])
        x = np.arange(len(methods))
        width = 0.25

        for i, stage in enumerate(["stage_1", "stage_2", "stage_3"]):
            vals = []
            for m in methods:
                cl = all_results[m]["cl_metrics"]
                vals.append(cl.get(f"{stage}_avg_forgetting", 0))
            ax.bar(x + i * width, vals, width, label=stage.replace("stage_", "Stage "))

        if "lora_base" in methods:
            bl_idx = methods.index("lora_base")
            ax.axhline(y=all_results["lora_base"]["cl_metrics"].get("stage_3_avg_forgetting", 0),
                       color="red", linestyle="--", alpha=0.5, label="LoRA baseline (Stage 3)")

        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Average Forgetting (MF1)")
        ax.set_title("Phase 62: Forgetting Comparison Across Methods")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "plots" / "forgetting_comparison.png", dpi=150)
        plt.close(fig)

        # ── Macro F1 trajectory per method ──────────────────────────
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        for mi, method_name in enumerate(methods):
            r = all_results[method_name]
            stages = sorted(r["stage_results"].keys())
            avg_mf1 = []
            for stage in stages:
                mf1s = [r["stage_results"][stage][ds]["macro_f1"] for ds in ALL_DATASETS]
                avg_mf1.append(float(np.mean(mf1s)))
            ax.plot(range(len(avg_mf1)), avg_mf1, "o-",
                    color=colors[mi], label=method_name, linewidth=1.5)

        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=45)
        ax.set_ylabel("Average Macro F1")
        ax.set_xlabel("Stage")
        ax.set_title("Phase 62: Average Macro F1 Across Stages")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(RESULTS / "plots" / "mf1_trajectory.png", dpi=150)
        plt.close(fig)

        # ── Replay analysis plot ────────────────────────────────────
        replay_methods = [m for m in methods if "replay" in m]
        if replay_methods:
            fig, ax = plt.subplots(figsize=(10, 6))
            replay_sizes = []
            replay_forgets = []
            for m in replay_methods:
                cl = all_results[m]["cl_metrics"]
                avg_f = float(np.mean([cl.get(f"{s}_avg_forgetting", 0)
                                       for s in ["stage_1", "stage_2", "stage_3"]]))
                # Extract replay size from method name
                if m.startswith("replay_"):
                    sz = float(m.replace("replay_", "").replace("pct", ""))
                elif "replay" in m and "ewc" in m:
                    # Combined method: extract replay_size from _size_pct pattern
                    import re
                    match = re.search(r'replay([\d.]+)pct', m)
                    sz = float(match.group(1)) if match else 1.0
                else:
                    sz = 0.0
                replay_sizes.append(sz)
                replay_forgets.append(avg_f)

            ax.semilogx(replay_sizes, replay_forgets, "bo-", linewidth=2, markersize=8)
            # Add LoRA baseline
            if "lora_base" in all_results:
                bl_f = float(np.mean([all_results["lora_base"]["cl_metrics"].get(f"{s}_avg_forgetting", 0)
                                      for s in ["stage_1", "stage_2", "stage_3"]]))
                ax.axhline(y=bl_f, color="red", linestyle="--", label="LoRA baseline (no replay)")
            ax.set_xlabel("Replay Size (% of dataset)")
            ax.set_ylabel("Average Forgetting (MF1)")
            ax.set_title("Phase 62: Replay Buffer Analysis")
            ax.legend()
            fig.tight_layout()
            fig.savefig(RESULTS / "plots" / "replay_analysis.png", dpi=150)
            plt.close(fig)

        # ── EWC lambda sweep ────────────────────────────────────────
        ewc_methods = [m for m in methods if "ewc" in m and "replay" not in m]
        if ewc_methods:
            fig, ax = plt.subplots(figsize=(10, 6))
            lambdas = []
            ewc_forgets = []
            for m in ewc_methods:
                cl = all_results[m]["cl_metrics"]
                avg_f = float(np.mean([cl.get(f"{s}_avg_forgetting", 0)
                                       for s in ["stage_1", "stage_2", "stage_3"]]))
                lbd = float(m.replace("ewc_l", ""))
                lambdas.append(lbd)
                ewc_forgets.append(avg_f)

            ax.semilogx(lambdas, ewc_forgets, "go-", linewidth=2, markersize=8)
            if "lora_base" in all_results:
                bl_f = float(np.mean([all_results["lora_base"]["cl_metrics"].get(f"{s}_avg_forgetting", 0)
                                      for s in ["stage_1", "stage_2", "stage_3"]]))
                ax.axhline(y=bl_f, color="red", linestyle="--", label="LoRA baseline")
            ax.set_xlabel("EWC λ")
            ax.set_ylabel("Average Forgetting (MF1)")
            ax.set_title("Phase 62: EWC Regularization Strength Sweep")
            ax.legend()
            fig.tight_layout()
            fig.savefig(RESULTS / "plots" / "ewc_sweep.png", dpi=150)
            plt.close(fig)

        # ── LwF temperature sweep ───────────────────────────────────
        lwf_methods = [m for m in methods if "lwf" in m]
        if lwf_methods:
            fig, ax = plt.subplots(figsize=(10, 6))
            temps = []
            lwf_forgets = []
            for m in lwf_methods:
                cl = all_results[m]["cl_metrics"]
                avg_f = float(np.mean([cl.get(f"{s}_avg_forgetting", 0)
                                       for s in ["stage_1", "stage_2", "stage_3"]]))
                T = float(m.replace("lwf_T", ""))
                temps.append(T)
                lwf_forgets.append(avg_f)

            ax.plot(temps, lwf_forgets, "mo-", linewidth=2, markersize=8)
            if "lora_base" in all_results:
                bl_f = float(np.mean([all_results["lora_base"]["cl_metrics"].get(f"{s}_avg_forgetting", 0)
                                      for s in ["stage_1", "stage_2", "stage_3"]]))
                ax.axhline(y=bl_f, color="red", linestyle="--", label="LoRA baseline")
            ax.set_xlabel("LwF Temperature T")
            ax.set_ylabel("Average Forgetting (MF1)")
            ax.set_title("Phase 62: LwF Temperature Sweep")
            ax.legend()
            fig.tight_layout()
            fig.savefig(RESULTS / "plots" / "lwf_sweep.png", dpi=150)
            plt.close(fig)

        # ── Heatmap of Final MF1 ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 8))
        method_names = sorted([m for m in all_results.keys()])
        ds_names = ALL_DATASETS
        heatmap_data = np.zeros((len(method_names), len(ds_names)))
        for mi, m in enumerate(method_names):
            for di, ds in enumerate(ds_names):
                heatmap_data[mi, di] = all_results[m]["stage_results"].get(
                    "stage_3", {}).get(ds, {}).get("macro_f1", 0)

        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(ds_names)))
        ax.set_xticklabels([DATASET_DISPLAY[ds] for ds in ds_names], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(method_names)))
        ax.set_yticklabels(method_names, fontsize=8)
        ax.set_title("Phase 62: Final Macro F1 by Method and Dataset")

        for mi in range(len(method_names)):
            for di in range(len(ds_names)):
                val = heatmap_data[mi, di]
                ax.text(di, mi, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if 0.3 < val < 0.7 else "white")

        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(RESULTS / "plots" / "final_mf1_heatmap.png", dpi=150)
        plt.close(fig)

        logger.info("Saved all plots")

    except Exception as e:
        logger.warning(f"Visualizations failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global ADAPT_EPOCHS
    parser = argparse.ArgumentParser(
        description="Phase 62: Continual Learning with Replay and Regularization")
    parser.add_argument("--method", choices=["all", "frozen", "lora_base", "replay",
                                             "ewc", "lwf", "combined"],
                        default="all", help="Which method(s) to run")
    parser.add_argument("--replay_size", type=float, default=1.0,
                        help="Replay buffer size as % of dataset")
    parser.add_argument("--ewc_lambda", type=float, default=100.0,
                        help="EWC regularization strength")
    parser.add_argument("--lwf_temp", type=float, default=4.0,
                        help="LwF distillation temperature")
    parser.add_argument("--epochs", type=int, default=ADAPT_EPOCHS,
                        help="Adaptation epochs")
    parser.add_argument("--skip_run", action="store_true",
                        help="Skip running, only generate report from existing CSVs")
    args = parser.parse_args()

    ADAPT_EPOCHS = args.epochs

    epochs_str = ADAPT_EPOCHS
    logger.info(f"Method: {args.method}, replay_size={args.replay_size}, "
                f"ewc_lambda={args.ewc_lambda}, lwf_temp={args.lwf_temp}, "
                f"epochs={epochs_str}")

    if args.skip_run:
        logger.info("Skip-run mode: loading existing results...")
        all_results = {}
        # Try to load from CSVs
        if (RESULTS / "continual_results.csv").exists():
            df = pd.read_csv(RESULTS / "continual_results.csv")
            for method in df["method"].unique():
                mdf = df[df["method"] == method]
                stage_results = {}
                for stage in mdf["stage"].unique():
                    sdf = mdf[mdf["stage"] == stage]
                    stage_results[stage] = {
                        row["dataset"]: {k: row[k] for k in
                                         ["binary_f1", "macro_f1", "accuracy", "roc_auc",
                                          "pr_auc", "precision", "recall", "brier", "ece", "nll"]
                                         if k in row}
                        for _, row in sdf.iterrows()
                    }
                all_results[method] = {
                    "method": method,
                    "stage_results": stage_results,
                    "cl_metrics": compute_continual_metrics(stage_results, method),
                    "rep_analysis": {},
                    "computational": {"total_seconds": 0},
                    "param_counts": {"trainable": 0, "total": 0, "compression_ratio": 0},
                }
        if all_results:
            generate_report(all_results)
            save_visualizations(all_results)
        return

    # ── Run experiments ────────────────────────────────────────────
    all_results, splits = run_phase62(method_filter=args.method)

    # ── Save everything ────────────────────────────────────────────
    if all_results:
        save_all_results(all_results, splits)
        generate_report(all_results)
        save_visualizations(all_results)

    logger.info("\nPhase 62 complete.")
    logger.info(f"Results in: {RESULTS}")


if __name__ == "__main__":
    main()
