# Baseline Model Suite v1.0

> **Phase 36 — Deliverable 6 of 8**
> Defines the mandatory baseline models every submission must report.
> Provides reference implementations and expected performance ranges.
> Date: 2026-06-24

---

## 1. Purpose

Every benchmark needs a fixed set of baselines to contextualize new results.
Without uniform baselines, researchers claim "state-of-the-art" against arbitrarily
chosen comparisons.

This document defines **7 mandatory baselines** spanning three complexity tiers.
Every submission to the Phase 36 benchmark MUST report results for all 7 baselines
alongside their proposed model.

---

## 2. Baseline Tiers

| Tier | Baselines | Purpose |
|------|-----------|---------|
| **Classical** | Logistic Regression, Random Forest, XGBoost | Upper bound for non-neural methods |
| **Neural** | MLP, Transformer IDS | Upper bound for deep learning IDS |
| **Domain Adaptation** | DANN, CORAL | Upper bound for transfer learning |

---

## 3. Classical Baselines

### 3.1 Logistic Regression

**Purpose:** Simplest possible baseline. If a neural method cannot beat logistic
regression, the problem's complexity does not warrant deep learning.

**Reference Implementation:**

```python
# File: baselines/phase36/logistic_regression.py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        random_state=42
    ))
])
```

**Hyperparameter Search Space:**

| Parameter | Values |
|-----------|--------|
| C | 0.01, 0.1, 1.0, 10.0, 100.0 |
| penalty | l2 (only) |
| solver | lbfgs, saga |
| max_iter | 1000, 5000 |

**Training:** 5-fold cross-validation on training split. Select best C by mean Macro F1.

**Expected Performance (In-Distribution, Run A):** MF1 0.82 — 0.88

### 3.2 Random Forest

**Purpose:** Non-linear ensemble method. Captures feature interactions and is
robust to outliers. Represents the best non-boosted tree ensemble.

**Reference Implementation:**

```python
# File: baselines/phase36/random_forest.py
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**Hyperparameter Search Space:**

| Parameter | Values |
|-----------|--------|
| n_estimators | 100, 200, 500 |
| max_depth | 10, 20, 30, None |
| min_samples_split | 2, 5, 10 |
| min_samples_leaf | 1, 2, 4 |
| max_features | sqrt, log2, None |

**Training:** Random search (100 iterations) on training split. 3-fold CV.
Select by mean Macro F1.

**Expected Performance (In-Distribution, Run A):** MF1 0.88 — 0.94

### 3.3 XGBoost

**Purpose:** Boosted tree ensemble. Often the best classical method for
tabular data. Represents the upper bound of gradient-boosted decision trees.

**Reference Implementation:**

```python
# File: baselines/phase36/xgboost.py
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    scale_pos_weight=None,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
```

**Hyperparameter Search Space:**

| Parameter | Values |
|-----------|--------|
| n_estimators | 100, 200, 500 |
| max_depth | 4, 6, 8, 12 |
| learning_rate | 0.01, 0.05, 0.1, 0.3 |
| subsample | 0.6, 0.8, 1.0 |
| colsample_bytree | 0.6, 0.8, 1.0 |
| reg_lambda | 0.0, 0.1, 1.0, 10.0 |
| reg_alpha | 0.0, 0.1, 1.0 |

**Training:** Bayesian optimization (50 iterations) on training split. 3-fold CV.
Early stopping at 50 rounds with validation set. Multi-class softmax objective.

**Expected Performance (In-Distribution, Run A):** MF1 0.89 — 0.95

---

## 4. Neural Baselines

### 4.1 MLP (Multi-Layer Perceptron)

**Purpose:** Simple feedforward neural network. Establishes the baseline for
deep learning IDS without domain adaptation.

**Reference Implementation:**

```python
# File: baselines/phase36/mlp.py
import torch
import torch.nn as nn

class MLPIDS(nn.Module):
    """4-layer MLP for IDS classification."""
    def __init__(self, input_dim=22, num_classes=7, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

**Training Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 (cosine annealing to 1e-5) |
| Batch size | 256 |
| Epochs | 100 (early stop patience 15) |
| Weight decay | 1e-4 |
| Loss | Cross-entropy with class weights |
| Class weights | Inverse frequency capped at 10.0 |

**Expected Performance (In-Distribution, Run A):** MF1 0.87 — 0.93

### 4.2 Transformer IDS

**Purpose:** Self-attention model for IDS. Captures long-range dependencies
between flow features and protocol interactions. Represents the state of the art
in deep learning IDS.

**Reference Implementation:**

```python
# File: baselines/phase36/transformer_ids.py
import torch
import torch.nn as nn
import math

class TransformerIDS(nn.Module):
    """Transformer encoder for flow-level IDS."""
    def __init__(self, input_dim=22, num_classes=7, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=512, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, input_dim) — single flow as sequence
        x = self.input_proj(x)  # (batch, 1, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        return self.classifier(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

**Training Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 (warmup 10%, cosine decay) |
| Batch size | 128 |
| Epochs | 150 (early stop patience 20) |
| Weight decay | 1e-4 |
| Dropout | 0.2 |
| Loss | Cross-entropy with label smoothing (ε=0.1) |
| Label smoothing | Reduces overfitting to majority classes |

**Expected Performance (In-Distribution, Run A):** MF1 0.88 — 0.95

---

## 5. Domain Adaptation Baselines

### 5.1 DANN (Domain-Adversarial Neural Network)

**Purpose:** Domain-adversarial training (Ganin et al., 2016). The canonical
unsupervised domain adaptation method. If DANN fails to transfer, the domains
are fundamentally incompatible.

**Reference Implementation:**

```python
# File: baselines/phase36/dann.py
import torch
import torch.nn as nn
import numpy as np

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DANNIDS(nn.Module):
    """Domain-Adversarial Neural Network for IDS transfer."""
    def __init__(self, input_dim=22, num_classes=7, num_domains=2):
        super().__init__()
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        labels = self.label_classifier(features)
        features_rev = GradientReversal.apply(features, alpha)
        domains = self.domain_classifier(features_rev)
        return labels, domains
```

**Training Procedure:**

1. Alternate source and target batches in 1:1 ratio
2. Gradient reversal coefficient α ramped from 0 → 1 over first 10 epochs
3. Loss = L_label + λ * L_domain (λ = 0.5)
4. Evaluate on target domain **without** target labels (unsupervised)

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD with momentum 0.9 |
| Learning rate | 1e-3 (no decay) |
| Batch size | 128 (64 source + 64 target) |
| Epochs | 100 |
| λ (domain loss weight) | 0.5 |
| α ramp | Linear 0→1 over epochs 0-10 |

**Expected Performance (Cross-Org, Run A → Run C):** MF1 0.70 — 0.82

### 5.2 CORAL (Correlation Alignment)

**Purpose:** Second-order feature alignment (Sun & Saenko, 2016). Aligns
covariance matrices of source and target feature distributions. Simpler than DANN
but often competitive.

**Reference Implementation:**

```python
# File: baselines/phase36/coral.py
import torch
import torch.nn as nn

class CORALIDS(nn.Module):
    """Deep CORAL for IDS — aligns feature covariances."""
    def __init__(self, input_dim=22, num_classes=7, backbone_dims=[256, 128]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in backbone_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def coral_loss(source_features, target_features):
    """Compute CORAL loss between source and target feature covariances."""
    d = source_features.size(1)
    source_cov = covariance(source_features)
    target_cov = covariance(target_features)
    return torch.sum((source_cov - target_cov) ** 2) / (4 * d ** 2)

def covariance(features):
    n = features.size(0)
    centered = features - features.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (n - 1)
    return cov
```

**Training Procedure:**

1. Train on source labeled data with standard cross-entropy
2. Augment loss with CORAL loss between source and target feature representations
3. Loss = L_CE + λ * L_CORAL (λ = 0.1)

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 (reduce on plateau) |
| Batch size | 128 |
| Epochs | 100 |
| λ (CORAL weight) | 0.01, 0.1, 1.0 (tune on val) |

**Expected Performance (Cross-Org, Run A → Run C):** MF1 0.68 — 0.80

---

## 6. Submission Requirements

### 6.1 Baseline Reproducibility

All baseline results MUST be produced using the **reference implementations**
provided in:

```
baselines/phase36/
├── logistic_regression.py
├── random_forest.py
├── xgboost.py
├── mlp.py
├── transformer_ids.py
├── dann.py
└── coral.py
```

Hyperparameter tuning is permitted but must use only the search spaces defined in
this document. Any deviation must be reported in the submission.

### 6.2 Computational Budget

All baselines should be runnable on a single consumer GPU (NVIDIA RTX 3080 or
equivalent, ≤ 16GB VRAM) within 48 hours total for all 7 baselines across all
5 evaluation regimes.

| Baseline | Approx. Runtime (RTX 3080) |
|----------|---------------------------|
| Logistic Regression | 5 minutes |
| Random Forest | 15 minutes |
| XGBoost | 30 minutes |
| MLP | 2 hours |
| Transformer IDS | 6 hours |
| DANN | 8 hours |
| CORAL | 4 hours |

### 6.3 Beyond Baselines

Submissions may include additional models beyond the 7 mandatory baselines.
When doing so, the novelty claim must be supported by statistically significant
improvement over the best baseline in the relevant regime.

---

## 7. Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-06-24 | Initial release — 7 baselines across 3 tiers |
