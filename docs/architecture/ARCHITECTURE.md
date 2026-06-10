# HELIX-IDS Architecture

## Scope

This document describes the current canonical architecture used by this repository.
It supersedes older descriptions based on binary-only or 32-feature pipelines.

## System Shape

HELIX-IDS is a multi-dataset intrusion detection pipeline with a shared feature space and a multi-task model.

- Datasets: NSL-KDD, UNSW-NB15, CICIDS-2018
- Unified input: 31 features total
- Model outputs:
  - Binary head: Normal vs Attack (2 classes)
  - Family head: attack-family taxonomy (7 classes)

## Data Architecture

The harmonization contract is implemented in `src/helix_ids/data/feature_harmonization.py`.

- Common features: 28
- Dataset-origin one-hot features: 3 (`is_nsl_kdd`, `is_unsw`, `is_cicids`)
- Total model input dimension: 31

The split/normalization contract is implemented in `src/helix_ids/data/multi_dataset_loader.py`.

- Per-dataset normalization is enforced to avoid cross-dataset leakage
- Stratified splitting is used where feasible
- Harmonized tensors are emitted for training/evaluation scripts

## Model Architecture

The canonical model is implemented in `src/helix_ids/models/helix_ids_full.py`.

- Shared MLP backbone (configurable hidden dimensions)
- Binary classification head (2 logits)
- Family classification head (7 logits)
- Multi-task optimization via `MultiTaskLoss`

Default config currently uses:

- Input dim: 31
- Hidden dims: `(512, 384, 256, 128)`
- Dropout: `(0.3, 0.3, 0.25, 0.2)`

## Training/Inference Topology

Primary operational entrypoints live in `scripts/`.

- Train: `scripts/train_helix_ids_full.py`
- Quantize lite: `scripts/quantize_helix_lite.py`
- Quantize micro: `scripts/quantize_helix_micro.py`
- Benchmark variants: `scripts/benchmark_helix_quantization.py`

Artifacts are written to:

- `models/helix_full/`
- `models/quantized/`
- `results/benchmarks/`

## Governance Layer

Governance is implemented under `src/helix_ids/governance/`.

- Stage orchestrator: `orchestrator.py`
- Entry wrapping: `entrypoint.py`
- Determinism controls: `determinism.py`
- Fingerprinting/lineage: `fingerprinting.py`, `run_registry.py`
- Promotion consensus: `promotion.py`

Tests for this layer are in `tests/test_governance/`.

## Repository Boundaries

Canonical code placement:

- Production package: `src/helix_ids/`
- Operational scripts: `scripts/`
- Tests: `tests/`
- Documentation: `docs/`

For file placement policy and cleanup behavior, see `docs/REPOSITORY_LAYOUT.md`.

## Legacy System Architecture

> **Document Version**: 1.0  
> **Last Updated**: 2026-04-08  
> **Status**: Production Ready  
> **Methodology**: SPARC (Architecture Phase)

---

## 1. System Overview

HELIX-IDS employs a modular, layered architecture designed for:

- **Portability**: Single codebase, multiple deployment targets
- **Scalability**: From ESP32 to cloud servers
- **Maintainability**: Clear separation of concerns
- **Performance**: Optimized inference pipelines

---

## 2. High-Level Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        A[Network Traffic] --> B[Flow Extractor]
        B --> C[Raw Features<br/>41 dimensions]
    end
    
    subgraph "Feature Engineering Layer"
        C --> D[Feature Selector]
        D --> E[Feature Transformer]
        E --> F[Normalizer<br/>MinMaxScaler]
        F --> G[Engineered Features<br/>32 dimensions]
    end
    
    subgraph "Model Layer"
        G --> H{Platform<br/>Selection}
        H -->|Production| I[MLP Full<br/>32→64→32→16→2]
        H -->|RPi 4| J[MLP Standard<br/>32→64→32→16→2]
        H -->|RPi Zero| K[MLP Lite<br/>32→32→16→2]
        H -->|ESP32| L[MLP Nano<br/>32→16→8→2]
    end
    
    subgraph "Output Layer"
        I --> M[Prediction]
        J --> M
        K --> M
        L --> M
        M --> N[Confidence Score]
        N --> O[Alert/Log]
    end
    
    style A fill:#e1f5fe
    style G fill:#fff3e0
    style M fill:#e8f5e9
    style O fill:#ffebee
```

---

## 3. Component Architecture

### 3.1 Data Flow Pipeline

```mermaid
flowchart LR
    subgraph Input["1. Input Processing"]
        A1[Raw Packet] --> A2[Flow Aggregation]
        A2 --> A3[Feature Extraction]
    end
    
    subgraph Engineering["2. Feature Engineering"]
        B1[Selection<br/>41→32 features] --> B2[Transformation]
        B2 --> B3[Scaling]
    end
    
    subgraph Inference["3. Model Inference"]
        C1[Load Model] --> C2[Forward Pass]
        C2 --> C3[Softmax]
    end
    
    subgraph Output["4. Output"]
        D1[Classification] --> D2[Confidence]
        D2 --> D3[Logging]
    end
    
    A3 --> B1
    B3 --> C1
    C3 --> D1
```

### 3.2 Module Dependency Graph

```mermaid
graph TD
    subgraph Core["Core Modules"]
        M1[helix_ids.data]
        M2[helix_ids.features]
        M3[helix_ids.models]
        M4[helix_ids.inference]
    end
    
    subgraph Utils["Utility Modules"]
        U1[helix_ids.utils.metrics]
        U2[helix_ids.utils.logging]
        U3[helix_ids.utils.config]
    end
    
    subgraph Deploy["Deployment Modules"]
        D1[deploy.production]
        D2[deploy.rpi]
        D3[deploy.esp32]
    end
    
    M1 --> M2
    M2 --> M3
    M3 --> M4
    
    U1 --> M3
    U2 --> M1
    U2 --> M4
    U3 --> M1
    U3 --> M3
    
    M4 --> D1
    M4 --> D2
    M3 --> D3
```

---

## 4. Model Architecture

### 4.1 MLP Network Architecture

```mermaid
graph LR
    subgraph Input["Input Layer"]
        I[32 Features]
    end
    
    subgraph Hidden1["Hidden Layer 1"]
        H1_1[Neuron 1]
        H1_2[Neuron 2]
        H1_N[...]
        H1_64[Neuron 64]
    end
    
    subgraph BN1["BatchNorm + ReLU + Dropout"]
        BN1_op[64 units]
    end
    
    subgraph Hidden2["Hidden Layer 2"]
        H2_1[Neuron 1]
        H2_N[...]
        H2_32[Neuron 32]
    end
    
    subgraph BN2["BatchNorm + ReLU + Dropout"]
        BN2_op[32 units]
    end
    
    subgraph Hidden3["Hidden Layer 3"]
        H3_1[Neuron 1]
        H3_N[...]
        H3_16[Neuron 16]
    end
    
    subgraph BN3["BatchNorm + ReLU + Dropout"]
        BN3_op[16 units]
    end
    
    subgraph Output["Output Layer"]
        O1[Normal]
        O2[Attack]
    end
    
    I --> H1_1
    I --> H1_2
    I --> H1_N
    I --> H1_64
    
    H1_1 --> BN1_op
    H1_64 --> BN1_op
    BN1_op --> H2_1
    BN1_op --> H2_32
    
    H2_1 --> BN2_op
    H2_32 --> BN2_op
    BN2_op --> H3_1
    BN2_op --> H3_16
    
    H3_1 --> BN3_op
    H3_16 --> BN3_op
    BN3_op --> O1
    BN3_op --> O2
```

### 4.2 Layer Specifications

| Layer             | Input Dim | Output Dim | Parameters | Activation |
| ----------------- | --------- | ---------- | ---------- | ---------- |
| Linear 1          | 32        | 64         | 2,112      | -          |
| BatchNorm 1       | 64        | 64         | 128        | -          |
| ReLU 1            | 64        | 64         | 0          | ReLU       |
| Dropout 1         | 64        | 64         | 0          | -          |
| Linear 2          | 64        | 32         | 2,080      | -          |
| BatchNorm 2       | 32        | 32         | 64         | -          |
| ReLU 2            | 32        | 32         | 0          | ReLU       |
| Dropout 2         | 32        | 32         | 0          | -          |
| Linear 3          | 32        | 16         | 528        | -          |
| BatchNorm 3       | 16        | 16         | 32         | -          |
| ReLU 3            | 16        | 16         | 0          | ReLU       |
| Dropout 3         | 16        | 16         | 0          | -          |
| Linear 4 (Output) | 16        | 2          | 34         | Softmax    |
| **Total**         | -         | -          | **4,978**  | -          |

---

## 5. Multi-Platform Deployment Architecture

### 5.1 Platform Comparison

```mermaid
graph TB
    subgraph Production["Production Server"]
        P1[Full Model<br/>4,978 params]
        P2[PyTorch Runtime]
        P3[FP32 Precision]
        P4["Latency: <1ms"]
    end
    
    subgraph RPi4["Raspberry Pi 4"]
        R4_1[Full Model<br/>4,978 params]
        R4_2[ONNX Runtime]
        R4_3[FP32/INT8]
        R4_4["Latency: <5ms"]
    end
    
    subgraph RPi0["Raspberry Pi Zero"]
        R0_1[Lite Model<br/>~2,000 params]
        R0_2[NumPy Runtime]
        R0_3[INT8 Quantized]
        R0_4["Latency: <10ms"]
    end
    
    subgraph ESP32["ESP32"]
        E1[Nano Model<br/><1,000 params]
        E2[C Runtime]
        E3[INT8 Fixed-Point]
        E4["Latency: <100ms"]
    end
    
    style Production fill:#c8e6c9
    style RPi4 fill:#b3e5fc
    style RPi0 fill:#fff9c4
    style ESP32 fill:#ffccbc
```

### 5.2 Deployment Pipeline

```mermaid
flowchart TD
    A[Trained PyTorch Model] --> B{Export Format}
    
    B -->|Production| C[PyTorch .pt]
    B -->|ONNX| D[ONNX .onnx]
    B -->|TFLite| E[TFLite .tflite]
    B -->|C Header| F[C .h]
    
    C --> G[Production Server<br/>Docker Container]
    D --> H[RPi 4<br/>ONNX Runtime]
    D --> I[RPi Zero<br/>NumPy Fallback]
    F --> J[ESP32<br/>Native C]
    
    subgraph Quantization
        K[INT8 Quantization]
        L[Weight Pruning 50%]
    end
    
    E --> K
    F --> K
    F --> L
    
    K --> H
    K --> I
    K --> J
    L --> J
```

---

## 6. Feature Engineering Architecture

### 6.1 Feature Transformation Pipeline

```mermaid
flowchart LR
    subgraph Raw["Raw Features (41)"]
        R1[Duration]
        R2[Bytes]
        R3[Flags]
        R4[Counts]
        R5[Rates]
    end
    
    subgraph Transform["Transformations"]
        T1[Log Transform]
        T2[Ratio Calculation]
        T3[Interaction Features]
        T4[Passthrough]
    end
    
    subgraph Engineered["Engineered Features (32)"]
        E1[log_duration]
        E2[byte_ratio]
        E3[rate_count_interaction]
        E4[original features]
    end
    
    R1 --> T1 --> E1
    R2 --> T2 --> E2
    R4 --> T3
    R5 --> T3
    T3 --> E3
    R3 --> T4 --> E4
```

### 6.2 Feature Categories

```mermaid
pie title Feature Distribution (32 total)
    "Byte Statistics" : 4
    "Rate Features" : 8
    "Count Features" : 6
    "Interaction Features" : 4
    "Log Transforms" : 5
    "Binary Flags" : 5
```

---

## 7. Training Architecture

### 7.1 Training Pipeline

```mermaid
flowchart TD
    subgraph Data["Data Preparation"]
        D1[NSL-KDD<br/>148K samples] --> D3[Combined Dataset]
        D2[UNSW-NB15<br/>100K samples] --> D3
        D3 --> D4[Feature Alignment]
        D4 --> D5[Train/Val/Test Split<br/>70/15/15]
    end
    
    subgraph Validation["Validation Gate"]
        V1[Logistic Regression]
        V2{F1 >= 0.75?}
        V1 --> V2
        V2 -->|No| V3[Feature Engineering<br/>Iteration]
        V2 -->|Yes| T1
        V3 --> D4
    end
    
    subgraph Training["Model Training"]
        T1[Initialize MLP]
        T2[Training Loop]
        T3[Early Stopping]
        T4[Best Model Checkpoint]
    end
    
    subgraph Evaluation["Evaluation"]
        E1[Test Set Metrics]
        E2[Cross-Dataset Validation]
        E3[Latency Benchmarks]
    end
    
    D5 --> V1
    T1 --> T2 --> T3 --> T4
    T4 --> E1 --> E2 --> E3
```

### 7.2 Training Hyperparameters

```mermaid
mindmap
  root((Training Config))
    Optimizer
      Adam
      LR: 0.001
      Weight Decay: 0.0001
    Loss
      CrossEntropy
      Class Weights
    Regularization
      Dropout: 0.2
      BatchNorm
      Gradient Clip: 1.0
    Schedule
      Epochs: 100 max
      Patience: 20
      LR Decay: 0.9 per 10 epochs
    Data
      Batch Size: 256
      Stratified Split
      Validation: 15%
```

---

## 8. Cross-Dataset Generalization

### 8.1 Feature Alignment Architecture

```mermaid
flowchart LR
    subgraph NSL["NSL-KDD Features"]
        N1[duration]
        N2[src_bytes]
        N3[serror_rate]
        N4[logged_in]
    end
    
    subgraph Mapping["Feature Mapping"]
        M1[Direct Mapping<br/>6 features]
        M2[Synthetic Engineering<br/>20 features]
        M3[Passthrough<br/>6 features]
    end
    
    subgraph UNSW["UNSW-NB15 Features"]
        U1[dur]
        U2[sbytes]
        U3[state-derived]
        U4[computed]
    end
    
    subgraph Unified["Unified Features (32)"]
        UF[Cross-Dataset<br/>Compatible]
    end
    
    N1 --> M1
    U1 --> M1
    M1 --> UF
    
    N3 --> M2
    U3 --> M2
    M2 --> UF
    
    N4 --> M3
    M3 --> UF
```

### 8.2 Label Alignment

```mermaid
graph LR
    subgraph UNSW_Labels["UNSW-NB15 Categories"]
        UL1[Fuzzers]
        UL2[Analysis]
        UL3[Backdoor]
        UL4[DoS]
        UL5[Exploits]
        UL6[Generic]
        UL7[Reconnaissance]
        UL8[Shellcode]
        UL9[Worms]
        UL10[Normal]
    end
    
    subgraph NSL_Labels["NSL-KDD Classes"]
        NL1[Normal]
        NL2[DoS]
        NL3[Probe]
        NL4[R2L]
        NL5[U2R]
    end
    
    UL10 --> NL1
    UL4 --> NL2
    UL6 --> NL2
    UL1 --> NL3
    UL2 --> NL3
    UL7 --> NL3
    UL3 --> NL4
    UL5 --> NL4
    UL9 --> NL4
    UL8 --> NL5
```

---

## 9. Inference Architecture

### 9.1 Real-Time Inference Pipeline

```mermaid
sequenceDiagram
    participant C as Client
    participant API as API Server
    participant FE as Feature Engine
    participant M as Model
    participant L as Logger
    
    C->>API: POST /predict (raw_features)
    API->>FE: extract_features(raw)
    FE->>FE: engineer_features()
    FE->>FE: scale_features()
    FE-->>API: engineered_features
    API->>M: forward(features)
    M->>M: layer1(features)
    M->>M: layer2(hidden1)
    M->>M: layer3(hidden2)
    M->>M: output(hidden3)
    M-->>API: logits
    API->>API: softmax(logits)
    API->>L: log_prediction()
    API-->>C: {prediction, confidence, latency}
```

### 9.2 Batch Processing Architecture

```mermaid
flowchart TB
    subgraph Input["Batch Input"]
        I1[Samples 1-256]
        I2[Samples 257-512]
        I3[Samples ...]
    end
    
    subgraph Processing["Parallel Processing"]
        P1[Feature Batch 1]
        P2[Feature Batch 2]
        P3[Feature Batch N]
    end
    
    subgraph Model["Model Inference"]
        M1[Batch Forward Pass]
        M2[GPU/CPU Vectorization]
    end
    
    subgraph Output["Aggregated Output"]
        O1[Predictions Array]
        O2[Confidence Array]
        O3[Metrics Summary]
    end
    
    I1 --> P1
    I2 --> P2
    I3 --> P3
    
    P1 --> M1
    P2 --> M1
    P3 --> M1
    
    M1 --> M2
    M2 --> O1
    M2 --> O2
    O1 --> O3
    O2 --> O3
```

---

## 10. Directory Structure

```text
helix-ids/
├── config/
│   ├── helix_config.yaml       # Model configurations
│   ├── platform_configs.yaml   # Platform-specific settings
│   ├── training.yaml           # Training hyperparameters
│   └── attack_params.yaml      # Attack classification params
│
├── data/
│   ├── nsl_kdd/                # NSL-KDD dataset
│   ├── unsw_nb15/              # UNSW-NB15 dataset
│   ├── processed/              # Cleaned and aligned data
│   └── splits/                 # Train/val/test indices
│
├── docs/
│   ├── SPECIFICATION.md        # System requirements
│   ├── PSEUDOCODE.md           # Algorithms
│   ├── ARCHITECTURE.md         # This document
│   ├── REFINEMENT.md           # Optimization history
│   └── COMPLETION.md           # Final metrics & deployment
│
├── models/
│   ├── production/             # Production model artifacts
│   ├── rpi_4/                  # RPi 4 optimized model
│   ├── rpi_zero/               # RPi Zero optimized model
│   └── esp32/                  # ESP32 C header model
│
├── scripts/
│   ├── feature_engineering.py  # Feature pipeline
│   ├── train_platform_models.py
│   ├── deploy.py               # Deployment automation
│   └── benchmark_e2e.py        # End-to-end benchmarks
│
├── src/
│   └── helix_ids/
│       ├── __init__.py
│       ├── data/               # Data loading modules
│       ├── features/           # Feature engineering
│       ├── models/             # Model definitions
│       ├── inference/          # Inference pipelines
│       └── utils/              # Utilities
│
├── tests/
│   ├── test_data/
│   ├── test_models/
│   └── test_utils/
│
└── results/
    ├── benchmarks/             # Performance results
    ├── experiments/            # Experiment outputs
    └── figures/                # Visualizations
```

---

## 11. API Interface Architecture

### 11.1 REST API Endpoints

```text
Production API Endpoints:

POST /api/v1/predict
  - Input: {"features": [...]}
  - Output: {"prediction": 0/1, "confidence": 0.95, "latency_ms": 0.5}

POST /api/v1/predict/batch
  - Input: {"samples": [[...], [...], ...]}
  - Output: {"predictions": [...], "metrics": {...}}

GET /api/v1/health
  - Output: {"status": "healthy", "model_version": "1.0.0"}

GET /api/v1/model/info
  - Output: {"architecture": "MLP", "params": 4978, "features": 32}
```

### 11.2 Data Contracts

```python
# Request Schema
PredictRequest = {
    "features": List[float],  # 32 engineered features
    "return_probabilities": bool  # Optional, default False
}

# Response Schema
PredictResponse = {
    "prediction": int,           # 0=Normal, 1=Attack
    "class_name": str,           # "Normal" or "Attack"
    "confidence": float,         # 0.0-1.0
    "probabilities": {           # Optional
        "Normal": float,
        "Attack": float
    },
    "latency_ms": float,
    "model_version": str
}
```

---

## 12. Security Architecture

### 12.1 Model Security

```mermaid
flowchart TB
    subgraph Input["Input Validation"]
        IV1[Feature Range Checks]
        IV2[Type Validation]
        IV3[Anomaly Detection]
    end
    
    subgraph Model["Model Protection"]
        MP1[Input Sanitization]
        MP2[Rate Limiting]
        MP3[Model Encryption]
    end
    
    subgraph Output["Output Security"]
        OS1[Confidence Thresholding]
        OS2[Audit Logging]
        OS3[Alert Rate Limiting]
    end
    
    IV1 --> MP1
    IV2 --> MP1
    IV3 --> MP2
    MP1 --> Model_Inference
    Model_Inference --> OS1
    OS1 --> OS2
    OS2 --> OS3
```

### 12.2 Adversarial Robustness

- **Input perturbation tolerance**: ±5% feature noise
- **Confidence calibration**: Platt scaling for reliable uncertainty
- **Out-of-distribution detection**: Mahalanobis distance threshold
- **Model integrity**: SHA-256 checksum validation

---

## 13. Scalability Considerations

### 13.1 Horizontal Scaling (Production)

```mermaid
graph TB
    LB[Load Balancer] --> S1[Server 1]
    LB --> S2[Server 2]
    LB --> S3[Server N]
    
    S1 --> Cache[Redis Cache]
    S2 --> Cache
    S3 --> Cache
    
    S1 --> DB[(Metrics DB)]
    S2 --> DB
    S3 --> DB
```

### 13.2 Edge Scaling (IoT)

- **Hub-and-Spoke**: Central RPi 4 with ESP32 nodes
- **Federated Updates**: Model updates pushed from cloud
- **Local Aggregation**: Batch predictions on hub

---

*Document generated following SPARC Architecture methodology. See REFINEMENT.md for optimization history.*
