# HELIX-IDS Metrics Module

Comprehensive evaluation and monitoring utilities for the HELIX-IDS intrusion detection system.

## Overview

The metrics module provides tools for:
- **Per-class performance evaluation** (precision, recall, F1, support)
- **Threshold-based quality assurance** (automated violation detection)
- **AUC-ROC computation** for probabilistic models
- **Formatted reporting** with visual indicators
- **JSON serialization** for logging and analysis

## Core Classes

### PerClassMetrics

Main class for computing and tracking per-class evaluation metrics.

```python
from helix_ids.metrics import PerClassMetrics

# Initialize with IDS class names
pcm = PerClassMetrics([
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
])

# Compute metrics
result = pcm.compute(y_true, y_pred, y_proba=None)

# Print formatted report
pcm.print_report(result, show_cm=True)
```

### Target Thresholds

Default F1 score thresholds per class (configurable):

| Class | Threshold | Rationale |
|-------|-----------|-----------|
| Normal | 0.98 | Majority class, highest expectations |
| DoS | 0.95 | High-volume attacks, well-balanced |
| Probe | 0.90 | Moderate frequency, reasonable threshold |
| R2L | 0.80 | Rare but critical, relaxed threshold |
| U2R | 0.60 | Very rare, challenge to detect |

### Custom Thresholds

Override defaults with custom thresholds:

```python
custom_thresholds = {
    "Normal": 0.99,
    "DoS": 0.96,
    "Probe": 0.92,
    "R2L": 0.85,
    "U2R": 0.70
}

pcm = PerClassMetrics(class_names, thresholds=custom_thresholds)
```

## API Reference

### `PerClassMetrics.compute(y_true, y_pred, y_proba=None)`

Compute comprehensive per-class metrics.

**Parameters:**
- `y_true` (array-like): True labels
- `y_pred` (array-like): Predicted labels
- `y_proba` (array-like, optional): Probability predictions (n_samples × n_classes)

**Returns:**
- `PerClassMetricsResult`: Object with metrics, violations, and confusion matrix

**Example:**
```python
result = pcm.compute(y_test, predictions)

# Access individual class metrics
for class_name in ["Normal", "DoS", "Probe", "R2L", "U2R"]:
    metrics = result.per_class[class_name]
    print(f"{class_name}: F1={metrics.f1:.4f}, Recall={metrics.recall:.4f}")

# Check violations
if result.violations:
    for violation in result.violations:
        print(f"⚠️  {violation}")
```

### `PerClassMetrics.print_report(result, show_cm=False)`

Print a formatted evaluation report.

**Parameters:**
- `result` (PerClassMetricsResult): Metrics result from `compute()`
- `show_cm` (bool): Include confusion matrix in report

**Features:**
- Per-class metrics in formatted table
- Threshold comparison with pass/fail status
- Visual indicators (✓/✗)
- Violation alerts
- Optional confusion matrix

### `PerClassMetrics.get_summary(result)`

Get a concise single-line summary.

**Returns:** String summary of key metrics

**Example:**
```python
summary = pcm.get_summary(result)
# Output: "Macro-F1: 0.8920, Weighted-F1: 0.9234, Violations: 2"
```

## Data Classes

### ClassMetrics

Metrics for a single class:
- `precision`: Positive prediction accuracy
- `recall`: True positive detection rate
- `f1`: Harmonic mean of precision and recall
- `support`: Number of samples in this class
- `auc_roc`: Area under ROC curve (optional)

```python
from helix_ids.metrics import ClassMetrics

metrics = ClassMetrics(
    precision=0.95,
    recall=0.92,
    f1=0.935,
    support=500,
    auc_roc=0.98
)

# Convert to dictionary
metrics_dict = metrics.to_dict()
```

### PerClassMetricsResult

Complete evaluation result:
- `per_class`: Dict[str, ClassMetrics] for each class
- `macro_f1`: Unweighted average F1 across classes
- `weighted_f1`: Weighted average F1 by support
- `confusion_matrix`: sklearn confusion matrix
- `violations`: List of threshold violation messages

```python
from helix_ids.metrics import PerClassMetricsResult

result = pcm.compute(y_true, y_pred)

# JSON-serializable dictionary
metrics_dict = result.to_dict()

# Serialization example
import json
json_str = json.dumps(metrics_dict)
```

## Usage Examples

### Example 1: Basic Evaluation

```python
import numpy as np
from helix_ids.metrics import PerClassMetrics

# Create tracker
pcm = PerClassMetrics(["Normal", "DoS", "Probe", "R2L", "U2R"])

# Sample predictions
y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
y_pred = np.array([0, 0, 1, 0, 2, 2, 3, 3, 4, 3])

# Evaluate
result = pcm.compute(y_true, y_pred)

# Report
pcm.print_report(result)
```

### Example 2: With Probability Predictions

```python
# Compute AUC-ROC for each class
y_proba = model.predict_proba(X_test)
result = pcm.compute(y_test, y_pred, y_proba=y_proba)

# Report includes AUC-ROC column
pcm.print_report(result)
```

### Example 3: Threshold Monitoring

```python
# Check for violations
result = pcm.compute(y_true, y_pred)

if result.violations:
    print("⚠️  Quality alerts detected:")
    for violation in result.violations:
        print(f"  {violation}")
else:
    print("✓ All classes meet quality thresholds")
```

### Example 4: Logging and Storage

```python
# Convert to JSON-serializable format
metrics_dict = result.to_dict()

# Log to MLflow, database, or file
import json
with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
```

## Integration

### With sklearn models

```python
from sklearn.ensemble import RandomForestClassifier
from helix_ids.metrics import PerClassMetrics

# Train model
model = RandomForestClassifier(n_classes=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

pcm = PerClassMetrics(["Normal", "DoS", "Probe", "R2L", "U2R"])
result = pcm.compute(y_test, y_pred, y_proba=y_proba)
pcm.print_report(result)
```

### With PyTorch models

```python
import torch
from helix_ids.metrics import PerClassMetrics

# Get predictions
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    probs = torch.softmax(outputs, dim=1)

y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
y_proba = probs.cpu().numpy()

# Evaluate
pcm = PerClassMetrics(["Normal", "DoS", "Probe", "R2L", "U2R"])
result = pcm.compute(y_test, y_pred, y_proba=y_proba)
pcm.print_report(result)
```

## Production Deployment

For production monitoring:

```python
# Monitor on validation set
val_result = pcm.compute(y_val, val_preds)

# Check quality gates
if val_result.violations:
    # Alert or retrain
    logger.error(f"Model quality degradation: {val_result.violations}")
else:
    logger.info(f"✓ Model metrics: {pcm.get_summary(val_result)}")

# Log metrics
for class_name, metrics in val_result.per_class.items():
    mlflow.log_metric(f"f1_{class_name}", metrics.f1)
    mlflow.log_metric(f"recall_{class_name}", metrics.recall)
```

## Testing

Run the test suite:

```bash
pytest tests/test_per_class_metrics.py -v
```

Run examples:

```bash
python examples/per_class_metrics_example.py
```

## See Also

- `src/helix_ids/utils/metrics.py` - Additional metrics utilities
- `src/helix_ids/metrics/adversarial_test.py` - Adversarial robustness testing
- `src/helix_ids/metrics/fn_tracker.py` - False negative tracking
