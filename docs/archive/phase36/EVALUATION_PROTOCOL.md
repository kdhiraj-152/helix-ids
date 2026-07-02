# Evaluation Protocol v1.0

> **Phase 36 — Deliverable 4 of 8**
> Defines five evaluation regimes for cross-dataset IDS transfer research.
> Every benchmark submission must report results for ALL five regimes.
> Date: 2026-06-24

---

## 1. Purpose

Current IDS evaluation practices use a single train-test split from the same dataset.
This measures **in-dataset memorization**, not **transfer generalization** — the ability
to detect attacks in unseen environments.

This protocol defines **five distinct evaluation regimes** that collectively measure
all aspects of cross-dataset generalization. Every submission to the benchmark must
report results for all five regimes.

---

## 2. Common Evaluation Settings

### 2.1 Metrics

Every regime reports:

| Metric | Definition | Primary? |
|--------|-----------|----------|
| **Macro F1** | Unweighted mean of per-class F1 scores | **Primary** |
| **Balanced Accuracy** | Mean of per-class recall | Secondary |
| **Precision** | Micro-averaged across all classes | Supporting |
| **Recall** | Micro-averaged across all classes | Supporting |
| **AUROC** | One-vs-rest AUC averaged across classes | Supporting |
| **Transfer Ratio** | Cross-dataset Macro F1 / In-distribution Macro F1 | Diagnostic |

**Macro F1 is the primary metric.** Cross-dataset IDS evaluation must treat all
classes equally — minority attacks (Privilege Escalation, Lateral Movement) are
as important as majority classes.

### 2.2 Dataset Structure

All evaluations use the Phase 36 Unified Dataset (PHASE36-UD), which consists of
multiple **collection runs** (see COLLECTION_PROTOCOL.md).

| Run | Description | Year-Month (simulated) |
|-----|-------------|------------------------|
| Run A | Collection week 1-3 variation | 2026-07 |
| Run B | Collection week 4-6 variation | 2026-08 |
| Run C | Same topology, different hardware | 2026-09 |
| Run D | Different organization topology | 2026-10 |
| Run E | Different organization, different IoT stack | 2027-01 |

### 2.3 Data Partitioning

Each run is split:

| Split | % of Run | Use | Stratification |
|-------|---------|-----|----------------|
| Train | 60% | Model fitting | By Level-1 class |
| Val | 15% | Hyperparameter selection | By Level-1 class |
| Test | 25% | Final evaluation | By Level-1 class |

Stratification ensures rare classes (Privilege Escalation) are proportionally
represented in all splits. If a class has fewer than 50 samples in a run, it is
excluded from that run's evaluation — but MUST still be reported as "insufficient data."

---

## 3. Regime 1: In-Distribution Evaluation

### 3.1 Objective

Measure the oracle performance — how well a model can learn within a single
controlled environment. This establishes the **upper bound** for all transfer regimes.

### 3.2 Procedure

```
Train on:  Run A (Train split)
Validate:  Run A (Val split)
Test on:   Run A (Test split)

Report:    Macro F1, Balanced Accuracy, Precision, Recall, AUROC
```

### 3.3 Interpretation

| Macro F1 Threshold | Interpretation |
|--------------------|----------------|
| ≥ 0.90 | Excellent in-distribution detection |
| 0.75 — 0.89 | Good in-distribution detection |
| 0.60 — 0.74 | Marginal in-distribution detection |
| < 0.60 | Model or feature set insufficient for in-distribution use |

In-distribution performance is the **ceiling** for transfer. If in-distribution MF1
< 0.75, the model is not suitable for transfer evaluation.

### 3.4 Expected Baselines

| Model | Expected MF1 (Run A → Run A Test) |
|-------|-----------------------------------|
| Logistic Regression | 0.82 — 0.88 |
| Random Forest | 0.88 — 0.94 |
| XGBoost | 0.89 — 0.95 |
| MLP | 0.87 — 0.93 |
| DANN | 0.86 — 0.92 |
| CORAL | 0.86 — 0.92 |
| Transformer IDS | 0.88 — 0.95 |

---

## 4. Regime 2: Cross-Organization Evaluation

### 4.1 Objective

Measure transfer between different organizations using the same collection protocol
but with different network hardware, traffic profiles, and personnel.

### 4.2 Procedure

```
Train on:  Run C (Train) — different hardware, same protocol
Validate:  Run C (Val)
Test on:   Run A (Test) — original hardware

Also reverse: Train on Run A, Test on Run C.
```

### 4.3 Interpretation

| MF1 Drop vs Regime 1 | Interpretation |
|----------------------|----------------|
| < 0.05 | Excellent cross-organization transfer |
| 0.05 — 0.15 | Good cross-organization transfer |
| 0.15 — 0.30 | Marginal — significant hardware/traffic sensitivity |
| > 0.30 | Poor — model does not generalize across organizations |

### 4.4 Expected Baselines

| Model | Expected MF1 (Run A → Run C Test) |
|-------|-----------------------------------|
| Logistic Regression | 0.60 — 0.72 |
| Random Forest | 0.65 — 0.78 |
| XGBoost | 0.67 — 0.80 |
| MLP | 0.62 — 0.75 |
| DANN | 0.70 — 0.82 |
| CORAL | 0.68 — 0.80 |
| Transformer IDS | 0.72 — 0.85 |

---

## 5. Regime 3: Cross-Time Evaluation

### 5.1 Objective

Measure temporal generalization — does the model detect attacks seen 3-6 months
after training? This captures concept drift in attack tools and benign traffic
patterns.

### 5.2 Procedure

```
Train on:  Run A (Jul 2026, Train split)
Validate:  Run A (Jul 2026, Val split)
Test on:   Run B (Aug 2026, Full test)     — 1 month later
           Run D (Oct 2026, Full test)     — 3 months later
           Run E (Jan 2027, Full test)     — 6 months later

Report MF1 for each time horizon.
```

### 5.3 Interpretation

| MF1 Decay per Month | Interpretation |
|---------------------|----------------|
| < 0.01/month | Excellent temporal stability |
| 0.01 — 0.03/month | Moderate temporal drift |
| 0.03 — 0.06/month | Significant temporal drift |
| > 0.06/month | Unacceptable — model requires daily retraining |

### 5.4 Expected Baselines

| Model | MF1 at +1mo | MF1 at +3mo | MF1 at +6mo |
|-------|:-----------:|:-----------:|:-----------:|
| Logistic Regression | 0.78 — 0.85 | 0.70 — 0.78 | 0.62 — 0.70 |
| Random Forest | 0.82 — 0.90 | 0.75 — 0.83 | 0.67 — 0.76 |
| XGBoost | 0.84 — 0.91 | 0.77 — 0.85 | 0.70 — 0.78 |
| MLP | 0.80 — 0.88 | 0.72 — 0.80 | 0.64 — 0.72 |
| DANN | 0.85 — 0.92 | 0.78 — 0.86 | 0.72 — 0.80 |
| CORAL | 0.83 — 0.90 | 0.76 — 0.84 | 0.69 — 0.77 |
| Transformer IDS | 0.86 — 0.93 | 0.80 — 0.88 | 0.74 — 0.82 |

---

## 6. Regime 4: Cross-Network Evaluation

### 6.1 Objective

Measure transfer between network tiers. Can a model trained on server-focused traffic
detect attacks in the IoT subnet? This is the most **practically relevant** regime
for real-world deployment.

### 6.2 Procedure

```
Train on:  C1 + C2 captures (DMZ + Internal — server traffic)
Test on:   C4 capture (IoT subnet traffic)

Also:
Train on:  C4 capture (IoT subnet)
Test on:   C1 + C2 captures (server traffic)
```

### 6.3 Interpretation

| MF1 | Interpretation |
|-----|----------------|
| ≥ 0.80 | Excellent cross-tier transfer |
| 0.60 — 0.79 | Good cross-tier transfer — usable with fine-tuning |
| 0.40 — 0.59 | Marginal — significant tier-specific behavior |
| < 0.40 | Poor — traffic patterns are environment-specific |

### 6.4 Expected Baselines

| Model | MF1 (Server → IoT) | MF1 (IoT → Server) |
|-------|:------------------:|:------------------:|
| Logistic Regression | 0.45 — 0.58 | 0.52 — 0.65 |
| Random Forest | 0.52 — 0.65 | 0.58 — 0.72 |
| XGBoost | 0.55 — 0.68 | 0.60 — 0.74 |
| MLP | 0.48 — 0.62 | 0.55 — 0.68 |
| DANN | 0.60 — 0.74 | 0.65 — 0.78 |
| CORAL | 0.56 — 0.70 | 0.62 — 0.75 |
| Transformer IDS | 0.65 — 0.78 | 0.68 — 0.80 |

---

## 7. Regime 5: Zero-Shot Transfer

### 7.1 Objective

The most challenging regime: can a model trained on a complete labeled environment
detect attacks in a **completely unseen** environment with **no labels**? This is
the goal of practical IDS deployment.

### 7.2 Procedure

```
Train on:  Run A (all data — Train + Val + Test)
           Run C (all data)
Test on:   Run E (completely unseen — different org, different IoT stack, 
           different time period, no labels available)

The model receives Run E features only — prediction only, no adaptation.
```

### 7.3 Additional Metrics

In addition to standard metrics, zero-shot evaluation reports:

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **Per-class MF1** | F1 for each Level-1 class separately | Identify transferable vs non-transferable classes |
| **Confusion matrix** | 7×7 matrix of predicted vs true classes | Characterize transfer failures |
| **Dataset-ID accuracy** | Accuracy of dataset origin classifier | Measure "dataset fingerprint" (target < 70%) |

### 7.4 Interpretation

| MF1 | Interpretation |
|-----|----------------|
| ≥ 0.75 | Excellent zero-shot — model truly generalizes |
| 0.50 — 0.74 | Good zero-shot — usable with minimal fine-tuning |
| 0.25 — 0.49 | Marginal — some classes transfer, others don't |
| < 0.25 | Poor — model cannot function in unseen environments |

### 7.5 Expected Baselines

| Model | Expected MF1 (Zero-Shot) |
|-------|--------------------------|
| Logistic Regression | 0.20 — 0.35 |
| Random Forest | 0.28 — 0.42 |
| XGBoost | 0.30 — 0.45 |
| MLP | 0.22 — 0.38 |
| DANN | 0.35 — 0.52 |
| CORAL | 0.32 — 0.48 |
| Transformer IDS | 0.40 — 0.58 |

---

## 8. Reporting Template

All submissions MUST use the following reporting format:

```markdown
### Submission: [Name]
### Date: [YYYY-MM-DD]

## Regime 1: In-Distribution
| Model | Macro F1 | Bal Acc | Precision | Recall | AUROC |
|-------|:--------:|:-------:|:---------:|:------:|:-----:|
| MyModel | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |

## Regime 2: Cross-Organization
| Direction | Macro F1 | Bal Acc | Precision | Recall | AUROC |
|-----------|:--------:|:-------:|:---------:|:------:|:-----:|
| Run A → Run C | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Run C → Run A | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |

## Regime 3: Cross-Time
| Horizon | Macro F1 | Bal Acc | Precision | Recall | AUROC |
|---------|:--------:|:-------:|:---------:|:------:|:-----:|
| +1 month | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| +3 months | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| +6 months | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |

## Regime 4: Cross-Network
| Direction | Macro F1 | Bal Acc | Precision | Recall | AUROC |
|-----------|:--------:|:-------:|:---------:|:------:|:-----:|
| Server → IoT | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| IoT → Server | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |

## Regime 5: Zero-Shot
| Metric | Value |
|--------|:-----:|
| Macro F1 | 0.XX |
| Balanced Accuracy | 0.XX |
| Precision | 0.XX |
| Recall | 0.XX |
| AUROC | 0.XX |
| Dataset-ID Accuracy | 0.XX |
| Highest-Class MF1 | class_name (0.XX) |
| Lowest-Class MF1 | class_name (0.XX) |
```

---

## 9. Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-06-24 | Initial release — five regime evaluation protocol |
