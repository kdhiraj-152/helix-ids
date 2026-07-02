# Benchmark Governance v1.0

> **Phase 36 — Deliverable 7 of 8**
> Defines the operational rules for the Phase 36 Unified IDS Benchmark.
> Covers versioning, data validation, submission rules, and reproducibility.
> Date: 2026-06-24

---

## 1. Purpose

A benchmark is only as credible as its governance. Without formal rules for
versioning, validation, submissions, and reproducibility, the benchmark becomes
another publication-laundry dataset where "state-of-the-art" claims proliferate
without verification.

This document establishes the **Phase 36 Benchmark Council** and the operational
rules that govern all benchmark releases and submissions.

---

## 2. Benchmark Council

### 2.1 Structure

| Role | Count | Appointment | Term |
|------|-------|-------------|------|
| Chair | 1 | Elected by council | 2 years |
| Dataset Curators | 2-3 | Appointed by chair | Per-release |
| Evaluation Officers | 2 | Appointed by chair | 1 year |
| Community Representatives | 2 | Elected by submitters | 1 year |

### 2.2 Responsibilities

| Role | Responsibilities |
|------|-----------------|
| Chair | Sets strategic direction, approves new collection runs, resolves disputes |
| Dataset Curators | Validate quality metrics, maintain collection protocol, approve dataset releases |
| Evaluation Officers | Maintain evaluation codebase, verify submission reproducibility, approve baseline implementations |
| Community Representatives | Advocate for community interests, propose rule changes, monitor submission fairness |

### 2.3 Meetings

- **Quarterly**: Regular benchmark status meeting
- **Ad-hoc**: Collection run approval meeting (as needed)
- **Annual**: Full benchmark release cycle review

---

## 3. Versioning

### 3.1 Semantic Versioning

The benchmark uses semantic versioning: `MAJOR.MINOR.PATCH`

| Component | Bump Condition | Example |
|-----------|---------------|---------|
| **MAJOR** | Breaking changes to feature spec, ontology, or evaluation protocol | 1.0.0 → 2.0.0 |
| **MINOR** | New collection runs, new baselines, non-breaking protocol changes | 1.0.0 → 1.1.0 |
| **PATCH** | Bug fixes, documentation updates, label corrections | 1.0.0 → 1.0.1 |

### 3.2 Version Manifest

Each benchmark release includes a version manifest:

```yaml
benchmark_version: "1.0.0"
release_date: 2026-06-24
ontology_version: "1.0.0"
feature_spec_version: "1.0.0"
collection_protocol_version: "1.0.0"
evaluation_protocol_version: "1.0.0"
quality_metrics_version: "1.0.0"
baseline_suite_version: "1.0.0"
included_collection_runs: ["A", "B", "C"]
```

### 3.3 Deprecation Policy

| Action | Notice Period | Grandfathering |
|--------|---------------|----------------|
| Deprecate a collection run | 6 months | Submissions using deprecated run accepted for 12 months |
| Change evaluation protocol | 12 months | Previous protocol results valid for 24 months |
| Remove a baseline | 6 months | Removed baselines still reportable for 12 months |
| Major ontology change | 18 months | Old ontology accepted in parallel for 24 months |

---

## 4. Data Validation

### 4.1 Collection Run Approval

A collection run enters the benchmark ONLY after passing all four quality gates
(see QUALITY_METRICS.md):

```
Gate 1: DOS ≥ 0.30    (Dataset-ID ≤ 70%)
Gate 2: LCS ≥ 0.80    (Label overlap)
Gate 3: SOS ≥ 0.60    (Semantic consistency)
Gate 4: DIC ≥ 0.50    (Transfer ceiling)
```

### 4.2 Automated Validation Pipeline

The automated validation pipeline (`scripts/phase36/validate_collection.py`)
checks all conditions and produces a signed validation report:

```bash
python scripts/phase36/validate_collection.py \
    --collection /path/to/collection \
    --ontology docs/phase36/ATTACK_ONTOLOGY_V1.md \
    --features docs/phase36/CANONICAL_FEATURE_SPEC.md \
    --output validation_report.json
```

The validation report includes:

1. **Schema conformance** (22 features × correct types)
2. **Quality metrics** (DOS, LCS, SOS, DIC with confidence intervals)
3. **Label distribution** (per-class counts and ratios)
4. **Temporal continuity** (capture coverage analysis)
5. **Protocol compliance** (collection protocol deviation report)

### 4.3 Cryptographic Signing

Every validated collection run is signed:

```bash
gpg --armor --detach-sign \
    --default-key "Phase36 Benchmark Council" \
    validation_report.json
```

Subsequent users verify:

```bash
gpg --verify validation_report.json.asc validation_report.json
```

The council's public key is distributed via the benchmark repository.

---

## 5. Submission Rules

### 5.1 Eligibility

A submission is eligible for benchmark inclusion if:

1. **Full evaluation**: Results reported for ALL 5 regimes (not cherry-picked)
2. **All 7 baselines**: Results reported for all mandatory baselines
3. **Reproducible**: Training code and configuration provided
4. **Hardware-disclosed**: Training/inference hardware fully specified
5. **Seed-reported**: All results include ±std across 5 random seeds

### 5.2 Submission Package

A complete submission includes:

```
submission_<timestamp>/
├── results/
│   ├── regime1_in_distribution.csv
│   ├── regime2_cross_organization.csv
│   ├── regime3_cross_time.csv
│   ├── regime4_cross_network.csv
│   ├── regime5_zero_shot.csv
│   └── submission_summary.json
├── code/
│   ├── model_definition.py
│   ├── training_config.yaml
│   ├── inference.py
│   └── requirements.txt
├── trained_models/
│   └── final_model.pth (or equivalent)
├── metadata/
│   ├── hardware_spec.yaml
│   ├── training_logs.txt
│   └── runtime_measurements.csv
└── SUBMISSION.md
```

### 5.3 Allowed Deviations

| Aspect | Allowed? | Constraint |
|--------|----------|------------|
| Custom feature engineering | Yes | Must include 22 canonical features + optional extras |
| Ensemble methods | Yes | Must report individual + ensemble results |
| Pre-training on external data | Yes | Must be disclosed and subtractable |
| Test-time adaptation | Yes | Must report both before and after adaptation |
| Custom loss functions | Yes | Must be fully specified |

### 5.4 Prohibited Practices

| Practice | Rationale |
|----------|-----------|
| Training on test data | Invalidates evaluation — immediate disqualification |
| Cherry-picking regimes | Must report all 5 regimes |
| Selective seed reporting | Must report mean ± std across all 5 seeds |
| Post-hoc label correction | Ground truth labels fixed at release time |
| Architecture search on test | Model selection must use validation split only |
| Withholding failure modes | All results must be reported, including negative |

### 5.5 Submission Review

Submissions are reviewed by the Evaluation Officer:

| Stage | Timeline | Action |
|-------|----------|--------|
| Initial submission | Day 1 | Submission received and logged |
| Automated validation | Day 1-2 | Schema, format, and completeness checks |
| Baseline replication | Day 3-7 | Council replicates baselines (spot-check 2 of 7) |
| Result verification | Day 8-14 | Reproduce submitted model results |
| Review decision | Day 15 | Accept / Conditional / Reject |

Conditional acceptance: Minor issues (missing metadata, unclear documentation)
resolvable within 14 days. Rejection: Irreproducible results, protocol violation,
or data contamination.

---

## 6. Reproducibility Requirements

### 6.1 Computational Reproducibility

All submissions must be reproducible within **±0.01 Macro F1** on the same hardware.

Requirements:

1. **Fixed random seeds**: All random operations (data loading, weight init, dropout,
   augmentation) must be seeded and documented.
2. **Deterministic algorithms**: Use deterministic GPU algorithms
   (`torch.backends.cudnn.deterministic = True`).
3. **Exact environment**: A `requirements.txt` or `environment.yaml` with pinned
   versions of every dependency.
4. **Docker**: A Dockerfile reproducing the exact execution environment is recommended.

### 6.2 Statistical Reproducibility

Results must be reported as:

```
mean ± std across 5 independent runs with different random seeds
```

### 6.3 Reproducibility Audit

The council maintains a **reproducibility bank** — a trusted compute cluster where
submissions are re-run. Any submission where re-run results deviate by > 0.01 MF1
from the submitted values is flagged for investigation.

| Deviation | Action |
|-----------|--------|
| 0.01 — 0.03 MF1 | Flag; request submitter to disclose additional implementation details |
| 0.03 — 0.05 MF1 | Conditional acceptance with note; require Docker environment |
| > 0.05 MF1 | Rejected — irreproducible |

---

## 7. Leaderboard

### 7.1 Structure

A public leaderboard is maintained at:

```
https://benchmarks.helix-ids.org/phase36/leaderboard
```

| Column | Description |
|--------|-------------|
| Rank | By average Macro F1 across all 5 regimes |
| Model | Model name |
| Submission Date | Date of acceptance |
| Regime 1 MF1 | In-distribution |
| Regime 2 MF1 | Cross-organization |
| Regime 3 MF1 | Cross-time |
| Regime 4 MF1 | Cross-network |
| Regime 5 MF1 | Zero-shot |
| Average MF1 | Macro average of all 5 regimes |
| Transfer Ratio | Regime 5 MF1 / Regime 1 MF1 |
| Council Verified | Yes/No — whether council reproduced results |

### 7.2 Ranking Rules

1. Primary ranking: Average MF1 across all 5 regimes
2. Tie-breaker: Standard deviation (lower = better)
3. Secondary tie-breaker: Transfer Ratio (higher = better)
4. Only council-verified submissions appear on the leaderboard

---

## 8. Code of Conduct

### 8.1 Submission Ethics

All submitters agree to:

1. Report results in full — no omission of unfavorable results
2. Disclose any data leakage or contamination discovered post-submission
3. Not train on test set features in any form
4. Not use test set labels for model selection at any stage
5. Acknowledge all prior work on which the submission builds

### 8.2 Sanctions

| Violation | First Offense | Second Offense |
|-----------|---------------|----------------|
| Protocol violation | Conditional acceptance with warning | 6-month submission ban |
| Irreproducible results | Flagged on leaderboard | 12-month submission ban |
| Data contamination | Immediate removal + public notice | Permanent ban |
| Intentional fraud | Permanent ban + public notice | — |

---

## 9. Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-06-24 | Initial release — Benchmark Council, versioning, validation, submission rules |
