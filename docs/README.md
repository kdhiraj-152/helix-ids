# HELIX-IDS Documentation

## Structure

```
docs/
├── README.md                  # This file
├── architecture/              # System architecture, model design, schemas
│   ├── ARCHITECTURE.md
│   ├── ARCHITECTURE_FULL.md
│   ├── MODEL_ARCHITECTURE.md
│   └── SCHEMA_CONTRACT.md
├── development/               # Training methodology, data pipeline, features
│   ├── TRAINING_METHODOLOGY.md
│   ├── DATA_PIPELINE.md
│   ├── DATASET_REPORT.md
│   ├── EXPERIMENTAL_SETUP.md
│   └── FEATURE_HARMONIZATION.md
├── governance/                # ADRs, hash authority, schema contracts
│   ├── ADR-001-governance-philosophy.md
│   ├── ADR-002-schema-lifecycle.md
│   ├── ADR-003-hash-authority.md
│   ├── ADR-004-enforcement-pipeline.md
│   ├── HASH_AUTHORITY.md
│   ├── IMMUTABLE_SCHEMA_CONTRACT.md
│   ├── MANIFEST_SCHEMA_GOVERNANCE.md
│   ├── PHASE_4A_GOVERNANCE_COVERAGE_AUDIT.md
│   ├── PHASE_4B_ASSUMPTION_ELIMINATION.md
│   ├── REPRODUCIBILITY_GAP.md
│   └── RESULT_SCHEMA_GOVERNANCE.md
├── operations/                # Deployment runbooks, checkpoint audit
│   ├── OPERATIONS_DEPLOYMENT_RUNBOOK.md
│   └── CHECKPOINT_AUDIT.md
├── reports/                   # Audits, reviews, analyses, benchmarks
│   ├── BENCHMARK_PROTOCOL.md
│   ├── EXPORT_CONTRACT_REPORT.md
│   ├── GOVERNANCE_AND_PROVENANCE.md
│   ├── HELIX_FORENSIC_CANONICALIZATION_AUDIT.md
│   ├── LIMITATIONS_AND_THREATS.md
│   ├── PAPER_READINESS_AUDIT.md
│   ├── PRI_FRAMEWORK.md
│   ├── REPRODUCIBILITY.md
│   ├── SECURITY_REVIEW.md
│   └── target_repository_layout.md
├── results/                   # Staging validation artifacts
│   ├── results/
│   └── fig/
├── manuscript/                # Paper drafts and figures
│   └── ...
├── archives/                  # Historical phase documentation
│   └── phase5/
└── fig_revamp/                # Revamped figures for manuscript
```

## Quick Reference

| Area | Key Doc | Purpose |
|------|---------|---------|
| Architecture | `architecture/ARCHITECTURE.md` | Package boundaries, model/runtime scope |
| Operations | `operations/OPERATIONS_DEPLOYMENT_RUNBOOK.md` | Deployment gates, metrics, rollout |
| Governance | `governance/ADR-001-governance-philosophy.md` | ADR-001: Governance philosophy |
| Manuscript | `manuscript/HELIX_submission_ready.md` | Paper draft |
