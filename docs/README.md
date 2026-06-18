# HELIX-IDS Documentation

## Structure

```
docs/
├── README.md                  # This file
├── architecture/              # System architecture, model design, schemas
│   ├── ARCHITECTURE.md        # Canonical package boundaries, model/runtime scope
│   ├── ARCHITECTURE_FULL.md   # Full architecture description
│   ├── CHECKPOINT_CERTIFICATION.md
│   ├── CONFIG_GOVERNANCE.md
│   ├── EXPERIMENTAL_SETUP.md
│   ├── FAILURE_MODES.md
│   ├── FEATURE_HARMONIZATION.md
│   ├── FINAL_METRICS.md
│   ├── MODEL_ARCHITECTURE.md
│   ├── PRODUCTION_READINESS.md
│   ├── RC3_READINESS_VERDICT.md
│   ├── REPRODUCIBILITY_AUDIT.md
│   ├── SCHEMA_CONTRACT.md
│   ├── TECHNICAL_DEBT_REGISTER.md
│   ├── TECHNICAL_DEBT_ROADMAP.md
│   ├── TRAINER_FINAL_AUDIT.md
│   ├── TRAINING_METHODOLOGY.md
│   └── dependency_graph.md + dependency_graph.json
├── audits/                    # Phase 23 audit deliverables
│   ├── DEAD_FILE_AUDIT.md
│   ├── DELETE_CANDIDATES.md
│   ├── REPOSITORY_STRUCTURE.md
│   ├── NAMING_STANDARDIZATION.md
│   ├── GITIGNORE_AUDIT.md
│   ├── DOC_RATIONALIZATION.md
│   ├── ARTIFACT_RETENTION_POLICY.md
│   ├── TEST_SUITE_MAP.md
│   └── DEPENDENCY_AUDIT.md
├── archive/                   # Historical phase documentation (archived)
│   ├── phase4/                # Phase 4A/4B governance audits
│   ├── phase11a/              # Phase 11A cleanup report
│   ├── phase13/               # Phase 13B architecture audit
│   ├── phase19/               # Phase 19 architecture freeze
│   ├── phase22/               # Phase 22 reliability plan
│   ├── phase23/               # Phase 23 CI/CD consolidation
│   └── superseded/            # Superseded docs (dead code, dependency, etc.)
├── compliance/                # License policy, supply chain
│   ├── LICENSE_POLICY.md
│   └── SUPPLY_CHAIN.md
├── development/               # Project status
│   └── PROJECT_STATUS.md
├── governance/                # ADRs, hash authority, schema contracts
│   ├── ADR-001-governance-philosophy.md
│   ├── ADR-002-schema-lifecycle.md
│   ├── ADR-003-hash-authority.md
│   ├── ADR-004-enforcement-pipeline.md
│   ├── IMMUTABLE_SCHEMA_CONTRACT.md
│   ├── PRI_FRAMEWORK.md
│   ├── hash_authority.md
│   ├── manifest_schema_governance.md
│   └── result_schema_governance.md
├── manuscript/                # Paper drafts
│   ├── HELIX_ieee_variant.md
│   └── HELIX_submission_ready.md
├── operations/                # Deployment runbooks, branch governance
│   ├── BRANCH_GOVERNANCE_APPLIED.md
│   ├── BRANCH_GOVERNANCE_FINAL.md
│   ├── OPERATIONS_CERTIFICATION.md
│   ├── OPERATIONS_DEPLOYMENT_RUNBOOK.md
│   └── RELEASE_PIPELINE_CERTIFICATION.md
├── releases/                  # Release certification docs
│   ├── RC1_READINESS.md
│   ├── RC2_CERTIFICATION.md
│   └── RC2_READINESS.md
├── reports/                   # Analysis reports
│   ├── BENCHMARK_PROTOCOL.md
│   ├── DATASET_REPORT.md
│   ├── LIMITATIONS_AND_THREATS.md
│   └── MUTATION_SCORECARD.md
├── reproducibility/           # Reproducibility guides
│   ├── CONTAINER_REPRODUCIBILITY.md
│   ├── DATA_PIPELINE.md
│   └── REPRODUCIBLE_BUILD_GUIDE.md
├── security/                  # Security posture
│   ├── SECURITY_POSTURE.md
│   └── SECURITY_REVIEW.md
└── figures/                   # Figures (6 PNGs for manuscript)
```

## Quick Reference

| Area | Key Doc | Purpose |
|------|---------|---------|
| Architecture | `architecture/ARCHITECTURE.md` | Package boundaries, model/runtime scope |
| Operations | `operations/OPERATIONS_DEPLOYMENT_RUNBOOK.md` | Deployment gates, metrics, rollout |
| Governance | `governance/ADR-001-governance-philosophy.md` | ADR-001: Governance philosophy |
| Manuscript | `manuscript/HELIX_submission_ready.md` | Paper draft |
| Reproducibility | `reproducibility/REPRODUCIBLE_BUILD_GUIDE.md` | Container-based reproducible build |
| Security | `security/SECURITY_POSTURE.md` | Security posture and review |
| Releases | `releases/RC2_CERTIFICATION.md` | RC2 certification status |
| Reports | `reports/BENCHMARK_PROTOCOL.md` | Benchmark and mutation testing |
