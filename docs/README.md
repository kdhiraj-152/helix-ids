# HELIX-IDS Documentation

This directory is the single source of truth for HELIX-IDS documentation.
It is intentionally minimal — each topic has exactly one authoritative document.

```
docs/
├── README.md                              # This file
├── architecture/                          # System design
│   ├── SYSTEM_ARCHITECTURE.md             # High-level architecture, model, training, inference
│   ├── DATA_FLOW.md                       # Data pipeline from raw captures to predictions
│   ├── GOVERNANCE.md                      # Governance policies, ADRs, config governance
│   └── DECISIONS.md                       # Architecture Decision Record summaries
├── development/                           # Developer guidance
│   ├── TESTING.md                         # All testing: unit, integration, mutation, chaos
│   ├── CODING_STANDARDS.md                # Style, linting, type annotations, conventions
│   ├── CONTRIBUTING.md                    # How to contribute
│   └── RELEASE_PROCESS.md                 # Release pipeline, CI workflows, branch governance
├── operations/                            # Deployment and runtime
│   ├── DEPLOYMENT.md                      # Full deployment runbook (6 phases)
│   ├── MONITORING.md                      # Metrics, alerting, telemetry
│   ├── RECOVERY.md                        # Failure modes, detection, recovery
│   └── SOAK_TESTING.md                    # Soak and load test certification
├── api/                                   # API reference
│   └── API_REFERENCE.md                   # REST endpoints, CLI commands
├── reports/                               # Certification and audit
│   ├── RC3_READINESS_VERDICT.md           # RC3 certification verdict
│   └── AUDIT_BASELINE_2026.md             # Consolidated audit findings
├── changelog/                             # Phase history
│   └── CHANGELOG.md                       # All notable changes
├── manuscript/                            # Paper drafts (preserved as-is)
│   ├── HELIX_submission_ready.md
│   └── HELIX_ieee_variant.md
├── final/                                 # Publication-ready paper + supporting docs
│   ├── PAPER_DRAFT.md                     # Full paper draft
│   ├── MASTER_RESULTS_TABLE.md            # Consolidated results
│   ├── REPRODUCIBILITY.md                 # Reproducibility details
│   ├── FAILURE_ANALYSIS.md                # Failure mode analysis
│   ├── RESEARCH_TIMELINE.md               # Research timeline
│   ├── THREATS_TO_VALIDITY.md             # Threats to validity
│   └── FUTURE_WORK.md                     # Future work
├── redteam/                               # Red/blue team security audits
│   ├── PHASE37_RED_TEAM_AUDIT.md
│   └── PHASE38_BLUE_TEAM_REBUTTAL_COMPLETE.md
├── releases/                              # Phase certification reports
│   ├── PHASE23_OPERATIONAL_CERTIFICATION.md
│   ├── PHASE24B_REPOSITORY_CLEANUP.md
│   └── ... (Phase 25–36 certification reports)
├── phase31–43h/                           # Research phase documentation
│   ├── phase31/  phase32/  phase33/  phase34/  phase36/
│   └── phase43a/ phase43b/ phase43c/ phase43d/ phase43e/ phase43g/ phase43h/
├── figures/                               # Paper figures (6 PNGs, gitignored)
└── archive/                               # Historical docs (not authoritative)
    ├── phase4/                            # Phase 4A/4B governance audits
    ├── phase11a/                          # Phase 11A cleanup report
    ├── phase13/                           # Phase 13B architecture audit
    ├── phase19/                           # Phase 19 architecture freeze
    ├── phase22/                           # Phase 22 reliability plan
    ├── phase23/                           # Phase 23 CI/CD consolidation
    ├── phase24a/                          # Phase 24A repository cleanup
    └── superseded/                        # Superseded reports and analyses
```

## Quick Reference

| Area | Document | Purpose |
|------|----------|---------|
| Architecture | `architecture/SYSTEM_ARCHITECTURE.md` | Package boundaries, model, training/inference, governance |
| Testing | `development/TESTING.md` | All test types, coverage, CI gates |
| Deployment | `operations/DEPLOYMENT.md` | Runbook, stages, gates, commands |
| Monitoring | `operations/MONITORING.md` | Metrics, alerting, thresholds |
| Recovery | `operations/RECOVERY.md` | Failure modes, detection, recovery |
| API | `api/API_REFERENCE.md` | REST endpoints, CLI flags |
| Manuscript | `manuscript/HELIX_submission_ready.md` | Paper draft |
| Final Paper | `final/PAPER_DRAFT.md` | Publication-ready paper |
| Changelog | `changelog/CHANGELOG.md` | Phase history |
| Certifications | `releases/PHASE23_OPERATIONAL_CERTIFICATION.md` | Phase certification reports |

> **Note:** Historical phase documentation in `archive/` is preserved for
> reference but is **not authoritative**. If something in archive/ contradicts
> the active docs above, the active doc is correct. Phase research docs
> (`phase31/`–`phase43h/`) document intermediate experiments and results.