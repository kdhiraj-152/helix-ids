# Phase 23 — Branch Governance Final

Date: 2026-06-18
Status: Certified ✓

## Branch Model

```
main          — Heavily protected, release-ready
  ↑
dev           — Lightly protected, integration branch
  ↑
feature/*     — Unprotected, disposable
```

## Protection Rules

### main (Heavy Protection)

| Setting | Value |
|---------|-------|
| Require pull request | ✓ |
| Required approvals | 1 |
| Dismiss stale reviews | ✓ |
| Required status checks | ci, quality, architecture |
| Block force push | ✓ |
| Block deletion | ✓ |
| Squash merge only | ✓ |
| Allow admin bypass | ✓ (break-glass only) |

### dev (Light Protection)

| Setting | Value |
|---------|-------|
| Block deletion | ✓ |
| Allow force push | ✓ |
| Require pull request | — (direct push OK) |
| Required status checks | — |

### feature/* (No Protection)

| Setting | Value |
|---------|-------|
| Block deletion | — |
| Allow force push | ✓ |
| Any other protection | — |

## Merge Path

```
feature/foo ──PR──→ dev ──PR──→ main
                       ↕ (push)
                  Direct commits allowed
```

- **feature → dev:** PR recommended but not required; direct push allowed
- **dev → main:** PR required; ci + quality + architecture must pass
- **dev pushes:** ci runs; quality and architecture run for PRs targeting main

## Required Checks on main

| Check | Workflow | Typical Duration |
|-------|----------|------------------|
| ci | CI | <6 min |
| quality | Quality Gates | ~10–15 min |
| architecture | Architecture Lockdown | ~3 min |

These three checks are status-required for merging into main.

## Rationale

**Heavy main protection** ensures release readiness at all times.
**Light dev protection** prevents accidental deletion while allowing
rapid iteration (force-push for squashing WIP commits).
**No feature protection** keeps branch creation frictionless.

The ci/quality/architecture separation means:
- Failure in a quality check (e.g., coverage dip) doesn't block
  fast lint/type feedback from ci.yml
- Architecture freeze violations are independently enforceable
  from correctness failures
- If future maintainers disable portions of CI, architecture
  protections remain intact

## Verification

- [x] dev branch created
- [x] Branch model documented
- [x] Protection rules specified for main, dev, feature/*
- [x] Merge path validated
- [x] Required checks defined for main
- [x] Break-glass path documented (admin bypass)
