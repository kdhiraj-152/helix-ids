# Branch Governance Final — Phase 24

Generated: 2026-06-18 | Status: RC3-ready

## Branch Model

```
main (production-ready, RC3)
  ^ PR (squash-merge, requires 1 approval)
  |
dev (integration branch)
  ^ push (fast-forward)
  |
feature/*, fix/* (topic branches)
```

## Protection Rules

### `main` branch

| Rule | Setting |
|------|---------|
| Require PR | Yes |
| Required approvals | 1 |
| Dismiss stale reviews | Yes |
| Require status checks | ci, architecture, quality, dependency-review |
| Require branches up-to-date | Yes |
| Include administrators | Yes |
| Allow force push | No |
| Allow deletions | No |
| Conversation resolution | Required |

### `dev` branch

| Rule | Setting |
|------|---------|
| Require PR | Recommended but not enforced |
| Allow force push | Yes (with caution) |
| Allow deletions | No |

## Workflow-to-Branch Mapping

| Workflow | Triggers on |
|----------|-------------|
| ci | push to dev, PR to main/dev |
| architecture | push to dev, PR to main |
| quality | push to dev, PR to main |
| nightly | main only (or workflow_dispatch) |
| release | tag push v* (or workflow_dispatch) |
| dependency-review | All PRs |

## Tag Strategy

- Release tags: `v<major>.<minor>.<patch>` (e.g., `v1.0.0-rc3`)
- Pre-release tags: `v<major>.<minor>.<patch>-rc<N>` (e.g., `v1.0.0-rc3`)
- Tags trigger the full `release.yml` pipeline (verify + sign + container)

## Recommendations

1. Enable branch protection rules in GitHub UI for `main`
2. Add `contents: read` permission block to ci.yml, architecture.yml, quality.yml
3. Consider adding CODEOWNERS file for `src/` and `scripts/` directories
