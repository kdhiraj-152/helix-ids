# Branch Governance — Actions Performed

Date: 2026-06-18
Status: Complete ✓

## Summary

- Repository made public (by user)
- Branch protection fully applied for `main` and `dev`
- Squash-merge only on main
- 4 stale remote branches deleted (3 Dependabot, 1 old release)
- 2 branches remain: `main` (protected) + `dev` (light protection)
- Stale branches cleaned up (4 deleted)

## Branch Protection — main

| Setting | Value |
|---------|-------|
| Require pull request | ✓ |
| Required approvals | 1 |
| Dismiss stale reviews | ✓ |
| Required status checks | CI / ci, Quality Gates / quality, Architecture Lockdown / architecture_check |
| Require branches up to date | ✓ (strict) |
| Block force push | ✓ |
| Block deletion | ✓ |
| Admin bypass | ✓ (enforce_admins: false) |

## Branch Protection — dev

| Setting | Value |
|---------|-------|
| Allow force push | ✓ |
| Block deletion | ✓ |
| No other restrictions | — |

## Branches on Remote

- `main` — active, protected
- `dev` — active, lightly protected
- (4 stale branches deleted: 3 Dependabot, 1 old release)

## Repository Settings

| Setting | Value |
|---------|-------|
| Merge commits | ✗ |
| Squash merging | ✓ |
| Rebase merging | ✗ |

## Commands Executed

```bash
# Apply main protection
gh api --method PUT repos/kdhiraj-152/helix-ids/branches/main/protection \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["CI / ci", "Quality Gates / quality", "Architecture Lockdown / architecture_check"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON

# Apply dev protection
gh api --method PUT repos/kdhiraj-152/helix-ids/branches/dev/protection \
  --input - <<'JSON'
{
  "allow_force_pushes": true,
  "allow_deletions": false
}
JSON

# Delete stale branches
git push origin --delete dependabot/pip/cryptography-49.0.0
git push origin --delete dependabot/pip/misc-74dc46496d
git push origin --delete dependabot/pip/setuptools-82.0.1
git push origin --delete release/canonical-contract-freeze

# Set squash-merge only
gh api --method PATCH repos/kdhiraj-152/helix-ids \
  -f allow_merge_commit=false \
  -f allow_rebase_merge=false \
  -f allow_squash_merge=true
```
