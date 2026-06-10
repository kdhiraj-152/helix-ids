# AGENTS.md

Guidance for AI coding agents working in this repository.

## Mission Constraints

- Treat this repo as formalization mode: no new features, no new scripts, no broad refactors.
- Prefer minimal, targeted fixes in existing modules.
- Keep runtime pipeline behavior stable unless explicitly requested.

## Read First

- [README.md](README.md): project status, reproducibility path, staging artifacts.
- [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md): canonical package boundaries and model/runtime scope.
- [docs/operations/OPERATIONS_DEPLOYMENT_RUNBOOK.md](docs/operations/OPERATIONS_DEPLOYMENT_RUNBOOK.md): deployment gates, metrics, rollout guards.
- [scripts/README.md](scripts/README.md): script-domain layout and wrapper expectations.

## Environment And Commands

- Use `python3`.
- Activate the local venv when available: `source .venv311/bin/activate`.
- Use `PYTHONPATH=src` for script invocations to match documented project workflows.

Common validation commands:

```bash
pytest -q
pytest tests/test_operations -q
ruff check src scripts tests
mypy src
```

Notes:

- Pyright is configured with type checking disabled in [pyrightconfig.json](pyrightconfig.json).
- Pytest defaults and coverage behavior are configured in [pyproject.toml](pyproject.toml).

## Codebase Map

- Core package: [src/helix_ids](src/helix_ids)
- Operational scripts: [scripts](scripts)
  - training: [scripts/training](scripts/training)
  - operations: [scripts/operations](scripts/operations)
  - evaluation: [scripts/evaluation](scripts/evaluation)
  - data: [scripts/data](scripts/data)
  - deployment: [scripts/deployment](scripts/deployment)
- Tests: [tests](tests)

## Project-Specific Pitfalls

- Feature-space history is mixed across modules (legacy 32/41-feature paths and current harmonized/full-model paths). Confirm target module contracts before changing dimensions or schemas.
- Some architecture text includes legacy sections; treat current code under [src/helix_ids](src/helix_ids) and active scripts under [scripts](scripts) as source of truth.
- Service gating depends on `helix_coverage_override_rate` and `helix_degraded_state`; preserve the operational threshold behavior in [scripts/operations/serve_rest.py](scripts/operations/serve_rest.py) and [scripts/operations/staging_gate_check.py](scripts/operations/staging_gate_check.py).

## Code-Review-Graph Workflow (Default Post-Change Review)

Use `mcp_code-review-g` tools after non-trivial edits.

1. Initialize compact context:
   - `mcp_code-review-g_get_minimal_context_tool`
2. Build or refresh graph cache:
   - first time: `mcp_code-review-g_build_or_update_graph_tool(full_rebuild=true, postprocess="minimal")`
   - then enrich (optional): `mcp_code-review-g_build_or_update_graph_tool(postprocess="full")`
3. Get review guidance:
   - `mcp_code-review-g_get_review_context_tool`
4. Explore dependencies when needed:
   - `mcp_code-review-g_query_graph_tool` (`callers_of`, `callees_of`, `imports_of`, `tests_for`)

Repository-specific guidance:

- `.code-review-graph/` is intentionally gitignored in [.gitignore](.gitignore).
- If the worktree is noisy, pass explicit `changed_files` to review-context tools instead of relying on `HEAD~1` auto-diff.
- In this repo, structural graph metrics like communities/flows may be sparse; rely on minimal-context plus review-context outputs for actionable review decisions.

## Editing Expectations

- Preserve existing naming and script-domain placement conventions from [scripts/README.md](scripts/README.md).
- Keep patches small and test the nearest affected unit/integration tests first.
- Prefer linking existing docs instead of duplicating long procedural content in code comments or new docs.
