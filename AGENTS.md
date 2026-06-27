# AGENTS.md

Guidance for AI coding agents working in this repository.

## Mission Constraints

- Treat this repo as formalization mode: no new features, no new scripts, no broad refactors.
- Prefer minimal, targeted fixes in existing modules.
- Keep runtime pipeline behavior stable unless explicitly requested.

## Read First

- [README.md](README.md): project overview, quick start, install, train, evaluate, deploy.
- [docs/architecture/SYSTEM_ARCHITECTURE.md](docs/architecture/SYSTEM_ARCHITECTURE.md): canonical package boundaries, model/runtime scope, governance, recovery.
- [docs/development/TESTING.md](docs/development/TESTING.md): test organization, types, coverage expectations, CI gates.
- [docs/development/CODING_STANDARDS.md](docs/development/CODING_STANDARDS.md): code style, linting, type annotations, forbidden patterns.
- [docs/operations/DEPLOYMENT.md](docs/operations/DEPLOYMENT.md): deployment stages, gate criteria, runbook commands.

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
  - benchmarks: [scripts/benchmarks](scripts/benchmarks)
  - ci: [scripts/ci](scripts/ci)
- Tests: [tests](tests)
- Documentation: [docs](docs)
  - architecture: system design, data flow, governance, decisions
  - development: testing, coding standards, contributing, release process
  - operations: deployment, monitoring, recovery, soak testing
  - api: CLI & REST API reference
  - reports: RC3 readiness verdict, audit baseline
  - changelog: phase history
  - manuscript: paper drafts
  - figures: paper figures
  - archive: historical phase documentation
  - final: publication paper draft and supporting docs
  - redteam: red/blue team security audits
  - releases: phase certification reports
  - phase31-43h: intermediate research phase documentation

## Project-Specific Pitfalls

- Feature-space history is mixed across modules (legacy 32/41-feature paths and current harmonized/full-model paths). Confirm target module contracts before changing dimensions or schemas. The canonical input dim is **17** features (not 41).
- Some architecture text includes legacy sections; treat current code under [src/helix_ids](src/helix_ids) and active scripts under [scripts](scripts) as source of truth.
- Service gating depends on `helix_coverage_override_rate` and `helix_degraded_state`; preserve the operational threshold behavior in [scripts/operations/serve_rest.py](scripts/operations/serve_rest.py) and [scripts/operations/staging_gate_check.py](scripts/operations/staging_gate_check.py).
- Training produces governed checkpoints only — single-seed runs are research-only and not deployable.
- The lockfile (`requirements-lock.txt`) must be synchronized with `requirements.in` — CI enforces this.

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

## Documentation Authority

The following files are single sources of truth:

| Topic | Document |
|-------|----------|
| System design | `docs/architecture/SYSTEM_ARCHITECTURE.md` |
| Testing | `docs/development/TESTING.md` |
| Deployment | `docs/operations/DEPLOYMENT.md` |
| Governance | `docs/architecture/GOVERNANCE.md` |
| API reference | `docs/api/API_REFERENCE.md` |

Historical documentation lives in `docs/archive/` and is not authoritative.
Intermediate research phase docs (`docs/phase31/`–`docs/phase43h/`) are experiment records, not curriculum.
