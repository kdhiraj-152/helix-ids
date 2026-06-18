# Security Findings

## Finding C-01: `eval()` of JSON Architecture String

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Category** | Security |
| **Evidence** | `scripts/evaluation/benchmark_e2e.py:53`: `hidden_dims = eval(card["architecture"])` |
| **Risk** | Arbitrary code execution if `card["architecture"]` contains user-controlled input. `eval()` of parsed JSON allows executing arbitrary Python expressions. |
| **Remediation** | Replace `eval()` with `ast.literal_eval()` for safe evaluation of Python literals. If the architecture is a list, `json.loads()` is even safer. |
| **Effort** | 30 minutes — single-line change + test update |
| **Status** | UNRESOLVED |

## Finding C-02: `pickle.load` with `weights_only=False`

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Category** | Security |
| **Evidence** | `src/helix_ids/models/adaptation/transfer_learning.py:1185`: `torch.load(f, weights_only=False)` |
| **Risk** | Arbitrary code execution during model loading. `weights_only=False` allows pickle to execute arbitrary code from a crafted checkpoint. |
| **Remediation** | Change to `weights_only=True`. If the checkpoint contains optimizer state (non-tensor data), restructure to store optimizer state separately or use safetensors. Multiple test files also use `weights_only=False` — fix those too. |
| **Effort** | 1-2 hours — needs verification that checkpoint structure supports restricted loading |
| **Status** | UNRESOLVED |

## Finding C-03: `pickle.load` in Test Code

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Category** | Security |
| **Evidence** | `tests/test_callbacks.py:339`, `tests/test_fault_injection.py:306`, `tests/test_export_quantization_deployment.py:243` |
| **Risk** | Test code deserializing untrusted artifacts. Lower severity because tests load controlled fixtures, but establishes a pattern that could be copied to production. |
| **Remediation** | Change to `weights_only=True` in all test `torch.load()` calls. Replace `pickle.load(f)` with safe alternative where applicable. |
| **Effort** | 1 hour |
| **Status** | UNRESOLVED |

## Finding C-04: Assert Statements Disabled Under `-O`

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Category** | Security/Reliability |
| **Evidence** | `scripts/training/core/trainer_facade.py:133-218`: 21 sequential `assert` statements guarding against None manager objects. `scripts/deployment/deploy.py:147-230`: 6 assertions |
| **Risk** | Python's `assert` statements are silently removed when the interpreter runs with `-O` (optimized) flag. In production Docker containers, `python -O` is common. Critical guard logic vanishes without warning. |
| **Remediation** | Replace ALL production `assert` preconditions with explicit `if x is None: raise ValueError(...)` guards. Reserve `assert` for test-only invariants. |
| **Effort** | 2-3 hours across ~40 sites in 8 files |
| **Status** | UNRESOLVED |

## Finding C-05: SSRF Risk via `urlopen` with `# nosec`

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Category** | Security |
| **Evidence** | `scripts/operations/staging_gate_check.py:30`: `urlopen(url, ...)` with comment `# nosec B310`. `scripts/operations/traffic_expansion_guard.py:19`: same pattern |
| **Risk** | Server-Side Request Forgery if the URL parameter is user-supplied. The `# nosec` suppression disables bandit's warning without addressing the underlying risk. |
| **Remediation** | Add URL allow-list validation before making requests. Only allow known endpoint patterns (e.g., local Prometheus server). Remove `# nosec` suppression. |
| **Effort** | 2 hours |
| **Status** | UNRESOLVED |

## Finding C-06: Secrets Pattern in Session Logs

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Category** | Security |
| **Evidence** | `session_logs/` directory contains chat logs mentioning `API_KEY`, `HF_TOKEN`, etc. — instructional messages, not actual credentials |
| **Risk** | Low, but session logs in VCS can contain accidental credential exposure from future conversations |
| **Remediation** | Add `session_logs/` to `.gitignore` or move to a non-repo location. Consider using `.gitignore` entry. |
| **Effort** | 10 minutes |
| **Status** | UNRESOLVED |

## Finding C-07: Insufficient File Permission Hardening

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Category** | Security |
| **Evidence** | `Dockerfile` and `.dockerignore` set to 600 (owner-only) ✅. Most other files at 644. No world-writable files. |
| **Risk** | No immediate risk. Permissions are standard for a development repository. |
| **Remediation** | Consider tightening permissions on config files containing service endpoints. Not urgent. |
| **Effort** | 30 minutes |
| **Status** | ACCEPTABLE — no action required |

## Finding C-08: Dependency CVEs (Not Scanned)

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Category** | Security |
| **Evidence** | `pip-audit` timed out during this audit. No automated CVE scanning integrated into CI. |
| **Risk** | Unknown CVEs in 18 outdated packages (chardet 5.2→7.4 is a major version gap). No vulnerability database check is performed. |
| **Remediation** | Integrate `pip-audit` or `safety` into CI pipeline. Add `pip-audit` step to CI workflow. Pin versions in lockfiles. |
| **Effort** | 2 hours |
| **Status** | UNRESOLVED |

## Security Scorecard

| Check | Status | Note |
|-------|--------|------|
| Secrets in repo | ✅ PASS | No credentials found |
| Unsafe deserialization | ❌ FAIL | 1 production + 3 test locations |
| eval() usage | ❌ FAIL | `benchmark_e2e.py:53` |
| Shell injection | ⚠️ WARNING | `subprocess.run` in 3+ scripts |
| Path traversal | ✅ PASS | URL allow-lists recommended |
| Temp file safety | ⚠️ WARNING | `NamedTemporaryFile(delete=False)` in tests |
| Logging leakage | ✅ PASS | No sensitive data logged |
| File permissions | ✅ PASS | Standard, well-configured |
| Assert safety | ❌ FAIL | 40+ production asserts disablable |
| Bare except | ✅ PASS | Governance tooling enforces |
| Docker hardening | ✅ PASS | Multi-stage, pinned digest |
| CVE scanning | ❌ FAIL | Not integrated in CI |
