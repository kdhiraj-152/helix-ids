#!/usr/bin/env python3
"""Generate SLSA provenance attestations for release artifacts.

Produces SLSA v1.0 predicate (https://slsa.dev/spec/v1.0/provenance) in
in-toto attestation format, compatible with cosign/sigstore verification.

Usage:
    python3 scripts/ci/generate_slsa_provenance.py <build-type> [output-dir]

Output:
    results/provenance/slsa-attestation.json — SLSA v1.0 provenance attestation
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BUILD_ARTIFACTS = [
    "requirements-lock.txt",
    "results/sbom/sbom.json",
    "results/checksums.sha256",
    "results/licenses/licenses.json",
    "results/licenses/licenses.csv",
]

BUILD_CONFIG = {
    "commit": os.environ.get("GITHUB_SHA", "unknown"),
    "repository": os.environ.get("GITHUB_REPOSITORY", "kdhiraj/helix-ids"),
    "ref": os.environ.get("GITHUB_REF", "refs/tags/unknown"),
    "workflow": os.environ.get("GITHUB_WORKFLOW", "release-integrity"),
    "runner": os.environ.get("RUNNER_NAME", "unknown"),
    "run_id": os.environ.get("GITHUB_RUN_ID", "0"),
    "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def get_git_command(cmd: list[str]) -> str:
    """Run a git command and return output."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def generate_provenance(build_type: str) -> dict:
    """Generate SLSA v1.0 provenance attestation in in-toto format."""

    repo_uri = f"git+https://github.com/{BUILD_CONFIG['repository']}"

    # Build resolved dependencies from git
    git_sha = get_git_command(["git", "rev-parse", "HEAD"])
    git_tree = get_git_command(["git", "rev-parse", "HEAD:"])
    git_branch = get_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    subjects = []
    for artifact_path in BUILD_ARTIFACTS:
        p = Path(artifact_path)
        if p.exists():
            subjects.append({
                "name": artifact_path,
                "digest": {"sha256": sha256_file(p).replace("sha256:", "")},
            })

    now = datetime.now(timezone.utc)

    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": subjects,
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": f"https://github.com/{BUILD_CONFIG['repository']}/actions/{BUILD_CONFIG['workflow']}",
                "externalParameters": {
                    "buildType": build_type,
                    "git": {
                        "commit": git_sha or BUILD_CONFIG["commit"],
                        "repository": repo_uri,
                        "ref": BUILD_CONFIG["ref"],
                        "branch": git_branch,
                    },
                    "workflow": {
                        "run_id": BUILD_CONFIG["run_id"],
                        "run_attempt": BUILD_CONFIG["run_attempt"],
                        "runner": BUILD_CONFIG["runner"],
                    },
                },
                "internalParameters": {
                    "git_tree_hash": git_tree,
                },
                "resolvedDependencies": subjects.copy(),
            },
            "runDetails": {
                "builder": {
                    "id": f"https://github.com/{BUILD_CONFIG['repository']}/.github/workflows/{BUILD_CONFIG['workflow']}",
                },
                "metadata": {
                    "invocationId": f"{BUILD_CONFIG['repository']}/actions/runs/{BUILD_CONFIG['run_id']}/attempts/{BUILD_CONFIG['run_attempt']}",
                    "startedOn": now.isoformat(),
                    "finishedOn": now.isoformat(),
                },
            },
        },
    }


def main() -> int:
    args = sys.argv[1:]
    build_type = args[0] if args else "release"
    output_dir = Path(args[1]) if len(args) > 1 else Path("results/provenance")
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = generate_provenance(build_type)

    output_path = output_dir / "slsa-attestation.json"
    with open(output_path, "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"SLSA provenance attestation generated: {output_path}")
    print(f"  Build type: {build_type}")
    print(f"  Subjects: {len(provenance['subject'])} artifacts")
    print(f"  Predicate: {provenance['predicateType']}")

    # Also write a compact version for signing
    compact_path = output_dir / "slsa-attestation-compact.json"
    with open(compact_path, "w") as f:
        json.dump(provenance, f, separators=(",", ":"))
    print(f"Compact attestation: {compact_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
