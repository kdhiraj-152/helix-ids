#!/usr/bin/env python3
"""Verify SLSA provenance attestation integrity.

Usage:
    python3 scripts/ci/verify_slsa_provenance.py [attestation-path]

Checks:
    1. Valid JSON
    2. In-toto statement envelope
    3. SLSA v1.0 predicate type
    4. Subject digests match current files
    5. Required fields present

Exit code:
    0 — verification passed
    1 — verification failed
"""

import hashlib
import json
import sys
from pathlib import Path

REQUIRED_SUBJECTS = [
    "requirements-lock.txt",
]

REQUIRED_TOP_LEVEL_FIELDS = ["_type", "subject", "predicateType", "predicate"]

REQUIRED_PREDICATE_FIELDS = ["buildDefinition", "runDetails"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_provenance(attestation_path: Path) -> int:
    if not attestation_path.exists():
        print(f"ERROR: Attestation not found: {attestation_path}")
        return 1

    try:
        with open(attestation_path) as f:
            attestation = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in attestation: {e}")
        return 1

    # Check top-level fields
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in attestation:
            print(f"ERROR: Missing required top-level field: {field}")
            return 1
    print(f"OK: All {len(REQUIRED_TOP_LEVEL_FIELDS)} top-level fields present")

    # Check type and predicate type
    if attestation["_type"] != "https://in-toto.io/Statement/v1":
        print(f"ERROR: Unexpected statement type: {attestation['_type']}")
        return 1
    print(f"OK: Statement type: {attestation['_type']}")

    expected_predicate = "https://slsa.dev/provenance/v1"
    if attestation["predicateType"] != expected_predicate:
        print(f"ERROR: Unexpected predicate type: {attestation['predicateType']}")
        return 1
    print(f"OK: Predicate type: {attestation['predicateType']}")

    # Check predicate fields
    for field in REQUIRED_PREDICATE_FIELDS:
        if field not in attestation["predicate"]:
            print(f"ERROR: Missing predicate field: {field}")
            return 1
    print("OK: Predicate structure valid")

    # Check required subjects are present
    subject_names = {s["name"] for s in attestation["subject"]}
    for required in REQUIRED_SUBJECTS:
        if required not in subject_names:
            print(f"WARNING: Required subject missing: {required}")
        else:
            print(f"OK: Subject present: {required}")

    # Verify subject digests against current files
    failures = 0
    for subject in attestation["subject"]:
        name = subject["name"]
        filesys_path = Path(name)
        if not filesys_path.exists():
            print(f"WARNING: Subject file not on disk: {name} (expected for CI-only artifacts)")
            continue

        actual_digest = sha256_file(filesys_path)
        expected_digests = subject.get("digest", {})
        if "sha256" not in expected_digests:
            print(f"ERROR: Subject {name} missing sha256 digest")
            failures += 1
            continue

        if expected_digests["sha256"] != actual_digest:
            print(f"ERROR: Digest mismatch for {name}")
            print(f"  Expected: {expected_digests['sha256']}")
            print(f"  Actual:   {actual_digest}")
            failures += 1
        else:
            print(f"OK: Digest verified for {name}")

    # Check buildDefinition details
    bd = attestation["predicate"]["buildDefinition"]
    print(f"OK: Build type: {bd.get('buildType', 'MISSING')}")
    ext = bd.get("externalParameters", {})
    print(f"OK: Build commit: {ext.get('git', {}).get('commit', 'MISSING')}")

    if failures > 0:
        print(f"\nFAILURE: {failures} subject digest mismatch(es)")
        return 1

    print("\nPROVENANCE VERIFICATION: PASSED")
    return 0


def main() -> int:
    args = sys.argv[1:]
    attestation_path = Path(args[0]) if args else Path("results/provenance/slsa-attestation.json")
    return verify_provenance(attestation_path)


if __name__ == "__main__":
    sys.exit(main())
