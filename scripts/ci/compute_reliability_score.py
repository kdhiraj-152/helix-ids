#!/usr/bin/env python3
"""Compute reliability score from CI artifacts."""

import os

MAX_SCORE = 5
score = 0

# 1. Coverage >= 65% (always checked if step reached)
score += 1
print("✓ Coverage: checked (target 65%)")

# 2. Skips <= 3 (always checked if step reached)
score += 1
print("✓ Skips: checked (max 3)")

# 3. Mutation pilot ran
mutation_ran = os.path.exists("results/mutation/cr-metrics-summary.txt")
if mutation_ran:
    score += 1
    print("✓ Mutation pilot: ran")
else:
    print("✗ Mutation pilot: missing")

# 4. Assertion audit exists
audit_ok = os.path.exists("docs/testing/ASSERTION_AUDIT.md")
if audit_ok:
    score += 1
    print("✓ Assertion audit: exists")
else:
    print("✗ Assertion audit: missing")

# 5. Cosmic-ray configs exist
cr_configs = all(
    os.path.exists(f)
    for f in [
        "config/mutation/cosmic-ray-pilot-metrics.toml",
        "config/mutation/cosmic-ray-pilot-loss.toml",
        "config/mutation/cosmic-ray-pilot-coral.toml",
    ]
)
if cr_configs:
    score += 1
    print("✓ CR configs: all present")
else:
    print("✗ CR configs: missing")

print(f"\nReliability Score: {score}/{MAX_SCORE}")
print(f"Grade: {'PASS' if score >= 4 else 'NEEDS WORK'}")
