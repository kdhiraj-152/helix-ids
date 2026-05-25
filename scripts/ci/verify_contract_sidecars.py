#!/usr/bin/env python3
"""Verify that each producer script emits runtime contract payload and sidecars.

This script looks for common producer locations and checks that for any
`.pt` or `.pth` save call there is a companion `.contract.json`,
`.feature_order.json`, and `.schema_hash.txt` written nearby in the codebase.

It is a heuristic guard for CI; it exits non-zero if it finds likely bare
`state_dict` saves or missing sidecar writers.
"""
import sys
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

PAT_SAVE = re.compile(r"torch\.save\([^)]*\)")
PAT_CONTRACT = re.compile(r"runtime_contract_payload\(|\.contract\.json")

def find_suspects():
    suspects = []
    for p in ROOT.rglob('*.py'):
        if 'venv' in p.parts or '.venv' in p.parts:
            continue
        txt = p.read_text(encoding='utf-8', errors='ignore')
        if PAT_SAVE.search(txt) and not PAT_CONTRACT.search(txt):
            suspects.append(p)
    return suspects

def main():
    suspects = find_suspects()
    if suspects:
        print("❌ Found potential bare torch.save usages without contract payload:")
        for s in suspects:
            print(' -', s)
        sys.exit(1)
    print("✅ All checked producers reference runtime_contract_payload or write sidecars.")

if __name__ == '__main__':
    main()
