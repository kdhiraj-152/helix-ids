#!/usr/bin/env python3
"""Check HelixFullTrainer size limits (LOC ≤ 2000, methods ≤ 100)."""
import ast
from pathlib import Path

root = Path('scripts/training')
f = root / 'train_helix_ids_full.py'
tree = ast.parse(f.read_text('utf-8'))
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'HelixFullTrainer':
        loc = node.end_lineno - node.lineno + 1
        print(f'HelixFullTrainer: {loc} LOC (gate: ≤ 2000)')
        assert loc <= 2000, f'FAIL: {loc} LOC exceeds freeze gate of 2000'
        methods = sum(1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        print(f'HelixFullTrainer: {methods} methods (gate: ≤ 100)')
        assert methods <= 100, f'FAIL: {methods} methods exceeds freeze gate of 100'
        break
print('PASS: Trainer size within freeze limits')
