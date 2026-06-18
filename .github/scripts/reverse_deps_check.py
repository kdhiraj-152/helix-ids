#!/usr/bin/env python3
"""Check zero reverse dependencies (src -> scripts imports)."""
import ast
from pathlib import Path

src = Path('src/helix_ids')
project_root = Path('.')
offenders = []
for pyfile in sorted(src.rglob('*.py')):
    if pyfile.name == '__init__.py':
        continue
    try:
        tree = ast.parse(pyfile.read_text('utf-8'))
    except SyntaxError:
        continue
    targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets.add(node.module.split('.')[0])
    if 'scripts' in targets:
        rel = pyfile.relative_to(project_root).as_posix()
        offenders.append(rel)
if offenders:
    print(f'FAIL: Reverse deps found ({len(offenders)}):')
    for o in offenders:
        print(f'  {o}')
    exit(1)
print('PASS: Zero reverse dependencies (src -> scripts = 0)')
