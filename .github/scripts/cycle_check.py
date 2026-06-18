#!/usr/bin/env python3
"""Check zero package-level cycles between helix_ids and scripts.training."""
import ast
from pathlib import Path

project_root = Path('.')
boundaries = {
    'src/helix_ids': 'helix_ids',
    'scripts/training': 'scripts.training',
}
boundary_by_prefix = {str(project_root / d): label for d, label in boundaries.items()}
edges = []
for dirpath_str, label in boundary_by_prefix.items():
    dirpath = Path(dirpath_str)
    if not dirpath.is_dir():
        continue
    for pyfile in sorted(dirpath.rglob('*.py')):
        if any(p.startswith('.') for p in pyfile.parent.relative_to(dirpath).parts):
            continue
        try:
            tree = ast.parse(pyfile.read_text('utf-8'))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mod_name = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_name = alias.name
                    break
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod_name = node.module
            if mod_name:
                mod_path = mod_name.replace('.', '/')
                for _dp, lbl in boundary_by_prefix.items():
                    boundary_top = lbl.split('.')[0]
                    if mod_path.split('/')[0] == boundary_top:
                        if lbl != label:
                            edges.append((label, lbl))
                        break
labels = list(boundaries.values())
adj = {lbl: [] for lbl in labels}
for src, tgt in edges:
    adj[src].append(tgt)
WHITE, GRAY, BLACK = 0, 1, 2
color = dict.fromkeys(labels, WHITE)
def dfs(node, path=None):
    if path is None:
        path = []
    color[node] = GRAY
    path.append(node)
    for neighbor in adj.get(node, []):
        if color[neighbor] == GRAY:
            cycle_start = path.index(neighbor)
            return path[cycle_start:] + [neighbor]
        if color[neighbor] == WHITE:
            result = dfs(neighbor, path)
            if result:
                return result
    path.pop()
    color[node] = BLACK
    return None
for lbl in labels:
    if color[lbl] == WHITE:
        cycle = dfs(lbl)
        if cycle:
            print(f'FAIL: Cycle detected: {" -> ".join(cycle)}')
            exit(1)
print('PASS: Zero package-level cycles')
