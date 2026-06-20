#!/usr/bin/env python3
"""
Comprehensive Architecture Audit - Part A
Dependency graph, cycle detection, size analysis.
Outputs structured JSON to stdout.
"""

import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/Users/kdhiraj/Downloads/RP-2").resolve()

EXCLUDE_DIRS = {
    ".venv311", ".git", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".hypothesis",
}

def is_excluded(p: Path) -> bool:
    for part in p.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False

def short_path(p: Path) -> str:
    return str(p.relative_to(PROJECT_ROOT))

def find_py_files() -> list[Path]:
    result = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        root_p = Path(root)
        if is_excluded(root_p):
            continue
        for f in files:
            if f.endswith(".py"):
                fp = root_p / f
                if not is_excluded(fp):
                    result.append(fp)
    return sorted(result)

def resolve_import(module_name: str, file_path: Path) -> str | None:
    parts = module_name.split(".")
    candidates = []
    for i in range(len(parts), 0, -1):
        prefix = parts[:i]
        suffix = parts[i:]
        dir_path = PROJECT_ROOT.joinpath(*prefix)
        if dir_path.is_dir():
            init_file = dir_path / "__init__.py"
            if init_file.exists():
                if suffix:
                    remaining = list(suffix)
                    file_candidate = dir_path.joinpath(*remaining).with_suffix(".py")
                    candidates.append(file_candidate)
                else:
                    candidates.append(init_file)
        file_path_candidate = PROJECT_ROOT.joinpath(*prefix).with_suffix(".py")
        candidates.append(file_path_candidate)
    for cand in candidates:
        try:
            cand = cand.resolve()
            if cand.exists() and str(cand).startswith(str(PROJECT_ROOT)):
                return short_path(cand)
        except (ValueError, OSError):
            continue
    return None

def analyze_file(file_path: Path) -> dict:
    with open(file_path, encoding="utf-8", errors="replace") as f:
        source = f.read()
    lines = source.split("\n")
    loc = len(lines)
    stripped_lines = [l.strip() for l in lines]
    blank_lines = sum(1 for l in stripped_lines if l == "")
    code_loc = loc - blank_lines
    tree = None
    parse_error = None
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        parse_error = str(e)
    imports = []
    function_defs = []
    class_defs = []
    top_level_assignments = []
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "level": level,
                        "line": node.lineno,
                    })
            if isinstance(node, ast.FunctionDef):
                function_defs.append({
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "total_lines": node.end_lineno - node.lineno + 1 if node.end_lineno else 0,
                })
            elif isinstance(node, ast.AsyncFunctionDef):
                function_defs.append({
                    "name": f"async {node.name}",
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "total_lines": node.end_lineno - node.lineno + 1 if node.end_lineno else 0,
                })
            elif isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                class_defs.append({
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "total_lines": node.end_lineno - node.lineno + 1 if node.end_lineno else 0,
                    "methods": methods,
                })
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        top_level_assignments.append({"name": target.id, "line": node.lineno})
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                top_level_assignments.append({"name": elt.id, "line": node.lineno})
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    top_level_assignments.append({"name": node.target.id, "line": node.lineno})
    return {
        "file": short_path(file_path),
        "loc": loc,
        "code_loc": code_loc,
        "blank_lines": blank_lines,
        "parse_error": parse_error,
        "imports": imports,
        "function_defs": function_defs,
        "class_defs": class_defs,
        "top_level_assignments": top_level_assignments,
        "num_imports": len(imports),
        "num_functions": len(function_defs),
        "num_classes": len(class_defs),
        "num_top_level_assignments": len(top_level_assignments),
    }

def build_dependency_graph(file_infos: list[dict]) -> dict:
    graph = defaultdict(list)
    reverse_map = defaultdict(list)
    for fi in file_infos:
        fpath = fi["file"]
        resolved_set = set()
        for imp in fi["imports"]:
            if imp["type"] == "import":
                module = imp["module"]
            else:
                level = imp.get("level", 0)
                if level > 0:
                    abs_path = PROJECT_ROOT / fpath
                    rel_to_project = abs_path.relative_to(PROJECT_ROOT)
                    rel_parent = rel_to_project.parent
                    if level > len(rel_parent.parts):
                        continue
                    base = Path(*rel_parent.parts[:-level]) if level > 0 else rel_parent
                    mod_path = imp["module"]
                    if mod_path:
                        full_mod_path = base / mod_path.replace(".", "/")
                    else:
                        full_mod_path = base
                    cand_paths = []
                    cand_paths.append(PROJECT_ROOT / full_mod_path.with_suffix(".py"))
                    cand_paths.append(PROJECT_ROOT / full_mod_path / "__init__.py")
                    resolved = None
                    for cp in cand_paths:
                        if cp.exists() and str(cp).startswith(str(PROJECT_ROOT)):
                            resolved = short_path(cp)
                            break
                    if resolved and resolved not in resolved_set:
                        resolved_set.add(resolved)
                        graph[fpath].append((resolved, imp))
                        reverse_map[resolved].append((fpath, imp))
                    continue
                else:
                    module = imp["module"]
            resolved = resolve_import(module, Path(PROJECT_ROOT) / fpath)
            if resolved and resolved not in resolved_set:
                resolved_set.add(resolved)
                graph[fpath].append((resolved, imp))
                reverse_map[resolved].append((fpath, imp))
    for fpath in list(graph.keys()):
        graph[fpath] = [(r, i) for r, i in graph[fpath] if r != fpath]
    for fpath in list(reverse_map.keys()):
        reverse_map[fpath] = [(r, i) for r, i in reverse_map[fpath] if r != fpath]
    fan_out = {}
    for fpath, edges in graph.items():
        unique_targets = set(r for r, _ in edges)
        fan_out[fpath] = len(unique_targets)
    fan_in = {}
    for fpath, edges in reverse_map.items():
        unique_sources = set(r for r, _ in edges)
        fan_in[fpath] = len(unique_sources)
    return {"graph": dict(graph), "reverse_map": dict(reverse_map), "fan_out": fan_out, "fan_in": fan_in}

def detect_cycles(graph: dict) -> list[list[str]]:
    from collections import defaultdict
    adj = defaultdict(list)
    all_nodes = set()
    for src, targets in graph.items():
        all_nodes.add(src)
        for tgt, _ in targets:
            adj[src].append(tgt)
            all_nodes.add(tgt)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(all_nodes, WHITE)
    path = []
    cycles = []
    def dfs(u):
        color[u] = GRAY
        path.append(u)
        for v in adj[u]:
            if v not in color:
                color[v] = WHITE
            if color[v] == GRAY:
                cycle_start_idx = path.index(v)
                cycle = path[cycle_start_idx:]
                cycles.append(list(cycle))
            elif color[v] == WHITE:
                dfs(v)
        path.pop()
        color[u] = BLACK
    for node in sorted(all_nodes):
        if color.get(node, WHITE) == WHITE:
            dfs(node)
    return cycles

def find_large_items(file_infos: list[dict]) -> dict:
    large_files = []
    large_classes = []
    large_functions = []
    for fi in file_infos:
        if fi["loc"] > 1000:
            large_files.append({"file": fi["file"], "loc": fi["loc"], "code_loc": fi["code_loc"]})
        for cls in fi.get("class_defs", []):
            if cls["total_lines"] > 500:
                large_classes.append({"file": fi["file"], "class_name": cls["name"], "start_line": cls["start_line"], "end_line": cls["end_line"], "total_lines": cls["total_lines"]})
        for func in fi.get("function_defs", []):
            if func["total_lines"] > 100:
                large_functions.append({"file": fi["file"], "function_name": func["name"], "start_line": func["start_line"], "end_line": func["end_line"], "total_lines": func["total_lines"]})
    large_files.sort(key=lambda x: x["loc"], reverse=True)
    large_classes.sort(key=lambda x: x["total_lines"], reverse=True)
    large_functions.sort(key=lambda x: x["total_lines"], reverse=True)
    return {"large_files": large_files, "large_classes": large_classes, "large_functions": large_functions}

def main():
    py_files = find_py_files()
    file_infos = []
    errors = []
    for fp in py_files:
        try:
            info = analyze_file(fp)
            file_infos.append(info)
        except Exception as e:
            errors.append({"file": short_path(fp), "error": str(e)})
    dep_info = build_dependency_graph(file_infos)
    cycles = detect_cycles(dep_info["graph"])
    deduped = set()
    unique_cycles = []
    for cycle in cycles:
        if len(cycle) < 2:
            continue
        min_idx = cycle.index(min(cycle))
        normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
        if normalized not in deduped:
            deduped.add(normalized)
            unique_cycles.append(list(normalized))
    large_items = find_large_items(file_infos)
    fan_in_sorted = sorted(dep_info["fan_in"].items(), key=lambda x: x[1], reverse=True)
    fan_out_sorted = sorted(dep_info["fan_out"].items(), key=lambda x: x[1], reverse=True)
    top_15_fan_in = [{"file": f, "count": c} for f, c in fan_in_sorted[:15]]
    top_15_fan_out = [{"file": f, "count": c} for f, c in fan_out_sorted[:15]]
    total_loc = sum(fi["loc"] for fi in file_infos)
    total_code_loc = sum(fi["code_loc"] for fi in file_infos)
    total_functions = sum(fi["num_functions"] for fi in file_infos)
    total_classes = sum(fi["num_classes"] for fi in file_infos)
    total_imports = sum(fi["num_imports"] for fi in file_infos)
    total_assignments = sum(fi["num_top_level_assignments"] for fi in file_infos)
    result = {
        "meta": {"project_root": str(PROJECT_ROOT), "total_files": len(file_infos), "parse_errors": len(errors), "parse_error_details": errors},
        "summary_stats": {"total_loc": total_loc, "total_code_loc": total_code_loc, "total_blank_lines": total_loc - total_code_loc, "total_functions": total_functions, "total_classes": total_classes, "total_imports": total_imports, "total_top_level_assignments": total_assignments, "total_graph_nodes": len(dep_info["graph"]), "total_graph_edges": sum(len(v) for v in dep_info["graph"].values()), "total_reverse_map_keys": len(dep_info["reverse_map"]), "total_cycles_found": len(unique_cycles)},
        "file_analysis": file_infos,
        "dependency_graph": {"edges": {k: [t for t, _ in v] for k, v in dep_info["graph"].items()}},
        "reverse_dependency_map": {k: [s for s, _ in v] for k, v in dep_info["reverse_map"].items()},
        "fan_analysis": {"top_15_fan_in": top_15_fan_in, "top_15_fan_out": top_15_fan_out, "all_fan_in": dict(sorted(dep_info["fan_in"].items(), key=lambda x: x[1], reverse=True)), "all_fan_out": dict(sorted(dep_info["fan_out"].items(), key=lambda x: x[1], reverse=True))},
        "cycles": {"total_unique_cycles": len(unique_cycles), "cycles": unique_cycles},
        "large_items": large_items,
    }
    json.dump(result, sys.stdout, indent=2, default=str)
    print()

if __name__ == "__main__":
    main()
