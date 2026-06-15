"""
Regression test: detect conflicting THREAT_WEIGHTS definitions.

If any module in helix_ids defines its own threat-weight dict that
differs from the canonical source (models.loss), this test fails.
New threat weights must be derived from or imported from the canonical source.
"""

import importlib
import pkgutil
import ast
import os
from pathlib import Path


def _find_threat_weight_definitions():
    """
    Walk every Python module under src/helix_ids/ and find all
    module-level assignments to THREAT_WEIGHTS or DEFAULT_THREAT_WEIGHTS
    that are literal dict/tensor definitions (not imports).
    """
    src_root = Path(__file__).parent.parent / "src"
    canonical = "helix_ids.models.loss"
    offenders = []

    for root, dirs, files in os.walk(src_root / "helix_ids"):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            # Build the module name relative to src
            rel = os.path.relpath(path, src_root).replace(os.sep, "/").replace(".py", "")
            # Remove /__init__ suffix
            if rel.endswith("/__init__"):
                rel = rel[: -len("/__init__")] or "helix_ids"
            mod = rel.replace("/", ".")

            with open(path) as fh:
                source = fh.read()

            tree = ast.parse(source)

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in (
                            "THREAT_WEIGHTS",
                            "DEFAULT_THREAT_WEIGHTS",
                        ):
                            # Skip if it's an import (not a literal definition)
                            if _is_dict_or_tensor_literal(node.value):
                                offenders.append((mod, target.id, node.lineno))

    return offenders, canonical


def _is_dict_or_tensor_literal(node):
    """Check if an AST node is a dict literal or torch.tensor(...) call."""
    if isinstance(node, ast.Dict):
        return True
    # torch.tensor([...]) — skip if it's just a tensor of canonical values
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "tensor":
            return True
        if isinstance(func, ast.Name) and func.id == "tensor":
            return True
    return False


def test_no_conflicting_threat_weights():
    """Fail if any module redefines THREAT_WEIGHTS as a literal instead of importing."""
    offenders, canonical = _find_threat_weight_definitions()
    # The canonical module (models.loss) is allowed to define both
    allowed = {
        (canonical, "THREAT_WEIGHTS"),
        (canonical, "DEFAULT_THREAT_WEIGHTS"),
    }
    # utils/export.py defines DEFAULT_THREAT_WEIGHTS as a 7-class weight dict
    # (Normal, DoS, Probe, R2L, U2R, Generic, Backdoor) used ONLY for ONNX export
    # metadata. The 5-class overlap values are intentionally distinct because
    # the 7-class model has a different classification schema. This is not a
    # training-weight duplicate — it is export-only metadata about a combined model.
    allowed.add(("helix_ids.utils.export", "DEFAULT_THREAT_WEIGHTS"))
    unexpected = [
        (mod, name, line)
        for (mod, name, line) in offenders
        if (mod, name) not in allowed
    ]

    if unexpected:
        msg_parts = [
            f"Found {len(unexpected)} non-canonical THREAT_WEIGHTS definition(s):",
        ]
        for mod, name, line in unexpected:
            msg_parts.append(f"  {mod}:{line} — defines {name} as a literal")
        msg_parts.append(
            f"\nCanonical source is {canonical}. "
            f"Import instead of redefining."
        )
        assert False, "\n".join(msg_parts)
