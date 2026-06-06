from __future__ import annotations

from pathlib import Path

import pytest

from helix_ids.governance.ast_validator import ASTValidatorConfig, main, validate_paths


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_allowlisted_derivation_allows_fillna(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "feature_engineering.py",
        "import pandas as pd\n"
        "df = pd.DataFrame()\n"
        "df.fillna(0)\n",
    )
    violations = validate_paths([path])
    assert not any(violation.rule_id == "GOV001" for violation in violations)


def test_forbidden_fillna_rejected_outside_allowlist(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "runtime_module.py",
        "import pandas as pd\n"
        "df = pd.DataFrame()\n"
        "df.fillna(0)\n",
    )
    violations = validate_paths([path], config=ASTValidatorConfig(allowlist_suffixes=()))
    assert any(violation.rule_id == "GOV001" for violation in violations)


def test_forbidden_fallback_rejected_in_sensitive_paths(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "runtime" / "runtime_module.py",
        "import warnings\n"
        "warnings.warn('fallback')\n",
    )
    violations = validate_paths([path])
    assert any(violation.rule_id == "GOV010" for violation in violations)


def test_forbidden_torch_save_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "export" / "exporter.py",
        "import torch\n"
        "state_dict = {}\n"
        "torch.save(state_dict, 'artifact.pt')\n",
    )
    violations = validate_paths([path])
    assert any(violation.rule_id == "GOV021" for violation in violations)


def test_runtime_schema_mutation_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "inference" / "runtime.py",
        "df = None\n"
        "df.columns = ['a']\n",
    )
    violations = validate_paths([path])
    assert any(violation.rule_id == "GOV030" for violation in violations)


def test_pickle_dump_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "runtime" / "serializer.py",
        "import pickle\n"
        "pickle.dump({'a': 1}, open('artifact.bin', 'wb'))\n",
    )
    violations = validate_paths([path])
    assert any(violation.rule_id == "GOV022" for violation in violations)


def test_joblib_dump_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "runtime" / "serializer.py",
        "import joblib\n"
        "joblib.dump({'a': 1}, 'artifact.bin')\n",
    )
    violations = validate_paths([path])
    assert any(violation.rule_id == "GOV022" for violation in violations)


def test_approved_serialization_module_allows_wiring(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "export" / "exporter.py",
        "import torch\n"
        "from helix_ids.governance.provenance import build_artifact_manifest\n"
        "manifest = build_artifact_manifest(model_architecture='x')\n"
        "torch.save({'weights': 1}, 'artifact.pt')\n",
    )
    config = ASTValidatorConfig(approved_serialization_suffixes=("export/exporter.py",))
    violations = validate_paths([path], config=config)
    assert not any(violation.rule_id in {"GOV020", "GOV021"} for violation in violations)


def test_ci_mode_fail_fast(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "runtime" / "module.py",
        "import pandas as pd\n"
        "df = pd.DataFrame()\n"
        "df.fillna(0)\n",
    )
    exit_code = main(["--paths", str(path), "--ci", "--json"])
    assert exit_code == 1
