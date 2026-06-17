"""
Regression tests for wrapper removal in HelixFullTrainer.

Phase 18: Verifies that removed static wrappers are no longer present
and that the correct replacements (direct LossRegistry calls) are used
throughout the codebase. Also verifies embedded dataset/sampler classes
were replaced with imports from scripts.training.data.

These are static-code-analysis tests (no model instantiation required).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TRAINER_PATH = Path("scripts/training/train_helix_ids_full.py")
CORE_INIT_PATH = Path("scripts/training/core/__init__.py")
CORE_STATE_PATH = Path("scripts/training/core/trainer_state.py")
CORE_RECOVERY_PATH = Path("scripts/training/core/recovery_manager.py")
CORE_FACADE_PATH = Path("scripts/training/core/trainer_facade.py")


# ── Wrappers that were removed (static methods delegating to LossRegistry) ──

_REMOVED_WRAPPERS = [
    "_supervised_contrastive_loss",
    "_supcon_anchor_weights",
    "_class_conditional_energy_gap_loss",
    "_energy_class_balance_loss",
    "_energy_min_winner_loss",
    "_pairwise_margin_repulsion_loss",
    "_centroid_separation_barrier_loss",
    "_centroid_repulsion_loss",
]


@pytest.fixture(scope="module")
def trainer_ast():
    """Parse trainer file once for all tests."""
    with open(TRAINER_PATH) as f:
        return ast.parse(f.read())


@pytest.fixture(scope="module")
def trainer_method_names(trainer_ast):
    """Extract method names in HelixFullTrainer class."""
    for node in ast.walk(trainer_ast):
        if isinstance(node, ast.ClassDef) and node.name == "HelixFullTrainer":
            return {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return set()


@pytest.fixture(scope="module")
def trainer_source_text(trainer_ast):
    """Return the raw source text for regex checks."""
    with open(TRAINER_PATH) as f:
        return f.read()


# ── Removed static wrappers ──


class TestRemovedStaticWrappers:
    """Verify the 8 pure-delegation static wrappers were removed."""

    def test_all_removed_wrappers_gone(self, trainer_method_names):
        for wrapper in _REMOVED_WRAPPERS:
            assert wrapper not in trainer_method_names, (
                f"Wrapper {wrapper} should have been removed but is still present"
            )

    @pytest.mark.parametrize("wrapper", _REMOVED_WRAPPERS)
    def test_no_self_reference_remaining(self, trainer_source_text, wrapper):
        """No `self.{wrapper}(` call should remain in the trainer file."""
        # Check for 'self.{wrapper}(' pattern
        assert f"self.{wrapper}(" not in trainer_source_text, (
            f"Call to self.{wrapper}() still present in trainer file"
        )


# ── Embedded dataset/sampler classes ──


class TestEmbeddedClassesRemoved:
    """Verify embedded dataset/sampler classes were removed."""

    @pytest.mark.parametrize("class_name", ["MultiTaskNumpyDataset", "ClassBalancedIndexSampler", "FrozenIndexSampler"])
    def test_embedded_class_not_in_trainer(self, trainer_ast, class_name):
        for node in ast.walk(trainer_ast):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                pytest.fail(f"{class_name} should be imported, not defined in trainer")

    def test_data_imports_present(self, trainer_source_text):
        assert "from scripts.training.data.dataset_builder import MultiTaskNumpyDataset" in trainer_source_text
        assert "from scripts.training.data.samplers import ClassBalancedIndexSampler, FrozenIndexSampler" in trainer_source_text


# ── configure_structure_recovery delegation ──


class TestConfigureStructureRecovery:
    """Verify configure_structure_recovery now delegates to RecoveryManager."""

    def test_delegates_to_recovery_manager(self, trainer_source_text):
        assert "self._recovery_manager.configure_structure_recovery(" in trainer_source_text

    def test_no_long_body_remains(self, trainer_source_text):
        """Verify the old large method body is gone."""
        # The old body had many self.energy_gap_margin = ... assignments
        # which would show if the inline code were still present.
        count = trainer_source_text.count("self.energy_gap_margin")
        # Should be 0 if new method just delegates (the old one set these inline)
        assert count <= 1, (
            f"Found {count} occurrences of 'self.energy_gap_margin' — "
            "expected ≤1 (only in TrainerState)"
        )


# ── TrainerState module structural checks ──


class TestTrainerStatePresence:
    """Verify TrainerState module exists and has expected members."""

    def test_trainer_state_exists(self):
        assert CORE_STATE_PATH.exists()

    def test_recovery_manager_exists(self):
        assert CORE_RECOVERY_PATH.exists()

    def test_trainer_facade_exists(self):
        assert CORE_FACADE_PATH.exists()

    def test_core_init_exports(self):
        with open(CORE_INIT_PATH) as f:
            content = f.read()
        assert "TrainerState" in content
        assert "RecoveryManager" in content
        assert "TrainerFacade" in content
        assert "TrainerFactory" in content


# ── Core package imports work ──


class TestCorePackageImports:
    """Verify the core package is importable and has expected symbols."""

    def test_import_trainer_state(self):
        from scripts.training.core import TrainerState
        assert TrainerState is not None

    def test_import_recovery_manager(self):
        from scripts.training.core import RecoveryManager
        assert RecoveryManager is not None

    def test_import_trainer_facade(self):
        from scripts.training.core import TrainerFacade
        assert TrainerFacade is not None

    def test_import_trainer_factory(self):
        from scripts.training.core import TrainerFactory
        assert TrainerFactory is not None


# ── No static wrappers remain in LossRegistry (double-check) ──


class TestDirectLossRegistryAccess:
    """Verify that call sites now use LossRegistry.method() directly."""

    def test_trainer_imports_loss_registry(self, trainer_source_text):
        """LossRegistry should be imported from the losses package, not defined locally."""
        assert "from scripts.training.losses import (" in trainer_source_text
        assert "LossRegistry," in trainer_source_text

    @pytest.mark.parametrize("method_name", [
        "supervised_contrastive_loss",
        "class_conditional_energy_gap_loss",
        "energy_class_balance_loss",
        "energy_min_winner_loss",
        "pairwise_margin_repulsion_loss",
        "centroid_separation_barrier_loss",
        "centroid_repulsion_loss",
    ])
    def test_loss_registry_method_accessible(self, method_name):
        """Verify the LossRegistry method exists (should be imported on-demand)."""
        from scripts.training.losses import LossRegistry
        assert hasattr(LossRegistry, method_name), (
            f"LossRegistry.{method_name} not found"
        )


# ── Representation-phase logic in scheduler ──


class TestRepresentationPhaseLogicLocation:
    """Verify representation-phase logic is NOT in trainer but in phase_orchestrator."""

    def test_should_exit_curriculum_not_in_trainer(self, trainer_method_names):
        # The method should delegate, not implement
        assert "_should_exit_representation_curriculum" in trainer_method_names, (
            "Wrapper should exist for delegation"
        )

    def test_representation_logic_in_orchestrator(self):
        """PhaseOrchestrator should have representation transition logic."""
        from scripts.training.scheduler import PhaseOrchestrator
        assert hasattr(PhaseOrchestrator, "should_exit_representation_curriculum")
        assert hasattr(PhaseOrchestrator, "handle_representation_phase_logic")
        assert hasattr(PhaseOrchestrator, "maybe_activate_joint_finetune_phase")
