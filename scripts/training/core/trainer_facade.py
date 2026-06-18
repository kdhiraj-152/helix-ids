"""
trainer_facade: Thin composition root for HelixFullTrainer.

Phase 18: HelixFullTrainer uses TrainerFacade as a single entry point
to all delegates and sub-managers, reducing the trainer to a thin
coordination layer.

Dependencies:
    TrainerState, RecoveryManager, and all delegate classes.
"""

from __future__ import annotations

from scripts.training.core.recovery_manager import RecoveryManager
from scripts.training.core.trainer_state import TrainerState
from scripts.training.diagnostics import (
    ClusterAnalyzer,
    GeometryAnalyzer,
    RepresentationDiagnostics,
)
from scripts.training.evaluation import EvaluationOrchestrator
from scripts.training.execution import (
    BatchProcessor,
    EpochRunner,
    TrainingOrchestrator,
    WarmupManager,
)
from scripts.training.losses import LossRegistry
from scripts.training.representation import CentroidManager, RepresentationCoordinator
from scripts.training.scheduler import (
    EarlyStoppingManager,
    FreezeManager,
    LRScheduler,
    PhaseManager,
    PhaseOrchestrator,
)
from scripts.training.validation import ValidationOrchestrator


class TrainerFacade:
    """Single-access facade for all HelixFullTrainer delegate subsystems.

    The trainer creates one TrainerFacade during __init__ and accesses
    all subsystems through self._trainer_facade.* properties.
    """

    def __init__(self, state: TrainerState) -> None:
        self._state = state

        # ── Delegate references (populated by build()) ───────────────────
        self._phase_manager: PhaseManager | None = None
        self._early_stopping_manager: EarlyStoppingManager | None = None
        self._freeze_manager: FreezeManager | None = None
        self._lr_scheduler: LRScheduler | None = None
        self._evaluation_orchestrator: EvaluationOrchestrator | None = None
        self._validation_orchestrator: ValidationOrchestrator | None = None
        self._geometry_analyzer: GeometryAnalyzer | None = None
        self._cluster_analyzer: ClusterAnalyzer | None = None
        self._rep_diagnostics: RepresentationDiagnostics | None = None
        self._centroid_manager: CentroidManager | None = None
        self._phase_orchestrator: PhaseOrchestrator | None = None
        self._representation_coordinator: RepresentationCoordinator | None = None
        self._loss_registry: LossRegistry | None = None
        self._batch_processor: BatchProcessor | None = None
        self._warmup_manager: WarmupManager | None = None
        self._epoch_runner: EpochRunner | None = None
        self._training_orchestrator: TrainingOrchestrator | None = None
        self._recovery_manager: RecoveryManager | None = None

    # ── Builder ─────────────────────────────────────────────────────────

    def build(self) -> TrainerFacade:
        """Construct all delegates from TrainerState. Called once during init."""
        from scripts.training.core.recovery_manager import RecoveryManager

        # ── Scheduler layer (Phase 13A-2) ────────────────────────────────
        self._phase_manager = self._state.build_phase_manager()
        self._early_stopping_manager = self._state.build_early_stopping_manager()
        self._freeze_manager = self._state.build_freeze_manager()
        self._lr_scheduler = self._state.build_lr_scheduler()

        # ── Diagnostics layer (Phase 13A-1) ──────────────────────────────
        self._geometry_analyzer = self._state.build_geometry_analyzer()
        self._cluster_analyzer = self._state.build_cluster_analyzer()
        self._rep_diagnostics = self._state.build_rep_diagnostics()

        # ── Representation layer (Phase 13A-3) ───────────────────────────
        self._centroid_manager = self._state.build_centroid_manager()
        self._representation_coordinator = self._state.build_representation_coordinator()

        # ── Loss registry (Phase 14) ─────────────────────────────────────
        self._loss_registry = self._state.build_loss_registry()

        # ── Phase orchestrator (Phase 15, depends on scheduler + diag) ───
        self._phase_orchestrator = self._state.build_phase_orchestrator(
            phase_manager=self._phase_manager,
            early_stopping_manager=self._early_stopping_manager,
            geometry_analyzer=self._geometry_analyzer,
            cluster_analyzer=self._cluster_analyzer,
            centroid_manager=self._centroid_manager,
            rep_diagnostics=self._rep_diagnostics,
        )

        # ── Orchestration layers (Phase 16/17) ───────────────────────────
        self._evaluation_orchestrator = self._state.build_evaluation_orchestrator()
        self._validation_orchestrator = self._state.build_validation_orchestrator()
        self._batch_processor = self._state.build_batch_processor(
            loss_registry=self._loss_registry
        )
        self._warmup_manager = self._state.build_warmup_manager()
        self._epoch_runner = self._state.build_epoch_runner(
            batch_processor=self._batch_processor,
            warmup_manager=self._warmup_manager,
        )
        self._training_orchestrator = self._state.build_training_orchestrator(
            epoch_runner=self._epoch_runner,
        )

        # ── Recovery manager (Phase 18) ──────────────────────────────────
        self._recovery_manager = RecoveryManager(
            state=self._state,
            loss_registry=self._loss_registry,
            centroid_manager=self._centroid_manager,
            logger=self._state.logger,
        )

        return self

    # ── Delegate accessors ──────────────────────────────────────────────

    @property
    def phase_manager(self) -> PhaseManager:
        if self._phase_manager is None:
            raise RuntimeError("phase_manager not initialized — call init() first")
        return self._phase_manager

    @property
    def early_stopping_manager(self) -> EarlyStoppingManager:
        if self._early_stopping_manager is None:
            raise RuntimeError("early_stopping_manager not initialized — call init() first")
        return self._early_stopping_manager

    @property
    def freeze_manager(self) -> FreezeManager:
        if self._freeze_manager is None:
            raise RuntimeError("freeze_manager not initialized — call init() first")
        return self._freeze_manager

    @property
    def lr_scheduler(self) -> LRScheduler:
        if self._lr_scheduler is None:
            raise RuntimeError("lr_scheduler not initialized — call init() first")
        return self._lr_scheduler

    @property
    def evaluation_orchestrator(self) -> EvaluationOrchestrator:
        if self._evaluation_orchestrator is None:
            raise RuntimeError("evaluation_orchestrator not initialized — call init() first")
        return self._evaluation_orchestrator

    @property
    def validation_orchestrator(self) -> ValidationOrchestrator:
        if self._validation_orchestrator is None:
            raise RuntimeError("validation_orchestrator not initialized — call init() first")
        return self._validation_orchestrator

    @property
    def geometry_analyzer(self) -> GeometryAnalyzer:
        if self._geometry_analyzer is None:
            raise RuntimeError("geometry_analyzer not initialized — call init() first")
        return self._geometry_analyzer

    @property
    def cluster_analyzer(self) -> ClusterAnalyzer:
        if self._cluster_analyzer is None:
            raise RuntimeError("cluster_analyzer not initialized — call init() first")
        return self._cluster_analyzer

    @property
    def rep_diagnostics(self) -> RepresentationDiagnostics:
        if self._rep_diagnostics is None:
            raise RuntimeError("rep_diagnostics not initialized — call init() first")
        return self._rep_diagnostics

    @property
    def centroid_manager(self) -> CentroidManager:
        if self._centroid_manager is None:
            raise RuntimeError("centroid_manager not initialized — call init() first")
        return self._centroid_manager

    @property
    def phase_orchestrator(self) -> PhaseOrchestrator:
        if self._phase_orchestrator is None:
            raise RuntimeError("phase_orchestrator not initialized — call init() first")
        return self._phase_orchestrator

    @property
    def representation_coordinator(self) -> RepresentationCoordinator:
        if self._representation_coordinator is None:
            raise RuntimeError("representation_coordinator not initialized — call init() first")
        return self._representation_coordinator

    @property
    def loss_registry(self) -> LossRegistry:
        if self._loss_registry is None:
            raise RuntimeError("loss_registry not initialized — call init() first")
        return self._loss_registry

    @property
    def batch_processor(self) -> BatchProcessor:
        if self._batch_processor is None:
            raise RuntimeError("batch_processor not initialized — call init() first")
        return self._batch_processor

    @property
    def warmup_manager(self) -> WarmupManager:
        if self._warmup_manager is None:
            raise RuntimeError("warmup_manager not initialized — call init() first")
        return self._warmup_manager

    @property
    def epoch_runner(self) -> EpochRunner:
        if self._epoch_runner is None:
            raise RuntimeError("epoch_runner not initialized — call init() first")
        return self._epoch_runner

    @property
    def training_orchestrator(self) -> TrainingOrchestrator:
        if self._training_orchestrator is None:
            raise RuntimeError("training_orchestrator not initialized — call init() first")
        return self._training_orchestrator

    @property
    def recovery_manager(self) -> RecoveryManager:
        if self._recovery_manager is None:
            raise RuntimeError("recovery_manager not initialized — call init() first")
        return self._recovery_manager
