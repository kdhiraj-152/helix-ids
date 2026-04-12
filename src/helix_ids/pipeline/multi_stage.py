"""
Multi-stage IDS pipeline with ESP32 → RPi → Server escalation.

Architecture:
    Stage 1: ESP32 Fast Reject (<100μs)
        - Binary model (Normal vs Attack)
        - Pass 85-90% of normal traffic (exit pipeline)
        - Flag suspicious → Stage 2

    Stage 2: RPi Refined Classification (<5ms)
        - Hierarchical model (4 attack families)
        - Attack family identification (DoS/Probe/R2L/U2R)
        - Uncertain/critical → Stage 3

    Stage 3: Server Deep Analysis
        - Full 5-class classification
        - Confidence calibration
        - Detailed threat report
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


class AttackFamily(Enum):
    """Attack classification families."""

    NORMAL = "Normal"
    DOS = "DoS"
    PROBE = "Probe"
    R2L = "R2L"
    U2R = "U2R"


@dataclass
class StageResult:
    """Result from a single stage."""

    stage: int
    classification: str
    confidence: float
    family: Optional[str] = None
    latency_ms: float = 0.0
    should_escalate: bool = False
    details: Optional[dict[str, Any]] = None


@dataclass
class PipelineResult:
    """Final result from multi-stage pipeline."""

    stage: int  # Which stage made final decision (1, 2, or 3)
    classification: str  # Final classification
    family: Optional[str] = None  # Attack family if applicable
    confidence: float = 0.0  # Confidence in classification
    is_normal: bool = False  # Is traffic normal?
    is_attack: bool = False  # Is traffic attack?
    escalation_path: list[int] = field(default_factory=list)  # Stages traversed
    total_latency_ms: float = 0.0  # Total time across all stages
    stage_latencies: list[float] = field(default_factory=list)
    details: Optional[dict[str, Any]] = None
    threat_score: float = 0.0  # 0-1, higher = more threatening


@dataclass
class PipelineMetrics:
    """Metrics tracking for pipeline performance."""

    total_samples: int = 0
    stage1_exits: int = 0  # Normal exits at Stage 1
    stage2_exits: int = 0  # Attack classification at Stage 2
    stage3_exits: int = 0  # Full analysis at Stage 3
    stage1_latencies: list[float] = field(default_factory=list)
    stage2_latencies: list[float] = field(default_factory=list)
    stage3_latencies: list[float] = field(default_factory=list)
    escalation_rate: float = 0.0  # % of samples escalated to next stage
    attack_detection_rate: float = 0.0

    def update(self, result: PipelineResult) -> None:
        """Update metrics with pipeline result."""
        self.total_samples += 1

        if result.stage == 1:
            self.stage1_exits += 1
        elif result.stage == 2:
            self.stage2_exits += 1
        elif result.stage == 3:
            self.stage3_exits += 1

        self.stage1_latencies.extend(
            [result.stage_latencies[0]] if len(result.stage_latencies) > 0 else []
        )
        if len(result.stage_latencies) > 1:
            self.stage2_latencies.append(result.stage_latencies[1])
        if len(result.stage_latencies) > 2:
            self.stage3_latencies.append(result.stage_latencies[2])

        if result.is_attack:
            self.attack_detection_rate = (
                self.attack_detection_rate * (self.total_samples - 1) / self.total_samples
                + 1.0 / self.total_samples
            )

        self.escalation_rate = (self.stage2_exits + self.stage3_exits) / max(1, self.total_samples)

    def summary(self) -> dict[str, Any]:
        """Get summary of pipeline metrics."""
        return {
            "total_samples": self.total_samples,
            "stage1_exits": self.stage1_exits,
            "stage1_exit_pct": 100 * self.stage1_exits / max(1, self.total_samples),
            "stage2_exits": self.stage2_exits,
            "stage2_exit_pct": 100 * self.stage2_exits / max(1, self.total_samples),
            "stage3_exits": self.stage3_exits,
            "stage3_exit_pct": 100 * self.stage3_exits / max(1, self.total_samples),
            "escalation_rate": self.escalation_rate,
            "attack_detection_rate": self.attack_detection_rate,
            "avg_stage1_latency_us": (
                1000 * np.mean(self.stage1_latencies) if self.stage1_latencies else 0.0
            ),
            "avg_stage2_latency_ms": (
                np.mean(self.stage2_latencies) if self.stage2_latencies else 0.0
            ),
            "avg_stage3_latency_ms": (
                np.mean(self.stage3_latencies) if self.stage3_latencies else 0.0
            ),
        }


class ESP32Stage:
    """
    Stage 1: ESP32 Fast Binary Classifier.

    - Decision time: <100μs
    - Pass ~85-90% of normal traffic (immediate exit)
    - Flag suspicious → Stage 2
    """

    def __init__(self, model: nn.Module, threshold: float = 0.7):
        """
        Initialize ESP32 stage.

        Args:
            model: Binary classification model (Normal vs Attack)
            threshold: Confidence threshold for exit decision
        """
        self.model = model
        self.threshold = threshold
        self.device = next(model.parameters()).device

    def predict(self, flow_features: np.ndarray) -> StageResult:
        """
        Make binary classification decision.

        Args:
            flow_features: Shape (1, 41) or (batch_size, 41)

        Returns:
            StageResult with binary classification
        """
        start_time = time.time()

        # Convert to tensor
        if isinstance(flow_features, np.ndarray):
            features = torch.from_numpy(flow_features).float().to(self.device)
        else:
            features = flow_features.float().to(self.device)

        # Ensure batch dimension
        if features.dim() == 1:
            features = features.unsqueeze(0)

        with torch.no_grad():
            output = self.model(features)

            # Extract binary classification (Normal=0, Attack=1)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Sigmoid for binary classification
            probs = torch.sigmoid(logits[:, 0])  # Prob of normal
            confidence = probs.item() if probs.dim() == 0 else probs[0].item()

        elapsed_ms = (time.time() - start_time) * 1000

        is_normal = confidence > self.threshold
        classification = "Normal" if is_normal else "Suspicious"

        return StageResult(
            stage=1,
            classification=classification,
            confidence=confidence,
            latency_ms=elapsed_ms,
            should_escalate=not is_normal,
            details={
                "is_normal": is_normal,
                "normal_probability": confidence,
                "attack_probability": 1.0 - confidence,
            },
        )


class RPiStage:
    """
    Stage 2: RPi Refined Classification.

    - Decision time: <5ms
    - Hierarchical classification (4 attack families)
    - Identify attack family (DoS/Probe/R2L/U2R)
    """

    def __init__(self, model: nn.Module, threshold: float = 0.7):
        """
        Initialize RPi stage.

        Args:
            model: Hierarchical classification model
            threshold: Confidence threshold for exit decision
        """
        self.model = model
        self.threshold = threshold
        self.device = next(model.parameters()).device
        self.family_names = [
            AttackFamily.DOS.value,
            AttackFamily.PROBE.value,
            AttackFamily.R2L.value,
            AttackFamily.U2R.value,
        ]

    def predict(self, flow_features: np.ndarray) -> StageResult:
        """
        Make hierarchical family classification.

        Args:
            flow_features: Shape (batch_size, 41)

        Returns:
            StageResult with attack family classification
        """
        start_time = time.time()

        # Convert to tensor
        if isinstance(flow_features, np.ndarray):
            features = torch.from_numpy(flow_features).float().to(self.device)
        else:
            features = flow_features.float().to(self.device)

        if features.dim() == 1:
            features = features.unsqueeze(0)

        with torch.no_grad():
            output = self.model(features)

            # Handle hierarchical output
            if isinstance(output, tuple):
                family_logits = output[1] if len(output) > 1 else output[0]
            else:
                family_logits = output

            # Get family classification
            family_probs = torch.softmax(family_logits[:, :4], dim=-1)
            family_idx = int(torch.argmax(family_probs, dim=-1).item())
            confidence = family_probs[0, family_idx].item()

        elapsed_ms = (time.time() - start_time) * 1000

        family = self.family_names[family_idx]
        should_escalate = confidence < self.threshold

        return StageResult(
            stage=2,
            classification=family,
            family=family,
            confidence=confidence,
            latency_ms=elapsed_ms,
            should_escalate=should_escalate,
            details={"family_probabilities": family_probs[0, :4].cpu().numpy().tolist()},
        )


class ServerStage:
    """
    Stage 3: Server Deep Analysis.

    - Full 5-class classification
    - Confidence calibration
    - Detailed threat report
    """

    def __init__(self, model: nn.Module):
        """
        Initialize server stage.

        Args:
            model: Full classification model (5 classes)
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.class_names = [
            AttackFamily.NORMAL.value,
            AttackFamily.DOS.value,
            AttackFamily.PROBE.value,
            AttackFamily.R2L.value,
            AttackFamily.U2R.value,
        ]

    def full_analysis(self, flow_features: np.ndarray) -> StageResult:
        """
        Perform comprehensive 5-class analysis.

        Args:
            flow_features: Shape (batch_size, 41)

        Returns:
            StageResult with full analysis
        """
        start_time = time.time()

        # Convert to tensor
        if isinstance(flow_features, np.ndarray):
            features = torch.from_numpy(flow_features).float().to(self.device)
        else:
            features = flow_features.float().to(self.device)

        if features.dim() == 1:
            features = features.unsqueeze(0)

        with torch.no_grad():
            output = self.model(features)

            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # 5-class softmax
            probs = torch.softmax(logits[:, :5], dim=-1)
            class_idx = int(torch.argmax(probs, dim=-1).item())
            confidence = probs[0, class_idx].item()

        elapsed_ms = (time.time() - start_time) * 1000

        classification = self.class_names[class_idx]
        is_normal = class_idx == 0

        # Compute threat score (0 = normal, 1 = critical threat)
        threat_score = 0.0 if is_normal else confidence
        if class_idx == 4:  # U2R (highest threat)
            threat_score = min(1.0, confidence * 1.2)

        return StageResult(
            stage=3,
            classification=classification,
            family=classification if class_idx > 0 else None,
            confidence=confidence,
            latency_ms=elapsed_ms,
            details={
                "class_probabilities": probs[0, :5].cpu().numpy().tolist(),
                "threat_score": threat_score,
                "all_classes": self.class_names,
            },
        )


class MultiStagePipeline:
    """
    Three-stage intrusion detection pipeline.

    Escalation path:
        Stage 1 (ESP32) → Binary fast reject
        Stage 2 (RPi) → Family classification
        Stage 3 (Server) → Full analysis
    """

    def __init__(
        self,
        esp32_model: nn.Module,
        rpi_model: nn.Module,
        server_model: nn.Module,
        stage1_threshold: float = 0.7,
        stage2_threshold: float = 0.7,
    ):
        """
        Initialize multi-stage pipeline.

        Args:
            esp32_model: Binary classifier for stage 1
            rpi_model: Hierarchical classifier for stage 2
            server_model: Full classifier for stage 3
            stage1_threshold: Exit threshold for stage 1
            stage2_threshold: Exit threshold for stage 2
        """
        self.stage1 = ESP32Stage(esp32_model, threshold=stage1_threshold)
        self.stage2 = RPiStage(rpi_model, threshold=stage2_threshold)
        self.stage3 = ServerStage(server_model)
        self.metrics = PipelineMetrics()

    def classify(
        self,
        flow_features: np.ndarray,
        stage1_threshold: Optional[float] = None,
        stage2_threshold: Optional[float] = None,
    ) -> PipelineResult:
        """
        Classify flow through multi-stage pipeline.

        Pipeline Logic:
            1. Stage 1: Binary normal/attack decision
               - If high confidence normal → Exit (stage=1)
               - Else → Escalate to Stage 2

            2. Stage 2: Attack family classification
               - If high confidence → Exit (stage=2)
               - Else or critical family → Escalate to Stage 3

            3. Stage 3: Full analysis
               - Return final 5-class classification (stage=3)

        Args:
            flow_features: Network flow features (1, 41) or (batch, 41)
            stage1_threshold: Override stage 1 threshold
            stage2_threshold: Override stage 2 threshold

        Returns:
            PipelineResult with classification and metadata
        """
        total_start = time.time()
        stage_latencies = []

        # Stage 1: Binary classification
        if stage1_threshold is not None:
            old_threshold = self.stage1.threshold
            self.stage1.threshold = stage1_threshold
            result1 = self.stage1.predict(flow_features)
            self.stage1.threshold = old_threshold
        else:
            result1 = self.stage1.predict(flow_features)

        stage_latencies.append(result1.latency_ms)

        # Check if exit at stage 1 (high confidence normal)
        if result1.classification == "Normal" and result1.confidence > self.stage1.threshold:
            pipeline_result = PipelineResult(
                stage=1,
                classification="Normal",
                is_normal=True,
                is_attack=False,
                confidence=result1.confidence,
                escalation_path=[1],
                total_latency_ms=time.time() - total_start,
                stage_latencies=stage_latencies,
                details=result1.details,
                threat_score=0.0,
            )
            self.metrics.update(pipeline_result)
            return pipeline_result

        # Stage 2: Attack family classification
        if stage2_threshold is not None:
            old_threshold = self.stage2.threshold
            self.stage2.threshold = stage2_threshold
            result2 = self.stage2.predict(flow_features)
            self.stage2.threshold = old_threshold
        else:
            result2 = self.stage2.predict(flow_features)

        stage_latencies.append(result2.latency_ms)

        # Check if exit at stage 2 (confident attack family, not critical)
        if not result2.should_escalate and result2.family != AttackFamily.U2R.value:
            pipeline_result = PipelineResult(
                stage=2,
                classification=result2.family or result2.classification,
                family=result2.family,
                is_attack=True,
                is_normal=False,
                confidence=result2.confidence,
                escalation_path=[1, 2],
                total_latency_ms=time.time() - total_start,
                stage_latencies=stage_latencies,
                details=result2.details,
                threat_score=result2.confidence,
            )
            self.metrics.update(pipeline_result)
            return pipeline_result

        # Stage 3: Full analysis
        result3 = self.stage3.full_analysis(flow_features)
        stage_latencies.append(result3.latency_ms)

        is_normal = result3.classification == "Normal"
        is_attack = not is_normal
        threat_score = (
            result3.details.get("threat_score", 0.0) if result3.details else result3.confidence
        )

        pipeline_result = PipelineResult(
            stage=3,
            classification=result3.classification,
            family=result3.family,
            is_attack=is_attack,
            is_normal=is_normal,
            confidence=result3.confidence,
            escalation_path=[1, 2, 3],
            total_latency_ms=(time.time() - total_start) * 1000,
            stage_latencies=stage_latencies,
            details=result3.details,
            threat_score=threat_score,
        )

        self.metrics.update(pipeline_result)
        return pipeline_result

    def classify_batch(self, flow_features: np.ndarray) -> list[PipelineResult]:
        """
        Classify multiple flows.

        Args:
            flow_features: Shape (batch_size, 41)

        Returns:
            List of PipelineResult objects
        """
        if flow_features.ndim == 1:
            flow_features = np.expand_dims(flow_features, axis=0)

        results = []
        for i in range(flow_features.shape[0]):
            result = self.classify(flow_features[i : i + 1])
            results.append(result)

        return results

    def get_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics summary."""
        return self.metrics.summary()

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self.metrics = PipelineMetrics()
