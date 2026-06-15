"""HelixIDS-Full evaluation pipeline extracted from HelixFullTrainer.

Extracted in Phase 12B-6 to separate evaluation concerns from training.
Can be used independently for evaluation-only workflows.

All methods preserve the exact behavioral equivalence of the original
HelixFullTrainer methods they were extracted from.
"""

from __future__ import annotations

import logging
import math
from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from helix_ids.models.full import HelixIDSFull, MultiTaskLoss


class HelixFullEvaluator:
    """Inference-time evaluator for HelixIDS-Full model.

    Encapsulates all evaluation logic formerly embedded in HelixFullTrainer.
    Stateless during a single evaluation call — all configuration is fixed at
    construction time.

    Parameters
    ----------
    model : HelixIDSFull
        The model to evaluate.
    device : str
        Target device for evaluation.
    loss_fn : MultiTaskLoss
        Loss function used for computing validation loss.
    binary_class_weights : torch.Tensor or None
        Per-class weights for binary classification.
    family_class_weights : torch.Tensor or None
        Per-class weights for family classification.
    logger : logging.Logger or None
        Logger instance.
    family_log_prior : torch.Tensor or None
        Log-prior for family logit correction (evaluation mode).
    use_energy_based_family_objective : bool
        Whether to apply temperature scaling from energy objective.
    energy_logit_temperature : float
        Temperature for logit scaling.
    active_family_class_ids : set[int] or None
        Class IDs that must appear in predictions (prediction floor).
    class4_logit_shift : float
        Inference-time class-4 logit shift delta.
    class4_logit_shift_class_id : int
        Index of the class to apply logit shift to.
    disable_integrity_hard_stops : bool
        If True, skip hard-stop integrity checks (for testing).
    """

    def __init__(
        self,
        *,
        model: HelixIDSFull,
        device: str,
        loss_fn: MultiTaskLoss,
        binary_class_weights: torch.Tensor | None = None,
        family_class_weights: torch.Tensor | None = None,
        logger: logging.Logger | None = None,
        # Logit control parameters (evaluation mode: no emergence bias)
        family_log_prior: torch.Tensor | None = None,
        use_energy_based_family_objective: bool = True,
        energy_logit_temperature: float = 2.0,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float = 0.0,
        class4_logit_shift_class_id: int = 4,
        # Diagnostic/state parameters
        disable_integrity_hard_stops: bool = False,
    ):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.binary_class_weights = binary_class_weights
        self.family_class_weights = family_class_weights
        self.logger = logger or logging.getLogger(__name__)

        # Logit control state
        self.family_log_prior = family_log_prior
        self.use_energy_based_family_objective = use_energy_based_family_objective
        self.energy_logit_temperature = energy_logit_temperature
        self.active_family_class_ids = active_family_class_ids or set()
        self.class4_logit_shift = class4_logit_shift
        self.class4_logit_shift_class_id = class4_logit_shift_class_id

        # Diagnostic state
        self.disable_integrity_hard_stops = disable_integrity_hard_stops

    # ------------------------------------------------------------------ #
    # Logit helpers (evaluation-only: no emergence bias, no train state)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_eval_class4_logit_shift(
        family_logits: torch.Tensor,
        *,
        shift: float,
        class_id: int,
    ) -> torch.Tensor:
        """Apply inference-time class N logit shift (logit_N <- logit_N - delta).

        Equivalent to HelixFullTrainer._apply_eval_class4_logit_shift.
        """
        if int(family_logits.ndim) != 2 or int(family_logits.shape[0]) <= 0:
            return family_logits
        delta = float(shift or 0.0)
        if delta <= 0.0:
            return family_logits
        cid = int(class_id)
        if cid < 0 or cid >= int(family_logits.shape[1]):
            return family_logits
        shifted = family_logits.clone()
        shifted[:, cid] = shifted[:, cid] - delta
        return shifted

    def _apply_eval_logit_controls(self, family_logits: torch.Tensor) -> torch.Tensor:
        """Evaluation-only family logit controls: temperature + prior.

        Corresponds to HelixFullTrainer._apply_family_logit_controls called
        with ``apply_emergence_bias=False`` — emergence bias is a training-time
        mechanism and is not applied during evaluation.
        """
        controlled = family_logits

        if self.use_energy_based_family_objective:
            controlled = controlled / max(1e-6, float(self.energy_logit_temperature))

        if self.family_log_prior is not None:
            if int(self.family_log_prior.shape[-1]) != int(family_logits.shape[-1]):
                raise RuntimeError(
                    "family prior dimension mismatch: "
                    f"priors={int(self.family_log_prior.shape[-1])} logits={int(family_logits.shape[-1])}"
                )
            controlled = controlled - self.family_log_prior

        return controlled

    @staticmethod
    def _apply_inference_prediction_floor(
        family_logits: torch.Tensor,
        family_pred: torch.Tensor,
        *,
        active_class_ids: set[int],
    ) -> torch.Tensor:
        """Inference-only prediction floor to guarantee per-batch class presence.

        Equivalent to HelixFullTrainer._apply_inference_prediction_floor.
        """
        if int(family_logits.ndim) != 2 or int(family_logits.shape[0]) <= 0:
            return family_pred

        class_count = int(family_logits.shape[1])
        active_ids = [
            int(cls)
            for cls in active_class_ids
            if 0 <= int(cls) < class_count
        ]
        if not active_ids:
            return family_pred

        adjusted_pred = family_pred.clone()
        predicted_set = {int(v) for v in adjusted_pred.tolist()}
        missing_ids = [cls for cls in active_ids if cls not in predicted_set]
        if not missing_ids:
            return adjusted_pred

        used_rows: set[int] = set()
        for cls in missing_ids:
            class_scores = family_logits[:, cls].clone()
            if used_rows:
                for row_idx in used_rows:
                    class_scores[row_idx] = float("-inf")
            row = int(torch.argmax(class_scores).item())
            adjusted_pred[row] = cls
            used_rows.add(row)

        return adjusted_pred

    # ------------------------------------------------------------------ #
    # F1 statistics (stateless static helper)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_f1_stats_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
        """Compute F1-related statistics from confusion matrix counts.

        Equivalent to HelixFullTrainer._compute_f1_stats_from_confusion.
        """
        if confusion.numel() == 0:
            return {
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "minority_recall_min": 0.0,
                "zero_prediction_classes": [],
            }

        conf = confusion.to(device="cpu", dtype=torch.float64)
        support = conf.sum(dim=1)
        predicted = conf.sum(dim=0)
        tp = torch.diag(conf)

        precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
        recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
        denom = precision + recall
        f1 = torch.where(denom > 0, 2.0 * precision * recall / denom, torch.zeros_like(tp))

        active_classes = (support + predicted) > 0
        macro_f1 = float(f1[active_classes].mean().item()) if bool(active_classes.any()) else 0.0

        total_support = float(support.sum().item())
        weighted_f1 = (
            float((f1 * support).sum().item() / total_support) if total_support > 0 else 0.0
        )

        present_classes = support > 0
        minority_present = torch.where(present_classes)[0].tolist()
        minority_recalls = [float(recall[idx].item()) for idx in minority_present if int(idx) != 0]
        minority_recall_min = float(min(minority_recalls)) if minority_recalls else 0.0

        zero_prediction_classes = sorted(
            int(idx)
            for idx in torch.where((support > 0) & (predicted == 0))[0].tolist()
        )

        return {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "minority_recall_min": minority_recall_min,
            "zero_prediction_classes": zero_prediction_classes,
        }

    # ------------------------------------------------------------------ #
    # Single-loader evaluation
    # ------------------------------------------------------------------ #

    def _resolve_active_class_ids(
        self,
        *,
        active_family_class_ids: set[int] | None = None,
    ) -> set[int]:
        """Resolve active class IDs, preferring per-call override over constructor default."""
        if active_family_class_ids is not None:
            return active_family_class_ids
        return self.active_family_class_ids

    @torch.no_grad()
    def _evaluate_loader(
        self,
        loader: DataLoader,
        dataset_name: str = "unknown",
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate metrics on a single dataset loader.

        Exact copy of HelixFullTrainer._evaluate_loader, with trainer-internal
        method calls replaced by equivalent evaluator-level implementations.

        Parameters
        ----------
        active_family_class_ids, class4_logit_shift, class4_logit_shift_class_id :
            Optional per-call overrides for mutable trainer attributes. When
            provided, these take precedence over the constructor values.
        """
        total_loss = 0.0
        total_calibrated_loss = 0.0
        total_binary_correct = 0
        total_family_correct = 0
        total_samples = 0

        binary_prob_chunks: list[torch.Tensor] = []
        binary_label_chunks: list[torch.Tensor] = []
        family_confusion: torch.Tensor | None = None
        family_entropy_sum = 0.0
        family_class_count = 0
        family_pred_counts: torch.Tensor | None = None
        family_logit_sums: torch.Tensor | None = None
        top2_logit_gap_sum = 0.0
        top2_logit_gap_count = 0
        binary_weights_cpu = (
            self.binary_class_weights.to(device="cpu") if self.binary_class_weights is not None else None
        )
        family_weights_cpu = (
            self.family_class_weights.to(device="cpu") if self.family_class_weights is not None else None
        )

        for x, y_binary, y_family in loader:
            x = x.to(self.device, non_blocking=True)
            y_binary_cpu = y_binary.to(device="cpu", dtype=torch.long, non_blocking=True)
            y_family_cpu = y_family.to(device="cpu", dtype=torch.long, non_blocking=True)

            binary_logits_dev, family_logits_dev = self.model(x)
            family_logits_dev = self._apply_eval_logit_controls(family_logits_dev)
            binary_logits = binary_logits_dev.to(device="cpu")
            family_logits = family_logits_dev.to(device="cpu")
            effective_shift = class4_logit_shift if class4_logit_shift is not None else self.class4_logit_shift
            effective_class_id = class4_logit_shift_class_id if class4_logit_shift_class_id is not None else self.class4_logit_shift_class_id
            family_logits = self._apply_eval_class4_logit_shift(
                family_logits,
                shift=effective_shift,
                class_id=effective_class_id,
            )

            loss, _ = self.loss_fn(
                binary_logits,
                y_binary_cpu,
                family_logits,
                y_family_cpu,
                binary_class_weights=binary_weights_cpu,
                family_class_weights=family_weights_cpu,
            )
            calibrated_loss = loss

            batch_size = int(y_binary_cpu.shape[0])
            binary_pred = torch.argmax(binary_logits, dim=1)
            family_pred = torch.argmax(family_logits, dim=1)
            effective_active_ids = self._resolve_active_class_ids(
                active_family_class_ids=active_family_class_ids,
            )
            family_pred = self._apply_inference_prediction_floor(
                family_logits, family_pred,
                active_class_ids=effective_active_ids,
            )
            total_loss += float(loss.item()) * batch_size
            total_calibrated_loss += float(calibrated_loss.item()) * batch_size
            total_binary_correct += int((binary_pred == y_binary_cpu).sum().item())
            total_family_correct += int((family_pred == y_family_cpu).sum().item())
            total_samples += batch_size

            binary_prob_chunks.append(torch.softmax(binary_logits, dim=1)[:, 1].detach())
            binary_label_chunks.append(y_binary_cpu.detach())

            if family_confusion is None:
                family_class_count = int(family_logits.shape[1])
                family_confusion = torch.zeros(
                    (family_class_count, family_class_count),
                    dtype=torch.int64,
                )

            if not self.disable_integrity_hard_stops:
                invalid_label_mask = (y_family_cpu < 0) | (y_family_cpu >= family_class_count)
                if bool(torch.any(invalid_label_mask)):
                    raise RuntimeError(
                        "Hard-stop integrity guard triggered: invalid_family_labels_in_eval_"
                        f"{dataset_name}:min={int(torch.min(y_family_cpu).item())}"
                        f":max={int(torch.max(y_family_cpu).item())}"
                        f":class_count={family_class_count}"
                    )

            family_index = (
                y_family_cpu.to(dtype=torch.int64) * family_class_count
                + family_pred.detach().to(dtype=torch.int64)
            )
            family_confusion += torch.bincount(
                family_index,
                minlength=family_class_count * family_class_count,
            ).reshape(family_class_count, family_class_count)

            family_prob = torch.softmax(family_logits, dim=1)
            safe_family_prob = torch.clamp(family_prob, min=1e-10, max=1.0)
            batch_entropy = -torch.sum(family_prob * torch.log(safe_family_prob), dim=1)
            batch_entropy = batch_entropy / math.log(float(family_prob.shape[1]))
            family_entropy_sum += float(batch_entropy.sum().item())

            if family_pred_counts is None:
                family_pred_counts = torch.zeros(family_class_count, dtype=torch.int64)
                family_logit_sums = torch.zeros(family_class_count, dtype=torch.float32)
            family_pred_counts += torch.bincount(
                family_pred.detach().to(device="cpu", dtype=torch.int64),
                minlength=family_class_count,
            )
            if family_logit_sums is not None:
                family_logit_sums += family_logits.detach().to(device="cpu", dtype=torch.float32).sum(
                    dim=0
                )

            top2_values = torch.topk(family_logits.detach(), k=2, dim=1).values
            top2_logit_gap_sum += float((top2_values[:, 0] - top2_values[:, 1]).sum().item())
            top2_logit_gap_count += int(top2_values.shape[0])

        if binary_prob_chunks:
            binary_probs = torch.cat(binary_prob_chunks, dim=0).to(device="cpu").numpy()
            binary_labels = torch.cat(binary_label_chunks, dim=0).to(device="cpu").numpy()
        else:
            binary_probs = np.array([])
            binary_labels = np.array([])

        if binary_labels.size > 0 and np.unique(binary_labels).size > 1:
            binary_auroc = float(roc_auc_score(binary_labels, binary_probs))
            binary_auprc = float(average_precision_score(binary_labels, binary_probs))
        else:
            binary_auroc = 0.0
            binary_auprc = 0.0

        family_stats = self._compute_f1_stats_from_confusion(
            family_confusion if family_confusion is not None else torch.zeros((0, 0), dtype=torch.int64)
        )
        if family_confusion is not None and int(family_confusion.numel()) > 0:
            support = family_confusion.sum(dim=1).to(dtype=torch.float64)
            tp = torch.diag(family_confusion).to(dtype=torch.float64)
            recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
            recall_payload = {
                int(idx): float(recall[idx].item())
                for idx in range(int(recall.shape[0]))
                if float(support[idx].item()) > 0.0
            }
            self.logger.info("ValDiag[%s] per_class_recall=%s", dataset_name, recall_payload)
        family_entropy = family_entropy_sum / max(1, total_samples)
        val_top2_logit_gap = top2_logit_gap_sum / max(1, top2_logit_gap_count)

        if family_pred_counts is not None and family_logit_sums is not None and total_samples > 0:
            pred_count_payload = {
                int(idx): int(count)
                for idx, count in enumerate(family_pred_counts.tolist())
            }
            avg_logit_payload = {
                int(idx): float(total / max(1, total_samples))
                for idx, total in enumerate(family_logit_sums.tolist())
            }
            self.logger.info(
                "ValDiag[%s] per_class_prediction_count=%s per_class_avg_logit=%s top2_logit_gap=%.4f",
                dataset_name,
                pred_count_payload,
                avg_logit_payload,
                val_top2_logit_gap,
            )

        return {
            "num_samples": float(total_samples),
            "val_loss": total_loss / max(1, total_samples),
            "val_calibrated_loss": total_calibrated_loss / max(1, total_samples),
            "val_binary_acc": total_binary_correct / max(1, total_samples),
            "val_family_acc": total_family_correct / max(1, total_samples),
            "val_binary_auroc": binary_auroc,
            "val_binary_auprc": binary_auprc,
            "val_family_macro_f1": float(family_stats["macro_f1"]),
            "val_family_minority_recall_min": float(family_stats["minority_recall_min"]),
            "val_family_entropy": family_entropy,
            "val_family_zero_prediction_classes": float(
                len(cast(list[int], family_stats["zero_prediction_classes"]))
            ),
            "val_family_predicted_class_count": float(
                int((family_confusion.sum(dim=0) > 0).sum().item()) if family_confusion is not None else 0
            ),
            "val_family_top2_logit_gap": float(val_top2_logit_gap),
        }

    # ------------------------------------------------------------------ #
    # Validation (multi-dataset aggregation)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def validate(
        self,
        val_loaders: dict[str, DataLoader],
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> dict[str, float]:
        """Validate per dataset with strict isolation (worst-case aggregation).

        Equivalent to HelixFullTrainer.validate — iterates all configured
        validation loaders, delegates to _evaluate_loader, and aggregates
        with the same min/max logic used by the trainer.
        """
        self.model.eval()
        if not val_loaders:
            raise RuntimeError("No validation loaders configured")

        dataset_metrics: dict[str, dict[str, Any]] = {}
        for dataset_name, loader in val_loaders.items():
            metrics = self._evaluate_loader(
                loader,
                dataset_name=dataset_name,
                active_family_class_ids=active_family_class_ids,
                class4_logit_shift=class4_logit_shift,
                class4_logit_shift_class_id=class4_logit_shift_class_id,
            )
            dataset_metrics[dataset_name] = metrics
            self.logger.info(
                f"Val[{dataset_name}] loss={metrics['val_loss']:.4f}, "
                f"bin_acc={metrics['val_binary_acc']:.4f}, "
                f"fam_acc={metrics['val_family_acc']:.4f}, "
                f"entropy={metrics['val_family_entropy']:.4f}, "
                f"top2_gap={metrics.get('val_family_top2_logit_gap', 0.0):.4f}"
            )

        total_samples = sum(metric["num_samples"] for metric in dataset_metrics.values())
        if total_samples <= 0:
            raise RuntimeError("Validation metrics are empty; no samples found in val loaders")

        metric_values = list(dataset_metrics.values())
        entropy_missing_same_dataset = any(
            metric["val_family_entropy"] < 0.12
            and metric["val_family_zero_prediction_classes"] > 0
            for metric in metric_values
        )
        return {
            "val_loss": float(max(metric["val_loss"] for metric in metric_values)),
            "val_calibrated_loss": float(
                max(metric["val_calibrated_loss"] for metric in metric_values)
            ),
            "val_binary_acc": float(min(metric["val_binary_acc"] for metric in metric_values)),
            "val_family_acc": float(min(metric["val_family_acc"] for metric in metric_values)),
            "val_binary_auroc": float(min(metric["val_binary_auroc"] for metric in metric_values)),
            "val_binary_auprc": float(min(metric["val_binary_auprc"] for metric in metric_values)),
            "val_family_macro_f1": float(
                min(metric["val_family_macro_f1"] for metric in metric_values)
            ),
            "val_family_minority_recall_min": float(
                min(metric["val_family_minority_recall_min"] for metric in metric_values)
            ),
            "val_family_entropy": float(min(metric["val_family_entropy"] for metric in metric_values)),
            "val_family_zero_prediction_classes": float(
                max(metric["val_family_zero_prediction_classes"] for metric in metric_values)
            ),
            "val_family_predicted_class_count": float(
                min(float(metric.get("val_family_predicted_class_count", 0.0)) for metric in metric_values)
            ),
            "val_family_top2_logit_gap": float(
                min(float(metric.get("val_family_top2_logit_gap", 0.0)) for metric in metric_values)
            ),
            "val_entropy_missing_same_dataset": float(entropy_missing_same_dataset),
        }

    # ------------------------------------------------------------------ #
    # Test evaluation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _process_test_batch(
        self,
        x: torch.Tensor,
        y_binary: torch.Tensor,
        y_family: torch.Tensor,
        binary_confusion: torch.Tensor,
        family_confusion: torch.Tensor | None,
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int, float]:
        """Process one test batch and accumulate metrics.

        Equivalent to HelixFullTrainer._process_test_batch.
        """
        x = x.to(self.device, non_blocking=True)
        y_binary_cpu = y_binary.to(device="cpu", dtype=torch.long, non_blocking=True)
        y_family_cpu = y_family.to(device="cpu", dtype=torch.long, non_blocking=True)

        binary_logits_dev, family_logits_dev = self.model(x)
        family_logits_dev = self._apply_eval_logit_controls(family_logits_dev)
        binary_logits = binary_logits_dev.to(device="cpu")
        family_logits = family_logits_dev.to(device="cpu")
        effective_shift = class4_logit_shift if class4_logit_shift is not None else self.class4_logit_shift
        effective_class_id = class4_logit_shift_class_id if class4_logit_shift_class_id is not None else self.class4_logit_shift_class_id
        family_logits = self._apply_eval_class4_logit_shift(
            family_logits,
            shift=effective_shift,
            class_id=effective_class_id,
        )

        binary_prob = torch.softmax(binary_logits, dim=1)
        family_prob = torch.softmax(family_logits, dim=1)
        binary_pred = torch.argmax(binary_logits, dim=1)
        family_pred = torch.argmax(family_logits, dim=1)
        effective_active_ids_for_batch = self._resolve_active_class_ids(
            active_family_class_ids=active_family_class_ids,
        )
        family_pred = self._apply_inference_prediction_floor(
            family_logits, family_pred,
            active_class_ids=effective_active_ids_for_batch,
        )

        batch_size = int(y_binary_cpu.shape[0])

        # Binary confusion
        binary_index = (
            y_binary_cpu.to(dtype=torch.int64) * 2 + binary_pred.detach().to(dtype=torch.int64)
        )
        binary_confusion = binary_confusion + torch.bincount(binary_index, minlength=4).reshape(2, 2)

        # Family confusion
        if family_confusion is None:
            family_class_count = int(family_logits.shape[1])
            family_confusion = torch.zeros((family_class_count, family_class_count), dtype=torch.int64)
        else:
            family_class_count = int(family_confusion.shape[0])

        if not self.disable_integrity_hard_stops:
            invalid_label_mask = (y_family_cpu < 0) | (y_family_cpu >= family_class_count)
            if bool(torch.any(invalid_label_mask)):
                raise RuntimeError("Hard-stop integrity guard triggered: invalid_family_labels_in_test")

        family_index = (
            y_family_cpu.to(dtype=torch.int64) * family_class_count
            + family_pred.detach().to(dtype=torch.int64)
        )
        family_confusion = family_confusion + torch.bincount(
            family_index,
            minlength=family_class_count * family_class_count,
        ).reshape(family_class_count, family_class_count)

        # Entropy
        safe_family_prob = torch.clamp(family_prob, min=1e-12, max=1.0)
        batch_entropy = -torch.sum(family_prob * torch.log(safe_family_prob), dim=1)
        batch_entropy = batch_entropy / math.log(float(family_prob.shape[1]))
        entropy_sum = float(batch_entropy.sum().item())

        return binary_prob[:, 1].detach(), y_binary_cpu.detach(), binary_confusion, family_confusion, batch_size, entropy_sum

    @torch.no_grad()
    def _evaluate_test_loader(
        self,
        test_loader: DataLoader,
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> dict[str, float]:
        """Evaluate one test loader with tensor-first aggregation.

        Equivalent to HelixFullTrainer._evaluate_test_loader.
        """
        binary_prob_chunks: list[torch.Tensor] = []
        binary_label_chunks: list[torch.Tensor] = []
        binary_confusion = torch.zeros((2, 2), dtype=torch.int64)
        family_confusion: torch.Tensor | None = None
        family_entropy_sum = 0.0
        total_samples = 0

        for x, y_binary, y_family in test_loader:
            binary_prob, y_binary_cpu, binary_confusion, family_confusion, batch_size, entropy_sum = (
                self._process_test_batch(
                    x, y_binary, y_family, binary_confusion, family_confusion,
                    active_family_class_ids=active_family_class_ids,
                    class4_logit_shift=class4_logit_shift,
                    class4_logit_shift_class_id=class4_logit_shift_class_id,
                )
            )

            binary_prob_chunks.append(binary_prob)
            binary_label_chunks.append(y_binary_cpu)
            family_entropy_sum += entropy_sum
            total_samples += batch_size

        binary_probs_arr = (
            torch.cat(binary_prob_chunks, dim=0).to(device="cpu").numpy()
            if binary_prob_chunks
            else np.array([])
        )
        binary_labels_arr = (
            torch.cat(binary_label_chunks, dim=0).to(device="cpu").numpy()
            if binary_label_chunks
            else np.array([])
        )

        binary_total = int(binary_confusion.sum().item())
        binary_accuracy = (
            float(torch.diag(binary_confusion).sum().item() / binary_total)
            if binary_total > 0
            else 0.0
        )

        family_total = int(family_confusion.sum().item()) if family_confusion is not None else 0
        family_accuracy = (
            float(torch.diag(family_confusion).sum().item() / family_total)
            if family_total > 0 and family_confusion is not None
            else 0.0
        )

        if binary_labels_arr.size > 0 and np.unique(binary_labels_arr).size > 1:
            binary_auroc = float(roc_auc_score(binary_labels_arr, binary_probs_arr))
            binary_auprc = float(average_precision_score(binary_labels_arr, binary_probs_arr))
        else:
            binary_auroc = 0.0
            binary_auprc = 0.0

        family_entropy = float(family_entropy_sum / max(1, total_samples)) if total_samples > 0 else 0.0

        binary_stats = self._compute_f1_stats_from_confusion(binary_confusion)
        family_stats = self._compute_f1_stats_from_confusion(
            family_confusion if family_confusion is not None else torch.zeros((0, 0), dtype=torch.int64)
        )

        return {
            "binary_accuracy": binary_accuracy,
            "binary_f1": float(binary_stats["weighted_f1"]),
            "binary_auroc": binary_auroc,
            "binary_auprc": binary_auprc,
            "family_accuracy": family_accuracy,
            "family_f1": float(family_stats["weighted_f1"]),
            "family_macro_f1": float(family_stats["macro_f1"]),
            "family_minority_recall_min": float(family_stats["minority_recall_min"]),
            "family_entropy": family_entropy,
            "family_zero_prediction_classes": float(
                len(cast(list[int], family_stats["zero_prediction_classes"]))
            ),
        }

    @torch.no_grad()
    def evaluate_per_dataset(
        self,
        test_loaders: dict[str, DataLoader],
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate on per-dataset test sets.

        Equivalent to HelixFullTrainer.evaluate_per_dataset.
        """
        self.model.eval()
        results: dict[str, dict[str, float]] = {}
        for dataset_name, test_loader in test_loaders.items():
            results[dataset_name] = self._evaluate_test_loader(
                test_loader,
                active_family_class_ids=active_family_class_ids,
                class4_logit_shift=class4_logit_shift,
                class4_logit_shift_class_id=class4_logit_shift_class_id,
            )
        return results
