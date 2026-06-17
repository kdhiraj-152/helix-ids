"""Epoch runner — single training epoch iteration."""

import time
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.training.execution.batch_processor import BatchProcessor
from scripts.training.execution.warmup_manager import WarmupManager


class EpochRunner:
    """Run one training epoch: batch iteration, accumulation, logging.

    The constructor receives stable references (model, data loader,
    batch processor, warmup manager).  Per-epoch configuration values
    and callable hooks for trainer state mutation are passed as keyword
    arguments to ``run_epoch``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        config: Any,
        device: torch.device,
        logger: Any,
        batch_processor: BatchProcessor,
        warmup_manager: WarmupManager,
        step10_symmetry_logged: bool = False,
    ) -> None:
        self._model = model
        self._train_loader = train_loader
        self._config = config
        self._device = device
        self._logger = logger
        self._batch_processor = batch_processor
        self._warmup_manager = warmup_manager
        self.step10_symmetry_logged = step10_symmetry_logged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_epoch(
        self,
        *,
        epoch: int,
        global_step: int,
        warmup_steps: int,
        representation_diagnostic_mode: bool,
        use_energy_based_family_objective: bool,
        active_family_class_ids: Optional[list[int]],
        enforce_all_classes_per_batch: bool,
        step_coverage_checked: bool,
        head_phase_start_step: int,
        representation_phase_active: bool,
        in_representation_window: bool,
        representation_curriculum_complete: bool,
        binary_class_weights: Optional[torch.Tensor],
        family_class_weights: Optional[torch.Tensor],
        family_log_prior: Optional[torch.Tensor],
        energy_logit_temperature: float,
        disable_tail_focal_regularizer: bool,
        optimizer: torch.optim.Optimizer,
        backbone_params: Optional[list[nn.Parameter]],
        # Callable hooks for stateful orchestration steps
        check_backbone_freeze_state: Callable[[], None],
        handle_representation_phase_logic: Callable[[bool], None],
        maybe_activate_joint_finetune_phase: Callable[[], None],
        check_family_class_coverage: Callable[[torch.Tensor], None],
        check_step_coverage: Callable[[int, Optional[torch.Tensor]], None],
        freeze_epoch_centroid_snapshot: Callable[[], None],
        update_centroids_from_epoch_buffer: Callable[[], None],
    ) -> dict[str, Any]:
        """Train for one epoch and return aggregated metrics.

        Returns a dict with:
          metrics (dict[str, float])  — epoch-level training metrics
          step10_symmetry_logged (bool)  — updated flag
          class_starvation_streak (int)  — updated streak counter
          global_step (int)  — updated global step counter
          family_pred_counts (Optional[torch.Tensor])
          family_logit_sums (Optional[torch.Tensor])
        """
        self._model.train()

        total_loss = 0.0
        total_calibrated_loss = 0.0
        total_binary_correct = 0
        total_family_correct = 0
        total_samples = 0
        train_logit_max = float("-inf")
        train_logit_min = float("inf")
        family_pred_counts: Optional[torch.Tensor] = None
        family_logit_sums: Optional[torch.Tensor] = None
        top2_logit_gap_sum = 0.0
        top2_logit_gap_count = 0
        class_starvation_streak = 0

        step_log_interval = max(1, int(self._config.log_interval))
        total_steps = len(self._train_loader)
        epoch_start = time.perf_counter()

        active_strategy = self._set_epoch_loss_strategy(
            representation_diagnostic_mode,
        )
        self._logger.info(
            "Epoch %d start | steps=%d | step_log_interval=%d | loss_strategy=%s",
            epoch,
            total_steps,
            step_log_interval,
            active_strategy,
        )

        # --- Warmup ---
        warmup_result = self._warmup_manager.run_warmup(
            self._train_loader.dataset,
            epoch=epoch,
            model_training=self._model.training,
            global_step=global_step,
            warmup_steps=warmup_steps,
            active_family_class_ids=active_family_class_ids,
            use_energy_based_family_objective=use_energy_based_family_objective,
            binary_class_weights=binary_class_weights,
            family_class_weights=family_class_weights,
            backbone_params=backbone_params,
            logger=self._logger,
        )
        gs = global_step
        if warmup_result["warmup_executed"]:
            gs += warmup_result["global_step_increment"]

        if not use_energy_based_family_objective:
            freeze_epoch_centroid_snapshot()

        # --- Batch loop ---
        for batch_idx, (x, y_binary, y_family) in enumerate(self._train_loader):
            if batch_idx % step_log_interval == 0:
                print(
                    f"step {batch_idx}/{total_steps} status=start epoch={epoch}",
                    flush=True,
                )

            check_backbone_freeze_state()

            x = x.to(self._device, non_blocking=True)
            y_binary = y_binary.to(self._device, non_blocking=True)
            y_family = y_family.to(self._device, non_blocking=True)

            unique_classes_in_batch = int(
                torch.unique(y_family, dim=0).numel()
            )
            if unique_classes_in_batch < 2:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: "
                    "batch_diversity_violation_lt2"
                )
            self._logger.info(
                "BatchClassCoverage step=%d batch=%d unique_classes_in_batch=%d",
                int(gs),
                int(batch_idx),
                unique_classes_in_batch,
            )
            if unique_classes_in_batch < 3:
                class_starvation_streak += 1
                if class_starvation_streak >= 5:
                    self._logger.warning(
                        "BatchClassCoverage starvation_detected streak=%d "
                        "(<3 unique classes)",
                        int(class_starvation_streak),
                    )
            else:
                class_starvation_streak = 0

            check_family_class_coverage(y_family)

            in_representation_phase = (
                representation_diagnostic_mode
                and self._is_representation_window_step(
                    gs, representation_diagnostic_mode,
                )
            )
            handle_representation_phase_logic(in_representation_phase)
            maybe_activate_joint_finetune_phase()

            in_step_warmup = gs < warmup_steps
            result = self._batch_processor.process_batch(
                x,
                y_binary,
                y_family,
                in_step_warmup=in_step_warmup,
                in_representation_phase=in_representation_phase,
                optimizer=optimizer,
                backbone_params=backbone_params,
                global_step=gs,
                warmup_steps=warmup_steps,
                binary_class_weights=binary_class_weights,
                family_class_weights=family_class_weights,
                active_family_class_ids=(
                    active_family_class_ids if active_family_class_ids else []
                ),
                use_energy_based_family_objective=use_energy_based_family_objective,
                disable_tail_focal_regularizer=disable_tail_focal_regularizer,
                energy_logit_temperature=energy_logit_temperature,
                family_log_prior=family_log_prior,
            )
            gs += result["global_step_increment"]

            raw_family_logits = result["raw_family_logits"]
            train_logit_max = max(
                train_logit_max, float(raw_family_logits.max().item())
            )
            train_logit_min = min(
                train_logit_min, float(raw_family_logits.min().item())
            )

            loss = result["loss"]
            batch_size = result["batch_size"]
            total_loss += float(loss.item()) * batch_size
            total_calibrated_loss += float(loss.item()) * batch_size
            total_binary_correct += result["binary_correct"]
            total_family_correct += result["family_correct"]
            total_samples += batch_size

            family_pred_counts, family_logit_sums = self._update_train_batch_stats(
                family_pred_counts,
                family_logit_sums,
                result["family_pred"],
                raw_family_logits,
            )

            check_step_coverage(batch_idx, family_pred_counts)

            top2_values = torch.topk(
                raw_family_logits.detach(), k=2, dim=1
            ).values
            top2_logit_gap_sum += float(
                (top2_values[:, 0] - top2_values[:, 1]).sum().item()
            )
            top2_logit_gap_count += int(top2_values.shape[0])

            self._log_step10_diagnostics(raw_family_logits, gs)

            self._log_batch_progress(
                batch_idx,
                total_steps,
                step_log_interval,
                total_loss,
                total_binary_correct,
                total_family_correct,
                total_samples,
                epoch,
            )

        # --- Epoch completion ---
        self._log_epoch_completion(
            epoch_start,
            train_logit_min,
            train_logit_max,
            family_pred_counts,
            family_logit_sums,
            total_samples,
            top2_logit_gap_sum,
            top2_logit_gap_count,
            epoch,
        )

        if not use_energy_based_family_objective:
            update_centroids_from_epoch_buffer()

        metrics = {
            "train_loss": total_loss / max(1, total_samples),
            "train_calibrated_loss": total_calibrated_loss / max(1, total_samples),
            "train_binary_acc": total_binary_correct / max(1, total_samples),
            "train_family_acc": total_family_correct / max(1, total_samples),
            "train_family_logit_max": train_logit_max,
            "train_family_logit_min": train_logit_min,
        }

        return {
            "metrics": metrics,
            "global_step": gs,
            "class_starvation_streak": class_starvation_streak,
            "family_pred_counts": family_pred_counts,
            "family_logit_sums": family_logit_sums,
            "step10_symmetry_logged": self.step10_symmetry_logged,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_epoch_loss_strategy(
        representation_diagnostic_mode: bool,
    ) -> str:
        """Determine and log the active loss strategy for this epoch."""
        return "rep_diag" if representation_diagnostic_mode else "standard"

    @staticmethod
    def _is_representation_window_step(
        step: int,
        representation_diagnostic_mode: bool,
    ) -> bool:
        """Check if current step falls within representation window."""
        if not representation_diagnostic_mode:
            return False
        return 0 <= step <= 499  # hardcoded to match trainer

    @staticmethod
    def _update_train_batch_stats(
        family_pred_counts: Optional[torch.Tensor],
        family_logit_sums: Optional[torch.Tensor],
        family_pred: torch.Tensor,
        raw_family_logits: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Update prediction counts and logit sums for diagnostics."""
        class_count = int(raw_family_logits.shape[1])
        if family_pred_counts is None:
            family_pred_counts = torch.zeros(class_count, dtype=torch.int64)
            family_logit_sums = torch.zeros(class_count, dtype=torch.float32)

        family_pred_counts += torch.bincount(
            family_pred.detach().to(device="cpu", dtype=torch.int64),
            minlength=class_count,
        )

        if family_logit_sums is not None:
            family_logit_sums += raw_family_logits.detach().to(
                device="cpu", dtype=torch.float32
            ).sum(dim=0)

        return family_pred_counts, family_logit_sums

    def _log_step10_diagnostics(
        self,
        raw_family_logits: torch.Tensor,
        global_step: int,
    ) -> None:
        """Log per-class logit statistics at step 10."""
        if (not self.step10_symmetry_logged) and global_step >= 10:
            mean_logits = raw_family_logits.detach().to(
                device="cpu", dtype=torch.float32
            ).mean(dim=0)
            std_logits = raw_family_logits.detach().to(
                device="cpu", dtype=torch.float32
            ).std(dim=0)
            mean_payload = {
                int(i): float(v) for i, v in enumerate(mean_logits.tolist())
            }
            std_payload = {
                int(i): float(v) for i, v in enumerate(std_logits.tolist())
            }
            self._logger.info(
                "Step10Diag: per_class_mean_logit=%s per_class_std_logit=%s",
                mean_payload,
                std_payload,
            )
            self.step10_symmetry_logged = True

    def _log_batch_progress(
        self,
        batch_idx: int,
        total_steps: int,
        step_log_interval: int,
        total_loss: float,
        total_binary_correct: int,
        total_family_correct: int,
        total_samples: int,
        epoch: int,
    ) -> None:
        """Log batch training progress."""
        if batch_idx % step_log_interval == 0:
            avg_loss = total_loss / max(1, total_samples)
            binary_acc = total_binary_correct / max(1, total_samples)
            family_acc = total_family_correct / max(1, total_samples)
            print(
                f"step {batch_idx}/{total_steps} loss {avg_loss:.4f} "
                f"binary_acc {binary_acc:.4f} family_acc {family_acc:.4f}",
                flush=True,
            )
            self._logger.info(
                f"Epoch {epoch} [{batch_idx}/{total_steps}] "
                f"Loss: {avg_loss:.4f} | "
                f"Binary Acc: {binary_acc:.4f} | "
                f"Family Acc: {family_acc:.4f} | "
            )

    def _log_epoch_completion(
        self,
        epoch_start: float,
        train_logit_min: float,
        train_logit_max: float,
        family_pred_counts: Optional[torch.Tensor],
        family_logit_sums: Optional[torch.Tensor],
        total_samples: int,
        top2_logit_gap_sum: float,
        top2_logit_gap_count: int,
        epoch: int,
    ) -> None:
        """Log metrics at end of training epoch."""
        elapsed = time.perf_counter() - epoch_start
        self._logger.info(
            "Epoch %d complete | elapsed=%.2fs", epoch, elapsed,
        )
        self._logger.info(
            "Epoch %d logit_range raw_family[min=%.4f max=%.4f]",
            epoch,
            train_logit_min,
            train_logit_max,
        )
        avg_top2_logit_gap = top2_logit_gap_sum / max(1, top2_logit_gap_count)
        if (
            family_pred_counts is not None
            and family_logit_sums is not None
            and total_samples > 0
        ):
            pred_count_payload = {
                int(idx): int(count)
                for idx, count in enumerate(family_pred_counts.tolist())
            }
            avg_logit_payload = {
                int(idx): float(total / max(1, total_samples))
                for idx, total in enumerate(family_logit_sums.tolist())
            }
            self._logger.info(
                "Epoch %d diagnostics: per_class_prediction_count=%s "
                "per_class_avg_logit=%s top2_logit_gap=%.4f",
                epoch,
                pred_count_payload,
                avg_logit_payload,
                avg_top2_logit_gap,
            )
        if train_logit_max > 10.0 or train_logit_min < -10.0:
            self._logger.warning(
                "Epoch %d logit saturation risk detected: "
                "raw_family[min=%.4f max=%.4f]",
                epoch,
                train_logit_min,
                train_logit_max,
            )
