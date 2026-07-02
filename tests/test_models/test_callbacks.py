"""
Focused unit tests for helix_ids/utils/callbacks.py
Raises coverage from ~20% to 70%+.
"""
import json
import logging
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.helix_ids.utils.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    TrainingLogger,
)

# =========================================================================
# Helpers
# =========================================================================


@pytest.fixture
def simple_model():
    """A tiny torch model for callback tests."""
    return nn.Linear(4, 2)


@pytest.fixture
def simple_optimizer(simple_model):
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def tmp_dir():
    """Yield a temporary directory path that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# =========================================================================
# EarlyStopping
# =========================================================================


class TestEarlyStopping:
    """Coverage target: EarlyStopping class (~140 lines)."""

    def test_max_mode_improves_and_stops(self):
        """Max mode: improving metric resets wait, patience exceeded stops."""
        es = EarlyStopping(monitor="f1", patience=2, mode="max", min_delta=0.0, verbose=False)

        # Epoch 0: f1=0.8  — improvement (best goes from -inf to 0.8)
        assert es.on_epoch_end(0, {"f1": 0.8}) is True
        assert es.best == 0.8
        assert es.wait == 0

        # Epoch 1: f1=0.85 — improvement
        assert es.on_epoch_end(1, {"f1": 0.85}) is True
        assert es.best == 0.85
        assert es.best_epoch == 1

        # Epoch 2: f1=0.83 — no improvement (not > 0.85)
        assert es.on_epoch_end(2, {"f1": 0.83}) is True
        assert es.wait == 1

        # Epoch 3: f1=0.82 — no improvement, wait=2 >= patience=2 -> stop
        assert es.on_epoch_end(3, {"f1": 0.82}) is False
        assert es.stopped_epoch == 3
        assert es.wait == 2

    def test_min_mode(self):
        """Min mode: lower metric values are better."""
        es = EarlyStopping(monitor="loss", patience=2, mode="min", verbose=False)

        # Epoch 0: loss=1.0 — improvement (best goes from +inf to 1.0)
        assert es.on_epoch_end(0, {"loss": 1.0}) is True
        assert es.best == 1.0

        # Epoch 1: loss=0.9 — improvement (0.9 < 1.0 - 0.0)
        assert es.on_epoch_end(1, {"loss": 0.9}) is True
        assert es.best == 0.9

        # Epoch 2: loss=0.9 — no improvement (not < 0.9 - 0.0), wait=1 < 2 -> continues
        assert es.on_epoch_end(2, {"loss": 0.9}) is True
        assert es.wait == 1

        # Epoch 3: loss=0.95 — no improvement, wait=2 >= patience=2 -> stop
        assert es.on_epoch_end(3, {"loss": 0.95}) is False

    def test_patience_zero_stops_immediately(self):
        """patience=0 means stop even after improvement (wait < 0 is False)."""
        es = EarlyStopping(monitor="acc", patience=0, mode="max", verbose=False)

        # Wait starts at 0, improvement sets wait back to 0, but 0 < 0 is False
        assert es.on_epoch_end(0, {"acc": 0.9}) is False

    def test_min_delta_prevents_noise(self):
        """min_delta filters out small fluctuations."""
        es = EarlyStopping(monitor="f1", patience=2, mode="max", min_delta=0.1, verbose=False)

        # Epoch 0: f1=0.8 — improvement
        assert es.on_epoch_end(0, {"f1": 0.8}) is True
        assert es.best == 0.8

        # Epoch 1: f1=0.85 — not enough improvement (0.85 > 0.8 + 0.1? No)
        assert es.on_epoch_end(1, {"f1": 0.85}) is True
        assert es.wait == 1          # not an improvement
        assert es.best == 0.8        # unchanged

        # Epoch 2: f1=0.91 — enough (0.91 > 0.8 + 0.1) -> improvement
        assert es.on_epoch_end(2, {"f1": 0.91}) is True
        assert es.best == 0.91
        assert es.wait == 0

    def test_metric_missing_warns_and_continues(self, caplog):
        """When monitored metric is absent warn and continue."""
        es = EarlyStopping(monitor="missing_metric", patience=2, verbose=False)
        caplog.set_level(logging.WARNING)

        result = es.on_epoch_end(0, {"loss": 0.5})
        assert result is True
        assert "missing_metric" in caplog.text

    def test_baseline_max_mode(self):
        """baseline sets the initial 'best' threshold; must exceed to improve."""
        es = EarlyStopping(monitor="f1", patience=2, mode="max", baseline=0.9, verbose=False)

        # Epoch 0: f1=0.85 — not > baseline 0.9, no improvement
        assert es.on_epoch_end(0, {"f1": 0.85}) is True
        assert es.best == 0.9  # stays at baseline
        assert es.wait == 1

        # Epoch 1: f1=0.95 — exceeds baseline, improvement
        assert es.on_epoch_end(1, {"f1": 0.95}) is True
        assert es.best == 0.95
        assert es.wait == 0

    def test_baseline_min_mode(self):
        """baseline for min mode: metric must go below baseline."""
        es = EarlyStopping(monitor="loss", patience=2, mode="min", baseline=1.0, verbose=False)

        # Epoch 0: loss=1.5 — not < baseline 1.0, no improvement
        assert es.on_epoch_end(0, {"loss": 1.5}) is True
        assert es.best == 1.0
        assert es.wait == 1

        # Epoch 1: loss=0.5 — below baseline, improvement
        assert es.on_epoch_end(1, {"loss": 0.5}) is True
        assert es.best == 0.5
        assert es.wait == 0

    def test_get_best_metric_and_epoch(self):
        """get_best_metric / get_best_epoch return correct values."""
        es = EarlyStopping(monitor="f1", patience=3, mode="max", verbose=False)

        es.on_epoch_end(0, {"f1": 0.7})
        es.on_epoch_end(1, {"f1": 0.9})
        es.on_epoch_end(2, {"f1": 0.8})

        assert es.get_best_metric() == 0.9
        assert es.get_best_epoch() == 1

    def test_restore_best_weights(self, simple_model):
        """restore_best_weights saves and restores model state dict."""
        # Set different weights so we can verify restoration
        nn.init.constant_(simple_model.weight, 1.0)
        nn.init.constant_(simple_model.bias, 0.0)
        original_weights = {k: v.clone() for k, v in simple_model.state_dict().items()}

        es = EarlyStopping(
            monitor="f1", patience=1, mode="max",
            restore_best_weights=True, verbose=False,
        )
        es.set_model(simple_model)

        # Epoch 0: improvement — saves weights (weight=1.0)
        es.on_epoch_end(0, {"f1": 0.8})
        assert es.best_weights is not None

        # Change model weights drastically
        nn.init.constant_(simple_model.weight, 99.0)
        nn.init.constant_(simple_model.bias, 99.0)

        # Epoch 1: no improvement — triggers restore
        es.on_epoch_end(1, {"f1": 0.7})

        # Best weights should be restored
        for key in original_weights:
            assert torch.allclose(simple_model.state_dict()[key], original_weights[key]), \
                f"{key} was not restored"

    def test_on_train_begin_resets_state(self):
        """on_train_begin resets all internal state."""
        es = EarlyStopping(monitor="f1", patience=2, mode="max", baseline=0.5, verbose=False)

        # Run some epochs
        es.on_epoch_end(0, {"f1": 0.9})
        es.on_epoch_end(1, {"f1": 0.8})
        es.on_epoch_end(2, {"f1": 0.7})  # patience exceeded

        assert es.stopped_epoch == 2
        assert es.wait == 2

        # Reset
        es.on_train_begin()
        assert es.wait == 0
        assert es.stopped_epoch == 0
        assert es.best_epoch == 0
        assert es.best_weights is None
        assert es.best == 0.5  # re-initialized to baseline

    def test_no_model_no_crash_on_restore(self):
        """restore_best_weights with no model set doesn't crash."""
        es = EarlyStopping(monitor="f1", patience=1, mode="max", restore_best_weights=True)
        es.on_epoch_end(0, {"f1": 0.8})
        # No model set → _handle_improvement skips saving weights
        assert es.best_weights is None
        # _handle_no_improvement checks for None before restoring
        es.on_epoch_end(1, {"f1": 0.7})  # should not crash


# =========================================================================
# Callback (base class)
# =========================================================================


class TestCallback:
    def test_base_class_methods(self):
        """Verify base Callback methods are callable and return None/True."""
        cb = Callback()

        cb.set_model(nn.Linear(2, 2))
        assert cb.model is not None

        cb.set_optimizer(torch.optim.SGD(cb.model.parameters(), lr=0.01))
        assert cb.optimizer is not None

        assert cb.on_train_begin() is None
        assert cb.on_train_end() is None
        assert cb.on_epoch_begin(0) is None
        assert cb.on_epoch_end(0) is True
        assert cb.on_batch_begin(0) is None
        assert cb.on_batch_end(0) is None


# =========================================================================
# ModelCheckpoint
# =========================================================================


class TestModelCheckpoint:
    """Coverage target: ModelCheckpoint class (~170 lines)."""

    def test_save_on_improvement(self, simple_model, tmp_dir):
        """Model is saved when metric improves (min mode)."""
        ckpt = ModelCheckpoint(
            filepath=tmp_dir / "model_{epoch}_{metric}.pt",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=False,
        )
        ckpt.set_model(simple_model)
        ckpt.on_train_begin({})

        # Epoch 0: loss=1.0 — save (first)
        assert ckpt.on_epoch_end(0, {"val_loss": 1.0}) is True
        assert (tmp_dir / "model_001_1.0000.pt").exists()
        assert ckpt.best == 1.0

        # Epoch 1: loss=0.5 — improvement, save again
        assert ckpt.on_epoch_end(1, {"val_loss": 0.5}) is True
        assert (tmp_dir / "model_002_0.5000.pt").exists()
        assert ckpt.best == 0.5

    def test_save_best_only_skips_non_improving(self, simple_model, tmp_dir):
        """When save_best_only=True and metric doesn't improve, no save."""
        ckpt = ModelCheckpoint(
            filepath=tmp_dir / "ckpt_{epoch}.pt",
            monitor="acc",
            mode="max",
            save_best_only=True,
            verbose=False,
        )
        ckpt.set_model(simple_model)
        ckpt.on_train_begin({})

        ckpt.on_epoch_end(0, {"acc": 0.9})
        n_files_after_epoch0 = len(list(tmp_dir.iterdir()))

        # Epoch 1: acc=0.8 — worse, should NOT save
        ckpt.on_epoch_end(1, {"acc": 0.8})
        n_files_after_epoch1 = len(list(tmp_dir.iterdir()))
        assert n_files_after_epoch1 == n_files_after_epoch0, \
            "No new file should be created for non-improving metric"

    def test_save_every_epoch_when_not_best_only(self, simple_model, tmp_dir):
        """When save_best_only=False, every epoch saves."""
        ckpt = ModelCheckpoint(
            filepath=tmp_dir / "epoch_{epoch}.pt",
            monitor="acc",
            mode="max",
            save_best_only=False,
            verbose=False,
        )
        ckpt.set_model(simple_model)
        ckpt.on_train_begin({})

        ckpt.on_epoch_end(0, {"acc": 0.9})
        ckpt.on_epoch_end(1, {"acc": 0.8})  # worse but still saved
        ckpt.on_epoch_end(2, {"acc": 0.95})

        assert (tmp_dir / "epoch_001.pt").exists()
        assert (tmp_dir / "epoch_002.pt").exists()
        assert (tmp_dir / "epoch_003.pt").exists()

    def test_save_weights_only(self, simple_model, tmp_dir):
        """save_weights_only=True still saves (no crash)."""
        ckpt = ModelCheckpoint(
            filepath=tmp_dir / "w_{epoch}.pt",
            monitor="loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=False,
        )
        ckpt.set_model(simple_model)
        ckpt.on_train_begin({})
        ckpt.on_epoch_end(0, {"loss": 0.5})

        assert (tmp_dir / "w_001.pt").exists()
        # Verify it's a valid checkpoint
        data = torch.load(tmp_dir / "w_001.pt", map_location="cpu", weights_only=False)
        assert "model_state_dict" in data
        assert "epoch" in data

    def test_filepath_formatting_with_placeholders(self, simple_model, tmp_dir):
        """{epoch}, {metric}, {monitor} and arbitrary metric keys are replaced."""
        ckpt = ModelCheckpoint(
            filepath=tmp_dir / "{monitor}_{epoch}_{val_f1}.pt",
            monitor="val_loss",
            mode="min",
            verbose=False,
        )
        ckpt.set_model(simple_model)
        ckpt.on_train_begin({})
        ckpt.on_epoch_end(0, {"val_loss": 0.5, "val_f1": 0.92})

        expected = tmp_dir / "val_loss_001_0.9200.pt"
        assert expected.exists(), f"Expected {expected}"

    def test_no_model_set_does_not_crash(self, caplog, tmp_dir):
        """Checkpoint without a model logs warning but doesn't crash."""
        ckpt = ModelCheckpoint(
            filepath=tmp_dir / "nope.pt",
            monitor="loss",
            mode="min",
            verbose=False,
        )
        # No model set
        caplog.set_level(logging.WARNING)
        result = ckpt.on_epoch_end(0, {"loss": 0.5})

        assert result is True
        assert "No model set" in caplog.text

    def test_metric_missing_warns_and_continues(self, caplog, tmp_dir):
        """When monitored metric is absent log warning and continue."""
        ckpt = ModelCheckpoint(
            filepath=tmp_dir / "x.pt",
            monitor="missing",
            verbose=False,
        )
        ckpt.set_model(simple_model)
        caplog.set_level(logging.WARNING)
        result = ckpt.on_epoch_end(0, {"loss": 0.5})

        assert result is True
        assert "not found in logs" in caplog.text

    def test_get_best_filepath(self, simple_model, tmp_dir):
        """get_best_filepath returns last saved path."""
        ckpt = ModelCheckpoint(
            filepath=tmp_dir / "best_{epoch}.pt",
            monitor="acc",
            mode="max",
            verbose=False,
        )
        ckpt.set_model(simple_model)
        ckpt.on_train_begin({})
        ckpt.on_epoch_end(0, {"acc": 0.9})

        assert ckpt.get_best_filepath() is not None
        assert ckpt.get_best_filepath().name == "best_001.pt"

    def test_on_train_begin_creates_directory(self, simple_model, tmp_dir):
        """on_train_begin creates parent directories."""
        deep_dir = tmp_dir / "sub" / "nested"
        ckpt = ModelCheckpoint(
            filepath=deep_dir / "model.pt",
            monitor="loss",
            mode="min",
            verbose=False,
        )
        ckpt.set_model(simple_model)
        ckpt.on_train_begin({})
        assert deep_dir.exists()


# =========================================================================
# CallbackList
# =========================================================================


class TestCallbackList:
    """Coverage target: CallbackList class (~70 lines)."""

    def test_append(self):
        """append adds a callback."""
        cl = CallbackList()
        assert len(cl) == 0
        cl.append(Callback())
        assert len(cl) == 1
        cl.append(Callback())
        assert len(cl) == 2

    def test_extend(self):
        """extend adds multiple callbacks."""
        cl = CallbackList()
        cl.extend([Callback(), Callback()])
        assert len(cl) == 2

    def test_set_model_propagates(self):
        """set_model is called on all children."""
        cb1 = Callback()
        cb2 = Callback()
        cl = CallbackList([cb1, cb2])

        model = nn.Linear(2, 2)
        cl.set_model(model)
        assert cb1.model is model
        assert cb2.model is model

    def test_set_optimizer_propagates(self):
        """set_optimizer is called on all children."""
        cb1 = Callback()
        cb2 = Callback()
        cl = CallbackList([cb1, cb2])

        model = nn.Linear(2, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        cl.set_optimizer(opt)
        assert cb1.optimizer is opt
        assert cb2.optimizer is opt

    def test_on_epoch_end_returns_true_all_continue(self):
        """When all callbacks return True, on_epoch_end returns True."""
        cb1 = Callback()
        cb2 = Callback()
        cl = CallbackList([cb1, cb2])
        assert cl.on_epoch_end(0) is True

    def test_on_epoch_end_returns_false_if_any_stops(self):
        """When any callback returns False, on_epoch_end returns False."""
        class StoppingCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                return False

        cl = CallbackList([Callback(), StoppingCallback()])
        assert cl.on_epoch_end(0) is False

    def test_on_epoch_end_continues_if_others_fail(self):
        """All callbacks are called even if one returns False."""
        calls = []

        class TrackingCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                calls.append(epoch)
                return True

        class StoppingCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                calls.append(epoch)
                return False

        cl = CallbackList([TrackingCallback(), StoppingCallback(), TrackingCallback()])
        result = cl.on_epoch_end(42)
        assert result is False
        assert calls == [42, 42, 42]

    def test_iter_and_len(self):
        """__iter__ and __len__ work."""
        cb1, cb2 = Callback(), Callback()
        cl = CallbackList([cb1, cb2])
        assert len(cl) == 2
        assert list(cl) == [cb1, cb2]

    def test_delegates_hooks(self):
        """All hooks are delegated to child callbacks."""
        calls = {"train_begin": 0, "train_end": 0, "epoch_begin": 0, "batch_begin": 0, "batch_end": 0}

        class CountingCallback(Callback):
            def on_train_begin(self, logs=None):
                calls["train_begin"] += 1
            def on_train_end(self, logs=None):
                calls["train_end"] += 1
            def on_epoch_begin(self, epoch, logs=None):
                calls["epoch_begin"] += 1
            def on_batch_begin(self, batch, logs=None):
                calls["batch_begin"] += 1
            def on_batch_end(self, batch, logs=None):
                calls["batch_end"] += 1

        cl = CallbackList([CountingCallback(), CountingCallback()])
        cl.on_train_begin()
        cl.on_epoch_begin(0)
        cl.on_batch_begin(0)
        cl.on_batch_end(0)
        cl.on_epoch_end(0)
        cl.on_train_end()

        assert calls == {"train_begin": 2, "train_end": 2, "epoch_begin": 2, "batch_begin": 2, "batch_end": 2}

    def test_empty_callback_list(self):
        """An empty CallbackList is safe to call methods on."""
        cl = CallbackList()
        assert len(cl) == 0
        assert cl.on_epoch_end(0) is True
        cl.on_train_begin()  # no error
        cl.on_train_end()
        cl.set_model(nn.Linear(2, 2))
        cl.set_optimizer(torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.01))


# =========================================================================
# LearningRateScheduler
# =========================================================================


class TestLearningRateScheduler:
    """Coverage target: LearningRateScheduler class (~130 lines)."""

    def test_warmup_cosine_epoch_0(self, simple_optimizer):
        """Epoch 0 in warmup phase gives minimal LR."""
        sched = LearningRateScheduler(
            schedule_type="warmup_cosine",
            initial_lr=1e-3,
            warmup_epochs=5,
            total_epochs=100,
            min_lr=1e-6,
            verbose=False,
        )
        sched.set_optimizer(simple_optimizer)
        sched.on_epoch_begin(0, {})

        expected_lr = 1e-3 * (1 / 5)  # 2e-4
        assert abs(sched.current_lr - expected_lr) < 1e-8
        assert len(sched.lr_history) == 1

    def test_warmup_cosine_mid(self, simple_optimizer):
        """Epoch in warmup phase."""
        sched = LearningRateScheduler(
            schedule_type="warmup_cosine",
            initial_lr=1e-3,
            warmup_epochs=5,
            total_epochs=100,
            min_lr=1e-6,
            verbose=False,
        )
        sched.set_optimizer(simple_optimizer)
        sched.on_epoch_begin(2, {})

        expected_lr = 1e-3 * (3 / 5)  # 6e-4
        assert abs(sched.current_lr - expected_lr) < 1e-8

    def test_warmup_cosine_after_warmup(self, simple_optimizer):
        """Epoch after warmup: cosine decay."""
        sched = LearningRateScheduler(
            schedule_type="warmup_cosine",
            initial_lr=1e-3,
            warmup_epochs=2,
            total_epochs=10,
            min_lr=1e-6,
            verbose=False,
        )
        sched.set_optimizer(simple_optimizer)
        sched.on_epoch_begin(5, {})

        # Cosine decay at epoch 5: progress = (5-2)/(10-2) = 3/8
        # cos_decay = 0.5 * (1 + cos(pi * 3/8))
        import math
        progress = (5 - 2) / (10 - 2)
        expected_lr = 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + math.cos(math.pi * progress))
        assert abs(sched.current_lr - expected_lr) < 1e-8

    def test_step_lr_with_step_size(self, simple_optimizer):
        """Step decay using step_size."""
        sched = LearningRateScheduler(
            schedule_type="step",
            initial_lr=1e-3,
            step_size=3,
            gamma=0.1,
            verbose=False,
        )
        sched.set_optimizer(simple_optimizer)

        sched.on_epoch_begin(0, {})
        assert abs(sched.current_lr - 1e-3) < 1e-8

        sched.on_epoch_begin(2, {})
        assert abs(sched.current_lr - 1e-3) < 1e-8

        sched.on_epoch_begin(3, {})
        # epoch 3 // 3 = 1 => lr = 1e-3 * 0.1^1 = 1e-4
        assert abs(sched.current_lr - 1e-4) < 1e-8

    def test_step_lr_with_milestones(self, simple_optimizer):
        """Step decay using milestones list."""
        sched = LearningRateScheduler(
            schedule_type="step",
            initial_lr=1e-3,
            milestones=[2, 5],
            gamma=0.1,
            verbose=False,
        )
        sched.set_optimizer(simple_optimizer)

        sched.on_epoch_begin(0, {})
        assert abs(sched.current_lr - 1e-3) < 1e-8

        sched.on_epoch_begin(2, {})
        assert abs(sched.current_lr - 1e-4) < 1e-8  # *0.1 once

        sched.on_epoch_begin(5, {})
        assert abs(sched.current_lr - 1e-5) < 1e-8  # *0.1 twice

    def test_custom_schedule_fn(self, simple_optimizer):
        """Custom schedule function is called and result applied."""
        def my_schedule(epoch, initial_lr):
            return initial_lr * (0.5 ** epoch)

        sched = LearningRateScheduler(
            schedule_type="custom",
            initial_lr=1e-3,
            schedule_fn=my_schedule,
            verbose=False,
        )
        sched.set_optimizer(simple_optimizer)
        sched.on_epoch_begin(2, {})
        assert abs(sched.current_lr - 2.5e-4) < 1e-8

    def test_get_lr_history(self, simple_optimizer):
        """get_lr_history returns recorded LR values."""
        sched = LearningRateScheduler(verbose=False)
        sched.set_optimizer(simple_optimizer)
        sched.on_epoch_begin(0, {})
        sched.on_epoch_begin(1, {})

        history = sched.get_lr_history()
        assert len(history) == 2

    def test_no_optimizer_no_crash(self):
        """Without optimizer, on_epoch_begin still computes LR and logs."""
        sched = LearningRateScheduler(verbose=False)
        # No optimizer set
        sched.on_epoch_begin(0, {})
        assert sched.current_lr > 0
        assert len(sched.lr_history) == 1

    def test_verbose_output(self, capsys, simple_optimizer):
        """Verbose mode prints LR changes."""
        sched = LearningRateScheduler(
            schedule_type="step",
            initial_lr=1e-3,
            step_size=2,
            gamma=0.5,
            verbose=True,
        )
        sched.set_optimizer(simple_optimizer)

        sched.on_epoch_begin(0, {})
        captured = capsys.readouterr()
        # No change on epoch 0 (same LR), verbose should not print
        assert "LearningRateScheduler" not in captured.out

        sched.on_epoch_begin(2, {})  # lr changes
        captured = capsys.readouterr()
        assert "LearningRateScheduler" in captured.out

    def test_unsupported_schedule_falls_back_to_initial(self, simple_optimizer):
        """Unsupported schedule_type returns initial_lr."""
        sched = LearningRateScheduler(
            schedule_type="unknown_type",
            initial_lr=0.42,
            verbose=False,
        )
        sched.set_optimizer(simple_optimizer)
        sched.on_epoch_begin(5, {})
        assert abs(sched.current_lr - 0.42) < 1e-8


# =========================================================================
# TrainingLogger
# =========================================================================


class TestTrainingLogger:
    """Coverage target: TrainingLogger class (~180 lines)."""

    def test_initialization(self, tmp_dir):
        """Basic init creates logger with default state."""
        logger = TrainingLogger(log_dir=tmp_dir, verbose=False)
        assert logger.log_dir == tmp_dir
        assert logger.history == []
        assert logger.start_time is None
        assert logger.tb_writer is None
        assert logger._tb_available is False

    def test_on_train_begin_creates_dir_and_prints(self, tmp_dir, capsys):
        """on_train_begin creates log directory and prints banner."""
        logger = TrainingLogger(log_dir=tmp_dir, verbose=True)
        logger.on_train_begin()
        assert tmp_dir.exists()

        captured = capsys.readouterr()
        assert "HELIX-IDS TRAINING" in captured.out

    def test_on_epoch_end_records_history(self, tmp_dir):
        """Each epoch adds a log entry."""
        logger = TrainingLogger(log_dir=tmp_dir, verbose=False)
        logger.on_train_begin()
        logger.on_epoch_end(0, {"loss": 1.0, "acc": 0.8})
        logger.on_epoch_end(1, {"loss": 0.5, "acc": 0.9})

        assert len(logger.history) == 2
        assert logger.history[0]["epoch"] == 1
        assert logger.history[0]["loss"] == 1.0
        assert logger.history[1]["epoch"] == 2
        assert logger.history[1]["loss"] == 0.5

    def test_on_epoch_end_saves_json(self, tmp_dir):
        """JSON log file is written after each epoch."""
        logger = TrainingLogger(log_dir=tmp_dir, verbose=False)
        logger.on_train_begin()
        logger.on_epoch_end(0, {"loss": 1.0})

        log_file = tmp_dir / "training_log.json"
        assert log_file.exists()

        data = json.loads(log_file.read_text())
        assert data["start_time"] is not None
        assert len(data["history"]) == 1

    def test_on_train_end_finalizes_and_closes(self, tmp_dir):
        """on_train_end finalizes the log and writes summary."""
        logger = TrainingLogger(log_dir=tmp_dir, verbose=False)
        logger.on_train_begin()
        logger.on_epoch_end(0, {"loss": 1.0})
        logger.on_epoch_end(1, {"loss": 0.5})
        logger.on_train_end()

        log_file = tmp_dir / "training_log.json"
        data = json.loads(log_file.read_text())
        assert data["end_time"] is not None
        assert data["total_epochs"] == 2
        assert data["duration_seconds"] is not None

    def test_verbose_epoch_end_prints(self, tmp_dir, capsys):
        """Verbose mode prints epoch summary."""
        logger = TrainingLogger(log_dir=tmp_dir, verbose=True, log_every_n_epochs=1)
        logger.on_train_begin()
        logger.on_epoch_end(0, {"train_loss": 1.0, "val_accuracy": 0.85, "val_macro_f1": 0.82})

        captured = capsys.readouterr()
        assert "Epoch    1" in captured.out
        assert "Loss: 1.0000" in captured.out

    def test_console_format_simple(self, tmp_dir, capsys):
        """Non-default console format prints key:value pairs."""
        logger = TrainingLogger(
            log_dir=tmp_dir,
            verbose=True,
            log_every_n_epochs=1,
            console_format="simple",
        )
        logger.on_train_begin()
        logger.on_epoch_end(0, {"loss": 0.5, "acc": 0.9})

        captured = capsys.readouterr()
        assert "loss: 0.5000" in captured.out or "loss: 0.5" in captured.out

    def test_get_history(self, tmp_dir):
        """get_history returns a copy of the history list."""
        logger = TrainingLogger(log_dir=tmp_dir, verbose=False)
        logger.on_train_begin()
        logger.on_epoch_end(0, {"loss": 1.0})

        hist = logger.get_history()
        assert hist == logger.history
        # Ensure it's a copy
        hist.append({"epoch": 99})
        assert len(logger.history) == 1

    def test_on_epoch_end_empty_logs(self, tmp_dir):
        """on_epoch_end with no logs still records."""
        logger = TrainingLogger(log_dir=tmp_dir, verbose=False)
        logger.on_train_begin()
        logger.on_epoch_end(0, {})
        assert len(logger.history) == 1
        assert "epoch" in logger.history[0]
