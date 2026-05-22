"""
Trainer — epoch loop, MLflow metric logging, and checkpoint management.

Supports two modes:
  • Dual-head (v1): wall + room decoder, MultiTaskLoss
  • Wall-only (v2): wall decoder only, WallLoss ± LwF regularisation

Optional features:
  • LwF (Learning without Forgetting): a frozen teacher model produces soft
    wall-logit targets; KL divergence between teacher and student is added
    as a lightweight regulariser (lwf_weight controls its contribution).
  • Early stopping: training halts when the tracked val metric shows no
    improvement of >= min_delta for `patience` consecutive epochs.

The Trainer does NOT call mlflow.start_run(); the caller (train.py) owns
the MLflow run context so it can log params before training starts.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_all, compute_wall
from src.training.losses import LwFLoss


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class _EarlyStopper:
    """Returns True when the tracked metric stalls for `patience` epochs."""

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self._best     = 0.0
        self._counter  = 0

    def __call__(self, metric: float) -> bool:
        if metric > self._best + self.min_delta:
            self._best    = metric
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self.patience


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model:          torch.nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        criterion:      torch.nn.Module,
        optimizer:      torch.optim.Optimizer,
        scheduler,                               # any LR scheduler or None
        cfg:            SimpleNamespace,
        device:         torch.device,
        checkpoint_dir: Path,
        # ── optional features ──────────────────────────────────────────────
        wall_only:               bool  = False,
        teacher_model:           torch.nn.Module | None = None,
        lwf_weight:              float = 0.3,
        early_stopping_patience: int | None = None,
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.cfg            = cfg
        self.device         = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._wall_only  = wall_only
        self._teacher    = teacher_model
        self._lwf_weight = lwf_weight

        if teacher_model is not None:
            lwf_temp = getattr(getattr(cfg, "lwf", SimpleNamespace()), "temperature", 2.0)
            self._lwf_loss = LwFLoss(temperature=lwf_temp).to(device)

        self._early_stopper = (
            _EarlyStopper(early_stopping_patience) if early_stopping_patience else None
        )

        self._metric_key  = cfg.evaluation.best_metric.replace("val_", "")
        self._best_metric = 0.0

        if not wall_only:
            self._num_room_classes = model.num_room_classes

    # ── Public ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        epochs = self.cfg.training.epochs
        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_stats = self._run_epoch(epoch, train=True)
            val_stats   = self._run_epoch(epoch, train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t0
            self._log_epoch(epoch, train_stats, val_stats)
            self._save_checkpoints(epoch, val_stats)
            self._print_epoch(epoch, epochs, train_stats, val_stats, elapsed)

            if self._early_stopper is not None:
                val_metric = val_stats.get(self._metric_key, 0.0)
                if self._early_stopper(val_metric):
                    print(
                        f"Early stopping: {self._metric_key} did not improve "
                        f"for {self._early_stopper.patience} consecutive epochs."
                    )
                    break

    # ── Epoch ──────────────────────────────────────────────────────────────

    def _run_epoch(self, epoch: int, train: bool) -> dict[str, float]:
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        prefix = "train" if train else "val"

        total_loss = wall_loss_sum = room_loss_sum = lwf_loss_sum = 0.0
        all_metrics: list[dict] = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in tqdm(loader, desc=f"[{prefix}] epoch {epoch}", leave=False):
                images       = batch["image"].to(self.device)
                wall_targets = batch["wall_mask"].to(self.device)

                if self._wall_only:
                    wall_logits = self.model(images)
                    l_wall      = self.criterion(wall_logits, wall_targets)
                    loss        = l_wall

                    # LwF regularisation applied during training only
                    if train and self._teacher is not None:
                        with torch.no_grad():
                            teacher_logits = self._teacher(images)
                        lwf           = self._lwf_loss(wall_logits, teacher_logits)
                        lwf_loss_sum += lwf.item()
                        loss          = loss + self._lwf_weight * lwf

                    all_metrics.append(compute_wall(wall_logits.detach(), wall_targets))

                else:
                    room_targets             = batch["room_mask"].to(self.device)
                    wall_logits, room_logits = self.model(images)
                    loss, l_wall, l_room     = self.criterion(
                        wall_logits, room_logits, wall_targets, room_targets
                    )
                    room_loss_sum += l_room.item()
                    all_metrics.append(compute_all(
                        wall_logits.detach(), room_logits.detach(),
                        wall_targets, room_targets,
                        self._num_room_classes,
                    ))

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss    += loss.item()
                wall_loss_sum += l_wall.item()

        n = len(loader)
        stats: dict[str, float] = {
            "loss":      total_loss    / n,
            "wall_loss": wall_loss_sum / n,
        }
        if not self._wall_only:
            stats["room_loss"] = room_loss_sum / n
        if train and self._teacher is not None:
            stats["lwf_loss"] = lwf_loss_sum / n
        for key in all_metrics[0]:
            stats[key] = sum(d[key] for d in all_metrics) / len(all_metrics)
        return stats

    # ── Logging ────────────────────────────────────────────────────────────

    def _log_epoch(
        self,
        epoch:       int,
        train_stats: dict[str, float],
        val_stats:   dict[str, float],
    ) -> None:
        metrics = {}
        for k, v in train_stats.items():
            metrics[f"train_{k}"] = v
        for k, v in val_stats.items():
            metrics[f"val_{k}"] = v
        mlflow.log_metrics(metrics, step=epoch)

    # ── Checkpoints ────────────────────────────────────────────────────────

    def _save_checkpoints(self, epoch: int, val_stats: dict[str, float]) -> None:
        current = val_stats.get(self._metric_key, 0.0)
        state = {
            "epoch":                    epoch,
            "model_state":              self.model.state_dict(),
            "optimizer_state":          self.optimizer.state_dict(),
            f"val_{self._metric_key}":  current,
        }

        torch.save(state, self.checkpoint_dir / "last.pt")

        if current > self._best_metric:
            self._best_metric = current
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            mlflow.log_artifact(str(best_path), artifact_path="checkpoints")

    # ── Console ────────────────────────────────────────────────────────────

    def _print_epoch(
        self,
        epoch:        int,
        total_epochs: int,
        train_stats:  dict[str, float],
        val_stats:    dict[str, float],
        elapsed:      float,
    ) -> None:
        if self._wall_only:
            lwf_str = (
                f"  lwf={train_stats['lwf_loss']:.4f}"
                if "lwf_loss" in train_stats else ""
            )
            print(
                f"Epoch {epoch:03d}/{total_epochs}  "
                f"loss={train_stats['loss']:.4f}{lwf_str}  "
                f"wall_iou={train_stats['wall_iou']:.3f}  |  "
                f"val_loss={val_stats['loss']:.4f}  "
                f"val_wall_iou={val_stats['wall_iou']:.3f}  "
                f"({elapsed:.0f}s)"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{total_epochs}  "
                f"loss={train_stats['loss']:.4f}  "
                f"wall_iou={train_stats['wall_iou']:.3f}  "
                f"room_miou={train_stats['room_miou']:.3f}  |  "
                f"val_loss={val_stats['loss']:.4f}  "
                f"val_wall_iou={val_stats['wall_iou']:.3f}  "
                f"val_room_miou={val_stats['room_miou']:.3f}  "
                f"({elapsed:.0f}s)"
            )
