"""
Trainer — epoch loop, MLflow metric logging, and checkpoint management.

Responsibilities:
    - Run train + val epochs
    - Log per-epoch losses and metrics to MLflow (step = epoch)
    - Save best checkpoint (by val_room_miou) and last checkpoint
    - Print a one-line summary after each epoch

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

from src.evaluation.metrics import compute_all


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

        self._best_miou = 0.0
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

    # ── Epoch ──────────────────────────────────────────────────────────────

    def _run_epoch(self, epoch: int, train: bool) -> dict[str, float]:
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        prefix = "train" if train else "val"

        total_loss = wall_loss_sum = room_loss_sum = 0.0
        all_wall_metrics: list[dict] = []
        all_room_metrics: list[dict] = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in tqdm(loader, desc=f"[{prefix}] epoch {epoch}", leave=False):
                images      = batch["image"].to(self.device)
                wall_targets = batch["wall_mask"].to(self.device)
                room_targets = batch["room_mask"].to(self.device)

                wall_logits, room_logits = self.model(images)

                loss, l_wall, l_room = self.criterion(
                    wall_logits, room_logits,
                    wall_targets, room_targets,
                )

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss    += loss.item()
                wall_loss_sum += l_wall.item()
                room_loss_sum += l_room.item()

                m = compute_all(
                    wall_logits.detach(), room_logits.detach(),
                    wall_targets, room_targets,
                    self.model.num_room_classes,
                )
                all_wall_metrics.append({k: m[k] for k in m if k.startswith("wall")})
                all_room_metrics.append({k: m[k] for k in m if k.startswith("room")})

        n = len(loader)
        stats = {
            "loss":      total_loss    / n,
            "wall_loss": wall_loss_sum / n,
            "room_loss": room_loss_sum / n,
        }
        for key in all_wall_metrics[0]:
            stats[key] = sum(d[key] for d in all_wall_metrics) / len(all_wall_metrics)
        for key in all_room_metrics[0]:
            stats[key] = sum(d[key] for d in all_room_metrics) / len(all_room_metrics)
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
        state = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_room_miou":   val_stats.get("room_miou", 0.0),
            "val_wall_iou":    val_stats.get("wall_iou",  0.0),
        }

        # Always save last
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(state, last_path)

        # Save best if improved
        current_miou = val_stats.get("room_miou", 0.0)
        if current_miou > self._best_miou:
            self._best_miou = current_miou
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            mlflow.log_artifact(str(best_path), artifact_path="checkpoints")

    # ── Console ────────────────────────────────────────────────────────────

    def _print_epoch(
        self,
        epoch:       int,
        total_epochs: int,
        train_stats: dict[str, float],
        val_stats:   dict[str, float],
        elapsed:     float,
    ) -> None:
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
