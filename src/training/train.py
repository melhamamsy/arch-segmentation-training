"""
Main training entry point.

Usage:
    python src/training/train.py --config configs/unet_r18_v1.yaml
    python src/training/train.py --config configs/unet_r18_v1.yaml --device cpu
    python src/training/train.py --config configs/unet_r18_v1.yaml --version v2

The --version flag overrides the dataset_version in the config, making it
easy to run the same model on multiple dataset versions without editing YAML:
    python src/training/train.py --config configs/unet_r18_v1.yaml --version v1
    python src/training/train.py --config configs/unet_r18_v1.yaml --version v2
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

# Optional IPEX — used only for ipex.optimize() on XPU (not required for basic XPU usage;
# torch.xpu is built into PyTorch 2.6+ natively).
try:
    import intel_extension_for_pytorch as ipex
    _IPEX_AVAILABLE = True
except ImportError:
    ipex = None
    _IPEX_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ColorMapper, FloorPlanDataset
from src.models.backbone import DualHeadUNet
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer
from src.utils.config import load_config

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'experiments' / 'mlflow.db'}"


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _xpu_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if preference == "xpu" and not _xpu_available():
        raise RuntimeError(
            "Device 'xpu' requested but no Intel XPU was found.\n"
            "Make sure you have Intel Arc GPU drivers installed and a torch build\n"
            "with XPU support (torch 2.6+ from the standard PyPI release, not +cpu)."
        )
    return torch.device(preference)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DualHeadUNet on floor plan data.")
    parser.add_argument("--config",  required=True,  help="Path to YAML config.")
    parser.add_argument("--version", default=None,   help="Override dataset_version in config.")
    parser.add_argument("--device",  default="auto", help="cuda | cpu | auto")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = resolve_device(args.device)

    # CLI override for dataset version
    if args.version:
        cfg.dataset_version = args.version

    print(f"Config          : {args.config}")
    print(f"Dataset version : {cfg.dataset_version}")
    print(f"Device          : {device}")

    # ── Load version split ────────────────────────────────────────────────
    version_path = ROOT / "data" / "versions" / f"{cfg.dataset_version}.json"
    if not version_path.exists():
        raise FileNotFoundError(
            f"Version '{cfg.dataset_version}' not found.\n"
            f"Run: python src/data/versioning.py list"
        )
    version = json.loads(version_path.read_text())
    print(f"Train samples   : {len(version['train'])}")
    print(f"Val   samples   : {len(version['val'])}")

    # ── Color palette ─────────────────────────────────────────────────────
    palette_dir  = ROOT / cfg.paths.palette_dir
    palette_path = palette_dir / f"{cfg.dataset_version}_palette.json"

    if palette_path.exists():
        print(f"Loading palette : {palette_path.relative_to(ROOT)}")
        color_mapper = ColorMapper.load(palette_path)
    else:
        print("Building colour palette from training samples ...")
        color_mapper = ColorMapper.build_from_samples(version["train"])
        palette_path.parent.mkdir(parents=True, exist_ok=True)
        color_mapper.save(palette_path)
        print(f"Palette saved   : {palette_path.relative_to(ROOT)}")

    print(f"Room classes    : {color_mapper.num_classes}  (incl. background)")

    # ── Datasets & loaders ───────────────────────────────────────────────
    img_size    = cfg.training.image_size
    num_workers = cfg.training.num_workers

    train_ds = FloorPlanDataset(version["train"], color_mapper, image_size=img_size)
    val_ds   = FloorPlanDataset(version["val"],   color_mapper, image_size=img_size)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"),
        # pin_memory intentionally off for xpu — IPEX handles host-device transfers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = DualHeadUNet(
        encoder_name=cfg.model.encoder,
        encoder_weights=cfg.model.encoder_weights,
        num_room_classes=color_mapper.num_classes,
        decoder_channels=tuple(cfg.model.decoder_channels),
    ).to(device)
    print(model)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = MultiTaskLoss(
        wall_weight=cfg.training.loss_wall_weight,
        room_weight=cfg.training.loss_room_weight,
        bce_weight=cfg.training.wall_bce_weight,
        dice_weight=cfg.training.wall_dice_weight,
        wall_pos_weight=cfg.training.wall_pos_weight,
    ).to(device)

    # ── Optimizer (differential LR) ───────────────────────────────────────
    encoder_lr = cfg.training.lr * cfg.training.encoder_lr_scale
    decoder_lr = cfg.training.lr
    optimizer  = torch.optim.AdamW(
        model.param_groups(encoder_lr, decoder_lr),
        weight_decay=cfg.training.weight_decay,
    )

    # ── IPEX optimization (XPU only) ─────────────────────────────────────
    if device.type == "xpu" and _IPEX_AVAILABLE:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
        print("IPEX optimization applied for XPU.")

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler_name = cfg.scheduler.name.lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.scheduler.T_max
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=f"{cfg.experiment_name}_{cfg.dataset_version}"):

        mlflow.log_params({
            "dataset_version":   cfg.dataset_version,
            "encoder":           cfg.model.encoder,
            "encoder_weights":   cfg.model.encoder_weights,
            "num_room_classes":  color_mapper.num_classes,
            "epochs":            cfg.training.epochs,
            "batch_size":        cfg.training.batch_size,
            "image_size":        cfg.training.image_size,
            "lr":                cfg.training.lr,
            "encoder_lr":        encoder_lr,
            "weight_decay":      cfg.training.weight_decay,
            "loss_wall_weight":  cfg.training.loss_wall_weight,
            "loss_room_weight":  cfg.training.loss_room_weight,
            "wall_bce_weight":   cfg.training.wall_bce_weight,
            "wall_dice_weight":  cfg.training.wall_dice_weight,
            "wall_pos_weight":   cfg.training.wall_pos_weight,
            "scheduler":         scheduler_name,
        })

        # Log version JSON as artifact for full reproducibility
        mlflow.log_artifact(str(version_path), artifact_path="dataset")
        mlflow.log_artifact(str(palette_path), artifact_path="dataset")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            device=device,
            checkpoint_dir=ROOT / cfg.paths.checkpoint_dir,
        )
        trainer.run()

    print("Training complete.")


if __name__ == "__main__":
    main()
