"""
Main training entry point.

Usage (v1 — dual-head):
    python src/training/train.py --config configs/unet_r18_v1.yaml

Usage (v2 — wall-only, transfer from v1):
    python src/training/train.py --config configs/unet_r18_v2.yaml
    python src/training/train.py --config configs/unet_r18_v2.yaml --lwf
    python src/training/train.py --config configs/unet_r18_v2.yaml --lwf --early_stopping 7

Flags:
    --device           cuda | cpu | xpu | auto  (default: auto)
    --data_version     override dataset_version in config
    --lwf              enable Learning without Forgetting (v2 only)
    --early_stopping N stop after N epochs without improvement (default: off)
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlflow
import torch
from torch.utils.data import DataLoader

# Optional IPEX — used only for ipex.optimize() on XPU (not required for basic
# XPU usage; torch.xpu is built into PyTorch 2.6+ natively).
try:
    import intel_extension_for_pytorch as ipex
    _IPEX_AVAILABLE = True
except ImportError:
    ipex = None
    _IPEX_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ColorMapper, FloorPlanDataset
from src.models.backbone import DualHeadUNet, WallOnlyUNet
from src.training.losses import MultiTaskLoss, WallLoss
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
    parser = argparse.ArgumentParser(description="Train wall/room segmentation model.")
    parser.add_argument("--config",          required=True,      help="Path to YAML config.")
    parser.add_argument("--data_version",    default=None,       help="Override dataset_version in config.")
    parser.add_argument("--device",          default="auto",     help="cuda | cpu | xpu | auto")
    parser.add_argument("--lwf",             action="store_true",help="Enable Learning without Forgetting (v2 only).")
    parser.add_argument("--early_stopping",  type=int, default=None, metavar="N",
                        help="Stop after N epochs without improvement in the best metric.")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = resolve_device(args.device)

    if args.data_version:
        cfg.dataset_version = args.data_version

    model_version = getattr(cfg.model, "version", "v1")
    wall_only     = (model_version == "v2")

    print(f"Config          : {args.config}")
    print(f"Model version   : {model_version}")
    print(f"Dataset version : {cfg.dataset_version}")
    print(f"Device          : {device}")
    if wall_only:
        print(f"LwF             : {'enabled' if args.lwf else 'disabled'}")
        if args.early_stopping:
            print(f"Early stopping  : patience={args.early_stopping}")

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

    if not wall_only:
        print(f"Room classes    : {color_mapper.num_classes}  (incl. background)")

    # ── Datasets & loaders ───────────────────────────────────────────────
    img_size    = cfg.training.image_size
    num_workers = cfg.training.num_workers

    train_ds = FloorPlanDataset(version["train"], color_mapper, image_size=img_size)
    val_ds   = FloorPlanDataset(version["val"],   color_mapper, image_size=img_size)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    encoder_lr = cfg.training.lr * cfg.training.encoder_lr_scale
    decoder_lr = cfg.training.lr

    if wall_only:
        v1_ckpt = ROOT / cfg.model.v1_checkpoint
        if not v1_ckpt.exists():
            raise FileNotFoundError(
                f"v1 checkpoint not found: {v1_ckpt}\n"
                f"Train v1 first: python src/training/train.py --config configs/unet_r18_v1.yaml"
            )
        model = WallOnlyUNet.from_dual_head_checkpoint(
            v1_ckpt,
            encoder_name=cfg.model.encoder,
            decoder_channels=tuple(cfg.model.decoder_channels),
            device=str(device),
        ).to(device)
        print(model)

        criterion = WallLoss(
            bce_weight=cfg.training.wall_bce_weight,
            dice_weight=cfg.training.wall_dice_weight,
            pos_weight=cfg.training.wall_pos_weight,
        ).to(device)

        # Teacher model for LwF: same checkpoint, frozen
        teacher_model = None
        if args.lwf:
            teacher_model = WallOnlyUNet.from_dual_head_checkpoint(
                v1_ckpt,
                encoder_name=cfg.model.encoder,
                decoder_channels=tuple(cfg.model.decoder_channels),
                device=str(device),
            ).to(device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            print("LwF teacher loaded and frozen.")

    else:
        model = DualHeadUNet(
            encoder_name=cfg.model.encoder,
            encoder_weights=cfg.model.encoder_weights,
            num_room_classes=color_mapper.num_classes,
            decoder_channels=tuple(cfg.model.decoder_channels),
        ).to(device)
        print(model)

        criterion = MultiTaskLoss(
            wall_weight=cfg.training.loss_wall_weight,
            room_weight=cfg.training.loss_room_weight,
            bce_weight=cfg.training.wall_bce_weight,
            dice_weight=cfg.training.wall_dice_weight,
            wall_pos_weight=cfg.training.wall_pos_weight,
        ).to(device)

        teacher_model = None

    # ── Optimizer (differential LR) ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.param_groups(encoder_lr, decoder_lr),
        weight_decay=cfg.training.weight_decay,
    )

    # ── IPEX optimization (XPU only) ─────────────────────────────────────
    if device.type == "xpu" and _IPEX_AVAILABLE:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
        if teacher_model is not None:
            teacher_model = ipex.optimize(teacher_model, dtype=torch.float32)
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
    lwf_cfg    = getattr(cfg, "lwf", SimpleNamespace(weight=0.3, temperature=2.0))
    lwf_weight = lwf_cfg.weight if args.lwf else 0.0

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=f"{cfg.experiment_name}_{cfg.dataset_version}"):

        common_params = {
            "model_version":     model_version,
            "dataset_version":   cfg.dataset_version,
            "encoder":           cfg.model.encoder,
            "epochs":            cfg.training.epochs,
            "batch_size":        cfg.training.batch_size,
            "image_size":        cfg.training.image_size,
            "lr":                cfg.training.lr,
            "encoder_lr":        encoder_lr,
            "weight_decay":      cfg.training.weight_decay,
            "wall_bce_weight":   cfg.training.wall_bce_weight,
            "wall_dice_weight":  cfg.training.wall_dice_weight,
            "wall_pos_weight":   cfg.training.wall_pos_weight,
            "scheduler":         scheduler_name,
            "early_stopping":    args.early_stopping,
        }
        if wall_only:
            common_params.update({
                "v1_checkpoint":  str(cfg.model.v1_checkpoint),
                "lwf_enabled":    args.lwf,
                "lwf_weight":     lwf_weight,
                "lwf_temperature": lwf_cfg.temperature,
            })
        else:
            common_params.update({
                "encoder_weights":  cfg.model.encoder_weights,
                "num_room_classes": color_mapper.num_classes,
                "loss_wall_weight": cfg.training.loss_wall_weight,
                "loss_room_weight": cfg.training.loss_room_weight,
            })
        mlflow.log_params(common_params)

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
            wall_only=wall_only,
            teacher_model=teacher_model,
            lwf_weight=lwf_weight,
            early_stopping_patience=args.early_stopping,
        )
        trainer.run()

    print("Training complete.")


if __name__ == "__main__":
    main()
