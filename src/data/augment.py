"""
Augmentation pipeline for floor-plan datasets.

Reads images directly from a raw dataset directory, applies spatial and/or
photometric transforms, and writes augmented copies to a new directory under
data/raw/. All three mask channels (walls, colors, footprints) are transformed
jointly with the image to maintain alignment.

Usage:
  python src/data/augment.py --source pseudo-12k --strategy geometric
  python src/data/augment.py --source pseudo-12k --strategy geometric+photometric --copies 2
  python src/data/augment.py --list-strategies

Output directory:
  data/raw/<source>_aug_<strategy>/
    images/   <prefix>_<id>.png
    masks/
      walls/      <prefix>_<id>.png
      colors/     <prefix>_<id>.png
      footprints/ <prefix>_<id>.png

Strategies can be combined with '+'. Augmented files inherit the source prefix
(pseudo_ or manual_) and are structured identically to raw downloads so that
versioning.py can include them in a dataset version without special handling.
"""

import argparse
import logging
import random
import sys
import traceback
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
MASK_COLS = ["walls", "colors", "footprints"]


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
# Add new strategies here and document them in augmentation.md.
# Each builder returns an albumentations Compose pipeline.
# Spatial transforms are applied to image + all masks simultaneously.
# Photometric transforms are applied to image only (masks unchanged).
# ---------------------------------------------------------------------------

def _geometric() -> A.Compose:
    """Spatial transforms safe for floor-plan annotations."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5,
        ),
        A.Affine(shear=(-5, 5), p=0.3),
    ])


def _photometric() -> A.Compose:
    """Pixel-level transforms applied to image only."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.2),
    ])


def _scale_crop() -> A.Compose:
    """Random crop + resize to force handling of partial floor plans."""
    return A.Compose([
        A.RandomResizedCrop(
            size=(512, 512), scale=(0.5, 1.0), ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR, p=0.6,
        ),
        A.PadIfNeeded(min_height=512, min_width=512,
                      border_mode=cv2.BORDER_CONSTANT, value=255, p=1.0),
    ])


def _elastic() -> A.Compose:
    """Non-rigid distortions for imperfect/hand-drawn floor plans."""
    return A.Compose([
        A.ElasticTransform(alpha=40, sigma=6, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.2),
    ])


STRATEGIES: dict[str, callable] = {
    "geometric":   _geometric,
    "photometric": _photometric,
    "scale_crop":  _scale_crop,
    "elastic":     _elastic,
}


def build_pipeline(strategy_name: str) -> A.Compose:
    """Build a combined pipeline from '+'-separated strategy names."""
    parts = [s.strip() for s in strategy_name.split("+")]
    unknown = [p for p in parts if p not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Available: {list(STRATEGIES)}")
    if len(parts) == 1:
        return STRATEGIES[parts[0]]()
    transforms = []
    for part in parts:
        transforms.extend(STRATEGIES[part]().transforms)
    return A.Compose(transforms)


# ---------------------------------------------------------------------------
# Prefix detection
# ---------------------------------------------------------------------------

def _detect_prefix(images_dir: Path) -> str:
    """Infer the file prefix (pseudo/manual) from the filenames present."""
    for candidate in ("pseudo", "manual"):
        if any(images_dir.glob(f"{candidate}_*.png")):
            return candidate
    raise ValueError(f"Cannot detect prefix in {images_dir}. "
                     "Expected files named pseudo_*.png or manual_*.png")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("augment")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(stream=open(sys.stdout.fileno(), "w",
                                           encoding="utf-8", closefd=False))
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Core augmentation
# ---------------------------------------------------------------------------

def augment_source(
    source_name: str,
    strategy_name: str,
    copies_per_sample: int = 1,
    seed: int = 42,
    log: logging.Logger | None = None,
) -> Path:
    """
    Augment all samples in a raw dataset directory.

    Args:
        source_name:       name of the source dir under data/raw/ (e.g. "pseudo-12k")
        strategy_name:     one or more '+'-joined strategy names
        copies_per_sample: augmented copies to generate per original image
        seed:              random seed for reproducibility

    Returns:
        Path to the output directory.
    """
    if log is None:
        log = logging.getLogger("augment")

    random.seed(seed)
    np.random.seed(seed)

    src_dir = RAW_DIR / source_name
    images_dir = src_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Source not found: {images_dir}")

    prefix = _detect_prefix(images_dir)
    pipeline = build_pipeline(strategy_name)

    safe_strategy = strategy_name.replace("+", "_")
    out_name = f"{source_name}_aug_{safe_strategy}"
    out_dir = RAW_DIR / out_name

    log.info(f"Source   : {src_dir.relative_to(ROOT)}")
    log.info(f"Strategy : {strategy_name}")
    log.info(f"Copies   : {copies_per_sample}x")
    log.info(f"Output   : {out_dir.relative_to(ROOT)}")

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for col in MASK_COLS:
        (out_dir / "masks" / col).mkdir(parents=True, exist_ok=True)

    src_images = sorted(images_dir.glob(f"{prefix}_*.png"))
    log.info(f"Found {len(src_images)} source images")

    out_idx = 0
    errors = 0

    for img_path in tqdm(src_images, desc=f"Augmenting {out_name}", unit="img"):
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            masks = []
            for col in MASK_COLS:
                m_path = src_dir / "masks" / col / img_path.name
                masks.append(np.array(Image.open(m_path).convert("RGB")))

            for _ in range(copies_per_sample):
                result = pipeline(image=image, masks=masks)
                aug_img = result["image"]
                aug_masks = result["masks"]

                out_fname = f"{prefix}_{out_idx:05d}.png"
                Image.fromarray(aug_img).save(out_dir / "images" / out_fname)
                for col, aug_mask in zip(MASK_COLS, aug_masks):
                    Image.fromarray(aug_mask).save(out_dir / "masks" / col / out_fname)

                out_idx += 1

        except Exception as exc:
            log.warning(f"{img_path.name} failed: {exc}")
            log.debug(traceback.format_exc())
            errors += 1

    total = len(src_images) * copies_per_sample - errors
    log.info(f"Saved {total} augmented samples -> {out_dir.relative_to(ROOT)}")
    if errors:
        log.warning(f"{errors} sample(s) had errors and were skipped")

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Augment a floor-plan dataset directory.")
    parser.add_argument("--source",
                        help="Source dataset name under data/raw/ (e.g. pseudo-12k).")
    parser.add_argument("--strategy", default="geometric",
                        help="Strategy or '+'-joined combination. Default: geometric")
    parser.add_argument("--copies", type=int, default=1,
                        help="Augmented copies per original. Default: 1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-strategies", action="store_true",
                        help="Print available strategies and exit.")
    args = parser.parse_args()

    if args.list_strategies:
        print("Available strategies:")
        for name in STRATEGIES:
            print(f"  {name}")
        print("\nCombine with '+', e.g.: geometric+photometric")
        return

    if not args.source:
        parser.error("--source is required unless using --list-strategies")

    log_file = RAW_DIR / f"{args.source}_aug_{args.strategy.replace('+', '_')}" / "augment.log"
    log = _setup_logging(log_file)

    try:
        augment_source(args.source, args.strategy,
                       copies_per_sample=args.copies, seed=args.seed, log=log)
    except Exception as exc:
        log.error(f"Augmentation failed: {exc}")
        log.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        for h in log.handlers[:]:
            h.close()
            log.removeHandler(h)


if __name__ == "__main__":
    main()
