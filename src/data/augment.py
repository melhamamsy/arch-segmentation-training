"""
Apply an ordered pipeline of augmentation strategies to floor-plan images.

Strategies are listed in order via --strategies; the combined name drives the
output directory and file prefix:

  --strategies greyscale
      output dir : data/augmented/{source}/greyscale/
      file prefix: greyscale_pseudo_00000.png

  --strategies greyscale geometric
      output dir : data/augmented/{source}/greyscale_geometric/
      file prefix: greyscale_geometric_pseudo_00000.png

Output layout:
  data/augmented/{source}/{combined}/images/{combined}_{orig_name}
  data/augmented/{source}/{combined}/masks/walls/{combined}_{orig_name}
  data/augmented/{source}/{combined}/masks/colors/{combined}_{orig_name}
  data/augmented/{source}/{combined}/masks/footprints/{combined}_{orig_name}

Color-only strategies (greyscale, photometric) leave masks unchanged.
Spatial strategies (geometric, scale_crop, elastic) apply the same random
transform jointly to the image and every mask to preserve alignment.

Usage:
  python src/data/augment.py --strategies greyscale
  python src/data/augment.py --strategies greyscale geometric
  python src/data/augment.py --strategies geometric --source pseudo-12k --force
  python src/data/augment.py --list-strategies
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
AUG_DIR = ROOT / "data" / "augmented"

MASK_COLS = ["walls", "colors", "footprints"]
SOURCES   = ["pseudo-12k", "manual-1k"]

# (image_rgb_np [H,W,3], masks [list of H,W,3]) -> (image_rgb_np, masks)
StrategyFn = Callable[
    [np.ndarray, list[np.ndarray]],
    tuple[np.ndarray, list[np.ndarray]],
]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _greyscale(img: np.ndarray, masks: list[np.ndarray]) \
        -> tuple[np.ndarray, list[np.ndarray]]:
    """Convert image to greyscale (L -> 3-channel RGB). Masks unchanged."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB), masks


def _geometric(img: np.ndarray, masks: list[np.ndarray]) \
        -> tuple[np.ndarray, list[np.ndarray]]:
    """Spatial flips, rotations and small affine shifts applied to image + masks."""
    pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5,
        ),
        A.Affine(shear=(-5, 5), p=0.3),
    ])
    result = pipeline(image=img, masks=masks)
    return result["image"], result["masks"]


def _photometric(img: np.ndarray, masks: list[np.ndarray]) \
        -> tuple[np.ndarray, list[np.ndarray]]:
    """Pixel-level colour/brightness changes applied to image only."""
    pipeline = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=10, p=0.4),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.2),
    ])
    return pipeline(image=img)["image"], masks


def _scale_crop(img: np.ndarray, masks: list[np.ndarray]) \
        -> tuple[np.ndarray, list[np.ndarray]]:
    """Random crop + resize to simulate partial floor plans."""
    pipeline = A.Compose([
        A.RandomResizedCrop(
            size=(512, 512), scale=(0.5, 1.0), ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR, p=0.6,
        ),
        A.PadIfNeeded(min_height=512, min_width=512,
                      border_mode=cv2.BORDER_CONSTANT, value=255, p=1.0),
    ])
    result = pipeline(image=img, masks=masks)
    return result["image"], result["masks"]


def _elastic(img: np.ndarray, masks: list[np.ndarray]) \
        -> tuple[np.ndarray, list[np.ndarray]]:
    """Non-rigid distortions for hand-drawn / imperfect floor plans."""
    pipeline = A.Compose([
        A.ElasticTransform(alpha=40, sigma=6, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.2),
    ])
    result = pipeline(image=img, masks=masks)
    return result["image"], result["masks"]


def _quantize(img: np.ndarray, n_bins: int,
              masks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Snap each channel to the nearer boundary of its percentile bin.

    Boundaries are computed per-channel from the image itself, so the
    quantization adapts to each image's actual intensity distribution.

    n_bins=4  -> boundaries at [0, 25, 50, 75, 100]th percentile (quartiles)
    n_bins=10 -> boundaries at [0, 10, 20, ..., 100]th percentile (deciles)

    Example (n_bins=4, channel R, boundaries=[0, 80, 150, 200, 255]):
      pixel value 95 falls in bin [80, 150].
      |95 - 80| = 15  <  |95 - 150| = 55  →  snapped to 80.
    """
    pcts = np.linspace(0, 100, n_bins + 1)   # n_bins+1 boundary values
    out  = np.empty_like(img)

    for c in range(3):
        ch    = img[:, :, c].astype(np.float32)
        bnd   = np.percentile(ch, pcts)         # sorted, length = n_bins+1
        flat  = ch.ravel()

        # searchsorted(side='right') gives the first index where bnd[idx] > v,
        # so bnd[idx-1] <= v < bnd[idx] is the bin containing v.
        idx   = np.searchsorted(bnd, flat, side="right")
        idx   = np.clip(idx, 1, len(bnd) - 1)  # keep inside valid range

        left  = bnd[idx - 1]
        right = bnd[idx]
        snapped = np.where(np.abs(flat - left) <= np.abs(flat - right), left, right)
        out[:, :, c] = np.clip(snapped.reshape(ch.shape), 0, 255).astype(np.uint8)

    return out, masks


def _quantization4(img: np.ndarray, masks: list[np.ndarray]) \
        -> tuple[np.ndarray, list[np.ndarray]]:
    """Snap each channel to the nearest quartile boundary (5 possible values). Masks unchanged."""
    return _quantize(img, 4, masks)


def _quantization10(img: np.ndarray, masks: list[np.ndarray]) \
        -> tuple[np.ndarray, list[np.ndarray]]:
    """Snap each channel to the nearest decile boundary (11 possible values). Masks unchanged."""
    return _quantize(img, 10, masks)


def _bleaching(img: np.ndarray, masks: list[np.ndarray],
               threshold: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Preserve greyscale pixels; turn all coloured pixels white.

    A pixel is considered greyscale when max(R,G,B) - min(R,G,B) <= threshold
    (i.e. its chroma is low). Greyscale pixels are averaged across channels so
    any residual tint is removed. All other pixels become (255, 255, 255).

    This keeps structural information (walls, furniture, text — all near-grey)
    while bleaching away solid room-colour fills.
    """
    chroma  = img.max(axis=2).astype(np.int16) - img.min(axis=2).astype(np.int16)
    is_grey = chroma <= threshold                                # [H, W] bool

    avg = (img.astype(np.int32).sum(axis=2) // 3).astype(np.uint8)  # [H, W]

    # expand to 3 channels for broadcasting
    is_grey3 = is_grey[:, :, np.newaxis].repeat(3, axis=2)     # [H, W, 3]
    avg3     = avg[:, :, np.newaxis].repeat(3, axis=2)          # [H, W, 3]

    out = np.where(is_grey3, avg3, np.uint8(255)).astype(np.uint8)
    return out, masks


def _chroma15(img: np.ndarray, masks: list[np.ndarray]) \
        -> tuple[np.ndarray, list[np.ndarray]]:
    """Turn coloured pixels white; keep greyscale pixels (chroma <= 15) as averaged grey."""
    return _bleaching(img, masks, threshold=15)


STRATEGIES: dict[str, StrategyFn] = {
    "greyscale":      _greyscale,
    "geometric":      _geometric,
    "photometric":    _photometric,
    "scale_crop":     _scale_crop,
    "elastic":        _elastic,
    "quantization4":  _quantization4,
    "quantization10": _quantization10,
    "chroma15":       _chroma15,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def apply_pipeline(
    strategy_names: list[str],
    img: np.ndarray,
    masks: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    for name in strategy_names:
        img, masks = STRATEGIES[name](img, masks)
    return img, masks


# ---------------------------------------------------------------------------
# Worker (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> int:
    """Process a single image. Returns 1 if written, 0 if skipped or failed."""
    img_path, src_dir, strategies, out_dir, combined, force = args

    out_name = f"{combined}_{img_path.name}"
    img_out  = out_dir / "images" / out_name

    if not force and img_out.exists():
        return 0

    try:
        img_np = np.array(Image.open(img_path).convert("RGB"))
        masks: list[np.ndarray] = []
        for col in MASK_COLS:
            m_path = src_dir / "masks" / col / img_path.name
            masks.append(
                np.array(Image.open(m_path).convert("RGB"))
                if m_path.exists() else np.zeros_like(img_np)
            )

        out_img_np, out_masks = apply_pipeline(strategies, img_np, masks)

        Image.fromarray(out_img_np).save(img_out)
        for col, mask_np in zip(MASK_COLS, out_masks):
            Image.fromarray(mask_np).save(out_dir / "masks" / col / out_name)

        return 1
    except Exception as exc:
        print(f"\nWARN  {img_path.name}: {exc}", flush=True)
        return 0


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------

def _detect_prefix(images_dir: Path) -> str:
    for candidate in ("pseudo", "manual"):
        if any(images_dir.glob(f"{candidate}_*.png")):
            return candidate
    raise ValueError(f"Cannot detect prefix in {images_dir}")


def augment_source(
    source: str,
    strategies: list[str],
    force: bool = False,
    workers: int | None = None,
) -> int:
    src_dir    = RAW_DIR / source
    images_dir = src_dir / "images"
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist. Run download.py first.")
        sys.exit(1)

    prefix   = _detect_prefix(images_dir)
    combined = "_".join(strategies)
    out_dir  = AUG_DIR / source / combined
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for col in MASK_COLS:
        (out_dir / "masks" / col).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob(f"{prefix}_*.png"))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return 0

    n_workers = workers or os.cpu_count() or 1
    print(f"\nSource   : {source}  ({len(image_paths)} images)")
    print(f"Pipeline : {' -> '.join(strategies)}")
    print(f"Workers  : {n_workers}")
    print(f"Output   : {out_dir.relative_to(ROOT)}")

    tasks = [
        (img_path, src_dir, strategies, out_dir, combined, force)
        for img_path in image_paths
    ]

    written = 0
    with Pool(n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_one, tasks),
            total=len(tasks),
            desc=f"{combined}/{source}",
            unit="img",
        ):
            written += result

    skipped = len(image_paths) - written
    if skipped:
        print(f"Skipped {skipped} already-existing files (--force to overwrite).")
    print(f"Written: {written} -> {out_dir.relative_to(ROOT)}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment floor-plan images with an ordered strategy pipeline.")
    parser.add_argument(
        "--strategies", nargs="+",
        choices=list(STRATEGIES.keys()),
        metavar="STRATEGY",
        help=f"Ordered strategies to apply. Available: {list(STRATEGIES.keys())}",
    )
    parser.add_argument(
        "--source", choices=SOURCES,
        help="Single source to process (default: both).")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes (default: all CPU cores).")
    parser.add_argument(
        "--list-strategies", action="store_true",
        help="Print available strategies and exit.")

    args = parser.parse_args()

    if args.list_strategies:
        print("Available strategies:")
        for name, fn in STRATEGIES.items():
            doc = fn.__doc__.strip().splitlines()[0] if fn.__doc__ else ""
            print(f"  {name:<12}  {doc}")
        return

    if not args.strategies:
        parser.error("--strategies is required unless using --list-strategies")

    targets = [args.source] if args.source else SOURCES
    total = 0
    for source in targets:
        total += augment_source(source, args.strategies,
                                force=args.force, workers=args.workers)

    print(f"\nTotal written: {total}")
    print("Done.")


if __name__ == "__main__":
    main()
