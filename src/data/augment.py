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

Output layout (one masks/ subdir per mask column the source actually has):
  data/augmented/{source}/{combined}/images/{combined}_{orig_name}
  data/augmented/{source}/{combined}/masks/walls/{combined}_{orig_name}
  data/augmented/{source}/{combined}/masks/colors/{combined}_{orig_name}       (pseudo/manual only)
  data/augmented/{source}/{combined}/masks/footprints/{combined}_{orig_name}   (pseudo/manual only)

Sources (see SOURCES) span two on-disk layouts; both are handled automatically:
  pseudo-12k, manual-1k                 — flat, masks: walls + colors + footprints
  cubicasa5k/{high_quality,             — nested one level, masks: walls only
              high_quality_architectural,
              colorful}
The file-name prefix and the set of mask columns are detected per source, so
cubicasa runs carry only the walls mask (no zero-filled colors/footprints).

Color-only strategies (greyscale, photometric) leave masks unchanged.
Spatial strategies (geometric, scale_crop, elastic) apply the same random
transform jointly to the image and every mask to preserve alignment.

Usage:
  python src/data/augment.py --strategies greyscale
  python src/data/augment.py --strategies greyscale geometric
  python src/data/augment.py --strategies geometric --source pseudo-12k --force
  python src/data/augment.py --strategies howallow5 --source cubicasa5k/high_quality
  python src/data/augment.py --list-strategies
"""

import argparse
import hashlib
import math
import os
import random
import re
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

# All mask columns we know how to carry through the pipeline, in canonical
# order. "walls" must stay first: howallow-style strategies read masks[0].
# Not every source has every column — cubicasa5k only ships "walls" — so the
# columns actually used for a given source are detected from disk per-source
# (see _detect_mask_cols). "walls_check"/"model" under cubicasa are NOT masks.
ALL_MASK_COLS = ["walls", "colors", "footprints"]

# Sources live at data/raw/{source}/. cubicasa5k sources are nested one level
# deeper (data/raw/cubicasa5k/{quality}/); pathlib handles the "/" on Windows.
SOURCES = [
    "pseudo-12k",
    "manual-1k",
    "cubicasa5k/high_quality",
    "cubicasa5k/high_quality_architectural",
    "cubicasa5k/colorful",
]

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


# Parameterized strategy: chroma<T> bleaches coloured pixels (chroma > T) white,
# keeping greyscale pixels (chroma <= T) as averaged grey. T is read from the
# strategy name at runtime (chroma15, chroma25, ...) — any T in 0..255 works.
_CHROMA_RE = re.compile(r"^chroma(\d+)$")


def _make_chroma(threshold: int) -> StrategyFn:
    """Build a chroma strategy fn for a given chroma threshold."""
    def _fn(img: np.ndarray, masks: list[np.ndarray]) \
            -> tuple[np.ndarray, list[np.ndarray]]:
        return _bleaching(img, masks, threshold=threshold)
    return _fn


def _resolve_chroma(name: str) -> StrategyFn | None:
    """Return a chroma strategy fn if *name* is chroma<T> (0<=T<=255), else None."""
    m = _CHROMA_RE.match(name)
    if not m:
        return None
    threshold = int(m.group(1))
    if threshold > 255:
        raise ValueError(f"chroma threshold must be <= 255, got {threshold!r} in {name!r}")
    return _make_chroma(threshold)


def _hollow_walls(img: np.ndarray, masks: list[np.ndarray],
                  border: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Turn solid walls into hollow walls: keep a *border*-pixel rim, white interior.

    Training plans have solid (filled) walls, but many inference plans draw walls
    as outlines only (hollow, white inside). This bridges that domain gap.

    The walls mask (masks[0], white walls on black) defines the wall region. We
    sweep inward from every wall edge and keep the first *border* pixels of wall
    on each side, then blank the interior to white. Sweeping from all four sides
    and keeping a fixed margin from the nearest edge is exactly a square (L-inf)
    erosion: a wall pixel is "interior" iff it lies more than *border* pixels from
    the nearest background pixel along any axis. Eroding the binary wall mask by
    *border* yields that interior; those pixels are set white in the image.

    Walls thinner than 2*border survive intact (erosion leaves no interior), so
    thin walls and the rims themselves stay solid. Internal walls, junctions and
    closed room outlines are handled automatically because both sides of every
    wall are background in the mask. Masks are returned unchanged.
    """
    walls = masks[0]                                   # [H, W, 3] white-on-black
    wall_bin = (walls.mean(axis=2) > 128).astype(np.uint8)

    k = 2 * border + 1                                 # rect kernel keeps `border` px/side
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    interior = cv2.erode(wall_bin, kernel, iterations=1).astype(bool)  # [H, W]

    out = img.copy()
    out[interior] = 255                                # blank wall interior to white
    return out, masks


# Parameterized strategy: howallow<N> hollows walls keeping an N-pixel rim.
# N is read from the strategy name at runtime (howallow2, howallow5, ...), so
# any positive integer works without registering a fixed entry.
_HOWALLOW_RE = re.compile(r"^howallow(\d+)$")


def _make_howallow(border: int) -> StrategyFn:
    """Build a howallow strategy fn for a given rim width (px)."""
    def _fn(img: np.ndarray, masks: list[np.ndarray]) \
            -> tuple[np.ndarray, list[np.ndarray]]:
        return _hollow_walls(img, masks, border=border)
    return _fn


def _resolve_howallow(name: str) -> StrategyFn | None:
    """Return a howallow strategy fn if *name* is howallow<N> (N>=1), else None."""
    m = _HOWALLOW_RE.match(name)
    if not m:
        return None
    border = int(m.group(1))
    if border < 1:
        raise ValueError(f"howallow border must be >= 1, got {border!r} in {name!r}")
    return _make_howallow(border)


STRATEGIES: dict[str, StrategyFn] = {
    "greyscale":      _greyscale,
    "geometric":      _geometric,
    "photometric":    _photometric,
    "scale_crop":     _scale_crop,
    "elastic":        _elastic,
    "quantization4":  _quantization4,
    "quantization10": _quantization10,
}

# rand-crop strategies: name -> (crop_pct, n_crops)
# n = floor(100 / crop_pct_int) — how many crops per original image.
# Each crop is a random (crop_pct × H, crop_pct × W) region resized back to
# the original resolution. Masks receive the IDENTICAL spatial crop so
# walls / colors / footprints stay perfectly aligned with the image.
RAND_CROP_STRATEGIES: dict[str, tuple[float, int]] = {
    "rand80crop": (0.80, math.floor(100 / 80)),   # crop_pct=0.80, n=1
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def apply_pipeline(
    strategy_names: list[str],
    img: np.ndarray,
    masks: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Apply all regular strategies in order. rand-crop strategies are skipped
    here and handled separately in _process_one."""
    for name in strategy_names:
        if name in RAND_CROP_STRATEGIES:
            continue
        fn = STRATEGIES.get(name) or _resolve_howallow(name) or _resolve_chroma(name)
        if fn is None:
            raise KeyError(f"Unknown strategy: {name!r}")
        img, masks = fn(img, masks)
    return img, masks


# ---------------------------------------------------------------------------
# Worker (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> int:
    """
    Process a single image through the strategy pipeline.

    For pipelines that include a rand-crop strategy, n crops are written
    with the crop index appended to the strategy name in the filename:
        chroma15_rand80crop1_pseudo_00000.png
        chroma15_rand80crop2_pseudo_00000.png  (if n > 1)

    For all other pipelines, one file is written:
        chroma15_pseudo_00000.png

    Returns the number of files actually written (0 if all already existed).
    """
    img_path, src_dir, strategies, out_dir, combined, force, base_seed, mask_cols = args

    try:
        img_np = np.array(Image.open(img_path).convert("RGB"))
        masks: list[np.ndarray] = []
        for col in mask_cols:
            m_path = src_dir / "masks" / col / img_path.name
            masks.append(
                np.array(Image.open(m_path).convert("RGB"))
                if m_path.exists() else np.zeros_like(img_np)
            )

        # Apply all regular strategies (rand-crop is skipped inside apply_pipeline)
        img_np, masks = apply_pipeline(strategies, img_np, masks)

        # Check for a rand-crop strategy in the pipeline
        rand_strat = next((s for s in strategies if s in RAND_CROP_STRATEGIES), None)

        if rand_strat is None:
            # ── Single output ────────────────────────────────────────────────
            out_name = f"{combined}_{img_path.name}"
            img_out  = out_dir / "images" / out_name
            if not force and img_out.exists():
                return 0
            Image.fromarray(img_np).save(img_out)
            for col, mask_np in zip(mask_cols, masks):
                Image.fromarray(mask_np).save(out_dir / "masks" / col / out_name)
            return 1

        # ── Rand-crop: n outputs, each with a different deterministic seed ──
        crop_pct, n = RAND_CROP_STRATEGIES[rand_strat]
        H, W        = img_np.shape[:2]
        crop_h      = int(H * crop_pct)
        crop_w      = int(W * crop_pct)

        # Seed per (image, crop_index): stable across runs, unique per image
        path_hash = int(hashlib.sha256(img_path.name.encode()).hexdigest()[:8], 16)

        transform = A.Compose([
            A.RandomCrop(height=crop_h, width=crop_w, p=1.0),
            A.Resize(height=H, width=W, interpolation=cv2.INTER_LINEAR, p=1.0),
        ])

        written = 0
        for i in range(1, n + 1):
            # Replace "rand80crop" with "rand80crop1", "rand80crop2", etc.
            out_name = f"{combined.replace(rand_strat, f'{rand_strat}{i}')}_{img_path.name}"
            img_out  = out_dir / "images" / out_name
            if not force and img_out.exists():
                continue

            crop_seed = (base_seed + path_hash + i) & 0xFFFF_FFFF
            np.random.seed(crop_seed)
            random.seed(crop_seed)

            result = transform(image=img_np, masks=masks)
            Image.fromarray(result["image"]).save(img_out)
            for col, mask_np in zip(mask_cols, result["masks"]):
                Image.fromarray(mask_np).save(out_dir / "masks" / col / out_name)
            written += 1

        return written

    except Exception as exc:
        print(f"\nWARN  {img_path.name}: {exc}", flush=True)
        return 0


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------

# File-name prefixes per source family. cubicasa images are named
# cubicasa_{c,hq,hqa}_00000.png; the existing sources use pseudo_/manual_.
_PREFIX_CANDIDATES = ("pseudo", "manual",
                      "cubicasa_c", "cubicasa_hqa", "cubicasa_hq")


def _detect_prefix(images_dir: Path) -> str:
    # Longest-prefix-first so "cubicasa_hqa" wins over "cubicasa_hq".
    for candidate in sorted(_PREFIX_CANDIDATES, key=len, reverse=True):
        if any(images_dir.glob(f"{candidate}_*.png")):
            return candidate
    raise ValueError(f"Cannot detect prefix in {images_dir}")


def _detect_mask_cols(src_dir: Path) -> list[str]:
    """Mask columns that actually exist for this source, in canonical order.

    cubicasa5k ships only `walls` (plus non-mask `walls_check`/`model` dirs we
    ignore); pseudo-12k / manual-1k ship walls + colors + footprints. Carrying
    only the present columns avoids writing zero-filled colors/footprints masks.
    """
    cols = [c for c in ALL_MASK_COLS if (src_dir / "masks" / c).is_dir()]
    if not cols:
        raise ValueError(f"No known mask columns under {src_dir / 'masks'}")
    return cols


def augment_source(
    source: str,
    strategies: list[str],
    force: bool = False,
    workers: int | None = None,
    seed: int = 42,
) -> int:
    src_dir    = RAW_DIR / source
    images_dir = src_dir / "images"
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist. Run download.py first.")
        sys.exit(1)

    prefix    = _detect_prefix(images_dir)
    mask_cols = _detect_mask_cols(src_dir)
    combined  = "_".join(strategies)
    out_dir   = AUG_DIR / source / combined
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for col in mask_cols:
        (out_dir / "masks" / col).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob(f"{prefix}_*.png"))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return 0

    # How many output files per input image
    rand_strat  = next((s for s in strategies if s in RAND_CROP_STRATEGIES), None)
    n_per_img   = RAND_CROP_STRATEGIES[rand_strat][1] if rand_strat else 1
    total_files = len(image_paths) * n_per_img

    n_workers = workers or os.cpu_count() or 1
    print(f"\nSource   : {source}  ({len(image_paths)} images → {total_files} output files)")
    print(f"Pipeline : {' -> '.join(strategies)}")
    print(f"Workers  : {n_workers}")
    print(f"Masks    : {', '.join(mask_cols)}")
    print(f"Output   : {out_dir.relative_to(ROOT)}")

    tasks = [
        (img_path, src_dir, strategies, out_dir, combined, force, seed, mask_cols)
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

    skipped = total_files - written
    if skipped:
        print(f"Skipped {skipped} already-existing files (--force to overwrite).")
    print(f"Written: {written} -> {out_dir.relative_to(ROOT)}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _valid_strategy(name: str) -> bool:
    """True if *name* is a known strategy (fixed, rand-crop, howallow<N>, chroma<T>)."""
    if name in STRATEGIES or name in RAND_CROP_STRATEGIES:
        return True
    try:
        # malformed params (e.g. howallow0, chroma300) raise ValueError -> unknown
        return (_resolve_howallow(name) is not None
                or _resolve_chroma(name) is not None)
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment floor-plan images with an ordered strategy pipeline.")
    # howallow<N> and chroma<T> are parameterized, so argparse `choices` can't
    # enumerate them — validate manually below instead.
    fixed_strategies = list(STRATEGIES.keys()) + list(RAND_CROP_STRATEGIES.keys())
    parser.add_argument(
        "--strategies", nargs="+",
        metavar="STRATEGY",
        help=f"Ordered strategies to apply. Available: {fixed_strategies} "
             f"plus howallow<N> (e.g. howallow2, howallow5) "
             f"and chroma<T> (e.g. chroma15, chroma25).",
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
        "--seed", type=int, default=42,
        help="Base random seed for rand-crop strategies (default: 42).")
    parser.add_argument(
        "--list-strategies", action="store_true",
        help="Print available strategies and exit.")

    args = parser.parse_args()

    if args.list_strategies:
        print("Available strategies:")
        for name, fn in STRATEGIES.items():
            doc = fn.__doc__.strip().splitlines()[0] if fn.__doc__ else ""
            print(f"  {name:<14}  {doc}")
        print()
        for name, (pct, n) in RAND_CROP_STRATEGIES.items():
            print(f"  {name:<14}  Random crop {int(pct*100)}% of image, resize back. "
                  f"Generates {n} crop(s) per image.")
        print()
        print(f"  {'howallow<N>':<14}  Hollow out walls keeping an N-pixel rim, "
              f"white interior (e.g. howallow2, howallow5).")
        print(f"  {'chroma<T>':<14}  Bleach coloured pixels (chroma > T) white; "
              f"keep greyscale as averaged grey (e.g. chroma15, chroma25).")
        return

    if not args.strategies:
        parser.error("--strategies is required unless using --list-strategies")

    bad = [s for s in args.strategies if not _valid_strategy(s)]
    if bad:
        parser.error(
            f"unknown strateg{'y' if len(bad) == 1 else 'ies'}: {', '.join(bad)}. "
            f"Available: {list(STRATEGIES.keys()) + list(RAND_CROP_STRATEGIES.keys())} "
            f"plus howallow<N> and chroma<T>.")

    targets = [args.source] if args.source else SOURCES
    total = 0
    for source in targets:
        total += augment_source(source, args.strategies,
                                force=args.force, workers=args.workers,
                                seed=args.seed)

    print(f"\nTotal written: {total}")
    print("Done.")


if __name__ == "__main__":
    main()
