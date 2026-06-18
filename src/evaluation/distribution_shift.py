"""
distribution_shift.py — measure how pixel-intensity distributions differ
between a training source and an external (out-of-distribution) source.

Floor plans are line drawings: a large near-uniform background with thin
ink strokes (walls, text, furniture).  A naive global histogram is therefore
dominated by background and hides the signal that matters.  So this tool
splits every image into:

    background  — the single dominant colour bucket (per image)
    foreground  — the "ink": everything far enough from that background

and compares the two sources on BOTH, separately.

Methodology — the noise floor
-----------------------------
We draw THREE disjoint, seeded samples from the training source.  Comparing
those three against each other tells us how much two same-distribution
samples naturally differ (the "noise floor").  The train-vs-external
distance is only meaningful insofar as it clearly exceeds that floor.

Distance metric
---------------
1-D Wasserstein / Earth-Mover's Distance (scipy).  It is expressed in
intensity units (0..255), is robust to empty histogram bins, and is
directly comparable across the foreground / background checks.

Usage
-----
    python -m src.evaluation.distribution_shift \
        --train-dir data/raw/pseudo-12k/images \
        --external-dir data/external/images \
        --n 200 --seed 42 --out outputs/dist_shift

v1 scope: grayscale intensity, foreground + background EMD, overlaid
histograms.  Per-RGB-channel and 2-D colour histograms are the next
iteration.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Coarse bin width (per channel) used to find the dominant background bucket.
# 16 → 16 buckets/channel; the modal bucket is the paper colour.
_BG_BIN = 16

# A pixel is "foreground" (ink) if its max per-channel distance from the
# dominant background colour exceeds this many intensity levels.
_FG_DELTA = 40


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def list_images(directory: Path) -> list[Path]:
    """All image files directly under `directory`, sorted for determinism."""
    files = [p for p in sorted(directory.iterdir())
             if p.suffix.lower() in _IMG_EXTS]
    if not files:
        raise FileNotFoundError(f"No images found in {directory}")
    return files


def load_rgb(path: Path) -> np.ndarray:
    """
    Load an image as HxWx3 uint8 RGB.

    RGBA is composited over white — transparent regions on architectural
    approvals are paper, so white is the correct fill and keeps the
    background-detection honest.
    """
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    return np.asarray(im)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma, returned as uint8 HxW."""
    return (rgb @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)


# ---------------------------------------------------------------------------
# Background / foreground split
# ---------------------------------------------------------------------------

def dominant_color(rgb: np.ndarray) -> np.ndarray:
    """
    Most frequent coarse colour bucket → its mean RGB (float32, len 3).

    Coarse binning makes this robust to anti-aliasing and JPEG noise: a
    scanned off-white page lands in one bucket even though no two pixels
    share the exact value.
    """
    flat = rgb.reshape(-1, 3)
    binned = (flat // _BG_BIN).astype(np.int32)
    keys = binned[:, 0] * 1_000_000 + binned[:, 1] * 1_000 + binned[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[counts.argmax()]
    # Mean of the actual pixels in the modal bucket (not the bucket centre).
    sel = keys == top
    return flat[sel].mean(axis=0).astype(np.float32)


def foreground_mask(rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Boolean HxW: pixels whose Chebyshev distance from bg exceeds _FG_DELTA."""
    dist = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    return dist > _FG_DELTA


# ---------------------------------------------------------------------------
# Per-source accumulation
# ---------------------------------------------------------------------------

def collect(paths: list[Path], max_fg_per_img: int = 50_000) -> dict:
    """
    Walk a list of image paths and accumulate:
        fg_intensities  — grayscale values of foreground pixels (subsampled)
        bg_intensities  — grayscale value of each image's dominant background
        fg_fraction     — per-image fraction of pixels that are foreground
    """
    fg_chunks: list[np.ndarray] = []
    bg_levels: list[float] = []
    fg_fracs: list[float] = []
    rng = np.random.default_rng(0)  # only for subsampling fg pixels, fixed

    for p in paths:
        rgb = load_rgb(p)
        gray = to_gray(rgb)
        bg = dominant_color(rgb)
        bg_levels.append(float(to_gray(bg[None, None, :])[0, 0]))

        mask = foreground_mask(rgb, bg)
        fg_fracs.append(float(mask.mean()))

        fg_vals = gray[mask]
        if fg_vals.size > max_fg_per_img:
            idx = rng.choice(fg_vals.size, max_fg_per_img, replace=False)
            fg_vals = fg_vals[idx]
        if fg_vals.size:
            fg_chunks.append(fg_vals)

    return {
        "fg": np.concatenate(fg_chunks) if fg_chunks else np.array([], dtype=np.uint8),
        "bg": np.asarray(bg_levels, dtype=np.float32),
        "fg_frac": np.asarray(fg_fracs, dtype=np.float32),
        "n_images": len(paths),
    }


def summary(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def three_disjoint_samples(paths: list[Path], n: int, seed: int) -> list[list[Path]]:
    """
    Three non-overlapping samples of size n (capped so they fit).

    If the source can't supply 3*n distinct images we shrink n and warn —
    the noise-floor estimate just gets noisier, it doesn't break.
    """
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    n_eff = min(n, len(shuffled) // 3)
    if n_eff < n:
        print(f"[warn] only {len(shuffled)} train images; "
              f"shrinking sample size {n} -> {n_eff} to keep 3 disjoint sets")
    if n_eff == 0:
        raise ValueError("Not enough training images for 3 disjoint samples")

    return [shuffled[i * n_eff:(i + 1) * n_eff] for i in range(3)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def overlay_hist(ax, series: dict[str, np.ndarray], bins, title: str):
    for label, arr in series.items():
        if arr.size:
            ax.hist(arr, bins=bins, density=True, histtype="step",
                    linewidth=1.6, label=f"{label} (n={arr.size:,})")
    ax.set_title(title)
    ax.set_xlabel("intensity (0–255)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True, type=Path,
                    help="directory of training source images")
    ap.add_argument("--external-dir", required=True, type=Path,
                    help="directory of external / OOD images")
    ap.add_argument("--n", type=int, default=200,
                    help="images per train sample (3 disjoint samples drawn)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("outputs/dist_shift"),
                    help="output directory for plots + report")
    args = ap.parse_args()

    # Windows consoles default to cp1252; force UTF-8 so the report prints clean.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args.out.mkdir(parents=True, exist_ok=True)

    train_paths = list_images(args.train_dir)
    ext_paths = list_images(args.external_dir)
    samples = three_disjoint_samples(train_paths, args.n, args.seed)

    print(f"train source : {args.train_dir}  ({len(train_paths):,} images)")
    print(f"external     : {args.external_dir}  ({len(ext_paths):,} images)")
    print(f"3 disjoint train samples of {len(samples[0])} each; "
          f"external uses all {len(ext_paths)}\n")

    train_stats = [collect(s) for s in samples]
    ext_stats = collect(ext_paths)

    # ── EMD: noise floor (train-vs-train) vs train-vs-external ────────────
    def emd(a, b):
        if a.size == 0 or b.size == 0:
            return float("nan")
        return float(wasserstein_distance(a, b))

    # pairwise among the three train samples
    floor_fg = [emd(train_stats[i]["fg"], train_stats[j]["fg"])
                for i, j in [(0, 1), (0, 2), (1, 2)]]
    floor_bg = [emd(train_stats[i]["bg"], train_stats[j]["bg"])
                for i, j in [(0, 1), (0, 2), (1, 2)]]

    # train (pooled across 3 samples) vs external
    train_fg_pool = np.concatenate([t["fg"] for t in train_stats])
    train_bg_pool = np.concatenate([t["bg"] for t in train_stats])
    shift_fg = emd(train_fg_pool, ext_stats["fg"])
    shift_bg = emd(train_bg_pool, ext_stats["bg"])

    floor_fg_mean = float(np.nanmean(floor_fg))
    floor_bg_mean = float(np.nanmean(floor_bg))

    def verdict(shift, floor):
        if np.isnan(shift) or np.isnan(floor) or floor == 0:
            return "?"
        r = shift / floor
        return (f"{r:.1f}x floor  " +
                ("<< within-distribution noise" if r < 1.5 else
                 "MILD shift" if r < 3 else
                 "CLEAR shift" if r < 6 else "SEVERE shift"))

    # ── Report ───────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("PIXEL-INTENSITY DISTRIBUTION SHIFT  —  train vs external")
    lines.append("=" * 70)
    lines.append(f"train dir    : {args.train_dir}")
    lines.append(f"external dir : {args.external_dir}")
    lines.append(f"sample size  : {len(samples[0])} x3 (train), "
                 f"{ext_stats['n_images']} (external)")
    lines.append("")
    lines.append("FOREGROUND (ink) intensity")
    lines.append(f"  noise floor EMD (train-vs-train) : "
                 f"{floor_fg_mean:6.2f}   {[round(x,2) for x in floor_fg]}")
    lines.append(f"  shift EMD       (train-vs-ext)   : "
                 f"{shift_fg:6.2f}   -> {verdict(shift_fg, floor_fg_mean)}")
    lines.append(f"  train fg stats : {summary(train_fg_pool)}")
    lines.append(f"  ext   fg stats : {summary(ext_stats['fg'])}")
    lines.append(f"  fg fraction    : train "
                 f"{np.concatenate([t['fg_frac'] for t in train_stats]).mean():.3f}"
                 f" | ext {ext_stats['fg_frac'].mean():.3f}   "
                 "(ink density — proxy for stroke thickness/clutter)")
    lines.append("")
    lines.append("BACKGROUND (paper) intensity")
    lines.append(f"  noise floor EMD (train-vs-train) : "
                 f"{floor_bg_mean:6.2f}   {[round(x,2) for x in floor_bg]}")
    lines.append(f"  shift EMD       (train-vs-ext)   : "
                 f"{shift_bg:6.2f}   -> {verdict(shift_bg, floor_bg_mean)}")
    lines.append(f"  train bg stats : {summary(train_bg_pool)}")
    lines.append(f"  ext   bg stats : {summary(ext_stats['bg'])}")
    lines.append("=" * 70)
    report = "\n".join(lines)
    print(report)
    (args.out / "report.txt").write_text(report, encoding="utf-8")

    # ── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    overlay_hist(
        axes[0],
        {"train (pooled)": train_fg_pool, "external": ext_stats["fg"]},
        bins=np.linspace(0, 255, 64),
        title=f"Foreground intensity  (EMD={shift_fg:.2f}, floor={floor_fg_mean:.2f})",
    )
    overlay_hist(
        axes[1],
        {"train (pooled)": train_bg_pool, "external": ext_stats["bg"]},
        bins=np.linspace(0, 255, 64),
        title=f"Background intensity  (EMD={shift_bg:.2f}, floor={floor_bg_mean:.2f})",
    )
    fig.suptitle("Pixel-intensity distribution shift: train vs external", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out / "histograms.png", dpi=120)
    print(f"\nwrote {args.out/'report.txt'} and {args.out/'histograms.png'}")


if __name__ == "__main__":
    main()
