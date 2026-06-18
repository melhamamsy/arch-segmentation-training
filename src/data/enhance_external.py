"""
enhance_external.py — pull external (out-of-distribution) floor plans toward
the training pixel-intensity distribution via per-channel histogram matching.

Why histogram matching (not an iterative search)
-------------------------------------------------
Matching the foreground intensity CDF of the external image onto the training
CDF is exact 1-D optimal transport: it is the monotonic map that minimises the
Earth-Mover's distance between the two distributions, in closed form.  No
iteration, no learning rate — one pass over the histogram drives the
distribution shift to its theoretical minimum.

The diagnostic (src/evaluation/distribution_shift.py) found the external ink is
crisp pure-black (spike at 0) while training ink sits ~170, and external paper
is brighter (254 vs ~245).  A per-channel LUT corrects both:

    * foreground (ink)  -> histogram-matched onto the training ink CDF
    * background (paper) -> anchored to the training paper level
    * the two are stitched into ONE monotonic 256-entry LUT per channel, so
      structure is preserved (a brighter input never becomes darker) and there
      is no seam between ink and paper.

Why foreground-only matching
----------------------------
External plans are ~12% ink; training plans are ~48% ink.  Matching the WHOLE
image histogram would try to push 48% of external pixels dark — turning white
paper into gray to hit that mass target.  Building the ink LUT from foreground
pixels only, then anchoring the paper separately, avoids that.

What this canNOT fix
--------------------
The ink-density / stroke-thickness / clutter gap is spatial, not tonal — a
tone curve cannot add ink.  This step closes the colour/intensity gap only.

Usage
-----
    python -m src.data.enhance_external \
        --train-dir data/raw/pseudo-12k/images \
        --external-dir data/external/images \
        --n 200 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance

from src.evaluation.distribution_shift import (
    list_images,
    load_rgb,
    to_gray,
    dominant_color,
    foreground_mask,
)

_REF_PER_IMG = 30_000      # foreground px sampled per training image
_REF_TOTAL   = 500_000     # cap on pooled reference px per channel


# ---------------------------------------------------------------------------
# Training reference (per-channel foreground histogram + background level)
# ---------------------------------------------------------------------------

def build_reference(paths: list[Path], n: int, seed: int) -> dict:
    """Pool per-channel foreground intensities + mean background from a sample."""
    rng = np.random.default_rng(seed)
    sample = list(paths)
    rng.shuffle(sample)
    sample = sample[:n]

    fg_ch: list[list[np.ndarray]] = [[], [], []]
    bg_levels: list[np.ndarray] = []

    for p in sample:
        rgb = load_rgb(p)
        bg = dominant_color(rgb)
        bg_levels.append(bg)
        mask = foreground_mask(rgb, bg)
        fg = rgb[mask]
        if fg.shape[0] == 0:
            continue
        if fg.shape[0] > _REF_PER_IMG:
            idx = rng.choice(fg.shape[0], _REF_PER_IMG, replace=False)
            fg = fg[idx]
        for c in range(3):
            fg_ch[c].append(fg[:, c])

    ref_fg = []
    for c in range(3):
        pooled = np.concatenate(fg_ch[c]).astype(np.uint8)
        if pooled.size > _REF_TOTAL:
            idx = rng.choice(pooled.size, _REF_TOTAL, replace=False)
            pooled = pooled[idx]
        ref_fg.append(pooled)

    return {
        "fg": ref_fg,                                 # list of 3 uint8 arrays
        "bg": np.stack(bg_levels).mean(axis=0),       # mean bg colour [3] float
        "n": len(sample),
    }


# ---------------------------------------------------------------------------
# Histogram matching -> monotonic 256-entry LUT
# ---------------------------------------------------------------------------

def _cdf(vals: np.ndarray) -> np.ndarray:
    """Empirical CDF of uint8 values, evaluated on the grid 0..255."""
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    c = np.cumsum(hist)
    return c / c[-1]


def build_lut(
    ext_fg: np.ndarray,      # external foreground values for this channel (uint8)
    ref_fg: np.ndarray,      # training foreground reference (uint8)
    ext_bg: float,           # external background level for this channel
    ref_bg: float,           # training background level for this channel
) -> np.ndarray:
    """
    One monotonic uint8 LUT[256] = foreground histogram-match  +  paper anchor.

    Foreground part : classic histogram matching (ext_fg CDF -> ref_fg CDF).
    Background part : the external paper level is pinned to the training paper
                      level; values between the ink range and paper are linearly
                      bridged.  np.maximum.accumulate guarantees monotonicity.
    """
    grid = np.arange(256)

    # --- foreground histogram match ---------------------------------------
    ext_cdf = _cdf(ext_fg)
    ref_vals, ref_counts = np.unique(ref_fg, return_counts=True)
    ref_q = np.cumsum(ref_counts) / ref_counts.sum()
    fg_map = np.interp(ext_cdf, ref_q, ref_vals)      # [256] ink mapping

    # Ink only occupies roughly [0, fg_hi]; above that we hand over to paper.
    fg_hi = int(np.percentile(ext_fg, 99))
    fg_hi = min(fg_hi, int(round(ext_bg)) - 1)
    fg_hi = max(fg_hi, 1)

    # --- stitch ink range + paper anchor into control points --------------
    ctrl_in  = list(grid[: fg_hi + 1])
    ctrl_out = list(fg_map[: fg_hi + 1])
    # cap ink output below the training paper level so it can't exceed paper
    ctrl_out = [min(v, ref_bg - 1.0) for v in ctrl_out]

    ctrl_in  += [int(round(ext_bg)), 255]
    ctrl_out += [ref_bg, ref_bg]

    lut = np.interp(grid, ctrl_in, ctrl_out)
    lut = np.maximum.accumulate(lut)                  # enforce monotonic
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


def apply_luts(rgb: np.ndarray, luts: list[np.ndarray]) -> np.ndarray:
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = luts[c][rgb[..., c]]
    return out


# ---------------------------------------------------------------------------
# Grayscale fg EMD — same units as the shift diagnostic, for before/after
# ---------------------------------------------------------------------------

def gray_fg_emd(rgb: np.ndarray, mask: np.ndarray, ref_gray: np.ndarray) -> float:
    vals = to_gray(rgb)[mask].astype(np.float32)
    if vals.size == 0:
        return float("nan")
    return float(wasserstein_distance(vals, ref_gray))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True, type=Path)
    ap.add_argument("--external-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200, help="training images for reference")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="enhanceV1_", help="output filename prefix")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    train_paths = list_images(args.train_dir)
    ext_paths = [p for p in list_images(args.external_dir)
                 if not p.name.startswith(args.prefix)]
    if not ext_paths:
        print("No un-enhanced external images found.")
        return

    print(f"building reference from {min(args.n, len(train_paths))} train images...")
    ref = build_reference(train_paths, args.n, args.seed)
    ref_gray = (0.299 * ref["fg"][0]
                + 0.587 * ref["fg"][1]
                + 0.114 * ref["fg"][2]).astype(np.float32)
    print(f"reference: mean paper = {ref['bg'].round(1).tolist()}, "
          f"ink per-channel medians = "
          f"{[int(np.median(ref['fg'][c])) for c in range(3)]}\n")

    summary = []
    for p in ext_paths:
        rgb = load_rgb(p)
        bg = dominant_color(rgb)
        mask = foreground_mask(rgb, bg)
        if mask.sum() == 0:
            print(f"[skip] {p.name}: no foreground detected")
            continue

        luts = [build_lut(rgb[..., c][mask], ref["fg"][c], float(bg[c]), float(ref["bg"][c]))
                for c in range(3)]
        enhanced = apply_luts(rgb, luts)

        before = gray_fg_emd(rgb, mask, ref_gray)
        after = gray_fg_emd(enhanced, mask, ref_gray)   # mask unchanged: map is monotonic

        out_path = p.with_name(args.prefix + p.name)
        Image.fromarray(enhanced).save(out_path)

        summary.append((p.name, before, after))
        drop = 100 * (before - after) / before if before else 0.0
        print(f"{p.name}")
        print(f"    gray-fg EMD  {before:6.2f} -> {after:6.2f}   ({drop:+.0f}%)")
        print(f"    saved {out_path.name}")

    print("\n" + "=" * 60)
    print("SUMMARY  grayscale foreground EMD vs training")
    print("=" * 60)
    for name, b, a in summary:
        print(f"  {name:42s} {b:6.2f} -> {a:6.2f}")
    if summary:
        bs = np.array([b for _, b, _ in summary])
        as_ = np.array([a for _, _, a in summary])
        print(f"  {'MEAN':42s} {bs.mean():6.2f} -> {as_.mean():6.2f}")


if __name__ == "__main__":
    main()
