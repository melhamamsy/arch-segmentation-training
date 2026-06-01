"""
Create dataset version v3: chroma15-cropped fine-tuning split.

v3 design
---------
  Train : ALL chroma15_rand80crop pseudo-12k  (v1 train indices, 10 800)
          + 10% original pseudo-12k            (sampled from v1 train, ~1 080)
          + 5%  chroma15 pseudo-12k            (sampled from v1 train, ~540)
          Total: ~12 420

  Val   : ALL chroma15_rand80crop pseudo-12k  (v1 val indices, 1 200)
          + 10% original pseudo-12k            (sampled from v1 val,  ~120)
          + 5%  chroma15 pseudo-12k            (sampled from v1 val,   ~60)
          Total: ~1 380

  Test  : original manual-1k                  (same 1 007 as v1)
          + chroma15 manual-1k                 (1 007)
          + chroma15_rand80crop manual-1k      (1 007)
          Total: 3 021

Focus: wall detection fine-tuning on colour-reduced, spatially cropped data.
       Colors and footprints masks are preserved in all records.

Prerequisites — run these before create_v3.py:
  python src/data/augment.py --strategies chroma15
  python src/data/augment.py --strategies chroma15 rand80crop

Usage:
  python src/data/create_v3.py
  python src/data/create_v3.py --orig-pct 0.10 --chroma-pct 0.05 --seed 42
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.versioning import (
    AUG_DIR,
    VERSIONS_DIR,
    MASK_COLS,
    _compute_stats,
    _split_composition,
    _log_to_mlflow,
    _setup_logging,
)

# The rand-crop combined strategy directory name
CHROMA_CROP_PIPELINE = "chroma15_rand80crop"
CHROMA_PIPELINE      = "chroma15"
CROP_N               = 1          # n=floor(100/80)=1 for rand80crop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_augmented_record(record: dict, pipeline: str, crop_idx: int = 0) -> dict:
    """
    Build an augmented sample record from an original v1 record.

    pipeline   : e.g. "chroma15" or "chroma15_rand80crop"
    crop_idx   : 0  -> no crop suffix (plain strategy, e.g. "chroma15_pseudo_00000.png")
                 i>0 -> crop index inserted before orig filename
                        (e.g. "chroma15_rand80crop1_pseudo_00000.png")
    """
    orig_fname = record["filename"]
    source     = record["source"]

    if crop_idx > 0:
        # Insert crop index into the rand-crop part of the combined name:
        #   "chroma15_rand80crop" + 1 + "_" + orig -> "chroma15_rand80crop1_pseudo_00000.png"
        aug_fname = f"{CHROMA_CROP_PIPELINE}{crop_idx}_{orig_fname}"
    else:
        aug_fname = f"{pipeline}_{orig_fname}"

    aug_dir = AUG_DIR / source / pipeline
    aug_img = aug_dir / "images" / aug_fname

    masks = {}
    for col in MASK_COLS:
        m = aug_dir / "masks" / col / aug_fname
        if m.exists():
            masks[col] = m.relative_to(ROOT).as_posix()

    return {
        "filename":     aug_fname,
        "source":       source,
        "augmentation": pipeline,
        "image_path":   aug_img.relative_to(ROOT).as_posix(),
        "masks":        masks,
    }


def _validate_coverage(records: list[dict], label: str, log) -> None:
    missing = [r["filename"] for r in records
               if not (ROOT / r["image_path"]).exists()]
    if missing:
        log.error(
            f"{len(missing)} {label} image(s) not found. "
            f"First missing: {missing[0]}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_v3(orig_pct: float = 0.10, chroma_pct: float = 0.05,
              seed: int = 42) -> Path:
    log = _setup_logging()

    v1_path = VERSIONS_DIR / "v1.json"
    if not v1_path.exists():
        log.error("v1.json not found. Run versioning.py create first.")
        sys.exit(1)

    v1 = json.loads(v1_path.read_text())
    v1_train = v1["train"]   # 10 800 pseudo-12k originals
    v1_val   = v1["val"]     # 1 200  pseudo-12k originals
    v1_test  = v1["test"]    # 1 007  manual-1k  originals

    log.info(f"v1 loaded: train={len(v1_train)}  val={len(v1_val)}  test={len(v1_test)}")

    rng = random.Random(seed)

    # ── Train ────────────────────────────────────────────────────────────────
    # All chroma15_rand80crop images (n=1 per original → same count as v1_train)
    crop_train = [_make_augmented_record(r, CHROMA_CROP_PIPELINE, crop_idx=1)
                  for r in v1_train]
    _validate_coverage(crop_train, f"{CHROMA_CROP_PIPELINE}/train", log)

    n_orig_train   = max(1, round(len(v1_train) * orig_pct))
    n_chroma_train = max(1, round(len(v1_train) * chroma_pct))
    orig_train     = rng.sample(v1_train, n_orig_train)
    chroma_train_pool = [_make_augmented_record(r, CHROMA_PIPELINE) for r in v1_train]
    chroma_train   = rng.sample(chroma_train_pool, n_chroma_train)

    train_samples  = crop_train + orig_train + chroma_train
    rng.shuffle(train_samples)

    # ── Val ──────────────────────────────────────────────────────────────────
    crop_val = [_make_augmented_record(r, CHROMA_CROP_PIPELINE, crop_idx=1)
                for r in v1_val]
    _validate_coverage(crop_val, f"{CHROMA_CROP_PIPELINE}/val", log)

    n_orig_val   = max(1, round(len(v1_val) * orig_pct))
    n_chroma_val = max(1, round(len(v1_val) * chroma_pct))
    orig_val     = rng.sample(v1_val, n_orig_val)
    chroma_val_pool = [_make_augmented_record(r, CHROMA_PIPELINE) for r in v1_val]
    chroma_val   = rng.sample(chroma_val_pool, n_chroma_val)

    val_samples  = crop_val + orig_val + chroma_val

    # ── Test ─────────────────────────────────────────────────────────────────
    chroma_test = [_make_augmented_record(r, CHROMA_PIPELINE) for r in v1_test]
    crop_test   = [_make_augmented_record(r, CHROMA_CROP_PIPELINE, crop_idx=1)
                   for r in v1_test]
    _validate_coverage(chroma_test, f"{CHROMA_PIPELINE}/test", log)
    _validate_coverage(crop_test,   f"{CHROMA_CROP_PIPELINE}/test", log)

    test_samples = list(v1_test) + chroma_test + crop_test

    log.info(
        f"v3: train={len(train_samples)} "
        f"({len(crop_train)} chroma15_crop + {n_orig_train} orig + {n_chroma_train} chroma15)  "
        f"val={len(val_samples)} "
        f"({len(crop_val)} chroma15_crop + {n_orig_val} orig + {n_chroma_val} chroma15)  "
        f"test={len(test_samples)} "
        f"({len(v1_test)} orig + {len(v1_test)} chroma15 + {len(v1_test)} chroma15_crop)"
    )

    # ── Statistics & composition ─────────────────────────────────────────────
    all_sources = ["pseudo-12k", "manual-1k"]
    grand_total = len(train_samples) + len(val_samples) + len(test_samples)

    stats = {
        "train": _compute_stats(train_samples, all_sources),
        "val":   _compute_stats(val_samples,   all_sources),
        "test":  _compute_stats(test_samples,  all_sources),
        "grand_total": grand_total,
    }
    splits = {
        "train": _split_composition(train_samples, grand_total),
        "val":   _split_composition(val_samples,   grand_total),
        "test":  _split_composition(test_samples,  grand_total),
    }

    source_meta = {
        "pseudo-12k": {
            "local_dir": "data/raw/pseudo-12k", "augmentation": "original",
            "total_samples": len(v1_train) + len(v1_val),
        },
        f"pseudo-12k/{CHROMA_PIPELINE}": {
            "local_dir": f"data/augmented/pseudo-12k/{CHROMA_PIPELINE}",
            "augmentation": CHROMA_PIPELINE,
            "total_samples": len(v1_train) + len(v1_val),
        },
        f"pseudo-12k/{CHROMA_CROP_PIPELINE}": {
            "local_dir": f"data/augmented/pseudo-12k/{CHROMA_CROP_PIPELINE}",
            "augmentation": CHROMA_CROP_PIPELINE,
            "total_samples": (len(v1_train) + len(v1_val)) * CROP_N,
        },
        "manual-1k": {
            "local_dir": "data/raw/manual-1k", "augmentation": "original",
            "total_samples": len(v1_test),
        },
        f"manual-1k/{CHROMA_PIPELINE}": {
            "local_dir": f"data/augmented/manual-1k/{CHROMA_PIPELINE}",
            "augmentation": CHROMA_PIPELINE,
            "total_samples": len(v1_test),
        },
        f"manual-1k/{CHROMA_CROP_PIPELINE}": {
            "local_dir": f"data/augmented/manual-1k/{CHROMA_CROP_PIPELINE}",
            "augmentation": CHROMA_CROP_PIPELINE,
            "total_samples": len(v1_test) * CROP_N,
        },
    }

    payload = {
        "version_name": "v3",
        "description": (
            f"Wall-focused fine-tuning. "
            f"Train/val: ALL chroma15_rand80crop pseudo-12k (v1 indices, n={CROP_N}) "
            f"+ {orig_pct*100:.0f}% original ({n_orig_train}/{n_orig_val}) "
            f"+ {chroma_pct*100:.0f}% chroma15 ({n_chroma_train}/{n_chroma_val}). "
            f"Test: original + chroma15 + chroma15_rand80crop of manual-1k."
        ),
        "created_utc":   datetime.now(timezone.utc).isoformat(),
        "seed":          seed,
        "orig_pct":      orig_pct,
        "chroma_pct":    chroma_pct,
        "base_version":  "v1",
        "train_sources": [f"pseudo-12k/{CHROMA_CROP_PIPELINE}",
                          "pseudo-12k",
                          f"pseudo-12k/{CHROMA_PIPELINE}"],
        "test_sources":  ["manual-1k",
                          f"manual-1k/{CHROMA_PIPELINE}",
                          f"manual-1k/{CHROMA_CROP_PIPELINE}"],
        "splits":        splits,
        "statistics":    stats,
        "sources":       source_meta,
        "train":         train_samples,
        "val":           val_samples,
        "test":          test_samples,
    }

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    version_file = VERSIONS_DIR / "v3.json"
    if version_file.exists():
        log.warning("v3.json already exists — overwriting.")
    version_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info(f"Version file -> {version_file.relative_to(ROOT)}")

    _log_to_mlflow("v3", payload, version_file, log)

    return version_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create v3 dataset version (chroma15-cropped wall fine-tuning).")
    parser.add_argument("--orig-pct",   type=float, default=0.10,
                        help="Fraction of original images in train/val (default: 0.10).")
    parser.add_argument("--chroma-pct", type=float, default=0.05,
                        help="Fraction of chroma15-only images in train/val (default: 0.05).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_v3(orig_pct=args.orig_pct, chroma_pct=args.chroma_pct, seed=args.seed)


if __name__ == "__main__":
    main()
