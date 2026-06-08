"""
Create dataset version v4: CubiCasa5k wall-detection split.

v4 design
---------
  Primary data: data/raw/cubicasa5k  (colorful / high_quality /
  high_quality_architectural — real, high-quality architectural drawings).
  Split membership is taken verbatim from the CubiCasa5k trail files:
        trials/cubicasa5k/{train,val,test}.txt

  Each trail line is an index like  "/high_quality_architectural/1191/"
  which maps to up to TWO on-disk images (skip whichever is missing):
        data/raw/cubicasa5k/high_quality_architectural/images/cubicasa_hqa_01191.png
        data/raw/cubicasa5k/high_quality_architectural/images/cubicasa_hqa_01191_model.png
  Category → filename prefix:
        high_quality_architectural -> hqa
        high_quality               -> hq
        colorful                   -> c

  Train : ALL cubicasa5k train images
          + 5% chroma15_rand80crop pseudo-12k  (sampled from v1 train)
          + 5% original pseudo-12k             (sampled from v1 train)
          + 5% chroma15 pseudo-12k             (sampled from v1 train)

  Val   : ALL cubicasa5k val images
          + 5% chroma15_rand80crop pseudo-12k  (sampled from v1 val)
          + 5% original pseudo-12k             (sampled from v1 val)
          + 5% chroma15 pseudo-12k             (sampled from v1 val)

  Test  : ALL cubicasa5k test images
          + original manual-1k                 (same 1 007 as v1, = v3 test)
          + chroma15 manual-1k                 (1 007)
          + chroma15_rand80crop manual-1k      (1 007)

Focus: wall detection on real architectural drawings, with a small slice of
       the synthetic pseudo-12k pipeline retained to avoid catastrophic drift.
       CubiCasa5k provides only 'walls' masks (no colours/footprints), which
       is exactly what the wall-only (v2) model needs.

Prerequisites — run these before create_v4.py:
  python src/data/augment.py --strategies chroma15
  python src/data/augment.py --strategies chroma15 rand80crop
  (and CubiCasa5k must be extracted to data/raw/cubicasa5k/)

Usage:
  python src/data/create_v4.py
  python src/data/create_v4.py --pseudo-pct 0.05 --seed 42
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
    RAW_DIR,
    VERSIONS_DIR,
    MASK_COLS,
    _compute_stats,
    _split_composition,
    _log_to_mlflow,
    _setup_logging,
)

# Reuse v3's augmented-record builder for the pseudo-12k / manual-1k slices
from src.data.create_v3 import (
    CHROMA_PIPELINE,
    CHROMA_CROP_PIPELINE,
    CROP_N,
    _make_augmented_record,
    _validate_coverage,
)

# ── CubiCasa5k layout ────────────────────────────────────────────────────────
TRIALS_DIR    = ROOT / "trials" / "cubicasa5k"
CUBICASA_DIR  = RAW_DIR / "cubicasa5k"
CUBICASA_SRC  = "cubicasa5k"

# Trail-file category  ->  on-disk filename prefix
CATEGORY_PREFIX = {
    "high_quality_architectural": "hqa",
    "high_quality":               "hq",
    "colorful":                   "c",
}

# Each index maps to a base render and a "_model" render (either may be absent)
IMAGE_VARIANTS = ("", "_model")


# ---------------------------------------------------------------------------
# CubiCasa5k helpers
# ---------------------------------------------------------------------------

def _parse_trail_line(line: str) -> tuple[str, str] | None:
    """
    "/high_quality_architectural/1191/"  ->  ("high_quality_architectural", "1191")
    Returns None for blank / malformed lines.
    """
    parts = line.strip().strip("/").split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _cubicasa_records(category: str, index: str) -> list[dict]:
    """
    Build sample records for one CubiCasa5k index.

    An index yields 0, 1, or 2 records (base + _model). Variants whose image
    file does not exist on disk are skipped.
    """
    prefix = CATEGORY_PREFIX.get(category)
    if prefix is None:
        return []

    idx5     = f"{int(index):05d}"
    base_dir = CUBICASA_DIR / category
    records: list[dict] = []

    for variant in IMAGE_VARIANTS:
        fname   = f"cubicasa_{prefix}_{idx5}{variant}.png"
        img     = base_dir / "images" / fname
        if not img.exists():
            continue

        masks = {}
        for col in MASK_COLS:                       # only 'walls' exists for cubicasa
            m = base_dir / "masks" / col / fname
            if m.exists():
                masks[col] = m.relative_to(ROOT).as_posix()

        records.append({
            "filename":     fname,
            "source":       CUBICASA_SRC,
            "augmentation": "original",
            "image_path":   img.relative_to(ROOT).as_posix(),
            "masks":        masks,
        })

    return records


def _load_cubicasa_split(split_name: str, log) -> list[dict]:
    """Read trials/cubicasa5k/<split_name>.txt and resolve to sample records."""
    trail = TRIALS_DIR / f"{split_name}.txt"
    if not trail.exists():
        log.error(f"Trail file not found: {trail}")
        sys.exit(1)

    records: list[dict] = []
    n_ids = n_missing = 0
    for line in trail.read_text().splitlines():
        parsed = _parse_trail_line(line)
        if parsed is None:
            continue
        n_ids += 1
        recs = _cubicasa_records(*parsed)
        if not recs:
            n_missing += 1
        records.extend(recs)

    log.info(
        f"cubicasa5k/{split_name}: {n_ids} ids -> {len(records)} images "
        f"({n_missing} ids had no image on disk, skipped)"
    )
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_v4(pseudo_pct: float = 0.05, seed: int = 42) -> Path:
    log = _setup_logging()

    v1_path = VERSIONS_DIR / "v1.json"
    if not v1_path.exists():
        log.error("v1.json not found. Run versioning.py create first.")
        sys.exit(1)

    v1       = json.loads(v1_path.read_text())
    v1_train = v1["train"]   # 10 800 pseudo-12k originals
    v1_val   = v1["val"]     # 1 200  pseudo-12k originals
    v1_test  = v1["test"]    # 1 007  manual-1k  originals
    log.info(f"v1 loaded: train={len(v1_train)}  val={len(v1_val)}  test={len(v1_test)}")

    rng = random.Random(seed)

    # ── CubiCasa5k splits (primary data) ──────────────────────────────────────
    cuba_train = _load_cubicasa_split("train", log)
    cuba_val   = _load_cubicasa_split("val",   log)
    cuba_test  = _load_cubicasa_split("test",  log)
    _validate_coverage(cuba_train, "cubicasa5k/train", log)
    _validate_coverage(cuba_val,   "cubicasa5k/val",   log)
    _validate_coverage(cuba_test,  "cubicasa5k/test",  log)

    # ── pseudo-12k retention slice (5% each: crop + original + chroma15) ──────
    def _pseudo_slice(pool: list[dict], pct: float) -> tuple[list, list, list]:
        n = max(1, round(len(pool) * pct))
        picks       = rng.sample(pool, n)
        crop_recs   = [_make_augmented_record(r, CHROMA_CROP_PIPELINE, crop_idx=1) for r in picks]
        orig_recs   = list(picks)
        chroma_recs = [_make_augmented_record(r, CHROMA_PIPELINE) for r in picks]
        return crop_recs, orig_recs, chroma_recs

    crop_train, orig_train, chroma_train = _pseudo_slice(v1_train, pseudo_pct)
    _validate_coverage(crop_train,   f"{CHROMA_CROP_PIPELINE}/train", log)
    _validate_coverage(chroma_train, f"{CHROMA_PIPELINE}/train",      log)

    crop_val, orig_val, chroma_val = _pseudo_slice(v1_val, pseudo_pct)
    _validate_coverage(crop_val,   f"{CHROMA_CROP_PIPELINE}/val", log)
    _validate_coverage(chroma_val, f"{CHROMA_PIPELINE}/val",      log)

    # ── Assemble train / val ──────────────────────────────────────────────────
    train_samples = cuba_train + crop_train + orig_train + chroma_train
    rng.shuffle(train_samples)

    val_samples = cuba_val + crop_val + orig_val + chroma_val

    # ── Test: cubicasa test + v3 test (manual-1k orig + chroma15 + crop) ──────
    chroma_test = [_make_augmented_record(r, CHROMA_PIPELINE) for r in v1_test]
    crop_test   = [_make_augmented_record(r, CHROMA_CROP_PIPELINE, crop_idx=1) for r in v1_test]
    _validate_coverage(chroma_test, f"{CHROMA_PIPELINE}/test",      log)
    _validate_coverage(crop_test,   f"{CHROMA_CROP_PIPELINE}/test", log)

    test_samples = cuba_test + list(v1_test) + chroma_test + crop_test

    n_pseudo_train = len(orig_train)
    n_pseudo_val   = len(orig_val)
    log.info(
        f"v4: train={len(train_samples)} "
        f"({len(cuba_train)} cubicasa + 3x{n_pseudo_train} pseudo slices)  "
        f"val={len(val_samples)} "
        f"({len(cuba_val)} cubicasa + 3x{n_pseudo_val} pseudo slices)  "
        f"test={len(test_samples)} "
        f"({len(cuba_test)} cubicasa + {len(v1_test)} orig + {len(v1_test)} chroma15 "
        f"+ {len(v1_test)} chroma15_crop manual-1k)"
    )

    # ── Statistics & composition ─────────────────────────────────────────────
    all_sources = ["cubicasa5k", "pseudo-12k", "manual-1k"]
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
        "cubicasa5k": {
            "local_dir": "data/raw/cubicasa5k", "augmentation": "original",
            "total_samples": len(cuba_train) + len(cuba_val) + len(cuba_test),
            "categories": list(CATEGORY_PREFIX.keys()),
            "trail_files": "trials/cubicasa5k/{train,val,test}.txt",
        },
        "pseudo-12k": {
            "local_dir": "data/raw/pseudo-12k", "augmentation": "original",
            "total_samples": n_pseudo_train + n_pseudo_val,
        },
        f"pseudo-12k/{CHROMA_PIPELINE}": {
            "local_dir": f"data/augmented/pseudo-12k/{CHROMA_PIPELINE}",
            "augmentation": CHROMA_PIPELINE,
            "total_samples": n_pseudo_train + n_pseudo_val,
        },
        f"pseudo-12k/{CHROMA_CROP_PIPELINE}": {
            "local_dir": f"data/augmented/pseudo-12k/{CHROMA_CROP_PIPELINE}",
            "augmentation": CHROMA_CROP_PIPELINE,
            "total_samples": (n_pseudo_train + n_pseudo_val) * CROP_N,
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
        "version_name": "v4",
        "description": (
            f"CubiCasa5k wall detection. "
            f"Train/val: ALL cubicasa5k (trail-file split) "
            f"+ {pseudo_pct*100:.0f}% each of chroma15_rand80crop / original / chroma15 "
            f"pseudo-12k ({n_pseudo_train}/{n_pseudo_val} per slice). "
            f"Test: cubicasa5k test + original + chroma15 + chroma15_rand80crop of manual-1k."
        ),
        "created_utc":   datetime.now(timezone.utc).isoformat(),
        "seed":          seed,
        "pseudo_pct":    pseudo_pct,
        "base_version":  "v1",
        "train_sources": ["cubicasa5k",
                          f"pseudo-12k/{CHROMA_CROP_PIPELINE}",
                          "pseudo-12k",
                          f"pseudo-12k/{CHROMA_PIPELINE}"],
        "test_sources":  ["cubicasa5k",
                          "manual-1k",
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
    version_file = VERSIONS_DIR / "v4.json"
    if version_file.exists():
        log.warning("v4.json already exists — overwriting.")
    version_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info(f"Version file -> {version_file.relative_to(ROOT)}")

    _log_to_mlflow("v4", payload, version_file, log)

    return version_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create v4 dataset version (CubiCasa5k wall detection).")
    parser.add_argument("--pseudo-pct", type=float, default=0.05,
                        help="Fraction of v1 pseudo-12k retained per slice "
                             "(crop/original/chroma15) in train/val (default: 0.05).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_v4(pseudo_pct=args.pseudo_pct, seed=args.seed)


if __name__ == "__main__":
    main()
