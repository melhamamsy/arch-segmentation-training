"""
Create dataset version v5: CubiCasa5k hollow-wall split.

v5 design
---------
  Primary data: data/augmented/cubicasa5k/*/howallow3 — the CubiCasa5k drawings
  with solid walls turned HOLLOW (5px→3px rim, white interior) to match
  inference-time plans that draw walls as outlines rather than filled blocks.
  Split membership is taken verbatim from the same CubiCasa5k trail files used
  by v4:
        trials/cubicasa5k/{train,val,test}.txt

  Each trail line is an index like  "/high_quality_architectural/1191/"
  which maps to up to TWO on-disk howallow3 images (skip whichever is missing):
        data/augmented/cubicasa5k/high_quality_architectural/howallow3/images/
            howallow3_cubicasa_hqa_01191.png
            howallow3_cubicasa_hqa_01191_model.png
  Category → filename prefix:
        high_quality_architectural -> hqa
        high_quality               -> hq
        colorful                   -> c

  Train : 5%  cubicasa5k/*/howallow3            (sampled from the train trail)
          + 100% chroma15 pseudo-12k            (ALL v1 train)
          + 10%  original pseudo-12k            (sampled from v1 train)
          + 10%  chroma15_rand80crop pseudo-12k (SAME 10% picks as original)

  Val   : 5%  cubicasa5k/*/howallow3            (sampled from the val trail)
          + 100% chroma15 pseudo-12k            (ALL v1 val)
          + 10%  original pseudo-12k            (sampled from v1 val)
          + 10%  chroma15_rand80crop pseudo-12k (SAME 10% picks as original)

  Test  : v4 test  ==  ALL cubicasa5k test (ORIGINAL, solid walls)
                       + original manual-1k             (same 1 007 as v1)
                       + chroma15 manual-1k             (1 007)
                       + chroma15_rand80crop manual-1k  (1 007)
          + ALL cubicasa5k/*/howallow3 test            (hollow walls)

Focus: wall detection that generalises to hollow / outline-drawn walls. Training
       sees mostly hollow cubicasa walls (small 5% slice) plus the synthetic
       pseudo-12k chroma pipeline; the test set evaluates BOTH solid (v4) and
       hollow cubicasa walls so the hollow-domain gain is measurable.
       CubiCasa5k provides only 'walls' masks (no colours/footprints), which is
       exactly what the wall-only model needs. howallow3 leaves masks unchanged,
       so wall annotations still describe the full (solid) wall footprint.

Prerequisites — run these before create_v5.py:
  python src/data/augment.py --strategies chroma15
  python src/data/augment.py --strategies chroma15 rand80crop
  python src/data/augment.py --strategies howallow3 \
      --source cubicasa5k/high_quality
  python src/data/augment.py --strategies howallow3 \
      --source cubicasa5k/high_quality_architectural
  python src/data/augment.py --strategies howallow3 --source cubicasa5k/colorful
  (and CubiCasa5k must be extracted to data/raw/cubicasa5k/)

Usage:
  python src/data/create_v5.py
  python src/data/create_v5.py --cubicasa-pct 0.05 --pseudo-pct 0.10 --seed 42
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

# Reuse v4's CubiCasa5k layout + trail parsing + original-record builder
from src.data.create_v4 import (
    TRIALS_DIR,
    CUBICASA_SRC,
    CATEGORY_PREFIX,
    IMAGE_VARIANTS,
    _parse_trail_line,
    _load_cubicasa_split,   # builds ORIGINAL (solid-wall) cubicasa records
)

# The hollow-wall augmentation directory name (data/augmented/cubicasa5k/*/<this>)
HOWALLOW_PIPELINE = "howallow3"


# ---------------------------------------------------------------------------
# CubiCasa5k howallow3 helpers (augmented, hollow-wall variant)
# ---------------------------------------------------------------------------

def _cubicasa_howallow3_records(category: str, index: str) -> list[dict]:
    """
    Build howallow3 sample records for one CubiCasa5k index.

    Mirrors create_v4._cubicasa_records but points at the augmented hollow-wall
    tree and prefixes filenames with "howallow3_". An index yields 0, 1, or 2
    records (base + _model); variants absent on disk are skipped.
    """
    prefix = CATEGORY_PREFIX.get(category)
    if prefix is None:
        return []

    idx5    = f"{int(index):05d}"
    aug_dir = AUG_DIR / "cubicasa5k" / category / HOWALLOW_PIPELINE
    records: list[dict] = []

    for variant in IMAGE_VARIANTS:
        fname = f"{HOWALLOW_PIPELINE}_cubicasa_{prefix}_{idx5}{variant}.png"
        img   = aug_dir / "images" / fname
        if not img.exists():
            continue

        masks = {}
        for col in MASK_COLS:                       # only 'walls' exists for cubicasa
            m = aug_dir / "masks" / col / fname
            if m.exists():
                masks[col] = m.relative_to(ROOT).as_posix()

        records.append({
            "filename":     fname,
            "source":       CUBICASA_SRC,
            "augmentation": HOWALLOW_PIPELINE,
            "image_path":   img.relative_to(ROOT).as_posix(),
            "masks":        masks,
        })

    return records


def _load_cubicasa_howallow3_split(split_name: str, log) -> list[dict]:
    """Read trials/cubicasa5k/<split_name>.txt and resolve to howallow3 records."""
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
        recs = _cubicasa_howallow3_records(*parsed)
        if not recs:
            n_missing += 1
        records.extend(recs)

    log.info(
        f"cubicasa5k/{HOWALLOW_PIPELINE}/{split_name}: {n_ids} ids -> "
        f"{len(records)} images ({n_missing} ids had no image on disk, skipped)"
    )
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_v5(cubicasa_pct: float = 0.05, pseudo_pct: float = 0.10,
              seed: int = 42) -> Path:
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

    # ── CubiCasa5k howallow3 splits (primary data, 5% of train/val) ───────────
    cuba_train_full = _load_cubicasa_howallow3_split("train", log)
    cuba_val_full   = _load_cubicasa_howallow3_split("val",   log)
    cuba_test_hollow = _load_cubicasa_howallow3_split("test", log)   # FULL test
    _validate_coverage(cuba_train_full, f"cubicasa5k/{HOWALLOW_PIPELINE}/train", log)
    _validate_coverage(cuba_val_full,   f"cubicasa5k/{HOWALLOW_PIPELINE}/val",   log)
    _validate_coverage(cuba_test_hollow, f"cubicasa5k/{HOWALLOW_PIPELINE}/test", log)

    def _sample_pct(pool: list[dict], pct: float) -> list[dict]:
        n = max(1, round(len(pool) * pct))
        return rng.sample(pool, n)

    cuba_train = _sample_pct(cuba_train_full, cubicasa_pct)
    cuba_val   = _sample_pct(cuba_val_full,   cubicasa_pct)

    # ── CubiCasa5k ORIGINAL test (solid walls, same as v4) ────────────────────
    cuba_test_orig = _load_cubicasa_split("test", log)
    _validate_coverage(cuba_test_orig, "cubicasa5k/test", log)

    # ── pseudo-12k slices: 100% chroma15 + shared 10% (original + crop) ───────
    def _pseudo_slice(pool: list[dict], pct: float) -> tuple[list, list, list]:
        # chroma15: every image in the split (100%)
        chroma_recs = [_make_augmented_record(r, CHROMA_PIPELINE) for r in pool]
        # original + chroma15_rand80crop share the SAME pct picks
        picks     = _sample_pct(pool, pct)
        orig_recs = list(picks)
        crop_recs = [_make_augmented_record(r, CHROMA_CROP_PIPELINE, crop_idx=1)
                     for r in picks]
        return chroma_recs, orig_recs, crop_recs

    chroma_train, orig_train, crop_train = _pseudo_slice(v1_train, pseudo_pct)
    _validate_coverage(chroma_train, f"{CHROMA_PIPELINE}/train",      log)
    _validate_coverage(crop_train,   f"{CHROMA_CROP_PIPELINE}/train", log)

    chroma_val, orig_val, crop_val = _pseudo_slice(v1_val, pseudo_pct)
    _validate_coverage(chroma_val, f"{CHROMA_PIPELINE}/val",      log)
    _validate_coverage(crop_val,   f"{CHROMA_CROP_PIPELINE}/val", log)

    # ── Assemble train / val ──────────────────────────────────────────────────
    train_samples = cuba_train + chroma_train + orig_train + crop_train
    rng.shuffle(train_samples)

    val_samples = cuba_val + chroma_val + orig_val + crop_val

    # ── Test: v4 test (cubicasa orig + manual slices) + cubicasa howallow3 ────
    chroma_test = [_make_augmented_record(r, CHROMA_PIPELINE) for r in v1_test]
    crop_test   = [_make_augmented_record(r, CHROMA_CROP_PIPELINE, crop_idx=1)
                   for r in v1_test]
    _validate_coverage(chroma_test, f"{CHROMA_PIPELINE}/test",      log)
    _validate_coverage(crop_test,   f"{CHROMA_CROP_PIPELINE}/test", log)

    test_samples = (cuba_test_orig + list(v1_test) + chroma_test + crop_test
                    + cuba_test_hollow)

    n_orig_train, n_crop_train = len(orig_train), len(crop_train)
    n_orig_val,   n_crop_val   = len(orig_val),   len(crop_val)
    log.info(
        f"v5: train={len(train_samples)} "
        f"({len(cuba_train)}/{len(cuba_train_full)} cubicasa howallow3 + "
        f"{len(chroma_train)} chroma15 + {n_orig_train} orig + {n_crop_train} crop)  "
        f"val={len(val_samples)} "
        f"({len(cuba_val)}/{len(cuba_val_full)} cubicasa howallow3 + "
        f"{len(chroma_val)} chroma15 + {n_orig_val} orig + {n_crop_val} crop)  "
        f"test={len(test_samples)} "
        f"({len(cuba_test_orig)} cubicasa orig + {len(v1_test)} manual orig + "
        f"{len(v1_test)} chroma15 + {len(v1_test)} crop + "
        f"{len(cuba_test_hollow)} cubicasa howallow3)"
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

    n_cuba_hollow = len(cuba_train) + len(cuba_val) + len(cuba_test_hollow)
    n_pseudo_chroma = len(chroma_train) + len(chroma_val)
    n_pseudo_orig   = n_orig_train + n_orig_val
    n_pseudo_crop   = n_crop_train + n_crop_val

    source_meta = {
        "cubicasa5k": {
            "local_dir": "data/raw/cubicasa5k", "augmentation": "original",
            "total_samples": len(cuba_test_orig),   # solid-wall, test only
            "categories": list(CATEGORY_PREFIX.keys()),
            "trail_files": "trials/cubicasa5k/{train,val,test}.txt",
        },
        f"cubicasa5k/{HOWALLOW_PIPELINE}": {
            "local_dir": f"data/augmented/cubicasa5k/*/{HOWALLOW_PIPELINE}",
            "augmentation": HOWALLOW_PIPELINE,
            "total_samples": n_cuba_hollow,
            "categories": list(CATEGORY_PREFIX.keys()),
            "trail_files": "trials/cubicasa5k/{train,val,test}.txt",
        },
        "pseudo-12k": {
            "local_dir": "data/raw/pseudo-12k", "augmentation": "original",
            "total_samples": n_pseudo_orig,
        },
        f"pseudo-12k/{CHROMA_PIPELINE}": {
            "local_dir": f"data/augmented/pseudo-12k/{CHROMA_PIPELINE}",
            "augmentation": CHROMA_PIPELINE,
            "total_samples": n_pseudo_chroma,
        },
        f"pseudo-12k/{CHROMA_CROP_PIPELINE}": {
            "local_dir": f"data/augmented/pseudo-12k/{CHROMA_CROP_PIPELINE}",
            "augmentation": CHROMA_CROP_PIPELINE,
            "total_samples": n_pseudo_crop * CROP_N,
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
        "version_name": "v5",
        "description": (
            f"CubiCasa5k hollow-wall ({HOWALLOW_PIPELINE}) detection. "
            f"Train/val: {cubicasa_pct*100:.0f}% cubicasa5k/{HOWALLOW_PIPELINE} "
            f"(trail-file split) + 100% chroma15 + {pseudo_pct*100:.0f}% each of "
            f"original / chroma15_rand80crop pseudo-12k (shared picks). "
            f"Test: v4 test (cubicasa5k original + original/chroma15/"
            f"chroma15_rand80crop manual-1k) + ALL cubicasa5k/{HOWALLOW_PIPELINE} test."
        ),
        "created_utc":    datetime.now(timezone.utc).isoformat(),
        "seed":           seed,
        "cubicasa_pct":   cubicasa_pct,
        "pseudo_pct":     pseudo_pct,
        "base_version":   "v1",
        "train_sources":  [f"cubicasa5k/{HOWALLOW_PIPELINE}",
                           f"pseudo-12k/{CHROMA_PIPELINE}",
                           "pseudo-12k",
                           f"pseudo-12k/{CHROMA_CROP_PIPELINE}"],
        "test_sources":   ["cubicasa5k",
                           f"cubicasa5k/{HOWALLOW_PIPELINE}",
                           "manual-1k",
                           f"manual-1k/{CHROMA_PIPELINE}",
                           f"manual-1k/{CHROMA_CROP_PIPELINE}"],
        "splits":         splits,
        "statistics":     stats,
        "sources":        source_meta,
        "train":          train_samples,
        "val":            val_samples,
        "test":           test_samples,
    }

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    version_file = VERSIONS_DIR / "v5.json"
    if version_file.exists():
        log.warning("v5.json already exists — overwriting.")
    version_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info(f"Version file -> {version_file.relative_to(ROOT)}")

    _log_to_mlflow("v5", payload, version_file, log)

    return version_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create v5 dataset version (CubiCasa5k hollow-wall detection).")
    parser.add_argument("--cubicasa-pct", type=float, default=0.05,
                        help="Fraction of cubicasa5k/howallow3 retained per "
                             "train/val split (default: 0.05).")
    parser.add_argument("--pseudo-pct", type=float, default=0.10,
                        help="Fraction of v1 pseudo-12k used for the shared "
                             "original + chroma15_rand80crop slices in train/val "
                             "(default: 0.10). chroma15 always uses 100%%.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_v5(cubicasa_pct=args.cubicasa_pct, pseudo_pct=args.pseudo_pct,
              seed=args.seed)


if __name__ == "__main__":
    main()
