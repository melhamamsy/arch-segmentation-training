"""
Create dataset version v2: colour-reduced fine-tuning split.

v2 design
---------
  Train : greyscale(v1_train, 10 800)
          + chroma15(v1_train, 10 800)
          + 10% original pseudo-12k sampled from v1_train (~1 080)
          Total: ~22 680

  Val   : greyscale(v1_val, 1 200)
          + chroma15(v1_val, 1 200)
          + 10% original pseudo-12k sampled from v1_val (~120)
          Total: ~2 520

  Test  : original manual-1k (same 1 007 as v1)
          + greyscale(manual-1k, 1 007)
          + chroma15(manual-1k,  1 007)
          Total: 3 021

Prerequisites — run these before create_v2.py:
  python src/data/augment.py --strategies greyscale
  python src/data/augment.py --strategies chroma15

Usage:
  python src/data/create_v2.py
  python src/data/create_v2.py --orig-pct 0.10 --seed 42
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

AUG_STRATEGIES = ["greyscale", "chroma15"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_augmented_record(record: dict, strategy: str) -> dict:
    """
    Derive an augmented sample record from an original record.
    Maps  {source}/{orig_fname}  ->  data/augmented/{source}/{strategy}/{strategy}_{orig_fname}
    """
    orig_fname = record["filename"]          # e.g. pseudo_00042.png / manual_00042.png
    aug_fname  = f"{strategy}_{orig_fname}"
    source     = record["source"]            # e.g. pseudo-12k / manual-1k

    aug_img = AUG_DIR / source / strategy / "images" / aug_fname
    masks = {}
    for col in MASK_COLS:
        m = AUG_DIR / source / strategy / "masks" / col / aug_fname
        if m.exists():
            masks[col] = m.relative_to(ROOT).as_posix()

    return {
        "filename":   aug_fname,
        "source":     source,
        "augmentation": strategy,
        "image_path": aug_img.relative_to(ROOT).as_posix(),
        "masks":      masks,
    }


def _validate_coverage(records: list[dict], strategy: str, log) -> None:
    missing = [r["filename"] for r in records
               if not (ROOT / r["image_path"]).exists()]
    if missing:
        log.error(
            f"{len(missing)} {strategy} image(s) not found on disk. "
            f"Run: python src/data/augment.py --strategies {strategy}. "
            f"First missing: {missing[0]}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_v2(orig_pct: float = 0.10, seed: int = 42) -> Path:
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
    aug_train = []
    for strategy in AUG_STRATEGIES:
        records = [_make_augmented_record(r, strategy) for r in v1_train]
        _validate_coverage(records, strategy, log)
        aug_train.extend(records)

    n_orig_train = max(1, round(len(v1_train) * orig_pct))
    orig_train   = rng.sample(v1_train, n_orig_train)
    train_samples = aug_train + orig_train
    rng.shuffle(train_samples)

    # ── Val ──────────────────────────────────────────────────────────────────
    aug_val = []
    for strategy in AUG_STRATEGIES:
        records = [_make_augmented_record(r, strategy) for r in v1_val]
        _validate_coverage(records, strategy, log)
        aug_val.extend(records)

    n_orig_val  = max(1, round(len(v1_val) * orig_pct))
    orig_val    = rng.sample(v1_val, n_orig_val)
    val_samples = aug_val + orig_val

    # ── Test ─────────────────────────────────────────────────────────────────
    aug_test = []
    for strategy in AUG_STRATEGIES:
        records = [_make_augmented_record(r, strategy) for r in v1_test]
        _validate_coverage(records, strategy, log)
        aug_test.extend(records)

    test_samples = list(v1_test) + aug_test   # original + greyscale + chroma15

    # ── Log sizes ────────────────────────────────────────────────────────────
    n_aug_per_strat_train = len(v1_train)
    n_aug_per_strat_val   = len(v1_val)
    log.info(
        f"v2: train={len(train_samples)} "
        f"({n_aug_per_strat_train} grey + {n_aug_per_strat_train} chroma15 + {n_orig_train} orig)  "
        f"val={len(val_samples)} "
        f"({n_aug_per_strat_val} grey + {n_aug_per_strat_val} chroma15 + {n_orig_val} orig)  "
        f"test={len(test_samples)} "
        f"({len(v1_test)} orig + {len(v1_test)} grey + {len(v1_test)} chroma15)"
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
            "local_dir":     "data/raw/pseudo-12k",
            "augmentation":  "original",
            "total_samples": len(v1_train) + len(v1_val),
        },
        **{
            f"pseudo-12k/{s}": {
                "local_dir":     f"data/augmented/pseudo-12k/{s}",
                "augmentation":  s,
                "total_samples": len(v1_train) + len(v1_val),
            }
            for s in AUG_STRATEGIES
        },
        "manual-1k": {
            "local_dir":     "data/raw/manual-1k",
            "augmentation":  "original",
            "total_samples": len(v1_test),
        },
        **{
            f"manual-1k/{s}": {
                "local_dir":     f"data/augmented/manual-1k/{s}",
                "augmentation":  s,
                "total_samples": len(v1_test),
            }
            for s in AUG_STRATEGIES
        },
    }

    # ── Build payload ────────────────────────────────────────────────────────
    payload = {
        "version_name": "v2",
        "description": (
            f"Colour-reduced fine-tuning split (greyscale + chroma15). "
            f"Train/val: greyscale + chroma15 of pseudo-12k (v1 indices) "
            f"+ {orig_pct*100:.0f}% original pseudo-12k "
            f"({n_orig_train} train / {n_orig_val} val). "
            f"Test: original + greyscale + chroma15 of manual-1k."
        ),
        "created_utc":   datetime.now(timezone.utc).isoformat(),
        "seed":          seed,
        "orig_pct":      orig_pct,
        "base_version":  "v1",
        "aug_strategies": AUG_STRATEGIES,
        "train_sources": [f"pseudo-12k/{s}" for s in AUG_STRATEGIES] + ["pseudo-12k"],
        "test_sources":  ["manual-1k"] + [f"manual-1k/{s}" for s in AUG_STRATEGIES],
        "splits":        splits,
        "statistics":    stats,
        "sources":       source_meta,
        "train":         train_samples,
        "val":           val_samples,
        "test":          test_samples,
    }

    # ── Write JSON ───────────────────────────────────────────────────────────
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    version_file = VERSIONS_DIR / "v2.json"
    if version_file.exists():
        log.warning("v2.json already exists — overwriting.")
    version_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info(f"Version file -> {version_file.relative_to(ROOT)}")

    _log_to_mlflow("v2", payload, version_file, log)

    return version_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create v2 dataset version (colour-reduced fine-tuning split).")
    parser.add_argument(
        "--orig-pct", type=float, default=0.10,
        help="Fraction of original images mixed into train/val (default: 0.10 = 10%%).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_v2(orig_pct=args.orig_pct, seed=args.seed)


if __name__ == "__main__":
    main()
