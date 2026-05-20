"""
Download zimhe floor-plan datasets from HuggingFace and save to disk.

Dataset schema (both datasets, identical columns):
  indices    str        8-char unique ID
  plans      Image      512x512 floor plan (RGBA for pseudo, RGB for manual)
  walls      Image      512x512 RGB wall mask          <- wall detection target
  colors     Image      512x512 RGB room color map     <- segmentation target
  footprints Image      512x512 RGB footprint mask
  captions   str        natural-language description

Output layout (files are prefixed by dataset: pseudo_ or manual_):
  data/raw/<name>/images/<prefix>_<id>.png
  data/raw/<name>/masks/walls/<prefix>_<id>.png
  data/raw/<name>/masks/colors/<prefix>_<id>.png
  data/raw/<name>/masks/footprints/<prefix>_<id>.png
  data/raw/<name>/metadata.json

Usage:
  python src/data/download.py                        # download both
  python src/data/download.py --dataset pseudo-12k   # single dataset
  python src/data/download.py --force                # re-download even if complete
  python src/data/download.py --rename               # rename existing files to add prefix
  python src/data/download.py --dry-run              # inspect columns only
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DATASETS: dict[str, dict] = {
    "pseudo-12k": {
        "hf_path": "zimhe/pseudo-floor-plan-12k",
        "split": "train",
        "prefix": "pseudo",
        "expected_samples": 12000,
    },
    "manual-1k": {
        "hf_path": "zimhe/manual-floor-plan-1k",
        "split": "train",
        "prefix": "manual",
        "expected_samples": 1007,
    },
}

MASK_COLS = ["walls", "colors", "footprints"]


def _setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("download")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(stream=open(sys.stdout.fileno(), "w",
                                           encoding="utf-8", closefd=False))
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _to_rgb(img) -> Image.Image:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    return img.convert("RGB")


def _count_saved(out_dir: Path, prefix: str) -> int:
    images_dir = out_dir / "images"
    if not images_dir.exists():
        return 0
    return sum(1 for _ in images_dir.glob(f"{prefix}_*.png"))


def rename_existing(name: str, cfg: dict, log: logging.Logger) -> None:
    """Add prefix to any files that are still named without it (idempotent)."""
    prefix = cfg["prefix"]
    out_dir = RAW_DIR / name
    dirs_to_rename = [out_dir / "images"] + [out_dir / "masks" / col for col in MASK_COLS]
    total = 0
    for d in dirs_to_rename:
        if not d.exists():
            continue
        files = [f for f in d.glob("*.png") if not f.name.startswith(f"{prefix}_")]
        for f in files:
            f.rename(f.parent / f"{prefix}_{f.name}")
            total += 1
    log.info(f"Renamed {total} files in {out_dir.relative_to(ROOT)}")


def save_dataset(
    name: str,
    cfg: dict,
    log: logging.Logger,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict]:
    hf_path = cfg["hf_path"]
    split = cfg["split"]
    prefix = cfg["prefix"]
    expected = cfg.get("expected_samples")
    out_dir = RAW_DIR / name

    log.info("=" * 60)
    log.info(f"Dataset  : {hf_path}  (split={split})")
    log.info(f"Output   : {out_dir.relative_to(ROOT)}")

    if not force and not dry_run:
        saved = _count_saved(out_dir, prefix)
        if expected and saved >= expected:
            log.info(f"Already complete ({saved}/{expected} images). Skipping.")
            log.info("  Re-run with --force to overwrite.")
            return _rebuild_manifest(out_dir, prefix, name)
        elif saved > 0:
            log.warning(f"Partial save detected ({saved}/{expected}). Use --force to restart.")

    log.info("Loading from HuggingFace ...")
    try:
        ds = load_dataset(hf_path, split=split)
    except Exception:
        log.error(f"Failed to load '{hf_path}'")
        log.debug(traceback.format_exc())
        raise

    log.info(f"Samples  : {len(ds)}")
    log.info(f"Features : {list(ds.features.keys())}")

    if dry_run:
        sample = ds[0]
        log.info("First sample:")
        for k, v in sample.items():
            if hasattr(v, "size"):
                log.info(f"  {k}: {type(v).__name__}  size={v.size}  mode={v.mode}")
            else:
                log.info(f"  {k}: {type(v).__name__}  value={str(v)[:80]}")
        log.info("[dry-run] Not saving.")
        return []

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for col in MASK_COLS:
        (out_dir / "masks" / col).mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    metadata: list[dict] = []
    errors: list[dict] = []

    for i, sample in enumerate(tqdm(ds, desc=f"Saving {name}", unit="img")):
        fname = f"{prefix}_{i:05d}.png"
        try:
            img_path = out_dir / "images" / fname
            _to_rgb(sample["plans"]).save(img_path)

            mask_paths: dict[str, str] = {}
            for col in MASK_COLS:
                m_path = out_dir / "masks" / col / fname
                _to_rgb(sample[col]).save(m_path)
                mask_paths[col] = m_path.relative_to(ROOT).as_posix()

            manifest.append({
                "filename": fname,
                "source": name,
                "original_index": sample["indices"],
                "image_path": img_path.relative_to(ROOT).as_posix(),
                "masks": mask_paths,
            })
            metadata.append({
                "filename": fname,
                "original_index": sample["indices"],
                "caption": sample["captions"],
            })
        except Exception as exc:
            log.warning(f"Sample {i} failed: {exc}")
            log.debug(traceback.format_exc())
            errors.append({"id": i, "error": str(exc)})

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    if errors:
        err_file = out_dir / "download_errors.json"
        err_file.write_text(json.dumps(errors, indent=2))
        log.warning(f"{len(errors)} sample(s) failed -> {err_file.relative_to(ROOT)}")

    log.info(f"Saved {len(manifest)}/{len(ds)} samples -> {out_dir.relative_to(ROOT)}")
    return manifest


def _rebuild_manifest(out_dir: Path, prefix: str, source_name: str) -> list[dict]:
    """Reconstruct a manifest from files already on disk."""
    images_dir = out_dir / "images"
    manifest = []
    for img_path in sorted(images_dir.glob(f"{prefix}_*.png")):
        fname = img_path.name
        masks = {col: (out_dir / "masks" / col / fname).relative_to(ROOT).as_posix()
                 for col in MASK_COLS}
        manifest.append({
            "filename": fname,
            "source": source_name,
            "image_path": img_path.relative_to(ROOT).as_posix(),
            "masks": masks,
        })
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download floor-plan datasets.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if images already exist.")
    parser.add_argument("--rename", action="store_true",
                        help="Rename existing files to add prefix (idempotent).")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()))
    args = parser.parse_args()

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    ok = True

    for name, cfg in targets.items():
        log_file = RAW_DIR / name / "download.log"
        log = _setup_logging(log_file)
        log.info(f"Job: {name}")
        try:
            if args.rename:
                rename_existing(name, cfg, log)
            else:
                save_dataset(name, cfg, log, dry_run=args.dry_run, force=args.force)
            log.info(f"Done: {name}")
        except Exception as exc:
            log.error(f"'{name}' failed: {exc}")
            log.debug(traceback.format_exc())
            ok = False
        finally:
            for h in log.handlers[:]:
                h.close()
                log.removeHandler(h)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
