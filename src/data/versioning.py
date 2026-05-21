"""
Dataset version management for floor-plan model training.

A "version" defines the exact train / val / test split used for one round
of experimentation, including which raw directories contribute to each split
and what augmentation strategy (if any) was applied.

Every version is:
  - Stored as a human-readable JSON in data/versions/<version_name>.json
  - Logged to MLflow under the "dataset_versions" experiment so each
    training run can reference it by name and the full split is reproducible

Version JSON structure:
  version_name    human-readable ID, e.g. "v1"
  description     free-text explanation of what this version is
  created_utc     ISO-8601 timestamp
  seed            random seed used for the train/val shuffle
  splits          per-split pool_ratio + source composition (count + pct)
  statistics      per-split counts: total, by_source, by_augmentation
  sources         metadata about each contributing directory
  train / val / test  lists of sample records (see below)

Each sample record:
  {
    "filename":    "pseudo_00042.png",   <- prefixed, unique within its dir
    "source":      "pseudo-12k",         <- raw directory it came from
    "augmentation":"original",           <- "original" or strategy name
    "image_path":  "data/raw/pseudo-12k/images/pseudo_00042.png",
    "masks": {
      "walls":      "data/raw/pseudo-12k/masks/walls/pseudo_00042.png",
      "colors":     "data/raw/pseudo-12k/masks/colors/pseudo_00042.png",
      "footprints": "data/raw/pseudo-12k/masks/footprints/pseudo_00042.png"
    }
  }

Usage:
  # Create the baseline version (pseudo-12k -> train+val, manual-1k -> test)
  python src/data/versioning.py create \\
      --name v1 \\
      --description "Baseline: pseudo-12k 90/10 split. manual-1k held out as test." \\
      --train pseudo-12k \\
      --test  manual-1k

  # Include augmented data in training
  python src/data/versioning.py create \\
      --name v2 \\
      --description "pseudo-12k + geometric augmentation (2x train size)." \\
      --train pseudo-12k pseudo-12k_aug_geometric \\
      --test  manual-1k

  # List versions
  python src/data/versioning.py list

  # Show version details
  python src/data/versioning.py show v1
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
AUG_DIR = ROOT / "data" / "augmented"
VERSIONS_DIR = ROOT / "data" / "versions"
MASK_COLS = ["walls", "colors", "footprints"]

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'experiments' / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "dataset_versions"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    log = logging.getLogger("versioning")
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return log
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(stream=open(sys.stdout.fileno(), "w",
                                           encoding="utf-8", closefd=False))
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    return log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_prefix(images_dir: Path) -> str:
    for candidate in ("pseudo", "manual"):
        if any(images_dir.glob(f"{candidate}_*.png")):
            return candidate
    raise ValueError(f"Cannot detect prefix in {images_dir}")


def _get_augmentation(dir_name: str) -> str:
    """Extract augmentation label from directory name."""
    if "_aug_" in dir_name:
        # pseudo-12k_aug_geometric_photometric -> geometric+photometric
        raw_strategy = dir_name.split("_aug_", 1)[1]
        return raw_strategy.replace("_", "+")
    return "original"


def _collect_samples(source_name: str) -> list[dict]:
    """
    Collect all sample records for a source.

    source_name can be:
      "pseudo-12k"           -> data/raw/pseudo-12k/
      "pseudo-12k/greyscale" -> data/augmented/pseudo-12k/greyscale/
    """
    if "/" in source_name:
        base, strategy = source_name.split("/", 1)
        src_dir = AUG_DIR / base / strategy
        augmentation = strategy
        source_label = base
        glob_pattern = "*.png"
    else:
        src_dir = RAW_DIR / source_name
        augmentation = _get_augmentation(source_name)
        source_label = source_name
        prefix = _detect_prefix(src_dir / "images")
        glob_pattern = f"{prefix}_*.png"

    images_dir = src_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {images_dir}\n"
            f"Run download.py first, or run augment_greyscale.py to create '{source_name}'."
        )

    samples = []
    for img_path in sorted(images_dir.glob(glob_pattern)):
        fname = img_path.name
        masks = {}
        for col in MASK_COLS:
            m = src_dir / "masks" / col / fname
            if m.exists():
                masks[col] = m.relative_to(ROOT).as_posix()
        samples.append({
            "filename": fname,
            "source": source_label,
            "augmentation": augmentation,
            "image_path": img_path.relative_to(ROOT).as_posix(),
            "masks": masks,
        })

    return samples


def _compute_stats(samples: list[dict], all_sources: list[str]) -> dict:
    total = len(samples)
    original_count = sum(1 for s in samples if s["augmentation"] == "original")
    augmented_count = total - original_count

    # Always list every known source, even if its count is 0
    by_source = {src: 0 for src in all_sources}
    for s in samples:
        by_source[s["source"]] = by_source.get(s["source"], 0) + 1

    # Only list strategies that are actually augmentations (not "original")
    by_strategy = dict(Counter(
        s["augmentation"] for s in samples if s["augmentation"] != "original"
    ))

    return {
        "total": total,
        "original": original_count,
        "augmented": augmented_count,
        "by_source": by_source,
        "by_strategy": by_strategy,   # empty dict when no augmentation
    }


def _split_composition(samples: list[dict], grand_total: int) -> dict:
    """
    Flat breakdown of a split.
    Keys: 'source' for original data, 'source/strategy' for augmented.
    pool_ratio: this split's size as fraction of the grand total.
    """
    counts: Counter = Counter()
    for s in samples:
        key = s["source"] if s["augmentation"] == "original" else f"{s['source']}/{s['augmentation']}"
        counts[key] += 1
    result = dict(counts)
    result["total"] = len(samples)
    result["pool_ratio"] = round(len(samples) / grand_total, 4) if grand_total else 0.0
    return result


def _source_meta(source_name: str, n_samples: int) -> dict:
    if "/" in source_name:
        base, strategy = source_name.split("/", 1)
        src_dir = AUG_DIR / base / strategy
        augmentation = strategy
    else:
        src_dir = RAW_DIR / source_name
        augmentation = _get_augmentation(source_name)
    return {
        "local_dir": src_dir.relative_to(ROOT).as_posix(),
        "augmentation": augmentation,
        "total_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Create version
# ---------------------------------------------------------------------------

def create_version(
    version_name: str,
    description: str,
    train_sources: list[str],
    test_sources: list[str],
    train_ratio: float = 0.9,
    seed: int = 42,
    log: logging.Logger | None = None,
) -> Path:
    """
    Build a train/val/test split and write a version JSON + MLflow run.

    Args:
        version_name:    short ID, e.g. "v1". Used as filename.
        description:     human-readable explanation of this version.
        train_sources:   list of raw dir names for train+val pool.
        test_sources:    list of raw dir names used as test set (no shuffle).
        train_ratio:     fraction of train_sources assigned to train (rest -> val).
        seed:            shuffle seed for reproducibility.
    """
    if log is None:
        log = _setup_logging()

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    version_file = VERSIONS_DIR / f"{version_name}.json"
    if version_file.exists():
        log.warning(f"{version_file.name} already exists — overwriting.")

    # ── Collect samples ────────────────────────────────────────────────────
    log.info(f"Collecting train sources: {train_sources}")
    train_pool: list[dict] = []
    source_meta: dict[str, dict] = {}
    for src in train_sources:
        samples = _collect_samples(src)
        train_pool.extend(samples)
        source_meta[src] = _source_meta(src, len(samples))
        log.info(f"  {src}: {len(samples)} samples")

    log.info(f"Collecting test sources: {test_sources}")
    test_samples: list[dict] = []
    for src in test_sources:
        samples = _collect_samples(src)
        test_samples.extend(samples)
        source_meta[src] = _source_meta(src, len(samples))
        log.info(f"  {src}: {len(samples)} samples")

    # ── Split train -> train + val ─────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(train_pool)
    split_idx = int(len(train_pool) * train_ratio)
    train_samples = train_pool[:split_idx]
    val_samples = train_pool[split_idx:]

    # ── Statistics & split composition ────────────────────────────────────
    all_sources = list(dict.fromkeys(train_sources + test_sources))  # ordered, deduped
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

    # ── Build payload ──────────────────────────────────────────────────────
    payload = {
        "version_name": version_name,
        "description": description,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "splits": splits,
        "train_sources": train_sources,
        "test_sources": test_sources,
        "statistics": stats,
        "sources": source_meta,
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    version_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info(f"Version file -> {version_file.relative_to(ROOT)}")

    # ── MLflow logging ─────────────────────────────────────────────────────
    _log_to_mlflow(version_name, payload, version_file, log)

    _print_summary(version_name, stats, log)
    return version_file


def _log_to_mlflow(
    version_name: str,
    payload: dict,
    version_file: Path,
    log: logging.Logger,
) -> None:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        exp = mlflow.set_experiment(MLFLOW_EXPERIMENT)
        log.info(f"MLflow experiment: '{MLFLOW_EXPERIMENT}' (id={exp.experiment_id})")

        with mlflow.start_run(run_name=version_name):
            # Tags
            mlflow.set_tags({
                "version_name": version_name,
                "train_sources": ", ".join(payload["train_sources"]),
                "test_sources":  ", ".join(payload["test_sources"]),
            })

            # Params (configuration)
            mlflow.log_params({
                "seed":            payload["seed"],
                "train_ratio":     payload["splits"]["train"]["pool_ratio"],
                "train_sources":   ", ".join(payload["train_sources"]),
                "test_sources":    ", ".join(payload["test_sources"]),
                "description":     payload["description"][:250],
            })

            # Metrics (counts — comparable across versions in the MLflow UI)
            stats = payload["statistics"]
            mlflow.log_metrics({
                "train_total":   stats["train"]["total"],
                "val_total":     stats["val"]["total"],
                "test_total":    stats["test"]["total"],
                "grand_total":   stats["grand_total"],
            })

            # Per-source counts for train
            for src, count in stats["train"]["by_source"].items():
                safe = src.replace("-", "_").replace("+", "_")
                mlflow.log_metric(f"train_{safe}", count)

            # Augmentation breakdown for train
            mlflow.log_metric("train_original",  stats["train"]["original"])
            mlflow.log_metric("train_augmented", stats["train"]["augmented"])
            for strategy, count in stats["train"]["by_strategy"].items():
                safe = strategy.replace("-", "_").replace("+", "_")
                mlflow.log_metric(f"train_strategy_{safe}", count)

            # Artifact: the full version JSON
            mlflow.log_artifact(str(version_file), artifact_path="dataset_versions")

        log.info(f"Logged to MLflow: run '{version_name}' in experiment '{MLFLOW_EXPERIMENT}'")

    except Exception as exc:
        log.warning(f"MLflow logging failed (version JSON was still saved): {exc}")


def _print_summary(version_name: str, stats: dict, log: logging.Logger) -> None:
    log.info("-" * 60)
    log.info(f"Version '{version_name}' created:")
    for split in ("train", "val", "test"):
        s = stats[split]
        by_src = "  ".join(f"{k}={v}" for k, v in s["by_source"].items())
        aug_str = f"augmented={s['augmented']}"
        if s["by_strategy"]:
            aug_str += "  (" + "  ".join(f"{k}={v}" for k, v in s["by_strategy"].items()) + ")"
        log.info(f"  {split:5s}  total={s['total']:6d}  original={s['original']}  {aug_str}  | {by_src}")
    log.info(f"  grand_total={stats['grand_total']}")
    log.info("-" * 60)


# ---------------------------------------------------------------------------
# List / show
# ---------------------------------------------------------------------------

def list_versions(log: logging.Logger) -> None:
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(VERSIONS_DIR.glob("*.json"))
    if not files:
        log.info("No versions found in data/versions/")
        return
    log.info(f"{'NAME':<20} {'TRAIN':>7} {'VAL':>6} {'TEST':>6}  DESCRIPTION")
    log.info("-" * 72)
    for f in files:
        try:
            v = json.loads(f.read_text())
            s = v["statistics"]
            name = v["version_name"]
            desc = v["description"][:45]
            log.info(f"{name:<20} {s['train']['total']:>7} {s['val']['total']:>6} "
                     f"{s['test']['total']:>6}  {desc}")
        except Exception:
            log.info(f"{f.stem:<20}  (could not parse)")


def show_version(version_name: str, log: logging.Logger) -> None:
    version_file = VERSIONS_DIR / f"{version_name}.json"
    if not version_file.exists():
        log.error(f"Version '{version_name}' not found in {VERSIONS_DIR}")
        return
    v = json.loads(version_file.read_text())
    log.info(f"Version   : {v['version_name']}")
    log.info(f"Created   : {v['created_utc']}")
    log.info(f"Description: {v['description']}")
    log.info(f"Seed      : {v['seed']}")
    for sp in ("train", "val", "test"):
        comp = "  ".join(f"{src}={d['count']} ({d['pct']}%)"
                         for src, d in v["splits"][sp]["composition"].items())
        log.info(f"  {sp:5s}  pool_ratio={v['splits'][sp]['pool_ratio']}  | {comp}")
    log.info("Statistics:")
    for split in ("train", "val", "test"):
        s = v["statistics"][split]
        log.info(f"  {split}")
        log.info(f"    total          : {s['total']}")
        log.info(f"    by_source      : {s['by_source']}")
        log.info(f"    by_augmentation: {s['by_augmentation']}")
    log.info(f"Grand total: {v['statistics']['grand_total']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create and inspect dataset versions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a new version.")
    p_create.add_argument("--name", required=True, help="Version name, e.g. v1")
    p_create.add_argument("--description", required=True)
    p_create.add_argument("--train", nargs="+", required=True,
                          metavar="DIR",
                          help="Raw dir name(s) for train+val pool.")
    p_create.add_argument("--test", nargs="+", required=True,
                          metavar="DIR",
                          help="Raw dir name(s) for test set.")
    p_create.add_argument("--train-ratio", type=float, default=0.9)
    p_create.add_argument("--seed", type=int, default=42)

    # list
    sub.add_parser("list", help="List all versions.")

    # show
    p_show = sub.add_parser("show", help="Show details of a version.")
    p_show.add_argument("version_name")

    args = parser.parse_args()
    log = _setup_logging()

    if args.cmd == "create":
        create_version(
            version_name=args.name,
            description=args.description,
            train_sources=args.train,
            test_sources=args.test,
            train_ratio=args.train_ratio,
            seed=args.seed,
            log=log,
        )
    elif args.cmd == "list":
        list_versions(log)
    elif args.cmd == "show":
        show_version(args.version_name, log)


if __name__ == "__main__":
    main()
