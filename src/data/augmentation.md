# Augmentation Strategies

All strategies are implemented in `augment.py` using [albumentations](https://albumentations.ai/).
Every spatial transform is applied jointly to the image and its mask to preserve annotation alignment.

---

## Available Strategies

### `geometric`
Spatial transforms that change the layout of the floor plan.

| Transform | Params | Purpose |
|---|---|---|
| HorizontalFlip | p=0.5 | Mirror floor plan left/right |
| VerticalFlip | p=0.5 | Mirror floor plan top/bottom |
| RandomRotate90 | p=0.5 | 90° rotations |
| ShiftScaleRotate | shift=±5%, scale=±10%, rot=±15°, p=0.5 | Mild perspective change |
| Affine (shear) | shear=±5°, p=0.3 | Slight shear distortion |

**When to use:** Always — floor plans can appear in any orientation. Safe to combine with everything.

---

### `photometric`
Pixel-level transforms applied to the image only (mask is unchanged).

| Transform | Params | Purpose |
|---|---|---|
| RandomBrightnessContrast | ±20%, p=0.5 | Simulate scan quality variation |
| HueSaturationValue | hue±10, sat±20, val±10, p=0.4 | Color variation across datasets |
| GaussNoise | std 1–5%, p=0.3 | Simulate digitization noise |
| GaussianBlur | kernel 3–5, p=0.2 | Simulate out-of-focus scans |
| Sharpen | alpha 0.1–0.3, p=0.2 | Counterpart to blur |

**When to use:** Useful when combining pseudo (generated) and manual (scanned) datasets — bridges the domain gap.

---

### `scale_crop`
Random crop and resize to force the model to handle partial floor plans.

| Transform | Params | Purpose |
|---|---|---|
| RandomResizedCrop | scale 50–100%, ratio 0.75–1.33, p=0.6 | Zoom into sub-regions |
| PadIfNeeded | 512×512 with white padding, p=1.0 | Maintain fixed input size |

**When to use:** Combine with `geometric` for best coverage. Helps with rooms near image borders.

---

### `elastic`
Non-rigid distortions that simulate imperfect hand-drawn floor plans.

| Transform | Params | Purpose |
|---|---|---|
| ElasticTransform | alpha=40, sigma=6, p=0.3 | Local deformation |
| GridDistortion | 5 steps, limit=0.15, p=0.3 | Grid-based warp |
| OpticalDistortion | distort=0.1, shift=0.05, p=0.2 | Lens-like distortion |

**When to use:** Particularly useful for `manual-floor-plan-1k` which contains hand-drawn scans. Use carefully — high alpha values can destroy thin wall annotations.

---

## Combining Strategies

Strategies are combined with `+`. The transforms from each strategy are concatenated into a single pipeline (order = left to right).

```bash
# Geometric only (baseline augmentation)
python src/data/augment.py --version <id> --strategy geometric

# Geometric + photometric (recommended starting point)
python src/data/augment.py --version <id> --strategy geometric+photometric

# Full combination
python src/data/augment.py --version <id> --strategy geometric+photometric+scale_crop

# With 2 copies per original image (doubles dataset size)
python src/data/augment.py --version <id> --strategy geometric+photometric --copies 2
```

---

## Version Tracking

Each augmented dataset gets its own version manifest at `data/versions/<version-id>.json`.
The version ID encodes: `<dataset>_aug_<strategy>_<manifest-hash>`.

The manifest records:
- Source version ID (full lineage)
- Strategy name applied
- Number of copies per sample
- Random seed (for reproducibility)
- Per-sample mapping: `{id, source_id, copy, image, mask}`

In MLflow, log `dataset_version` as a run parameter so each experiment is traceable to its exact data version.

---

## Recommended Experiment Plan

| Version | Strategy | Dataset size | Notes |
|---|---|---|---|
| v1 | original | ~13k | Baseline |
| v2 | geometric | ~13k | Basic spatial aug |
| v3 | geometric+photometric | ~13k | Domain gap reduction |
| v4 | geometric+photometric ×2 copies | ~26k | Double size |
| v5 | geometric+photometric+elastic | ~13k | Max diversity |

Run a full training cycle on v1 first to establish a baseline before adding augmentation.

---

## Adding a New Strategy

1. Add a new function `_my_strategy() -> A.Compose` in `augment.py`
2. Register it in the `STRATEGIES` dict
3. Document it in this file with the table format above
