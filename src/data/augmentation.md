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

### `chroma<T>`
Bleach away room-colour fills: turn **coloured** pixels white while keeping
near-grey structure (walls, furniture, text). Masks unchanged.

`T` is the chroma threshold and is read from the strategy name at runtime —
pass `chroma15`, `chroma25`, or any integer in `0..255`.

| Step | Detail |
|---|---|
| Compute chroma | `max(R,G,B) − min(R,G,B)` per pixel |
| Keep greyscale | pixels with chroma ≤ T → averaged grey (residual tint removed) |
| Bleach colour | pixels with chroma > T → white (255,255,255) |

Higher `T` keeps more (faintly-tinted) pixels as grey; lower `T` bleaches more
aggressively. Useful for bridging the gap between coloured (pseudo) and
greyscale/scanned (manual) plans by collapsing solid room fills.

**When to use:** When room-colour fills are noise for the task and only the
grey line-work (walls, text) matters.

---

### `howallow<N>`
Turn solid walls into **hollow** walls (outline only, white inside) to match
inference-time floor plans that draw walls as borders rather than filled blocks.

`N` is the rim width in pixels and is read from the strategy name at runtime —
pass `howallow2`, `howallow5`, `howallow12`, or any positive integer. No fixed
registration: the border is parsed from `howallow<N>`.

| Step | Detail |
|---|---|
| Read walls mask | `masks/walls` (white walls on black) defines the wall region |
| Erode by N px | Square erosion = sweep inward from every wall edge, keep N px per side |
| Blank interior | Eroded interior pixels → white (255,255,255) in the image |

Sweeping from all sides and keeping a fixed N px margin from the nearest edge is
exactly a square (L∞) erosion: a wall pixel is *interior* iff it sits more than
N px from the nearest background pixel along any axis. Internal walls, junctions
and closed room outlines are handled automatically (both sides of every wall are
background in the mask). Walls thinner than 2·N px survive intact — no interior to
blank — so thin walls and the N px rims stay solid. **Masks are unchanged**, so
wall annotations still describe the full (solid) wall footprint.

**Choosing N:** match it to wall thickness. `pseudo-12k` / `manual-1k` walls are
~8–10px (use `howallow2`); `cubicasa5k` walls are ~14–43px (use `howallow5` or
larger). If `N` ≥ half the wall thickness the strategy is a no-op (nothing to
hollow).

**When to use:** When training data has solid walls but inference plans have
hollow/outlined walls. Place after any spatial strategy (e.g. `geometric howallow5`)
so the mask used for hollowing stays aligned with the transformed image.

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
