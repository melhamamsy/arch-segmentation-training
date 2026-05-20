"""
FloorPlanDataset  — PyTorch Dataset for wall detection + room segmentation.

Reads from a version JSON split (train / val / test list).
Each sample yields:
    image       FloatTensor [3, H, W]   ImageNet-normalised RGB floor plan
    wall_mask   LongTensor  [H, W]      0 = background, 1 = wall
    room_mask   LongTensor  [H, W]      0 = background, 1..N = room class

Wall binarisation assumption (validated in notebooks/explore.ipynb):
    pixels whose all three RGB channels >= wall_bg_threshold are background.
    Default threshold = 200.  If your dataset uses a dark background,
    pass wall_bg_threshold=55 and set wall_dark_bg=True.

Room colour mapping:
    ColorMapper scans a subset of training colour masks to discover the
    unique RGB values used per room type.  It assigns stable integer indices
    and saves the mapping to  data/versions/<version>_palette.json  so every
    run uses the exact same class numbering.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]

# ImageNet stats used by all torchvision/timm pre-trained encoders
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# ColorMapper
# ---------------------------------------------------------------------------

class ColorMapper:
    """
    Bidirectional map between RGB tuples (room colours in 'colors' masks)
    and integer class indices.

    Index 0 is always background (near-white pixels).
    Indices 1..N are room types, sorted by RGB value for reproducibility.
    """

    def __init__(self, color_to_idx: dict[tuple, int], idx_to_color: dict[int, tuple]):
        self.color_to_idx = color_to_idx
        self.idx_to_color = idx_to_color
        self.num_classes   = len(idx_to_color)

    # ── Build ──────────────────────────────────────────────────────────────

    @classmethod
    def build_from_samples(
        cls,
        samples: list[dict],
        num_scan: int = 200,
        bg_threshold: int = 240,
    ) -> "ColorMapper":
        """
        Scan a subset of colour masks to discover all unique room colours.

        Args:
            samples:       version split list (train recommended)
            num_scan:      how many masks to inspect (evenly spaced)
            bg_threshold:  pixels with all channels >= this are background
        """
        unique: set[tuple] = set()
        step = max(1, len(samples) // num_scan)

        for sample in samples[::step]:
            mask_path = (sample.get("masks") or {}).get("colors")
            if not mask_path:
                continue
            img = np.array(Image.open(ROOT / mask_path).convert("RGB"))
            for color in np.unique(img.reshape(-1, 3), axis=0):
                unique.add(tuple(int(c) for c in color))

        bg_colors   = [c for c in unique if all(v >= bg_threshold for v in c)]
        room_colors = sorted(c for c in unique if not all(v >= bg_threshold for v in c))

        color_to_idx: dict[tuple, int] = {}
        idx_to_color: dict[int, tuple] = {}

        # Assign index 0 to all background shades (map all → 0)
        for c in bg_colors:
            color_to_idx[c] = 0
        idx_to_color[0] = bg_colors[0] if bg_colors else (255, 255, 255)

        for i, c in enumerate(room_colors, start=1):
            color_to_idx[c] = i
            idx_to_color[i] = c

        return cls(color_to_idx, idx_to_color)

    # ── Conversion ─────────────────────────────────────────────────────────

    def rgb_to_index(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert HxWx3 uint8 RGB mask → HxW int64 class map.
        Unknown colours are mapped to 0 (background).
        """
        H, W = mask.shape[:2]
        # Pack RGB into a single int32 for fast comparison
        packed = (mask[:, :, 0].astype(np.int32) << 16
                  | mask[:, :, 1].astype(np.int32) << 8
                  | mask[:, :, 2].astype(np.int32))
        lut = {(r << 16 | g << 8 | b): idx
               for (r, g, b), idx in self.color_to_idx.items()}
        result = np.zeros((H, W), dtype=np.int64)
        for packed_color, idx in lut.items():
            result[packed == packed_color] = idx
        return result

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        payload = {
            "num_classes": self.num_classes,
            "color_to_idx": {str(k): v for k, v in self.color_to_idx.items()},
            "idx_to_color": {str(k): list(v) for k, v in self.idx_to_color.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "ColorMapper":
        data = json.loads(Path(path).read_text())
        color_to_idx = {eval(k): v for k, v in data["color_to_idx"].items()}
        idx_to_color = {int(k): tuple(v) for k, v in data["idx_to_color"].items()}
        return cls(color_to_idx, idx_to_color)

    def __repr__(self) -> str:
        return f"ColorMapper(num_classes={self.num_classes})"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FloorPlanDataset(Dataset):
    """
    Loads floor plan images and their dual segmentation targets from disk.

    Args:
        samples:            list of sample dicts from a version JSON split
        color_mapper:       ColorMapper instance (build once, share across splits)
        image_size:         spatial size to resize all images/masks to
        wall_bg_threshold:  RGB threshold above which a pixel is background
        wall_dark_bg:       set True if the wall mask has a dark background
        augment_transform:  optional albumentations Compose for on-the-fly aug
    """

    def __init__(
        self,
        samples: list[dict],
        color_mapper: ColorMapper,
        image_size: int = 512,
        wall_bg_threshold: int = 200,
        wall_dark_bg: bool = False,
        augment_transform=None,
    ):
        self.samples            = samples
        self.color_mapper       = color_mapper
        self.image_size         = image_size
        self.wall_bg_threshold  = wall_bg_threshold
        self.wall_dark_bg       = wall_dark_bg
        self.augment_transform  = augment_transform

        self._img_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        masks  = sample.get("masks") or {}

        # ── Floor plan image ───────────────────────────────────────────────
        img_pil = Image.open(ROOT / sample["image_path"]).convert("RGB")

        # ── Wall mask ──────────────────────────────────────────────────────
        wall_path = masks.get("walls")
        if wall_path:
            wall_rgb = np.array(
                Image.open(ROOT / wall_path).convert("RGB")
                     .resize((self.image_size, self.image_size), Image.NEAREST)
            )
            is_bg = np.all(wall_rgb >= self.wall_bg_threshold, axis=2)
            wall_mask = (~is_bg if not self.wall_dark_bg
                         else np.all(wall_rgb <= (255 - self.wall_bg_threshold), axis=2)
                         ).astype(np.int64)
        else:
            wall_mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)

        # ── Room colour mask ───────────────────────────────────────────────
        color_path = masks.get("colors")
        if color_path:
            color_rgb = np.array(
                Image.open(ROOT / color_path).convert("RGB")
                     .resize((self.image_size, self.image_size), Image.NEAREST)
            )
            room_mask = self.color_mapper.rgb_to_index(color_rgb)
        else:
            room_mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)

        # ── Optional on-the-fly augmentation ──────────────────────────────
        if self.augment_transform is not None:
            img_np = np.array(img_pil.resize((self.image_size, self.image_size)))
            out = self.augment_transform(
                image=img_np,
                masks=[wall_mask.astype(np.uint8), room_mask.astype(np.uint8)],
            )
            img_pil   = Image.fromarray(out["image"])
            wall_mask = out["masks"][0].astype(np.int64)
            room_mask = out["masks"][1].astype(np.int64)

        return {
            "image":     self._img_tf(img_pil),
            "wall_mask": torch.from_numpy(wall_mask),
            "room_mask": torch.from_numpy(room_mask),
            "filename":  sample["filename"],
        }
