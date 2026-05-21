"""
Inference script — run DualHeadUNet on a directory of floor plan images.

Loads all .png images from  <input_dir>/images/
Writes predictions to:
  <input_dir>/masks/walls/<filename>.png    — binary wall mask (white = wall)
  <input_dir>/masks/colors/<filename>.png   — colorized room segmentation

Preprocessing exactly mirrors FloorPlanDataset:
  1. enhance_colors  (placeholder — currently identity)
  2. Resize to image_size × image_size  (bilinear)
  3. ToTensor + ImageNet normalise

Output masks are saved at the original image resolution (upsampled from
model output via nearest-neighbour to preserve label boundaries).

Usage:
  python src/inference.py \\
      --input_dir  data/external/ \\
      --model_path checkpoints/unet_r18_v1/best.pt

  # with explicit palette and config:
  python src/inference.py \\
      --input_dir  data/external/ \\
      --model_path checkpoints/unet_r18_v1/best.pt \\
      --palette    data/versions/v1_palette.json \\
      --config     configs/unet_r18_v1.yaml

  # override specific params:
  python src/inference.py \\
      --input_dir  data/external/ \\
      --model_path checkpoints/unet_r18_v1/best.pt \\
      --encoder    resnet34 \\
      --num_classes 12 \\
      --image_size 512 \\
      --threshold  0.4 \\
      --batch_size 8 \\
      --device     cuda
"""

import argparse
import cv2
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.backbone import DualHeadUNet
from src.data.dataset import ColorMapper
from src.utils.config import load_config

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Color enhancement  (placeholder)
# ---------------------------------------------------------------------------

def enhance_colors(pil_image: Image.Image, save_path=None) -> Image.Image:
    """
    Enhance the quality of a PIL image to improve OCR/OD detection.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Source image.
    save_path : str, optional
        If provided, the enhanced image is written to this path.

    Returns
    -------
    PIL.Image.Image
        The enhanced image.
    """
    # --- Convert PIL → OpenCV (RGB → BGR) ------------------------------------
    cv_img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    # 1) Grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 2) Denoise with Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) CLAHE contrast equalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalised = clahe.apply(blurred)

    # 4) Sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalised, -1, kernel)

    # --- Convert OpenCV → PIL (GRAY → RGB) -----------------------------------
    pil_enhanced = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))

    # 5) Brightness & contrast boost
    pil_enhanced = ImageEnhance.Brightness(pil_enhanced).enhance(1.5)
    pil_enhanced = ImageEnhance.Contrast(pil_enhanced).enhance(2.0)

    # Optional save
    if save_path:
        os.makedirs(
            os.path.dirname(save_path), exist_ok=True
        )
        pil_enhanced.save(save_path)

    return pil_enhanced



# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _infer_num_classes(state_dict: dict) -> int:
    """Read num_room_classes from the room segmentation head weight shape."""
    for key in sorted(state_dict.keys()):
        if "_room.segmentation_head" in key and key.endswith(".weight"):
            return state_dict[key].shape[0]
    raise ValueError(
        "Cannot infer num_room_classes from checkpoint. "
        "Pass --num_classes or --palette."
    )


def load_model(
    model_path:       str | Path,
    encoder_name:     str,
    num_room_classes: int,
    device:           torch.device,
) -> DualHeadUNet:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)

    model = DualHeadUNet(
        encoder_name=encoder_name,
        encoder_weights=None,          # weights come from checkpoint
        num_room_classes=num_room_classes,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Colorisation helpers
# ---------------------------------------------------------------------------

def _default_palette(num_classes: int) -> dict[int, tuple]:
    """
    Generate a visually distinct palette when no training palette is available.
    Uses evenly-spaced hues in HSV space.
    """
    import colorsys
    palette = {0: (255, 255, 255)}      # background = white
    for i in range(1, num_classes):
        hue = (i - 1) / max(num_classes - 1, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
        palette[i] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def _colorize_room_mask(class_map: np.ndarray, palette: dict[int, tuple]) -> Image.Image:
    """Convert HxW int class map to HxWx3 RGB image using the palette."""
    H, W = class_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, color in palette.items():
        rgb[class_map == idx] = color
    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _postprocess_walls(
    logits:     torch.Tensor,       # [B, 1, H, W]
    threshold:  float,
    orig_sizes: list[tuple[int, int]],   # [(W, H), ...]
) -> list[Image.Image]:
    probs = torch.sigmoid(logits.squeeze(1))    # [B, H, W]
    results = []
    for i, (W, H) in enumerate(orig_sizes):
        prob_img = (probs[i].cpu().numpy() * 255).astype(np.uint8)
        pil_prob = Image.fromarray(prob_img, mode="L")
        pil_resized = pil_prob.resize((W, H), Image.NEAREST)
        binary = (np.array(pil_resized) >= int(threshold * 255)).astype(np.uint8) * 255
        results.append(Image.fromarray(binary, mode="L"))
    return results


def _postprocess_rooms(
    logits:     torch.Tensor,       # [B, N, H, W]
    palette:    dict[int, tuple],
    orig_sizes: list[tuple[int, int]],   # [(W, H), ...]
) -> list[Image.Image]:
    class_maps = logits.argmax(dim=1).cpu().numpy()  # [B, H, W]
    results = []
    for i, (W, H) in enumerate(orig_sizes):
        class_pil = Image.fromarray(class_maps[i].astype(np.uint8), mode="L")
        class_resized = np.array(class_pil.resize((W, H), Image.NEAREST))
        results.append(_colorize_room_mask(class_resized, palette))
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_inference(
    input_dir:          Path,
    model_path:         Path,
    encoder_name:       str          = "resnet18",
    num_room_classes:   int | None   = None,
    image_size:         int          = 512,
    threshold:          float        = 0.5,
    batch_size:         int          = 4,
    device:             torch.device | None = None,
    palette_path:       Path | None  = None,
    enhance_greyscale:  bool         = False,
) -> None:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device     : {device}")
    print(f"Model      : {model_path}")
    print(f"Input dir  : {input_dir}")

    # ── Palette ───────────────────────────────────────────────────────────
    if palette_path and palette_path.exists():
        color_mapper = ColorMapper.load(palette_path)
        palette = {k: tuple(v) for k, v in color_mapper.idx_to_color.items()}
        if num_room_classes is None:
            num_room_classes = color_mapper.num_classes
        print(f"Palette    : {palette_path.name}  ({num_room_classes} classes)")
    else:
        if palette_path:
            print(f"Palette    : {palette_path} not found — using generated colours")
        else:
            print("Palette    : not provided — using generated colours")

    # ── Load checkpoint (need state_dict before we can infer num_classes) ──
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)

    if num_room_classes is None:
        num_room_classes = _infer_num_classes(state_dict)
        print(f"num_classes: {num_room_classes}  (inferred from checkpoint)")

    if not (palette_path and palette_path.exists()):
        palette = _default_palette(num_room_classes)

    # ── Model ──────────────────────────────────────────────────────────────
    model = DualHeadUNet(
        encoder_name=encoder_name,
        encoder_weights=None,
        num_room_classes=num_room_classes,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Encoder    : {encoder_name}  |  room classes: {num_room_classes}")

    # ── Discover input images ──────────────────────────────────────────────
    images_dir = input_dir / "images"
    get_enhp = lambda p: p.parent.with_name("images_enhanced") / p.name # Enhance path
    
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist.")
        sys.exit(1)

    image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No .png / .jpg images found in {images_dir}")
        return

    print(f"Images     : {len(image_paths)}")

    # ── Output directories ─────────────────────────────────────────────────
    walls_out  = input_dir / "masks" / "walls"
    colors_out = input_dir / "masks" / "colors"
    walls_out.mkdir(parents=True, exist_ok=True)
    colors_out.mkdir(parents=True, exist_ok=True)

    transform = _build_transform(image_size)

    # ── Batch inference ────────────────────────────────────────────────────
    for batch_start in tqdm(range(0, len(image_paths), batch_size),
                            desc="Inferring", unit="batch"):

        batch_paths = image_paths[batch_start: batch_start + batch_size]
        pil_images  = [Image.open(p).convert("RGB") for p in batch_paths]
        orig_sizes  = [img.size for img in pil_images]   # (W, H) per PIL convention

        enhanced   = [enhance_colors(img, get_enhp(p)) for p, img in zip(batch_paths, pil_images)] \
                     if enhance_greyscale else pil_images
        tensors    = torch.stack([transform(img) for img in enhanced]).to(device)

        with torch.no_grad():
            wall_logits, room_logits = model(tensors)

        wall_masks  = _postprocess_walls(wall_logits,  threshold,  orig_sizes)
        room_colors = _postprocess_rooms(room_logits,  palette,    orig_sizes)

        for path, wall_img, room_img in zip(batch_paths, wall_masks, room_colors):
            stem = path.stem
            wall_img.save(walls_out  / f"{stem}.png")
            room_img.save(colors_out / f"{stem}.png")

    print(f"\nSaved wall masks  -> {walls_out.relative_to(input_dir.parent)}")
    print(f"Saved room colors -> {colors_out.relative_to(input_dir.parent)}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _auto_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_from_config(cfg_path: str | None) -> dict:
    """Pull model / data params from a training YAML config if provided."""
    if not cfg_path:
        return {}
    cfg = load_config(cfg_path)
    out = {}
    try: out["encoder_name"] = cfg.model.encoder
    except AttributeError: pass
    try: out["image_size"]   = cfg.training.image_size
    except AttributeError: pass
    try: out["threshold"]    = cfg.evaluation.wall_pred_threshold
    except AttributeError: pass
    try: out["palette_dir"]  = cfg.paths.palette_dir
    except AttributeError: pass
    try: out["dataset_version"] = cfg.dataset_version
    except AttributeError: pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DualHeadUNet inference on a directory of floor plan images.")

    # Required
    parser.add_argument("--input_dir",   required=True,
                        help="Directory containing images/ subfolder.")
    parser.add_argument("--model_path",  required=True,
                        help="Path to .pt checkpoint.")

    # Optional — config auto-fills missing params
    parser.add_argument("--config",      default=None,
                        help="Training YAML config to read model params from.")
    parser.add_argument("--palette",     default=None,
                        help="Path to <version>_palette.json for room colours. "
                             "If omitted, a generated palette is used.")

    # Optional overrides
    parser.add_argument("--encoder",     default=None,
                        help="Encoder name, e.g. resnet18 (default: from config or resnet18).")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of room classes (default: inferred from checkpoint).")
    parser.add_argument("--image_size",  type=int, default=None,
                        help="Inference image size (default: from config or 512).")
    parser.add_argument("--threshold",   type=float, default=None,
                        help="Sigmoid threshold for wall prediction (default: 0.5).")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--enhance_greyscale", type=lambda x: x.lower() == "true",
                        default=False, metavar="true|false",
                        help="Apply greyscale enhancement before inference (default: false).")
    parser.add_argument("--device",      default=None,
                        help="cpu / cuda / mps (default: auto-detect).")

    args = parser.parse_args()

    # ── Resolve params: explicit args > config > hardcoded defaults ────────
    cfg_params = _resolve_from_config(args.config)

    encoder_name = args.encoder or cfg_params.get("encoder_name", "resnet18")
    image_size   = args.image_size or cfg_params.get("image_size", 512)
    threshold    = args.threshold  or cfg_params.get("threshold",  0.5)

    # Palette resolution: explicit --palette > infer from config paths
    palette_path = None
    if args.palette:
        palette_path = Path(args.palette)
    elif "palette_dir" in cfg_params and "dataset_version" in cfg_params:
        palette_path = ROOT / cfg_params["palette_dir"] / f"{cfg_params['dataset_version']}_palette.json"

    run_inference(
        input_dir          = Path(args.input_dir),
        model_path         = Path(args.model_path),
        encoder_name       = encoder_name,
        num_room_classes   = args.num_classes,
        image_size         = image_size,
        threshold          = threshold,
        batch_size         = args.batch_size,
        device             = _auto_device(args.device),
        palette_path       = palette_path,
        enhance_greyscale  = args.enhance_greyscale,
    )


if __name__ == "__main__":
    main()
