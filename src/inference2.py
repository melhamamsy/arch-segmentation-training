"""
Wall-only inference script — runs WallOnlyUNet on a directory of floor plan images.

Loads all .png / .jpg images from  <input_dir>/images/
Writes binary wall masks to:        <input_dir>/masks/walls/<filename>.png
  white (255) = wall, black (0) = background

Preprocessing mirrors FloorPlanDataset:
  1. Optional greyscale enhancement (CLAHE + sharpen + brightness boost)
  2. Resize to image_size x image_size  (bilinear)
  3. ToTensor + ImageNet normalise

Output masks are upsampled to the original image resolution via nearest-
neighbour to preserve hard wall boundaries.

Usage:
  python src/inference2.py \\
      --input_dir  data/external/ \\
      --model_path checkpoints/unet_r18_v2/best.pt

  # with config auto-fill:
  python src/inference2.py \\
      --input_dir  data/external/ \\
      --model_path checkpoints/unet_r18_v2/best.pt \\
      --config     configs/unet_r18_v2.yaml

  # explicit overrides:
  python src/inference2.py \\
      --input_dir  data/external/ \\
      --model_path checkpoints/unet_r18_v2/best.pt \\
      --encoder    resnet18 \\
      --image_size 512 \\
      --threshold  0.4 \\
      --batch_size 8 \\
      --device     cuda
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.backbone import WallOnlyUNet
from src.utils.config import load_config

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Greyscale enhancement  (same pipeline as inference.py)
# ---------------------------------------------------------------------------

def enhance_colors(pil_image: Image.Image, save_path=None) -> Image.Image:
    return pil_image
    cv_img    = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    gray      = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalised = clahe.apply(blurred)
    kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalised, -1, kernel)
    enhanced  = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
    enhanced  = ImageEnhance.Brightness(enhanced).enhance(1.5)
    enhanced  = ImageEnhance.Contrast(enhanced).enhance(2.0)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        enhanced.save(save_path)
    return enhanced


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
# Device resolution
# ---------------------------------------------------------------------------

def _resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path:   str | Path,
    encoder_name: str,
    device:       torch.device,
) -> WallOnlyUNet:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    model = WallOnlyUNet(encoder_name=encoder_name, encoder_weights=None)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _postprocess_walls(
    logits:     torch.Tensor,            # [B, 1, H, W]
    threshold:  float,
    orig_sizes: list[tuple[int, int]],   # [(W, H), ...]
) -> list[Image.Image]:
    probs   = torch.sigmoid(logits.squeeze(1))   # [B, H, W]

    # --- diagnostic: how many pixels sit in the "decidable" mid-range? ---
    p = probs.detach().float().cpu()
    frac_lt_01 = (p < 0.1).float().mean().item()
    frac_gt_09 = (p > 0.9).float().mean().item()
    frac_mid   = ((p >= 0.1) & (p <= 0.9)).float().mean().item()
    print(f"[probs] min={p.min():.4f} max={p.max():.4f} mean={p.mean():.4f} "
          f"| <0.1: {frac_lt_01:.2%}  mid: {frac_mid:.2%}  >0.9: {frac_gt_09:.2%}")

    results = []
    for i, (W, H) in enumerate(orig_sizes):
        prob_np  = (probs[i].cpu().numpy() * 255).astype(np.uint8)
        pil_prob = Image.fromarray(prob_np, mode="L")
        resized  = np.array(pil_prob.resize((W, H), Image.NEAREST))
        binary   = (resized >= int(threshold * 255)).astype(np.uint8) * 255
        results.append(Image.fromarray(binary, mode="L"))
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_inference(
    input_dir:         Path,
    model_path:        Path,
    encoder_name:      str   = "resnet18",
    image_size:        int   = 512,
    threshold:         float = 0.3,
    batch_size:        int   = 4,
    device:            torch.device | None = None,
    enhance_greyscale: bool  = False,
) -> None:

    if device is None:
        device = _resolve_device(None)

    print(f"Device     : {device}")
    print(f"Model      : {model_path}")
    print(f"Input dir  : {input_dir}")

    model = load_model(model_path, encoder_name, device)
    print(f"Encoder    : {encoder_name}  |  threshold: {threshold}")

    images_dir = input_dir / "images"
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist.")
        sys.exit(1)

    image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No .png / .jpg images found in {images_dir}")
        return
    print(f"Images     : {len(image_paths)}")

    walls_out = input_dir / "masks" / "walls"
    walls_out.mkdir(parents=True, exist_ok=True)

    transform    = _build_transform(image_size)
    get_enh_path = lambda p: p.parent.with_name("images_enhanced") / p.name

    for batch_start in tqdm(range(0, len(image_paths), batch_size),
                            desc="Inferring", unit="batch"):
        batch_paths = image_paths[batch_start: batch_start + batch_size]
        pil_images  = [Image.open(p).convert("RGB") for p in batch_paths]
        orig_sizes  = [img.size for img in pil_images]

        inputs = (
            [enhance_colors(img, get_enh_path(p)) for p, img in zip(batch_paths, pil_images)]
            if enhance_greyscale else pil_images
        )
        tensors = torch.stack([transform(img) for img in inputs]).to(device)

        with torch.no_grad():
            wall_logits = model(tensors)

        wall_masks = _postprocess_walls(wall_logits, threshold, orig_sizes)

        for path, wall_img in zip(batch_paths, wall_masks):
            wall_img.save(walls_out / f"{path.stem}.png")

    print(f"\nSaved wall masks -> {walls_out}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_from_config(cfg_path: str | None) -> dict:
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
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run WallOnlyUNet (v2) inference on a directory of floor plan images.")

    parser.add_argument("--input_dir",          required=True,
                        help="Directory containing an images/ subfolder.")
    parser.add_argument("--model_path",         required=True,
                        help="Path to WallOnlyUNet checkpoint (.pt).")
    parser.add_argument("--config",             default=None,
                        help="Training YAML config to auto-fill encoder / image_size / threshold.")
    parser.add_argument("--encoder",            default=None,
                        help="Encoder name (default: from config or resnet18).")
    parser.add_argument("--image_size",         type=int,   default=None,
                        help="Inference resolution (default: from config or 512).")
    parser.add_argument("--threshold",          type=float, default=None,
                        help="Sigmoid threshold for wall prediction (default: 0.3).")
    parser.add_argument("--batch_size",         type=int,   default=4)
    parser.add_argument("--enhance_greyscale",
                        type=lambda x: x.lower() == "true",
                        default=False, metavar="true|false",
                        help="Apply greyscale enhancement before inference (default: false).")
    parser.add_argument("--device",             default=None,
                        help="cpu | cuda | xpu (default: auto-detect).")

    args = parser.parse_args()

    cfg_params   = _resolve_from_config(args.config)
    encoder_name = args.encoder    or cfg_params.get("encoder_name", "resnet18")
    image_size   = args.image_size or cfg_params.get("image_size",   512)
    threshold    = args.threshold  or cfg_params.get("threshold",    0.3)

    run_inference(
        input_dir         = Path(args.input_dir),
        model_path        = Path(args.model_path),
        encoder_name      = encoder_name,
        image_size        = image_size,
        threshold         = threshold,
        batch_size        = args.batch_size,
        device            = _resolve_device(args.device),
        enhance_greyscale = args.enhance_greyscale,
    )


if __name__ == "__main__":
    main()
