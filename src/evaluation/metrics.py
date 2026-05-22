"""
Segmentation metrics for wall detection and room segmentation.

All functions accept raw logits and integer targets (on any device).
They return plain Python floats so they can be logged directly to MLflow.

Wall metrics  (binary segmentation):
    wall_iou          — Intersection over Union for the wall class
    wall_f1           — F1 / Dice score for the wall class
    wall_pixel_acc    — pixel-level accuracy (wall vs background)

Room metrics  (multi-class segmentation):
    room_miou         — mean IoU across all N classes (macro average)
    room_pixel_acc    — pixel-level accuracy across all classes
    room_per_class_iou — per-class IoU dict  {class_idx: iou}
"""

import torch


_EPS = 1e-6


# ---------------------------------------------------------------------------
# Wall metrics (binary)
# ---------------------------------------------------------------------------

def wall_iou(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Binary IoU for wall pixels.

    Args:
        logits:    [B, 1, H, W]  raw logits
        targets:   [B, H, W]    int64  (0 or 1)
        threshold: sigmoid threshold for wall prediction
    Returns:
        float  IoU averaged over the batch
    """
    with torch.no_grad():
        preds   = (torch.sigmoid(logits.squeeze(1)) >= threshold)
        targets = targets.bool()
        inter   = (preds & targets).float().sum(dim=(1, 2))
        union   = (preds | targets).float().sum(dim=(1, 2))
        return ((inter + _EPS) / (union + _EPS)).mean().item()


def wall_f1(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Dice / F1 score for wall pixels."""
    with torch.no_grad():
        preds   = (torch.sigmoid(logits.squeeze(1)) >= threshold).float()
        targets = targets.float()
        tp    = (preds * targets).sum(dim=(1, 2))
        denom = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        return ((2.0 * tp + _EPS) / (denom + _EPS)).mean().item()


def wall_pixel_acc(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Pixel-level accuracy for binary wall mask."""
    with torch.no_grad():
        preds = (torch.sigmoid(logits.squeeze(1)) >= threshold).long()
        return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# Room metrics (multi-class)
# ---------------------------------------------------------------------------

def room_miou(
    logits:       torch.Tensor,
    targets:      torch.Tensor,
    num_classes:  int,
    ignore_index: int = -1,
) -> float:
    """
    Mean IoU across all room classes (macro average).
    Classes absent from both predictions and targets are excluded from the mean.

    Args:
        logits:      [B, N, H, W]
        targets:     [B, H, W]    int64
        num_classes: N
        ignore_index: class index to skip
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        ious  = []
        for c in range(num_classes):
            if c == ignore_index:
                continue
            pred_c = (preds   == c)
            true_c = (targets == c)
            inter  = (pred_c & true_c).float().sum()
            union  = (pred_c | true_c).float().sum()
            if union > 0:
                ious.append(((inter + _EPS) / (union + _EPS)).item())
        return sum(ious) / len(ious) if ious else 0.0


def room_pixel_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Pixel-level accuracy across all room classes."""
    with torch.no_grad():
        return (logits.argmax(dim=1) == targets).float().mean().item()


def room_per_class_iou(
    logits:      torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int,
) -> dict[int, float]:
    """Per-class IoU as {class_idx: iou}. Absent classes are omitted."""
    with torch.no_grad():
        preds  = logits.argmax(dim=1)
        result = {}
        for c in range(num_classes):
            pred_c = (preds   == c)
            true_c = (targets == c)
            inter  = (pred_c & true_c).float().sum()
            union  = (pred_c | true_c).float().sum()
            if union > 0:
                result[c] = ((inter + _EPS) / (union + _EPS)).item()
        return result


# ---------------------------------------------------------------------------
# Convenience: all metrics in one call
# ---------------------------------------------------------------------------

def compute_wall(
    wall_logits:  torch.Tensor,
    wall_targets: torch.Tensor,
    threshold:    float = 0.5,
) -> dict[str, float]:
    """Wall-only metrics dict (no room head required)."""
    return {
        "wall_iou":       wall_iou(wall_logits, wall_targets, threshold),
        "wall_f1":        wall_f1(wall_logits, wall_targets, threshold),
        "wall_pixel_acc": wall_pixel_acc(wall_logits, wall_targets, threshold),
    }


def compute_all(
    wall_logits:      torch.Tensor,
    room_logits:      torch.Tensor,
    wall_targets:     torch.Tensor,
    room_targets:     torch.Tensor,
    num_room_classes: int,
) -> dict[str, float]:
    """Flat dict of all metrics, ready to log to MLflow."""
    return {
        "wall_iou":       wall_iou(wall_logits, wall_targets),
        "wall_f1":        wall_f1(wall_logits, wall_targets),
        "wall_pixel_acc": wall_pixel_acc(wall_logits, wall_targets),
        "room_miou":      room_miou(room_logits, room_targets, num_room_classes),
        "room_pixel_acc": room_pixel_acc(room_logits, room_targets),
    }
