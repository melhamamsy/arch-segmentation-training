"""
Loss functions for dual-head wall detection + room segmentation.

Wall head: class imbalance (walls cover ~5-10% of pixels) makes vanilla BCE
underfit.  We combine BCE + Dice, which pushes the model to optimise overlap
rather than raw pixel accuracy.

Room head: standard cross-entropy over N room classes.

MultiTaskLoss: weighted sum  L = alpha * L_wall + beta * L_room.
Both sub-losses are returned individually for MLflow logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Works on raw logits (applies sigmoid internally).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, 1, H, W] raw logits
            targets: [B, H, W]    int64  (0/1)
        """
        probs   = torch.sigmoid(logits).squeeze(1)          # [B, H, W]
        targets = targets.float()
        probs   = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Wall loss  (BCE + Dice)
# ---------------------------------------------------------------------------

class WallLoss(nn.Module):
    """
    Combined BCE + Dice for binary wall segmentation.

    Args:
        bce_weight:  contribution of binary cross-entropy  (default 0.5)
        dice_weight: contribution of Dice loss             (default 0.5)
        pos_weight:  optional scalar to up-weight wall pixels (helps if
                     walls are very sparse; set to ~10 for 5% wall density)
    """

    def __init__(
        self,
        bce_weight:  float = 0.5,
        dice_weight: float = 0.5,
        pos_weight:  float | None = None,
    ):
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, 1, H, W]
            targets: [B, H, W]   int64
        """
        targets_f = targets.float().unsqueeze(1)
        return self.bce_w  * self.bce(logits, targets_f) \
             + self.dice_w * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Room loss  (cross-entropy)
# ---------------------------------------------------------------------------

class RoomLoss(nn.Module):
    """
    Cross-entropy loss for multi-class room segmentation.

    Args:
        ignore_index: class index to ignore in loss (default -1 = none)
        label_smoothing: optional label smoothing  (default 0.0)
    """

    def __init__(self, ignore_index: int = -1, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, N, H, W]
            targets: [B, H, W]   int64
        """
        return self.ce(logits, targets)


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    L_total = alpha * L_wall + beta * L_room

    Returns the total loss and both sub-losses for logging.
    """

    def __init__(
        self,
        wall_weight:     float = 1.0,
        room_weight:     float = 1.0,
        bce_weight:      float = 0.5,
        dice_weight:     float = 0.5,
        wall_pos_weight: float | None = None,
    ):
        super().__init__()
        self.wall_loss = WallLoss(bce_weight, dice_weight, wall_pos_weight)
        self.room_loss = RoomLoss()
        self.alpha = wall_weight
        self.beta  = room_weight

    def forward(
        self,
        wall_logits: torch.Tensor,
        room_logits: torch.Tensor,
        wall_targets: torch.Tensor,
        room_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, wall_loss, room_loss
        """
        l_wall = self.wall_loss(wall_logits, wall_targets)
        l_room = self.room_loss(room_logits, room_targets)
        total  = self.alpha * l_wall + self.beta * l_room
        return total, l_wall, l_room


# ---------------------------------------------------------------------------
# LwF loss  (Learning without Forgetting — KL divergence for binary case)
# ---------------------------------------------------------------------------

class LwFLoss(nn.Module):
    """
    Soft-label distillation loss for binary wall segmentation.

    Compares student logits against a frozen teacher's predictions using
    temperature-scaled binary cross-entropy, which is equivalent to the
    KL divergence between two Bernoulli distributions.

    Args:
        temperature: T > 1 softens the teacher distribution, making
                     the student less rigidly constrained (default 2.0).
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [B, 1, H, W]  — new model (gradients flow here)
            teacher_logits: [B, 1, H, W]  — frozen old model (detached)
        """
        soft_labels = torch.sigmoid(teacher_logits.detach() / self.T)
        return F.binary_cross_entropy_with_logits(student_logits / self.T, soft_labels)
