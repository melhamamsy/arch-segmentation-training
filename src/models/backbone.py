"""
DualHeadUNet — shared encoder, two independent U-Net decoders.

Architecture:
    Input [B, 3, H, W]
        |
    Shared Encoder (ResNet18 by default, ImageNet pre-trained)
        |
    ┌───┴───────────────┐
    Wall Decoder        Room Decoder
    (U-Net)             (U-Net)
        |                   |
    [B, 1, H, W]        [B, N, H, W]
    wall logits         room logits

The encoder runs exactly once per forward pass.  Both decoders receive the
same feature pyramid, so there is no redundant computation.

Decoupled parameter groups make it easy to apply a lower learning rate to
the pre-trained encoder and a higher rate to the decoder heads.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DualHeadUNet(nn.Module):
    """
    Args:
        encoder_name:     any smp-supported encoder, e.g. "resnet18", "resnet34"
        encoder_weights:  "imagenet" (default) or None
        num_room_classes: total classes including background (class 0)
        decoder_channels: U-Net decoder channel progression (coarse → fine)
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: str = "imagenet",
        num_room_classes: int = 10,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__()

        # ── Wall branch (also owns the shared encoder) ─────────────────────
        self._wall = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=decoder_channels,
        )

        # ── Room branch (encoder will be replaced with the shared one) ─────
        self._room = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,        # weights loaded only once
            in_channels=3,
            classes=num_room_classes,
            activation=None,
            decoder_channels=decoder_channels,
        )
        # Share encoder: room branch reuses wall branch encoder
        self._room.encoder = self._wall.encoder

        self.encoder_name    = encoder_name
        self.num_room_classes = num_room_classes

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] float32

        Returns:
            wall_logits: [B, 1, H, W]   (apply sigmoid for probabilities)
            room_logits: [B, N, H, W]   (apply softmax / argmax for classes)
        """
        features = self._wall.encoder(x)          # shared forward

        # smp >= 0.4: decoder takes a list, not *args
        wall_logits = self._wall.segmentation_head(self._wall.decoder(features))
        room_logits = self._room.segmentation_head(self._room.decoder(features))
        return wall_logits, room_logits

    # ── Parameter groups (for differential learning rates) ─────────────────

    def encoder_parameters(self):
        return self._wall.encoder.parameters()

    def decoder_parameters(self):
        """All decoder + head parameters from both branches."""
        return (
            list(self._wall.decoder.parameters())
            + list(self._wall.segmentation_head.parameters())
            + list(self._room.decoder.parameters())
            + list(self._room.segmentation_head.parameters())
        )

    def param_groups(self, encoder_lr: float, decoder_lr: float) -> list[dict]:
        """
        Returns optimizer param groups with separate learning rates.
        Typical usage: encoder_lr = base_lr * 0.1, decoder_lr = base_lr.
        """
        return [
            {"params": list(self.encoder_parameters()), "lr": encoder_lr},
            {"params": self.decoder_parameters(),       "lr": decoder_lr},
        ]

    # ── Utility ────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        total    = sum(p.numel() for p in self.parameters())
        encoder  = sum(p.numel() for p in self.encoder_parameters())
        decoders = sum(p.numel() for p in self.decoder_parameters())
        return {"total": total, "encoder": encoder, "decoders": decoders}

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"DualHeadUNet(\n"
            f"  encoder={self.encoder_name}, "
            f"  num_room_classes={self.num_room_classes}\n"
            f"  params: total={counts['total']:,}  "
            f"encoder={counts['encoder']:,}  "
            f"decoders={counts['decoders']:,}\n"
            f")"
        )


# ── Wall-only single-head model ────────────────────────────────────────────


class WallOnlyUNet(nn.Module):
    """
    Single-head U-Net for wall detection only.

    Can be initialised from scratch or loaded from a DualHeadUNet checkpoint
    (transfer learning — only the wall branch weights are used).
    """

    def __init__(
        self,
        encoder_name:     str = "resnet18",
        encoder_weights:  str | None = "imagenet",
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        self._wall = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=decoder_channels,
        )
        self.encoder_name = encoder_name

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns wall_logits [B, 1, H, W]."""
        features = self._wall.encoder(x)
        return self._wall.segmentation_head(self._wall.decoder(features))

    # ── Transfer learning ──────────────────────────────────────────────────

    @classmethod
    def from_dual_head_checkpoint(
        cls,
        checkpoint_path,
        encoder_name:     str = "resnet18",
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        device: str = "cpu",
    ) -> "WallOnlyUNet":
        """
        Load wall-branch weights from a DualHeadUNet (v1) checkpoint.

        v1 keys: _wall.encoder.*, _wall.decoder.*, _wall.segmentation_head.*, _room.*
        v2 keys: _wall.encoder.*, _wall.decoder.*, _wall.segmentation_head.*
        The _room.* keys are dropped silently.
        """
        model = cls(
            encoder_name=encoder_name,
            encoder_weights=None,   # weights come from checkpoint
            decoder_channels=decoder_channels,
        )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        v1_state = ckpt["model_state"]
        v2_state  = {k: v for k, v in v1_state.items() if k.startswith("_wall.")}
        model.load_state_dict(v2_state, strict=True)
        return model

    # ── Parameter groups ───────────────────────────────────────────────────

    def encoder_parameters(self):
        return self._wall.encoder.parameters()

    def decoder_parameters(self):
        return (
            list(self._wall.decoder.parameters())
            + list(self._wall.segmentation_head.parameters())
        )

    def param_groups(self, encoder_lr: float, decoder_lr: float) -> list[dict]:
        return [
            {"params": list(self.encoder_parameters()), "lr": encoder_lr},
            {"params": self.decoder_parameters(),       "lr": decoder_lr},
        ]

    # ── Utility ────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        total   = sum(p.numel() for p in self.parameters())
        encoder = sum(p.numel() for p in self.encoder_parameters())
        decoder = sum(p.numel() for p in self.decoder_parameters())
        return {"total": total, "encoder": encoder, "decoders": decoder}

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"WallOnlyUNet(\n"
            f"  encoder={self.encoder_name}\n"
            f"  params: total={counts['total']:,}  "
            f"encoder={counts['encoder']:,}  "
            f"decoder={counts['decoders']:,}\n"
            f")"
        )
