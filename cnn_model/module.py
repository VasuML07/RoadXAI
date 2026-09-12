"""
RoadXAI CNN Model.

Contains the convolutional neural network used for road-defect
segmentation, along with the loss function, optimizer and scheduler
builders required by the training pipeline.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ConvBlock(nn.Module):
    """
    Standard convolutional block.

    Structure:
        Conv2d -> BatchNorm -> ReLU -> Dropout
        Conv2d -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder block consisting of convolutional processing and pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)

        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution and skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = ConvBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        # Handle possible spatial-size differences caused by input dimensions.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat((skip, x), dim=1)

        return self.conv(x)


class RoadXAIUNet(nn.Module):
    """
    U-Net style CNN for binary road-defect segmentation.

    Input:
        [batch, 3, height, width]

    Output:
        [batch, 1, height, width]

    The model returns logits. Apply sigmoid only when converting
    logits into probabilities for inference or visualization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.encoder1 = EncoderBlock(
            in_channels,
            base_channels,
            dropout=dropout,
        )

        self.encoder2 = EncoderBlock(
            base_channels,
            base_channels * 2,
            dropout=dropout,
        )

        self.encoder3 = EncoderBlock(
            base_channels * 2,
            base_channels * 4,
            dropout=dropout,
        )

        self.encoder4 = EncoderBlock(
            base_channels * 4,
            base_channels * 8,
            dropout=dropout,
        )

        self.bottleneck = ConvBlock(
            base_channels * 8,
            base_channels * 16,
            dropout=dropout,
        )

        self.decoder4 = DecoderBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
            dropout=dropout,
        )

        self.decoder3 = DecoderBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            dropout=dropout,
        )

        self.decoder2 = DecoderBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            dropout=dropout,
        )

        self.decoder1 = DecoderBlock(
            base_channels * 2,
            base_channels,
            base_channels,
            dropout=dropout,
        )

        self.output_layer = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return segmentation logits."""

        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        return self.output_layer(x)


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.contiguous().view(
            probabilities.shape[0],
            -1,
        )
        targets = targets.float().contiguous().view(
            targets.shape[0],
            -1,
        )

        intersection = (probabilities * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice loss.

    This provides pixel-wise BCE supervision while also handling
    segmentation overlap through Dice loss.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative.")

        if bce_weight + dice_weight == 0:
            raise ValueError("At least one loss weight must be greater than zero.")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )


def build_model(
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
    dropout: float = 0.10,
) -> RoadXAIUNet:
    """Create and return the RoadXAI segmentation model."""

    return RoadXAIUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        dropout=dropout,
    )


def build_loss(
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> BCEDiceLoss:
    """Create the combined BCE + Dice segmentation loss."""

    return BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
    )


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> AdamW:
    """Create the AdamW optimizer."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")

    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    return AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 1e-6,
) -> ReduceLROnPlateau:
    """Create a validation-loss based learning-rate scheduler."""

    if not 0 < factor < 1:
        raise ValueError("factor must be between 0 and 1.")

    if patience < 0:
        raise ValueError("patience cannot be negative.")

    if min_lr < 0:
        raise ValueError("min_lr cannot be negative.")

    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def initialize_weights(model: nn.Module) -> None:
    """
    Initialize convolutional and normalization layers.

    Uses Kaiming initialization for convolutional layers and
    standard initialization for BatchNorm layers.
    """

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def predict_mask(
    model: nn.Module,
    images: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Generate binary segmentation masks.

    Returns:
        Tensor with shape [batch, 1, height, width] and values {0, 1}.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    was_training = model.training
    model.eval()

    with torch.no_grad():
        logits = model(images)
        probabilities = torch.sigmoid(logits)
        masks = (probabilities >= threshold).float()

    if was_training:
        model.train()

    return masks


__all__ = [
    "ConvBlock",
    "EncoderBlock",
    "DecoderBlock",
    "RoadXAIUNet",
    "DiceLoss",
    "BCEDiceLoss",
    "build_model",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "count_parameters",
    "initialize_weights",
    "predict_mask",
]