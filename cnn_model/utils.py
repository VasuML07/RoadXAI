"""
Utility functions for the RoadXAI CNN model.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """
    Select the best available computation device.

    Priority:
        CUDA -> CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def move_model_to_device(
    model: nn.Module,
    device: torch.device | None = None,
) -> nn.Module:
    """Move a model to the selected device."""

    if device is None:
        device = get_device()

    return model.to(device)


def validate_input_tensor(
    images: torch.Tensor,
    expected_channels: int = 3,
) -> None:
    """
    Validate an image tensor before passing it to the CNN.

    Expected format:
        [batch, channels, height, width]
    """

    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor.")

    if images.ndim != 4:
        raise ValueError(
            "Expected a 4D tensor with shape "
            "[batch, channels, height, width]."
        )

    if images.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} input channels, "
            f"but received {images.shape[1]}."
        )

    if images.shape[2] <= 0 or images.shape[3] <= 0:
        raise ValueError("Image height and width must be greater than zero.")


def validate_mask_tensor(
    masks: torch.Tensor,
) -> None:
    """
    Validate a segmentation mask tensor.

    Expected format:
        [batch, 1, height, width]
    """

    if not isinstance(masks, torch.Tensor):
        raise TypeError("masks must be a torch.Tensor.")

    if masks.ndim != 4:
        raise ValueError(
            "Expected a 4D mask tensor with shape "
            "[batch, 1, height, width]."
        )

    if masks.shape[1] != 1:
        raise ValueError(
            f"Expected one mask channel, but received {masks.shape[1]}."
        )


def normalize_mask(
    masks: torch.Tensor,
) -> torch.Tensor:
    """
    Convert masks to floating-point binary values.

    Any positive value becomes 1 and zero remains 0.
    """

    validate_mask_tensor(masks)

    return (masks > 0).float()


def calculate_batch_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate mean Intersection over Union for a batch.

    Args:
        predictions:
            Model logits or probabilities.
        targets:
            Ground-truth binary masks.
        threshold:
            Threshold used to convert probabilities to binary masks.
        smooth:
            Numerical stability constant.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have identical shapes."
        )

    if predictions.ndim != 4:
        raise ValueError("Expected tensors with shape [B, C, H, W].")

    probabilities = (
        torch.sigmoid(predictions)
        if predictions.min() < 0 or predictions.max() > 1
        else predictions
    )

    predicted_masks = (probabilities >= threshold).float()
    target_masks = (targets > 0.5).float()

    predicted_masks = predicted_masks.flatten(start_dim=1)
    target_masks = target_masks.flatten(start_dim=1)

    intersection = (
        predicted_masks * target_masks
    ).sum(dim=1)

    union = (
        predicted_masks
        + target_masks
        - predicted_masks * target_masks
    ).sum(dim=1)

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean()


def calculate_batch_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> torch.Tensor:
    """Calculate mean Dice coefficient for a batch."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have identical shapes."
        )

    probabilities = (
        torch.sigmoid(predictions)
        if predictions.min() < 0 or predictions.max() > 1
        else predictions
    )

    predicted_masks = (probabilities >= threshold).float()
    target_masks = (targets > 0.5).float()

    predicted_masks = predicted_masks.flatten(start_dim=1)
    target_masks = target_masks.flatten(start_dim=1)

    intersection = (
        predicted_masks * target_masks
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        predicted_masks.sum(dim=1)
        + target_masks.sum(dim=1)
        + smooth
    )

    return dice.mean()


def get_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> float:
    """Return the current learning rate of an optimizer."""

    if not optimizer.param_groups:
        raise ValueError("Optimizer contains no parameter groups.")

    return float(optimizer.param_groups[0]["lr"])


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    path: str | Path,
    loss: float | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Path:
    """
    Save a model checkpoint.

    The checkpoint contains:
        - model state
        - optimizer state
        - scheduler state when available
        - epoch
        - loss
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Load a model checkpoint.

    Returns:
        The complete checkpoint dictionary.
    """

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if device is None:
        device = get_device()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint


def model_summary(model: nn.Module) -> dict[str, int]:
    """
    Return basic model parameter statistics.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    non_trainable_parameters = (
        total_parameters - trainable_parameters
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "non_trainable_parameters": non_trainable_parameters,
    }


__all__ = [
    "get_device",
    "move_model_to_device",
    "validate_input_tensor",
    "validate_mask_tensor",
    "normalize_mask",
    "calculate_batch_iou",
    "calculate_batch_dice",
    "get_learning_rate",
    "save_checkpoint",
    "load_checkpoint",
    "model_summary",
]