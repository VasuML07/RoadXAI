"""
Utility functions for the RoadXAI training pipeline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior improves reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Return total, trainable and non-trainable parameter counts.
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def get_current_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> float:
    """Return the learning rate of the first parameter group."""

    if not optimizer.param_groups:
        raise ValueError(
            "Optimizer contains no parameter groups."
        )

    return float(
        optimizer.param_groups[0]["lr"]
    )


def calculate_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> float:
    """
    Calculate mean Dice score.

    Predictions may be logits or probabilities.
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have the same shape."
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = torch.sigmoid(predictions)

    predicted_masks = (
        probabilities >= threshold
    ).float()

    target_masks = (
        targets >= 0.5
    ).float()

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

    return float(dice.mean().item())


def calculate_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> float:
    """
    Calculate mean Intersection over Union.

    Predictions may be logits or probabilities.
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have the same shape."
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = torch.sigmoid(predictions)

    predicted_masks = (
        probabilities >= threshold
    ).float()

    target_masks = (
        targets >= 0.5
    ).float()

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

    return float(iou.mean().item())


def save_json(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    """Save a dictionary as formatted JSON."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected the JSON root to contain an object."
        )

    return data


def save_history(
    history: Any,
    path: str | Path,
) -> Path:
    """
    Save TrainingHistory as JSON.

    Supports dataclass-style objects containing lists
    and scalar training statistics.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(history, "__dataclass_fields__"):
        from dataclasses import asdict

        data = asdict(history)

    elif isinstance(history, dict):
        data = history

    else:
        raise TypeError(
            "history must be a dataclass instance or dictionary."
        )

    return save_json(
        data=data,
        path=output_path,
    )


def validate_dataloader(
    dataloader: Any,
    name: str = "DataLoader",
) -> None:
    """Perform basic DataLoader validation."""

    if dataloader is None:
        raise ValueError(
            f"{name} cannot be None."
        )

    if not hasattr(dataloader, "__iter__"):
        raise TypeError(
            f"{name} must be iterable."
        )

    try:
        length = len(dataloader)
    except TypeError:
        return

    if length == 0:
        raise ValueError(
            f"{name} contains zero batches."
        )


def format_seconds(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""

    if seconds < 0:
        raise ValueError(
            "seconds cannot be negative."
        )

    hours, remainder = divmod(
        int(seconds),
        3600,
    )

    minutes, seconds_remaining = divmod(
        remainder,
        60,
    )

    if hours > 0:
        return (
            f"{hours}h "
            f"{minutes}m "
            f"{seconds_remaining}s"
        )

    if minutes > 0:
        return (
            f"{minutes}m "
            f"{seconds_remaining}s"
        )

    return f"{seconds_remaining}s"


def checkpoint_exists(
    path: str | Path,
) -> bool:
    """Check whether a checkpoint file exists."""

    return Path(path).is_file()


def get_checkpoint_size_mb(
    path: str | Path,
) -> float:
    """Return checkpoint size in megabytes."""

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    return checkpoint_path.stat().st_size / (
        1024 * 1024
    )


__all__ = [
    "set_seed",
    "get_device",
    "count_parameters",
    "get_current_learning_rate",
    "calculate_dice",
    "calculate_iou",
    "save_json",
    "load_json",
    "save_history",
    "validate_dataloader",
    "format_seconds",
    "checkpoint_exists",
    "get_checkpoint_size_mb",
]