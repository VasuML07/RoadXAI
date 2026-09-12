"""
Utility functions for the RoadXAI Explainable AI pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


def normalize_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize an image to uint8 [0, 255].
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "image must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "image cannot be empty."
        )

    image = image.astype(np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum <= minimum:
        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    normalized = (
        (image - minimum)
        / (maximum - minimum)
        * 255.0
    )

    return np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)


def tensor_to_image(
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Convert a tensor image into a uint8 NumPy image.

    Supports:
        [C, H, W]
        [1, C, H, W]
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "tensor must be a torch.Tensor."
        )

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Only batch size 1 is supported."
            )

        tensor = tensor[0]

    if tensor.ndim != 3:
        raise ValueError(
            "Expected tensor shape [C, H, W]."
        )

    if tensor.shape[0] != 3:
        raise ValueError(
            "Expected an RGB image with 3 channels."
        )

    image = tensor.detach().cpu().float()

    image_min = image.min()
    image_max = image.max()

    if image_min < 0.0 or image_max > 1.0:
        image = normalize_tensor(image)

    image = image.clamp(
        0.0,
        1.0,
    )

    image = (
        image.permute(1, 2, 0)
        .numpy()
        * 255.0
    )

    return np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)


def normalize_tensor(
    tensor: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Min-max normalize a tensor to [0, 1].
    """

    minimum = tensor.amin()

    maximum = tensor.amax()

    return (
        tensor - minimum
    ) / (
        maximum - minimum + eps
    )


def resize_heatmap(
    heatmap: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize a heatmap.

    Args:
        heatmap:
            2D heatmap.

        target_size:
            (width, height).
    """

    if not isinstance(heatmap, np.ndarray):
        raise TypeError(
            "heatmap must be a NumPy array."
        )

    if heatmap.ndim != 2:
        raise ValueError(
            "heatmap must be a 2D array."
        )

    width, height = target_size

    if width <= 0 or height <= 0:
        raise ValueError(
            "target dimensions must be positive."
        )

    return cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def apply_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Convert a normalized heatmap into a color visualization.
    """

    if not isinstance(heatmap, np.ndarray):
        raise TypeError(
            "heatmap must be a NumPy array."
        )

    if heatmap.ndim != 2:
        raise ValueError(
            "heatmap must be a 2D array."
        )

    normalized = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    heatmap_uint8 = (
        normalized * 255.0
    ).astype(np.uint8)

    return cv2.applyColorMap(
        heatmap_uint8,
        colormap,
    )


def blend_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a CAM heatmap with an image.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape [H, W, 3]."
        )

    if heatmap.ndim != 2:
        raise ValueError(
            "heatmap must have shape [H, W]."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    image_uint8 = np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)

    if heatmap.shape != image_uint8.shape[:2]:
        heatmap = resize_heatmap(
            heatmap,
            (
                image_uint8.shape[1],
                image_uint8.shape[0],
            ),
        )

    colored = apply_colormap(
        heatmap,
        colormap,
    )

    return cv2.addWeighted(
        image_uint8,
        1.0 - alpha,
        colored,
        alpha,
        0,
    )


def save_heatmap(
    heatmap: np.ndarray,
    path: str | Path,
) -> Path:
    """
    Save a heatmap as a PNG image.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if heatmap.ndim != 2:
        raise ValueError(
            "heatmap must be a 2D array."
        )

    heatmap_uint8 = (
        np.clip(
            heatmap,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(np.uint8)

    success = cv2.imwrite(
        str(output_path),
        heatmap_uint8,
    )

    if not success:
        raise IOError(
            f"Failed to save heatmap: {output_path}"
        )

    return output_path


def save_overlay(
    overlay: np.ndarray,
    path: str | Path,
) -> Path:
    """
    Save an explanation overlay as an image.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if overlay.ndim != 3 or overlay.shape[2] != 3:
        raise ValueError(
            "overlay must have shape [H, W, 3]."
        )

    success = cv2.imwrite(
        str(output_path),
        overlay,
    )

    if not success:
        raise IOError(
            f"Failed to save overlay: {output_path}"
        )

    return output_path


def get_layer_by_name(
    model: nn.Module,
    layer_name: str,
) -> nn.Module:
    """
    Retrieve a model layer using its dotted module name.

    Example:
        get_layer_by_name(model, "decoder1.conv.block.0")
    """

    if not layer_name:
        raise ValueError(
            "layer_name cannot be empty."
        )

    modules = dict(
        model.named_modules()
    )

    if layer_name not in modules:
        available = list(
            modules.keys()
        )

        raise ValueError(
            f"Layer '{layer_name}' was not found. "
            f"Available layers: {available}"
        )

    return modules[layer_name]


def list_conv_layers(
    model: nn.Module,
) -> list[str]:
    """
    Return names of all Conv2d layers in a model.
    """

    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]


def select_last_conv_layer(
    model: nn.Module,
) -> tuple[str, nn.Module]:
    """
    Select the final Conv2d layer in the model.
    """

    layers = [
        (
            name,
            module,
        )
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]

    if not layers:
        raise ValueError(
            "No Conv2d layers found in model."
        )

    return layers[-1]


def validate_heatmap(
    heatmap: np.ndarray,
) -> None:
    """
    Validate a heatmap before visualization or saving.
    """

    if not isinstance(
        heatmap,
        np.ndarray,
    ):
        raise TypeError(
            "heatmap must be a NumPy array."
        )

    if heatmap.ndim != 2:
        raise ValueError(
            "heatmap must be 2-dimensional."
        )

    if heatmap.size == 0:
        raise ValueError(
            "heatmap cannot be empty."
        )

    if not np.isfinite(
        heatmap
    ).all():
        raise ValueError(
            "heatmap contains NaN or infinite values."
        )


def heatmap_statistics(
    heatmap: np.ndarray,
) -> dict[str, float]:
    """
    Calculate basic statistics for an explanation heatmap.
    """

    validate_heatmap(
        heatmap
    )

    return {
        "minimum": float(
            heatmap.min()
        ),
        "maximum": float(
            heatmap.max()
        ),
        "mean": float(
            heatmap.mean()
        ),
        "std": float(
            heatmap.std()
        ),
        "active_ratio": float(
            (heatmap >= 0.5).mean()
        ),
    }


def explanation_metadata(
    method: str,
    target_class: int,
    heatmap: np.ndarray,
) -> dict[str, Any]:
    """
    Create metadata describing an explanation result.
    """

    validate_heatmap(
        heatmap
    )

    statistics = heatmap_statistics(
        heatmap
    )

    return {
        "method": method,
        "target_class": int(target_class),
        "height": int(heatmap.shape[0]),
        "width": int(heatmap.shape[1]),
        "statistics": statistics,
    }


__all__ = [
    "normalize_image",
    "tensor_to_image",
    "normalize_tensor",
    "resize_heatmap",
    "apply_colormap",
    "blend_heatmap",
    "save_heatmap",
    "save_overlay",
    "get_layer_by_name",
    "list_conv_layers",
    "select_last_conv_layer",
    "validate_heatmap",
    "heatmap_statistics",
    "explanation_metadata",
]