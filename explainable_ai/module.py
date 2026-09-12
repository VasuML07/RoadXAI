"""
RoadXAI Explainable AI Module.

Implements:
    - Grad-CAM
    - Grad-CAM++
    - Score-CAM
    - Heatmap generation
    - Heatmap overlay

The methods are designed for the RoadXAI U-Net segmentation model.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationHook:
    """
    Capture activations and gradients from a target layer.
    """

    def __init__(self, layer: nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if isinstance(output, torch.Tensor):
            self.activations = output

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0]

    def remove(self) -> None:
        """Remove registered hooks."""

        self.forward_handle.remove()
        self.backward_handle.remove()


def _validate_image_tensor(
    image: torch.Tensor,
) -> None:
    """Validate an input image tensor."""

    if not isinstance(image, torch.Tensor):
        raise TypeError(
            "image must be a torch.Tensor."
        )

    if image.ndim != 4:
        raise ValueError(
            "Expected image shape [B, C, H, W]."
        )

    if image.shape[0] < 1:
        raise ValueError(
            "Image batch cannot be empty."
        )

    if image.shape[1] != 3:
        raise ValueError(
            "RoadXAI expects RGB input with 3 channels."
        )

    if image.shape[-2] <= 0 or image.shape[-1] <= 0:
        raise ValueError(
            "Image height and width must be positive."
        )

    if not torch.is_floating_point(image):
        raise TypeError(
            "image must be a floating-point tensor."
        )


def _prepare_autograd_image(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Convert an input tensor into a normal tensor usable by autograd.

    Streamlit inference uses torch.inference_mode() for normal
    prediction. Grad-CAM requires autograd, so the inference tensor
    must be converted outside inference mode before the explanation
    is generated.
    """

    _validate_image_tensor(image)

    with torch.inference_mode(False):
        prepared = image.detach().clone()

    prepared.requires_grad_(False)

    return prepared


def _get_target_score(
    output: torch.Tensor,
    target_class: int = 0,
) -> torch.Tensor:
    """
    Get a scalar target score for segmentation.

    For the binary RoadXAI model:
        target_class 0 -> defect output channel

    The spatial mean is used so the explanation represents the
    model's overall defect prediction.
    """

    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "Model output must be a torch.Tensor."
        )

    if output.ndim != 4:
        raise ValueError(
            "Expected model output shape [B, C, H, W]."
        )

    if not 0 <= target_class < output.shape[1]:
        raise ValueError(
            f"target_class must be between 0 and "
            f"{output.shape[1] - 1}."
        )

    return output[:, target_class].mean()


def _normalize_heatmap(
    heatmap: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize a heatmap to [0, 1]."""

    heatmap = torch.relu(heatmap)

    minimum = heatmap.amin(
        dim=(-2, -1),
        keepdim=True,
    )

    maximum = heatmap.amax(
        dim=(-2, -1),
        keepdim=True,
    )

    denominator = maximum - minimum

    normalized = torch.where(
        denominator > eps,
        (heatmap - minimum)
        / denominator.clamp_min(eps),
        torch.zeros_like(heatmap),
    )

    return normalized.clamp(
        0.0,
        1.0,
    )


def _resize_heatmap(
    heatmap: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Resize heatmap to image spatial dimensions."""

    if heatmap.ndim != 3:
        raise ValueError(
            "Expected heatmap shape [B, H, W]."
        )

    return F.interpolate(
        heatmap.unsqueeze(1),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def grad_cam(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: nn.Module,
    target_class: int = 0,
) -> torch.Tensor:
    """
    Generate a Grad-CAM heatmap.

    Args:
        model:
            Trained CNN segmentation model.

        image:
            Input tensor with shape [B, 3, H, W].

        target_layer:
            Convolutional layer to explain.

        target_class:
            Output channel being explained.

    Returns:
        Heatmap with shape [B, H, W], normalized to [0, 1].
    """

    image = _prepare_autograd_image(image)

    if not isinstance(target_layer, nn.Module):
        raise TypeError(
            "target_layer must be a torch.nn.Module."
        )

    model.eval()

    hook = ActivationHook(target_layer)

    try:
        model.zero_grad(set_to_none=True)

        with torch.inference_mode(False):
            with torch.enable_grad():
                output = model(image)

                score = _get_target_score(
                    output,
                    target_class,
                )

                score.backward()

        if hook.activations is None:
            raise RuntimeError(
                "Target-layer activations were not captured."
            )

        if hook.gradients is None:
            raise RuntimeError(
                "Target-layer gradients were not captured."
            )

        activations = hook.activations
        gradients = hook.gradients

        if activations.ndim != 4:
            raise RuntimeError(
                "Target-layer activations must have shape "
                "[B, C, H, W]."
            )

        if gradients.shape != activations.shape:
            raise RuntimeError(
                "Target-layer gradients and activations have "
                "different shapes."
            )

        weights = gradients.mean(
            dim=(-2, -1),
            keepdim=True,
        )

        heatmap = (
            weights * activations
        ).sum(dim=1)

        heatmap = _normalize_heatmap(
            heatmap
        )

        heatmap = _resize_heatmap(
            heatmap,
            image.shape[-2:],
        )

        return heatmap.detach()

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def grad_cam_plus_plus(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: nn.Module,
    target_class: int = 0,
) -> torch.Tensor:
    """
    Generate a Grad-CAM++ heatmap.

    Uses higher-order gradient information to calculate
    activation importance weights.
    """

    image = _prepare_autograd_image(image)

    if not isinstance(target_layer, nn.Module):
        raise TypeError(
            "target_layer must be a torch.nn.Module."
        )

    model.eval()

    hook = ActivationHook(target_layer)

    try:
        model.zero_grad(set_to_none=True)

        with torch.inference_mode(False):
            with torch.enable_grad():
                output = model(image)

                score = _get_target_score(
                    output,
                    target_class,
                )

                if hook.activations is None:
                    raise RuntimeError(
                        "Target-layer activations were not captured."
                    )

                first_gradients = torch.autograd.grad(
                    score,
                    output,
                    create_graph=True,
                    retain_graph=True,
                )[0]

                score.backward(
                    retain_graph=True
                )

        if hook.gradients is None:
            raise RuntimeError(
                "Target-layer gradients were not captured."
            )

        activations = hook.activations
        gradients = hook.gradients

        if activations.ndim != 4:
            raise RuntimeError(
                "Target-layer activations must have shape "
                "[B, C, H, W]."
            )

        if gradients.shape != activations.shape:
            raise RuntimeError(
                "Target-layer gradients and activations have "
                "different shapes."
            )

        positive_gradients = torch.relu(
            gradients
        )

        gradient_squared = (
            positive_gradients ** 2
        )

        gradient_cubed = (
            positive_gradients ** 3
        )

        activation_weighted_cubed = (
            activations
            * gradient_cubed
        ).sum(
            dim=(-2, -1),
            keepdim=True,
        )

        alpha_denominator = (
            2.0 * gradient_squared
            + activation_weighted_cubed
        )

        alpha_denominator = torch.where(
            alpha_denominator.abs() > 1e-8,
            alpha_denominator,
            torch.ones_like(alpha_denominator),
        )

        alphas = (
            gradient_squared
            / alpha_denominator
        )

        weights = (
            alphas
            * positive_gradients
        ).sum(
            dim=(-2, -1),
            keepdim=True,
        )

        heatmap = (
            weights * activations
        ).sum(dim=1)

        if first_gradients is not None:
            gradient_scale = (
                first_gradients
                .detach()
                .abs()
                .mean()
            )

            if torch.isfinite(gradient_scale):
                heatmap = heatmap * (
                    1.0 + gradient_scale
                )

        heatmap = _normalize_heatmap(
            heatmap
        )

        heatmap = _resize_heatmap(
            heatmap,
            image.shape[-2:],
        )

        return heatmap.detach()

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def score_cam(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: nn.Module,
    target_class: int = 0,
) -> torch.Tensor:
    """
    Generate a Score-CAM heatmap.

    Score-CAM uses activation maps as masks and measures the
    change in the target model score produced by each mask.

    Returns:
        Heatmap with shape [B, H, W].
    """

    _validate_image_tensor(image)

    if not isinstance(target_layer, nn.Module):
        raise TypeError(
            "target_layer must be a torch.nn.Module."
        )

    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def forward_hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if isinstance(output, torch.Tensor):
            captured["activations"] = output.detach()

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:
        original_output = model(image)

        original_score = (
            _get_target_score(
                original_output,
                target_class,
            ).detach()
        )

        if "activations" not in captured:
            raise RuntimeError(
                "Target-layer activations were not captured."
            )

        activations = captured["activations"]

        if activations.ndim != 4:
            raise RuntimeError(
                "Target-layer activations must have shape "
                "[B, C, H, W]."
            )

        batch_size = image.shape[0]

        if batch_size != 1:
            raise ValueError(
                "Score-CAM currently supports batch size 1."
            )

        activation_maps = activations[0]

        activation_maps = _normalize_heatmap(
            activation_maps.unsqueeze(0)
        )[0]

        input_height = image.shape[-2]
        input_width = image.shape[-1]

        activation_maps = F.interpolate(
            activation_maps.unsqueeze(1),
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        weights: list[torch.Tensor] = []

        for activation_map in activation_maps:
            masked_image = (
                image
                * activation_map.unsqueeze(0).unsqueeze(0)
            )

            masked_output = model(
                masked_image
            )

            masked_score = _get_target_score(
                masked_output,
                target_class,
            )

            weight = (
                masked_score - original_score
            )

            weights.append(
                weight
            )

        channel_weights = torch.stack(
            weights
        ).view(
            1,
            -1,
            1,
            1,
        )

        heatmap = (
            channel_weights
            * activation_maps.unsqueeze(0)
        ).sum(dim=1)

        heatmap = _normalize_heatmap(
            heatmap
        )

        return heatmap.detach()

    finally:
        handle.remove()


def generate_heatmap(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: nn.Module,
    method: str = "gradcam",
    target_class: int = 0,
) -> torch.Tensor:
    """
    Generate an explanation using the selected method.

    Supported methods:
        gradcam
        grad-cam
        gradcam++
        grad-cam++
        scorecam
        score-cam
    """

    if not isinstance(method, str):
        raise TypeError(
            "method must be a string."
        )

    method_name = (
        method
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    if method_name == "gradcam":
        return grad_cam(
            model=model,
            image=image,
            target_layer=target_layer,
            target_class=target_class,
        )

    if method_name == "gradcam++":
        return grad_cam_plus_plus(
            model=model,
            image=image,
            target_layer=target_layer,
            target_class=target_class,
        )

    if method_name == "scorecam":
        return score_cam(
            model=model,
            image=image,
            target_layer=target_layer,
            target_class=target_class,
        )

    raise ValueError(
        "Unsupported explanation method. "
        "Use 'gradcam', 'gradcam++', or 'scorecam'."
    )


def heatmap_to_numpy(
    heatmap: torch.Tensor,
) -> np.ndarray:
    """
    Convert a torch heatmap to a NumPy array.

    For a batch of one, returns [H, W].
    """

    if not isinstance(heatmap, torch.Tensor):
        raise TypeError(
            "heatmap must be a torch.Tensor."
        )

    if heatmap.ndim == 3:
        if heatmap.shape[0] != 1:
            raise ValueError(
                "heatmap_to_numpy expects batch size 1."
            )

        heatmap = heatmap[0]

    if heatmap.ndim != 2:
        raise ValueError(
            "Expected heatmap shape [H, W] or [1, H, W]."
        )

    heatmap = (
        heatmap
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    if not np.isfinite(heatmap).all():
        raise ValueError(
            "heatmap contains NaN or infinite values."
        )

    return np.clip(
        heatmap,
        0.0,
        1.0,
    )


def create_colormap(
    heatmap: torch.Tensor,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Convert a normalized heatmap into a BGR color map.
    """

    heatmap_np = heatmap_to_numpy(
        heatmap
    )

    heatmap_uint8 = (
        heatmap_np * 255.0
    ).astype(np.uint8)

    return cv2.applyColorMap(
        heatmap_uint8,
        colormap,
    )


def overlay_heatmap(
    image: np.ndarray,
    heatmap: torch.Tensor | np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay an explanation heatmap on an image.

    Args:
        image:
            RGB or BGR uint8 image with shape [H, W, 3].

        heatmap:
            Normalized heatmap.

        alpha:
            Heatmap contribution in the final image.

        colormap:
            OpenCV colormap identifier.

    Returns:
        BGR uint8 visualization.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "image must be a NumPy array."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected image shape [H, W, 3]."
        )

    if image.size == 0:
        raise ValueError(
            "image cannot be empty."
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

    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap_to_numpy(
            heatmap
        )
    elif isinstance(heatmap, np.ndarray):
        heatmap_np = heatmap

        if heatmap_np.ndim == 3:
            if heatmap_np.shape[0] != 1:
                raise ValueError(
                    "NumPy heatmap batch must have size 1."
                )

            heatmap_np = heatmap_np[0]

        if heatmap_np.ndim != 2:
            raise ValueError(
                "Expected heatmap shape [H, W]."
            )

        if not np.isfinite(heatmap_np).all():
            raise ValueError(
                "heatmap contains NaN or infinite values."
            )

        heatmap_np = np.clip(
            heatmap_np,
            0.0,
            1.0,
        ).astype(np.float32)

    else:
        raise TypeError(
            "heatmap must be a torch.Tensor or NumPy array."
        )

    if heatmap_np.shape != image_uint8.shape[:2]:
        heatmap_np = cv2.resize(
            heatmap_np,
            (
                image_uint8.shape[1],
                image_uint8.shape[0],
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    heatmap_uint8 = (
        heatmap_np * 255.0
    ).astype(np.uint8)

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        colormap,
    )

    overlay = cv2.addWeighted(
        image_uint8,
        1.0 - alpha,
        colored_heatmap,
        alpha,
        0,
    )

    return overlay


def get_default_target_layer(
    model: nn.Module,
) -> nn.Module:
    """
    Find a suitable final convolutional layer.

    The search walks through model modules in reverse order and
    returns the last Conv2d layer.

    This preserves the existing RoadXAI behavior and allows
    architecture-independent CAM generation.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    convolutional_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]

    if not convolutional_layers:
        raise ValueError(
            "No Conv2d layer was found in the model."
        )

    return convolutional_layers[-1]


__all__ = [
    "ActivationHook",
    "grad_cam",
    "grad_cam_plus_plus",
    "score_cam",
    "generate_heatmap",
    "heatmap_to_numpy",
    "create_colormap",
    "overlay_heatmap",
    "get_default_target_layer",
]