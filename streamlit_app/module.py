"""
RoadXAI Streamlit Application Module.

Provides reusable Streamlit components for the RoadXAI inference
workflow, including image upload, segmentation, explainability,
engineering analysis, and 3D visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


@dataclass
class InferenceOutput:
    """Container for a complete RoadXAI inference result."""

    original_image: np.ndarray
    input_tensor: torch.Tensor
    prediction_mask: np.ndarray
    probability_map: np.ndarray
    confidence: float


def load_image(
    image_source: Union[str, Path, bytes, Image.Image],
) -> Image.Image:
    """
    Load an image from a path, bytes, or PIL Image.

    The image is converted to RGB.
    """
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    if isinstance(image_source, (str, Path)):
        path = Path(image_source)

        if not path.exists():
            raise FileNotFoundError(
                f"Image file not found: {path}"
            )

        return Image.open(path).convert("RGB")

    if isinstance(image_source, bytes):
        from io import BytesIO

        return Image.open(
            BytesIO(image_source)
        ).convert("RGB")

    raise TypeError(
        "image_source must be a file path, bytes, or PIL Image."
    )


def image_to_numpy(
    image: Image.Image,
) -> np.ndarray:
    """
    Convert a PIL image into an RGB NumPy array.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image.")

    return np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )


def preprocess_image(
    image: Image.Image,
    image_size: Tuple[int, int] = (512, 512),
) -> torch.Tensor:
    """
    Convert an RGB image into a model-ready tensor.

    Output shape:
        [1, 3, H, W]
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image.")

    if len(image_size) != 2:
        raise ValueError(
            "image_size must contain height and width."
        )

    height, width = image_size

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image dimensions must be positive."
        )

    resized = image.convert("RGB").resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        resized,
        dtype=np.float32,
    ) / 255.0

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1)
    )

    tensor = tensor.unsqueeze(0)

    return tensor.contiguous()


def postprocess_prediction(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert model logits into a binary mask and probability map.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor.")

    if logits.ndim != 4:
        raise ValueError(
            f"Expected logits [B,C,H,W], got {logits.shape}."
        )

    if logits.shape[0] != 1:
        raise ValueError(
            "Streamlit inference currently expects batch size 1."
        )

    if logits.shape[1] != 1:
        raise ValueError(
            "RoadXAI expects one binary segmentation channel."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = torch.sigmoid(logits)

    probability_map = (
        probabilities[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    prediction_mask = (
        probability_map >= threshold
    ).astype(np.uint8)

    defect_pixels = prediction_mask.sum()

    if defect_pixels > 0:
        confidence = float(
            probability_map[prediction_mask > 0].mean()
        )
    else:
        confidence = float(
            1.0 - probability_map.mean()
        )

    return (
        prediction_mask,
        probability_map.astype(np.float32),
        confidence,
    )


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    image: Image.Image,
    device: Optional[Union[str, torch.device]] = None,
    image_size: Tuple[int, int] = (512, 512),
    threshold: float = 0.5,
) -> InferenceOutput:
    """
    Run complete RoadXAI segmentation inference.
    """
    if device is None:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    original_image = image_to_numpy(image)

    input_tensor = preprocess_image(
        image,
        image_size=image_size,
    ).to(device)

    logits = model(input_tensor)

    (
        prediction_mask,
        probability_map,
        confidence,
    ) = postprocess_prediction(
        logits,
        threshold=threshold,
    )

    return InferenceOutput(
        original_image=original_image,
        input_tensor=input_tensor,
        prediction_mask=prediction_mask,
        probability_map=probability_map,
        confidence=confidence,
    )


def resize_mask_to_image(
    mask: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a binary prediction mask to the original image size.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D.")

    width, height = image_size

    if width <= 0 or height <= 0:
        raise ValueError(
            "Image dimensions must be positive."
        )

    mask_image = Image.fromarray(
        (mask > 0).astype(np.uint8) * 255
    )

    resized = mask_image.resize(
        (width, height),
        Image.Resampling.NEAREST,
    )

    return (
        np.asarray(resized) > 127
    ).astype(np.uint8)


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a segmentation mask on an RGB image.

    The defect region is displayed using a fixed red channel.
    """
    image_array = np.asarray(
        image,
        dtype=np.uint8,
    )

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            "image must have shape [H, W, 3]."
        )

    if mask.shape != image_array.shape[:2]:
        raise ValueError(
            "mask and image dimensions do not match."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    overlay = image_array.astype(np.float32).copy()

    defect = mask > 0

    overlay[defect, 0] = (
        alpha * 255.0
        + (1.0 - alpha) * overlay[defect, 0]
    )

    overlay[defect, 1] *= 1.0 - alpha
    overlay[defect, 2] *= 1.0 - alpha

    return np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)


def calculate_defect_percentage(
    mask: np.ndarray,
) -> float:
    """
    Calculate the percentage of pixels classified as defective.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D.")

    total_pixels = mask.size

    if total_pixels == 0:
        return 0.0

    return float(
        np.count_nonzero(mask) / total_pixels * 100.0
    )


def build_dashboard_summary(
    inference: InferenceOutput,
) -> Dict[str, Any]:
    """
    Build a compact summary suitable for Streamlit metrics.
    """
    mask = inference.prediction_mask

    return {
        "confidence": float(inference.confidence),
        "defect_percentage": calculate_defect_percentage(
            mask
        ),
        "defect_pixels": int(
            np.count_nonzero(mask)
        ),
        "image_height": int(
            inference.original_image.shape[0]
        ),
        "image_width": int(
            inference.original_image.shape[1]
        ),
    }


def display_image(
    image: np.ndarray,
    caption: str = "",
    use_column_width: bool = True,
) -> None:
    """
    Display an image through Streamlit.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required for the application."
        ) from exc

    st.image(
        image,
        caption=caption,
        use_container_width=use_column_width,
    )


def display_inference_summary(
    inference: InferenceOutput,
) -> None:
    """
    Display basic inference metrics in Streamlit.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required for the application."
        ) from exc

    summary = build_dashboard_summary(
        inference
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Defect Area",
        f"{summary['defect_percentage']:.2f}%",
    )

    col2.metric(
        "Confidence",
        f"{summary['confidence']:.2%}",
    )

    col3.metric(
        "Defect Pixels",
        f"{summary['defect_pixels']:,}",
    )


def create_upload_section():
    """
    Create the Streamlit image-upload component.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required for the application."
        ) from exc

    return st.file_uploader(
        "Upload a road image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp",
        ],
        accept_multiple_files=False,
    )


def uploaded_file_to_image(
    uploaded_file,
) -> Optional[Image.Image]:
    """
    Convert a Streamlit UploadedFile into a PIL image.
    """
    if uploaded_file is None:
        return None

    return load_image(
        uploaded_file.getvalue()
    )


def create_app_header(
    title: str = "RoadXAI",
    subtitle: str = (
        "AI-Powered Road Defect Detection "
        "and Engineering Analysis"
    ),
) -> None:
    """
    Render the main Streamlit application header.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required for the application."
        ) from exc

    st.title(title)
    st.caption(subtitle)


def create_settings_sidebar(
    default_threshold: float = 0.5,
    default_image_size: int = 512,
) -> Tuple[float, int]:
    """
    Create inference settings in the Streamlit sidebar.

    Returns
    -------
    tuple
        (threshold, image_size)
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required for the application."
        ) from exc

    with st.sidebar:
        st.header("Inference Settings")

        threshold = st.slider(
            "Segmentation Threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(default_threshold),
            step=0.05,
        )

        image_size = st.selectbox(
            "Model Image Size",
            options=[
                256,
                384,
                512,
                640,
                768,
            ],
            index=[
                256,
                384,
                512,
                640,
                768,
            ].index(default_image_size)
            if default_image_size
            in [256, 384, 512, 640, 768]
            else 2,
        )

    return threshold, image_size


def get_device() -> torch.device:
    """
    Return the best available PyTorch device.
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> torch.nn.Module:
    """
    Load a trained RoadXAI model checkpoint.
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict"
        )

        if state_dict is None:
            state_dict = checkpoint.get(
                "state_dict"
            )

        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model