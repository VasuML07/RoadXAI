"""
RoadXAI Streamlit Application Utilities.

Helper functions for Streamlit display, image conversion,
mask visualization, model-output formatting, and application state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image


def numpy_to_pil(
    image: np.ndarray,
) -> Image.Image:
    """
    Convert a NumPy image into a PIL RGB image.
    """
    array = np.asarray(image)

    if array.ndim == 2:
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(
            "Expected image with shape [H, W, 3] or [H, W, 4]."
        )

    if np.issubdtype(array.dtype, np.floating):
        if array.max() <= 1.0:
            array = array * 255.0

    array = np.clip(array, 0, 255).astype(np.uint8)

    if array.shape[2] == 4:
        return Image.fromarray(array, mode="RGBA")

    return Image.fromarray(array, mode="RGB")


def pil_to_numpy(
    image: Image.Image,
) -> np.ndarray:
    """
    Convert a PIL image into a uint8 RGB NumPy array.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image.")

    return np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )


def mask_to_uint8(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Convert a binary mask into a displayable 0-255 image.
    """
    array = np.asarray(mask)

    if array.ndim != 2:
        raise ValueError("mask must be 2D.")

    return (
        (array > 0).astype(np.uint8) * 255
    )


def probability_to_uint8(
    probability_map: np.ndarray,
) -> np.ndarray:
    """
    Convert a probability map in [0, 1] to 8-bit grayscale.
    """
    array = np.asarray(
        probability_map,
        dtype=np.float32,
    )

    if array.ndim != 2:
        raise ValueError(
            "probability_map must be 2D."
        )

    array = np.clip(
        array,
        0.0,
        1.0,
    )

    return (
        array * 255.0
    ).astype(np.uint8)


def create_side_by_side(
    image: np.ndarray,
    mask: np.ndarray,
    overlay: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine original image, mask, and optional overlay horizontally.
    """
    original = pil_to_numpy(
        numpy_to_pil(image)
    )

    binary_mask = mask_to_uint8(mask)

    mask_rgb = np.stack(
        [binary_mask] * 3,
        axis=-1,
    )

    if mask_rgb.shape[:2] != original.shape[:2]:
        mask_rgb = np.asarray(
            numpy_to_pil(mask_rgb).resize(
                (
                    original.shape[1],
                    original.shape[0],
                ),
                Image.Resampling.NEAREST,
            )
        )

    panels = [
        original,
        mask_rgb,
    ]

    if overlay is not None:
        overlay_array = np.asarray(overlay)

        if overlay_array.shape[:2] != original.shape[:2]:
            overlay_array = np.asarray(
                numpy_to_pil(
                    overlay_array
                ).resize(
                    (
                        original.shape[1],
                        original.shape[0],
                    ),
                    Image.Resampling.BILINEAR,
                )
            )

        panels.append(
            pil_to_numpy(
                numpy_to_pil(overlay_array)
            )
        )

    return np.concatenate(
        panels,
        axis=1,
    )


def normalize_display_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize an image for Streamlit display.
    """
    array = np.asarray(image)

    if array.ndim not in (2, 3):
        raise ValueError(
            "Image must be 2D or 3D."
        )

    if np.issubdtype(array.dtype, np.floating):
        minimum = float(array.min())
        maximum = float(array.max())

        if 0.0 <= minimum and maximum <= 1.0:
            array = array * 255.0
        elif maximum > minimum:
            array = (
                (array - minimum)
                / (maximum - minimum)
                * 255.0
            )

    return np.clip(
        array,
        0,
        255,
    ).astype(np.uint8)


def format_percentage(
    value: float,
    decimals: int = 2,
) -> str:
    """
    Format a numeric value as a percentage.
    """
    return f"{float(value):.{decimals}f}%"


def format_confidence(
    confidence: float,
    decimals: int = 2,
) -> str:
    """
    Format a confidence value as a percentage.
    """
    confidence = float(confidence)

    if 0.0 <= confidence <= 1.0:
        confidence *= 100.0

    return f"{confidence:.{decimals}f}%"


def format_area(
    area: float,
    unit: str = "px²",
    decimals: int = 2,
) -> str:
    """
    Format an area measurement.
    """
    return f"{float(area):.{decimals}f} {unit}"


def format_length(
    length: float,
    unit: str = "px",
    decimals: int = 2,
) -> str:
    """
    Format a length measurement.
    """
    return f"{float(length):.{decimals}f} {unit}"


def severity_label(
    severity: Any,
) -> str:
    """
    Convert a severity enum/value into a display label.
    """
    if hasattr(severity, "value"):
        severity = severity.value

    text = str(severity).strip()

    if not text:
        return "Unknown"

    return text.replace(
        "_",
        " ",
    ).replace(
        "-",
        " ",
    ).title()


def condition_label(
    condition: Any,
) -> str:
    """
    Convert a road-condition enum/value into a display label.
    """
    if hasattr(condition, "value"):
        condition = condition.value

    text = str(condition).strip()

    if not text:
        return "Unknown"

    return text.replace(
        "_",
        " ",
    ).replace(
        "-",
        " ",
    ).title()


def build_metric_dict(
    defect_percentage: float,
    confidence: float,
    defect_pixels: int,
    image_shape: tuple,
) -> Dict[str, Any]:
    """
    Build a standardized dictionary for dashboard metrics.
    """
    return {
        "defect_percentage": float(
            defect_percentage
        ),
        "confidence": float(confidence),
        "defect_pixels": int(defect_pixels),
        "image_height": int(image_shape[0]),
        "image_width": int(image_shape[1]),
    }


def safe_get(
    data: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve a value from dictionaries or objects.
    """
    if data is None:
        return default

    if isinstance(data, dict):
        return data.get(key, default)

    return getattr(
        data,
        key,
        default,
    )


def result_to_display_dict(
    result: Any,
) -> Dict[str, Any]:
    """
    Convert a dataclass/object/dictionary result into
    a JSON-friendly display dictionary.
    """
    if result is None:
        return {}

    if isinstance(result, dict):
        output = {}

        for key, value in result.items():
            if hasattr(value, "value"):
                output[key] = value.value
            elif isinstance(value, np.ndarray):
                output[key] = value.tolist()
            elif isinstance(value, np.generic):
                output[key] = value.item()
            elif isinstance(value, dict):
                output[key] = result_to_display_dict(
                    value
                )
            elif isinstance(value, (list, tuple)):
                output[key] = [
                    (
                        item.value
                        if hasattr(item, "value")
                        else item
                    )
                    for item in value
                ]
            else:
                output[key] = value

        return output

    if hasattr(result, "__dataclass_fields__"):
        from dataclasses import asdict

        return result_to_display_dict(
            asdict(result)
        )

    if hasattr(result, "__dict__"):
        return result_to_display_dict(
            vars(result)
        )

    return {"value": result}


def save_uploaded_file(
    uploaded_file: Any,
    output_directory: Union[str, Path],
) -> Optional[Path]:
    """
    Save a Streamlit UploadedFile to disk.

    Returns None when no file is supplied.
    """
    if uploaded_file is None:
        return None

    output_dir = Path(output_directory)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = Path(
        uploaded_file.name
    ).name

    if not filename:
        raise ValueError(
            "Uploaded file has no valid filename."
        )

    output_path = output_dir / filename

    with output_path.open("wb") as file:
        file.write(
            uploaded_file.getvalue()
        )

    return output_path


def create_download_bytes(
    data: Any,
    format: str = "png",
) -> bytes:
    """
    Convert common visualization data into downloadable bytes.

    Supported formats:
        png
        jpg/jpeg
    """
    image = numpy_to_pil(
        np.asarray(data)
    )

    from io import BytesIO

    buffer = BytesIO()

    normalized_format = format.lower()

    if normalized_format == "png":
        image.save(
            buffer,
            format="PNG",
        )
    elif normalized_format in {"jpg", "jpeg"}:
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=95,
        )
    else:
        raise ValueError(
            f"Unsupported image format: {format}"
        )

    return buffer.getvalue()


def reset_streamlit_state(
    keys: list[str],
) -> None:
    """
    Remove selected keys from Streamlit session state.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required."
        ) from exc

    for key in keys:
        st.session_state.pop(
            key,
            None,
        )


def initialize_streamlit_state(
    defaults: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Initialize Streamlit session-state values.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required."
        ) from exc

    defaults = defaults or {}

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_error(
    message: str,
    exception: Optional[Exception] = None,
) -> None:
    """
    Display a consistent Streamlit error message.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required."
        ) from exc

    if exception is not None:
        st.error(
            f"{message}: {exception}"
        )
    else:
        st.error(message)


def display_success(
    message: str,
) -> None:
    """
    Display a Streamlit success message.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required."
        ) from exc

    st.success(message)


def display_info(
    message: str,
) -> None:
    """
    Display a Streamlit informational message.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required."
        ) from exc

    st.info(message)