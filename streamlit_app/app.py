"""
RoadXAI Streamlit Application.

Two user-facing views:

1. Road Inspection
2. Advanced Analysis

The application provides AI-assisted road-defect screening,
engineering measurements, explainable AI, interactive 3D
visualization, and report generation.

Important:
A single RGB image does not provide validated physical depth.
3D depth is therefore visualization geometry unless calibrated
depth information is supplied.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "crack500"
    / "best.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"


# ============================================================================
# ROADXAI IMPORTS
# ============================================================================

from cnn_model.module import build_model
from deployment.module import load_model

from streamlit_app.module import (
    create_app_header,
    create_settings_sidebar,
    create_upload_section,
    uploaded_file_to_image,
    run_inference,
    display_image,
    display_inference_summary,
    resize_mask_to_image,
    create_mask_overlay,
)

from engineering_analysis.module import (
    analyze_road,
)

from explainable_ai.module import (
    generate_heatmap,
    heatmap_to_numpy,
    overlay_heatmap,
    get_default_target_layer,
)

from report_generation.module import (
    generate_report,
)

from report_generation.utils import (
    render_report_text,
    save_json_report,
    save_text_report,
    generate_pdf_report,
)


visualization_module = importlib.import_module(
    "3d_visualization.module"
)

generate_3d_visualization = (
    visualization_module.generate_3d_visualization
)


# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RoadXAI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# SAFE VALUE HELPERS
# ============================================================================


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a value to a finite float."""

    try:
        result = float(value)

        if np.isfinite(result):
            return result

        return default

    except (
        TypeError,
        ValueError,
    ):
        return default


def optional_float(
    value: Any,
) -> float | None:
    """Convert a value to a finite float or None."""

    if value is None:
        return None

    try:
        result = float(value)

        if np.isfinite(result):
            return result

        return None

    except (
        TypeError,
        ValueError,
    ):
        return None


def get_value(
    source: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Read a value from either a dictionary or an object."""

    if source is None:
        return default

    if isinstance(
        source,
        dict,
    ):
        return source.get(
            key,
            default,
        )

    return getattr(
        source,
        key,
        default,
    )


def readable_value(
    value: Any,
) -> str:
    """Convert enum-like values into readable text."""

    if value is None:
        return "Unknown"

    if hasattr(
        value,
        "value",
    ):
        value = value.value

    return str(
        value
    ).replace(
        "_",
        " ",
    )


# ============================================================================
# IMAGE STATE
# ============================================================================


def make_image_key(
    uploaded_file: Any,
    image: Any,
) -> str:
    """Create a stable identifier for the uploaded image."""

    if hasattr(
        image,
        "convert",
    ):
        image_array = np.asarray(
            image.convert("RGB")
        )
    else:
        image_array = np.asarray(
            image
        )

    if (
        image_array.ndim != 3
        or image_array.shape[2] != 3
    ):
        raise ValueError(
            "Uploaded image must have shape [H, W, 3]."
        )

    image_array = np.ascontiguousarray(
        image_array,
        dtype=np.uint8,
    )

    digest = hashlib.sha256(
        image_array.tobytes()
    ).hexdigest()[:16]

    filename = getattr(
        uploaded_file,
        "name",
        "image",
    )

    return (
        f"{filename}|"
        f"{digest}|"
        f"{image_array.shape}"
    )


def clear_stale_analysis_state(
    image_key: str,
) -> None:
    """Clear cached analysis results when a new image is uploaded."""

    previous_key = st.session_state.get(
        "analysis_image_key"
    )

    if previous_key == image_key:
        return

    st.session_state[
        "analysis_image_key"
    ] = image_key

    for key in (
        "xai_result",
        "xai_image_key",
        "visualization_result",
        "visualization_key",
        "report",
        "report_image_key",
    ):
        st.session_state.pop(
            key,
            None,
        )


# ============================================================================
# MODEL
# ============================================================================


@st.cache_resource
def load_roadxai_model():
    """Load the trained CRACK500 model once."""

    model = build_model(
        base_channels=16
    )

    return load_model(
        model,
        CHECKPOINT_PATH,
        device="cpu",
    )


# ============================================================================
# DEFECT RECORDS
# ============================================================================


def build_defect_records(
    engineering_result: Any,
) -> list[dict[str, Any]]:
    """Create UI-friendly defect records."""

    defects = list(
        get_value(
            engineering_result,
            "defects",
            [],
        )
        or []
    )

    severities = list(
        get_value(
            engineering_result,
            "severity",
            [],
        )
        or []
    )

    records: list[dict[str, Any]] = []

    for index, defect in enumerate(
        defects,
        start=1,
    ):
        severity = (
            severities[index - 1]
            if index - 1 < len(severities)
            else None
        )

        level = readable_value(
            get_value(
                severity,
                "level",
                "Unknown",
            )
        )

        score = safe_float(
            get_value(
                severity,
                "score",
                0.0,
            )
        )

        level_lower = level.lower()

        if (
            "critical" in level_lower
            or score >= 75
        ):
            priority = "Critical"

        elif (
            "high" in level_lower
            or score >= 50
        ):
            priority = "High"

        elif (
            "moderate" in level_lower
            or score >= 25
        ):
            priority = "Moderate"

        else:
            priority = "Low"

        records.append(
            {
                "id": index,
                "severity": level,
                "score": score,
                "priority": priority,
                "area_pixels": safe_float(
                    get_value(
                        defect,
                        "area_pixels",
                        0.0,
                    )
                ),
                "length_pixels": safe_float(
                    get_value(
                        defect,
                        "length_pixels",
                        0.0,
                    )
                ),
                "width_pixels": safe_float(
                    get_value(
                        defect,
                        "width_pixels",
                        0.0,
                    )
                ),
                "centroid_x": safe_float(
                    get_value(
                        defect,
                        "centroid_x",
                        0.0,
                    )
                ),
                "centroid_y": safe_float(
                    get_value(
                        defect,
                        "centroid_y",
                        0.0,
                    )
                ),
                "area_m2": optional_float(
                    get_value(
                        defect,
                        "area_m2",
                        None,
                    )
                ),
                "length_m": optional_float(
                    get_value(
                        defect,
                        "length_m",
                        None,
                    )
                ),
                "width_m": optional_float(
                    get_value(
                        defect,
                        "width_m",
                        None,
                    )
                ),
                "explanation": str(
                    get_value(
                        severity,
                        "explanation",
                        "",
                    )
                    or ""
                ),
            }
        )

    records.sort(
        key=lambda item: (
            item["score"],
            item["area_pixels"],
        ),
        reverse=True,
    )

    return records


# ============================================================================
# NUMBERED DEFECT OVERLAY
# ============================================================================


def make_numbered_defect_overlay(
    original_image: np.ndarray,
    prediction_mask: np.ndarray,
    engineering_result: Any,
) -> np.ndarray:
    """Draw the segmentation mask and numbered defect locations."""

    image = np.asarray(
        original_image
    )

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError(
            "original_image must have shape [H, W, 3]."
        )

    image = np.clip(
        image,
        0,
        255,
    ).astype(
        np.uint8
    )

    mask = np.squeeze(
        np.asarray(
            prediction_mask
        )
    )

    if mask.ndim != 2:
        raise ValueError(
            "prediction_mask must be 2D."
        )

    binary_mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    image_h, image_w = image.shape[:2]
    mask_h, mask_w = binary_mask.shape

    resized_mask = cv2.resize(
        binary_mask,
        (
            image_w,
            image_h,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    # Use the project's mask-overlay helper.
    try:
        result = create_mask_overlay(
            image.copy(),
            resized_mask,
            alpha=0.32,
        )
    except Exception:
        # Safe fallback if the helper has a different signature.
        result = image.copy()

        damage = resized_mask.astype(bool)

        overlay_color = np.zeros_like(
            result
        )

        overlay_color[
            damage
        ] = (
            255,
            60,
            60,
        )

        result = cv2.addWeighted(
            result,
            0.68,
            overlay_color,
            0.32,
            0,
        )

    defects = list(
        get_value(
            engineering_result,
            "defects",
            [],
        )
        or []
    )

    for index, defect in enumerate(
        defects,
        start=1,
    ):
        source_x = safe_float(
            get_value(
                defect,
                "centroid_x",
                0.0,
            )
        )

        source_y = safe_float(
            get_value(
                defect,
                "centroid_y",
                0.0,
            )
        )

        x = int(
            round(
                source_x
                * image_w
                / max(mask_w, 1)
            )
        )

        y = int(
            round(
                source_y
                * image_h
                / max(mask_h, 1)
            )
        )

        x = max(
            20,
            min(
                image_w - 20,
                x,
            ),
        )

        y = max(
            20,
            min(
                image_h - 20,
                y,
            ),
        )

        radius = (
            18
            if image_w >= 700
            else 14
        )

        cv2.circle(
            result,
            (x, y),
            radius + 3,
            (255, 255, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            result,
            (x, y),
            radius,
            (205, 30, 30),
            -1,
            lineType=cv2.LINE_AA,
        )

        label = str(index)

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = (
            0.55
            if index < 10
            else 0.43
        )

        thickness = 2

        text_size, _ = cv2.getTextSize(
            label,
            font,
            scale,
            thickness,
        )

        text_w, text_h = text_size

        cv2.putText(
            result,
            label,
            (
                x - text_w // 2,
                y + text_h // 2,
            ),
            font,
            scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return result


# ============================================================================
# PROBABILITY VISUALIZATION
# ============================================================================


def make_probability_visualization(
    probability_map: np.ndarray,
    original_image: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create probability-map visualizations."""

    probability = np.asarray(
        probability_map,
        dtype=np.float32,
    )

    probability = np.squeeze(
        probability
    )

    if probability.ndim != 2:
        raise ValueError(
            "Probability map must be 2D."
        )

    probability = np.nan_to_num(
        probability,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    probability = np.clip(
        probability,
        0.0,
        1.0,
    )

    probability_u8 = (
        probability * 255.0
    ).astype(
        np.uint8
    )

    color_bgr = cv2.applyColorMap(
        probability_u8,
        cv2.COLORMAP_JET,
    )

    color_rgb = cv2.cvtColor(
        color_bgr,
        cv2.COLOR_BGR2RGB,
    )

    image_rgb = np.asarray(
        original_image
    )

    image_rgb = np.clip(
        image_rgb,
        0,
        255,
    ).astype(
        np.uint8
    )

    if (
        color_rgb.shape[:2]
        != image_rgb.shape[:2]
    ):
        color_rgb = cv2.resize(
            color_rgb,
            (
                image_rgb.shape[1],
                image_rgb.shape[0],
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    overlay = cv2.addWeighted(
        image_rgb,
        0.55,
        color_rgb,
        0.45,
        0,
    )

    return {
        "grayscale": probability_u8,
        "color": color_rgb,
        "overlay": overlay,
    }


# ============================================================================
# XAI
# ============================================================================


def prepare_xai_tensor(
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Create a fresh tensor suitable for Grad-CAM autograd."""

    if not isinstance(
        input_tensor,
        torch.Tensor,
    ):
        raise TypeError(
            "input_tensor must be a torch.Tensor."
        )

    array = (
        input_tensor
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    return torch.tensor(
        array,
        dtype=torch.float32,
        device="cpu",
    )


def run_xai(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> dict[str, Any]:
    """Generate a robust Grad-CAM explanation."""

    xai_input = prepare_xai_tensor(
        input_tensor
    )

    target_layer = get_default_target_layer(
        model
    )

    raw_heatmap = generate_heatmap(
        model=model,
        image=xai_input,
        target_layer=target_layer,
        method="gradcam",
        target_class=0,
    )

    heatmap_tensor = heatmap_to_tensor(
        raw_heatmap
    )

    heatmap_array = heatmap_to_numpy(
        heatmap_tensor
    )

    if heatmap_array.ndim == 3:
        heatmap_array = heatmap_array[0]

    heatmap_array = np.asarray(
        heatmap_array,
        dtype=np.float32,
    )

    heatmap_array = np.nan_to_num(
        heatmap_array,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    minimum = float(
        heatmap_array.min()
    )

    maximum = float(
        heatmap_array.max()
    )

    if maximum > minimum:
        heatmap_array = (
            heatmap_array - minimum
        ) / (
            maximum - minimum
        )

    else:
        heatmap_array = np.zeros_like(
            heatmap_array
        )

    heatmap_array = np.clip(
        heatmap_array,
        0.0,
        1.0,
    )

    image_array = (
        xai_input[0]
        .detach()
        .cpu()
        .numpy()
        .transpose(
            1,
            2,
            0,
        )
    )

    image_array = np.clip(
        image_array * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    overlay = overlay_heatmap(
        image_array,
        heatmap_array,
        alpha=0.45,
    )

    return {
        "method": "Grad-CAM",
        "target_layer": target_layer,
        "heatmap": heatmap_array,
        "overlay": overlay,
    }


def heatmap_to_tensor(
    heatmap: Any,
) -> torch.Tensor:
    """Normalize a heatmap into [1, H, W]."""

    if isinstance(
        heatmap,
        torch.Tensor,
    ):
        result = (
            heatmap
            .detach()
            .cpu()
            .float()
        )

    elif isinstance(
        heatmap,
        np.ndarray,
    ):
        result = torch.from_numpy(
            np.asarray(
                heatmap,
                dtype=np.float32,
            ).copy()
        )

    else:
        raise TypeError(
            "Heatmap must be a torch.Tensor "
            "or NumPy array."
        )

    if result.ndim == 2:
        result = result.unsqueeze(0)

    if result.ndim != 3:
        raise ValueError(
            "Expected heatmap with shape [H,W] "
            "or [1,H,W]."
        )

    result = torch.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    minimum = result.amin(
        dim=(-2, -1),
        keepdim=True,
    )

    maximum = result.amax(
        dim=(-2, -1),
        keepdim=True,
    )

    result = (
        result - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    return result.clamp(
        0.0,
        1.0,
    )


# ============================================================================
# HEALTH / PRIORITY
# ============================================================================


def health_summary(
    engineering_result: Any,
) -> tuple[float | None, str]:
    """Extract road-health score and condition."""

    health = get_value(
        engineering_result,
        "road_health",
        None,
    )

    score = optional_float(
        get_value(
            health,
            "score",
            None,
        )
    )

    condition = readable_value(
        get_value(
            health,
            "condition",
            "Unknown",
        )
    )

    return (
        score,
        condition,
    )


def affected_percentage(
    engineering_result: Any,
    mask: np.ndarray,
) -> float:
    """Calculate percentage of image affected by detected defects."""

    health = get_value(
        engineering_result,
        "road_health",
        None,
    )

    ratio = optional_float(
        get_value(
            health,
            "defect_area_ratio",
            None,
        )
    )

    if ratio is None:
        ratio = optional_float(
            get_value(
                health,
                "area_ratio",
                None,
            )
        )

    if ratio is not None:
        return float(
            np.clip(
                ratio * 100.0,
                0.0,
                100.0,
            )
        )

    mask_array = np.asarray(
        mask
    )

    if mask_array.size == 0:
        return 0.0

    return float(
        np.count_nonzero(
            mask_array
        )
        / mask_array.size
        * 100.0
    )


def maintenance_priority(
    records: list[dict[str, Any]],
    affected: float,
) -> str:
    """Determine maintenance priority."""

    if any(
        item["priority"] == "Critical"
        for item in records
    ):
        return "Critical"

    if any(
        item["priority"] == "High"
        for item in records
    ):
        return "High"

    if affected >= 25:
        return "High"

    if affected >= 10:
        return "Moderate"

    if records:
        return "Moderate"

    return "Low"


def inspection_required(
    records: list[dict[str, Any]],
    condition: str,
    confidence: float,
) -> bool:
    """Determine whether field inspection should be prioritized."""

    serious = any(
        item["priority"]
        in {
            "High",
            "Critical",
        }
        for item in records
    )

    poor_condition = (
        condition.lower()
        in {
            "poor",
            "critical",
        }
    )

    low_confidence = (
        confidence < 0.65
    )

    return (
        serious
        or poor_condition
        or low_confidence
    )


# ============================================================================
# 3D
# ============================================================================


def generate_3d(
    inference: Any,
    prediction_mask: np.ndarray,
    engineering_result: Any,
    display_mode: str,
    visual_depth: float,
    show_markers: bool,
    show_boundaries: bool,
) -> dict[str, Any]:
    """Generate the interactive RoadXAI 3D visualization."""

    records = build_defect_records(
        engineering_result
    )

    mask = np.asarray(
        prediction_mask,
        dtype=np.uint8,
    )

    road_image = np.asarray(
        inference.original_image
    ).copy()

    probability_map = np.asarray(
        inference.probability_map,
        dtype=np.float32,
    )

    xai_heatmap = None

    xai_result = st.session_state.get(
        "xai_result"
    )

    if (
        xai_result is not None
        and st.session_state.get(
            "xai_image_key"
        )
        == st.session_state.get(
            "analysis_image_key"
        )
    ):
        xai_heatmap = xai_result.get(
            "heatmap"
        )

    try:
        supported = set(
            inspect.signature(
                generate_3d_visualization
            ).parameters
        )

    except Exception:
        supported = set()

    kwargs: dict[str, Any] = {
        "mask": mask,
        "max_depth": float(
            visual_depth
        ),
        "pixel_size": 1.0,
        "title": (
            "RoadXAI Interactive "
            "3D Inspection"
        ),
        "road_image": road_image,
    }

    optional_kwargs = {
        "engineering_result": engineering_result,
        "probability_map": probability_map,
        "xai_heatmap": xai_heatmap,
        "display_mode": display_mode,
        "show_markers": show_markers,
        "show_boundaries": show_boundaries,
        "vertical_exaggeration": float(
            visual_depth
        ),
        "confidence": safe_float(
            get_value(
                inference,
                "confidence",
                0.0,
            )
        ),
    }

    for key, value in optional_kwargs.items():
        if key in supported:
            kwargs[key] = value

    raw_result = generate_3d_visualization(
        **kwargs
    )

    if (
        isinstance(
            raw_result,
            tuple,
        )
        and len(raw_result) == 3
    ):
        heightmap_result = raw_result[0]
        mesh_result = raw_result[1]
        figure = raw_result[2]

        return {
            "heightmap": getattr(
                heightmap_result,
                "heightmap",
                None,
            ),
            "heightmap_result": (
                heightmap_result
            ),
            "mesh": mesh_result,
            "figure": figure,
            "records": records,
        }

    return {
        "figure": raw_result,
        "records": records,
    }


# ============================================================================
# REPORT
# ============================================================================


def render_report(
    inference: Any,
    engineering_result: Any,
    uploaded_file: Any,
    image_key: str,
) -> None:
    """Generate and display report controls."""

    st.subheader(
        "Inspection Report"
    )

    st.caption(
        "Generate JSON, TXT and PDF versions of the "
        "current RoadXAI assessment."
    )

    button_key = (
        f"generate_report_{image_key}"
    )

    if st.button(
        "Generate RoadXAI Report",
        type="primary",
        key=button_key,
    ):
        try:
            xai_result = st.session_state.get(
                "xai_result"
            )

            if (
                st.session_state.get(
                    "xai_image_key"
                )
                != image_key
            ):
                xai_result = None

            visualization_result = (
                st.session_state.get(
                    "visualization_result"
                )
            )

            REPORT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            filename = Path(
                getattr(
                    uploaded_file,
                    "name",
                    "road_inspection",
                )
            ).stem

            report_base = (
                REPORT_DIR / filename
            )

            with st.spinner(
                "Generating RoadXAI report..."
            ):
                report = generate_report(
                    engineering_result=engineering_result,
                    inference_result=inference,
                    xai_result=xai_result,
                    visualization_result=visualization_result,
                    image_name=getattr(
                        uploaded_file,
                        "name",
                        None,
                    ),
                    title=(
                        "RoadXAI Road "
                        "Inspection Report"
                    ),
                )

                json_path = save_json_report(
                    report,
                    report_base.with_suffix(
                        ".json"
                    ),
                )

                text_path = save_text_report(
                    report,
                    report_base.with_suffix(
                        ".txt"
                    ),
                )

                pdf_path = generate_pdf_report(
                    report,
                    report_base.with_suffix(
                        ".pdf"
                    ),
                )

            st.session_state[
                "report"
            ] = report

            st.session_state[
                "report_image_key"
            ] = image_key

            st.success(
                "Report generated successfully."
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.download_button(
                    "Download JSON",
                    data=json_path.read_bytes(),
                    file_name=json_path.name,
                    mime="application/json",
                    key=(
                        f"download_json_"
                        f"{image_key}"
                    ),
                )

            with c2:
                st.download_button(
                    "Download TXT",
                    data=text_path.read_bytes(),
                    file_name=text_path.name,
                    mime="text/plain",
                    key=(
                        f"download_txt_"
                        f"{image_key}"
                    ),
                )

            with c3:
                st.download_button(
                    "Download PDF",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=(
                        f"download_pdf_"
                        f"{image_key}"
                    ),
                )

        except Exception as exc:
            st.error(
                f"Report generation failed: {exc}"
            )

    report = st.session_state.get(
        "report"
    )

    if (
        report is not None
        and st.session_state.get(
            "report_image_key"
        )
        == image_key
    ):
        with st.expander(
            "Report Preview",
            expanded=False,
        ):
            st.text(
                render_report_text(
                    report
                )
            )


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main() -> None:
    """Run the RoadXAI Streamlit application."""

    create_app_header()

    # ------------------------------------------------------------------------
    # Model validation
    # ------------------------------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        st.error(
            "Model checkpoint not found:\n\n"
            f"{CHECKPOINT_PATH}"
        )
        st.stop()

    # ------------------------------------------------------------------------
    # Sidebar settings
    # ------------------------------------------------------------------------

    threshold, image_size = (
        create_settings_sidebar(
            default_threshold=0.50,
            default_image_size=256,
        )
    )

    # ------------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------------

    uploaded_file = (
        create_upload_section()
    )

    if uploaded_file is None:
        st.info(
            "Upload a road image to begin inspection."
        )
        return

    # ------------------------------------------------------------------------
    # 3D controls
    # ------------------------------------------------------------------------

    with st.sidebar:
        st.divider()

        st.subheader(
            "3D Inspection"
        )

        display_mode = st.selectbox(
            "Analysis mode",
            [
                "Inspection",
                "Road Surface",
                "Severity",
                "AI Confidence",
                "AI Attention",
            ],
            key="3d_display_mode",
        )

        visual_depth = st.slider(
            "Visual depth scale",
            min_value=0.20,
            max_value=2.00,
            value=0.90,
            step=0.10,
            help=(
                "Controls visualization exaggeration only. "
                "It is not physical pothole depth."
            ),
            key="3d_visual_depth",
        )

        show_markers = st.checkbox(
            "Show defect markers",
            value=True,
            key="3d_show_markers",
        )

        show_boundaries = st.checkbox(
            "Show defect boundaries",
            value=True,
            key="3d_show_boundaries",
        )

    # ------------------------------------------------------------------------
    # Image conversion
    # ------------------------------------------------------------------------

    try:
        image = uploaded_file_to_image(
            uploaded_file
        )

        if image is None:
            st.error(
                "Unable to read the uploaded image."
            )
            return

        if hasattr(
            image,
            "convert",
        ):
            image = image.convert(
                "RGB"
            )

        image_array = np.asarray(
            image
        )

        if (
            image_array.ndim != 3
            or image_array.shape[2] != 3
        ):
            raise ValueError(
                "Uploaded image must be RGB."
            )

        image_key = make_image_key(
            uploaded_file,
            image,
        )

        clear_stale_analysis_state(
            image_key
        )

    except Exception as exc:
        st.error(
            f"Image loading failed: {exc}"
        )
        return

    # ------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------

    try:
        model = load_roadxai_model()

        with st.spinner(
            "Analyzing road image..."
        ):
            inference = run_inference(
                model=model,
                image=image,
                device="cpu",
                image_size=(
                    image_size,
                    image_size,
                ),
                threshold=threshold,
            )

    except Exception as exc:
        st.error(
            f"Inference failed: {exc}"
        )
        return

    # ------------------------------------------------------------------------
    # Prediction mask
    # ------------------------------------------------------------------------

    try:
        prediction_mask = np.squeeze(
            np.asarray(
                inference.prediction_mask
            )
        )

        if prediction_mask.ndim != 2:
            raise ValueError(
                "Model prediction mask must be 2D."
            )

        prediction_mask = (
            prediction_mask > 0
        ).astype(
            np.uint8
        )

    except Exception as exc:
        st.error(
            f"Invalid model mask: {exc}"
        )
        return

    # ------------------------------------------------------------------------
    # Engineering analysis
    # ------------------------------------------------------------------------

    try:
        road_area_pixels = int(
            prediction_mask.shape[0]
            * prediction_mask.shape[1]
        )

        engineering_result = analyze_road(
            mask=prediction_mask,
            road_area_pixels=road_area_pixels,
            calibration=None,
            repair_rate_per_m2=None,
            currency="INR",
            min_area_pixels=10.0,
        )

    except Exception as exc:
        st.error(
            f"Engineering analysis failed: {exc}"
        )
        return

    # ------------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------------

    records = build_defect_records(
        engineering_result
    )

    confidence = safe_float(
        get_value(
            inference,
            "confidence",
            0.0,
        )
    )

    health_score, condition = (
        health_summary(
            engineering_result
        )
    )

    affected = affected_percentage(
        engineering_result,
        prediction_mask,
    )

    priority = maintenance_priority(
        records,
        affected,
    )

    needs_inspection = (
        inspection_required(
            records,
            condition,
            confidence,
        )
    )

    # ------------------------------------------------------------------------
    # Numbered overlay
    # ------------------------------------------------------------------------

    try:
        numbered_overlay = (
            make_numbered_defect_overlay(
                inference.original_image,
                prediction_mask,
                engineering_result,
            )
        )

    except Exception as exc:
        st.warning(
            "Could not create numbered overlay: "
            f"{exc}"
        )

        numbered_overlay = np.asarray(
            inference.original_image
        ).copy()

    # =========================================================================
    # TABS
    # =========================================================================

    inspection_tab, advanced_tab = st.tabs(
        [
            "Road Inspection",
            "Advanced Analysis",
        ]
    )

    # =========================================================================
    # ROAD INSPECTION
    # =========================================================================

    with inspection_tab:

        st.header(
            "Road Inspection"
        )

        st.caption(
            "AI-assisted screening of road defects "
            "from the uploaded image."
        )

        # ---------------------------------------------------------------------
        # Main images
        # ---------------------------------------------------------------------

        image_col, overlay_col = (
            st.columns(2)
        )

        with image_col:
            display_image(
                inference.original_image,
                caption="Original road image",
            )

        with overlay_col:
            display_image(
                numbered_overlay,
                caption=(
                    "Detected defects — "
                    "numbers match the measurements"
                ),
            )

        # ---------------------------------------------------------------------
        # Summary metrics
        # ---------------------------------------------------------------------

        st.subheader(
            "Inspection Summary"
        )

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )

        metric1.metric(
            "Detected Defects",
            f"{len(records):,}",
        )

        metric2.metric(
            "Affected Area",
            f"{affected:.2f}%",
        )

        metric3.metric(
            "Health Score",
            (
                f"{health_score:.2f}/100"
                if health_score is not None
                else "N/A"
            ),
        )

        metric4.metric(
            "Model Confidence",
            f"{confidence:.2%}",
        )

        # ---------------------------------------------------------------------
        # Condition
        # ---------------------------------------------------------------------

        condition_text = (
            condition.upper()
            if condition
            else "UNKNOWN"
        )

        st.info(
            f"Road condition: {condition_text}"
        )

        if needs_inspection:
            st.warning(
                "Field inspection is recommended "
                "before maintenance decisions."
            )
        else:
            st.success(
                "No immediate high-priority "
                "inspection trigger was identified."
            )

        # ---------------------------------------------------------------------
        # Defect measurements
        # ---------------------------------------------------------------------

        st.subheader(
            "Detected Defects"
        )

        if not records:
            st.success(
                "No defects detected at the selected threshold."
            )

        else:
            st.caption(
                "Measurements are in pixels because no "
                "pixel-to-meter calibration was supplied."
            )

            for item in records:

                with st.expander(
                    (
                        f"Defect {item['id']} — "
                        f"{item['priority']} "
                        f"(score {item['score']:.2f})"
                    ),
                    expanded=(
                        item["id"] == 1
                    ),
                ):
                    d1, d2, d3 = (
                        st.columns(3)
                    )

                    d1.metric(
                        "Area",
                        f"{item['area_pixels']:.2f} px²",
                    )

                    d2.metric(
                        "Length",
                        f"{item['length_pixels']:.2f} px",
                    )

                    d3.metric(
                        "Width",
                        f"{item['width_pixels']:.2f} px",
                    )

                    d4, d5, d6 = (
                        st.columns(3)
                    )

                    d4.metric(
                        "Centroid X",
                        f"{item['centroid_x']:.2f}",
                    )

                    d5.metric(
                        "Centroid Y",
                        f"{item['centroid_y']:.2f}",
                    )

                    d6.metric(
                        "Severity",
                        item["severity"].upper(),
                    )

                    if item["explanation"]:
                        st.caption(
                            item["explanation"]
                        )

        # ---------------------------------------------------------------------
        # Basic explanation
        # ---------------------------------------------------------------------

        with st.expander(
            "What do these results mean?",
            expanded=False,
        ):
            st.write(
                "RoadXAI first segments pixels that appear "
                "to belong to road defects. Connected regions "
                "are then measured individually."
            )

            st.write(
                "Area describes how many image pixels belong "
                "to a detected region. Length and width describe "
                "its estimated geometric dimensions in the image."
            )

            st.write(
                "The health score is an image-based indicator. "
                "It is not a structural safety certification."
            )

            st.write(
                "Physical meters, square meters and repair costs "
                "require validated calibration information."
            )

    # =========================================================================
    # ADVANCED ANALYSIS
    # =========================================================================

    with advanced_tab:

        st.header(
            "Advanced Analysis"
        )

        # ---------------------------------------------------------------------
        # Inference summary
        # ---------------------------------------------------------------------

        display_inference_summary(
            inference
        )

        # ---------------------------------------------------------------------
        # Probability map
        # ---------------------------------------------------------------------

        st.subheader(
            "Defect Probability"
        )

        probability_visuals = (
            make_probability_visualization(
                inference.probability_map,
                inference.original_image,
            )
        )

        p1, p2, p3 = st.columns(3)

        with p1:
            display_image(
                probability_visuals[
                    "grayscale"
                ],
                caption="Probability intensity",
            )

        with p2:
            display_image(
                probability_visuals[
                    "color"
                ],
                caption="Probability heatmap",
            )

        with p3:
            display_image(
                probability_visuals[
                    "overlay"
                ],
                caption="Probability over original image",
            )

        st.caption(
            f"Segmentation threshold: {threshold:.2f}. "
            "Higher probability means stronger model confidence "
            "that a pixel belongs to a defect."
        )

        # ---------------------------------------------------------------------
        # Engineering summary
        # ---------------------------------------------------------------------

        st.subheader(
            "Engineering Analysis"
        )

        e1, e2, e3, e4 = (
            st.columns(4)
        )

        e1.metric(
            "Defects",
            f"{len(records):,}",
        )

        e2.metric(
            "Affected Area",
            f"{affected:.2f}%",
        )

        e3.metric(
            "Condition",
            condition.upper(),
        )

        e4.metric(
            "Priority",
            priority.upper(),
        )

        # ---------------------------------------------------------------------
        # Severity distribution
        # ---------------------------------------------------------------------

        severity_counts = {
            "Low": 0,
            "Moderate": 0,
            "High": 0,
            "Critical": 0,
        }

        for item in records:
            level = item[
                "priority"
            ]

            if level in severity_counts:
                severity_counts[
                    level
                ] += 1

        st.write(
            "### Severity Distribution"
        )

        for level, count in severity_counts.items():
            st.write(
                f"**{level}:** {count}"
            )

        # ---------------------------------------------------------------------
        # Grad-CAM
        # ---------------------------------------------------------------------

        st.subheader(
            "Explainable AI"
        )

        st.caption(
            "Grad-CAM highlights image regions that contributed "
            "to the model prediction. It explains model attention; "
            "it does not measure physical damage."
        )

        if st.button(
            "Generate Grad-CAM Explanation",
            key=f"generate_xai_{image_key}",
        ):
            try:
                with st.spinner(
                    "Generating Grad-CAM..."
                ):
                    xai_result = run_xai(
                        model,
                        inference.input_tensor,
                    )

                st.session_state[
                    "xai_result"
                ] = xai_result

                st.session_state[
                    "xai_image_key"
                ] = image_key

                st.success(
                    "Grad-CAM explanation generated."
                )

            except Exception as exc:
                st.error(
                    f"XAI generation failed: {exc}"
                )

        xai_result = st.session_state.get(
            "xai_result"
        )

        if (
            xai_result is not None
            and st.session_state.get(
                "xai_image_key"
            )
            == image_key
        ):

            x1, x2 = st.columns(2)

            with x1:
                display_image(
                    xai_result["heatmap"],
                    caption=(
                        "Grad-CAM attention"
                    ),
                )

            with x2:
                display_image(
                    xai_result["overlay"],
                    caption=(
                        "Grad-CAM overlay"
                    ),
                )

            heatmap = np.asarray(
                xai_result["heatmap"],
                dtype=np.float32,
            )

            a1, a2, a3 = (
                st.columns(3)
            )

            a1.metric(
                "Mean Attention",
                f"{float(heatmap.mean()):.3f}",
            )

            a2.metric(
                "Maximum Attention",
                f"{float(heatmap.max()):.3f}",
            )

            a3.metric(
                "Pixels > 0.5",
                f"{int(np.count_nonzero(heatmap >= 0.5)):,}",
            )

            target_layer = xai_result[
                "target_layer"
            ]

            st.caption(
                "Method: Grad-CAM | "
                "Target layer: "
                f"{target_layer.__class__.__name__}"
            )

        else:
            st.info(
                "Generate Grad-CAM to inspect model attention."
            )

        # ---------------------------------------------------------------------
        # 3D visualization
        # ---------------------------------------------------------------------

        st.subheader(
            "Interactive 3D Visualization"
        )

        st.caption(
            "The 3D model is aligned with the uploaded image "
            "and segmentation mask. Vertical geometry is "
            "normalized visualization depth, not physical depth."
        )

        current_mask = np.asarray(
            prediction_mask,
            dtype=np.uint8,
        )

        current_3d_key = (
            f"{image_key}|"
            f"{current_mask.shape}|"
            f"{int(np.count_nonzero(current_mask))}|"
            f"{float(current_mask.mean()):.8f}|"
            f"{display_mode}|"
            f"{visual_depth}|"
            f"{show_markers}|"
            f"{show_boundaries}"
        )

        if st.session_state.get(
            "visualization_key"
        ) != current_3d_key:
            st.session_state.pop(
                "visualization_result",
                None,
            )

        if st.button(
            "Generate 3D Defect Visualization",
            type="primary",
            key=f"generate_3d_{image_key}",
        ):
            try:
                st.session_state.pop(
                    "visualization_result",
                    None,
                )

                with st.spinner(
                    "Building interactive 3D inspection model..."
                ):
                    result = generate_3d(
                        inference,
                        current_mask,
                        engineering_result,
                        display_mode,
                        visual_depth,
                        show_markers,
                        show_boundaries,
                    )

                st.session_state[
                    "visualization_result"
                ] = result

                st.session_state[
                    "visualization_key"
                ] = current_3d_key

                st.success(
                    "3D visualization generated."
                )

            except Exception as exc:
                st.error(
                    f"3D visualization failed: {exc}"
                )

        visualization_result = (
            st.session_state.get(
                "visualization_result"
            )
        )

        if (
            visualization_result is not None
            and st.session_state.get(
                "visualization_key"
            )
            == current_3d_key
        ):

            figure = visualization_result.get(
                "figure"
            )

            if figure is not None:
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "responsive": True,
                    },
                    key=(
                        f"plot3d_{image_key}"
                    ),
                )

            heightmap = visualization_result.get(
                "heightmap"
            )

            mesh = visualization_result.get(
                "mesh"
            )

            m1, m2, m3, m4 = (
                st.columns(4)
            )

            if heightmap is not None:
                m1.metric(
                    "3D Grid",
                    (
                        f"{heightmap.shape[0]} × "
                        f"{heightmap.shape[1]}"
                    ),
                )

                m4.metric(
                    "Max Visual Depth",
                    f"{float(np.max(heightmap)):.3f}",
                )

            else:
                m1.metric(
                    "3D Grid",
                    "N/A",
                )

            if mesh is not None:
                m2.metric(
                    "Vertices",
                    f"{len(mesh.vertices):,}",
                )

                m3.metric(
                    "Faces",
                    f"{len(mesh.faces):,}",
                )

            else:
                m2.metric(
                    "Vertices",
                    "N/A",
                )

                m3.metric(
                    "Faces",
                    "N/A",
                )

            st.caption(
                "Drag to rotate. Scroll to zoom. "
                "Right-drag to pan. Use the Plotly controls "
                "to reset the camera or export the visualization."
            )

        else:
            st.info(
                "Generate the 3D model to inspect detected "
                "damage spatially."
            )

        # ---------------------------------------------------------------------
        # Road decision summary
        # ---------------------------------------------------------------------

        st.subheader(
            "Road Decision Summary"
        )

        r1, r2 = st.columns(2)

        with r1:
            st.write(
                f"**Condition:** {condition}"
            )

            st.write(
                (
                    f"**Health score:** "
                    f"{health_score:.2f}/100"
                    if health_score is not None
                    else "**Health score:** N/A"
                )
            )

            st.write(
                f"**Affected road area:** "
                f"{affected:.2f}%"
            )

        with r2:
            st.write(
                f"**Maintenance priority:** "
                f"{priority}"
            )

            st.write(
                "**Field inspection:** "
                f"{'Required' if needs_inspection else 'Not immediately indicated'}"
            )

            st.write(
                f"**Model confidence:** "
                f"{confidence:.2%}"
            )

        # ---------------------------------------------------------------------
        # Report
        # ---------------------------------------------------------------------

        st.divider()

        render_report(
            inference,
            engineering_result,
            uploaded_file,
            image_key,
        )

    # =========================================================================
    # FOOTER
    # =========================================================================

    st.divider()

    st.caption(
        "RoadXAI is an AI-assisted road-inspection "
        "screening and decision-support system. "
        "Automated results should be validated through "
        "appropriate field inspection before engineering "
        "or maintenance decisions."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()
