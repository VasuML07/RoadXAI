
"""
RoadXAI Streamlit Application.

End-to-end image analysis interface for:
- road-defect segmentation
- probability visualization
- engineering measurements
- Grad-CAM explanation
- image-aligned 3D visualization
- report generation

Scope:
This application provides image-based inspection support. It does not
measure physical defect depth, certify road safety, determine structural
load capacity, or provide a guaranteed repair diagnosis/cost.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "crack500" / "best.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# RoadXAI imports
# ---------------------------------------------------------------------------

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

from engineering_analysis.module import analyze_road

from explainable_ai.module import (
    generate_heatmap,
    heatmap_to_numpy,
    overlay_heatmap,
    get_default_target_layer,
)

from report_generation.module import generate_report

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


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "crack500"
    / "best.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"


st.set_page_config(
    page_title="RoadXAI",
    page_icon="RoadXAI",
    layout="wide",
)


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


def prepare_xai_tensor(
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Create a normal CPU tensor safe for Grad-CAM autograd."""
    if not isinstance(
        input_tensor,
        torch.Tensor,
    ):
        raise TypeError(
            "Inference input must be a torch.Tensor."
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


def readable_value(value: Any) -> str:
    if value is None:
        return "Unknown"

    if hasattr(value, "value"):
        value = value.value

    return str(value).replace("_", " ")


def make_image_key(uploaded_file: Any, image: Any) -> str:
    if hasattr(image, "convert"):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Uploaded image must be an RGB image.")

    digest = hashlib.sha256(
        array.astype(np.uint8).tobytes()
    ).hexdigest()[:16]

    return f"{uploaded_file.name}|{digest}|{array.shape}"


def clear_image_state(image_key: str) -> None:
    previous = st.session_state.get("roadxai_image_key")

    if previous == image_key:
        return

    st.session_state["roadxai_image_key"] = image_key

    for key in (
        "xai_result",
        "xai_image_key",
        "visualization_result",
        "visualization_key",
        "report",
        "report_image_key",
    ):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Defect records
# ---------------------------------------------------------------------------

def build_defect_records(
    engineering_result: Any,
) -> list[dict[str, Any]]:
    defects = list(
        get_value(engineering_result, "defects", []) or []
    )

    severities = list(
        get_value(engineering_result, "severity", []) or []
    )

    records: list[dict[str, Any]] = []

    for index, defect in enumerate(defects, start=1):
        severity = (
            severities[index - 1]
            if index - 1 < len(severities)
            else None
        )

        level = readable_value(
            get_value(severity, "level", "Unknown")
        )

        score = safe_float(
            get_value(severity, "score", 0.0)
        )

        level_lower = level.lower()

        if "critical" in level_lower or score >= 75:
            priority = "Critical"
        elif "high" in level_lower or score >= 50:
            priority = "High"
        elif "moderate" in level_lower or score >= 25:
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
                    get_value(defect, "area_pixels", 0.0)
                ),
                "length_pixels": safe_float(
                    get_value(defect, "length_pixels", 0.0)
                ),
                "width_pixels": safe_float(
                    get_value(defect, "width_pixels", 0.0)
                ),
                "centroid_x": safe_float(
                    get_value(defect, "centroid_x", 0.0)
                ),
                "centroid_y": safe_float(
                    get_value(defect, "centroid_y", 0.0)
                ),
                "area_m2": optional_float(
                    get_value(defect, "area_m2", None)
                ),
                "length_m": optional_float(
                    get_value(defect, "length_m", None)
                ),
                "width_m": optional_float(
                    get_value(defect, "width_m", None)
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


# ---------------------------------------------------------------------------
# Numbered overlay
# ---------------------------------------------------------------------------

def make_numbered_defect_overlay(
    original_image: np.ndarray,
    prediction_mask: np.ndarray,
    engineering_result: Any,
) -> np.ndarray:
    image = np.asarray(
        original_image,
        dtype=np.uint8,
    ).copy()

    mask = np.squeeze(
        np.asarray(prediction_mask)
    )

    if mask.ndim != 2:
        raise ValueError("prediction_mask must be 2D.")

    image_h, image_w = image.shape[:2]
    mask_h, mask_w = mask.shape

    resized_mask = cv2.resize(
        (mask > 0).astype(np.uint8),
        (image_w, image_h),
        interpolation=cv2.INTER_NEAREST,
    )

    result = image.copy()

    # Use the existing overlay implementation for the mask.
    result = create_mask_overlay(
        result,
        resized_mask,
        alpha=0.32,
    )

    defects = list(
        get_value(
            engineering_result,
            "defects",
            [],
        )
        or []
    )

    for index, defect in enumerate(defects, start=1):
        x = safe_float(
            get_value(defect, "centroid_x", 0.0)
        )
        y = safe_float(
            get_value(defect, "centroid_y", 0.0)
        )

        x = int(round(x * image_w / max(mask_w, 1)))
        y = int(round(y * image_h / max(mask_h, 1)))

        x = max(20, min(image_w - 20, x))
        y = max(20, min(image_h - 20, y))

        radius = 18 if image_w >= 700 else 14

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

        text = str(index)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55 if index < 10 else 0.43
        thickness = 2

        text_size, _ = cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )

        text_w, text_h = text_size

        cv2.putText(
            result,
            text,
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


# ---------------------------------------------------------------------------
# XAI
# ---------------------------------------------------------------------------

def run_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor.")

    # Inference is executed under torch.inference_mode() in the
    # Streamlit inference helper. Grad-CAM needs a fresh normal tensor
    # that can participate in autograd.
    input_array = (
        input_tensor
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    xai_input = torch.tensor(
        input_array,
        dtype=torch.float32,
        device="cpu",
        requires_grad=True,
    )

    target_layer = get_default_target_layer(model)

    heatmap = generate_heatmap(
        model=model,
        image=xai_input,
        target_layer=target_layer,
        method="gradcam",
        target_class=0,
    )

    heatmap_array = heatmap_to_numpy(heatmap)
    heatmap_array = np.asarray(
        heatmap_array,
        dtype=np.float32,
    )

    if heatmap_array.ndim == 3:
        heatmap_array = heatmap_array[0]

    heatmap_array = np.nan_to_num(
        heatmap_array,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    minimum = float(heatmap_array.min())
    maximum = float(heatmap_array.max())

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

    input_image = (
        xai_input[0]
        .detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    input_image = np.clip(
        input_image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # Keep the heatmap tensor shape compatible with the XAI module.
    heatmap_for_overlay = heatmap_array

    overlay = overlay_heatmap(
        input_image,
        heatmap_for_overlay,
        alpha=0.45,
    )

    return {
        "method": "Grad-CAM",
        "target_layer": str(target_layer),
        "heatmap": heatmap_array,
        "overlay": overlay,
    }


# ---------------------------------------------------------------------------
# Health / priority
# ---------------------------------------------------------------------------

def health_summary(
    engineering_result: Any,
) -> tuple[float | None, str]:
    health = get_value(
        engineering_result,
        "road_health",
        None,
    )

    score = optional_float(
        get_value(health, "score", None)
    )

    condition = readable_value(
        get_value(health, "condition", "Unknown")
    )

    return score, condition


def affected_percentage(
    engineering_result: Any,
    mask: np.ndarray,
) -> float:
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

    return float(
        np.count_nonzero(mask)
        / max(mask.size, 1)
        * 100.0
    )


def maintenance_priority(
    records: list[dict[str, Any]],
    affected: float,
) -> str:
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
    serious = any(
        item["priority"]
        in {"High", "Critical"}
        for item in records
    )

    poor_condition = (
        condition.lower()
        in {"poor", "critical"}
    )

    low_confidence = confidence < 0.65

    return (
        serious
        or poor_condition
        or low_confidence
    )


# ---------------------------------------------------------------------------
# 3D helpers
# ---------------------------------------------------------------------------

def generate_3d(
    inference: Any,
    prediction_mask: np.ndarray,
    engineering_result: Any,
    display_mode: str,
    visual_depth: float,
    show_markers: bool,
    show_boundaries: bool,
) -> dict[str, Any]:
    records = build_defect_records(
        engineering_result
    )

    mask = np.asarray(prediction_mask, dtype=np.uint8)
    road_image = np.asarray(inference.original_image).copy()
    probability_map = np.asarray(inference.probability_map, dtype=np.float32)
    xai_heatmap = (
        st.session_state.get("xai_result", {}).get("heatmap")
        if st.session_state.get("xai_image_key")
        == st.session_state.get("roadxai_image_key")
        else None
    )

    try:
        import inspect
        supported = set(inspect.signature(generate_3d_visualization).parameters)
    except Exception:
        supported = set()

    kwargs = {
        "mask": mask,
        "max_depth": visual_depth,
        "pixel_size": 1.0,
        "title": "RoadXAI Interactive 3D Inspection",
        "road_image": road_image,
    }
    optional_kwargs = {
        "engineering_result": engineering_result,
        "probability_map": probability_map,
        "xai_heatmap": xai_heatmap,
        "display_mode": display_mode,
        "show_markers": show_markers,
        "show_boundaries": show_boundaries,
    }
    for key, value in optional_kwargs.items():
        if key in supported:
            kwargs[key] = value

    raw = generate_3d_visualization(**kwargs)

    if isinstance(raw, tuple) and len(raw) == 3:
        heightmap_result, mesh_result, figure = raw

        return {
            "heightmap": heightmap_result.heightmap,
            "heightmap_result": heightmap_result,
            "mesh": mesh_result,
            "figure": figure,
            "records": records,
        }

    return {
        "figure": raw,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@st.cache_resource
def load_roadxai_model():
    model = build_model(
        base_channels=16
    )

    return load_model(
        model,
        CHECKPOINT_PATH,
        device="cpu",
    )


def render_report(
    inference: Any,
    engineering_result: Any,
    uploaded_file: Any,
    image_key: str,
) -> None:
    st.subheader("Inspection Report")

    st.caption(
        "Generate one report for the current image. "
        "The report includes detection, engineering, XAI and 3D results."
    )

    if st.button(
        "Generate RoadXAI Report",
        type="primary",
        key=f"generate_report_{image_key}",
    ):
        try:
            xai_result = st.session_state.get(
                "xai_result"
            )

            if st.session_state.get(
                "xai_image_key"
            ) != image_key:
                xai_result = None

            visualization_result = st.session_state.get(
                "visualization_result"
            )

            REPORT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            report_base = (
                REPORT_DIR
                / Path(
                    uploaded_file.name
                ).stem
            )

            with st.spinner(
                "Generating report..."
            ):
                report = generate_report(
                    engineering_result=engineering_result,
                    inference_result=inference,
                    xai_result=xai_result,
                    visualization_result=visualization_result,
                    image_name=uploaded_file.name,
                    title="RoadXAI Road Inspection Report",
                )

                json_path = save_json_report(
                    report,
                    report_base.with_suffix(".json"),
                )

                text_path = save_text_report(
                    report,
                    report_base.with_suffix(".txt"),
                )

                pdf_path = generate_pdf_report(
                    report,
                    report_base.with_suffix(".pdf"),
                )

            st.session_state["report"] = report
            st.session_state["report_image_key"] = image_key

            st.success("Report generated.")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.download_button(
                    "Download JSON",
                    data=json_path.read_bytes(),
                    file_name=json_path.name,
                    mime="application/json",
                    key=f"download_json_{image_key}",
                )

            with c2:
                st.download_button(
                    "Download TXT",
                    data=text_path.read_bytes(),
                    file_name=text_path.name,
                    mime="text/plain",
                    key=f"download_txt_{image_key}",
                )

            with c3:
                st.download_button(
                    "Download PDF",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=f"download_pdf_{image_key}",
                )

        except Exception as exc:
            st.error(
                f"Report generation failed: {exc}"
            )

    report = st.session_state.get("report")

    if (
        report is not None
        and st.session_state.get(
            "report_image_key"
        ) == image_key
    ):
        with st.expander(
            "Report Preview",
            expanded=False,
        ):
            st.text(
                render_report_text(report)
            )


def main() -> None:
    create_app_header()

    st.sidebar.subheader("Model")

    model_name = st.sidebar.selectbox(
        "Defect model",
        options=list(CHECKPOINTS.keys()),
        index=0,
        help=(
            "Select which trained segmentation model should analyze "
            "the uploaded image."
        ),
    )

    checkpoint_path = CHECKPOINTS[model_name]

    if not checkpoint_path.is_file():
        st.error(
            "Model checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )
        st.stop()

    threshold, image_size = (
        create_settings_sidebar(
            default_threshold=0.50,
            default_image_size=256,
        )
    )

    uploaded_file = create_upload_section()

    if uploaded_file is None:
        st.info(
            "Upload a road image to begin inspection."
        )
        return

    # -----------------------------------------------------------------------
    # Sidebar: 3D controls
    # -----------------------------------------------------------------------

    with st.sidebar:
        st.divider()
        st.subheader("3D Inspection")

        display_mode = st.selectbox(
            "Color / analysis mode",
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
            "Visual depth exaggeration",
            0.20,
            2.00,
            0.90,
            0.10,
            help=(
                "Visualization scale only. "
                "This is not physical pothole depth."
            ),
            key="3d_visual_depth",
        )

        show_markers = st.checkbox(
            "Show defect markers",
            True,
            key="3d_show_markers",
        )

        show_boundaries = st.checkbox(
            "Show defect boundaries",
            True,
            key="3d_show_boundaries",
        )

    # -----------------------------------------------------------------------
    # Image
    # -----------------------------------------------------------------------

    try:
        image = uploaded_file_to_image(
            uploaded_file
        )

        if image is None:
            st.error("Unable to read the image.")
            return

        if hasattr(image, "convert"):
            image = image.convert("RGB")

        image_key = make_image_key(
            uploaded_file,
            image,
        )

        clear_image_state(
            image_key
        )

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

    prediction_mask = np.squeeze(
        np.asarray(
            inference.prediction_mask
        )
    )

    if prediction_mask.ndim != 2:
        st.error(
            "The model returned an invalid segmentation mask."
        )
        return

    # -----------------------------------------------------------------------
    # Engineering analysis
    # -----------------------------------------------------------------------

    try:
        engineering_result = analyze_road(
            mask=prediction_mask,
            road_area_pixels=int(
                prediction_mask.size
            ),
            calibration=None,
            repair_rate_per_m2=None,
            currency="INR",
            min_area_pixels=10.0,
        )

        numbered_overlay = (
            make_numbered_defect_overlay(
                inference.original_image,
                prediction_mask,
                engineering_result,
            )
        )

        # -----------------------------------------------------
        # Road Defect Analysis
        # -----------------------------------------------------

        st.subheader(
            "Road Defect Analysis"
        )

        col1, col2 = st.columns(2)

        with col1:
            display_image(
                inference.original_image,
                caption="Original Road Image",
            )

        with col2:
            display_image(
                numbered_overlay,
                caption=(
                    "Detected Defect Overlay — "
                    "Numbers match measurements"
                ),
            )

        st.caption(
            "Numbers identify separate detected regions. "
            "They match the engineering measurements and "
            "severity assessment below."
        )

        # -----------------------------------------------------
        # Inference Summary
        # -----------------------------------------------------

        st.subheader(
            "Inference Summary"
        )

        display_inference_summary(
            inference
        )

        # -----------------------------------------------------
        # Probability Map
        # -----------------------------------------------------

        probability_visuals = (
            make_probability_visualization(
                inference.probability_map,
                inference.original_image,
            )
        )

        probability = np.asarray(
            inference.probability_map,
            dtype=np.float32,
        )

    # -----------------------------------------------------------------------
    # Top-level tabs
    # -----------------------------------------------------------------------

    inspection_tab, advanced_tab = st.tabs(
        [
            "Road Inspection",
            "Advanced Analysis",
        ]
    )

    # -----------------------------------------------------------------------
    # Road Inspection
    # -----------------------------------------------------------------------

    with inspection_tab:

        st.header("Road Inspection")

        st.metric(
            "Road Condition",
            condition,
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Overall Health Score",
            (
                f"{health_score:.1f}/100"
                if health_score is not None
                else "N/A"
            ),
        )

        m2.metric(
            "Number of Defects",
            len(records),
        )

        m3.metric(
            "High / Critical",
            serious_count,
        )

        m4.metric(
            "Road Affected",
            f"{affected:.2f}%",
        )

        p1, p2 = st.columns(2)

        p1.metric(
            "Maintenance Priority",
            priority,
        )

        p2.metric(
            "Model Confidence",
            f"{confidence:.1%}",
        )

        if needs_inspection:
            st.warning(
                "Field inspection is recommended based on "
                "the current image-analysis result."
            )
        else:
            st.success(
                "No immediate field-inspection trigger was "
                "identified from this image."
            )

        st.divider()

        st.subheader(
            "Serious Defect Locations"
        )

        st.caption(
            "Defect numbers identify exact image locations. "
            "Coordinates are image pixels, not geographic GPS coordinates."
        )

        c1, c2 = st.columns(2)

        with c1:
            display_image(
                inference.original_image,
                caption="Original road image",
            )

        with c2:
            display_image(
                numbered_overlay,
                caption="Numbered defect map",
            )

        if records:
            st.subheader(
                "Detected Defects"
            )

            selected_id = st.selectbox(
                "Inspect defect",
                [
                    record["id"]
                    for record in records
                ],
                format_func=lambda x: (
                    f"Defect #{x}"
                ),
                key=f"defect_selector_{image_key}",
            )

            selected = next(
                item
                for item in records
                if item["id"] == selected_id
            )

            a1, a2, a3, a4 = st.columns(4)

            a1.metric(
                "Severity",
                selected["severity"].upper(),
            )

            a2.metric(
                "Severity Score",
                f"{selected['score']:.1f}/100",
            )

            a3.metric(
                "Area",
                f"{selected['area_pixels']:,.0f} px²",
            )

            a4.metric(
                "Priority",
                selected["priority"],
            )

            b1, b2 = st.columns(2)

            with b1:
                st.write(
                    f"**Image location:** "
                    f"({selected['centroid_x']:.1f}, "
                    f"{selected['centroid_y']:.1f}) px"
                )

                st.write(
                    f"**Length:** "
                    f"{selected['length_pixels']:.1f} px"
                )

                st.write(
                    f"**Width:** "
                    f"{selected['width_pixels']:.1f} px"
                )

            with b2:
                if selected["area_m2"] is not None:
                    st.write(
                        f"**Physical area:** "
                        f"{selected['area_m2']:.4f} m²"
                    )
                else:
                    st.write(
                        "**Physical area:** Not calibrated"
                    )

                if selected["explanation"]:
                    st.write(
                        selected["explanation"]
                    )

        else:
            st.info(
                "No defects passed the current segmentation threshold."
            )

        st.divider()

        # -------------------------------------------------------------------
        # 3D
        # -------------------------------------------------------------------

        st.header(
            "Interactive 3D Road Inspection"
        )

        st.info(
            "The vertical axis is normalized visualization geometry. "
            "It is not a physical depth measurement."
        )

        visualization_key = (
            f"{image_key}|"
            f"{display_mode}|"
            f"{visual_depth:.2f}|"
            f"{show_markers}|"
            f"{show_boundaries}"
        )

        if st.button(
            "Build / Refresh 3D Model",
            type="primary",
            key=f"build_3d_{image_key}",
        ):
            try:
                with st.spinner(
                    "Building interactive 3D inspection model..."
                ):
                    result = generate_3d(
                        inference,
                        prediction_mask,
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
                ] = visualization_key

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
            ) == visualization_key
        ):

            figure = visualization_result[
                "figure"
            ]

            st.plotly_chart(
                figure,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True,
                },
                key=f"plot3d_{image_key}",
            )

            heightmap = (
                visualization_result.get(
                    "heightmap"
                )
            )

            mesh = (
                visualization_result.get(
                    "mesh"
                )
            )

            s1, s2, s3, s4 = st.columns(4)

            if heightmap is not None:
                s1.metric(
                    "3D Grid",
                    (
                        f"{heightmap.shape[0]} × "
                        f"{heightmap.shape[1]}"
                    ),
                )
            else:
                s1.metric(
                    "3D Grid",
                    "N/A",
                )

            if mesh is not None:
                s2.metric(
                    "Vertices",
                    f"{len(mesh.vertices):,}",
                )

                s3.metric(
                    "Faces",
                    f"{len(mesh.faces):,}",
                )

            if heightmap is not None:
                s4.metric(
                    "Max Visual Depth",
                    f"{float(np.max(heightmap)):.3f}",
                )

            st.caption(
                "Interaction: drag to rotate, scroll to zoom, "
                "right-drag to pan, and use the Plotly modebar "
                "for camera/reset/export controls."
            )

        else:
            st.caption(
                "Build the 3D model to inspect the detected "
                "damage spatially."
            )

        st.divider()

    # -----------------------------------------------------------------------
    # Advanced Analysis
    # -----------------------------------------------------------------------

    with advanced_tab:

        st.header(
            "Advanced Analysis"
        )

        display_inference_summary(
            inference
        )

        st.subheader(
            "Segmentation"
        )

        c1, c2 = st.columns(2)

        with c1:
            display_image(
                inference.original_image,
                caption="Original image",
            )

        with c2:
            display_image(
                numbered_overlay,
                caption="Numbered segmentation",
            )

        st.subheader(
            "Probability Map"
        )

        probability = np.squeeze(
            np.asarray(
                inference.probability_map,
                dtype=np.float32,
            )
        )

        probability = np.clip(
            probability,
            0.0,
            1.0,
        )

        prob_u8 = (
            probability * 255.0
        ).astype(np.uint8)

        prob_color = cv2.applyColorMap(
            prob_u8,
            cv2.COLORMAP_TURBO,
        )

        prob_color = cv2.cvtColor(
            prob_color,
            cv2.COLOR_BGR2RGB,
        )

        if prob_color.shape[:2] != (
            inference.original_image.shape[:2]
        ):
            prob_color = cv2.resize(
                prob_color,
                (
                    inference.original_image.shape[1],
                    inference.original_image.shape[0],
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        prob_overlay = cv2.addWeighted(
            inference.original_image,
            0.55,
            prob_color,
            0.45,
            0,
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            display_image(
                prob_u8,
                caption="Probability intensity",
            )

        with q2:
            display_image(
                prob_color,
                caption="Probability heatmap",
            )

        with q3:
            display_image(
                prob_overlay,
                caption="Probability overlay",
            )

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Mean",
            f"{float(probability.mean()):.3f}",
        )

        q2.metric(
            "Maximum",
            f"{float(probability.max()):.3f}",
        )

        q3.metric(
            "Above Threshold",
            f"{int(np.count_nonzero(probability >= threshold)):,}",
        )

        q4.metric(
            "Threshold",
            f"{threshold:.2f}",
        )

        st.subheader(
            "Engineering Measurements"
        )

        if records:

            for record in records:
                with st.expander(
                    (
                        f"Defect #{record['id']} — "
                        f"{record['severity'].upper()} — "
                        f"{record['priority']}"
                    ),
                    expanded=False,
                ):
                    e1, e2, e3 = st.columns(3)

                    e1.metric(
                        "Area",
                        f"{record['area_pixels']:,.1f} px²",
                    )

                    e2.metric(
                        "Length",
                        f"{record['length_pixels']:.1f} px",
                    )

                    e3.metric(
                        "Width",
                        f"{record['width_pixels']:.1f} px",
                    )

                    st.write(
                        f"Centroid: "
                        f"({record['centroid_x']:.1f}, "
                        f"{record['centroid_y']:.1f}) px"
                    )

        else:
            st.info(
                "No engineering defect records are available."
            )

        st.subheader(
            "Explainable AI"
        )

        if st.button(
            "Generate Grad-CAM",
            key=f"generate_xai_{image_key}",
        ):
            try:
                with st.spinner(
                    "Generating Grad-CAM..."
                ):
                    xai_result = run_gradcam(
                        model,
                        inference.input_tensor,
                    )

                st.session_state[
                    "xai_result"
                ] = xai_result

                st.session_state[
                    "xai_image_key"
                ] = image_key

            except Exception as exc:
                st.error(
                    f"XAI generation failed: {exc}"
                )

        xai_result = (
            st.session_state.get(
                "xai_result"
            )
        )

                if st.session_state.get(
                    "xai_image_key"
                ) != image_key:
                    xai_result = None

                visualization_result = (
                    st.session_state.get(
                        "visualization_result"
                    )
                )

                if st.session_state.get(
                    "visualization_image_key"
                ) != current_3d_key:
                    visualization_result = None

                REPORT_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with st.spinner(
                    "Generating RoadXAI report..."
                ):
                    report = generate_report(
                        engineering_result=(
                            engineering_result
                        ),
                        inference_result=inference,
                        xai_result=xai_result,
                        visualization_result=(
                            visualization_result
                        ),
                        image_name=(
                            uploaded_file.name
                        ),
                        title=(
                            "RoadXAI Road Defect "
                            "Assessment"
                        ),
                    )

                    report_base = (
                        REPORT_DIR
                        / Path(
                            uploaded_file.name
                        ).stem
                    )

                    json_path = (
                        save_json_report(
                            report,
                            report_base.with_suffix(
                                ".json"
                            ),
                        )
                    )

                    text_path = (
                        save_text_report(
                            report,
                            report_base.with_suffix(
                                ".txt"
                            ),
                        )
                    )

                    pdf_path = (
                        generate_pdf_report(
                            report,
                            report_base.with_suffix(
                                ".pdf"
                            ),
                        )
                    )

                st.session_state[
                    "report"
                ] = report

                st.success(
                    "RoadXAI report generated successfully."
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button(
                        "Download JSON",
                        data=(
                            json_path.read_bytes()
                        ),
                        file_name=json_path.name,
                        mime="application/json",
                        key=f"download_json_{image_key}",
                    )

                with col2:
                    st.download_button(
                        "Download TXT",
                        data=(
                            text_path.read_bytes()
                        ),
                        file_name=text_path.name,
                        mime="text/plain",
                        key=f"download_txt_{image_key}",
                    )

                with col3:
                    st.download_button(
                        "Download PDF",
                        data=(
                            pdf_path.read_bytes()
                        ),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        key=f"download_pdf_{image_key}",
                    )

                with st.expander(
                    "Report Preview",
                    expanded=True,
                ):
                    st.text(
                        render_report_text(
                            report
                        )
                    )

            except Exception as exc:
                st.error(
                    "Report generation failed: "
                    f"{exc}"
                )

    except Exception as exc:
        st.error(
            f"Application error: {exc}"
        )


if __name__ == "__main__":
    main()
