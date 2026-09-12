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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
)
from explainable_ai.module import (
    generate_heatmap,
    heatmap_to_numpy,
    overlay_heatmap,
    get_default_target_layer,
)
from engineering_analysis.module import analyze_road
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
    page_icon="🛣️",
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


def heatmap_as_tensor(
    heatmap: Any,
) -> torch.Tensor:
    """Normalize a Tensor/NumPy XAI heatmap to [1,H,W]."""
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
            "XAI heatmap must be a torch.Tensor "
            "or NumPy array."
        )

    if result.ndim == 2:
        result = result.unsqueeze(0)

    if result.ndim != 3:
        raise ValueError(
            "Expected XAI heatmap with shape [B,H,W] "
            "or [H,W]."
        )

    if result.shape[0] != 1:
        raise ValueError(
            "XAI visualization supports batch size 1."
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


def run_xai(
    model,
    input_tensor,
):
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

    heatmap_tensor = heatmap_as_tensor(
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

    image_array = (
        xai_input[0]
        .detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    image_array = np.clip(
        image_array * 255.0,
        0,
        255,
    ).astype(np.uint8)

    xai_overlay = overlay_heatmap(
        image_array,
        heatmap_tensor,
        alpha=0.45,
    )

    return {
        "method": "Grad-CAM",
        "target_layer": target_layer,
        "heatmap": heatmap_array,
        "overlay": xai_overlay,
    }


def make_probability_visualization(
    probability_map: np.ndarray,
    original_image: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create grayscale, color, and image-overlay probability maps."""
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
    ).astype(np.uint8)

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

    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(
            image_rgb,
            0,
            255,
        ).astype(np.uint8)

    if color_rgb.shape[:2] != image_rgb.shape[:2]:
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


def make_numbered_defect_overlay(
    original_image: np.ndarray,
    prediction_mask: np.ndarray,
    engineering_result,
) -> np.ndarray:
    """Draw the predicted mask and numbered engineering defects."""
    image = np.asarray(
        original_image
    )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "original_image must have shape [H,W,3]."
        )

    image = np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)

    mask = np.squeeze(
        np.asarray(prediction_mask)
    )

    if mask.ndim != 2:
        raise ValueError(
            "prediction_mask must be 2D."
        )

    mask = (
        mask > 0
    ).astype(np.uint8)

    image_h, image_w = image.shape[:2]
    mask_h, mask_w = mask.shape

    resized_mask = cv2.resize(
        mask,
        (
            image_w,
            image_h,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    result = image.copy()

    defect_pixels = (
        resized_mask.astype(bool)
    )

    red_layer = np.zeros_like(
        image
    )

    red_layer[:, :, 0] = 220
    red_layer[:, :, 1] = 35
    red_layer[:, :, 2] = 35

    result[defect_pixels] = (
        0.55 * result[defect_pixels]
        + 0.45 * red_layer[defect_pixels]
    ).astype(np.uint8)

    defects = getattr(
        engineering_result,
        "defects",
        [],
    )

    for index, defect in enumerate(
        defects,
        start=1,
    ):
        x = float(
            getattr(
                defect,
                "centroid_x",
                0.0,
            )
        )

        y = float(
            getattr(
                defect,
                "centroid_y",
                0.0,
            )
        )

        x = int(
            round(
                x
                * image_w
                / max(mask_w, 1)
            )
        )

        y = int(
            round(
                y
                * image_h
                / max(mask_h, 1)
            )
        )

        x = max(
            18,
            min(
                image_w - 18,
                x,
            ),
        )

        y = max(
            18,
            min(
                image_h - 18,
                y,
            ),
        )

        radius = max(
            15,
            min(
                22,
                int(
                    min(
                        image_w,
                        image_h,
                    )
                    * 0.018
                ),
            ),
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
        font_scale = (
            0.55
            if index < 10
            else 0.45
        )

        thickness = 2

        text_size, _ = cv2.getTextSize(
            label,
            font,
            font_scale,
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
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return result


def build_visualization_result(
    result,
):
    """Convert the 3D tuple into a UI/report-friendly dictionary."""
    if not isinstance(
        result,
        tuple,
    ):
        return {
            "heightmap": getattr(
                result,
                "heightmap",
                None,
            ),
            "mesh": getattr(
                result,
                "mesh",
                None,
            ),
            "figure": result,
        }

    if len(result) != 3:
        raise ValueError(
            "Expected 3D result as "
            "(HeightMapResult, MeshResult, Figure)."
        )

    heightmap_result, mesh_result, figure = result

    return {
        "heightmap": heightmap_result.heightmap,
        "mesh": mesh_result,
        "figure": figure,
        "heightmap_result": heightmap_result,
    }


def make_image_key(
    uploaded_file,
    image: Any,
) -> str:
    """
    Create a stable key for the currently analyzed image.

    uploaded_file_to_image() may return a PIL.Image.Image rather
    than a NumPy array, so the image is explicitly converted before
    accessing shape/bytes.
    """
    if hasattr(image, "convert"):
        image_array = np.asarray(
            image.convert("RGB")
        )
    else:
        image_array = np.asarray(
            image
        )

    if image_array.ndim != 3:
        raise ValueError(
            "Uploaded image must be a 3-channel image."
        )

    image_bytes = image_array.tobytes()

    digest = hashlib.sha256(
        image_bytes
    ).hexdigest()[:16]

    return (
        f"{uploaded_file.name}|"
        f"{digest}|"
        f"{image_array.shape}"
    )


def clear_stale_analysis_state(
    image_key: str,
) -> None:
    """
    Clear XAI, 3D, and report results when a different image is loaded.
    """
    previous_key = st.session_state.get(
        "analysis_image_key"
    )

    if previous_key == image_key:
        return

    st.session_state[
        "analysis_image_key"
    ] = image_key

    st.session_state.pop(
        "xai_result",
        None,
    )

    st.session_state.pop(
        "visualization_result",
        None,
    )

    st.session_state.pop(
        "visualization_image_key",
        None,
    )

    st.session_state.pop(
        "report",
        None,
    )


def main():
    create_app_header()

    if not CHECKPOINT_PATH.exists():
        st.error(
            "Model checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )
        st.stop()

    threshold, image_size = (
        create_settings_sidebar(
            default_threshold=0.5,
            default_image_size=256,
        )
    )

    uploaded_file = (
        create_upload_section()
    )

    if uploaded_file is None:
        st.info(
            "Upload a road image to begin inference."
        )
        return

    try:
        image = uploaded_file_to_image(
            uploaded_file
        )

        if image is None:
            st.error(
                "Unable to load the uploaded image."
            )
            return

        # Normalize PIL/other image objects to RGB once at the
        # application boundary. This prevents PIL-vs-NumPy shape
        # errors throughout the rest of the app.
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        image_key = make_image_key(
            uploaded_file,
            image,
        )

        clear_stale_analysis_state(
            image_key
        )

        model = load_roadxai_model()

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

        prediction_mask = np.asarray(
            inference.prediction_mask
        )

        prediction_mask = np.squeeze(
            prediction_mask
        )

        if prediction_mask.ndim != 2:
            raise ValueError(
                "Model prediction mask must be 2D."
            )

        resized_mask = (
            resize_mask_to_image(
                prediction_mask,
                (
                    inference.original_image.shape[1],
                    inference.original_image.shape[0],
                ),
            )
        )

        # -----------------------------------------------------
        # Engineering analysis
        # -----------------------------------------------------

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

        probability = np.squeeze(
            probability
        )

        probability = np.clip(
            probability,
            0.0,
            1.0,
        )

        with st.expander(
            "Probability Map",
            expanded=True,
        ):
            st.caption(
                "Each pixel contains the model's estimated "
                "defect probability. The selected threshold "
                f"is {threshold:.2f}."
            )

            p1, p2, p3 = st.columns(3)

            with p1:
                display_image(
                    probability_visuals[
                        "grayscale"
                    ],
                    caption=(
                        "Probability Intensity"
                    ),
                )

            with p2:
                display_image(
                    probability_visuals[
                        "color"
                    ],
                    caption=(
                        "Color Probability Heatmap"
                    ),
                )

            with p3:
                display_image(
                    probability_visuals[
                        "overlay"
                    ],
                    caption=(
                        "Probability on Original Image"
                    ),
                )

            st.markdown(
                """
                **Scale**

                `0.00` — very low defect probability  
                `0.50` — example decision threshold  
                `1.00` — very high defect probability

                Warmer/brighter regions indicate higher model
                probability; they do not directly represent
                physical defect depth.
                """
            )

            pc1, pc2, pc3, pc4 = st.columns(4)

            pc1.metric(
                "Mean Probability",
                f"{float(probability.mean()):.3f}",
            )

            pc2.metric(
                "Maximum",
                f"{float(probability.max()):.3f}",
            )

            pc3.metric(
                "Pixels Above Threshold",
                f"{int(np.count_nonzero(probability >= threshold)):,}",
            )

            pc4.metric(
                "Decision Threshold",
                f"{threshold:.2f}",
            )

        st.divider()

        # -----------------------------------------------------
        # Engineering Analysis
        # -----------------------------------------------------

        st.subheader(
            "Engineering Analysis"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Detected Defects",
            str(
                len(
                    engineering_result.defects
                )
            ),
        )

        col2.metric(
            "Road Health Indicator",
            engineering_result.road_health.condition,
        )

        col3.metric(
            "Health Score",
            (
                f"{engineering_result.road_health.score:.2f}"
            ),
        )

        st.caption(
            "The health score is a deterministic image-based "
            "indicator derived from the detected defect area. "
            "It is not a structural safety certification."
        )

        if engineering_result.defects:
            st.write(
                "### Defect Measurements"
            )

            rows = []

            for index, defect in enumerate(
                engineering_result.defects,
                start=1,
            ):
                rows.append(
                    {
                        "Defect": index,
                        "Area (pixels)": round(
                            float(
                                defect.area_pixels
                            ),
                            2,
                        ),
                        "Length (pixels)": round(
                            float(
                                defect.length_pixels
                            ),
                            2,
                        ),
                        "Width (pixels)": round(
                            float(
                                defect.width_pixels
                            ),
                            2,
                        ),
                        "Centroid X": round(
                            float(
                                defect.centroid_x
                            ),
                            2,
                        ),
                        "Centroid Y": round(
                            float(
                                defect.centroid_y
                            ),
                            2,
                        ),
                    }
                )

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )

            st.write(
                "### Severity Assessment"
            )

            for index, severity in enumerate(
                engineering_result.severity,
                start=1,
            ):
                st.write(
                    f"**Defect {index}:** "
                    f"{severity.level.value.upper()} "
                    f"— score {severity.score:.2f}"
                )

                st.caption(
                    severity.explanation
                )

            st.write(
                "### Repair Cost"
            )

            st.info(
                "Physical repair cost is not calculated because "
                "this image analysis currently has no pixel-to-meter "
                "calibration or supplied repair rate."
            )

        else:
            st.success(
                "No defects detected at the selected threshold."
            )

        st.divider()

        # -----------------------------------------------------
        # Explainable AI
        # -----------------------------------------------------

        st.subheader(
            "Explainable AI"
        )

        st.caption(
            "Grad-CAM highlights image regions that contributed "
            "to the model's prediction. It is an explanation of "
            "model attention, not a measurement of defect size."
        )

        if st.button(
            "Generate Grad-CAM Explanation",
            key="generate_xai",
        ):
            try:
                with st.spinner(
                    "Generating Grad-CAM explanation..."
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

            except Exception as exc:
                st.error(
                    "XAI generation failed: "
                    f"{exc}"
                )

        xai_result = (
            st.session_state.get(
                "xai_result"
            )
        )

        if (
            xai_result is not None
            and st.session_state.get(
                "xai_image_key"
            ) == image_key
        ):
            col1, col2 = st.columns(2)

            with col1:
                display_image(
                    xai_result["heatmap"],
                    caption=(
                        "Grad-CAM Heatmap — "
                        "Model Attention"
                    ),
                )

            with col2:
                display_image(
                    xai_result["overlay"],
                    caption=(
                        "Grad-CAM Overlay"
                    ),
                )

            target_layer = (
                xai_result[
                    "target_layer"
                ]
            )

            st.caption(
                f"Method: {xai_result['method']} | "
                f"Target layer: "
                f"{target_layer.__class__.__name__}"
            )

            heatmap_np = np.asarray(
                xai_result["heatmap"],
                dtype=np.float32,
            )

            xc1, xc2, xc3 = st.columns(3)

            xc1.metric(
                "Mean Attention",
                f"{float(heatmap_np.mean()):.3f}",
            )

            xc2.metric(
                "Maximum Attention",
                f"{float(heatmap_np.max()):.3f}",
            )

            xc3.metric(
                "Attention Pixels > 0.5",
                f"{int(np.count_nonzero(heatmap_np >= 0.5)):,}",
            )

        st.divider()

        # -----------------------------------------------------
        # 3D Visualization
        # -----------------------------------------------------

        st.subheader(
            "3D Visualization"
        )

        st.caption(
            "The 3D view uses the current image and current "
            "segmentation mask. The surface is image-aligned. "
            "Vertical elevation is normalized visualization "
            "geometry, not measured physical road depth."
        )

        current_mask_for_3d = np.asarray(
            prediction_mask,
            dtype=np.uint8,
        ).copy()

        current_3d_key = (
            f"{image_key}|"
            f"{current_mask_for_3d.shape}|"
            f"{int(np.count_nonzero(current_mask_for_3d))}|"
            f"{float(current_mask_for_3d.mean()):.8f}"
        )

        if st.session_state.get(
            "visualization_image_key"
        ) != current_3d_key:
            st.session_state.pop(
                "visualization_result",
                None,
            )

        if st.button(
            "Generate 3D Defect Visualization",
            key="generate_3d",
        ):
            try:
                # Explicitly remove any previous image's result.
                st.session_state.pop(
                    "visualization_result",
                    None,
                )

                with st.spinner(
                    "Generating image-aligned 3D visualization..."
                ):
                    raw_result = (
                        generate_3d_visualization(
                            mask=current_mask_for_3d,
                            max_depth=1.0,
                            pixel_size=1.0,
                            title=(
                                "RoadXAI "
                                "Image-Aligned "
                                "3D Defect Visualization"
                            ),
                            road_image=np.asarray(
                                inference.original_image
                            ).copy(),
                        )
                    )

                visualization_result = (
                    build_visualization_result(
                        raw_result
                    )
                )

                st.session_state[
                    "visualization_result"
                ] = visualization_result

                st.session_state[
                    "visualization_image_key"
                ] = current_3d_key

            except Exception as exc:
                st.error(
                    "3D visualization failed: "
                    f"{exc}"
                )

        visualization_result = (
            st.session_state.get(
                "visualization_result"
            )
        )

        if (
            visualization_result is not None
            and st.session_state.get(
                "visualization_image_key"
            ) == current_3d_key
        ):
            figure = visualization_result[
                "figure"
            ]

            st.plotly_chart(
                figure,
                use_container_width=True,
                key=f"3d_plot_{current_3d_key}",
            )

            heightmap = (
                visualization_result[
                    "heightmap"
                ]
            )

            mesh = (
                visualization_result[
                    "mesh"
                ]
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Heightmap",
                (
                    f"{heightmap.shape[0]} × "
                    f"{heightmap.shape[1]}"
                ),
            )

            col2.metric(
                "Mesh Vertices",
                f"{len(mesh.vertices):,}",
            )

            col3.metric(
                "Mesh Faces",
                f"{len(mesh.faces):,}",
            )

        st.divider()

        # -----------------------------------------------------
        # Report Generation
        # -----------------------------------------------------

        st.subheader(
            "Report Generation"
        )

        st.caption(
            "Generate a structured RoadXAI assessment containing "
            "detection, engineering analysis, XAI, 3D visualization, "
            "and inspection-support recommendations."
        )

        if st.button(
            "Generate RoadXAI Report",
            key="generate_report",
        ):
            try:
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
