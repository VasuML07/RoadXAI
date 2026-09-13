"""
RoadXAI Interactive 3D Visualization Module
=============================================

Purpose
-------
Create a professional, image-aligned 3D road-inspection visualization.

The viewer is designed to help a non-technical user answer:

    - Where is the damage?
    - How many damaged regions are present?
    - Which regions are serious?
    - How large are they?
    - Where are they located in the image?
    - What does the AI think is important?
    - What does the visual depth representation mean?

Important scientific limitation
--------------------------------
A single RGB image does NOT provide reliable physical pothole depth.

Therefore:

    Z = normalized visual geometry

unless an external calibrated depth source is explicitly supplied.

The 3D model is a visualization and decision-support representation,
not a physical depth measurement in centimetres or millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import cv2
import numpy as np


ArrayLike = Any


# ============================================================
# RESULT TYPES
# ============================================================


@dataclass
class HeightMapResult:
    """
    Result produced by the mask -> visual depth conversion.
    """

    heightmap: np.ndarray
    min_height: float
    max_height: float
    mean_height: float
    defect_pixels: int
    depth_unit: str = "normalized"
    estimated_max_depth_cm: Optional[float] = None


@dataclass
class MeshResult:
    """
    Triangular mesh representation of the generated surface.
    """

    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class Defect3DInfo:
    """
    Engineering information attached to one 3D defect marker.

    Dimensions are pixel-based unless the engineering module
    supplies calibrated physical measurements.
    """

    defect_number: int

    severity: str = "Unknown"

    severity_score: Optional[float] = None

    area_pixels: Optional[float] = None

    length_pixels: Optional[float] = None

    width_pixels: Optional[float] = None

    centroid_x: Optional[float] = None

    centroid_y: Optional[float] = None

    confidence: Optional[float] = None


# ============================================================
# VALIDATION
# ============================================================


def validate_mask(
    mask: ArrayLike,
) -> np.ndarray:
    """
    Validate and convert a segmentation mask into H x W float32.

    Accepted input shapes include:

        H x W
        H x W x 1
        1 x H x W

    Any positive value is considered damaged.
    """

    array = np.asarray(mask)

    if array.size == 0:
        raise ValueError(
            "Mask cannot be empty."
        )

    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(
            "Mask must represent a 2D image after squeezing. "
            f"Received shape: {array.shape}"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "Mask contains NaN or infinite values."
        )

    return (
        array.astype(
            np.float32,
            copy=False,
        )
        > 0
    ).astype(
        np.float32
    )


def _validate_numeric_image(
    image: ArrayLike,
) -> np.ndarray:
    """
    Convert an image-like object to uint8 RGB.
    """

    array = np.asarray(image)

    if array.size == 0:
        raise ValueError(
            "Image cannot be empty."
        )

    if array.ndim == 2:
        array = np.stack(
            [array] * 3,
            axis=-1,
        )

    if array.ndim != 3:
        raise ValueError(
            "Road image must be a 2D grayscale or 3D RGB image."
        )

    if array.shape[2] == 1:
        array = np.repeat(
            array,
            3,
            axis=2,
        )

    if array.shape[2] != 3:
        raise ValueError(
            "Road image must have exactly 3 channels."
        )

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    if array.dtype != np.uint8:
        if float(array.max()) <= 1.0:
            array = array * 255.0

        array = np.clip(
            array,
            0,
            255,
        ).astype(
            np.uint8
        )

    return array


# ============================================================
# NUMERICAL HELPERS
# ============================================================


def normalize_array(
    array: ArrayLike,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize an array into [0, 1].
    """

    values = np.asarray(
        array,
        dtype=np.float32,
    )

    if values.size == 0:
        raise ValueError(
            "Array cannot be empty."
        )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    minimum = float(
        values.min()
    )

    maximum = float(
        values.max()
    )

    if maximum - minimum <= eps:
        return np.zeros_like(
            values,
            dtype=np.float32,
        )

    return (
        (values - minimum)
        / (
            maximum - minimum
        )
    ).astype(
        np.float32
    )


def _gaussian_blur(
    array: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Apply Gaussian smoothing safely.
    """

    if sigma <= 0:
        return np.asarray(
            array,
            dtype=np.float32,
        )

    kernel = max(
        3,
        int(
            round(
                float(sigma) * 6
            )
        )
        | 1,
    )

    return cv2.GaussianBlur(
        np.asarray(
            array,
            dtype=np.float32,
        ),
        (
            kernel,
            kernel,
        ),
        float(sigma),
    ).astype(
        np.float32
    )


# ============================================================
# IMAGE PREPARATION
# ============================================================


def _prepare_image(
    image: ArrayLike,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """
    Convert the road image to RGB and resize it to the
    exact heightmap dimensions.

    target_shape = (height, width)
    """

    result = _validate_numeric_image(
        image
    )

    target_height = int(
        target_shape[0]
    )

    target_width = int(
        target_shape[1]
    )

    if result.shape[:2] != (
        target_height,
        target_width,
    ):
        result = cv2.resize(
            result,
            (
                target_width,
                target_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    return result


# ============================================================
# DEPTH GENERATION
# ============================================================


def _component_depth_profile(
    component: np.ndarray,
    max_depth: float,
    min_visible_depth: float,
) -> np.ndarray:
    """
    Create a smooth visual depression for one connected component.

    The center of a broad damaged region becomes visually deeper.

    Thin cracks receive a minimum visible depth so they do not
    disappear from the 3D representation.
    """

    component_u8 = (
        component > 0
    ).astype(
        np.uint8
    )

    if not np.any(
        component_u8
    ):
        return np.zeros_like(
            component,
            dtype=np.float32,
        )

    distance = cv2.distanceTransform(
        component_u8,
        cv2.DIST_L2,
        5,
    ).astype(
        np.float32
    )

    distance_normalized = normalize_array(
        distance
    )

    area = int(
        np.count_nonzero(
            component_u8
        )
    )

    # Large regions receive slightly stronger visual relief.
    area_factor = np.clip(
        np.log1p(area) / 12.0,
        0.0,
        1.0,
    )

    depth_factor = (
        0.65
        + 0.35 * area_factor
    )

    visual_depth = (
        float(min_visible_depth)
        + (
            distance_normalized
            * (
                float(max_depth)
                - float(min_visible_depth)
            )
        )
        * depth_factor
    )

    visual_depth *= component_u8

    return np.clip(
        visual_depth,
        0.0,
        float(max_depth),
    ).astype(
        np.float32
    )


def create_heightmap(
    mask: ArrayLike,
    max_depth: float = 1.0,
    smoothing: float = 1.2,
    min_visible_depth: float = 0.12,
) -> HeightMapResult:
    """
    Convert a segmentation mask into normalized visual depth.

    This is NOT physical depth estimation.

    The output represents visual relief:

        road surface -> approximately 0
        damaged region -> positive visual depth
    """

    validated = validate_mask(
        mask
    )

    if max_depth <= 0:
        raise ValueError(
            "max_depth must be greater than zero."
        )

    if not (
        0.0
        <= min_visible_depth
        <= max_depth
    ):
        raise ValueError(
            "min_visible_depth must be between 0 and max_depth."
        )

    binary = (
        validated > 0
    ).astype(
        np.uint8
    )

    heightmap = np.zeros(
        binary.shape,
        dtype=np.float32,
    )

    defect_pixels = int(
        np.count_nonzero(
            binary
        )
    )

    if defect_pixels == 0:
        return HeightMapResult(
            heightmap=heightmap,
            min_height=0.0,
            max_height=0.0,
            mean_height=0.0,
            defect_pixels=0,
        )

    count, labels, _, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    for component_id in range(
        1,
        count,
    ):
        component = (
            labels == component_id
        ).astype(
            np.uint8
        )

        profile = (
            _component_depth_profile(
                component,
                max_depth=max_depth,
                min_visible_depth=(
                    min_visible_depth
                ),
            )
        )

        heightmap = np.maximum(
            heightmap,
            profile,
        )

    if smoothing > 0:
        blurred = _gaussian_blur(
            heightmap,
            sigma=smoothing,
        )

        # Never create artificial depth outside the segmentation.
        heightmap = (
            blurred * binary
        ).astype(
            np.float32
        )

    heightmap = np.clip(
        heightmap,
        0.0,
        float(max_depth),
    )

    return HeightMapResult(
        heightmap=heightmap,
        min_height=float(
            heightmap.min()
        ),
        max_height=float(
            heightmap.max()
        ),
        mean_height=float(
            heightmap.mean()
        ),
        defect_pixels=defect_pixels,
        depth_unit="normalized",
        estimated_max_depth_cm=None,
    )


# ============================================================
# COORDINATES
# ============================================================


def create_coordinate_grid(
    height: int,
    width: int,
    pixel_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create image-coordinate grids.

    X increases left -> right.
    Y increases top -> bottom.
    """

    height = int(height)
    width = int(width)

    if height <= 0 or width <= 0:
        raise ValueError(
            "height and width must be positive."
        )

    if pixel_size <= 0:
        raise ValueError(
            "pixel_size must be greater than zero."
        )

    x = (
        np.arange(
            width,
            dtype=np.float32,
        )
        * float(pixel_size)
    )

    y = (
        np.arange(
            height,
            dtype=np.float32,
        )
        * float(pixel_size)
    )

    x_grid, y_grid = np.meshgrid(
        x,
        y,
    )

    return (
        x_grid,
        y_grid,
    )


# ============================================================
# MESH
# ============================================================


def heightmap_to_mesh(
    heightmap: ArrayLike,
    pixel_size: float = 1.0,
) -> MeshResult:
    """
    Convert a heightmap into a triangular mesh.

    Z is stored as negative visual depth so a damaged region
    behaves visually like a depression.
    """

    height = np.asarray(
        heightmap,
        dtype=np.float32,
    )

    if height.ndim != 2:
        raise ValueError(
            "heightmap must be 2D."
        )

    if height.size == 0:
        raise ValueError(
            "heightmap cannot be empty."
        )

    rows, cols = height.shape

    x_grid, y_grid = (
        create_coordinate_grid(
            rows,
            cols,
            pixel_size,
        )
    )

    z_grid = -height

    vertices = np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
            z_grid.ravel(),
        )
    ).astype(
        np.float32
    )

    if rows < 2 or cols < 2:
        faces = np.empty(
            (
                0,
                3,
            ),
            dtype=np.int32,
        )

        return MeshResult(
            vertices=vertices,
            faces=faces,
        )

    row = np.arange(
        rows - 1,
        dtype=np.int32,
    )[:, None]

    col = np.arange(
        cols - 1,
        dtype=np.int32,
    )[None, :]

    top_left = (
        row * cols + col
    ).ravel()

    top_right = (
        top_left + 1
    )

    bottom_left = (
        (row + 1) * cols + col
    ).ravel()

    bottom_right = (
        bottom_left + 1
    )

    faces = np.vstack(
        (
            np.column_stack(
                (
                    top_left,
                    bottom_left,
                    top_right,
                )
            ),
            np.column_stack(
                (
                    top_right,
                    bottom_left,
                    bottom_right,
                )
            ),
        )
    ).astype(
        np.int32
    )

    return MeshResult(
        vertices=vertices,
        faces=faces,
    )


# ============================================================
# COLOR HELPERS
# ============================================================


def _rgb_strings(
    image: np.ndarray,
) -> list[str]:
    """
    Convert RGB pixels into Plotly rgb(...) strings.
    """

    flat = np.asarray(
        image,
        dtype=np.uint8,
    ).reshape(
        -1,
        3,
    )

    return [
        (
            f"rgb({int(pixel[0])},"
            f"{int(pixel[1])},"
            f"{int(pixel[2])})"
        )
        for pixel in flat
    ]


def _blend_rgb(
    image: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.58,
) -> np.ndarray:
    """
    Highlight detected damage while preserving the original road image.
    """

    base = np.asarray(
        image,
        dtype=np.float32,
    )

    binary = (
        np.asarray(mask)
        > 0
    ).astype(
        np.float32
    )

    damage_color = np.array(
        [
            239.0,
            68.0,
            68.0,
        ],
        dtype=np.float32,
    )

    alpha = (
        binary
        * float(strength)
    )[..., None]

    result = (
        base * (
            1.0 - alpha
        )
        + damage_color * alpha
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )


def _apply_heatmap(
    image: np.ndarray,
    values: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.72,
) -> np.ndarray:
    """
    Apply a Turbo heatmap over selected regions.

    Turbo is used because it gives a visually obvious progression
    from low -> medium -> high values.
    """

    normalized = normalize_array(
        values
    )

    heat = cv2.applyColorMap(
        (
            normalized
            * 255.0
        ).astype(
            np.uint8
        ),
        cv2.COLORMAP_TURBO,
    )

    heat = cv2.cvtColor(
        heat,
        cv2.COLOR_BGR2RGB,
    )

    result = image.astype(
        np.float32
    ).copy()

    if mask is None:
        alpha = np.full(
            normalized.shape,
            float(strength),
            dtype=np.float32,
        )
    else:
        alpha = (
            np.asarray(mask)
            > 0
        ).astype(
            np.float32
        ) * float(strength)

    alpha = alpha[..., None]

    result = (
        result * (
            1.0 - alpha
        )
        + heat.astype(
            np.float32
        ) * alpha
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# GENERIC OBJECT ACCESS
# ============================================================


def _value(
    source: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Read an attribute from either a dataclass/object or dictionary.
    """

    if source is None:
        return default

    if isinstance(
        source,
        Mapping,
    ):
        return source.get(
            name,
            default,
        )

    return getattr(
        source,
        name,
        default,
    )


def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        result = float(
            value
        )

        if not np.isfinite(
            result
        ):
            return default

        return result

    except (
        TypeError,
        ValueError,
    ):
        return default


def _severity_color(
    severity: str,
) -> str:
    """
    Professional severity palette.

    Green   = low
    Yellow  = moderate
    Orange  = high
    Red     = critical
    """

    text = str(
        severity
    ).lower()

    if "critical" in text:
        return "#ff1744"

    if "high" in text:
        return "#ff6d00"

    if "moderate" in text:
        return "#ffc107"

    if "low" in text:
        return "#00c853"

    return "#42a5f5"


# ============================================================
# DEFECT METADATA
# ============================================================


def _extract_defect_metadata(
    engineering_result: Any = None,
    confidence: Optional[float] = None,
) -> list[Defect3DInfo]:
    """
    Convert engineering-analysis output into 3D metadata.

    Supports both common structures:

        engineering_result.defects

    and

        engineering_result.measurements
    """

    if engineering_result is None:
        return []

    measurements = _value(
        engineering_result,
        "measurements",
        None,
    )

    if measurements is None:
        measurements = _value(
            engineering_result,
            "defects",
            [],
        )

    severity_values = _value(
        engineering_result,
        "severity",
        [],
    )

    if measurements is None:
        measurements = []

    if severity_values is None:
        severity_values = []

    result: list[
        Defect3DInfo
    ] = []

    for index, measurement in enumerate(
        measurements,
        start=1,
    ):
        severity_item = (
            severity_values[index - 1]
            if index - 1
            < len(severity_values)
            else None
        )

        severity_level = _value(
            severity_item,
            "level",
            "Unknown",
        )

        if hasattr(
            severity_level,
            "value",
        ):
            severity_level = (
                severity_level.value
            )

        result.append(
            Defect3DInfo(
                defect_number=int(
                    _value(
                        measurement,
                        "defect_number",
                        index,
                    )
                ),
                severity=str(
                    severity_level
                ).upper(),
                severity_score=_safe_float(
                    _value(
                        severity_item,
                        "score",
                        None,
                    )
                ),
                area_pixels=_safe_float(
                    _value(
                        measurement,
                        "area_pixels",
                        None,
                    )
                ),
                length_pixels=_safe_float(
                    _value(
                        measurement,
                        "length_pixels",
                        None,
                    )
                ),
                width_pixels=_safe_float(
                    _value(
                        measurement,
                        "width_pixels",
                        None,
                    )
                ),
                centroid_x=_safe_float(
                    _value(
                        measurement,
                        "centroid_x",
                        None,
                    )
                ),
                centroid_y=_safe_float(
                    _value(
                        measurement,
                        "centroid_y",
                        None,
                    )
                ),
                confidence=_safe_float(
                    confidence
                ),
            )
        )

    return result


# ============================================================
# COMPONENT INFORMATION
# ============================================================


def _connected_components(
    mask: np.ndarray,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return connected-component information.
    """

    binary = (
        mask > 0
    ).astype(
        np.uint8
    )

    return cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )


def _component_masks(
    mask: np.ndarray,
) -> list[np.ndarray]:
    """
    Return one binary mask for every detected region.
    """

    count, labels, _, _ = (
        _connected_components(
            mask
        )
    )

    result = []

    for component_id in range(
        1,
        count,
    ):
        result.append(
            (
                labels
                == component_id
            ).astype(
                np.uint8
            )
        )

    return result


# ============================================================
# SEVERITY MAP
# ============================================================


def _build_severity_map(
    mask: np.ndarray,
    metadata: Sequence[
        Defect3DInfo
    ],
) -> np.ndarray:
    """
    Build a numeric severity map.

    Values:

        Low       = 0.20
        Moderate  = 0.45
        High      = 0.72
        Critical  = 1.00
    """

    severity_map = np.zeros(
        mask.shape,
        dtype=np.float32,
    )

    components = _component_masks(
        mask
    )

    for index, component in enumerate(
        components
    ):
        if index < len(metadata):
            severity = metadata[
                index
            ].severity.lower()
        else:
            severity = "unknown"

        if "critical" in severity:
            value = 1.0
        elif "high" in severity:
            value = 0.72
        elif "moderate" in severity:
            value = 0.45
        elif "low" in severity:
            value = 0.20
        else:
            value = 0.55

        severity_map[
            component > 0
        ] = value

    return severity_map


# ============================================================
# MARKERS
# ============================================================


def _add_defect_markers(
    figure: Any,
    heightmap: np.ndarray,
    mask: np.ndarray,
    metadata: Sequence[
        Defect3DInfo
    ],
    pixel_size: float,
) -> None:
    """
    Add numbered defect markers with rich hover information.
    """

    import plotly.graph_objects as go

    components = _component_masks(
        mask
    )

    marker_x = []
    marker_y = []
    marker_z = []
    marker_text = []
    marker_colors = []
    customdata = []

    for index, component in enumerate(
        components,
        start=1,
    ):
        rows, cols = np.where(
            component > 0
        )

        if len(rows) == 0:
            continue

        if index - 1 < len(
            metadata
        ):
            item = metadata[
                index - 1
            ]
        else:
            item = Defect3DInfo(
                defect_number=index,
                centroid_x=float(
                    cols.mean()
                ),
                centroid_y=float(
                    rows.mean()
                ),
            )

        x = (
            item.centroid_x
            if item.centroid_x
            is not None
            else float(
                cols.mean()
            )
        )

        y = (
            item.centroid_y
            if item.centroid_y
            is not None
            else float(
                rows.mean()
            )
        )

        x_index = int(
            np.clip(
                round(x),
                0,
                heightmap.shape[1] - 1,
            )
        )

        y_index = int(
            np.clip(
                round(y),
                0,
                heightmap.shape[0] - 1,
            )
        )

        z = -float(
            heightmap[
                y_index,
                x_index,
            ]
        )

        marker_x.append(
            float(x)
            * pixel_size
        )

        marker_y.append(
            float(y)
            * pixel_size
        )

        marker_z.append(
            z
            - 0.035
        )

        marker_text.append(
            f"Defect #{item.defect_number}"
        )

        marker_colors.append(
            _severity_color(
                item.severity
            )
        )

        customdata.append(
            [
                item.defect_number,
                item.severity,
                (
                    item.severity_score
                    if item.severity_score
                    is not None
                    else "N/A"
                ),
                (
                    item.area_pixels
                    if item.area_pixels
                    is not None
                    else "N/A"
                ),
                (
                    item.length_pixels
                    if item.length_pixels
                    is not None
                    else "N/A"
                ),
                (
                    item.width_pixels
                    if item.width_pixels
                    is not None
                    else "N/A"
                ),
                x,
                y,
                (
                    f"{item.confidence:.1%}"
                    if item.confidence
                    is not None
                    else "N/A"
                ),
            ]
        )

    if not marker_x:
        return

    figure.add_trace(
        go.Scatter3d(
            x=marker_x,
            y=marker_y,
            z=marker_z,
            mode="markers+text",
            text=marker_text,
            textposition="top center",
            textfont=dict(
                size=11,
                color="white",
            ),
            marker=dict(
                size=8,
                color=marker_colors,
                line=dict(
                    width=2,
                    color="white",
                ),
                opacity=1.0,
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>Defect #%{customdata[0]}</b>"
                "<br><br>"
                "Severity: "
                "<b>%{customdata[1]}</b>"
                "<br>"
                "Severity score: "
                "%{customdata[2]}"
                "<br>"
                "Area: "
                "%{customdata[3]} px²"
                "<br>"
                "Length: "
                "%{customdata[4]} px"
                "<br>"
                "Width: "
                "%{customdata[5]} px"
                "<br>"
                "Image X: "
                "%{customdata[6]:.1f}"
                "<br>"
                "Image Y: "
                "%{customdata[7]:.1f}"
                "<br>"
                "Model confidence: "
                "%{customdata[8]}"
                "<extra></extra>"
            ),
            name="Defect Locations",
            showlegend=True,
        )
    )


# ============================================================
# BOUNDARIES
# ============================================================


def _add_defect_boundaries(
    figure: Any,
    heightmap: np.ndarray,
    mask: np.ndarray,
    pixel_size: float,
) -> None:
    """
    Draw connected-component boundaries in 3D.
    """

    import plotly.graph_objects as go

    binary = (
        mask > 0
    ).astype(
        np.uint8
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for index, contour in enumerate(
        contours,
        start=1,
    ):
        if len(contour) < 2:
            continue

        points = contour.reshape(
            -1,
            2,
        )

        xs = []
        ys = []
        zs = []

        for x, y in points:
            x_index = int(
                np.clip(
                    x,
                    0,
                    heightmap.shape[1] - 1,
                )
            )

            y_index = int(
                np.clip(
                    y,
                    0,
                    heightmap.shape[0] - 1,
                )
            )

            xs.append(
                float(x)
                * pixel_size
            )

            ys.append(
                float(y)
                * pixel_size
            )

            zs.append(
                -float(
                    heightmap[
                        y_index,
                        x_index,
                    ]
                )
                - 0.018
            )

        figure.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(
                    color="#ffffff",
                    width=4,
                ),
                hoverinfo="skip",
                name=(
                    "Damage Boundary"
                    if index == 1
                    else None
                ),
                showlegend=(
                    index == 1
                ),
            )
        )


# ============================================================
# 3D SURFACE
# ============================================================


def create_plotly_surface(
    heightmap: ArrayLike,
    pixel_size: float = 1.0,
    title: str = "RoadXAI Interactive Road Inspection",
    road_image: Optional[
        ArrayLike
    ] = None,
    defect_mask: Optional[
        ArrayLike
    ] = None,
    engineering_result: Any = None,
    confidence: Optional[
        float
    ] = None,
    show_markers: bool = True,
    show_boundaries: bool = True,
    display_mode: str = "inspection",
    vertical_exaggeration: float = 1.0,
    probability_map: Optional[
        ArrayLike
    ] = None,
    xai_heatmap: Optional[
        ArrayLike
    ] = None,
):
    """
    Create the main professional interactive 3D viewer.

    Supported display modes
    -----------------------

    inspection
        Original road photograph + red damage highlighting.

    original
        Original road photograph with minimal overlays.

    severity
        Severity-based colors.

    depth
        Visual-depth heatmap.

    probability
        Model probability heatmap.

    xai
        Grad-CAM / attention heatmap.

    ai_confidence
        Probability-oriented AI confidence view.

    ai_attention
        Explainability-oriented attention view.
    """

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for 3D visualization. "
            "Install it with: pip install plotly"
        ) from exc

    height = np.asarray(
        heightmap,
        dtype=np.float32,
    )

    if height.ndim != 2:
        raise ValueError(
            "heightmap must be 2D."
        )

    height = np.nan_to_num(
        height,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    h, w = height.shape

    if road_image is None:
        image = np.full(
            (
                h,
                w,
                3,
            ),
            92,
            dtype=np.uint8,
        )
    else:
        image = _prepare_image(
            road_image,
            (
                h,
                w,
            ),
        )

    if defect_mask is None:
        mask = (
            height > 0
        ).astype(
            np.float32
        )
    else:
        mask = validate_mask(
            defect_mask
        )

        if mask.shape != (
            h,
            w,
        ):
            mask = cv2.resize(
                mask,
                (
                    w,
                    h,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

    mode = str(
        display_mode
    ).strip().lower()

    aliases = {
        "inspection view": "inspection",
        "road surface": "original",
        "original road surface": "original",
        "severity view": "severity",
        "visual depth": "depth",
        "ai confidence": "probability",
        "ai attention": "xai",
        "probability view": "probability",
        "grad-cam": "xai",
    }

    mode = aliases.get(
        mode,
        mode,
    )

    allowed_modes = {
        "inspection",
        "original",
        "severity",
        "depth",
        "probability",
        "xai",
        "ai_confidence",
        "ai_attention",
    }

    if mode not in allowed_modes:
        mode = "inspection"

    # --------------------------------------------------------
    # Prepare external maps
    # --------------------------------------------------------

    probability = None

    if probability_map is not None:
        probability = np.asarray(
            probability_map,
            dtype=np.float32,
        )

        probability = np.squeeze(
            probability
        )

        if probability.ndim == 2:
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

            if probability.shape != (
                h,
                w,
            ):
                probability = cv2.resize(
                    probability,
                    (
                        w,
                        h,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
        else:
            probability = None

    xai = None

    if xai_heatmap is not None:
        xai = np.asarray(
            xai_heatmap,
            dtype=np.float32,
        )

        xai = np.squeeze(
            xai
        )

        if xai.ndim == 2:
            xai = np.nan_to_num(
                xai,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )

            xai = normalize_array(
                xai
            )

            if xai.shape != (
                h,
                w,
            ):
                xai = cv2.resize(
                    xai,
                    (
                        w,
                        h,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
        else:
            xai = None

    # --------------------------------------------------------
    # Surface appearance
    # --------------------------------------------------------

    if mode == "original":
        surface_image = image.copy()

    elif mode == "severity":
        severity_map = (
            _build_severity_map(
                mask,
                _extract_defect_metadata(
                    engineering_result,
                    confidence,
                ),
            )
        )

        surface_image = _apply_heatmap(
            image,
            severity_map,
            mask=mask,
            strength=0.78,
        )

    elif mode == "depth":
        surface_image = _apply_heatmap(
            image,
            normalize_array(
                height
            ),
            mask=mask,
            strength=0.82,
        )

    elif mode in {
        "probability",
        "ai_confidence",
    }:
        if probability is None:
            surface_image = _blend_rgb(
                image,
                mask,
                strength=0.60,
            )
        else:
            surface_image = _apply_heatmap(
                image,
                probability,
                mask=None,
                strength=0.82,
            )

    elif mode in {
        "xai",
        "ai_attention",
    }:
        if xai is None:
            surface_image = _blend_rgb(
                image,
                mask,
                strength=0.60,
            )
        else:
            surface_image = _apply_heatmap(
                image,
                xai,
                mask=None,
                strength=0.82,
            )

    else:
        surface_image = _blend_rgb(
            image,
            mask,
            strength=0.56,
        )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    x_grid, y_grid = (
        create_coordinate_grid(
            h,
            w,
            pixel_size,
        )
    )

    exaggeration = max(
        0.05,
        float(
            vertical_exaggeration
        ),
    )

    # Negative Z = downward depression.
    z_grid = (
        -height
        * exaggeration
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    figure = go.Figure()

    # --------------------------------------------------------
    # Main image-aligned 3D surface
    # --------------------------------------------------------

    figure.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            surfacecolor=np.zeros_like(
                z_grid
            ),
            colorscale=[
                [
                    0.0,
                    "rgb(0,0,0)",
                ],
                [
                    1.0,
                    "rgb(0,0,0)",
                ],
            ],
            showscale=False,
            opacity=1.0,
            customdata=np.stack(
                [
                    x_grid,
                    y_grid,
                    height,
                    mask,
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>Road Surface</b>"
                "<br><br>"
                "Image X: "
                "%{customdata[0]:.0f}"
                "<br>"
                "Image Y: "
                "%{customdata[1]:.0f}"
                "<br>"
                "Visual depth: "
                "%{customdata[2]:.3f}"
                "<br>"
                "Detected damage: "
                "%{customdata[3]:.0f}"
                "<extra></extra>"
            ),
            name="Road Surface",
            showlegend=True,
        )
    )

    # --------------------------------------------------------
    # Photo texture mesh
    #
    # Mesh3d supports per-vertex RGB colors and therefore
    # preserves the COMPLETE uploaded image.
    # --------------------------------------------------------

    vertices = np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
            z_grid.ravel(),
        )
    ).astype(
        np.float32
    )

    if h >= 2 and w >= 2:
        row = np.arange(
            h - 1,
            dtype=np.int32,
        )[:, None]

        col = np.arange(
            w - 1,
            dtype=np.int32,
        )[None, :]

        top_left = (
            row * w + col
        ).ravel()

        top_right = (
            top_left + 1
        )

        bottom_left = (
            (row + 1) * w + col
        ).ravel()

        bottom_right = (
            bottom_left + 1
        )

        faces = np.vstack(
            (
                np.column_stack(
                    (
                        top_left,
                        bottom_left,
                        top_right,
                    )
                ),
                np.column_stack(
                    (
                        top_right,
                        bottom_left,
                        bottom_right,
                    )
                ),
            )
        ).astype(
            np.int32
        )
    else:
        faces = np.empty(
            (
                0,
                3,
            ),
            dtype=np.int32,
        )

    vertex_colors = _rgb_strings(
        surface_image
    )

    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=(
                faces[:, 0]
                if len(faces)
                else []
            ),
            j=(
                faces[:, 1]
                if len(faces)
                else []
            ),
            k=(
                faces[:, 2]
                if len(faces)
                else []
            ),
            vertexcolor=vertex_colors,
            opacity=1.0,
            flatshading=False,
            lighting=dict(
                ambient=0.72,
                diffuse=0.92,
                specular=0.28,
                roughness=0.72,
                fresnel=0.08,
            ),
            hoverinfo="skip",
            name="Image-Aligned Surface",
            showlegend=False,
        )
    )

    # --------------------------------------------------------
    # Boundaries
    # --------------------------------------------------------

    if show_boundaries:
        _add_defect_boundaries(
            figure,
            height,
            mask,
            pixel_size,
        )

    # --------------------------------------------------------
    # Defect metadata
    # --------------------------------------------------------

    metadata = _extract_defect_metadata(
        engineering_result,
        confidence,
    )

    if show_markers:
        _add_defect_markers(
            figure,
            height,
            mask,
            metadata,
            pixel_size,
        )

    # --------------------------------------------------------
    # Health summary annotation
    # --------------------------------------------------------

    health = _value(
        engineering_result,
        "road_health",
        None,
    )

    health_score = _safe_float(
        _value(
            health,
            "score",
            None,
        )
    )

    health_condition = _value(
        health,
        "condition",
        None,
    )

    if hasattr(
        health_condition,
        "value",
    ):
        health_condition = (
            health_condition.value
        )

    annotation_lines = []

    if health_score is not None:
        annotation_lines.append(
            f"Road health: "
            f"{health_score:.1f}/100"
        )

    if health_condition is not None:
        annotation_lines.append(
            "Condition: "
            f"{str(health_condition).replace('_', ' ').title()}"
        )

    if confidence is not None:
        annotation_lines.append(
            f"AI confidence: "
            f"{float(confidence):.1%}"
        )

    if metadata:
        annotation_lines.append(
            f"Detected regions: "
            f"{len(metadata)}"
        )

    if annotation_lines:
        figure.add_annotation(
            x=0.015,
            y=0.98,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            text="<br>".join(
                annotation_lines
            ),
            showarrow=False,
            align="left",
            bgcolor="rgba(10,15,22,0.88)",
            bordercolor="#475569",
            borderwidth=1,
            borderpad=8,
            font=dict(
                size=12,
                color="#f8fafc",
            ),
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    max_dimension = max(
        float(w),
        float(h),
        1.0,
    )

    figure.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                size=18,
                color="#f8fafc",
            ),
        ),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        margin=dict(
            l=0,
            r=0,
            t=55,
            b=0,
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#475569",
            font=dict(
                color="#ffffff",
                size=13,
            ),
        ),
        legend=dict(
            bgcolor="rgba(15,23,42,0.82)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(
                color="#f8fafc",
                size=11,
            ),
        ),
        scene=dict(
            bgcolor="#0b0f14",
            xaxis=dict(
                title="Image X",
                backgroundcolor="#0b0f14",
                gridcolor="#334155",
                zerolinecolor="#475569",
                showspikes=False,
            ),
            yaxis=dict(
                title="Image Y",
                backgroundcolor="#0b0f14",
                gridcolor="#334155",
                zerolinecolor="#475569",
                autorange="reversed",
                showspikes=False,
            ),
            zaxis=dict(
                title="Visual Depth",
                backgroundcolor="#0b0f14",
                gridcolor="#334155",
                zerolinecolor="#475569",
                showspikes=False,
            ),
            aspectmode="manual",
            aspectratio=dict(
                x=max(
                    w / max_dimension,
                    0.35,
                ),
                y=max(
                    h / max_dimension,
                    0.35,
                ),
                z=0.48,
            ),
            camera=dict(
                eye=dict(
                    x=1.45,
                    y=1.45,
                    z=1.05,
                ),
                center=dict(
                    x=0.0,
                    y=0.0,
                    z=-0.05,
                ),
            ),
            dragmode="orbit",
        ),
        uirevision="roadxai-interactive-3d",
    )

    return figure


# ============================================================
# STANDALONE MESH VIEW
# ============================================================


def create_plotly_mesh(
    mesh: MeshResult,
    title: str = "RoadXAI 3D Mesh",
):
    """
    Create a standalone interactive triangular mesh.
    """

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for 3D visualization."
        ) from exc

    vertices = np.asarray(
        mesh.vertices
    )

    faces = np.asarray(
        mesh.faces
    )

    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
    ):
        raise ValueError(
            "vertices must have shape [N, 3]."
        )

    if (
        faces.ndim != 2
        or faces.shape[1] != 3
    ):
        raise ValueError(
            "faces must have shape [M, 3]."
        )

    figure = go.Figure()

    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=np.abs(
                vertices[:, 2]
            ),
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(
                title="Visual depth",
            ),
            opacity=0.92,
            lighting=dict(
                ambient=0.72,
                diffuse=0.92,
                specular=0.25,
            ),
            hovertemplate=(
                "Image X: %{x:.1f}"
                "<br>"
                "Image Y: %{y:.1f}"
                "<br>"
                "Visual depth: "
                "%{z:.3f}"
                "<extra></extra>"
            ),
            name="3D Mesh",
        )
    )

    figure.update_layout(
        title=title,
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0,
        ),
        scene=dict(
            bgcolor="#0b0f14",
            xaxis_title="Image X",
            yaxis_title="Image Y",
            zaxis_title="Visual Depth",
            yaxis=dict(
                autorange="reversed"
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=1.45,
                    y=1.45,
                    z=1.05,
                )
            ),
        ),
        uirevision="roadxai-mesh",
    )

    return figure


# ============================================================
# SAVE
# ============================================================


def save_plotly_figure(
    figure: Any,
    output_path: Union[
        str,
        Path,
    ],
) -> Path:
    """
    Save an interactive Plotly figure as standalone HTML.
    """

    path = Path(
        output_path
    )

    if path.suffix.lower() != ".html":
        path = path.with_suffix(
            ".html"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(
        str(path),
        include_plotlyjs=True,
    )

    return path


# ============================================================
# COMPLETE PIPELINE
# ============================================================


def generate_3d_visualization(
    mask: ArrayLike,
    max_depth: float = 1.0,
    pixel_size: float = 1.0,
    title: str = "RoadXAI Interactive Road Inspection",
    road_image: Optional[
        ArrayLike
    ] = None,
    engineering_result: Any = None,
    confidence: Optional[
        float
    ] = None,
    show_markers: bool = True,
    show_boundaries: bool = True,
    display_mode: str = "inspection",
    vertical_exaggeration: float = 1.0,
    probability_map: Optional[
        ArrayLike
    ] = None,
    xai_heatmap: Optional[
        ArrayLike
    ] = None,
):
    """
    Complete RoadXAI 3D inspection pipeline.

    Returns
    -------
    tuple
        (
            HeightMapResult,
            MeshResult,
            Plotly Figure
        )

    Backward compatibility
    ----------------------
    The original function signature is preserved. New features are
    optional, so older callers can continue using:

        generate_3d_visualization(
            mask=mask,
            road_image=image,
        )
    """

    validated_mask = validate_mask(
        mask
    )

    image = None

    if road_image is not None:
        image = _prepare_image(
            road_image,
            validated_mask.shape,
        )

    heightmap_result = create_heightmap(
        validated_mask,
        max_depth=float(
            max_depth
        ),
        smoothing=1.2,
        min_visible_depth=min(
            0.12,
            float(max_depth),
        ),
    )

    mesh_result = heightmap_to_mesh(
        heightmap_result.heightmap,
        pixel_size=float(
            pixel_size
        ),
    )

    figure = create_plotly_surface(
        heightmap_result.heightmap,
        pixel_size=float(
            pixel_size
        ),
        title=title,
        road_image=image,
        defect_mask=validated_mask,
        engineering_result=engineering_result,
        confidence=confidence,
        show_markers=bool(
            show_markers
        ),
        show_boundaries=bool(
            show_boundaries
        ),
        display_mode=display_mode,
        vertical_exaggeration=float(
            vertical_exaggeration
        ),
        probability_map=probability_map,
        xai_heatmap=xai_heatmap,
    )

    return (
        heightmap_result,
        mesh_result,
        figure,
    )


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "HeightMapResult",
    "MeshResult",
    "Defect3DInfo",
    "validate_mask",
    "normalize_array",
    "create_heightmap",
    "create_coordinate_grid",
    "heightmap_to_mesh",
    "create_plotly_surface",
    "create_plotly_mesh",
    "save_plotly_figure",
    "generate_3d_visualization",
]