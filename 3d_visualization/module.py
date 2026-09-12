"""
RoadXAI 3D Visualization Module.

Creates an image-aligned 3D representation of a road-defect
segmentation mask.

Important:
The Z axis is a normalized visualization depth unless a real
depth/calibration value is explicitly supplied. A single RGB image
cannot reliably determine the physical depth of a pothole or crack
in centimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


ArrayLike = np.ndarray


@dataclass
class HeightMapResult:
    """Container for a generated defect depth map."""

    heightmap: np.ndarray
    min_height: float
    max_height: float
    mean_height: float
    defect_pixels: int
    depth_unit: str = "normalized"
    estimated_max_depth_cm: Optional[float] = None


@dataclass
class MeshResult:
    """Container for 3D mesh data."""

    vertices: np.ndarray
    faces: np.ndarray


def validate_mask(mask: ArrayLike) -> np.ndarray:
    """Validate and normalize a 2D defect mask."""
    array = np.asarray(mask)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, received shape {array.shape}."
        )

    if array.size == 0:
        raise ValueError("Mask cannot be empty.")

    if not np.isfinite(array).all():
        raise ValueError("Mask contains NaN or infinite values.")

    return (array > 0).astype(np.float32)


def normalize_array(
    array: ArrayLike,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.ndarray:
    """Min-max normalize an array into the requested range."""
    values = np.asarray(array, dtype=np.float32)

    if values.size == 0:
        raise ValueError("Array cannot be empty.")

    if max_value < min_value:
        raise ValueError(
            "max_value must be greater than or equal to min_value."
        )

    current_min = float(values.min())
    current_max = float(values.max())

    if current_max - current_min <= 1e-8:
        return np.full_like(
            values,
            min_value,
            dtype=np.float32,
        )

    normalized = (
        values - current_min
    ) / (
        current_max - current_min
    )

    return (
        normalized * (max_value - min_value)
        + min_value
    ).astype(np.float32)


def _gaussian_blur(
    array: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Smooth an array without changing its data type."""
    if sigma <= 0:
        return array.astype(np.float32)

    try:
        import cv2
    except ImportError:
        return array.astype(np.float32)

    kernel_size = max(
        3,
        int(round(float(sigma) * 6)) | 1,
    )

    return cv2.GaussianBlur(
        array.astype(np.float32),
        (kernel_size, kernel_size),
        float(sigma),
    ).astype(np.float32)


def _component_depth_profile(
    component: np.ndarray,
    max_depth: float,
    min_visible_depth: float,
) -> np.ndarray:
    """
    Create a downward depth profile for one connected component.

    Every detected pixel receives a small non-zero depression so that
    very thin cracks remain visible. Wider regions become progressively
    deeper toward their interiors.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for 3D depth-map generation."
        ) from exc

    binary = (
        component.astype(np.uint8) * 255
    )

    distance = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5,
    ).astype(np.float32)

    positive = distance > 0

    if not positive.any():
        return (
            component.astype(np.float32)
            * float(min_visible_depth)
        )

    distance_max = float(
        distance[positive].max()
    )

    if distance_max <= 1e-8:
        normalized = np.ones_like(
            distance,
            dtype=np.float32,
        )
    else:
        normalized = (
            distance / distance_max
        ).astype(np.float32)

    # Keep a visible floor for thin cracks.
    normalized = (
        float(min_visible_depth / max_depth)
        + (
            1.0
            - float(min_visible_depth / max_depth)
        )
        * normalized
    )

    depth = (
        normalized
        * float(max_depth)
    )

    depth *= component.astype(
        np.float32
    )

    return depth.astype(np.float32)


def create_heightmap(
    mask: ArrayLike,
    max_depth: float = 1.0,
    smoothing: float = 0.8,
    min_visible_depth: float = 0.10,
) -> HeightMapResult:
    """
    Convert a binary defect mask into a downward synthetic depth map.

    Road surface:
        Z = 0

    Detected defect:
        Z < 0

    Thin cracks receive a minimum visible depression, while wider
    connected regions become deeper toward their interiors.

    This is a visualization of segmentation geometry. It is NOT a
    measurement of physical pothole/crack depth.
    """
    mask_array = validate_mask(mask)

    if max_depth <= 0:
        raise ValueError(
            "max_depth must be greater than zero."
        )

    if not 0.0 < min_visible_depth <= 1.0:
        raise ValueError(
            "min_visible_depth must be in the range (0, 1]."
        )

    defect_pixels = int(
        np.count_nonzero(mask_array)
    )

    if defect_pixels == 0:
        heightmap = np.zeros_like(
            mask_array,
            dtype=np.float32,
        )

        return HeightMapResult(
            heightmap=heightmap,
            min_height=0.0,
            max_height=0.0,
            mean_height=0.0,
            defect_pixels=0,
        )

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for 3D depth-map generation."
        ) from exc

    binary = (
        mask_array > 0
    ).astype(np.uint8)

    # Do NOT perform erosion/opening here. Thin cracks can be only
    # one or two pixels wide at 256x256 and morphology could erase them.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    depth_map = np.zeros_like(
        mask_array,
        dtype=np.float32,
    )

    # Process each connected defect independently so separate cracks/
    # potholes remain separate depressions rather than one large basin.
    for label in range(1, num_labels):
        component = (
            labels == label
        ).astype(np.float32)

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area <= 0:
            continue

        component_depth = _component_depth_profile(
            component,
            max_depth=float(max_depth),
            min_visible_depth=(
                float(max_depth)
                * float(min_visible_depth)
            ),
        )

        # Slightly deeper visualization for larger components, but
        # never beyond max_depth. This helps holes stand out while
        # keeping tiny cracks visible.
        if area > 20:
            area_factor = min(
                1.0,
                0.65
                + 0.35
                * np.sqrt(
                    min(area / 2000.0, 1.0)
                ),
            )
            component_depth *= float(
                area_factor
            )

        if smoothing > 0 and area >= 9:
            smoothed = _gaussian_blur(
                component_depth,
                smoothing,
            )

            # Restore the visible floor so narrow defects do not
            # disappear after smoothing.
            smoothed *= component

            floor = (
                component
                * float(max_depth)
                * float(min_visible_depth)
            )

            component_depth = np.maximum(
                smoothed,
                floor,
            )

        depth_map = np.maximum(
            depth_map,
            component_depth,
        )

    # Convert positive synthetic depth into a physical-looking
    # downward depression.
    heightmap = -np.clip(
        depth_map,
        0.0,
        float(max_depth),
    ).astype(np.float32)

    defect_values = (
        -heightmap[mask_array > 0]
    )

    return HeightMapResult(
        heightmap=heightmap,
        min_height=float(heightmap.min()),
        max_height=float(heightmap.max()),
        mean_height=float(
            heightmap[mask_array > 0].mean()
        ),
        defect_pixels=defect_pixels,
        depth_unit="normalized",
        estimated_max_depth_cm=None,
    )


def create_coordinate_grid(
    height: int,
    width: int,
    pixel_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create X/Y coordinate grids matching image rows and columns."""
    if height <= 0 or width <= 0:
        raise ValueError(
            "height and width must be positive."
        )

    if pixel_size <= 0:
        raise ValueError(
            "pixel_size must be positive."
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

    return np.meshgrid(
        x,
        y,
    )


def heightmap_to_mesh(
    heightmap: ArrayLike,
    pixel_size: float = 1.0,
) -> MeshResult:
    """Convert a 2D depth map into a triangular mesh."""
    height = np.asarray(
        heightmap,
        dtype=np.float32,
    )

    if height.ndim != 2:
        raise ValueError(
            f"Expected a 2D heightmap, received {height.shape}."
        )

    if height.size == 0:
        raise ValueError(
            "Heightmap cannot be empty."
        )

    x_grid, y_grid = create_coordinate_grid(
        height.shape[0],
        height.shape[1],
        pixel_size,
    )

    vertices = np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
            height.ravel(),
        )
    ).astype(np.float32)

    rows, cols = height.shape

    if rows < 2 or cols < 2:
        faces_array = np.empty(
            (0, 3),
            dtype=np.int32,
        )
    else:
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

        top_right = top_left + 1

        bottom_left = (
            (row + 1) * cols + col
        ).ravel()

        bottom_right = bottom_left + 1

        faces_array = np.vstack(
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
        ).astype(np.int32)

    return MeshResult(
        vertices=vertices,
        faces=faces_array,
    )


def _prepare_image(
    road_image: ArrayLike,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Convert an image to RGB uint8 with the exact target size."""
    image = np.asarray(
        road_image
    )

    if image.ndim == 2:
        image = np.repeat(
            image[:, :, None],
            3,
            axis=2,
        )

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(
            "road_image must have shape [H,W,3] or [H,W,4]."
        )

    image = image[:, :, :3]

    if image.shape[:2] != target_shape:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required to resize the road image."
            ) from exc

        image = cv2.resize(
            image,
            (
                target_shape[1],
                target_shape[0],
            ),
            interpolation=cv2.INTER_AREA,
        )

    values = np.asarray(
        image,
        dtype=np.float32,
    )

    if values.size and float(values.max()) <= 1.0:
        values *= 255.0

    return np.clip(
        values,
        0,
        255,
    ).astype(np.uint8)


def _make_vertex_colors(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create image-based vertex colors with subtle defect highlighting."""
    colors = image.copy().astype(
        np.float32
    )

    if mask is not None:
        defect = (
            np.asarray(mask) > 0
        )

        highlight = np.zeros_like(
            colors
        )

        highlight[:, :, 0] = 235.0
        highlight[:, :, 1] = 55.0
        highlight[:, :, 2] = 35.0

        colors[defect] = (
            0.55 * colors[defect]
            + 0.45 * highlight[defect]
        )

    colors = np.clip(
        colors,
        0,
        255,
    ).astype(np.uint8)

    flat = colors.reshape(
        -1,
        3,
    )

    return np.asarray(
        [
            f"rgb({int(r)},{int(g)},{int(b)})"
            for r, g, b in flat
        ],
        dtype=object,
    )


def _add_component_boundaries(
    figure,
    mask: np.ndarray,
    height: np.ndarray,
    pixel_size: float,
) -> None:
    """Add separate boundaries for every connected detected defect."""
    try:
        import cv2
        import plotly.graph_objects as go
    except ImportError:
        return

    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Keep all components, including very small ones.
    for contour_index, contour in enumerate(
        contours,
        start=1,
    ):
        if len(contour) < 2:
            continue

        points = contour.reshape(
            -1,
            2,
        )

        xs = np.clip(
            points[:, 0],
            0,
            height.shape[1] - 1,
        ).astype(np.int32)

        ys = np.clip(
            points[:, 1],
            0,
            height.shape[0] - 1,
        ).astype(np.int32)

        # Put the outline slightly above the depressed surface so it
        # remains visible from different camera angles.
        z = (
            height[ys, xs]
            + 0.018
        )

        figure.add_trace(
            go.Scatter3d(
                x=xs * float(pixel_size),
                y=ys * float(pixel_size),
                z=z,
                mode="lines",
                line=dict(
                    width=4,
                ),
                name=f"Defect {contour_index}",
                showlegend=False,
                hovertemplate=(
                    f"Defect {contour_index}"
                    "<extra></extra>"
                ),
            )
        )


def create_plotly_surface(
    heightmap: ArrayLike,
    pixel_size: float = 1.0,
    title: str = "Road Defect 3D Surface",
    road_image: Optional[ArrayLike] = None,
    defect_mask: Optional[ArrayLike] = None,
):
    """
    Create an image-aligned interactive 3D visualization.

    Defects are rendered downward from Z=0. If a road image is
    supplied, its pixels provide the surface appearance.

    The Z axis is normalized visualization depth, not physical depth.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for 3D visualization."
        ) from exc

    height = np.asarray(
        heightmap,
        dtype=np.float32,
    )

    if height.ndim != 2:
        raise ValueError(
            "heightmap must be a 2D array."
        )

    x_grid, y_grid = create_coordinate_grid(
        height.shape[0],
        height.shape[1],
        pixel_size,
    )

    figure = go.Figure()

    if road_image is not None:
        image = _prepare_image(
            road_image,
            height.shape,
        )

        mask = None

        if defect_mask is not None:
            mask = validate_mask(
                defect_mask
            )

            if mask.shape != height.shape:
                try:
                    import cv2
                except ImportError as exc:
                    raise ImportError(
                        "OpenCV is required to resize the defect mask."
                    ) from exc

                mask = cv2.resize(
                    mask,
                    (
                        height.shape[1],
                        height.shape[0],
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

        vertices = np.column_stack(
            (
                x_grid.ravel(),
                y_grid.ravel(),
                height.ravel(),
            )
        ).astype(np.float32)

        rows, cols = height.shape

        if rows < 2 or cols < 2:
            faces = np.empty(
                (0, 3),
                dtype=np.int32,
            )
        else:
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

            top_right = top_left + 1

            bottom_left = (
                (row + 1) * cols + col
            ).ravel()

            bottom_right = bottom_left + 1

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
            ).astype(np.int32)

        vertex_colors = _make_vertex_colors(
            image,
            mask,
        )

        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0] if len(faces) else [],
                j=faces[:, 1] if len(faces) else [],
                k=faces[:, 2] if len(faces) else [],
                vertexcolor=vertex_colors,
                opacity=1.0,
                flatshading=False,
                hovertemplate=(
                    "Image X: %{x:.0f}<br>"
                    "Image Y: %{y:.0f}<br>"
                    "Normalized depth: %{z:.3f}"
                    "<extra></extra>"
                ),
                name="Road surface",
            )
        )

    else:
        figure.add_trace(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=height,
                colorscale="Viridis",
                colorbar=dict(
                    title="Normalized depth",
                ),
                hovertemplate=(
                    "Image X: %{x:.0f}<br>"
                    "Image Y: %{y:.0f}<br>"
                    "Normalized depth: %{z:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    if defect_mask is not None:
        mask = validate_mask(
            defect_mask
        )

        if mask.shape != height.shape:
            try:
                import cv2
            except ImportError as exc:
                raise ImportError(
                    "OpenCV is required to resize the defect mask."
                ) from exc

            mask = cv2.resize(
                mask,
                (
                    height.shape[1],
                    height.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        _add_component_boundaries(
            figure,
            mask,
            height,
            pixel_size,
        )

    figure.update_layout(
        title=title,
        margin=dict(
            l=0,
            r=0,
            t=45,
            b=0,
        ),
        scene=dict(
            xaxis_title="Image X",
            yaxis_title="Image Y",
            zaxis_title="Normalized Depth",
            aspectmode="auto",
            yaxis=dict(
                autorange="reversed",
            ),
            camera=dict(
                eye=dict(
                    x=1.45,
                    y=1.45,
                    z=1.05,
                )
            ),
        ),
        showlegend=False,
    )

    return figure


def create_plotly_mesh(
    mesh: MeshResult,
    title: str = "Road Defect 3D Mesh",
):
    """Create an interactive Plotly triangular mesh visualization."""
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

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            "vertices must have shape [N, 3]."
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            "faces must have shape [M, 3]."
        )

    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.9,
                intensity=vertices[:, 2],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Normalized depth",
                ),
                hovertemplate=(
                    "X: %{x:.0f}<br>"
                    "Y: %{y:.0f}<br>"
                    "Depth: %{z:.3f}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    figure.update_layout(
        title=title,
        margin=dict(
            l=0,
            r=0,
            t=45,
            b=0,
        ),
        scene=dict(
            xaxis_title="Image X",
            yaxis_title="Image Y",
            zaxis_title="Normalized Depth",
            aspectmode="auto",
            yaxis=dict(
                autorange="reversed",
            ),
            camera=dict(
                eye=dict(
                    x=1.45,
                    y=1.45,
                    z=1.05,
                )
            ),
        ),
    )

    return figure


def save_plotly_figure(
    figure,
    output_path: Union[str, Path],
) -> Path:
    """Save a Plotly figure as an HTML file."""
    path = Path(
        output_path
    )

    if path.suffix.lower() != ".html":
        path = path.with_suffix(".html")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(
        str(path)
    )

    return path


def generate_3d_visualization(
    mask: ArrayLike,
    max_depth: float = 1.0,
    pixel_size: float = 1.0,
    title: str = "Road Defect 3D Visualization",
    road_image: Optional[ArrayLike] = None,
):
    """
    Complete mask-to-3D visualization pipeline.

    The generated defects are downward depressions. Very thin
    connected regions are retained and given a minimum visible depth.

    `road_image` keeps the 3D surface visually aligned with the current
    uploaded photograph.
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
        max_depth=max_depth,
        smoothing=0.8,
        min_visible_depth=0.10,
    )

    mesh_result = heightmap_to_mesh(
        heightmap_result.heightmap,
        pixel_size=pixel_size,
    )

    figure = create_plotly_surface(
        heightmap_result.heightmap,
        pixel_size=pixel_size,
        title=title,
        road_image=image,
        defect_mask=validated_mask,
    )

    return (
        heightmap_result,
        mesh_result,
        figure,
    )
