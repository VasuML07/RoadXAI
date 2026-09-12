"""
RoadXAI 3D Visualization Utilities.

Helper functions for preparing heightmaps, validating visualization
inputs, exporting 3D data, and generating visualization metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .module import (
    HeightMapResult,
    MeshResult,
    create_heightmap,
    heightmap_to_mesh,
)


def validate_visualization_input(
    mask: np.ndarray,
    pixel_size: float = 1.0,
    max_depth: float = 1.0,
) -> np.ndarray:
    """
    Validate inputs required for 3D visualization.

    Returns
    -------
    np.ndarray
        Validated binary mask.
    """
    array = np.asarray(mask)

    if array.ndim != 2:
        raise ValueError(
            f"Mask must be 2D, received shape {array.shape}."
        )

    if array.size == 0:
        raise ValueError("Mask cannot be empty.")

    if not np.isfinite(array).all():
        raise ValueError("Mask contains NaN or infinite values.")

    if pixel_size <= 0:
        raise ValueError("pixel_size must be greater than zero.")

    if max_depth < 0:
        raise ValueError("max_depth cannot be negative.")

    return (array > 0).astype(np.uint8)


def calculate_heightmap_statistics(
    heightmap: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate descriptive statistics for a heightmap.
    """
    values = np.asarray(heightmap, dtype=np.float32)

    if values.ndim != 2:
        raise ValueError("heightmap must be a 2D array.")

    if values.size == 0:
        raise ValueError("heightmap cannot be empty.")

    positive_values = values[values > 0]

    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "defect_mean": (
            float(positive_values.mean())
            if positive_values.size > 0
            else 0.0
        ),
        "defect_max": (
            float(positive_values.max())
            if positive_values.size > 0
            else 0.0
        ),
        "nonzero_pixels": int(np.count_nonzero(values)),
        "total_pixels": int(values.size),
    }


def calculate_mesh_statistics(
    mesh: MeshResult,
) -> Dict[str, int]:
    """
    Calculate basic statistics for a generated mesh.
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must have shape [N, 3].")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must have shape [M, 3].")

    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
    }


def prepare_heightmap(
    mask: np.ndarray,
    max_depth: float = 1.0,
    smoothing: float = 0.0,
) -> HeightMapResult:
    """
    Validate a mask and generate its heightmap.
    """
    validated_mask = validate_visualization_input(
        mask,
        max_depth=max_depth,
    )

    return create_heightmap(
        validated_mask,
        max_depth=max_depth,
        smoothing=smoothing,
    )


def prepare_mesh(
    heightmap: np.ndarray,
    pixel_size: float = 1.0,
) -> MeshResult:
    """
    Convert a validated heightmap into mesh data.
    """
    values = np.asarray(heightmap, dtype=np.float32)

    if values.ndim != 2:
        raise ValueError("heightmap must be a 2D array.")

    if not np.isfinite(values).all():
        raise ValueError(
            "heightmap contains NaN or infinite values."
        )

    return heightmap_to_mesh(
        values,
        pixel_size=pixel_size,
    )


def export_heightmap(
    heightmap: np.ndarray,
    output_path: Union[str, Path],
) -> Path:
    """
    Save a heightmap as a NumPy .npy file.
    """
    path = Path(output_path)

    if path.suffix.lower() != ".npy":
        path = path.with_suffix(".npy")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(path, np.asarray(heightmap, dtype=np.float32))

    return path


def load_heightmap(
    input_path: Union[str, Path],
) -> np.ndarray:
    """
    Load a heightmap from a .npy file.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Heightmap file not found: {path}"
        )

    values = np.load(path)

    if values.ndim != 2:
        raise ValueError(
            f"Expected a 2D heightmap, got {values.shape}."
        )

    return values.astype(np.float32)


def export_mesh(
    mesh: MeshResult,
    output_path: Union[str, Path],
) -> Path:
    """
    Export mesh vertices and faces to a compressed NumPy archive.
    """
    path = Path(output_path)

    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
    )

    return path


def load_mesh(
    input_path: Union[str, Path],
) -> MeshResult:
    """
    Load mesh data from a compressed NumPy archive.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Mesh file not found: {path}"
        )

    data = np.load(path)

    if "vertices" not in data or "faces" not in data:
        raise ValueError(
            "Mesh archive must contain 'vertices' and 'faces'."
        )

    vertices = data["vertices"].astype(np.float32)
    faces = data["faces"].astype(np.int32)

    return MeshResult(
        vertices=vertices,
        faces=faces,
    )


def build_visualization_metadata(
    heightmap_result: HeightMapResult,
    mesh_result: Optional[MeshResult] = None,
    pixel_size: float = 1.0,
) -> Dict[str, Any]:
    """
    Build JSON-serializable metadata for a 3D visualization.
    """
    metadata: Dict[str, Any] = {
        "heightmap": {
            "shape": list(
                heightmap_result.heightmap.shape
            ),
            "min_height": heightmap_result.min_height,
            "max_height": heightmap_result.max_height,
            "mean_height": heightmap_result.mean_height,
            "defect_pixels": heightmap_result.defect_pixels,
        },
        "pixel_size": float(pixel_size),
    }

    if mesh_result is not None:
        metadata["mesh"] = calculate_mesh_statistics(
            mesh_result
        )

    return metadata


def save_visualization_metadata(
    metadata: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Save visualization metadata as formatted JSON.
    """
    path = Path(output_path)

    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    return path


def generate_visualization_data(
    mask: np.ndarray,
    max_depth: float = 1.0,
    pixel_size: float = 1.0,
    smoothing: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate complete reusable 3D visualization data.

    Returns
    -------
    dict
        Heightmap result, mesh result, and metadata.
    """
    heightmap_result = prepare_heightmap(
        mask,
        max_depth=max_depth,
        smoothing=smoothing,
    )

    mesh_result = prepare_mesh(
        heightmap_result.heightmap,
        pixel_size=pixel_size,
    )

    metadata = build_visualization_metadata(
        heightmap_result,
        mesh_result,
        pixel_size=pixel_size,
    )

    return {
        "heightmap": heightmap_result.heightmap,
        "mesh": mesh_result,
        "metadata": metadata,
    }