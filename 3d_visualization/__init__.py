"""
RoadXAI 3D Visualization Package.

Provides image-aligned 3D road-defect visualization utilities.
"""

from .module import (
    HeightMapResult,
    MeshResult,
    validate_mask,
    normalize_array,
    create_heightmap,
    create_coordinate_grid,
    heightmap_to_mesh,
    create_plotly_surface,
    create_plotly_mesh,
    save_plotly_figure,
    generate_3d_visualization,
)

__version__ = "2.0.0"
__author__ = "RoadXAI"

__all__ = [
    "HeightMapResult",
    "MeshResult",
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