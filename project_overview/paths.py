"""
RoadXAI Project Paths
=====================

Centralized filesystem paths used by RoadXAI.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Main Directories
# ---------------------------------------------------------------------------

DATASETS_DIR = PROJECT_ROOT / "datasets"

PROJECT_OVERVIEW_DIR = PROJECT_ROOT / "project_overview"

DATASET_PIPELINE_DIR = PROJECT_ROOT / "dataset_pipeline"


# ---------------------------------------------------------------------------
# Dataset Directories
# ---------------------------------------------------------------------------

CRACK500_DIR = DATASETS_DIR / "CRACK500"

POTHOLE_YOLO_DIR = DATASETS_DIR / "Pothole_Segmentation_YOLOv8"

PUBLIC_POTHOLE_DIR = DATASETS_DIR / "PUBLIC POTHOLE DATASET"


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the RoadXAI project root directory."""
    return PROJECT_ROOT


def get_datasets_dir() -> Path:
    """Return the main datasets directory."""
    return DATASETS_DIR


def get_dataset_paths() -> dict[str, Path]:
    """Return paths for all configured datasets."""
    return {
        "CRACK500": CRACK500_DIR,
        "Pothole_Segmentation_YOLOv8": POTHOLE_YOLO_DIR,
        "PUBLIC POTHOLE DATASET": PUBLIC_POTHOLE_DIR,
    }


def dataset_exists(dataset_name: str) -> bool:
    """
    Check whether a configured dataset directory exists.

    Parameters
    ----------
    dataset_name : str
        Name of the configured dataset.

    Returns
    -------
    bool
        True if the dataset directory exists.
    """

    paths = get_dataset_paths()

    if dataset_name not in paths:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    return paths[dataset_name].is_dir()


if __name__ == "__main__":
    print("RoadXAI Project Paths")
    print("=" * 40)

    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Datasets    : {DATASETS_DIR}")

    print("\nDatasets:")

    for name, path in get_dataset_paths().items():
        status = "FOUND" if path.is_dir() else "MISSING"
        print(f"{name}: {status}")
        print(f"  {path}")