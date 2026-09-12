"""
RoadXAI Project Configuration


Central configuration for project-wide settings used across RoadXAI.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"

CRACK500_DIR = DATASETS_DIR / "CRACK500"

POTHOLE_YOLO_DIR = DATASETS_DIR / "Pothole_Segmentation_YOLOv8"

PUBLIC_POTHOLE_DIR = DATASETS_DIR / "PUBLIC POTHOLE DATASET"


# ---------------------------------------------------------------------------
# Image Configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE = (512, 512)

IMAGE_CHANNELS = 3

MASK_CHANNELS = 1


# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------

TRAIN_SPLIT = 0.80

VAL_SPLIT = 0.20

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Road Damage Configuration
# ---------------------------------------------------------------------------

NUM_CLASSES = 2

CLASS_NAMES = [
    "crack",
    "pothole",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config() -> bool:
    """
    Validate the project-wide configuration.

    Returns
    -------
    bool
        True when the configuration is valid.
    """

    if IMAGE_SIZE[0] <= 0 or IMAGE_SIZE[1] <= 0:
        return False

    if IMAGE_CHANNELS <= 0:
        return False

    if MASK_CHANNELS <= 0:
        return False

    if NUM_CLASSES != len(CLASS_NAMES):
        return False

    if not 0 < TRAIN_SPLIT < 1:
        return False

    if not 0 < VAL_SPLIT < 1:
        return False

    if abs((TRAIN_SPLIT + VAL_SPLIT) - 1.0) > 1e-6:
        return False

    if RANDOM_SEED < 0:
        return False

    return True


if __name__ == "__main__":
    print("RoadXAI Configuration")
    print("=" * 40)
    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Datasets     : {DATASETS_DIR}")
    print(f"Image Size   : {IMAGE_SIZE}")
    print(f"Classes      : {CLASS_NAMES}")
    print(f"Train Split  : {TRAIN_SPLIT}")
    print(f"Val Split    : {VAL_SPLIT}")
    print(f"Random Seed  : {RANDOM_SEED}")
    print(f"Validation   : {'PASS' if validate_config() else 'FAIL'}")