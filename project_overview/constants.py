"""
RoadXAI Project Constants
=========================

Centralized constants used by the RoadXAI project.
"""

# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

PROJECT_NAME = "RoadXAI"
PROJECT_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Image Processing
# ---------------------------------------------------------------------------

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS = 3
MASK_CHANNELS = 1


# ---------------------------------------------------------------------------
# Dataset Splits
# ---------------------------------------------------------------------------

TRAIN_SPLIT = 0.80
VALIDATION_SPLIT = 0.20

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Damage Classes
# ---------------------------------------------------------------------------

CRACK_CLASS = "crack"
POTHOLE_CLASS = "pothole"

CLASS_NAMES = (
    CRACK_CLASS,
    POTHOLE_CLASS,
)

NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Dataset Names
# ---------------------------------------------------------------------------

CRACK500_DATASET = "CRACK500"
POTHOLE_YOLO_DATASET = "Pothole_Segmentation_YOLOv8"
PUBLIC_POTHOLE_DATASET = "PUBLIC POTHOLE DATASET"

DATASET_NAMES = (
    CRACK500_DATASET,
    POTHOLE_YOLO_DATASET,
    PUBLIC_POTHOLE_DATASET,
)


# ---------------------------------------------------------------------------
# File Extensions
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_constants() -> bool:
    """Validate the project constants."""

    if IMAGE_WIDTH <= 0 or IMAGE_HEIGHT <= 0:
        return False

    if IMAGE_CHANNELS <= 0 or MASK_CHANNELS <= 0:
        return False

    if NUM_CLASSES != len(CLASS_NAMES):
        return False

    if len(CLASS_NAMES) == 0:
        return False

    if len(DATASET_NAMES) == 0:
        return False

    if not 0 < TRAIN_SPLIT < 1:
        return False

    if not 0 < VALIDATION_SPLIT < 1:
        return False

    if abs(TRAIN_SPLIT + VALIDATION_SPLIT - 1.0) > 1e-6:
        return False

    if RANDOM_SEED < 0:
        return False

    return True


if __name__ == "__main__":
    print("RoadXAI Constants")
    print("=" * 40)
    print(f"Project       : {PROJECT_NAME}")
    print(f"Version       : {PROJECT_VERSION}")
    print(f"Image Size    : {IMAGE_SIZE}")
    print(f"Classes       : {CLASS_NAMES}")
    print(f"Datasets      : {DATASET_NAMES}")
    print(f"Validation    : {'PASS' if validate_constants() else 'FAIL'}")