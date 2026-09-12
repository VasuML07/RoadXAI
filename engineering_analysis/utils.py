"""
Utility functions for RoadXAI engineering analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .module import (
    Calibration,
    DefectMeasurements,
    EngineeringAnalysisResult,
    RepairCostResult,
    RoadHealthResult,
    SeverityLevel,
    SeverityResult,
)


def mask_to_binary(
    mask: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert a probability or grayscale mask into a binary mask.

    Returns:
        uint8 mask containing 0 and 255.
    """

    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy array.")

    if mask.size == 0:
        raise ValueError("mask cannot be empty.")

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise ValueError(
                "Expected a single-channel mask."
            )

    if mask.ndim != 2:
        raise ValueError(
            "mask must have shape [H, W]."
        )

    if np.issubdtype(mask.dtype, np.integer):
        binary = mask > 0
    else:
        binary = mask >= threshold

    return (
        binary.astype(np.uint8) * 255
    )


def clean_binary_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Remove small holes/noise using morphological operations.
    """

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be a positive odd number."
        )

    if iterations <= 0:
        raise ValueError(
            "iterations must be greater than zero."
        )

    binary = mask_to_binary(mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations,
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations,
    )

    return cleaned


def remove_small_components(
    mask: np.ndarray,
    min_area_pixels: int = 10,
) -> np.ndarray:
    """
    Remove connected components smaller than min_area_pixels.
    """

    if min_area_pixels < 0:
        raise ValueError(
            "min_area_pixels cannot be negative."
        )

    binary = mask_to_binary(mask)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(
        binary
    )

    for label in range(
        1,
        num_labels,
    ):
        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area >= min_area_pixels:
            cleaned[labels == label] = 255

    return cleaned


def calculate_mask_area_ratio(
    mask: np.ndarray,
) -> float:
    """
    Calculate the fraction of image pixels occupied by defects.
    """

    binary = mask_to_binary(mask)

    total_pixels = binary.size

    if total_pixels == 0:
        return 0.0

    defect_pixels = np.count_nonzero(
        binary
    )

    return float(
        defect_pixels / total_pixels
    )


def calculate_physical_area(
    pixel_area: float,
    calibration: Calibration,
) -> float:
    """Convert pixel area to square meters."""

    if pixel_area < 0:
        raise ValueError(
            "pixel_area cannot be negative."
        )

    if not isinstance(
        calibration,
        Calibration,
    ):
        raise TypeError(
            "calibration must be a Calibration object."
        )

    return calibration.pixel_area_to_square_meters(
        pixel_area
    )


def calculate_physical_length(
    pixel_length: float,
    calibration: Calibration,
) -> float:
    """Convert pixel length to meters."""

    if pixel_length < 0:
        raise ValueError(
            "pixel_length cannot be negative."
        )

    if not isinstance(
        calibration,
        Calibration,
    ):
        raise TypeError(
            "calibration must be a Calibration object."
        )

    return calibration.pixel_length_to_meters(
        pixel_length
    )


def severity_to_numeric(
    severity: SeverityLevel | SeverityResult | str,
) -> int:
    """
    Convert severity into an ordinal numeric level.

    Mapping:
        low      -> 1
        moderate -> 2
        high     -> 3
        critical -> 4
    """

    if isinstance(
        severity,
        SeverityResult,
    ):
        severity = severity.level

    if isinstance(
        severity,
        SeverityLevel,
    ):
        severity_name = severity.value
    else:
        severity_name = str(
            severity
        ).lower().strip()

    mapping = {
        "low": 1,
        "moderate": 2,
        "high": 3,
        "critical": 4,
    }

    if severity_name not in mapping:
        raise ValueError(
            f"Unknown severity level: {severity_name}"
        )

    return mapping[severity_name]


def severity_summary(
    severities: Iterable[
        SeverityResult
    ],
) -> dict[str, Any]:
    """
    Summarize severity results.
    """

    severity_list = list(
        severities
    )

    counts = {
        level.value: 0
        for level in SeverityLevel
    }

    scores: list[float] = []

    for result in severity_list:
        counts[
            result.level.value
        ] += 1

        scores.append(
            float(result.score)
        )

    return {
        "total_defects": len(
            severity_list
        ),
        "low": counts["low"],
        "moderate": counts["moderate"],
        "high": counts["high"],
        "critical": counts["critical"],
        "average_score": (
            float(np.mean(scores))
            if scores
            else 0.0
        ),
        "maximum_score": (
            float(max(scores))
            if scores
            else 0.0
        ),
    }


def road_condition_priority(
    condition: str,
) -> int:
    """
    Convert road condition into maintenance priority.

    Higher number means higher priority.
    """

    mapping = {
        "excellent": 1,
        "good": 2,
        "fair": 3,
        "poor": 4,
        "critical": 5,
    }

    normalized = condition.lower().strip()

    if normalized not in mapping:
        raise ValueError(
            f"Unknown road condition: {condition}"
        )

    return mapping[normalized]


def format_area(
    area_m2: float | None,
    precision: int = 2,
) -> str:
    """Format an area value for display."""

    if area_m2 is None:
        return "N/A"

    if area_m2 < 0:
        raise ValueError(
            "area_m2 cannot be negative."
        )

    return f"{area_m2:.{precision}f} m²"


def format_length(
    length_m: float | None,
    precision: int = 2,
) -> str:
    """Format a length value for display."""

    if length_m is None:
        return "N/A"

    if length_m < 0:
        raise ValueError(
            "length_m cannot be negative."
        )

    return f"{length_m:.{precision}f} m"


def format_cost(
    cost: float,
    currency: str = "INR",
    precision: int = 2,
) -> str:
    """Format a repair-cost estimate."""

    if cost < 0:
        raise ValueError(
            "cost cannot be negative."
        )

    if not currency.strip():
        raise ValueError(
            "currency cannot be empty."
        )

    return (
        f"{currency.upper()} "
        f"{cost:,.{precision}f}"
    )


def engineering_result_to_dict(
    result: EngineeringAnalysisResult,
) -> dict[str, Any]:
    """
    Convert a complete engineering result into a JSON-safe dictionary.
    """

    if not isinstance(
        result,
        EngineeringAnalysisResult,
    ):
        raise TypeError(
            "result must be an EngineeringAnalysisResult."
        )

    defects = [
        asdict(defect)
        for defect in result.defects
    ]

    severities = []

    for severity in result.severity:
        severity_dict = asdict(
            severity
        )

        severity_dict["level"] = (
            severity.level.value
        )

        severities.append(
            severity_dict
        )

    road_health = asdict(
        result.road_health
    )

    repair_cost = asdict(
        result.repair_cost
    )

    return {
        "defects": defects,
        "severity": severities,
        "road_health": road_health,
        "repair_cost": repair_cost,
    }


def save_engineering_result(
    result: EngineeringAnalysisResult,
    path: str | Path,
) -> Path:
    """Save engineering-analysis results as JSON."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = engineering_result_to_dict(
        result
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def draw_defect_contours(
    image: np.ndarray,
    mask: np.ndarray,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detected defect boundaries on an image.
    """

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "image must be a NumPy array."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape [H, W, 3]."
        )

    if thickness <= 0:
        raise ValueError(
            "thickness must be greater than zero."
        )

    binary = mask_to_binary(
        mask
    )

    if binary.shape != image.shape[:2]:
        raise ValueError(
            "Image and mask spatial dimensions must match."
        )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    output = image.copy()

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 255),
        thickness,
    )

    return output


def draw_defect_labels(
    image: np.ndarray,
    defects: list[DefectMeasurements],
    severities: list[SeverityResult],
) -> np.ndarray:
    """
    Draw defect indices and severity labels.

    The two lists must correspond by index.
    """

    if len(defects) != len(severities):
        raise ValueError(
            "defects and severities must have the same length."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape [H, W, 3]."
        )

    output = image.copy()

    for index, (
        defect,
        severity,
    ) in enumerate(
        zip(
            defects,
            severities,
        ),
        start=1,
    ):
        x = int(
            round(defect.centroid_x)
        )

        y = int(
            round(defect.centroid_y)
        )

        label = (
            f"#{index} "
            f"{severity.level.value} "
            f"{severity.score:.0f}"
        )

        cv2.putText(
            output,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def validate_calibration(
    pixels_per_meter: float,
) -> Calibration:
    """
    Create and validate a Calibration object.
    """

    return Calibration(
        pixels_per_meter=float(
            pixels_per_meter
        )
    )


def summarize_repair_cost(
    result: RepairCostResult,
) -> dict[str, float | str]:
    """Return a compact repair-cost summary."""

    return {
        "area_m2": result.estimated_area_m2,
        "rate_per_m2": result.rate_per_m2,
        "estimated_cost": result.estimated_cost,
        "currency": result.currency,
    }


__all__ = [
    "mask_to_binary",
    "clean_binary_mask",
    "remove_small_components",
    "calculate_mask_area_ratio",
    "calculate_physical_area",
    "calculate_physical_length",
    "severity_to_numeric",
    "severity_summary",
    "road_condition_priority",
    "format_area",
    "format_length",
    "format_cost",
    "engineering_result_to_dict",
    "save_engineering_result",
    "draw_defect_contours",
    "draw_defect_labels",
    "validate_calibration",
    "summarize_repair_cost",
]