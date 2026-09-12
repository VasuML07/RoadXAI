"""
RoadXAI Engineering Analysis Module.

Provides engineering-oriented measurements and estimates for
detected road defects:

    - Defect area
    - Defect length
    - Defect width
    - Severity
    - Road health score
    - Repair cost estimate

Important:
Pixel measurements are converted to physical measurements only when
a calibration scale is supplied. Without calibration, measurements
remain in pixel units.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import cv2
import numpy as np


class SeverityLevel(str, Enum):
    """Road-defect severity categories."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Calibration:
    """
    Pixel-to-metric calibration.

    pixels_per_meter:
        Number of image pixels representing one physical meter.

    This value should ideally come from camera calibration,
    known road dimensions, or another validated reference.
    """

    pixels_per_meter: float

    def __post_init__(self) -> None:
        if self.pixels_per_meter <= 0:
            raise ValueError(
                "pixels_per_meter must be greater than zero."
            )

    @property
    def pixels_per_centimeter(self) -> float:
        """Return pixels per centimeter."""

        return self.pixels_per_meter / 100.0

    def pixel_length_to_meters(
        self,
        pixels: float,
    ) -> float:
        """Convert a pixel length to meters."""

        if pixels < 0:
            raise ValueError(
                "Pixel length cannot be negative."
            )

        return pixels / self.pixels_per_meter

    def pixel_area_to_square_meters(
        self,
        pixels: float,
    ) -> float:
        """Convert pixel area to square meters."""

        if pixels < 0:
            raise ValueError(
                "Pixel area cannot be negative."
            )

        return pixels / (
            self.pixels_per_meter ** 2
        )


@dataclass
class DefectMeasurements:
    """Measured geometric properties of one detected defect."""

    area_pixels: float
    length_pixels: float
    width_pixels: float

    area_m2: float | None = None
    length_m: float | None = None
    width_m: float | None = None

    bounding_box_width_pixels: float = 0.0
    bounding_box_height_pixels: float = 0.0

    centroid_x: float = 0.0
    centroid_y: float = 0.0


@dataclass
class SeverityResult:
    """Severity assessment for a road defect."""

    level: SeverityLevel
    score: float
    area_ratio: float
    explanation: str


@dataclass
class RoadHealthResult:
    """Overall road-health assessment."""

    score: float
    condition: str
    defect_area_ratio: float
    defect_count: int


@dataclass
class RepairCostResult:
    """Estimated repair cost."""

    estimated_area_m2: float
    rate_per_m2: float
    estimated_cost: float
    currency: str


@dataclass
class EngineeringAnalysisResult:
    """Complete engineering analysis."""

    defects: list[DefectMeasurements]
    severity: list[SeverityResult]
    road_health: RoadHealthResult
    repair_cost: RepairCostResult


def validate_mask(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Validate and normalize a binary segmentation mask.

    Accepted input:
        [H, W]
        [H, W, 1]

    Returns:
        uint8 mask containing only 0 and 255.
    """

    if not isinstance(mask, np.ndarray):
        raise TypeError(
            "mask must be a NumPy array."
        )

    if mask.size == 0:
        raise ValueError(
            "mask cannot be empty."
        )

    if mask.ndim == 3:
        if mask.shape[2] != 1:
            raise ValueError(
                "3D masks must have one channel."
            )

        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise ValueError(
            "mask must have shape [H, W]."
        )

    if not np.isfinite(mask).all():
        raise ValueError(
            "mask contains NaN or infinite values."
        )

    return np.where(
        mask > 0,
        255,
        0,
    ).astype(np.uint8)


def calculate_area_pixels(
    mask: np.ndarray,
) -> int:
    """Calculate defect area in pixels."""

    binary_mask = validate_mask(mask)

    return int(
        np.count_nonzero(binary_mask)
    )


def calculate_contours(
    mask: np.ndarray,
) -> list[np.ndarray]:
    """Extract external defect contours."""

    binary_mask = validate_mask(mask)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return list(contours)


def calculate_bounding_box(
    contour: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return x, y, width, height for a contour."""

    if contour.size == 0:
        raise ValueError(
            "Contour cannot be empty."
        )

    return cv2.boundingRect(
        contour
    )


def calculate_length_width(
    contour: np.ndarray,
) -> tuple[float, float]:
    """
    Estimate defect length and width using the minimum-area rectangle.

    Returns:
        length_pixels, width_pixels

    The larger rectangle dimension is treated as length.
    """

    if contour.size == 0:
        raise ValueError(
            "Contour cannot be empty."
        )

    rectangle = cv2.minAreaRect(
        contour
    )

    width, height = rectangle[1]

    if width <= height:
        length = height
        short_width = width
    else:
        length = width
        short_width = height

    return (
        float(length),
        float(short_width),
    )


def calculate_centroid(
    contour: np.ndarray,
) -> tuple[float, float]:
    """Calculate the centroid of a contour."""

    if contour.size == 0:
        raise ValueError(
            "Contour cannot be empty."
        )

    moments = cv2.moments(
        contour
    )

    if moments["m00"] == 0:
        x, y, _, _ = cv2.boundingRect(
            contour
        )

        return (
            float(x),
            float(y),
        )

    centroid_x = (
        moments["m10"]
        / moments["m00"]
    )

    centroid_y = (
        moments["m01"]
        / moments["m00"]
    )

    return (
        float(centroid_x),
        float(centroid_y),
    )


def measure_defect(
    contour: np.ndarray,
    calibration: Calibration | None = None,
) -> DefectMeasurements:
    """
    Measure one defect contour.

    Physical measurements are populated only when calibration
    is supplied.
    """

    area_pixels = float(
        cv2.contourArea(contour)
    )

    length_pixels, width_pixels = (
        calculate_length_width(contour)
    )

    x, y, box_width, box_height = (
        calculate_bounding_box(contour)
    )

    centroid_x, centroid_y = (
        calculate_centroid(contour)
    )

    area_m2 = None
    length_m = None
    width_m = None

    if calibration is not None:
        area_m2 = (
            calibration.pixel_area_to_square_meters(
                area_pixels
            )
        )

        length_m = (
            calibration.pixel_length_to_meters(
                length_pixels
            )
        )

        width_m = (
            calibration.pixel_length_to_meters(
                width_pixels
            )
        )

    return DefectMeasurements(
        area_pixels=area_pixels,
        length_pixels=length_pixels,
        width_pixels=width_pixels,
        area_m2=area_m2,
        length_m=length_m,
        width_m=width_m,
        bounding_box_width_pixels=float(
            box_width
        ),
        bounding_box_height_pixels=float(
            box_height
        ),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
    )


def measure_all_defects(
    mask: np.ndarray,
    calibration: Calibration | None = None,
    min_area_pixels: float = 10.0,
) -> list[DefectMeasurements]:
    """
    Measure every connected external defect region.

    Small regions below min_area_pixels are ignored.
    """

    if min_area_pixels < 0:
        raise ValueError(
            "min_area_pixels cannot be negative."
        )

    contours = calculate_contours(
        mask
    )

    measurements: list[
        DefectMeasurements
    ] = []

    for contour in contours:
        area = cv2.contourArea(
            contour
        )

        if area < min_area_pixels:
            continue

        measurements.append(
            measure_defect(
                contour,
                calibration,
            )
        )

    measurements.sort(
        key=lambda item: item.area_pixels,
        reverse=True,
    )

    return measurements


def calculate_area_ratio(
    defect_area_pixels: float,
    road_area_pixels: float,
) -> float:
    """
    Calculate the percentage of road surface occupied by defects.

    Returns:
        Ratio in [0, 1].
    """

    if defect_area_pixels < 0:
        raise ValueError(
            "defect_area_pixels cannot be negative."
        )

    if road_area_pixels <= 0:
        raise ValueError(
            "road_area_pixels must be greater than zero."
        )

    ratio = (
        defect_area_pixels
        / road_area_pixels
    )

    return float(
        np.clip(
            ratio,
            0.0,
            1.0,
        )
    )


def calculate_severity(
    defect_area_ratio: float,
    length_m: float | None = None,
    width_m: float | None = None,
) -> SeverityResult:
    """
    Calculate deterministic defect severity.

    Base score is derived from defect area ratio.
    Physical dimensions can increase the score when calibration
    is available.

    Score:
        0-24   -> low
        25-49  -> moderate
        50-74  -> high
        75-100 -> critical
    """

    if not 0.0 <= defect_area_ratio <= 1.0:
        raise ValueError(
            "defect_area_ratio must be between 0 and 1."
        )

    if length_m is not None and length_m < 0:
        raise ValueError(
            "length_m cannot be negative."
        )

    if width_m is not None and width_m < 0:
        raise ValueError(
            "width_m cannot be negative."
        )

    # Area contribution.
    area_score = min(
        defect_area_ratio * 1000.0,
        70.0,
    )

    dimension_score = 0.0

    if length_m is not None:
        dimension_score += min(
            length_m * 5.0,
            15.0,
        )

    if width_m is not None:
        dimension_score += min(
            width_m * 10.0,
            15.0,
        )

    score = float(
        min(
            area_score + dimension_score,
            100.0,
        )
    )

    if score < 25:
        level = SeverityLevel.LOW
        explanation = (
            "Small defect with relatively limited "
            "road-surface impact."
        )

    elif score < 50:
        level = SeverityLevel.MODERATE
        explanation = (
            "Noticeable defect requiring inspection "
            "and planned maintenance."
        )

    elif score < 75:
        level = SeverityLevel.HIGH
        explanation = (
            "Significant defect with substantial "
            "road-surface impact."
        )

    else:
        level = SeverityLevel.CRITICAL
        explanation = (
            "Severe defect requiring high-priority "
            "maintenance or repair."
        )

    return SeverityResult(
        level=level,
        score=score,
        area_ratio=defect_area_ratio,
        explanation=explanation,
    )


def calculate_road_health(
    road_area_pixels: int | float,
    defects: Iterable[DefectMeasurements],
) -> RoadHealthResult:
    """
    Calculate an overall road-health score.

    The score decreases according to the fraction of road area
    occupied by detected defects.

    Score:
        90-100 -> Excellent
        75-89  -> Good
        50-74  -> Fair
        25-49  -> Poor
        0-24   -> Critical
    """

    if road_area_pixels <= 0:
        raise ValueError(
            "road_area_pixels must be greater than zero."
        )

    defects_list = list(
        defects
    )

    total_defect_area = sum(
        max(
            defect.area_pixels,
            0.0,
        )
        for defect in defects_list
    )

    defect_area_ratio = calculate_area_ratio(
        total_defect_area,
        float(road_area_pixels),
    )

    # A deterministic penalty based on affected surface.
    score = max(
        0.0,
        100.0
        - defect_area_ratio * 100.0,
    )

    if score >= 90:
        condition = "Excellent"
    elif score >= 75:
        condition = "Good"
    elif score >= 50:
        condition = "Fair"
    elif score >= 25:
        condition = "Poor"
    else:
        condition = "Critical"

    return RoadHealthResult(
        score=float(score),
        condition=condition,
        defect_area_ratio=defect_area_ratio,
        defect_count=len(defects_list),
    )


def estimate_repair_cost(
    defects: Iterable[DefectMeasurements],
    rate_per_m2: float,
    currency: str = "INR",
) -> RepairCostResult:
    """
    Estimate repair cost from physically calibrated defect area.

    All defects must have area_m2 available.
    """

    if rate_per_m2 < 0:
        raise ValueError(
            "rate_per_m2 cannot be negative."
        )

    if not currency.strip():
        raise ValueError(
            "currency cannot be empty."
        )

    defects_list = list(
        defects
    )

    missing_calibration = [
        index
        for index, defect in enumerate(
            defects_list
        )
        if defect.area_m2 is None
    ]

    if missing_calibration:
        raise ValueError(
            "Physical area is unavailable for defects at "
            f"indices {missing_calibration}. "
            "Provide Calibration before estimating repair cost."
        )

    total_area = sum(
        float(defect.area_m2)
        for defect in defects_list
        if defect.area_m2 is not None
    )

    estimated_cost = (
        total_area * rate_per_m2
    )

    return RepairCostResult(
        estimated_area_m2=total_area,
        rate_per_m2=float(rate_per_m2),
        estimated_cost=float(estimated_cost),
        currency=currency.upper(),
    )


def analyze_road(
    mask: np.ndarray,
    road_area_pixels: int | float,
    calibration: Calibration | None = None,
    repair_rate_per_m2: float | None = None,
    currency: str = "INR",
    min_area_pixels: float = 10.0,
) -> EngineeringAnalysisResult:
    """
    Run the complete engineering-analysis pipeline.

    Steps:
        1. Extract defect regions.
        2. Measure area, length and width.
        3. Calculate severity for every defect.
        4. Calculate overall road health.
        5. Estimate repair cost when calibration and a repair rate
           are available.
    """

    defects = measure_all_defects(
        mask=mask,
        calibration=calibration,
        min_area_pixels=min_area_pixels,
    )

    severity_results: list[
        SeverityResult
    ] = []

    for defect in defects:
        ratio = calculate_area_ratio(
            defect.area_pixels,
            float(road_area_pixels),
        )

        severity_results.append(
            calculate_severity(
                defect_area_ratio=ratio,
                length_m=defect.length_m,
                width_m=defect.width_m,
            )
        )

    road_health = calculate_road_health(
        road_area_pixels=road_area_pixels,
        defects=defects,
    )

    if repair_rate_per_m2 is not None:
        repair_cost = estimate_repair_cost(
            defects=defects,
            rate_per_m2=repair_rate_per_m2,
            currency=currency,
        )

    else:
        repair_cost = RepairCostResult(
            estimated_area_m2=0.0,
            rate_per_m2=0.0,
            estimated_cost=0.0,
            currency=currency.upper(),
        )

    return EngineeringAnalysisResult(
        defects=defects,
        severity=severity_results,
        road_health=road_health,
        repair_cost=repair_cost,
    )


__all__ = [
    "SeverityLevel",
    "Calibration",
    "DefectMeasurements",
    "SeverityResult",
    "RoadHealthResult",
    "RepairCostResult",
    "EngineeringAnalysisResult",
    "validate_mask",
    "calculate_area_pixels",
    "calculate_contours",
    "calculate_bounding_box",
    "calculate_length_width",
    "calculate_centroid",
    "measure_defect",
    "measure_all_defects",
    "calculate_area_ratio",
    "calculate_severity",
    "calculate_road_health",
    "estimate_repair_cost",
    "analyze_road",
]