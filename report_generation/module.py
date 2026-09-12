"""
RoadXAI Report Generation Module.

Creates clear, structured, layman-friendly road-defect assessment
reports from inference, engineering, XAI, and 3D results.

The report deliberately separates:
    - model prediction
    - engineering measurements
    - severity
    - road-health interpretation
    - explainability
    - 3D visualization
    - recommendations

Important:
Physical dimensions and repair-cost estimates are only presented as
physical quantities when valid calibration/rate information exists.
The report never invents meters, square meters, or monetary values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import math


@dataclass
class ReportSection:
    """One human-readable report section."""

    title: str
    content: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoadXAIReport:
    """Complete RoadXAI assessment report."""

    report_id: str
    generated_at: str
    title: str
    image_name: Optional[str]
    sections: List[ReportSection]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return _json_safe(asdict(self))


def _json_safe(value: Any) -> Any:
    """Convert common Python/NumPy/dataclass values to JSON-safe data."""
    if value is None:
        return None

    if hasattr(value, "value") and value.__class__.__module__ != "builtins":
        try:
            return _json_safe(value.value)
        except Exception:
            pass

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, (str, int, bool)):
        return value

    return str(value)


def _get_value(
    source: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Read a field from either a dictionary or an object."""
    if source is None:
        return default

    if isinstance(source, dict):
        return source.get(key, default)

    return getattr(source, key, default)


def _number(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a value to a finite float."""
    if value is None:
        return default

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    return number if math.isfinite(number) else default


def _percentage(value: Any) -> Optional[float]:
    """Convert a ratio such as 0.123 to percentage 12.3."""
    number = _number(value)
    if number is None:
        return None

    if 0.0 <= number <= 1.0:
        return number * 100.0

    return number


def _severity_level(severity: Any) -> str:
    """Return a clean severity label."""
    level = _get_value(severity, "level")

    if level is None:
        level = severity

    value = _get_value(level, "value", level)

    text = str(value).strip().lower()

    if "critical" in text:
        return "critical"
    if "high" in text:
        return "high"
    if "moderate" in text:
        return "moderate"
    if "low" in text:
        return "low"

    return "unknown"


def _severity_explanation(level: str) -> str:
    """Return a layman-friendly explanation."""
    explanations = {
        "critical": (
            "This is a severe detected defect. It should receive "
            "high-priority field inspection and maintenance planning."
        ),
        "high": (
            "This is a significant detected defect. It should be "
            "inspected and considered for high-priority maintenance."
        ),
        "moderate": (
            "This is a noticeable defect. It should be inspected "
            "and included in planned maintenance."
        ),
        "low": (
            "This is a relatively small detected defect. It should "
            "be monitored and considered during routine maintenance."
        ),
        "unknown": (
            "The available result does not provide a recognized "
            "severity category."
        ),
    }

    return explanations[level]


def _health_explanation(condition: str, score: Optional[float]) -> str:
    """Translate road-health output into plain language."""
    normalized = str(condition or "").strip().lower()

    if normalized == "excellent":
        return (
            "The detected defect area occupies a relatively small "
            "portion of the analyzed image according to the current "
            "health-score calculation."
        )

    if normalized == "good":
        return (
            "The detected defects affect a limited portion of the "
            "analyzed image according to the current health-score calculation."
        )

    if normalized == "fair":
        return (
            "The detected defects affect a noticeable portion of the "
            "analyzed image and should be reviewed during maintenance planning."
        )

    if normalized == "poor":
        return (
            "The detected defects affect a substantial portion of the "
            "analyzed image and should receive closer inspection."
        )

    if normalized == "critical":
        return (
            "The detected defects occupy a large portion of the analyzed "
            "image and warrant high-priority field inspection."
        )

    return (
        "Road health is calculated from the detected defect area and the "
        "configured deterministic health-score rules."
    )


def create_report_id(prefix: str = "ROADXAI") -> str:
    """Create a unique report identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}-{timestamp}"


def create_summary_section(
    engineering_result: Any = None,
    inference_result: Any = None,
) -> ReportSection:
    """Create a concise, human-readable executive summary."""

    confidence = _number(
        _get_value(inference_result, "confidence")
    )

    defects = _get_value(
        engineering_result,
        "defects",
        [],
    ) or []

    road_health = _get_value(
        engineering_result,
        "road_health",
    )

    health_score = _number(
        _get_value(road_health, "score")
    )

    health_condition = _get_value(
        road_health,
        "condition",
        "Not available",
    )

    prediction_mask = _get_value(
        inference_result,
        "prediction_mask",
    )

    defect_percentage = None

    if prediction_mask is not None:
        try:
            import numpy as np

            mask = np.asarray(prediction_mask)
            if mask.size:
                defect_percentage = (
                    float(np.count_nonzero(mask))
                    / float(mask.size)
                    * 100.0
                )
        except Exception:
            pass

    if defects:
        defect_sentence = (
            f"The system identified {len(defects)} separate "
            f"detected defect region(s)."
        )
    else:
        defect_sentence = (
            "The system did not identify any defect regions "
            "above the configured minimum area."
        )

    if defect_percentage is not None:
        area_sentence = (
            f"Approximately {defect_percentage:.2f}% of the analyzed "
            "image pixels were classified as defect pixels."
        )
    else:
        area_sentence = ""

    if confidence is not None:
        confidence_sentence = (
            f"The model confidence indicator is {confidence:.2%}."
        )
    else:
        confidence_sentence = (
            "A model confidence value was not available."
        )

    health_sentence = (
        f"The calculated road-health result is "
        f"{str(health_condition).title()}"
    )

    if health_score is not None:
        health_sentence += f" ({health_score:.2f}/100)."

    health_sentence += " " + _health_explanation(
        str(health_condition),
        health_score,
    )

    content = " ".join(
        part
        for part in [
            defect_sentence,
            area_sentence,
            confidence_sentence,
            health_sentence,
        ]
        if part
    )

    data = {
        "defect_count": len(defects),
        "defect_percentage": defect_percentage,
        "model_confidence": confidence,
        "road_health_condition": health_condition,
        "road_health_score": health_score,
    }

    return ReportSection(
        title="Executive Summary",
        content=content,
        data=data,
    )


def create_detection_section(
    inference_result: Any,
) -> ReportSection:
    """Create a clear description of model detection."""

    prediction_mask = _get_value(
        inference_result,
        "prediction_mask",
    )

    confidence = _number(
        _get_value(inference_result, "confidence")
    )

    data: Dict[str, Any] = {}

    if prediction_mask is not None:
        try:
            import numpy as np

            mask = np.asarray(prediction_mask)

            if mask.ndim == 3:
                mask = np.squeeze(mask)

            data["image_height"] = int(mask.shape[0])
            data["image_width"] = int(mask.shape[1])
            data["total_pixels"] = int(mask.size)
            data["defect_pixels"] = int(np.count_nonzero(mask))
            data["defect_percentage"] = (
                float(np.count_nonzero(mask))
                / float(mask.size)
                * 100.0
                if mask.size
                else 0.0
            )
        except Exception:
            pass

    if confidence is not None:
        data["confidence"] = confidence

    defect_percentage = data.get(
        "defect_percentage",
        0.0,
    )

    content = (
        "RoadXAI uses image segmentation to estimate which pixels "
        "belong to road defects. In this assessment, "
        f"{defect_percentage:.2f}% of the analyzed image pixels "
        "were classified as defect pixels."
    )

    if confidence is not None:
        content += (
            f" The reported model confidence indicator is "
            f"{confidence:.2%}."
        )

    content += (
        " This is an automated image-analysis result and should be "
        "confirmed by field inspection before repair decisions."
    )

    return ReportSection(
        title="Defect Detection",
        content=content,
        data=data,
    )


def create_measurement_section(
    engineering_result: Any,
) -> ReportSection:
    """Create readable measurements for every detected region."""

    measurements = _get_value(
        engineering_result,
        "defects",
        [],
    ) or []

    if not isinstance(measurements, (list, tuple)):
        measurements = [measurements]

    rows: List[Dict[str, Any]] = []

    for index, measurement in enumerate(
        measurements,
        start=1,
    ):
        row: Dict[str, Any] = {
            "defect_number": index,
            "area_pixels": _number(
                _get_value(measurement, "area_pixels")
            ),
            "length_pixels": _number(
                _get_value(measurement, "length_pixels")
            ),
            "width_pixels": _number(
                _get_value(measurement, "width_pixels")
            ),
            "area_m2": _number(
                _get_value(measurement, "area_m2")
            ),
            "length_m": _number(
                _get_value(measurement, "length_m")
            ),
            "width_m": _number(
                _get_value(measurement, "width_m")
            ),
            "centroid_x": _number(
                _get_value(measurement, "centroid_x")
            ),
            "centroid_y": _number(
                _get_value(measurement, "centroid_y")
            ),
        }

        rows.append(row)

    data = {
        "defect_count": len(rows),
        "measurements": rows,
        "physical_calibration_available": any(
            row["area_m2"] is not None
            or row["length_m"] is not None
            or row["width_m"] is not None
            for row in rows
        ),
    }

    if not rows:
        content = (
            "No defect regions were available for engineering measurement."
        )
    else:
        physical_available = data[
            "physical_calibration_available"
        ]

        content = (
            f"Engineering analysis measured {len(rows)} detected "
            "defect region(s). Area, length, and width are reported "
            "in pixels unless a validated pixel-to-meter calibration "
            "was supplied."
        )

        if physical_available:
            content += (
                " Physical measurements are available for the "
                "calibrated results."
            )
        else:
            content += (
                " No physical calibration was supplied, so the "
                "pixel measurements must not be interpreted as meters "
                "or square meters."
            )

    return ReportSection(
        title="Engineering Measurements",
        content=content,
        data=data,
    )


def create_severity_section(
    engineering_result: Any,
) -> ReportSection:
    """Create a layman-friendly severity assessment."""

    severity_results = _get_value(
        engineering_result,
        "severity",
        [],
    ) or []

    if not isinstance(
        severity_results,
        (list, tuple),
    ):
        severity_results = [severity_results]

    road_health = _get_value(
        engineering_result,
        "road_health",
    )

    health_condition = _get_value(
        road_health,
        "condition",
        "Not available",
    )

    health_score = _number(
        _get_value(road_health, "score")
    )

    severity_rows: List[Dict[str, Any]] = []

    for index, severity in enumerate(
        severity_results,
        start=1,
    ):
        level = _severity_level(severity)
        score = _number(
            _get_value(severity, "score")
        )
        area_ratio = _number(
            _get_value(severity, "area_ratio")
        )

        severity_rows.append(
            {
                "defect_number": index,
                "level": level,
                "score": score,
                "defect_area_percentage": (
                    area_ratio * 100.0
                    if area_ratio is not None
                    else None
                ),
                "meaning": _severity_explanation(level),
            }
        )

    data = {
        "severity": severity_rows,
        "road_health": {
            "condition": health_condition,
            "score": health_score,
        },
    }

    if severity_rows:
        highest = max(
            severity_rows,
            key=lambda item: (
                {
                    "critical": 4,
                    "high": 3,
                    "moderate": 2,
                    "low": 1,
                    "unknown": 0,
                }.get(item["level"], 0),
                item["score"] or 0.0,
            ),
        )

        content = (
            f"The highest detected severity category is "
            f"{highest['level'].upper()}. "
            f"{_severity_explanation(highest['level'])}"
        )

        content += (
            f" Across the full image, the calculated road-health "
            f"result is {str(health_condition).title()}"
        )

        if health_score is not None:
            content += f" ({health_score:.2f}/100)"

        content += "."

        content += (
            " Severity is calculated deterministically from the "
            "detected defect area and, when available, calibrated "
            "physical dimensions."
        )
    else:
        content = (
            "No severity classifications were generated because "
            "no qualifying defect regions were available."
        )

    return ReportSection(
        title="Severity Assessment",
        content=content,
        data=data,
    )


def create_cost_section(
    engineering_result: Any,
) -> ReportSection:
    """Create a safe repair-cost section without inventing prices."""

    cost = _get_value(
        engineering_result,
        "repair_cost",
    )

    if cost is None:
        cost = _get_value(
            engineering_result,
            "repair_cost_result",
        )

    estimate_area = _number(
        _get_value(cost, "estimated_area_m2")
    )
    rate = _number(
        _get_value(cost, "rate_per_m2")
    )
    estimated_cost = _number(
        _get_value(cost, "estimated_cost")
    )
    currency = str(
        _get_value(cost, "currency", "INR")
    )

    physical_area_available = any(
        value is not None
        for value in (
            estimate_area,
            rate,
            estimated_cost,
        )
    ) and (
        estimate_area is not None
        and estimate_area > 0
    )

    if (
        cost is None
        or not physical_area_available
        or rate is None
        or rate <= 0
    ):
        content = (
            "A repair-cost amount is not provided for this assessment. "
            "RoadXAI requires a validated pixel-to-meter calibration "
            "and a user-supplied repair rate per square meter before "
            "a monetary estimate can be calculated. No price is "
            "invented by the system."
        )

        data = {
            "available": False,
            "reason": (
                "Calibration and/or repair rate was not supplied."
            ),
        }

    else:
        content = (
            f"Using the supplied calibration and repair rate, the "
            f"estimated affected area is {estimate_area:.4f} m² and "
            f"the estimated repair amount is "
            f"{currency.upper()} {estimated_cost:,.2f}. "
            "This is an indicative calculation, not a contractor quote."
        )

        data = {
            "available": True,
            "estimated_area_m2": estimate_area,
            "rate_per_m2": rate,
            "estimated_cost": estimated_cost,
            "currency": currency.upper(),
        }

    return ReportSection(
        title="Repair Cost Estimate",
        content=content,
        data=data,
    )


def create_xai_section(
    xai_result: Any,
) -> ReportSection:
    """Create a clear explanation of the XAI result."""

    if xai_result is None:
        return ReportSection(
            title="Explainable AI",
            content=(
                "No explainability result was generated for this assessment."
            ),
            data={"available": False},
        )

    method = _get_value(
        xai_result,
        "method",
        _get_value(xai_result, "technique", "Grad-CAM"),
    )

    target_layer = _get_value(
        xai_result,
        "target_layer",
    )

    heatmap = _get_value(
        xai_result,
        "heatmap",
    )

    shape = None

    try:
        shape = list(heatmap.shape)
    except Exception:
        pass

    target_layer_name = (
        target_layer.__class__.__name__
        if target_layer is not None
        else "Not specified"
    )

    content = (
        f"{method} was used to show which image regions most "
        "influenced the model's prediction. Warmer/brighter areas "
        "indicate stronger model attention, while cooler/darker "
        "areas indicate weaker attention."
    )

    content += (
        " This visualization explains the model's focus; it is not "
        "a separate measurement of defect size or physical damage."
    )

    if target_layer is not None:
        content += (
            f" The explanation was generated from the configured "
            f"{target_layer_name} target layer."
        )

    data = {
        "available": True,
        "method": _json_safe(method),
        "target_layer": target_layer_name,
        "heatmap_shape": shape,
    }

    return ReportSection(
        title="Explainable AI",
        content=content,
        data=data,
    )


def create_3d_section(
    visualization_result: Any,
) -> ReportSection:
    """Create a clear explanation of the 3D visualization."""

    if visualization_result is None:
        return ReportSection(
            title="3D Visualization",
            content=(
                "No 3D visualization was generated for this assessment."
            ),
            data={"available": False},
        )

    heightmap = _get_value(
        visualization_result,
        "heightmap",
    )

    mesh = _get_value(
        visualization_result,
        "mesh",
    )

    shape = None

    try:
        shape = list(heightmap.shape)
    except Exception:
        pass

    vertices = _get_value(mesh, "vertices")
    faces = _get_value(mesh, "faces")

    vertex_count = (
        len(vertices)
        if vertices is not None
        else None
    )

    face_count = (
        len(faces)
        if faces is not None
        else None
    )

    content = (
        "A 3D surface visualization was generated from the detected "
        "segmentation mask. Higher surface values represent stronger "
        "detected defect intensity in the visualization."
    )

    content += (
        " The 3D surface is a visual representation of the model "
        "output; without depth or camera calibration it must not be "
        "interpreted as a physically measured pothole depth."
    )

    data = {
        "available": True,
        "heightmap_shape": shape,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "physical_depth_measurement": False,
    }

    return ReportSection(
        title="3D Visualization",
        content=content,
        data=data,
    )


def create_recommendations_section(
    engineering_result: Any = None,
) -> ReportSection:
    """Create deterministic maintenance guidance."""

    severity_results = _get_value(
        engineering_result,
        "severity",
        [],
    ) or []

    if not isinstance(
        severity_results,
        (list, tuple),
    ):
        severity_results = [severity_results]

    levels = [
        _severity_level(item)
        for item in severity_results
    ]

    if "critical" in levels:
        priority = "IMMEDIATE / HIGH PRIORITY"
        action = (
            "Arrange prompt field inspection and prioritize the "
            "critical region(s) for maintenance planning."
        )
    elif "high" in levels:
        priority = "HIGH PRIORITY"
        action = (
            "Arrange field inspection and prioritize the high-severity "
            "region(s) for maintenance planning."
        )
    elif "moderate" in levels:
        priority = "PLANNED MAINTENANCE"
        action = (
            "Inspect the moderate-severity region(s) and include them "
            "in planned maintenance."
        )
    elif "low" in levels:
        priority = "ROUTINE MONITORING"
        action = (
            "Monitor the detected low-severity region(s) and consider "
            "them during routine maintenance."
        )
    else:
        priority = "FIELD REVIEW"
        action = (
            "Review the automated result and confirm conditions with "
            "a field inspection before making maintenance decisions."
        )

    content = (
        f"Recommended priority: {priority}. {action} "
        "RoadXAI should be treated as a screening and decision-support "
        "system; final engineering and repair decisions require "
        "appropriate field verification."
    )

    return ReportSection(
        title="Recommendations",
        content=content,
        data={
            "priority": priority,
            "recommended_action": action,
            "field_verification_required": True,
        },
    )


def generate_report(
    engineering_result: Any = None,
    inference_result: Any = None,
    xai_result: Any = None,
    visualization_result: Any = None,
    image_name: Optional[str] = None,
    title: str = "RoadXAI Road Defect Assessment",
) -> RoadXAIReport:
    """Generate the complete RoadXAI assessment."""

    sections: List[ReportSection] = [
        create_summary_section(
            engineering_result,
            inference_result,
        )
    ]

    if inference_result is not None:
        sections.append(
            create_detection_section(
                inference_result
            )
        )

    if engineering_result is not None:
        sections.append(
            create_measurement_section(
                engineering_result
            )
        )
        sections.append(
            create_severity_section(
                engineering_result
            )
        )
        sections.append(
            create_cost_section(
                engineering_result
            )
        )

    if xai_result is not None:
        sections.append(
            create_xai_section(
                xai_result
            )
        )

    if visualization_result is not None:
        sections.append(
            create_3d_section(
                visualization_result
            )
        )

    if engineering_result is not None:
        sections.append(
            create_recommendations_section(
                engineering_result
            )
        )

    return RoadXAIReport(
        report_id=create_report_id(),
        generated_at=datetime.now(
            timezone.utc
        ).isoformat(),
        title=title,
        image_name=image_name,
        sections=sections,
        summary=_json_safe(
            sections[0].data
            if sections
            else {}
        ),
    )


def save_json_report(
    report: RoadXAIReport,
    output_path: Union[str, Path],
) -> Path:
    """Save a structured JSON report."""
    path = Path(output_path)

    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            report.to_dict(),
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


def render_report_text(
    report: RoadXAIReport,
) -> str:
    """Render a professional plain-text report."""

    lines: List[str] = [
        report.title,
        "=" * len(report.title),
        "",
        f"Report ID: {report.report_id}",
        f"Generated: {report.generated_at}",
    ]

    if report.image_name:
        lines.append(
            f"Image: {report.image_name}"
        )

    lines.extend(
        [
            "",
            "IMPORTANT",
            "---------",
            (
                "This report is an automated image-analysis assessment. "
                "Field verification is required before engineering or "
                "repair decisions."
            ),
            "",
        ]
    )

    for number, section in enumerate(
        report.sections,
        start=1,
    ):
        lines.extend(
            [
                f"{number}. {section.title}",
                "-" * (
                    len(section.title) + 3
                ),
                section.content,
                "",
            ]
        )

        if section.title == "Engineering Measurements":
            measurements = section.data.get(
                "measurements",
                [],
            )

            if measurements:
                lines.append(
                    "Defect measurements:"
                )

                for item in measurements:
                    area = item.get("area_pixels")
                    length = item.get("length_pixels")
                    width = item.get("width_pixels")

                    lines.append(
                        f"  Defect {item['defect_number']}: "
                        f"area={_format_number(area)} px, "
                        f"length={_format_number(length)} px, "
                        f"width={_format_number(width)} px"
                    )

                lines.append("")

        if section.title == "Severity Assessment":
            severity_items = section.data.get(
                "severity",
                [],
            )

            for item in severity_items:
                score = item.get("score")

                score_text = (
                    f"{score:.2f}"
                    if score is not None
                    else "N/A"
                )

                lines.append(
                    f"  Defect {item['defect_number']}: "
                    f"{item['level'].upper()} "
                    f"(score {score_text}) — "
                    f"{item['meaning']}"
                )

            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _format_number(value: Any) -> str:
    """Format numeric values compactly."""
    number = _number(value)

    if number is None:
        return "N/A"

    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"

    return f"{number:,.2f}"


def save_text_report(
    report: RoadXAIReport,
    output_path: Union[str, Path],
) -> Path:
    """Save a readable text report."""
    path = Path(output_path)

    if path.suffix.lower() != ".txt":
        path = path.with_suffix(".txt")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        render_report_text(report),
        encoding="utf-8",
    )

    return path


def _pdf_paragraph(text: str) -> str:
    """Escape text for ReportLab Paragraph."""
    return escape(
        str(text)
    ).replace(
        "\n",
        "<br/>",
    )


def generate_pdf_report(
    report: RoadXAIReport,
    output_path: Union[str, Path],
) -> Path:
    """
    Generate a clean, human-readable PDF.

    The PDF intentionally does not dump raw JSON structures.
    """

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            HRFlowable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise ImportError(
            "ReportLab is required for PDF generation."
        ) from exc

    path = Path(output_path)

    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=report.title,
        author="RoadXAI",
    )

    base_styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "RoadXAITitle",
        parent=base_styles["Title"],
        fontSize=22,
        leading=27,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    meta_style = ParagraphStyle(
        "RoadXAIMeta",
        parent=base_styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#555555"),
    )

    heading_style = ParagraphStyle(
        "RoadXAIHeading",
        parent=base_styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=7,
    )

    body_style = ParagraphStyle(
        "RoadXAIBody",
        parent=base_styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=8,
    )

    small_style = ParagraphStyle(
        "RoadXAISmall",
        parent=base_styles["BodyText"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#555555"),
    )

    story = []

    story.append(
        Paragraph(
            _pdf_paragraph(report.title),
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Automated Road Defect Assessment",
            ParagraphStyle(
                "Subtitle",
                parent=base_styles["Normal"],
                alignment=TA_CENTER,
                fontSize=10,
                textColor=colors.HexColor("#555555"),
                spaceAfter=12,
            ),
        )
    )

    metadata = [
        [
            Paragraph("<b>Report ID</b>", small_style),
            Paragraph(
                _pdf_paragraph(report.report_id),
                small_style,
            ),
        ],
        [
            Paragraph("<b>Generated</b>", small_style),
            Paragraph(
                _pdf_paragraph(report.generated_at),
                small_style,
            ),
        ],
        [
            Paragraph("<b>Image</b>", small_style),
            Paragraph(
                _pdf_paragraph(
                    report.image_name or "Not specified"
                ),
                small_style,
            ),
        ],
    ]

    meta_table = Table(
        metadata,
        colWidths=[30 * mm, 145 * mm],
    )

    meta_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(meta_table)
    story.append(Spacer(1, 8))

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.8,
            color=colors.HexColor("#999999"),
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "<b>Important:</b> This is an automated image-analysis "
            "assessment. Field verification is required before "
            "engineering or repair decisions.",
            body_style,
        )
    )

    for index, section in enumerate(
        report.sections,
        start=1,
    ):
        story.append(
            Paragraph(
                _pdf_paragraph(
                    f"{index}. {section.title}"
                ),
                heading_style,
            )
        )

        story.append(
            Paragraph(
                _pdf_paragraph(section.content),
                body_style,
            )
        )

        if section.title == "Engineering Measurements":
            measurements = section.data.get(
                "measurements",
                [],
            )

            if measurements:
                table_data = [
                    [
                        "Defect",
                        "Area (px)",
                        "Length (px)",
                        "Width (px)",
                    ]
                ]

                for item in measurements:
                    table_data.append(
                        [
                            str(item["defect_number"]),
                            _format_number(
                                item.get("area_pixels")
                            ),
                            _format_number(
                                item.get("length_pixels")
                            ),
                            _format_number(
                                item.get("width_pixels")
                            ),
                        ]
                    )

                table = Table(
                    table_data,
                    colWidths=[
                        25 * mm,
                        42 * mm,
                        42 * mm,
                        42 * mm,
                    ],
                    repeatRows=1,
                )

                table.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor("#E8E8E8"),
                            ),
                            (
                                "TEXTCOLOR",
                                (0, 0),
                                (-1, 0),
                                colors.black,
                            ),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.35,
                                colors.HexColor("#AAAAAA"),
                            ),
                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, 0),
                                "Helvetica-Bold",
                            ),
                            (
                                "FONTSIZE",
                                (0, 0),
                                (-1, -1),
                                8.5,
                            ),
                            (
                                "ALIGN",
                                (0, 0),
                                (-1, -1),
                                "CENTER",
                            ),
                            (
                                "VALIGN",
                                (0, 0),
                                (-1, -1),
                                "MIDDLE",
                            ),
                            (
                                "TOPPADDING",
                                (0, 0),
                                (-1, -1),
                                5,
                            ),
                            (
                                "BOTTOMPADDING",
                                (0, 0),
                                (-1, -1),
                                5,
                            ),
                        ]
                    )
                )

                story.append(table)
                story.append(Spacer(1, 8))

        elif section.title == "Severity Assessment":
            severity_items = section.data.get(
                "severity",
                [],
            )

            if severity_items:
                severity_table = [
                    [
                        "Defect",
                        "Severity",
                        "Score",
                        "What it means",
                    ]
                ]

                for item in severity_items:
                    score = item.get("score")
                    severity_table.append(
                        [
                            str(item["defect_number"]),
                            item["level"].upper(),
                            (
                                f"{score:.2f}"
                                if score is not None
                                else "N/A"
                            ),
                            Paragraph(
                                _pdf_paragraph(
                                    item["meaning"]
                                ),
                                small_style,
                            ),
                        ]
                    )

                table = Table(
                    severity_table,
                    colWidths=[
                        20 * mm,
                        30 * mm,
                        25 * mm,
                        95 * mm,
                    ],
                    repeatRows=1,
                )

                table.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor("#E8E8E8"),
                            ),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.35,
                                colors.HexColor("#AAAAAA"),
                            ),
                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, 0),
                                "Helvetica-Bold",
                            ),
                            (
                                "FONTSIZE",
                                (0, 0),
                                (-1, -1),
                                8.5,
                            ),
                            (
                                "ALIGN",
                                (0, 0),
                                (2, -1),
                                "CENTER",
                            ),
                            (
                                "VALIGN",
                                (0, 0),
                                (-1, -1),
                                "TOP",
                            ),
                            (
                                "TOPPADDING",
                                (0, 0),
                                (-1, -1),
                                5,
                            ),
                            (
                                "BOTTOMPADDING",
                                (0, 0),
                                (-1, -1),
                                5,
                            ),
                        ]
                    )
                )

                story.append(table)
                story.append(Spacer(1, 8))

        elif section.title == "Repair Cost Estimate":
            if section.data.get("available"):
                cost_text = (
                    f"Estimated affected area: "
                    f"{section.data['estimated_area_m2']:.4f} m²<br/>"
                    f"Repair rate: "
                    f"{section.data['currency']} "
                    f"{section.data['rate_per_m2']:,.2f} per m²<br/>"
                    f"Estimated amount: "
                    f"{section.data['currency']} "
                    f"{section.data['estimated_cost']:,.2f}"
                )
            else:
                cost_text = (
                    "<b>No monetary estimate available.</b><br/>"
                    "A validated pixel-to-meter calibration and a "
                    "repair rate per square meter are required."
                )

            story.append(
                Paragraph(
                    cost_text,
                    body_style,
                )
            )

        story.append(Spacer(1, 5))

    document.build(story)

    return path


__all__ = [
    "ReportSection",
    "RoadXAIReport",
    "create_report_id",
    "create_summary_section",
    "create_detection_section",
    "create_measurement_section",
    "create_severity_section",
    "create_xai_section",
    "create_3d_section",
    "create_cost_section",
    "create_recommendations_section",
    "generate_report",
    "render_report_text",
    "save_json_report",
    "save_text_report",
    "generate_pdf_report",
]
