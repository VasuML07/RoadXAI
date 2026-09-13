"""
RoadXAI Report Generation Module.

Creates structured RoadXAI inspection reports from:

    - model inference
    - engineering analysis
    - explainable AI
    - 3D visualization

Important design rule
---------------------
Structured data is stored inside ReportSection.data for JSON export.

Human-readable report formats (TXT/PDF) use ONLY ReportSection.content.
They must never dump ReportSection.data into the document.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json


# ============================================================================
# REPORT DATA MODELS
# ============================================================================


@dataclass
class ReportSection:
    """One human-readable section of a RoadXAI report."""

    title: str
    content: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoadXAIReport:
    """Complete structured RoadXAI report."""

    report_id: str
    generated_at: str
    title: str
    image_name: Optional[str]
    sections: List[ReportSection]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report into JSON-compatible data."""
        return _json_safe(asdict(self))


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def _json_safe(value: Any) -> Any:
    """
    Convert common Python, NumPy, Enum and dataclass values
    into JSON-compatible values.
    """

    if value is None:
        return None

    # Enum-like values
    if hasattr(value, "value"):
        try:
            return _json_safe(value.value)
        except Exception:
            pass

    # NumPy scalar / ndarray
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
        return [
            _json_safe(item)
            for item in value
        ]

    # Dataclass
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))

    # Primitive values
    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    # Final safe fallback
    return str(value)


def _get_value(
    source: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Read a value from either a dictionary or an object."""

    if source is None:
        return default

    if isinstance(source, dict):
        return source.get(
            key,
            default,
        )

    return getattr(
        source,
        key,
        default,
    )


def _format_number(
    value: Any,
    digits: int = 2,
) -> str:
    """Safely format a numeric value."""

    try:
        return f"{float(value):.{digits}f}"
    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def _format_percent(
    value: Any,
) -> str:
    """Format either a fraction or percentage."""

    try:
        number = float(value)

        if 0.0 <= number <= 1.0:
            number *= 100.0

        return f"{number:.2f}%"

    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


def _format_optional(
    value: Any,
    suffix: str = "",
) -> str:
    """Format optional numeric values."""

    if value is None:
        return "Not available"

    return (
        f"{_format_number(value)}"
        f"{suffix}"
    )


# ============================================================================
# REPORT ID
# ============================================================================


def create_report_id(
    prefix: str = "ROADXAI",
) -> str:
    """Create a unique report identifier."""

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d%H%M%S%f"
    )

    return f"{prefix}-{timestamp}"


# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================


def create_summary_section(
    engineering_result: Any = None,
    inference_result: Any = None,
) -> ReportSection:
    """Create the executive summary."""

    confidence = _get_value(
        inference_result,
        "confidence",
    )

    prediction_mask = _get_value(
        inference_result,
        "prediction_mask",
    )

    defect_count = None

    road_health = _get_value(
        engineering_result,
        "road_health",
    )

    severity = _get_value(
        engineering_result,
        "severity",
    )

    if severity is None:
        severity = _get_value(
            engineering_result,
            "severity_level",
        )

    if road_health is not None:
        defect_count = _get_value(
            road_health,
            "defect_count",
        )

    if defect_count is None:
        defects = _get_value(
            engineering_result,
            "defects",
            [],
        )

        if defects is not None:
            try:
                defect_count = len(defects)
            except TypeError:
                defect_count = None

    defect_percentage = None

    if (
        prediction_mask is not None
    ):
        try:
            import numpy as np

            mask = np.asarray(
                prediction_mask
            )

            if mask.size > 0:
                defect_percentage = (
                    float(
                        np.count_nonzero(mask)
                    )
                    / float(mask.size)
                    * 100.0
                )
        except Exception:
            pass

    health_score = _get_value(
        road_health,
        "score",
    )

    condition = _get_value(
        road_health,
        "condition",
    )

    if condition is None:
        condition = _get_value(
            engineering_result,
            "condition",
        )

    summary_data: Dict[str, Any] = {}

    if defect_count is not None:
        summary_data[
            "defect_count"
        ] = int(defect_count)

    if defect_percentage is not None:
        summary_data[
            "defect_percentage"
        ] = float(defect_percentage)

    if confidence is not None:
        summary_data[
            "model_confidence"
        ] = float(confidence)

    if condition is not None:
        summary_data[
            "road_health_condition"
        ] = str(condition)

    if health_score is not None:
        summary_data[
            "road_health_score"
        ] = float(health_score)

    parts: List[str] = []

    if defect_count is not None:
        parts.append(
            "The system identified "
            f"{int(defect_count)} separate "
            "detected defect region(s)."
        )
    else:
        parts.append(
            "The system completed an "
            "automated road-defect assessment."
        )

    if defect_percentage is not None:
        parts.append(
            "Approximately "
            f"{defect_percentage:.2f}% of the "
            "analyzed image pixels were classified "
            "as defect pixels."
        )

    if confidence is not None:
        parts.append(
            "The model confidence indicator is "
            f"{float(confidence):.2%}."
        )

    if condition is not None:
        if health_score is not None:
            parts.append(
                "The calculated road-health result "
                f"is {condition} "
                f"({float(health_score):.2f}/100)."
            )
        else:
            parts.append(
                f"The calculated road condition "
                f"is {condition}."
            )

    if severity is not None:
        severity_text = _json_safe(
            severity
        )

        if isinstance(
            severity_text,
            (list, dict),
        ):
            parts.append(
                "Individual defect severity "
                "classifications are provided in "
                "the engineering section."
            )
        else:
            parts.append(
                "The assessment includes the "
                f"following severity classification: "
                f"{severity_text}."
            )

    parts.append(
        "This is an automated image-analysis "
        "result and should be confirmed by field "
        "inspection before repair decisions."
    )

    return ReportSection(
        title="Executive Summary",
        content=" ".join(parts),
        data=summary_data,
    )


# ============================================================================
# DEFECT DETECTION
# ============================================================================


def create_detection_section(
    inference_result: Any,
) -> ReportSection:
    """Create the defect-detection section."""

    prediction_mask = _get_value(
        inference_result,
        "prediction_mask",
    )

    confidence = _get_value(
        inference_result,
        "confidence",
    )

    original_image = _get_value(
        inference_result,
        "original_image",
    )

    data: Dict[str, Any] = {}

    image_height = None
    image_width = None

    if original_image is not None:
        try:
            shape = original_image.shape

            if len(shape) >= 2:
                image_height = int(shape[0])
                image_width = int(shape[1])
        except Exception:
            pass

    if (
        image_height is None
        and prediction_mask is not None
    ):
        try:
            shape = prediction_mask.shape

            if len(shape) >= 2:
                image_height = int(shape[-2])
                image_width = int(shape[-1])
        except Exception:
            pass

    defect_pixels = None
    total_pixels = None

    if prediction_mask is not None:
        try:
            import numpy as np

            mask = np.asarray(
                prediction_mask
            )

            total_pixels = int(
                mask.size
            )

            defect_pixels = int(
                np.count_nonzero(mask)
            )

        except Exception:
            pass

    if image_height is not None:
        data[
            "image_height"
        ] = image_height

    if image_width is not None:
        data[
            "image_width"
        ] = image_width

    if total_pixels is not None:
        data[
            "total_pixels"
        ] = total_pixels

    if defect_pixels is not None:
        data[
            "defect_pixels"
        ] = defect_pixels

    defect_percentage = None

    if (
        defect_pixels is not None
        and total_pixels
    ):
        defect_percentage = (
            defect_pixels
            / total_pixels
            * 100.0
        )

        data[
            "defect_percentage"
        ] = defect_percentage

    if confidence is not None:
        data[
            "confidence"
        ] = float(confidence)

    parts = [
        "RoadXAI uses image segmentation "
        "to estimate which pixels belong "
        "to road defects."
    ]

    if defect_percentage is not None:
        parts.append(
            f"In this assessment, "
            f"{defect_percentage:.2f}% of the "
            "analyzed image pixels were "
            "classified as defect pixels."
        )

    if confidence is not None:
        parts.append(
            "The reported model confidence "
            f"indicator is {float(confidence):.2%}."
        )

    parts.append(
        "This is an automated image-analysis "
        "result and should be confirmed by "
        "field inspection before repair decisions."
    )

    return ReportSection(
        title="Defect Detection",
        content=" ".join(parts),
        data=data,
    )


# ============================================================================
# ENGINEERING MEASUREMENTS
# ============================================================================


def create_measurement_section(
    engineering_result: Any,
) -> ReportSection:
    """Create the engineering-measurement section."""

    measurements = _get_value(
        engineering_result,
        "measurements",
    )

    if measurements is None:
        measurements = _get_value(
            engineering_result,
            "defects",
            [],
        )

    if measurements is None:
        measurements = []

    if not isinstance(
        measurements,
        (list, tuple),
    ):
        measurements = [measurements]

    serialized = [
        _json_safe(item)
        for item in measurements
    ]

    calibration = _get_value(
        engineering_result,
        "calibration",
    )

    data: Dict[str, Any] = {
        "defect_count": len(serialized),
        "measurements": serialized,
        "physical_calibration_available": (
            calibration is not None
        ),
    }

    physical_available = (
        calibration is not None
    )

    if physical_available:
        content = (
            "Engineering analysis measured "
            f"{len(serialized)} detected defect "
            "region(s). Physical measurements "
            "are available because a calibration "
            "reference was supplied."
        )
    else:
        content = (
            "Engineering analysis measured "
            f"{len(serialized)} detected defect "
            "region(s). Area, length, and width "
            "are reported in pixels because no "
            "validated pixel-to-meter calibration "
            "was supplied. These pixel measurements "
            "must not be interpreted as meters or "
            "square meters."
        )

    return ReportSection(
        title="Engineering Measurements",
        content=content,
        data=data,
    )


# ============================================================================
# SEVERITY AND ROAD HEALTH
# ============================================================================


def create_severity_section(
    engineering_result: Any,
) -> ReportSection:
    """Create severity and road-health information."""

    severity = _get_value(
        engineering_result,
        "severity",
    )

    if severity is None:
        severity = _get_value(
            engineering_result,
            "severity_level",
        )

    road_health = _get_value(
        engineering_result,
        "road_health",
    )

    condition = _get_value(
        road_health,
        "condition",
    )

    if condition is None:
        condition = _get_value(
            engineering_result,
            "condition",
        )

    health_score = _get_value(
        road_health,
        "score",
    )

    data = {
        "severity": _json_safe(
            severity
        ),
        "road_health": _json_safe(
            road_health
        ),
        "condition": _json_safe(
            condition
        ),
    }

    if health_score is not None:
        data[
            "road_health_score"
        ] = float(health_score)

    parts: List[str] = []

    if condition is not None:
        if health_score is not None:
            parts.append(
                "Overall road condition: "
                f"{condition} "
                f"({float(health_score):.2f}/100)."
            )
        else:
            parts.append(
                f"Overall road condition: "
                f"{condition}."
            )

    if severity is not None:
        severity_value = _json_safe(
            severity
        )

        if isinstance(
            severity_value,
            list,
        ):
            counts = {}

            for item in severity_value:
                level = _get_value(
                    item,
                    "level",
                )

                if level is not None:
                    level = str(level)
                    counts[level] = (
                        counts.get(level, 0)
                        + 1
                    )

            if counts:
                summary = ", ".join(
                    f"{count} {level}"
                    for level, count
                    in counts.items()
                )

                parts.append(
                    "Detected severity distribution: "
                    f"{summary}."
                )

        elif isinstance(
            severity_value,
            dict,
        ):
            parts.append(
                "Severity information is available "
                "for the detected defects."
            )
        else:
            parts.append(
                "Severity classification: "
                f"{severity_value}."
            )

    if not parts:
        parts.append(
            "No severity or road-condition "
            "information was provided."
        )

    return ReportSection(
        title="Severity Assessment",
        content=" ".join(parts),
        data=data,
    )


# ============================================================================
# REPAIR COST
# ============================================================================


def create_cost_section(
    engineering_result: Any,
) -> ReportSection:
    """Create repair-cost information."""

    repair_cost = _get_value(
        engineering_result,
        "repair_cost",
    )

    if repair_cost is None:
        return ReportSection(
            title="Repair Cost Estimate",
            content=(
                "No repair-cost estimate was "
                "generated because the required "
                "calibration or repair-rate "
                "information was not supplied."
            ),
            data={
                "available": False,
            },
        )

    estimated_area = _get_value(
        repair_cost,
        "estimated_area_m2",
    )

    rate = _get_value(
        repair_cost,
        "rate_per_m2",
    )

    estimated_cost = _get_value(
        repair_cost,
        "estimated_cost",
    )

    currency = _get_value(
        repair_cost,
        "currency",
        "INR",
    )

    data = _json_safe(
        repair_cost
    )

    if not isinstance(
        data,
        dict,
    ):
        data = {
            "repair_cost": data,
        }

    data["available"] = True

    parts = [
        "A repair-cost estimate was generated."
    ]

    if estimated_area is not None:
        parts.append(
            "Estimated repair area: "
            f"{_format_number(estimated_area)} m²."
        )

    if rate is not None:
        parts.append(
            "Applied repair rate: "
            f"{currency} "
            f"{float(rate):,.2f} per m²."
        )

    if estimated_cost is not None:
        parts.append(
            "Estimated repair cost: "
            f"{currency} "
            f"{float(estimated_cost):,.2f}."
        )

    return ReportSection(
        title="Repair Cost Estimate",
        content=" ".join(parts),
        data=data,
    )


# ============================================================================
# XAI
# ============================================================================


def create_xai_section(
    xai_result: Any,
) -> ReportSection:
    """Create explainable-AI information."""

    if xai_result is None:
        return ReportSection(
            title="Explainable AI",
            content=(
                "No explainable-AI result was "
                "generated."
            ),
            data={
                "available": False,
            },
        )

    method = _get_value(
        xai_result,
        "method",
    )

    if method is None:
        method = _get_value(
            xai_result,
            "algorithm",
        )

    target_class = _get_value(
        xai_result,
        "target_class",
    )

    data: Dict[str, Any] = {
        "available": True,
    }

    if method is not None:
        data[
            "method"
        ] = str(method)

    if target_class is not None:
        data[
            "target_class"
        ] = _json_safe(target_class)

    parts = [
        "An explainable-AI attention map "
        "was generated to show image regions "
        "that contributed to the model prediction."
    ]

    if method is not None:
        parts.append(
            f"The selected method was {method}."
        )

    if target_class is not None:
        parts.append(
            f"The analyzed target class was "
            f"{target_class}."
        )

    return ReportSection(
        title="Explainable AI",
        content=" ".join(parts),
        data=data,
    )


# ============================================================================
# 3D VISUALIZATION
# ============================================================================


def create_3d_section(
    visualization_result: Any,
) -> ReportSection:
    """Create 3D-visualization information."""

    if visualization_result is None:
        return ReportSection(
            title="3D Visualization",
            content=(
                "No 3D visualization was generated."
            ),
            data={
                "available": False,
            },
        )

    data: Dict[str, Any] = {
        "available": True,
    }

    heightmap = _get_value(
        visualization_result,
        "heightmap",
    )

    if heightmap is None:
        heightmap_result = _get_value(
            visualization_result,
            "heightmap_result",
        )

        heightmap = _get_value(
            heightmap_result,
            "heightmap",
        )

    mesh = _get_value(
        visualization_result,
        "mesh",
    )

    if heightmap is not None:
        try:
            data[
                "heightmap_shape"
            ] = list(
                heightmap.shape
            )
        except Exception:
            pass

    if mesh is not None:
        vertices = _get_value(
            mesh,
            "vertices",
        )

        faces = _get_value(
            mesh,
            "faces",
        )

        if vertices is not None:
            try:
                data[
                    "vertex_count"
                ] = int(len(vertices))
            except Exception:
                pass

        if faces is not None:
            try:
                data[
                    "face_count"
                ] = int(len(faces))
            except Exception:
                pass

    data[
        "depth_is_physical"
    ] = False

    return ReportSection(
        title="3D Visualization",
        content=(
            "An interactive 3D representation "
            "of the detected defect geometry "
            "was generated. The displayed depth "
            "is a normalized visualization "
            "coordinate unless validated physical "
            "depth information is supplied."
        ),
        data=data,
    )


# ============================================================================
# RECOMMENDATIONS
# ============================================================================


def create_recommendations_section(
    engineering_result: Any = None,
) -> ReportSection:
    """Create deterministic maintenance recommendations."""

    severity = _get_value(
        engineering_result,
        "severity",
    )

    if severity is None:
        severity = _get_value(
            engineering_result,
            "severity_level",
        )

    severity_text = str(
        _json_safe(severity)
    ).lower()

    if "critical" in severity_text:
        recommendation = (
            "Prioritize immediate inspection "
            "and repair planning."
        )

    elif "high" in severity_text:
        recommendation = (
            "Schedule high-priority inspection "
            "and maintenance."
        )

    elif "moderate" in severity_text:
        recommendation = (
            "Schedule routine inspection and "
            "maintenance planning."
        )

    elif "low" in severity_text:
        recommendation = (
            "Continue monitoring and include the "
            "location in routine maintenance."
        )

    else:
        recommendation = (
            "Review the generated measurements "
            "and schedule an appropriate field "
            "inspection."
        )

    return ReportSection(
        title="Recommendations",
        content=recommendation,
        data={
            "recommendation": recommendation,
        },
    )


# ============================================================================
# COMPLETE REPORT
# ============================================================================


def generate_report(
    engineering_result: Any = None,
    inference_result: Any = None,
    xai_result: Any = None,
    visualization_result: Any = None,
    image_name: Optional[str] = None,
    title: str = "RoadXAI Road Inspection Report",
) -> RoadXAIReport:
    """Generate a complete RoadXAI report."""

    sections: List[ReportSection] = []

    sections.append(
        create_summary_section(
            engineering_result=engineering_result,
            inference_result=inference_result,
        )
    )

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

        sections.append(
            create_recommendations_section(
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

    summary = (
        sections[0].data
        if sections
        else {}
    )

    return RoadXAIReport(
        report_id=create_report_id(),
        generated_at=datetime.now(
            timezone.utc
        ).isoformat(),
        title=title,
        image_name=image_name,
        sections=sections,
        summary=_json_safe(summary),
    )


# ============================================================================
# JSON EXPORT
# ============================================================================


def save_json_report(
    report: RoadXAIReport,
    output_path: Union[str, Path],
) -> Path:
    """Save the complete structured report as JSON."""

    if not isinstance(
        report,
        RoadXAIReport,
    ):
        raise TypeError(
            "report must be a RoadXAIReport."
        )

    path = Path(
        output_path
    )

    if path.suffix.lower() != ".json":
        path = path.with_suffix(
            ".json"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report.to_dict(),
            file,
            indent=4,
            ensure_ascii=False,
        )

    return path


# ============================================================================
# HUMAN-READABLE TEXT
# ============================================================================


def render_report_text(
    report: RoadXAIReport,
) -> str:
    """
    Render ONLY human-readable section content.

    ReportSection.data is deliberately excluded.
    """

    if not isinstance(
        report,
        RoadXAIReport,
    ):
        raise TypeError(
            "report must be a RoadXAIReport."
        )

    lines = [
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

    lines.append("")

    for section in report.sections:
        lines.extend(
            [
                section.title,
                "-" * len(section.title),
                section.content.strip(),
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def save_text_report(
    report: RoadXAIReport,
    output_path: Union[str, Path],
) -> Path:
    """Save the human-readable text report."""

    path = Path(
        output_path
    )

    if path.suffix.lower() != ".txt":
        path = path.with_suffix(
            ".txt"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        render_report_text(report),
        encoding="utf-8",
    )

    return path


# ============================================================================
# PDF EXPORT
# ============================================================================


def generate_pdf_report(
    report: RoadXAIReport,
    output_path: Union[str, Path],
) -> Path:
    """
    Generate a professional human-readable PDF.

    CRITICAL:
        ReportSection.data is NEVER rendered.

    Only:
        report metadata
        section.title
        section.content

    are written to the PDF.
    """

    if not isinstance(
        report,
        RoadXAIReport,
    ):
        raise TypeError(
            "report must be a RoadXAIReport."
        )

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import (
            TA_CENTER,
            TA_LEFT,
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )

    except ImportError as exc:
        raise ImportError(
            "ReportLab is required for PDF generation. "
            "Install it with: pip install reportlab"
        ) from exc

    path = Path(
        output_path
    )

    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(
            ".pdf"
        )

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
        subject="Automated Road Defect Inspection Report",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "RoadXAITitle",
        parent=styles["Title"],
        fontSize=21,
        leading=25,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    metadata_style = ParagraphStyle(
        "RoadXAIMetadata",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor(
            "#555555"
        ),
        spaceAfter=3,
    )

    section_style = ParagraphStyle(
        "RoadXAISection",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        alignment=TA_LEFT,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "RoadXAIBody",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=16,
        alignment=TA_LEFT,
        spaceAfter=8,
    )

    footer_style = ParagraphStyle(
        "RoadXAIFooter",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor(
            "#666666"
        ),
    )

    story = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    story.append(
        Paragraph(
            report.title,
            title_style,
        )
    )

    story.append(
        Paragraph(
            f"Report ID: {report.report_id}",
            metadata_style,
        )
    )

    story.append(
        Paragraph(
            f"Generated: {report.generated_at}",
            metadata_style,
        )
    )

    if report.image_name:
        story.append(
            Paragraph(
                f"Image: {report.image_name}",
                metadata_style,
            )
        )

    story.append(
        Spacer(
            1,
            10,
        )
    )

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    for section in report.sections:

        section_title = (
            str(section.title)
            .replace("&", "&amp;")
        )

        section_content = (
            str(section.content)
            .strip()
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

        block = [
            Paragraph(
                section_title,
                section_style,
            ),
            Paragraph(
                section_content,
                body_style,
            ),
        ]

        story.append(
            KeepTogether(block)
        )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    story.append(
        Spacer(
            1,
            12,
        )
    )

    story.append(
        Paragraph(
            "RoadXAI automated inspection report. "
            "Results should be validated through appropriate "
            "field inspection before maintenance decisions.",
            footer_style,
        )
    )

    def add_page_number(
        canvas,
        doc,
    ):
        canvas.saveState()

        canvas.setFont(
            "Helvetica",
            8,
        )

        canvas.setFillColor(
            colors.HexColor(
                "#666666"
            )
        )

        canvas.drawCentredString(
            A4[0] / 2,
            8 * mm,
            f"RoadXAI | Page {doc.page}",
        )

        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    return path


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "ReportSection",
    "RoadXAIReport",
    "create_report_id",
    "create_summary_section",
    "create_detection_section",
    "create_measurement_section",
    "create_severity_section",
    "create_cost_section",
    "create_xai_section",
    "create_3d_section",
    "create_recommendations_section",
    "generate_report",
    "save_json_report",
    "render_report_text",
    "save_text_report",
    "generate_pdf_report",
]