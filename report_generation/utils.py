"""
RoadXAI Report Generation Module.

Generates structured engineering reports from model predictions,
explainability results, engineering measurements, and 3D analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json


@dataclass
class ReportSection:
    """A section of the generated report."""

    title: str
    content: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoadXAIReport:
    """Complete RoadXAI engineering report."""

    report_id: str
    generated_at: str
    title: str
    image_name: Optional[str]
    sections: List[ReportSection]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report into a JSON-serializable dictionary."""
        return asdict(self)


def _json_safe(value: Any) -> Any:
    """Convert common Python/NumPy objects into JSON-safe values."""
    if value is None:
        return None

    if hasattr(value, "value"):
        return _json_safe(value.value)

    if hasattr(value, "tolist"):
        return value.tolist()

    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))

    return value


def _get_value(
    source: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Safely read a value from a dictionary or object."""
    if source is None:
        return default

    if isinstance(source, dict):
        return source.get(key, default)

    return getattr(source, key, default)


def create_report_id(
    prefix: str = "ROADXAI",
) -> str:
    """Create a unique report identifier."""
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d%H%M%S%f")

    return f"{prefix}-{timestamp}"


def create_summary_section(
    engineering_result: Any = None,
    inference_result: Any = None,
) -> ReportSection:
    """
    Create the executive summary section.
    """
    summary: Dict[str, Any] = {}

    confidence = _get_value(
        inference_result,
        "confidence",
    )

    if confidence is not None:
        summary["model_confidence"] = float(
            confidence
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

    if severity is not None:
        summary["severity"] = _json_safe(
            severity
        )

    road_health = _get_value(
        engineering_result,
        "road_health",
    )

    if road_health is not None:
        summary["road_health"] = _json_safe(
            road_health
        )

    defect_area = _get_value(
        engineering_result,
        "total_area",
    )

    if defect_area is not None:
        summary["defect_area"] = float(
            defect_area
        )

    content_parts = [
        "RoadXAI automated road-defect assessment."
    ]

    if confidence is not None:
        content_parts.append(
            f"Model confidence: {float(confidence):.2%}."
        )

    if severity is not None:
        content_parts.append(
            f"Detected severity: "
            f"{_json_safe(severity)}."
        )

    if road_health is not None:
        content_parts.append(
            f"Road-health assessment: "
            f"{_json_safe(road_health)}."
        )

    return ReportSection(
        title="Executive Summary",
        content=" ".join(content_parts),
        data=summary,
    )


def create_detection_section(
    inference_result: Any,
) -> ReportSection:
    """
    Create the defect-detection section.
    """
    data: Dict[str, Any] = {}

    prediction_mask = _get_value(
        inference_result,
        "prediction_mask",
    )

    if prediction_mask is not None:
        try:
            import numpy as np

            mask = np.asarray(prediction_mask)

            data["image_height"] = int(
                mask.shape[0]
            )
            data["image_width"] = int(
                mask.shape[1]
            )
            data["defect_pixels"] = int(
                np.count_nonzero(mask)
            )
            data["total_pixels"] = int(
                mask.size
            )

            if mask.size:
                data["defect_percentage"] = (
                    float(
                        np.count_nonzero(mask)
                        / mask.size
                        * 100.0
                    )
                )
        except Exception:
            pass

    confidence = _get_value(
        inference_result,
        "confidence",
    )

    if confidence is not None:
        data["confidence"] = float(
            confidence
        )

    content = (
        "The segmentation model identified "
        f"{data.get('defect_percentage', 0.0):.2f}% "
        "of the analyzed image area as defective."
    )

    if confidence is not None:
        content += (
            f" Estimated prediction confidence was "
            f"{float(confidence):.2%}."
        )

    return ReportSection(
        title="Defect Detection",
        content=content,
        data=data,
    )


def create_measurement_section(
    engineering_result: Any,
) -> ReportSection:
    """
    Create the engineering-measurement section.
    """
    data: Dict[str, Any] = {}

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

    serialized_measurements = []

    for measurement in measurements:
        serialized_measurements.append(
            _json_safe(measurement)
        )

    data["defects"] = serialized_measurements
    data["defect_count"] = len(
        serialized_measurements
    )

    total_area = _get_value(
        engineering_result,
        "total_area",
    )

    if total_area is not None:
        data["total_area"] = _json_safe(
            total_area
        )

    content = (
        f"Engineering analysis identified "
        f"{len(serialized_measurements)} "
        "measured defect region(s)."
    )

    if total_area is not None:
        content += (
            f" Total measured defect area: "
            f"{total_area}."
        )

    return ReportSection(
        title="Engineering Measurements",
        content=content,
        data=data,
    )


def create_severity_section(
    engineering_result: Any,
) -> ReportSection:
    """
    Create the severity and road-condition section.
    """
    data: Dict[str, Any] = {}

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
        engineering_result,
        "condition",
    )

    if condition is None:
        condition = _get_value(
            engineering_result,
            "road_condition",
        )

    data["severity"] = _json_safe(
        severity
    )
    data["road_health"] = _json_safe(
        road_health
    )
    data["condition"] = _json_safe(
        condition
    )

    content_parts = []

    if severity is not None:
        content_parts.append(
            f"Severity classification: "
            f"{_json_safe(severity)}."
        )

    if condition is not None:
        content_parts.append(
            f"Road condition: "
            f"{_json_safe(condition)}."
        )

    if road_health is not None:
        content_parts.append(
            f"Road health: "
            f"{_json_safe(road_health)}."
        )

    if not content_parts:
        content_parts.append(
            "No severity or road-condition "
            "information was provided."
        )

    return ReportSection(
        title="Severity Assessment",
        content=" ".join(content_parts),
        data=data,
    )


def create_cost_section(
    engineering_result: Any,
) -> ReportSection:
    """
    Create the repair-cost estimation section.
    """
    data: Dict[str, Any] = {}

    cost = _get_value(
        engineering_result,
        "repair_cost",
    )

    if cost is None:
        cost = _get_value(
            engineering_result,
            "cost_estimate",
        )

    if cost is None:
        cost = _get_value(
            engineering_result,
            "repair_cost_result",
        )

    data["estimate"] = _json_safe(cost)

    if cost is None:
        content = (
            "No repair-cost estimate was "
            "available for this assessment."
        )
    else:
        content = (
            "Estimated repair cost information "
            "is included in the report data."
        )

    return ReportSection(
        title="Repair Cost Estimate",
        content=content,
        data=data,
    )


def create_xai_section(
    xai_result: Any,
) -> ReportSection:
    """
    Create the explainability section.
    """
    data: Dict[str, Any] = {}

    method = _get_value(
        xai_result,
        "method",
    )

    if method is None:
        method = _get_value(
            xai_result,
            "technique",
        )

    heatmap = _get_value(
        xai_result,
        "heatmap",
    )

    if method is not None:
        data["method"] = _json_safe(
            method
        )

    if heatmap is not None:
        try:
            data["heatmap_shape"] = list(
                heatmap.shape
            )
        except AttributeError:
            pass

    if method is not None:
        content = (
            f"Explainability was generated using "
            f"{_json_safe(method)}."
        )
    else:
        content = (
            "Explainability output was generated "
            "for the model prediction."
        )

    return ReportSection(
        title="Explainable AI",
        content=content,
        data=data,
    )


def create_3d_section(
    visualization_result: Any,
) -> ReportSection:
    """
    Create the 3D visualization section.
    """
    data: Dict[str, Any] = {}

    heightmap = _get_value(
        visualization_result,
        "heightmap",
    )

    mesh = _get_value(
        visualization_result,
        "mesh",
    )

    if heightmap is not None:
        try:
            data["heightmap_shape"] = list(
                heightmap.shape
            )
        except AttributeError:
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
            data["vertex_count"] = len(
                vertices
            )

        if faces is not None:
            data["face_count"] = len(
                faces
            )

    content = (
        "A 3D representation of the detected "
        "defect geometry was generated."
    )

    return ReportSection(
        title="3D Visualization",
        content=content,
        data=data,
    )


def create_recommendations_section(
    engineering_result: Any = None,
) -> ReportSection:
    """
    Create deterministic maintenance recommendations.
    """
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
            "Prioritize immediate inspection and "
            "repair planning."
        )
    elif "high" in severity_text:
        recommendation = (
            "Schedule high-priority inspection and "
            "maintenance."
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
            "Review the generated measurements and "
            "schedule an appropriate field inspection."
        )

    return ReportSection(
        title="Recommendations",
        content=recommendation,
        data={
            "recommendation": recommendation,
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
    """
    Generate a complete RoadXAI engineering report.
    """
    sections: List[ReportSection] = []

    sections.append(
        create_summary_section(
            engineering_result,
            inference_result,
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

    summary = sections[0].data if sections else {}

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


def save_json_report(
    report: RoadXAIReport,
    output_path: Union[str, Path],
) -> Path:
    """
    Save a RoadXAI report as JSON.
    """
    path = Path(output_path)

    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")

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


def render_report_text(
    report: RoadXAIReport,
) -> str:
    """
    Render a report as readable plain text.
    """
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
                section.content,
                "",
            ]
        )

    return "\n".join(lines)


def save_text_report(
    report: RoadXAIReport,
    output_path: Union[str, Path],
) -> Path:
    """
    Save a readable text version of the report.
    """
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


def generate_pdf_report(
    report: RoadXAIReport,
    output_path: Union[str, Path],
) -> Path:
    """
    Generate a PDF version of the report.

    Requires ReportLab.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
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

    styles = getSampleStyleSheet()

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=report.title,
        author="RoadXAI",
    )

    story = []

    story.append(
        Paragraph(
            report.title,
            styles["Title"],
        )
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            f"Report ID: {report.report_id}",
            styles["Normal"],
        )
    )

    story.append(
        Paragraph(
            f"Generated: {report.generated_at}",
            styles["Normal"],
        )
    )

    if report.image_name:
        story.append(
            Paragraph(
                f"Image: {report.image_name}",
                styles["Normal"],
            )
        )

    story.append(
        Spacer(1, 16)
    )

    for section in report.sections:
        story.append(
            Paragraph(
                section.title,
                styles["Heading2"],
            )
        )

        story.append(
            Paragraph(
                section.content,
                styles["BodyText"],
            )
        )

        if section.data:
            formatted_data = json.dumps(
                _json_safe(section.data),
                indent=2,
                ensure_ascii=False,
            )

            formatted_data = (
                formatted_data
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
            )

            story.append(
                Spacer(1, 6)
            )

            story.append(
                Paragraph(
                    formatted_data,
                    styles["Code"],
                )
            )

        story.append(
            Spacer(1, 14)
        )

    document.build(story)

    return path