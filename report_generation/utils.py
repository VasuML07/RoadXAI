"""
RoadXAI Report Generation Utilities
====================================

Utilities for:

    - JSON report export
    - Human-readable TXT report export
    - Professional PDF report generation
    - Report preview formatting

Important:
The JSON file remains fully structured for machine/API use.

The TXT and PDF reports are intentionally human-readable and DO NOT
print internal Python dictionaries or raw JSON structures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from .module import (
    RoadXAIReport,
    ReportSection,
)


# ============================================================
# GENERIC HELPERS
# ============================================================


def _get_value(
    source: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Safely read an attribute from an object or dictionary."""

    if source is None:
        return default

    if isinstance(
        source,
        Mapping,
    ):
        return source.get(
            key,
            default,
        )

    return getattr(
        source,
        key,
        default,
    )


def _json_safe(
    value: Any,
) -> Any:
    """
    Convert common Python/NumPy/dataclass values into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if is_dataclass(value):
        return _json_safe(
            asdict(value)
        )

    if hasattr(
        value,
        "value",
    ):
        return _json_safe(
            value.value
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    return value


def _format_number(
    value: Any,
    decimals: int = 2,
    default: str = "N/A",
) -> str:
    """Format a numeric value for human-readable reports."""

    if value is None:
        return default

    try:
        number = float(value)

        if not np.isfinite(number):
            return default

        return f"{number:,.{decimals}f}"

    except (
        TypeError,
        ValueError,
    ):
        return default


def _format_percent(
    value: Any,
    decimals: int = 2,
) -> str:
    """Format a decimal probability as a percentage."""

    if value is None:
        return "N/A"

    try:
        number = float(value)

        if not np.isfinite(number):
            return "N/A"

        if abs(number) <= 1.0:
            number *= 100.0

        return f"{number:.{decimals}f}%"

    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


def _readable_enum(
    value: Any,
) -> str:
    """Convert Enum-like values to readable text."""

    if value is None:
        return "Unknown"

    if hasattr(
        value,
        "value",
    ):
        value = value.value

    return str(
        value
    ).replace(
        "_",
        " ",
    ).strip()


def _escape_pdf_text(
    value: Any,
) -> str:
    """
    Escape text before inserting it into a ReportLab Paragraph.
    """

    import html

    return html.escape(
        str(value)
    )


# ============================================================
# JSON EXPORT
# ============================================================


def save_json_report(
    report: RoadXAIReport,
    output_path: Union[
        str,
        Path,
    ],
) -> Path:
    """
    Save the complete structured report as JSON.

    JSON is intentionally machine-readable. This is the ONLY report
    format where the complete structured internal data is exported.
    """

    if not isinstance(
        report,
        RoadXAIReport,
    ):
        raise TypeError(
            "report must be a RoadXAIReport instance."
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

    payload = _json_safe(
        report.to_dict()
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return path


# ============================================================
# HUMAN-READABLE TEXT
# ============================================================


def _text_section(
    section: ReportSection,
) -> list[str]:
    """
    Convert a report section into readable text.

    Raw section.data is deliberately NOT printed.
    """

    title = str(
        section.title
    ).strip()

    content = str(
        section.content
        or ""
    ).strip()

    lines = [
        title,
        "-" * len(title),
    ]

    if content:
        lines.append(
            content
        )

    data = (
        section.data
        if isinstance(
            section.data,
            Mapping,
        )
        else {}
    )

    title_lower = title.lower()

    # --------------------------------------------------------
    # Executive Summary
    # --------------------------------------------------------

    if title_lower == "executive summary":
        confidence = _get_value(
            data,
            "model_confidence",
        )

        road_health = _get_value(
            data,
            "road_health",
        )

        if confidence is not None:
            lines.append(
                f"Model confidence: "
                f"{_format_percent(confidence)}"
            )

        if road_health is not None:
            condition = _readable_enum(
                _get_value(
                    road_health,
                    "condition",
                )
            )

            score = _get_value(
                road_health,
                "score",
            )

            if condition != "Unknown":
                lines.append(
                    f"Road health: "
                    f"{condition.title()}"
                )

            if score is not None:
                lines.append(
                    f"Road health score: "
                    f"{_format_number(score)} / 100"
                )

    # --------------------------------------------------------
    # Defect Detection
    # --------------------------------------------------------

    elif title_lower == "defect detection":

        image_height = _get_value(
            data,
            "image_height",
        )

        image_width = _get_value(
            data,
            "image_width",
        )

        defect_pixels = _get_value(
            data,
            "defect_pixels",
        )

        total_pixels = _get_value(
            data,
            "total_pixels",
        )

        defect_percentage = _get_value(
            data,
            "defect_percentage",
        )

        confidence = _get_value(
            data,
            "confidence",
        )

        if (
            image_height is not None
            and image_width is not None
        ):
            lines.append(
                f"Analyzed image: "
                f"{int(image_width)} × "
                f"{int(image_height)} pixels"
            )

        if defect_pixels is not None:
            lines.append(
                f"Detected defect pixels: "
                f"{int(defect_pixels):,}"
            )

        if total_pixels is not None:
            lines.append(
                f"Total analyzed pixels: "
                f"{int(total_pixels):,}"
            )

        if defect_percentage is not None:
            lines.append(
                f"Defect coverage: "
                f"{_format_number(defect_percentage)}%"
            )

        if confidence is not None:
            lines.append(
                f"Model confidence: "
                f"{_format_percent(confidence)}"
            )

    # --------------------------------------------------------
    # Engineering Measurements
    # --------------------------------------------------------

    elif title_lower == "engineering measurements":

        measurements = _get_value(
            data,
            "measurements",
            None,
        )

        if measurements is None:
            measurements = _get_value(
                data,
                "defects",
                [],
            )

        if measurements:
            lines.append(
                ""
            )

            lines.append(
                "Defect measurements:"
            )

            for index, item in enumerate(
                measurements,
                start=1,
            ):
                number = _get_value(
                    item,
                    "defect_number",
                    index,
                )

                area = _get_value(
                    item,
                    "area_pixels",
                )

                length = _get_value(
                    item,
                    "length_pixels",
                )

                width = _get_value(
                    item,
                    "width_pixels",
                )

                centroid_x = _get_value(
                    item,
                    "centroid_x",
                )

                centroid_y = _get_value(
                    item,
                    "centroid_y",
                )

                lines.append(
                    (
                        f"Defect #{number}: "
                        f"Area {_format_number(area)} px² | "
                        f"Length {_format_number(length)} px | "
                        f"Width {_format_number(width)} px | "
                        f"Location "
                        f"({_format_number(centroid_x, 1)}, "
                        f"{_format_number(centroid_y, 1)}) px"
                    )
                )

        calibration = _get_value(
            data,
            "physical_calibration_available",
        )

        if calibration is False:
            lines.append(
                ""
            )

            lines.append(
                "Physical calibration: Not available"
            )

            lines.append(
                "Physical dimensions are therefore "
                "not reported."
            )

    # --------------------------------------------------------
    # Severity Assessment
    # --------------------------------------------------------

    elif title_lower == "severity assessment":

        severity_items = _get_value(
            data,
            "severity",
            [],
        )

        if severity_items:
            lines.append(
                ""
            )

            lines.append(
                "Defect severity:"
            )

            for index, item in enumerate(
                severity_items,
                start=1,
            ):
                number = _get_value(
                    item,
                    "defect_number",
                    index,
                )

                level = _readable_enum(
                    _get_value(
                        item,
                        "level",
                        "Unknown",
                    )
                )

                score = _get_value(
                    item,
                    "score",
                )

                lines.append(
                    (
                        f"Defect #{number}: "
                        f"{level.title()} "
                        f"({_format_number(score)} / 100)"
                    )
                )

        road_health = _get_value(
            data,
            "road_health",
        )

        if road_health is not None:
            condition = _readable_enum(
                _get_value(
                    road_health,
                    "condition",
                )
            )

            score = _get_value(
                road_health,
                "score",
            )

            lines.append(
                ""
            )

            lines.append(
                f"Overall road condition: "
                f"{condition.title()}"
            )

            if score is not None:
                lines.append(
                    f"Overall road-health score: "
                    f"{_format_number(score)} / 100"
                )

    # --------------------------------------------------------
    # Repair Cost
    # --------------------------------------------------------

    elif title_lower == "repair cost estimate":

        available = _get_value(
            data,
            "available",
        )

        if available is False:
            lines.append(
                ""
            )

            lines.append(
                "Repair-cost estimate: Not available"
            )

            reason = _get_value(
                data,
                "reason",
            )

            if reason:
                lines.append(
                    f"Reason: {reason}"
                )

    # --------------------------------------------------------
    # Explainable AI
    # --------------------------------------------------------

    elif title_lower in {
        "explainable ai",
        "xai",
    }:

        method = _get_value(
            data,
            "method",
        )

        target_layer = _get_value(
            data,
            "target_layer",
        )

        if method:
            lines.append(
                f"Method: {method}"
            )

        if target_layer:
            lines.append(
                f"Target layer: {target_layer}"
            )

    # --------------------------------------------------------
    # 3D Visualization
    # --------------------------------------------------------

    elif title_lower == "3d visualization":

        available = _get_value(
            data,
            "available",
        )

        if available is not None:
            lines.append(
                f"3D visualization available: "
                f"{'Yes' if available else 'No'}"
            )

        shape = _get_value(
            data,
            "heightmap_shape",
        )

        if shape:
            lines.append(
                f"Heightmap: "
                f"{shape[0]} × {shape[1]}"
            )

        vertices = _get_value(
            data,
            "vertex_count",
        )

        faces = _get_value(
            data,
            "face_count",
        )

        if vertices is not None:
            lines.append(
                f"Mesh vertices: "
                f"{int(vertices):,}"
            )

        if faces is not None:
            lines.append(
                f"Mesh faces: "
                f"{int(faces):,}"
            )

        physical_depth = _get_value(
            data,
            "physical_depth_measurement",
        )

        if physical_depth is False:
            lines.append(
                "Physical depth measurement: "
                "Not available"
            )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    elif title_lower == "recommendations":

        recommendation = _get_value(
            data,
            "recommendation",
        )

        if recommendation:
            lines.append(
                ""
            )

            lines.append(
                f"Recommended action: "
                f"{recommendation}"
            )

    return lines


def render_report_text(
    report: RoadXAIReport,
) -> str:
    """
    Render the report as readable plain text.

    IMPORTANT:
    Internal dictionaries are never dumped into the report.
    """

    if not isinstance(
        report,
        RoadXAIReport,
    ):
        raise TypeError(
            "report must be a RoadXAIReport instance."
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
            _text_section(
                section
            )
        )

        lines.append("")

    return "\n".join(
        lines
    ).strip()


# ============================================================
# TXT EXPORT
# ============================================================


def save_text_report(
    report: RoadXAIReport,
    output_path: Union[
        str,
        Path,
    ],
) -> Path:
    """
    Save a human-readable text report.

    Raw JSON is intentionally excluded.
    """

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
        render_report_text(
            report
        ),
        encoding="utf-8",
    )

    return path


# ============================================================
# PDF STYLES
# ============================================================


def _build_pdf_styles():
    """
    Create professional ReportLab styles.
    """

    from reportlab.lib.enums import (
        TA_CENTER,
        TA_LEFT,
    )
    from reportlab.lib.styles import (
        ParagraphStyle,
        getSampleStyleSheet,
    )
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    base = getSampleStyleSheet()

    styles = {}

    styles["Title"] = ParagraphStyle(
        "RoadXAITitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    )

    styles["Subtitle"] = ParagraphStyle(
        "RoadXAISubtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor(
            "#475569"
        ),
        spaceAfter=7 * mm,
    )

    styles["Section"] = ParagraphStyle(
        "RoadXAISection",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor(
            "#0f172a"
        ),
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )

    styles["Body"] = ParagraphStyle(
        "RoadXAIBody",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor(
            "#1e293b"
        ),
        spaceAfter=2.5 * mm,
    )

    styles["Small"] = ParagraphStyle(
        "RoadXAISmall",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor(
            "#475569"
        ),
    )

    styles["Metric"] = ParagraphStyle(
        "RoadXAIMetric",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor(
            "#0f172a"
        ),
        spaceAfter=1.5 * mm,
    )

    styles["TableHeader"] = ParagraphStyle(
        "RoadXAIHeader",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=colors.white,
        alignment=TA_LEFT,
    )

    styles["TableBody"] = ParagraphStyle(
        "RoadXAITableBody",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor(
            "#1e293b"
        ),
    )

    return styles


# ============================================================
# PDF TABLE HELPERS
# ============================================================


def _make_table(
    data,
    col_widths,
    styles,
    repeat_rows: int = 1,
):
    """
    Create a consistent PDF table.
    """

    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=repeat_rows,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#1e3a5f"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor(
                        "#cbd5e1"
                    ),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor(
                            "#f8fafc"
                        ),
                    ],
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ]
        )
    )

    return table


# ============================================================
# PDF SECTION BUILDERS
# ============================================================


def _add_pdf_section(
    story,
    section: ReportSection,
    styles,
) -> None:
    """
    Add one report section to the PDF.

    The section's human-readable content is always shown.

    Structured data is selectively converted into:
        - metrics
        - tables
        - readable labels

    Raw dictionaries are NEVER printed.
    """

    from reportlab.lib.units import mm
    from reportlab.platypus import (
        KeepTogether,
        Paragraph,
        Spacer,
    )

    title = str(
        section.title
    ).strip()

    title_lower = title.lower()

    data = (
        section.data
        if isinstance(
            section.data,
            Mapping,
        )
        else {}
    )

    section_story = []

    section_story.append(
        Paragraph(
            _escape_pdf_text(
                title
            ),
            styles["Section"],
        )
    )

    content = str(
        section.content
        or ""
    ).strip()

    if content:
        section_story.append(
            Paragraph(
                _escape_pdf_text(
                    content
                ),
                styles["Body"],
            )
        )

    # --------------------------------------------------------
    # Executive Summary
    # --------------------------------------------------------

    if title_lower == "executive summary":

        rows = []

        confidence = _get_value(
            data,
            "model_confidence",
        )

        road_health = _get_value(
            data,
            "road_health",
        )

        if confidence is not None:
            rows.append(
                [
                    Paragraph(
                        "Model confidence",
                        styles["TableBody"],
                    ),
                    Paragraph(
                        _format_percent(
                            confidence
                        ),
                        styles["TableBody"],
                    ),
                ]
            )

        if road_health is not None:

            condition = _readable_enum(
                _get_value(
                    road_health,
                    "condition",
                )
            )

            score = _get_value(
                road_health,
                "score",
            )

            if condition != "Unknown":
                rows.append(
                    [
                        Paragraph(
                            "Road condition",
                            styles["TableBody"],
                        ),
                        Paragraph(
                            _escape_pdf_text(
                                condition.title()
                            ),
                            styles["TableBody"],
                        ),
                    ]
                )

            if score is not None:
                rows.append(
                    [
                        Paragraph(
                            "Road-health score",
                            styles["TableBody"],
                        ),
                        Paragraph(
                            (
                                f"{_format_number(score)} "
                                "/ 100"
                            ),
                            styles["TableBody"],
                        ),
                    ]
                )

        if rows:
            section_story.append(
                Spacer(
                    1,
                    2 * mm,
                )
            )

            table_rows = [
                [
                    Paragraph(
                        "Metric",
                        styles["TableHeader"],
                    ),
                    Paragraph(
                        "Result",
                        styles["TableHeader"],
                    ),
                ]
            ] + rows

            section_story.append(
                _make_table(
                    table_rows,
                    [
                        70 * mm,
                        65 * mm,
                    ],
                    styles,
                )
            )

    # --------------------------------------------------------
    # Defect Detection
    # --------------------------------------------------------

    elif title_lower == "defect detection":

        metrics = []

        values = [
            (
                "Analyzed image",
                (
                    f"{int(_get_value(data, 'image_width', 0))} × "
                    f"{int(_get_value(data, 'image_height', 0))} pixels"
                    if (
                        _get_value(data, "image_width")
                        is not None
                        and _get_value(data, "image_height")
                        is not None
                    )
                    else "N/A"
                ),
            ),
            (
                "Defect pixels",
                (
                    f"{int(_get_value(data, 'defect_pixels')):,}"
                    if _get_value(
                        data,
                        "defect_pixels",
                    )
                    is not None
                    else "N/A"
                ),
            ),
            (
                "Defect coverage",
                (
                    f"{_format_number(_get_value(data, 'defect_percentage'))}%"
                    if _get_value(
                        data,
                        "defect_percentage",
                    )
                    is not None
                    else "N/A"
                ),
            ),
            (
                "Model confidence",
                _format_percent(
                    _get_value(
                        data,
                        "confidence",
                    )
                ),
            ),
        ]

        for label, value in metrics:
            section_story.append(
                Paragraph(
                    (
                        f"<b>{_escape_pdf_text(label)}:</b> "
                        f"{_escape_pdf_text(value)}"
                    ),
                    styles["Metric"],
                )
            )

    # --------------------------------------------------------
    # Engineering Measurements
    # --------------------------------------------------------

    elif title_lower == "engineering measurements":

        measurements = _get_value(
            data,
            "measurements",
            None,
        )

        if measurements is None:
            measurements = _get_value(
                data,
                "defects",
                [],
            )

        if measurements:

            table_rows = [
                [
                    Paragraph(
                        "Defect",
                        styles["TableHeader"],
                    ),
                    Paragraph(
                        "Area (px²)",
                        styles["TableHeader"],
                    ),
                    Paragraph(
                        "Length (px)",
                        styles["TableHeader"],
                    ),
                    Paragraph(
                        "Width (px)",
                        styles["TableHeader"],
                    ),
                    Paragraph(
                        "Centroid (X,Y)",
                        styles["TableHeader"],
                    ),
                ]
            ]

            for index, item in enumerate(
                measurements,
                start=1,
            ):
                number = _get_value(
                    item,
                    "defect_number",
                    index,
                )

                area = _format_number(
                    _get_value(
                        item,
                        "area_pixels",
                    ),
                    1,
                )

                length = _format_number(
                    _get_value(
                        item,
                        "length_pixels",
                    ),
                    1,
                )

                width = _format_number(
                    _get_value(
                        item,
                        "width_pixels",
                    ),
                    1,
                )

                x = _format_number(
                    _get_value(
                        item,
                        "centroid_x",
                    ),
                    1,
                )

                y = _format_number(
                    _get_value(
                        item,
                        "centroid_y",
                    ),
                    1,
                )

                table_rows.append(
                    [
                        Paragraph(
                            f"#{number}",
                            styles["TableBody"],
                        ),
                        Paragraph(
                            area,
                            styles["TableBody"],
                        ),
                        Paragraph(
                            length,
                            styles["TableBody"],
                        ),
                        Paragraph(
                            width,
                            styles["TableBody"],
                        ),
                        Paragraph(
                            f"({x}, {y})",
                            styles["TableBody"],
                        ),
                    ]
                )

            section_story.append(
                Spacer(
                    1,
                    2 * mm,
                )
            )

            section_story.append(
                _make_table(
                    table_rows,
                    [
                        18 * mm,
                        30 * mm,
                        28 * mm,
                        26 * mm,
                        40 * mm,
                    ],
                    styles,
                )
            )

            calibration = _get_value(
                data,
                "physical_calibration_available",
            )

            if calibration is False:
                section_story.append(
                    Spacer(
                        1,
                        3 * mm,
                    )
                )

                section_story.append(
                    Paragraph(
                        (
                            "<b>Physical calibration:</b> "
                            "Not available. Pixel measurements "
                            "must not be interpreted as metres "
                            "or square metres."
                        ),
                        styles["Small"],
                    )
                )

    # --------------------------------------------------------
    # Severity Assessment
    # --------------------------------------------------------

    elif title_lower == "severity assessment":

        severity_items = _get_value(
            data,
            "severity",
            [],
        )

        if severity_items:

            table_rows = [
                [
                    Paragraph(
                        "Defect",
                        styles["TableHeader"],
                    ),
                    Paragraph(
                        "Severity",
                        styles["TableHeader"],
                    ),
                    Paragraph(
                        "Score",
                        styles["TableHeader"],
                    ),
                    Paragraph(
                        "Affected area",
                        styles["TableHeader"],
                    ),
                ]
            ]

            for index, item in enumerate(
                severity_items,
                start=1,
            ):
                number = _get_value(
                    item,
                    "defect_number",
                    index,
                )

                level = _readable_enum(
                    _get_value(
                        item,
                        "level",
                        "Unknown",
                    )
                )

                score = _format_number(
                    _get_value(
                        item,
                        "score",
                    ),
                    1,
                )

                area = _get_value(
                    item,
                    "defect_area_percentage",
                )

                if area is None:
                    area_text = "N/A"
                else:
                    area_text = (
                        f"{_format_number(area, 2)}%"
                    )

                table_rows.append(
                    [
                        Paragraph(
                            f"#{number}",
                            styles["TableBody"],
                        ),
                        Paragraph(
                            _escape_pdf_text(
                                level.title()
                            ),
                            styles["TableBody"],
                        ),
                        Paragraph(
                            f"{score}/100",
                            styles["TableBody"],
                        ),
                        Paragraph(
                            area_text,
                            styles["TableBody"],
                        ),
                    ]
                )

            section_story.append(
                Spacer(
                    1,
                    2 * mm,
                )
            )

            section_story.append(
                _make_table(
                    table_rows,
                    [
                        25 * mm,
                        45 * mm,
                        35 * mm,
                        45 * mm,
                    ],
                    styles,
                )
            )

        road_health = _get_value(
            data,
            "road_health",
        )

        if road_health is not None:

            condition = _readable_enum(
                _get_value(
                    road_health,
                    "condition",
                )
            )

            score = _get_value(
                road_health,
                "score",
            )

            section_story.append(
                Spacer(
                    1,
                    3 * mm,
                )
            )

            section_story.append(
                Paragraph(
                    (
                        f"<b>Overall road condition:</b> "
                        f"{_escape_pdf_text(condition.title())}"
                        "<br/>"
                        f"<b>Road-health score:</b> "
                        f"{_format_number(score)} / 100"
                    ),
                    styles["Body"],
                )
            )

    # --------------------------------------------------------
    # Repair Cost
    # --------------------------------------------------------

    elif title_lower == "repair cost estimate":

        available = _get_value(
            data,
            "available",
        )

        if available is False:

            reason = _get_value(
                data,
                "reason",
                "Calibration and repair rate were not supplied.",
            )

            section_story.append(
                Spacer(
                    1,
                    2 * mm,
                )
            )

            section_story.append(
                Paragraph(
                    (
                        "<b>Estimate unavailable.</b><br/>"
                        f"{_escape_pdf_text(reason)}"
                    ),
                    styles["Body"],
                )
            )

    # --------------------------------------------------------
    # Explainable AI
    # --------------------------------------------------------

    elif title_lower in {
        "explainable ai",
        "xai",
    }:

        method = _get_value(
            data,
            "method",
        )

        target_layer = _get_value(
            data,
            "target_layer",
        )

        if method:
            section_story.append(
                Paragraph(
                    (
                        f"<b>Method:</b> "
                        f"{_escape_pdf_text(method)}"
                    ),
                    styles["Metric"],
                )
            )

        if target_layer:
            section_story.append(
                Paragraph(
                    (
                        f"<b>Target layer:</b> "
                        f"{_escape_pdf_text(target_layer)}"
                    ),
                    styles["Small"],
                )
            )

    # --------------------------------------------------------
    # 3D Visualization
    # --------------------------------------------------------

    elif title_lower == "3d visualization":

        available = _get_value(
            data,
            "available",
        )

        shape = _get_value(
            data,
            "heightmap_shape",
        )

        vertices = _get_value(
            data,
            "vertex_count",
        )

        faces = _get_value(
            data,
            "face_count",
        )

        physical_depth = _get_value(
            data,
            "physical_depth_measurement",
        )

        rows = []

        if available is not None:
            rows.append(
                [
                    Paragraph(
                        "Available",
                        styles["TableBody"],
                    ),
                    Paragraph(
                        "Yes"
                        if available
                        else "No",
                        styles["TableBody"],
                    ),
                ]
            )

        if shape:
            rows.append(
                [
                    Paragraph(
                        "Heightmap",
                        styles["TableBody"],
                    ),
                    Paragraph(
                        f"{shape[0]} × {shape[1]}",
                        styles["TableBody"],
                    ),
                ]
            )

        if vertices is not None:
            rows.append(
                [
                    Paragraph(
                        "Mesh vertices",
                        styles["TableBody"],
                    ),
                    Paragraph(
                        f"{int(vertices):,}",
                        styles["TableBody"],
                    ),
                ]
            )

        if faces is not None:
            rows.append(
                [
                    Paragraph(
                        "Mesh faces",
                        styles["TableBody"],
                    ),
                    Paragraph(
                        f"{int(faces):,}",
                        styles["TableBody"],
                    ),
                ]
            )

        if physical_depth is False:
            rows.append(
                [
                    Paragraph(
                        "Physical depth",
                        styles["TableBody"],
                    ),
                    Paragraph(
                        "Not measured",
                        styles["TableBody"],
                    ),
                ]
            )

        if rows:
            table_rows = [
                [
                    Paragraph(
                        "Property",
                        styles["TableHeader"],
                    ),
                    Paragraph(
                        "Result",
                        styles["TableHeader"],
                    ),
                ]
            ] + rows

            section_story.append(
                Spacer(
                    1,
                    2 * mm,
                )
            )

            section_story.append(
                _make_table(
                    table_rows,
                    [
                        70 * mm,
                        65 * mm,
                    ],
                    styles,
                )
            )

        section_story.append(
            Spacer(
                1,
                3 * mm,
            )
        )

        section_story.append(
            Paragraph(
                (
                    "<b>Interpretation:</b> The 3D surface is "
                    "visualization geometry derived from the "
                    "segmentation mask. Without calibrated depth "
                    "data, it must not be interpreted as a physical "
                    "pothole-depth measurement."
                ),
                styles["Small"],
            )
        )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    elif title_lower == "recommendations":

        recommendation = _get_value(
            data,
            "recommendation",
        )

        if recommendation:
            section_story.append(
                Spacer(
                    1,
                    2 * mm,
                )
            )

            section_story.append(
                Paragraph(
                    (
                        f"<b>Recommended action:</b> "
                        f"{_escape_pdf_text(recommendation)}"
                    ),
                    styles["Body"],
                )
            )

    story.append(
        KeepTogether(
            section_story
        )
    )


# ============================================================
# PDF EXPORT
# ============================================================


def generate_pdf_report(
    report: RoadXAIReport,
    output_path: Union[
        str,
        Path,
    ],
) -> Path:
    """
    Generate a professional human-readable PDF report.

    IMPORTANT:
        Internal report dictionaries are NOT rendered.

    The PDF contains:
        - report metadata
        - readable summaries
        - engineering tables
        - severity tables
        - 3D summary
        - XAI summary
        - recommendations
        - scientific limitations
    """

    if not isinstance(
        report,
        RoadXAIReport,
    ):
        raise TypeError(
            "report must be a RoadXAIReport instance."
        )

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError as exc:
        raise ImportError(
            "ReportLab is required for PDF generation."
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

    styles = _build_pdf_styles()

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=report.title,
        author="RoadXAI",
        subject=(
            "AI-assisted road inspection report"
        ),
    )

    story = []

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    story.append(
        Paragraph(
            _escape_pdf_text(
                report.title
            ),
            styles["Title"],
        )
    )

    metadata_lines = [
        (
            f"<b>Report ID:</b> "
            f"{_escape_pdf_text(report.report_id)}"
        ),
        (
            f"<b>Generated:</b> "
            f"{_escape_pdf_text(report.generated_at)}"
        ),
    ]

    if report.image_name:
        metadata_lines.append(
            (
                f"<b>Image:</b> "
                f"{_escape_pdf_text(report.image_name)}"
            )
        )

    story.append(
        Paragraph(
            "<br/>".join(
                metadata_lines
            ),
            styles["Subtitle"],
        )
    )

    # --------------------------------------------------------
    # Important note
    # --------------------------------------------------------

    story.append(
        Paragraph(
            (
                "<b>Important:</b> RoadXAI provides "
                "AI-assisted image-based inspection support. "
                "Pixel measurements are not physical measurements "
                "unless validated calibration is supplied. "
                "The 3D vertical dimension is visualization geometry "
                "unless external depth data is available."
            ),
            styles["Small"],
        )
    )

    story.append(
        Spacer(
            1,
            3 * mm,
        )
    )

    # --------------------------------------------------------
    # Sections
    # --------------------------------------------------------

    for section in report.sections:
        _add_pdf_section(
            story,
            section,
            styles,
        )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    def add_page_number(
        canvas,
        doc,
    ):
        canvas.saveState()

        width, _ = A4

        canvas.setStrokeColor(
            colors.HexColor(
                "#cbd5e1"
            )
        )

        canvas.line(
            16 * mm,
            12 * mm,
            width - 16 * mm,
            12 * mm,
        )

        canvas.setFont(
            "Helvetica",
            7,
        )

        canvas.setFillColor(
            colors.HexColor(
                "#64748b"
            )
        )

        canvas.drawString(
            16 * mm,
            7 * mm,
            "RoadXAI — AI-assisted road inspection",
        )

        canvas.drawRightString(
            width - 16 * mm,
            7 * mm,
            f"Page {doc.page}",
        )

        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    return path


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "save_json_report",
    "render_report_text",
    "save_text_report",
    "generate_pdf_report",
]