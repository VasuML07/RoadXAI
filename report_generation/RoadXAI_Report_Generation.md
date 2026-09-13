# RoadXAI — Report Generation

The Report Generation module converts outputs from RoadXAI inference, engineering analysis, explainable AI, and 3D visualization into a structured inspection report and exports it in machine-readable and human-readable formats.

---

## 1. Report Generation — One View

```mermaid
flowchart TB
    A["RoadXAI Report Generation"]

    A --> B["Model Inference"]
    A --> C["Engineering Analysis"]
    A --> D["Explainable AI"]
    A --> E["3D Visualization"]

    B --> F["Detection Section"]
    C --> G["Engineering Measurements"]
    C --> H["Severity Assessment"]
    C --> I["Repair Cost Estimate"]
    C --> J["Recommendations"]
    D --> K["Explainable AI Section"]
    E --> L["3D Visualization Section"]

    F --> M["RoadXAIReport"]
    G --> M
    H --> M
    I --> M
    J --> M
    K --> M
    L --> M

    M --> N["JSON"]
    M --> O["TXT"]
    M --> P["PDF"]
```

---

## 2. Complete Report Pipeline

```mermaid
flowchart LR
    A["Inference Result"] --> B["generate_report()"]
    C["Engineering Result"] --> B
    D["XAI Result"] --> B
    E["Visualization Result"] --> B

    B --> F["RoadXAIReport"]

    F --> G["Executive Summary"]
    F --> H["Defect Detection"]
    F --> I["Engineering Measurements"]
    F --> J["Severity Assessment"]
    F --> K["Repair Cost Estimate"]
    F --> L["Recommendations"]
    F --> M["Explainable AI"]
    F --> N["3D Visualization"]

    F --> O["save_json_report()"]
    F --> P["save_text_report()"]
    F --> Q["generate_pdf_report()"]
```

---

## 3. Report Data Model

```mermaid
classDiagram
    class RoadXAIReport {
        +str report_id
        +str generated_at
        +str title
        +Optional~str~ image_name
        +List~ReportSection~ sections
        +Dict summary
        +to_dict()
    }

    class ReportSection {
        +str title
        +str content
        +Dict data
    }

    RoadXAIReport "1" --> "*" ReportSection
```

### Design rule

```mermaid
flowchart TB
    A["ReportSection"]

    A --> B["content"]
    A --> C["data"]

    B --> D["Human-readable"]
    C --> E["Structured / machine-readable"]

    D --> F["TXT"]
    D --> G["PDF"]

    C --> H["JSON"]
```

`ReportSection.data` stores structured information for JSON/API use, while the human-readable report formats are designed to avoid dumping internal Python dictionaries or raw JSON.

---

## 4. Report ID Generation

```mermaid
flowchart LR
    A["create_report_id()"] --> B["UTC timestamp"]
    B --> C["ROADXAI-YYYYMMDDHHMMSSffffff"]
    C --> D["Unique report identifier"]
```

The default prefix is `ROADXAI`.

---

## 5. Executive Summary

```mermaid
flowchart TB
    A["Inference Result"] --> D["Executive Summary"]
    B["Engineering Result"] --> D

    D --> E["Defect Count"]
    D --> F["Defect Pixel Percentage"]
    D --> G["Model Confidence"]
    D --> H["Road Condition"]
    D --> I["Road Health Score"]
    D --> J["Severity Information"]
    D --> K["Field Inspection Notice"]
```

The summary extracts available information from inference and engineering results and produces both readable text and structured summary data.

---

## 6. Defect Detection Section

```mermaid
flowchart TB
    A["Inference Result"]

    A --> B["Prediction Mask"]
    A --> C["Confidence"]
    A --> D["Original Image"]

    B --> E["Total Pixels"]
    B --> F["Defect Pixels"]
    E --> G["Defect Coverage %"]
    F --> G

    C --> H["Model Confidence"]
    D --> I["Image Dimensions"]

    G --> J["Defect Detection Section"]
    H --> J
    I --> J
```

The section reports the image dimensions, analyzed pixel count, defect pixels, defect coverage, and model confidence when available.

---

## 7. Engineering Measurements

```mermaid
flowchart TB
    A["Engineering Analysis"] --> B["Detected Defects"]

    B --> C["Area"]
    B --> D["Length"]
    B --> E["Width"]
    B --> F["Centroid"]

    A --> G["Calibration"]

    C --> H["Engineering Measurements Section"]
    D --> H
    E --> H
    F --> H
    G --> H

    H --> I["Pixel Measurements"]
    H --> J["Physical Measurements when calibrated"]
```

If no validated calibration is supplied, the report explicitly keeps measurements in pixels and does not interpret them as metres or square metres.

---

## 8. Severity Assessment

```mermaid
flowchart TB
    A["Engineering Result"] --> B["Severity"]
    A --> C["Road Health"]

    B --> D["Severity Classification"]
    B --> E["Severity Distribution"]

    C --> F["Road Condition"]
    C --> G["Road Health Score"]

    D --> H["Severity Assessment"]
    E --> H
    F --> H
    G --> H
```

---

## 9. Repair Cost Estimate

```mermaid
flowchart LR
    A["Engineering Result"] --> B{"Repair Cost Available?"}

    B -->|Yes| C["Estimated Area"]
    B -->|Yes| D["Rate per m²"]
    B -->|Yes| E["Estimated Cost"]
    B -->|Yes| F["Currency"]

    C --> G["Repair Cost Estimate"]
    D --> G
    E --> G
    F --> G

    B -->|No| H["Estimate Not Available"]
```

The report only presents a repair-cost estimate when the engineering result contains the required repair-cost information.

---

## 10. Deterministic Recommendations

```mermaid
flowchart TB
    A["Severity"] --> B{"Severity Level"}

    B -->|Critical| C["Immediate inspection and repair planning"]
    B -->|High| D["High-priority inspection and maintenance"]
    B -->|Moderate| E["Routine inspection and maintenance planning"]
    B -->|Low| F["Continue monitoring and routine maintenance"]

    B -->|Unknown / Other| G["Review measurements and schedule field inspection"]
```

Recommendations are generated deterministically from the reported severity information.

---

## 11. Explainable AI Section

```mermaid
flowchart LR
    A["XAI Result"] --> B["Method"]
    A --> C["Target Class"]

    B --> D["Explainable AI Section"]
    C --> D

    D --> E["Attention / Explanation Map Description"]
```

The report records the selected explainable-AI method and target class when provided.

The module describes the XAI output as showing image regions that contributed to the model prediction. It does not treat the explanation map as ground truth or causal proof.

---

## 12. 3D Visualization Section

```mermaid
flowchart TB
    A["3D Visualization Result"] --> B["Heightmap"]
    A --> C["Mesh"]

    B --> D["Heightmap Shape"]
    C --> E["Vertex Count"]
    C --> F["Face Count"]

    D --> G["3D Visualization Section"]
    E --> G
    F --> G

    G --> H["Normalized Visualization Geometry"]
    H --> I["Not Physical Depth"]
```

The report explicitly marks the vertical dimension as visualization geometry unless validated physical depth information is available.

---

## 13. Complete Section Assembly

```mermaid
flowchart TB
    A["generate_report()"]

    A --> B["create_summary_section()"]

    A --> C{"Inference Result?"}
    C -->|Yes| D["create_detection_section()"]

    A --> E{"Engineering Result?"}
    E -->|Yes| F["create_measurement_section()"]
    E -->|Yes| G["create_severity_section()"]
    E -->|Yes| H["create_cost_section()"]
    E -->|Yes| I["create_recommendations_section()"]

    A --> J{"XAI Result?"}
    J -->|Yes| K["create_xai_section()"]

    A --> L{"Visualization Result?"}
    L -->|Yes| M["create_3d_section()"]

    B --> N["Report Sections"]
    D --> N
    F --> N
    G --> N
    H --> N
    I --> N
    K --> N
    M --> N

    N --> O["RoadXAIReport"]
```

The executive summary is always created. Other sections are included when their corresponding input result is supplied.

---

## 14. Export Architecture

```mermaid
flowchart TB
    A["RoadXAIReport"]

    A --> B["JSON Export"]
    A --> C["TXT Export"]
    A --> D["PDF Export"]

    B --> E["Complete Structured Data"]
    C --> F["Human-readable Text"]
    D --> G["Professional Human-readable PDF"]
```

### JSON

```mermaid
flowchart LR
    A["RoadXAIReport"] --> B["to_dict()"]
    B --> C["JSON-safe conversion"]
    C --> D[".json"]
```

JSON preserves the complete structured report, including section data.

### TXT

```mermaid
flowchart LR
    A["RoadXAIReport"] --> B["render_report_text()"]
    B --> C["Section Titles"]
    B --> D["Section Content"]
    C --> E[".txt"]
    D --> E
```

### PDF

```mermaid
flowchart LR
    A["RoadXAIReport"] --> B["generate_pdf_report()"]
    B --> C["Report Metadata"]
    B --> D["Section Titles"]
    B --> E["Section Content"]
    C --> F["ReportLab PDF"]
    D --> F
    E --> F
```

---

## 15. Human-readable vs Structured Data

```mermaid
flowchart TB
    A["RoadXAIReport"]

    A --> B["Structured Data"]
    A --> C["Human-readable Content"]

    B --> D["JSON"]
    C --> E["TXT"]
    C --> F["PDF"]

    D --> G["Machine / API Consumption"]
    E --> H["Human Inspection"]
    F --> H
```

| Format | Purpose | Structured `data` |
|---|---|---|
| JSON | Machine/API use | Preserved |
| TXT | Human-readable report | Not dumped |
| PDF | Professional report | Not dumped |

---

## 16. PDF Generation Architecture

```mermaid
flowchart TB
    A["generate_pdf_report()"] --> B["ReportLab"]
    B --> C["A4 Document"]

    C --> D["Title"]
    C --> E["Report ID"]
    C --> F["Generation Time"]
    C --> G["Image Name"]
    C --> H["Report Sections"]
    C --> I["Footer"]

    H --> J["Section Title"]
    H --> K["Section Content"]
```

The PDF includes report metadata, human-readable sections, and a page footer. The current implementation uses ReportLab and A4 page formatting.

---

## 17. Utility Layer

```mermaid
flowchart TB
    A["Report Utilities"]

    A --> B["_get_value()"]
    A --> C["_json_safe()"]
    A --> D["_format_number()"]
    A --> E["_format_percent()"]
    A --> F["_format_optional()"]

    B --> G["Safe object/dictionary access"]
    C --> H["JSON-compatible conversion"]
    D --> I["Numeric formatting"]
    E --> J["Percentage formatting"]
    F --> K["Optional value formatting"]
```

These helpers allow report generation to work with both object-style and dictionary-style inputs and convert common NumPy, Enum, and dataclass values into safe representations.

---

## 18. Public API

```mermaid
flowchart TB
    A["Report Generation Public API"]

    A --> B["ReportSection"]
    A --> C["RoadXAIReport"]
    A --> D["create_report_id()"]
    A --> E["create_summary_section()"]
    A --> F["create_detection_section()"]
    A --> G["create_measurement_section()"]
    A --> H["create_severity_section()"]
    A --> I["create_cost_section()"]
    A --> J["create_xai_section()"]
    A --> K["create_3d_section()"]
    A --> L["create_recommendations_section()"]
    A --> M["generate_report()"]
    A --> N["save_json_report()"]
    A --> O["render_report_text()"]
    A --> P["save_text_report()"]
    A --> Q["generate_pdf_report()"]
```

---

## 19. Package Structure

```mermaid
flowchart TB
    A["report_generation"]

    A --> B["module.py"]
    A --> C["utils.py"]
    A --> D["__init__.py"]

    B --> E["Report Models"]
    B --> F["Report Section Builders"]
    B --> G["Complete Report Generator"]

    C --> H["JSON Export"]
    C --> I["TXT Rendering"]
    C --> J["PDF Generation"]
    C --> K["Formatting Helpers"]

    D --> L["Package Metadata"]
```

The package declares version `1.0.0` and author metadata in `__init__.py`.

---

## 20. End-to-End RoadXAI Reporting Flow

```mermaid
flowchart TB
    A["Road Image"]

    A --> B["CNN Segmentation"]
    B --> C["Prediction Mask"]

    C --> D["Engineering Analysis"]
    C --> E["Evaluation"]
    C --> F["Explainable AI"]
    C --> G["3D Visualization"]

    D --> H["Measurements"]
    D --> I["Severity"]
    D --> J["Road Health"]
    D --> K["Repair Cost"]

    H --> L["Report Generation"]
    I --> L
    J --> L
    K --> L
    C --> L
    F --> L
    G --> L

    L --> M["Executive Summary"]
    L --> N["Detection"]
    L --> O["Engineering"]
    L --> P["Severity"]
    L --> Q["Cost"]
    L --> R["Recommendations"]
    L --> S["XAI"]
    L --> T["3D"]

    M --> U["JSON"]
    N --> U
    O --> U
    P --> U
    Q --> U
    R --> U
    S --> U
    T --> U

    M --> V["TXT / PDF"]
    N --> V
    O --> V
    P --> V
    Q --> V
    R --> V
    S --> V
    T --> V
```

---

## 21. Research-Paper View

```mermaid
flowchart TB
    A["RoadXAI Report Generation"]

    A --> B["Multimodal Result Aggregation"]
    B --> C["Inference"]
    B --> D["Engineering Analysis"]
    B --> E["XAI"]
    B --> F["3D Visualization"]

    B --> G["Structured Inspection Representation"]

    G --> H["Executive Summary"]
    G --> I["Defect Information"]
    G --> J["Engineering Measurements"]
    G --> K["Severity / Road Health"]
    G --> L["Repair Cost"]
    G --> M["Recommendations"]
    G --> N["Interpretability"]
    G --> O["Visualization Summary"]

    G --> P["Machine-readable JSON"]
    G --> Q["Human-readable TXT"]
    G --> R["Professional PDF"]
```

### System role

The report generation layer acts as the **presentation and consolidation layer** of RoadXAI. It does not independently perform the underlying detection, segmentation, engineering calculations, XAI computation, or 3D generation. Instead, it consumes their results and produces a consistent inspection artifact.

---

## 22. Scientific / Engineering Limitations

```mermaid
flowchart TB
    A["Report Interpretation"]

    A --> B["AI-assisted result"]
    A --> C["Pixel measurements"]
    A --> D["3D visualization"]

    B --> E["Requires field confirmation"]
    C --> F["Physical dimensions require calibration"]
    D --> G["Physical depth requires validated depth data"]
```

Important limitations implemented in the reporting logic:

- AI-generated inspection results should be confirmed by appropriate field inspection before repair decisions.
- Pixel measurements are not physical measurements without validated calibration.
- The 3D vertical dimension is visualization geometry unless external physical depth information is available.
- A report is an aggregation of upstream outputs; it does not independently validate the correctness of those upstream outputs.

---

## 23. Final Architecture

```mermaid
flowchart TB
    A["RoadXAI"]

    A --> B["Inference"]
    A --> C["Engineering Analysis"]
    A --> D["Explainable AI"]
    A --> E["3D Visualization"]

    B --> F["Report Generation"]
    C --> F
    D --> F
    E --> F

    F --> G["RoadXAIReport"]

    G --> H["Executive Summary"]
    G --> I["Detection"]
    G --> J["Engineering Measurements"]
    G --> K["Severity Assessment"]
    G --> L["Repair Cost"]
    G --> M["Recommendations"]
    G --> N["Explainable AI"]
    G --> O["3D Visualization"]

    G --> P["JSON"]
    G --> Q["TXT"]
    G --> R["PDF"]
```

> **Scope note:** The Report Generation module consolidates existing RoadXAI outputs into structured and human-readable inspection reports. It does not perform model inference, segmentation, engineering analysis, XAI computation, or physical depth measurement.
