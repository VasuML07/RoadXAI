# RoadXAI Engineering Analysis Module

## 1. Engineering Analysis — One View

```mermaid
flowchart LR
    A[Predicted Defect Mask] --> B[Validate / Normalize]
    B --> C[Extract Defect Regions]
    C --> D[Measure Geometry]
    D --> E[Severity Analysis]
    D --> F[Road Health]
    D --> G[Repair Cost]
    E --> H[Engineering Result]
    F --> H
    G --> H
```

**Purpose:** convert segmentation masks into engineering-oriented measurements and road-condition indicators.

---

## 2. Module → Engineering Outputs

```mermaid
flowchart TD
    A[Defect Mask]
    --> B[Defect Detection]

    B --> C["Area"]
    B --> D["Length"]
    B --> E["Width"]
    B --> F["Centroid"]
    B --> G["Bounding Box"]

    C --> H[Severity]
    D --> H
    E --> H

    C --> I[Road Health]
    C --> J[Repair Cost]
```

---

## 3. Complete Internal Workflow

```mermaid
flowchart TD
    A[Segmentation Mask]
    --> B[Validate Mask]

    B --> C[External Contours]
    C --> D[Filter Small Components]

    D --> E[Measure Each Defect]

    E --> F["Area in Pixels"]
    E --> G["Length in Pixels"]
    E --> H["Width in Pixels"]
    E --> I["Centroid"]
    E --> J["Bounding Box"]

    F --> K[Area Ratio]
    G --> L[Severity]
    H --> L
    K --> L

    F --> M[Overall Road Health]

    E --> N{Calibration Available?}
    N -->|Yes| O[Physical Measurements]
    N -->|No| P[Pixel Measurements Only]

    O --> Q{Repair Rate Available?}
    Q -->|Yes| R[Repair Cost]
    Q -->|No| S[No Cost Estimate]

    L --> T[Engineering Analysis Result]
    M --> T
    R --> T
```

---

## 4. Mask Processing

```mermaid
flowchart LR
    A[Input Mask] --> B[Shape Validation]
    B --> C["Accept [H,W] or [H,W,1]"]
    C --> D[Finite Value Check]
    D --> E["Convert > 0 to 255"]
    E --> F[Binary uint8 Mask]
```

The module works on a normalized binary representation:

```text
Background → 0
Defect     → 255
```

---

## 5. Defect Region Extraction

```mermaid
flowchart LR
    A[Binary Mask]
    --> B[OpenCV findContours]
    --> C[External Contours]
    --> D[Individual Defect Regions]
```

```mermaid
flowchart TD
    A[All Contours] --> B{Area < 10 px?}
    B -->|Yes| C[Ignore]
    B -->|No| D[Measure Defect]
    C --> E[Next Contour]
    D --> E
```

Default minimum defect area:

```text
min_area_pixels = 10
```

---

## 6. Defect Geometry

```mermaid
flowchart TD
    A[Defect Contour]
    --> B[Contour Area]
    A --> C[Minimum-Area Rectangle]
    A --> D[Bounding Rectangle]
    A --> E[Contour Moments]

    B --> F["Area (pixels²)"]
    C --> G["Length = larger dimension"]
    C --> H["Width = smaller dimension"]
    D --> I["Bounding-box width / height"]
    E --> J["Centroid (x,y)"]
```

---

## 7. Pixel → Physical Measurements

```mermaid
flowchart LR
    A[Pixel Measurement]
    --> B{Calibration?}

    B -->|No| C[Keep Pixel Units]

    B -->|Yes| D["pixels_per_meter"]
    D --> E["Length ÷ pixels_per_meter"]
    D --> F["Area ÷ pixels_per_meter²"]

    E --> G[Length in meters]
    E --> H[Width in meters]
    F --> I[Area in m²]
```

Calibration is explicitly required for physical measurements.

```text
pixels_per_meter > 0
```

---

## 8. Severity Calculation

```mermaid
flowchart TD
    A[Defect Area Ratio]
    --> B["Area Score
min(ratio × 1000, 70)"]

    C[Length in meters] --> D["Length Score
min(length × 5, 15)"]
    E[Width in meters] --> F["Width Score
min(width × 10, 15)"]

    B --> G[Total Score]
    D --> G
    F --> G

    G --> H["Clamp to 0–100"]
    H --> I{Severity}

    I -->|0–24| J[LOW]
    I -->|25–49| K[MODERATE]
    I -->|50–74| L[HIGH]
    I -->|75–100| M[CRITICAL]
```

**Important:** physical dimensions contribute to severity only when calibration is available.

---

## 9. Road Health

```mermaid
flowchart LR
    A[All Defect Areas]
    --> B[Total Defect Area]

    B --> C[÷ Road Area]
    C --> D[Defect Area Ratio]

    D --> E["100 − ratio × 100"]
    E --> F[Road Health Score]
    F --> G[Condition]
```

```mermaid
flowchart LR
    A[Score] --> B{Condition}

    B -->|90–100| C[Excellent]
    B -->|75–89| D[Good]
    B -->|50–74| E[Fair]
    B -->|25–49| F[Poor]
    B -->|0–24| G[Critical]
```

The health score is based on the fraction of road area occupied by detected defects.

---

## 10. Repair Cost

```mermaid
flowchart TD
    A[Defects]
    --> B[Calibrated Area]

    B --> C[Total Defect Area m²]
    C --> D[Repair Rate per m²]

    D --> E["Estimated Cost
= Area × Rate"]

    E --> F[Currency]
    F --> G[Repair Cost Result]
```

Repair-cost estimation requires:

```text
Calibration
+
Area in m²
+
Repair rate per m²
```

The default currency is `INR`.

---

## 11. Full Engineering Result

```mermaid
flowchart TD
    A[analyze_road]
    --> B[Defect Measurements]
    A --> C[Severity Results]
    A --> D[Road Health Result]
    A --> E[Repair Cost Result]

    B --> F[EngineeringAnalysisResult]
    C --> F
    D --> F
    E --> F
```

### Result structure

```text
EngineeringAnalysisResult
├── defects
│   ├── area
│   ├── length
│   ├── width
│   ├── centroid
│   └── bounding box
├── severity
├── road_health
└── repair_cost
```

---

## 12. Engineering Analysis → RoadXAI

```mermaid
flowchart LR
    A[CNN Segmentation]
    --> B[Defect Mask]

    B --> C[Engineering Analysis]

    C --> D[Measurements]
    C --> E[Severity]
    C --> F[Road Health]
    C --> G[Repair Cost]

    D --> H[3D Visualization]
    E --> H
    F --> H
    G --> H

    C --> I[XAI / Reporting]
```

The module is the bridge between **pixel-level segmentation** and **engineering-oriented interpretation**.

---

## 13. Visualization Utilities

```mermaid
flowchart LR
    A[Image + Mask]
    --> B[Contours]
    --> C[Defect Boundaries]

    D[Defect Measurements]
    --> E[Defect Labels]

    C --> F[Annotated Image]
    E --> F
```

Utility functions also support:

```text
Mask cleaning
Small-component removal
Area / length formatting
Cost formatting
Severity summaries
JSON export
```

---

## 14. Result Export

```mermaid
flowchart LR
    A[EngineeringAnalysisResult]
    --> B[Convert Dataclasses]
    B --> C[JSON-Safe Dictionary]
    C --> D[JSON File]
```

This allows the engineering results to be stored and reused outside the visualization interface.

---

## 15. Why This Module Matters

```mermaid
flowchart LR
    A[AI Prediction]
    --> B[Quantitative Measurements]
    --> C[Severity]
    --> D[Road Condition]
    --> E[Maintenance Information]
```

Without engineering analysis:

```text
Image → Defect Mask
```

With engineering analysis:

```text
Image
  ↓
Defect Mask
  ↓
Area / Length / Width
  ↓
Severity
  ↓
Road Health
  ↓
Repair-Cost Estimate
```

---

## 16. Research-Paper View

```mermaid
flowchart TD
    A[Predicted Segmentation]
    --> B[Contour Extraction]
    --> C[Geometric Measurement]
    --> D[Calibration]
    --> E[Severity Assessment]
    --> F[Road Health Assessment]
    --> G[Repair Cost Estimation]
    --> H[Engineering Interpretation]
```

### Key Technical Points

| Component | Current implementation |
|---|---|
| Input | Binary segmentation mask |
| Region extraction | External OpenCV contours |
| Small-region filtering | Default 10 pixels |
| Area | Contour area |
| Length / width | Minimum-area rectangle |
| Centroid | Contour moments |
| Physical calibration | Pixels per meter |
| Severity | Deterministic 0–100 score |
| Severity levels | Low / Moderate / High / Critical |
| Road health | 0–100 score |
| Road conditions | Excellent / Good / Fair / Poor / Critical |
| Repair cost | Area × rate per m² |
| Default currency | INR |
| Output | Structured engineering result + JSON export |

> **Scientific limitation:** pixel measurements become physical measurements only when a valid calibration scale is supplied. The module does not independently infer real-world dimensions from an RGB image.
