# RoadXAI — 3D Visualization Module

## 1. Module in One Diagram

```mermaid
flowchart LR
    A[Road Image] --> B[AI Segmentation]
    B --> C[Defect Mask]
    C --> D[3D Visualization]
    E[Engineering Analysis] --> D
    F[AI Confidence] --> D
    G[XAI Heatmap] --> D
    D --> H[Interactive 3D Inspection]
```

**Goal:** turn 2D AI road-damage results into an easier-to-understand interactive 3D inspection view.

---

## 2. Core Pipeline

```mermaid
flowchart TD
    A[Defect Mask] --> B[Validate Mask]
    B --> C[Find Defect Regions]
    C --> D[Generate Visual Depth]
    D --> E[Heightmap]
    E --> F[3D Coordinates]
    F --> G[Triangular Mesh]
    G --> H[Image-Aligned 3D Surface]
    H --> I[Plotly Interactive Viewer]
```

---

## 3. What Happens to the Mask?

```mermaid
flowchart LR
    A["2D Segmentation Mask"] --> B["Connected Components"]
    B --> C["Distance Transform"]
    C --> D["Normalized Visual Depth"]
    D --> E["Smooth Heightmap"]
    E --> F["3D Depression"]
```

Conceptually:

```text
Road surface       ─────────────────
Defect boundary    ───────╲    ╱────
Defect center       ────────╲__╱─────
                         visual depth
```

---

## 4. Heightmap → 3D Mesh

```mermaid
flowchart LR
    A[Heightmap H × W] --> B[X/Y Coordinate Grid]
    B --> C[Z = -Visual Depth]
    C --> D[3D Vertices]
    D --> E[Triangular Faces]
    E --> F[3D Mesh]
```

```text
       X →
    ┌───────────────┐
 Y  │ •──•──•──•──• │
 ↓  │ ╱╲ ╱╲ ╱╲ ╱╲  │
    │ •──•──•──•──• │
    │ ╱╲ ╱╲ ╱╲ ╱╲  │
    └───────────────┘
             ↓
        3D surface
```

---

## 5. Image + 3D Surface

```mermaid
flowchart TD
    A[Original Road Image] --> C[Image-Aligned Surface]
    B[Visual Depth] --> C
    C --> D[3D Inspection]
```

```text
        Original image
    ┌────────────────────┐
    │      ROAD          │
    │   ███ defect       │
    │      ███           │
    └────────────────────┘
             +
       visual geometry
             ↓
    ┌────────────────────┐
    │      ROAD          │
    │       \___/        │
    │          \___      │
    └────────────────────┘
          3D view
```

---

## 6. Additional AI / Engineering Information

```mermaid
flowchart LR
    A[3D Surface] --> E[Interactive Inspection]

    B[Severity] --> E
    C[Probability] --> E
    D[XAI Heatmap] --> E
    F[Engineering Measurements] --> E
```

A defect can therefore be connected to:

```text
Defect #1
├── Severity
├── Severity score
├── Area
├── Length
├── Width
├── Centroid
└── AI confidence
```

---

## 7. Visualization Modes

```mermaid
flowchart TD
    A[Same 3D Surface]
    A --> B[Inspection]
    A --> C[Original]
    A --> D[Severity]
    A --> E[Depth]
    A --> F[Probability / Confidence]
    A --> G[XAI / Attention]
```

The geometry stays the same; the **visual information displayed on it changes**.

---

## 8. Defect Markers

```mermaid
flowchart LR
    A[Defect Region] --> B[Centroid]
    A --> C[Boundary]
    B --> D[Numbered 3D Marker]
    C --> E[3D Boundary]
    D --> F[Hover Information]
    E --> F
```

This lets the user connect a 3D location with its corresponding defect information.

---

## 9. XAI Connection

```mermaid
flowchart LR
    A[Road Image] --> D[3D Inspection]
    B[Segmentation] --> D
    C[XAI Attention] --> D
    E[Engineering Data] --> D
```

**Interpretation:**

```text
Where is damage?
        ↓
What region did AI detect?
        ↓
Where is model attention concentrated?
        ↓
What engineering information belongs to it?
```

---

## 10. Scientific Limitation

```mermaid
flowchart TD
    A[Single RGB Image] --> B[Segmentation]
    B --> C[Visual Depth]
    C --> D[3D Representation]

    E[Calibrated Depth Source] --> F[Physical Depth]
```

### Important

```text
RGB image
   ↓
Visual depth ≠ Physical depth
```

The current module represents **normalized visual geometry**.

It must **not** be reported as centimetres or millimetres unless a calibrated external depth source is supplied.

---

## 11. Why It Helps

```mermaid
flowchart LR
    A[2D AI Result] --> B[3D Visualization]
    B --> C[Spatial Understanding]
    B --> D[Defect Identification]
    B --> E[Severity / Engineering Context]
    B --> F[XAI Interpretation]
    C --> G[Better Road Inspection]
    D --> G
    E --> G
    F --> G
```

### Core contribution

> **Transforms complex AI road-inspection outputs into an interactive spatial representation that is easier to inspect and communicate.**

---

## 12. Outputs

```mermaid
flowchart LR
    A[Visualization Pipeline] --> B[Heightmap]
    A --> C[3D Mesh]
    A --> D[Plotly Figure]
    A --> E[Metadata]
    D --> F[Interactive HTML]
```

| Output | Purpose |
|---|---|
| Heightmap | Visual depth representation |
| Mesh | 3D geometric representation |
| Plotly figure | Interactive inspection |
| Metadata | Visualization statistics |
| HTML | Standalone interactive visualization |

---

## 13. Research-Paper Description

```text
AI Segmentation
      ↓
Normalized Visual Depth
      ↓
Image-Aligned 3D Surface
      ↓
Engineering + XAI Overlays
      ↓
Interactive Road-Defect Inspection
```

**Key terms:** image-aligned 3D visualization · normalized visual depth · road-defect inspection · interactive visualization · XAI integration
