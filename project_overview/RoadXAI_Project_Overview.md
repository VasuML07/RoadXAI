# RoadXAI — Project Overview

RoadXAI is an explainable AI-based road infrastructure intelligence system for detecting, segmenting, analyzing, and reporting road damage from images.

The project overview module provides **static project-level metadata and configuration**. It does not perform training, inference, dataset loading, segmentation, explainability, or engineering calculations.

---

## 1. Project Overview — One View

```mermaid
flowchart TB
    A["RoadXAI"] --> B["Project Metadata"]
    A --> C["Supported Damage Classes"]
    A --> D["Project Objectives"]
    A --> E["Project Configuration"]
    A --> F["Project Paths"]
    A --> G["Validation Utilities"]

    B --> B1["Name: RoadXAI"]
    B --> B2["Version: 0.1.0"]
    B --> B3["Description"]

    C --> C1["crack"]
    C --> C2["pothole"]

    D --> D1["Detection"]
    D --> D2["Classification"]
    D --> D3["Segmentation"]
    D --> D4["Severity"]
    D --> D5["Explainability"]
    D --> D6["Structured Inspection"]
    D --> D7["Engineering Assessment"]
    D --> D8["Inspection Reports"]

    E --> E1["512 × 512"]
    E --> E2["3 Image Channels"]
    E --> E3["1 Mask Channel"]
    E --> E4["80% Train / 20% Validation"]
    E --> E5["Random Seed 42"]

    F --> F1["Project Root"]
    F --> F2["datasets/"]
    F --> F3["CRACK500"]
    F --> F4["Pothole_Segmentation_YOLOv8"]
    F --> F5["PUBLIC POTHOLE DATASET"]
```

---

## 2. Project Identity

```mermaid
flowchart LR
    A["Project Identity"] --> B["RoadXAI"]
    A --> C["Version 0.1.0"]
    A --> D["Explainable AI"]
    A --> E["Road Infrastructure Intelligence"]
```

---

## 3. Supported Damage Classes

```mermaid
flowchart TB
    A["RoadXAI Damage Classes"] --> B["crack"]
    A --> C["pothole"]
```

The current project-level metadata defines exactly two supported damage classes:

| Class | Meaning |
|---|---|
| `crack` | Road crack |
| `pothole` | Road pothole |

---

## 4. Project Objectives

```mermaid
flowchart TB
    A["RoadXAI Objectives"]

    A --> B["Detect road damage from road images"]
    A --> C["Classify different types of road damage"]
    A --> D["Localize road damage using image segmentation"]
    A --> E["Estimate damage severity"]
    A --> F["Provide explainable AI outputs"]
    A --> G["Extract structured information from visual road inspections"]
    A --> H["Support engineering-oriented road condition assessment"]
    A --> I["Generate consistent and interpretable inspection reports"]
```

---

## 5. Project-Level Workflow

```mermaid
flowchart LR
    A["Road Image"] --> B["Damage Detection"]
    B --> C["Damage Classification"]
    C --> D["Segmentation"]
    D --> E["Damage Localization"]
    E --> F["Severity Estimation"]
    F --> G["Explainable AI"]
    G --> H["Structured Inspection"]
    H --> I["Engineering-Oriented Assessment"]
    I --> J["Inspection Report"]
```

This represents the intended project-level intelligence flow described by the project metadata. The project overview module itself only stores and exposes this information.

---

## 6. Relationship With RoadXAI Modules

```mermaid
flowchart TB
    A["Project Overview"]

    A --> B["Dataset Pipeline"]
    A --> C["CNN Segmentation"]
    A --> D["Engineering Analysis"]
    A --> E["Evaluation"]
    A --> F["Explainable AI"]
    A --> G["3D Visualization"]

    B --> H["Prepared Road Images + Masks"]
    C --> I["Segmentation Predictions"]
    I --> D
    I --> E
    I --> F
    I --> G
    D --> J["Severity / Road Condition / Repair Analysis"]
    E --> K["Model Performance"]
    F --> L["Heatmaps / Model Explanations"]
    G --> M["Visual Inspection"]
```

The overview package does not execute these modules; it provides the project-level identity, objectives, classes, configuration, and paths used to organize the system.

---

## 7. Project Configuration

```mermaid
flowchart TB
    A["Project Configuration"] --> B["Image Configuration"]
    A --> C["Dataset Configuration"]
    A --> D["Damage Configuration"]

    B --> B1["IMAGE_SIZE = 512 × 512"]
    B --> B2["IMAGE_CHANNELS = 3"]
    B --> B3["MASK_CHANNELS = 1"]

    C --> C1["TRAIN_SPLIT = 0.80"]
    C --> C2["VAL_SPLIT = 0.20"]
    C --> C3["RANDOM_SEED = 42"]

    D --> D1["NUM_CLASSES = 2"]
    D --> D2["crack"]
    D --> D3["pothole"]
```

---

## 8. Dataset Structure

```mermaid
flowchart TB
    A["RoadXAI Project Root"] --> B["datasets/"]

    B --> C["CRACK500"]
    B --> D["Pothole_Segmentation_YOLOv8"]
    B --> E["PUBLIC POTHOLE DATASET"]

    C --> F["Road Damage Images"]
    D --> G["Pothole Data"]
    E --> H["Public Pothole Data"]
```

The project paths module centralizes these dataset locations rather than scattering filesystem paths throughout the project.

---

## 9. Metadata API

```mermaid
flowchart LR
    A["Project Metadata"] --> B["get_project_info()"]
    A --> C["get_damage_classes()"]
    A --> D["get_project_objectives()"]

    B --> E["Complete Metadata Dictionary"]
    C --> F["Damage Class List"]
    D --> G["Objective List"]
```

The public project-overview interface exports:

```text
PROJECT_NAME
PROJECT_VERSION
PROJECT_DESCRIPTION
PROJECT_OBJECTIVES
DAMAGE_CLASSES
```

---

## 10. Project Schema

```mermaid
classDiagram
    class ProjectSchema {
        +str name
        +str version
        +str description
        +List~str~ damage_classes
        +List~str~ objectives
    }

    ProjectSchema --> "validated by" validate_project_schema
    ProjectSchema --> "created by" create_project_schema
```

The schema is an immutable (`frozen=True`) dataclass representing structured RoadXAI project information.

---

## 11. Validation Architecture

```mermaid
flowchart TB
    A["Project Metadata / Configuration"]

    A --> B["validate_project_metadata()"]
    A --> C["validate_project_schema()"]
    A --> D["validate_config()"]
    A --> E["validate_constants()"]

    B --> F{"Valid?"}
    C --> G{"Valid?"}
    D --> H{"Valid?"}
    E --> I{"Valid?"}

    F --> J["PASS / FAIL"]
    G --> J
    H --> J
    I --> J
```

Validation checks include:

- Non-empty project name
- Non-empty version
- Non-empty description
- At least one objective
- At least one damage class
- No empty objective/class entries
- Valid image dimensions and channels
- Consistent class count
- Train/validation proportions
- Valid random seed
- Consistent centralized configuration

---

## 12. Project Constants

| Parameter | Current Value |
|---|---:|
| Project | RoadXAI |
| Version | 0.1.0 |
| Image width | 512 |
| Image height | 512 |
| Image channels | 3 |
| Mask channels | 1 |
| Training split | 0.80 |
| Validation split | 0.20 |
| Random seed | 42 |
| Number of classes | 2 |
| Classes | crack, pothole |

---

## 13. Project Overview Package Boundary

```mermaid
flowchart TB
    A["project_overview"]

    A --> B["module.py"]
    A --> C["schema.py"]
    A --> D["config.py"]
    A --> E["constants.py"]
    A --> F["paths.py"]
    A --> G["utils.py"]
    A --> H["__init__.py"]

    B --> I["Static Metadata"]
    C --> J["Structured Schema"]
    D --> K["Project Configuration"]
    E --> L["Centralized Constants"]
    F --> M["Filesystem Paths"]
    G --> N["Display + Validation Utilities"]
    H --> O["Public Interface"]
```

---

## 14. Separation of Responsibilities

```mermaid
flowchart LR
    A["Project Overview"] --> B["Defines"]
    A --> C["Validates"]
    A --> D["Exposes"]

    B --> B1["Identity"]
    B --> B2["Objectives"]
    B --> B3["Damage Classes"]
    B --> B4["Configuration"]
    B --> B5["Paths"]

    C --> C1["Metadata"]
    C --> C2["Schema"]
    C --> C3["Configuration"]
    C --> C4["Constants"]

    D --> D1["Project Information"]
    D --> D2["Damage Classes"]
    D --> D3["Objectives"]
```

### Explicitly outside this module

```text
Training
Inference
Dataset loading
Image segmentation
Model evaluation
Grad-CAM / XAI computation
Engineering calculations
3D mesh generation
```

---

## 15. Research-Paper View

```mermaid
flowchart TB
    A["RoadXAI System Definition"]
    A --> B["Problem Definition"]
    A --> C["Target Damage Classes"]
    A --> D["System Objectives"]
    A --> E["Standardized Input Configuration"]
    A --> F["Dataset Organization"]
    A --> G["Module Boundaries"]
    A --> H["Validation"]

    B --> I["Road Damage Analysis"]
    C --> I
    D --> I
    E --> I
    F --> I
    G --> I
    H --> I
```

### System-level role

The project overview establishes the **formal specification of RoadXAI**:

1. Defines what the project is.
2. Defines which road-damage classes are currently supported.
3. Defines the project's intended objectives.
4. Establishes common image, mask, split, and class configuration.
5. Centralizes dataset paths.
6. Provides validation for project-level information.
7. Exposes a consistent public interface to other project components.

---

## 16. Complete RoadXAI Architecture

```mermaid
flowchart TB
    A["RoadXAI Project Overview"]

    A --> B["Datasets"]
    A --> C["CNN Segmentation"]
    A --> D["Evaluation"]
    A --> E["Engineering Analysis"]
    A --> F["Explainable AI"]
    A --> G["3D Visualization"]

    B --> H["Images + Ground-Truth Masks"]
    H --> C

    C --> I["Predicted Damage Masks"]

    I --> D
    I --> E
    I --> F
    I --> G

    D --> J["Accuracy / Precision / Recall / F1 / IoU / Dice"]
    E --> K["Defect Measurements / Severity / Road Health / Repair Cost"]
    F --> L["Grad-CAM / Grad-CAM++ / Score-CAM"]
    G --> M["Heightmap / Mesh / Visual Inspection"]

    J --> N["RoadXAI Inspection Intelligence"]
    K --> N
    L --> N
    M --> N
```

---

## 17. Key Technical Summary

| Component | Role |
|---|---|
| `module.py` | Project identity, description, objectives, damage classes |
| `schema.py` | Structured immutable project schema |
| `config.py` | Central project configuration |
| `constants.py` | Shared project constants |
| `paths.py` | Centralized project and dataset paths |
| `utils.py` | Project overview display and metadata validation |
| `__init__.py` | Public project-overview interface |

---

## 18. Final Architecture

```mermaid
flowchart TB
    A["RoadXAI"]

    A --> B["Project Definition"]
    A --> C["Data"]
    A --> D["Learning"]
    A --> E["Evaluation"]
    A --> F["Engineering"]
    A --> G["Explainability"]
    A --> H["Visualization"]

    B --> B1["Metadata"]
    B --> B2["Configuration"]
    B --> B3["Constants"]
    B --> B4["Paths"]
    B --> B5["Validation"]

    C --> C1["CRACK500"]
    C --> C2["Pothole_Segmentation_YOLOv8"]
    C --> C3["PUBLIC POTHOLE DATASET"]

    D --> D1["U-Net Segmentation"]

    E --> E1["Pixel Metrics"]
    E --> E2["Inference Timing"]

    F --> F1["Defect Geometry"]
    F --> F2["Severity"]
    F --> F3["Road Health"]
    F --> F4["Repair Cost"]

    G --> G1["Grad-CAM"]
    G --> G2["Grad-CAM++"]
    G --> G3["Score-CAM"]

    H --> H1["Heightmap"]
    H --> H2["3D Mesh"]
    H --> H3["Inspection Views"]
```

> **Scope note:** The project overview is a metadata/configuration layer. It defines and validates the system specification but does not itself execute the machine-learning, segmentation, evaluation, XAI, engineering-analysis, or visualization pipelines.
