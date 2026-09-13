# RoadXAI

> **AI-assisted road-surface inspection, segmentation, engineering analysis, explainability, 3D visualization, and reporting — in one pipeline.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-5c3ee8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Research%2FProject-lightgrey)](#license)

---

## `01` — What is RoadXAI?

RoadXAI converts a road image into an **AI-assisted inspection result**:

```text
ROAD IMAGE
    │
    ▼
┌─────────────────────┐
│ Dataset / Preprocess │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ U-Net Segmentation  │
└──────────┬──────────┘
           ▼
     DEFECT MASK
           │
     ┌─────┼──────────────┬──────────────┐
     ▼     ▼              ▼              ▼
  Metrics Engineering    XAI             3D
     │     │              │              │
     └─────┴──────────────┴──────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Road Inspection │
              │ + Decision View │
              └───────┬────────┘
                      ▼
              JSON / TXT / PDF
```

The system is designed as a modular research/engineering pipeline rather than a single prediction script.

---

## `02` — The Core Idea

**Detect → Measure → Explain → Visualize → Report**

| Stage | RoadXAI output |
|---|---|
| Detection | Pixel-level segmentation mask |
| Measurement | Defect area, length, width, centroid and related measurements |
| Severity | LOW / MODERATE / HIGH / CRITICAL |
| Road health | Health score + condition |
| Explainability | Grad-CAM-based attention visualization |
| 3D | Interactive visual surface representation |
| Decision support | Maintenance priority + inspection indication |
| Reporting | JSON, TXT and PDF |

> **Important:** RoadXAI is an AI-assisted screening and decision-support system. Physical measurements and engineering decisions require appropriate calibration and field validation.

---

<details>
<summary><strong>⚡ Quick Start</strong></summary>

### 1. Create environment

```bash
python -m venv .venv
```

### 2. Activate it

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows CMD**
```cmd
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch RoadXAI

```bash
streamlit run streamlit_app/app.py
```

### 5. Open the application

Streamlit will display the local application URL in the terminal.

</details>

---

## `03` — Project Architecture

```text
RoadXAI/
│
├── dataset_pipeline/
│   └── Dataset loading, preprocessing and validation
│
├── models/
│   └── CNN / U-Net segmentation architecture
│
├── training/
│   └── Training, validation, metrics, checkpoints and logging
│
├── engineering/
│   └── Defect measurements, severity, road health and cost logic
│
├── evaluation/
│   └── Model evaluation and dataset-level metrics
│
├── explainable_ai/
│   └── Grad-CAM family explainability
│
├── visualization_3d/
│   └── Interactive road-surface visualization
│
├── report_generation/
│   └── Structured JSON / TXT / PDF reports
│
├── streamlit_app/
│   └── Interactive RoadXAI application
│
├── checkpoints/
│   └── Trained model checkpoints
│
├── reports/
│   └── Generated inspection reports
│
├── runs/
│   └── TensorBoard logs
│
├── requirements.txt
└── .gitignore
```

---

## `04` — End-to-End Pipeline

```text
                    ┌──────────────────┐
                    │   ROAD IMAGES    │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │ Dataset Pipeline   │
                  │ resize / tensors   │
                  │ masks / loaders    │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │   U-Net CNN        │
                  │   segmentation     │
                  └─────────┬──────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ DEFECT MASK │
                     └──────┬──────┘
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
        ▼                   ▼                    ▼
 ┌─────────────┐    ┌───────────────┐    ┌─────────────┐
 │ Engineering │    │ Explainable AI│    │ 3D Surface  │
 │ Analysis    │    │ Grad-CAM      │    │ Visualization│
 └──────┬──────┘    └───────┬───────┘    └──────┬──────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            ▼
                    ┌───────────────┐
                    │ Streamlit UI  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Report Engine │
                    └───────┬───────┘
                            ▼
                   JSON / TXT / PDF
```

---

## `05` — Model

RoadXAI uses a **U-Net-style binary segmentation network**.

```text
INPUT
  │
  ▼
┌─────────┐
│ Encoder │ ───────────────┐
└────┬────┘                │
     ▼                     │ Skip
┌─────────┐                │ Connections
│ Encoder │ ───────────┐   │
└────┬────┘            │   │
     ▼                 │   │
┌─────────┐            │   │
│ Encoder │ ───────┐   │   │
└────┬────┘        │   │   │
     ▼             │   │   │
┌─────────────┐    │   │   │
│ Bottleneck  │    │   │   │
└──────┬──────┘    │   │   │
       ▼           │   │   │
┌─────────────┐    │   │   │
│   Decoder   │ ◄──┘   │   │
└──────┬──────┘        │   │
       ▼               │   │
┌─────────────┐        │   │
│   Decoder   │ ◄──────┘   │
└──────┬──────┘            │
       ▼                   │
┌─────────────┐            │
│   Decoder   │ ◄──────────┘
└──────┬──────┘
       ▼
┌─────────────┐
│ 1×1 Conv    │
│ 1 Logit     │
└──────┬──────┘
       ▼
BINARY DEFECT MASK
```

### Prediction

```text
logits
  │
  ▼
sigmoid
  │
  ▼
probability map
  │
  ▼
threshold = 0.5
  │
  ▼
binary mask
```

---

## `06` — Training

The training pipeline provides:

- Training loop
- Validation loop
- Dice and IoU tracking
- Automatic mixed precision on CUDA
- Gradient clipping
- Learning-rate scheduling
- Early stopping
- Best-model checkpointing
- TensorBoard logging
- Training-history persistence
- Checkpoint resume support

```text
TRAIN
  │
  ├── forward
  ├── loss
  ├── backward
  ├── gradient clipping
  └── optimizer step
  │
  ▼
VALIDATE
  │
  ├── validation loss
  ├── Dice
  └── IoU
  │
  ▼
SCHEDULER
  │
  ▼
BEST VALIDATION LOSS?
  ├── YES → best.pt
  └── NO  → early-stopping counter
```

Default training configuration:

```text
epochs                   = 10
threshold                = 0.5
early_stopping_patience  = 5
AMP                      = enabled on CUDA
max_grad_norm            = 1.0
save_best_only           = True
checkpoint_dir           = checkpoints
TensorBoard               = runs/roadxai
```

---

## `07` — Evaluation

RoadXAI evaluates segmentation using:

```text
Accuracy
Precision
Recall
F1 / Dice
IoU
Inference Time
```

The evaluation stage compares model behavior across supported datasets and provides quantitative evidence for model performance.

---

## `08` — Engineering Analysis

The segmentation mask is transformed into engineering-oriented information.

```text
MASK
 │
 ├── connected defect regions
 │
 ├── area
 ├── length
 ├── width
 ├── centroid
 └── optional calibrated physical dimensions
             │
             ▼
       SEVERITY SCORE
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
    LOW   MODERATE   HIGH
                    │
                    ▼
                  CRITICAL
```

Severity levels are based on the implemented scoring logic.

Road health is derived from defect-area ratio:

```text
Road Health Score
        │
        ├── Excellent
        ├── Good
        ├── Fair
        ├── Poor
        └── Critical
```

> Physical area, length, width and repair cost require appropriate calibration. Pixel measurements alone do not establish real-world dimensions.

---

## `09` — Explainable AI

RoadXAI provides model-attention visualization through:

```text
Grad-CAM
Grad-CAM++
Score-CAM
```

Conceptually:

```text
IMAGE
  │
  ▼
CNN
  │
  ├──────────────► SEGMENTATION
  │
  ▼
TARGET LAYER
  │
  ▼
ACTIVATIONS + GRADIENTS
  │
  ▼
CAM GENERATION
  │
  ▼
NORMALIZED HEATMAP
  │
  ▼
IMAGE OVERLAY
```

The current application uses Grad-CAM with the model's default target layer.

> XAI heatmaps indicate model activation/attention behavior. They are not ground-truth maps and are not causal proof.

---

## `10` — 3D Visualization

The 3D module converts segmentation information into an interactive visual surface.

```text
MASK
 │
 ├── connected components
 ├── distance transform
 ├── area factor
 ├── visual depth
 └── smoothing
       │
       ▼
   HEIGHT MAP
       │
       ▼
   MESH / SURFACE
       │
       ▼
     PLOTLY
       │
       ▼
INTERACTIVE 3D VIEW
```

Available inspection views include:

```text
Inspection
Road Surface
Severity
AI Confidence
AI Attention
```

Additional controls include visual depth, markers and boundaries.

> The generated vertical dimension is visualization geometry unless validated depth information is supplied.

---

## `11` — Streamlit Application

The application provides two primary views:

```text
┌──────────────────────────────────────────┐
│              RoadXAI                     │
├──────────────────────────────────────────┤
│ Upload Road Image                        │
│                                          │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ Road         │  │ Defect Overlay   │ │
│  │ Image        │  │ + Numbering      │ │
│  └──────────────┘  └──────────────────┘ │
│                                          │
│  Area        Confidence       Pixels     │
│                                          │
│  Condition / Inspection Requirement      │
└──────────────────────────────────────────┘
```

### Road Inspection

Focused on the operational result:

- Original image
- Numbered defect overlay
- Defect measurements
- Area
- Confidence
- Defect pixels
- Road condition
- Inspection indication
- Explanations

### Advanced Analysis

Provides:

- Inference summary
- Probability maps
- Engineering summary
- Severity distribution
- Grad-CAM
- Interactive 3D
- Road decision summary
- Report generation

---

## `12` — Reports

RoadXAI can generate:

```text
┌──────────────┐
│ Analysis     │
│ Results      │
└──────┬───────┘
       ▼
┌──────────────┐
│ Report Model │
└──────┬───────┘
       │
       ├──────► JSON
       ├──────► TXT
       └──────► PDF
```

Reports can consolidate:

- Inspection summary
- Defect detection
- Measurements
- Severity
- Road condition
- Cost-related information when available
- Recommendations
- XAI information
- 3D visualization information

Generated reports are stored under:

```text
reports/
```

---

## `13` — Dataset Support

The project contains a dedicated dataset pipeline for preparing segmentation data.

Validated datasets in the project include:

```text
CRACK500
Pothole_Segmentation_YOLOv8
Public Pothole Dataset
```

The dataset pipeline handles the conversion of image/mask data into model-ready tensors and DataLoaders.

---

## `14` — Current Evaluation Snapshot

The implemented evaluation pipeline produced the following recorded results:

| Dataset | Dice / F1 | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| CRACK500 | 0.8684 | 0.7674 | 0.9299 | 0.8145 |
| Public Pothole | 0.7023 | 0.5412 | 0.8082 | 0.6210 |
| YOLOv8 Pothole | 0.4273 | 0.2717 | 0.5330 | 0.3565 |

These values are **project evaluation results**, not claims of universal model performance.

---

## `15` — Runtime Flow

```text
                    USER
                     │
                     ▼
              Upload Road Image
                     │
                     ▼
              Streamlit Application
                     │
                     ▼
               Load Checkpoint
                     │
                     ▼
                 Preprocess
                     │
                     ▼
                U-Net Inference
                     │
                     ▼
               Prediction Mask
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     Engineering    XAI        3D
          │          │          │
          └──────────┼──────────┘
                     ▼
              Decision Summary
                     │
                     ▼
               Report Generator
                     │
              ┌──────┼──────┐
              ▼      ▼      ▼
             JSON   TXT     PDF
```

---

## `16` — Training → Deployment

```text
DATA
 │
 ▼
DATASET PIPELINE
 │
 ▼
TRAINING
 │
 ▼
checkpoints/best.pt
 │
 ├──────────────► EVALUATION
 │
 └──────────────► STREAMLIT
                       │
                       ├── Engineering
                       ├── XAI
                       ├── 3D
                       └── Reports
```

The checkpoint is the hand-off point between model development and application inference.

---

## `17` — Useful Commands

### Run the application

```bash
streamlit run streamlit_app/app.py
```

### Run the dataset validation test

```bash
python dataset_pipeline/test_dataset.py
```

### TensorBoard

```bash
tensorboard --logdir runs/roadxai
```

---

## `18` — Design Principles

```text
MODULAR
   ↓
Each major capability is isolated into a package.

REPRODUCIBLE
   ↓
Seeds, checkpoints, history and deterministic cuDNN settings
are provided by the training pipeline.

MEASURABLE
   ↓
Dice, IoU, precision, recall and inference performance
are explicitly evaluated.

EXPLAINABLE
   ↓
Grad-CAM-family methods expose model attention behavior.

ENGINEERING-AWARE
   ↓
Segmentation results are converted into defect and
road-condition indicators.

INTERACTIVE
   ↓
Streamlit + Plotly provide the inspection interface.

EXPORTABLE
   ↓
Results can be packaged as JSON, TXT and PDF reports.
```

---

## `19` — Repository Map

| Directory | Responsibility |
|---|---|
| `dataset_pipeline/` | Dataset preparation and validation |
| `models/` | Segmentation architecture |
| `training/` | Model optimization and checkpoints |
| `engineering/` | Measurements, severity and road health |
| `evaluation/` | Quantitative model evaluation |
| `explainable_ai/` | Grad-CAM-family XAI |
| `visualization_3d/` | Interactive 3D representation |
| `report_generation/` | Report construction and export |
| `streamlit_app/` | User-facing application |
| `checkpoints/` | Saved model states |
| `reports/` | Generated inspection artifacts |
| `runs/` | TensorBoard logs |

---

## `20` — Scientific Boundaries

RoadXAI should be interpreted as an **AI-assisted inspection and decision-support system**, not as an autonomous civil-engineering certification system.

The most important boundaries are:

1. **Segmentation quality depends on training data and annotations.**
2. **Pixel dimensions are not automatically physical dimensions.**
3. **Calibration is required for real-world measurement and cost estimation.**
4. **Grad-CAM explains model activation behavior, not causality.**
5. **The 3D surface is visual unless validated depth data is available.**
6. **Field inspection remains necessary for engineering decisions.**

---

## `21` — Project Status

### Implemented pipeline

```text
[✓] Dataset Pipeline
[✓] CNN / U-Net Segmentation
[✓] Training Pipeline
[✓] Evaluation
[✓] Engineering Analysis
[✓] Explainable AI
[✓] 3D Visualization
[✓] Report Generation
[✓] Streamlit Application
```

---

## `22` — The RoadXAI Loop

```text
              ┌───────────────────┐
              │   ROAD IMAGE      │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │     DETECT        │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │     MEASURE       │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │     EXPLAIN       │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │   VISUALIZE       │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │      REPORT       │
              └─────────┬─────────┘
                        │
                        └───────────────► INSPECTION DECISION
```

> **RoadXAI — From pixels to an interpretable road-inspection workflow.**

---

## License

This repository is intended as a research/project implementation. Add the project's final license terms here before public distribution.
