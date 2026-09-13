# ROADXAI

<div align="center">

# `ROADXAI`

### **See the defect. Measure the defect. Explain the AI. Inspect the road.**

AI-assisted road-surface inspection built around **pixel-level segmentation**, engineering analysis, explainable AI, interactive 3D visualization, and structured reporting.

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

</div>

---

## `01` · THE 10-SECOND VERSION

RoadXAI takes this:

```text
┌──────────────────────┐
│     ROAD IMAGE       │
│                      │
│   cracks / potholes  │
│   damaged surfaces   │
└──────────┬───────────┘
           │
           ▼
```

and turns it into this:

```text
┌──────────────────────────────────────────────────────────┐
│                    ROADXAI INSPECTION                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  DEFECT MASK       ENGINEERING       AI EXPLANATION      │
│  ████████          AREA / SIZE       GRAD-CAM            │
│  ██████            SEVERITY          ATTENTION           │
│                                                          │
│  ROAD HEALTH       3D VIEW           REPORT              │
│  CONDITION         SURFACE           JSON / TXT / PDF    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The system is not just a segmentation model.

It is a complete chain:

```text
IMAGE
  ↓
SEGMENT
  ↓
MEASURE
  ↓
SCORE
  ↓
EXPLAIN
  ↓
VISUALIZE
  ↓
REPORT
  ↓
INSPECT
```

---

# `02` · WHY ROADXAI?

Most computer-vision projects stop here:

```text
Image → Model → Mask
```

RoadXAI continues:

```text
Image
  │
  ▼
┌──────────────┐
│ Segmentation │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Measurement  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Severity     │
└──────┬───────┘
       │
       ├───────────────┐
       ▼               ▼
┌──────────────┐ ┌──────────────┐
│ Explainable  │ │ 3D Surface   │
│ AI           │ │ Visualization│
└──────┬───────┘ └──────┬───────┘
       │                │
       └────────┬───────┘
                ▼
        ┌──────────────┐
        │ Inspection   │
        │ Dashboard    │
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │ JSON / TXT   │
        │ PDF Report   │
        └──────────────┘
```

**The objective is interpretability and inspection workflow, not only prediction.**

---

# `03` · SYSTEM AT A GLANCE

| Layer | What it does | Main output |
|---|---|---|
| Dataset | Loads and prepares image/mask data | DataLoaders |
| Model | Pixel-level road-defect segmentation | Logits / mask |
| Training | Optimizes and validates the model | `best.pt` |
| Evaluation | Quantifies model performance | Dice / IoU / precision / recall |
| Engineering | Converts masks into defect information | Measurements / severity / health |
| XAI | Shows model attention | Heatmaps |
| 3D | Builds an interactive visual surface | 3D inspection view |
| Streamlit | Provides the user interface | Interactive inspection |
| Reports | Packages analysis results | JSON / TXT / PDF |

---

# `04` · THE ARCHITECTURE

```text
                         ┌─────────────────────┐
                         │     ROAD IMAGE      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ DATASET PIPELINE    │
                         │ images + masks      │
                         │ preprocessing       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ U-NET CNN           │
                         │ segmentation        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ DEFECT MASK         │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
     │ ENGINEERING    │    │ EXPLAINABLE AI │    │ 3D VISUALIZER  │
     │                │    │                │    │                │
     │ area           │    │ Grad-CAM       │    │ heightmap      │
     │ length         │    │ Grad-CAM++     │    │ mesh           │
     │ width          │    │ Score-CAM      │    │ surface        │
     │ severity       │    │ heatmap        │    │ modes          │
     │ road health    │    │                │    │                │
     └────────┬───────┘    └────────┬───────┘    └────────┬───────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ STREAMLIT APP       │
                         │ inspection UI       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ REPORT GENERATION   │
                         │ JSON / TXT / PDF    │
                         └─────────────────────┘
```

---

# `05` · THE MODEL

RoadXAI uses a **U-Net-style binary segmentation network**.

```text
INPUT
  │
  ▼
┌────────────┐
│ ENCODER 1  │──────────────┐
└─────┬──────┘              │
      ▼                     │
┌────────────┐              │
│ ENCODER 2  │──────────┐   │
└─────┬──────┘          │   │
      ▼                 │   │
┌────────────┐          │   │
│ ENCODER 3  │──────┐   │   │
└─────┬──────┘      │   │   │
      ▼             │   │   │
┌────────────┐      │   │   │
│ ENCODER 4  │──┐   │   │   │
└─────┬──────┘  │   │   │   │
      ▼         │   │   │   │
┌────────────┐  │   │   │   │
│ BOTTLENECK │  │   │   │   │
└─────┬──────┘  │   │   │   │
      ▼         │   │   │   │
┌────────────┐  │   │   │   │
│ DECODER 4  │◄─┘   │   │   │
└─────┬──────┘      │   │   │
      ▼             │   │   │
┌────────────┐      │   │   │
│ DECODER 3  │◄─────┘   │   │
└─────┬──────┘          │   │
      ▼                 │   │
┌────────────┐          │   │
│ DECODER 2  │◄─────────┘   │
└─────┬──────┘              │
      ▼                     │
┌────────────┐              │
│ DECODER 1  │◄─────────────┘
└─────┬──────┘
      ▼
┌────────────┐
│ 1×1 CONV   │
│ 1 CHANNEL  │
└─────┬──────┘
      ▼
DEFECT LOGITS
      │
      ▼
   SIGMOID
      │
      ▼
BINARY MASK
```

### Prediction path

```text
logits
  ↓
sigmoid
  ↓
probability map
  ↓
threshold = 0.5
  ↓
binary defect mask
```

---

# `06` · TRAINING ENGINE

The training system is built around a repeatable optimization loop:

```text
TRAIN
  │
  ├── forward
  ├── loss
  ├── backward
  ├── gradient clipping
  └── optimizer update
  │
  ▼
VALIDATE
  │
  ├── loss
  ├── Dice
  └── IoU
  │
  ▼
SCHEDULER
  │
  ▼
BEST VALIDATION LOSS?
  │
  ├── YES → best.pt
  │
  └── NO  → early-stop counter
```

Included:

- Automatic device selection
- CUDA mixed precision
- Gradient clipping
- Dice / IoU tracking
- Learning-rate scheduling
- Early stopping
- Best checkpoint saving
- Checkpoint restoration
- TensorBoard logging
- Training history
- Reproducibility controls

Default training configuration:

```text
epochs                  10
threshold               0.5
early stopping          5 epochs
AMP                     CUDA only
gradient max norm       1.0
save best only          True
checkpoint directory    checkpoints/
TensorBoard              runs/roadxai/
```

---

# `07` · WHAT THE MODEL ACTUALLY PRODUCES

A single inference is not treated as just `True / False`.

```text
                     MODEL OUTPUT
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
       Probability      Mask       Confidence
          Map
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
                    DEFECT ANALYSIS
```

The Streamlit application then derives:

```text
Defect percentage
Defect pixels
Defect regions
Area
Length
Width
Centroid
Severity
Road condition
Maintenance priority
Inspection requirement
```

---

# `08` · ENGINEERING LAYER

```text
                 DEFECT MASK
                      │
                      ▼
             ┌─────────────────┐
             │ Defect Regions  │
             └────────┬────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        AREA        LENGTH       WIDTH
          │           │           │
          └───────────┼───────────┘
                      ▼
               SEVERITY SCORE
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
         LOW       MODERATE      HIGH
                                  │
                                  ▼
                              CRITICAL
```

Road health:

```text
DEFECT AREA RATIO
        │
        ▼
ROAD HEALTH SCORE
        │
        ├── Excellent
        ├── Good
        ├── Fair
        ├── Poor
        └── Critical
```

The implementation also supports calibrated physical measurements and repair-cost logic when the required calibration/rate information is available.

---

# `09` · EXPLAINABILITY

RoadXAI includes:

```text
Grad-CAM
Grad-CAM++
Score-CAM
```

Conceptual path:

```text
ROAD IMAGE
    │
    ▼
   CNN
    │
    ▼
TARGET CONVOLUTION LAYER
    │
    ├── Activations
    └── Gradients
            │
            ▼
       CAM METHOD
            │
            ▼
      HEATMAP [0, 1]
            │
            ▼
       IMAGE OVERLAY
```

The application currently uses **Grad-CAM** with the default target layer.

The heatmap answers:

> **Where is the model's internal activation concentrated?**

It does not answer:

> **What is the ground-truth cause of the road damage?**

---

# `10` · 3D INSPECTION

RoadXAI converts segmentation information into a visual surface representation.

```text
DEFECT MASK
    │
    ▼
CONNECTED COMPONENTS
    │
    ▼
DISTANCE TRANSFORM
    │
    ▼
AREA / DEPTH FACTORS
    │
    ▼
SMOOTHING
    │
    ▼
HEIGHTMAP
    │
    ▼
MESH / SURFACE
    │
    ▼
INTERACTIVE 3D VIEW
```

Available application modes include:

```text
Inspection
Road Surface
Severity
AI Confidence
AI Attention
```

Controls include:

```text
Visual depth
Markers
Boundaries
```

> The 3D vertical dimension is visualization geometry, not automatically measured physical depth.

---

# `11` · THE STREAMLIT EXPERIENCE

The application is divided into two main inspection experiences.

```text
┌─────────────────────────────────────────────┐
│                 ROADXAI                      │
├─────────────────────────────────────────────┤
│                                             │
│  UPLOAD ROAD IMAGE                          │
│            │                                │
│            ▼                                │
│       AI INFERENCE                          │
│            │                                │
│     ┌──────┴──────┐                         │
│     ▼             ▼                         │
│ ROAD INSPECTION  ADVANCED ANALYSIS          │
│                                             │
└─────────────────────────────────────────────┘
```

### Road Inspection

Designed for the primary inspection result:

```text
Original Image
      +
Numbered Defect Overlay
      +
Summary Metrics
      +
Road Condition
      +
Inspection Requirement
      +
Defect Measurements
      +
Explanation
```

### Advanced Analysis

Provides:

```text
Inference summary
Probability maps
Engineering summary
Severity distribution
Grad-CAM
Interactive 3D
Road decision summary
Report generation
```

---

# `12` · REPORT ENGINE

RoadXAI packages the analysis into structured artifacts.

```text
                  ANALYSIS
                     │
                     ▼
              REPORT MODEL
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
        JSON        TXT        PDF
```

Reports can contain:

```text
Inspection summary
Detection results
Measurements
Severity
Road condition
Cost information when available
Recommendations
XAI information
3D information
```

Output directory:

```text
reports/
```

---

# `13` · DATASETS

The project dataset pipeline has been validated with:

| Dataset | Role |
|---|---|
| CRACK500 | Crack segmentation |
| Pothole_Segmentation_YOLOv8 | Pothole segmentation |
| Public Pothole Dataset | Pothole segmentation |

The dataset layer converts image/mask data into model-ready tensors and DataLoaders.

---

# `14` · EVALUATION SNAPSHOT

Recorded project evaluation results:

| Dataset | Accuracy | Precision | Recall | Dice / F1 | IoU |
|---|---:|---:|---:|---:|---:|
| CRACK500 | 0.9850 | 0.9299 | 0.8145 | 0.8684 | 0.7674 |
| Public Pothole | 0.8823 | 0.8082 | 0.6210 | 0.7023 | 0.5412 |
| YOLOv8 Pothole | 0.8453 | 0.5330 | 0.3565 | 0.4273 | 0.2717 |

Inference-time measurements were also collected during evaluation.

These numbers describe the recorded project experiments; they are not universal performance guarantees.

---

# `15` · PROJECT STRUCTURE

```text
RoadXAI/
│
├── dataset_pipeline/
│   └── Dataset loading, preprocessing, validation
│
├── models/
│   └── CNN / U-Net segmentation
│
├── training/
│   └── Training, validation, checkpoints, TensorBoard
│
├── engineering/
│   └── Measurements, severity, health, cost
│
├── evaluation/
│   └── Model evaluation
│
├── explainable_ai/
│   └── Grad-CAM family
│
├── visualization_3d/
│   └── Interactive 3D visualization
│
├── report_generation/
│   └── JSON / TXT / PDF reports
│
├── streamlit_app/
│   └── User-facing inspection application
│
├── checkpoints/
│   └── Trained model checkpoints
│
├── reports/
│   └── Generated reports
│
├── runs/
│   └── TensorBoard logs
│
├── requirements.txt
└── .gitignore
```

---

# `16` · QUICK START

## Create the environment

```bash
python -m venv .venv
```

## Activate it

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Launch RoadXAI

```bash
streamlit run streamlit_app/app.py
```

## TensorBoard

```bash
tensorboard --logdir runs/roadxai
```

---

# `17` · THE COMPLETE DATA JOURNEY

This is the central idea of the repository:

```text
┌─────────────────────────────────────────────────────────┐
│                    RAW ROAD IMAGE                        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    PREPROCESSING                         │
│                resize → tensor → batch                  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  U-NET SEGMENTATION                      │
│              image → defect probability                 │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    DEFECT MASK                           │
│                 pixel-level localization                 │
└───────────────┬───────────────┬───────────────┬─────────┘
                │               │               │
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ ENGINEERING  │ │     XAI      │ │      3D      │
        │              │ │              │ │              │
        │ measure      │ │ explain      │ │ visualize    │
        │ score        │ │ attention    │ │ surface      │
        └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
               │                │                │
               └────────────────┼────────────────┘
                                │
                                ▼
                    ┌────────────────────┐
                    │ STREAMLIT INSPECTOR│
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │ REPORT GENERATION  │
                    └──────────┬─────────┘
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
                JSON          TXT          PDF
```

---

# `18` · WHAT MAKES THE PIPELINE DIFFERENT

```text
                  ┌─────────────────────┐
                  │   PIXEL-LEVEL AI    │
                  └──────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
          ENGINEERING       XAI            3D
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                     HUMAN INSPECTION
```

The project connects **computer vision**, **engineering-oriented analysis**, **model interpretability**, and **interactive inspection** rather than exposing the neural network as an isolated prediction endpoint.

---

# `19` · ENGINEERING SAFETY BOUNDARIES

RoadXAI is an **AI-assisted screening and decision-support system**.

It should not be interpreted as autonomous engineering certification.

### Important boundaries

- Segmentation quality depends on training data and annotation quality.
- Dice and IoU measure segmentation overlap, not structural safety.
- Pixel dimensions are not automatically physical dimensions.
- Calibration is required for real-world measurement and cost estimation.
- Grad-CAM visualizations represent model activation behavior, not causal proof.
- The 3D representation is visualization geometry unless validated depth is supplied.
- Field inspection remains necessary for engineering decisions.

---

# `20` · CURRENT IMPLEMENTATION STATUS

```text
DATASET PIPELINE             [████████████████████] READY
CNN / U-NET                 [████████████████████] READY
TRAINING PIPELINE           [████████████████████] READY
EVALUATION                  [████████████████████] READY
ENGINEERING ANALYSIS        [████████████████████] READY
EXPLAINABLE AI              [████████████████████] READY
3D VISUALIZATION            [████████████████████] READY
REPORT GENERATION            [████████████████████] READY
STREAMLIT APPLICATION        [████████████████████] READY
```

---

# `21` · ONE COMMAND TO SEE IT

After installation:

```bash
streamlit run streamlit_app/app.py
```

Then the RoadXAI flow becomes:

```text
UPLOAD
  ↓
PREDICT
  ↓
INSPECT
  ↓
EXPLAIN
  ↓
ANALYZE
  ↓
VIEW IN 3D
  ↓
EXPORT
```

---

# `22` · RESEARCH VIEW

```text
                COMPUTER VISION
                       │
                       ▼
              SEMANTIC SEGMENTATION
                       │
                       ▼
              PIXEL-LEVEL DEFECT MAP
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
        ENGINEERING   XAI        3D
             │         │         │
             └─────────┼─────────┘
                       ▼
              DECISION SUPPORT
                       │
                       ▼
                HUMAN INSPECTION
```

RoadXAI therefore forms a single research pipeline from **raw visual evidence to interpretable inspection artifacts**.

---

# `23` · ROADXAI IN ONE LINE

> **RoadXAI transforms road images into segmentation-driven, explainable, engineering-aware inspection results.**

---

## License

This repository is intended as a research/project implementation. Add the final license terms here before public distribution.
