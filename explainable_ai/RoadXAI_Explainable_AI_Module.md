# RoadXAI Explainable AI Module

## 1. Explainable AI — One View

```mermaid
flowchart LR
    A[Road Image] --> B[RoadXAI U-Net]
    B --> C[Defect Prediction]

    B --> D[Internal Activations]
    B --> E[Gradients / Scores]

    D --> F[XAI Method]
    E --> F

    F --> G[Explanation Heatmap]
    G --> H[Overlay on Road Image]
```

**Purpose:** show which image regions contribute to the CNN's defect prediction.

---

## 2. XAI Pipeline

```mermaid
flowchart TD
    A[Input Image]
    --> B[Trained CNN]

    B --> C[Target Defect Output]
    B --> D[Target Conv Layer]

    C --> E[XAI Calculation]
    D --> E

    E --> F[Raw Importance Map]
    F --> G[ReLU / Normalize]
    G --> H["Heatmap 0–1"]
    H --> I[Resize to Image]
    I --> J[Color Map]
    J --> K[Overlay]
```

---

## 3. Three Explanation Methods

```mermaid
flowchart TD
    A[RoadXAI U-Net] --> B{Explanation Method}

    B --> C[Grad-CAM]
    B --> D[Grad-CAM++]
    B --> E[Score-CAM]

    C --> F[Heatmap]
    D --> F
    E --> F

    F --> G[Visual Explanation]
```

| Method | Main information used |
|---|---|
| Grad-CAM | Activations + gradients |
| Grad-CAM++ | Higher-order gradient information |
| Score-CAM | Activation maps + model score changes |

---

## 4. Grad-CAM

```mermaid
flowchart TD
    A[Input Image]
    --> B[CNN Forward Pass]

    B --> C[Target Layer Activations]
    B --> D[Target Defect Score]

    D --> E[Backpropagation]
    E --> F[Target Layer Gradients]

    F --> G[Average Gradients Spatially]
    G --> H[Channel Weights]

    C --> I[Weighted Activations]
    H --> I

    I --> J[Sum Channels]
    J --> K[ReLU + Normalize]
    K --> L[Grad-CAM Heatmap]
```

Core idea:

```text
Gradients → channel importance
Activations × importance → explanation
```

The target score is the **spatial mean of the selected output channel**.

---

## 5. Grad-CAM++ 

```mermaid
flowchart TD
    A[Input Image]
    --> B[CNN Forward Pass]

    B --> C[Target Activations]
    B --> D[Target Score]

    D --> E[Gradient Information]
    E --> F[Positive Gradients]
    F --> G[Gradient² + Gradient³]

    G --> H[Alpha Weights]
    H --> I[Channel Weights]

    C --> J[Weighted Activations]
    I --> J

    J --> K[Sum Channels]
    K --> L[ReLU + Normalize]
    L --> M[Grad-CAM++ Heatmap]
```

Grad-CAM++ uses additional gradient information to calculate more detailed activation importance weights.

---

## 6. Score-CAM

```mermaid
flowchart TD
    A[Input Image]
    --> B[CNN]
    B --> C[Target Layer Activations]

    C --> D[Normalize Activation Maps]
    D --> E[Resize Maps to Input]

    E --> F[Use Each Map as Image Mask]
    F --> G[CNN Forward Pass]

    G --> H[Target Score Change]
    H --> I[Channel Weight]

    I --> J[Weighted Activation Maps]
    J --> K[Sum Channels]
    K --> L[Normalize]
    L --> M[Score-CAM Heatmap]
```

Score-CAM uses the model's score response to activation-based masked inputs instead of backpropagated gradients.

---

## 7. Target Class

```mermaid
flowchart LR
    A[U-Net Output]
    --> B["[B, C, H, W]"]

    B --> C["target_class = 0"]
    C --> D[Selected Defect Output]
    D --> E[Spatial Mean]
    E --> F[Scalar Target Score]
```

For the binary RoadXAI segmentation model, `target_class=0` selects the single defect output channel.

---

## 8. Target Layer

```mermaid
flowchart TD
    A[RoadXAI CNN]
    --> B[Conv2d Layers]

    B --> C[Search Conv2d Layers]
    C --> D[Select Last Conv2d]
    D --> E[Default XAI Target Layer]
```

The implementation can also accept a specific convolutional layer.

Layer utilities support:

```text
Get layer by dotted name
List all Conv2d layers
Select final Conv2d layer
```

---

## 9. Activation + Gradient Capture

```mermaid
flowchart LR
    A[Target Conv Layer]
    --> B[Forward Hook]
    B --> C[Activations]

    A --> D[Backward Hook]
    D --> E[Gradients]

    C --> F[XAI Calculation]
    E --> F
```

Hooks are removed after explanation generation.

---

## 10. Heatmap Processing

```mermaid
flowchart LR
    A[Raw CAM]
    --> B[ReLU]
    --> C[Min-Max Normalization]
    --> D["Range 0–1"]
    --> E[Resize to Input]
```

The generated explanation is represented as a normalized spatial heatmap.

---

## 11. Heatmap → Visualization

```mermaid
flowchart LR
    A["Heatmap 0–1"]
    --> B["0–255 uint8"]
    --> C[OpenCV Colormap]
    --> D[Colored Heatmap]

    E[Road Image]
    --> F[Weighted Blend]

    D --> F
    F --> G[XAI Overlay]
```

Default overlay contribution:

```text
alpha = 0.45
```

---

## 12. What the Heatmap Means

```mermaid
flowchart TD
    A[XAI Heatmap]
    --> B{Pixel Importance}

    B -->|Low| C[Less contribution to explained prediction]
    B -->|High| D[Greater contribution to explained prediction]

    D --> E[Highlighted Road Region]
```

The heatmap is an **explanation of model activation/score sensitivity**, not a ground-truth defect mask.

```text
Segmentation Mask → What the model predicts
XAI Heatmap       → Where the model's decision is supported
```

---

## 13. XAI vs Segmentation

```mermaid
flowchart LR
    A[Road Image]
    --> B[CNN]

    B --> C[Segmentation Output]
    B --> D[XAI Explanation]

    C --> E["Defect Mask"]
    D --> F["Importance Heatmap"]

    E --> G[Engineering Analysis]
    F --> H[Model Interpretability]
```

Both use the same CNN prediction, but they answer different questions.

---

## 14. Utility Pipeline

```mermaid
flowchart TD
    A[Input Tensor]
    --> B[Tensor → Image]

    C[Heatmap]
    --> D[Validate]
    --> E[Statistics]

    C --> F[Resize]
    --> G[Colormap]
    --> H[Overlay]

    H --> I[Save PNG]
```

Available utilities include:

```text
Image normalization
Tensor → image conversion
Tensor min-max normalization
Heatmap resizing
Colormap generation
Heatmap blending
Heatmap validation
Heatmap statistics
Explanation metadata
Heatmap / overlay saving
```

---

## 15. Explanation Metadata

```mermaid
flowchart LR
    A[Explanation]
    --> B[Method]
    A --> C[Target Class]
    A --> D[Heatmap Size]
    A --> E[Heatmap Statistics]

    E --> F[Min]
    E --> G[Max]
    E --> H[Mean]
    E --> I[Std]
    E --> J[Active Ratio]
```

The active ratio represents the fraction of heatmap pixels with value `>= 0.5`.

---

## 16. Complete RoadXAI Connection

```mermaid
flowchart LR
    A[Dataset Pipeline]
    --> B[CNN]

    B --> C[Defect Mask]
    B --> D[XAI]

    C --> E[Evaluation]
    C --> F[Engineering Analysis]
    C --> G[3D Visualization]

    D --> H[Grad-CAM]
    D --> I[Grad-CAM++]
    D --> J[Score-CAM]

    H --> K[Heatmap]
    I --> K
    J --> K

    K --> L[Interpretability]
```

---

## 17. Why XAI Matters

```mermaid
flowchart LR
    A[CNN Prediction]
    --> B["Black-box result"]

    C[XAI]
    --> D["Visual explanation"]

    D --> E[Inspect model focus]
    E --> F[Understand prediction]
    F --> G[Research interpretability]
```

XAI adds an explanation layer to RoadXAI rather than changing the segmentation prediction itself.

---

## 18. Research-Paper View

```mermaid
flowchart TD
    A[Road Image]
    --> B[U-Net Segmentation]

    B --> C[Defect Prediction]

    B --> D[Target Layer Activations]
    C --> E[Target Score]

    D --> F[XAI]
    E --> F

    F --> G[Grad-CAM]
    F --> H[Grad-CAM++]
    F --> I[Score-CAM]

    G --> J[Heatmap]
    H --> J
    I --> J

    J --> K[Visual Model Explanation]
```

### Key Technical Points

| Component | Current implementation |
|---|---|
| Model | RoadXAI U-Net segmentation model |
| XAI methods | Grad-CAM, Grad-CAM++, Score-CAM |
| Input | `[B, 3, H, W]` floating-point RGB tensor |
| Target output | Selected segmentation output channel |
| Default target class | `0` |
| Default target layer | Last `Conv2d` layer |
| Grad-CAM | Activation-weighted by spatially averaged gradients |
| Grad-CAM++ | Higher-order gradient-based weighting |
| Score-CAM | Activation masking + score change |
| Heatmap range | 0–1 |
| Heatmap resize | Bilinear interpolation |
| Visualization | OpenCV colormap + weighted overlay |
| Default overlay alpha | 0.45 |
| Output | Heatmap / colored heatmap / overlay |
| Metadata | Method, target class, dimensions, statistics |

> **Scientific limitation:** the generated heatmaps explain the model's internal activation/score behavior for the selected target and layer. They should not be interpreted as ground-truth defect boundaries or as proof that highlighted pixels are causally responsible for the physical defect.
