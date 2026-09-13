# RoadXAI Evaluation Module

## 1. Evaluation — One View

```mermaid
flowchart LR
    A[Trained CNN] --> B[Test / Validation Data]
    B --> C[Model Inference]
    C --> D[Segmentation Logits]
    D --> E[Sigmoid + Threshold]
    E --> F[Binary Prediction]
    F --> G[Compare with Ground Truth]
    G --> H[Evaluation Metrics]
    C --> I[Inference Time]
```

**Purpose:** measure segmentation accuracy and inference performance of the RoadXAI CNN.

---

## 2. Evaluation Outputs

```mermaid
flowchart TD
    A[Predictions + Ground Truth]
    --> B[Confusion Matrix]

    B --> C[Accuracy]
    B --> D[Precision]
    B --> E[Recall]
    B --> F[F1]
    B --> G[IoU]
    B --> H[Dice]

    I[Model Inference]
    --> J[Inference Time]

    C --> K[Evaluation Result]
    D --> K
    E --> K
    F --> K
    G --> K
    H --> K
    J --> K
```

---

## 3. Complete Internal Workflow

```mermaid
flowchart TD
    A[Evaluation DataLoader]
    --> B[Prepare Batch]

    B --> C[Move Images + Masks to Device]
    C --> D[CNN Inference]
    D --> E[Logits]

    E --> F[Sigmoid]
    F --> G["Threshold = 0.5"]
    G --> H[Binary Prediction]

    H --> I[Compare with Target]
    I --> J[TP / TN / FP / FN]

    J --> K[Merge Batch Confusion Matrices]
    K --> L[Dataset-Level Metrics]

    D --> M[Measure Inference Time]

    L --> N[EvaluationResult]
    M --> N
```

---

## 4. Prediction Conversion

```mermaid
flowchart LR
    A[CNN Logits]
    --> B[Sigmoid]
    --> C[Pixel Probabilities]
    --> D["Threshold ≥ 0.5"]
    --> E[Binary Mask]
```

```text
Probability < 0.5 → Background
Probability ≥ 0.5 → Defect
```

The threshold is configurable but defaults to `0.5`.

---

## 5. Pixel-Level Confusion Matrix

```mermaid
flowchart TD
    A[Predicted Pixel] --> B{Compare with Ground Truth}

    B -->|Pred=1, True=1| C[True Positive]
    B -->|Pred=0, True=0| D[True Negative]
    B -->|Pred=1, True=0| E[False Positive]
    B -->|Pred=0, True=1| F[False Negative]
```

```mermaid
flowchart LR
    A[TP] --> B[Correct Defect Pixels]
    C[TN] --> D[Correct Background Pixels]
    E[FP] --> F[False Defect Pixels]
    G[FN] --> H[Missed Defect Pixels]
```

All segmentation metrics are derived from these pixel-level counts.

---

## 6. Accuracy

```mermaid
flowchart LR
    A[TP + TN] --> C["Accuracy
(TP + TN) / (TP + TN + FP + FN)"]
    B[All Pixels] --> C
```

Measures the fraction of correctly classified pixels.

---

## 7. Precision

```mermaid
flowchart LR
    A[TP] --> C["Precision
TP / (TP + FP)"]
    B[FP] --> C
```

Measures how many predicted defect pixels are actually defects.

---

## 8. Recall

```mermaid
flowchart LR
    A[TP] --> C["Recall
TP / (TP + FN)"]
    B[FN] --> C
```

Measures how many actual defect pixels are detected.

---

## 9. F1 Score

```mermaid
flowchart LR
    A[Precision] --> C["F1
2 × Precision × Recall / (Precision + Recall)"]
    B[Recall] --> C
```

F1 combines precision and recall into one score.

---

## 10. IoU

```mermaid
flowchart LR
    A[TP] --> C["IoU
TP / (TP + FP + FN)"]
    B[FP] --> C
    D[FN] --> C
```

IoU measures overlap between predicted and ground-truth defect regions.

---

## 11. Dice

```mermaid
flowchart LR
    A[2 × TP] --> C["Dice
2TP / (2TP + FP + FN)"]
    B[FP] --> C
    D[FN] --> C
```

Dice is another overlap measure and is equivalent to F1 for this binary pixel formulation.

---

## 12. Batch → Complete Dataset

```mermaid
flowchart TD
    A[Batch 1] --> B[Confusion Matrix 1]
    C[Batch 2] --> D[Confusion Matrix 2]
    E[Batch N] --> F[Confusion Matrix N]

    B --> G[Merge Counts]
    D --> G
    F --> G

    G --> H[Total Dataset Confusion Matrix]
    H --> I[Final Metrics]
```

Metrics are calculated from the accumulated confusion-matrix counts over the complete evaluation dataset.

---

## 13. Inference Performance

```mermaid
flowchart LR
    A[Input Batch]
    --> B[Warm-up / Model Ready]
    --> C[Start Timer]
    --> D[CNN Forward Pass]
    --> E[Stop Timer]
    --> F[Batch Latency]

    F --> G[Average per Sample]
```

```text
Reported:
Average inference time
Minimum inference time
Maximum inference time
```

The main `evaluate_model()` path reports average inference time per sample and excludes data-loading time.

---

## 14. Evaluation Pipeline

```mermaid
flowchart TD
    A[Trained Model]
    --> B[model.eval()]

    B --> C[Evaluation DataLoader]
    C --> D[Batch Preparation]
    D --> E[Forward Pass]
    E --> F[Logits]

    F --> G[Binary Predictions]
    G --> H[Confusion Matrix]

    H --> I[Accuracy]
    H --> J[Precision]
    H --> K[Recall]
    H --> L[F1]
    H --> M[IoU]
    H --> N[Dice]

    E --> O[Inference Latency]

    I --> P[EvaluationResult]
    J --> P
    K --> P
    L --> P
    M --> P
    N --> P
    O --> P
```

---

## 15. Model Comparison

```mermaid
flowchart LR
    A[Model A Metrics] --> D[Select Metric]
    B[Model B Metrics] --> D
    C[Model C Metrics] --> D

    D --> E[Sort Descending]
    E --> F[Ranked Model Comparison]
```

The comparison utility can rank models using a selected metric, with `f1` as the default.

---

## 16. Evaluation Results

```mermaid
flowchart TD
    A[EvaluationResult]
    --> B[Accuracy]
    A --> C[Precision]
    A --> D[Recall]
    A --> E[F1]
    A --> F[IoU]
    A --> G[Dice]
    A --> H[Inference Time]
    A --> I[Sample Count]
    A --> J[Confusion Matrix]
```

Results can be converted to a dictionary and saved as formatted JSON.

---

## 17. Prediction Storage

```mermaid
flowchart LR
    A[Prediction Tensor]
    --> B[Move to CPU]
    --> C[PyTorch File]

    C --> D[Load Predictions]
    D --> E[Tensor]
```

Prediction tensors can be saved and loaded independently for later analysis.

---

## 18. RoadXAI Pipeline Connection

```mermaid
flowchart LR
    A[Dataset Pipeline]
    --> B[CNN Training]
    --> C[Evaluation]

    C --> D[Segmentation Metrics]
    C --> E[Inference Performance]

    B --> F[Trained Model]
    F --> C

    C --> G[Engineering Analysis]
    C --> H[3D Visualization]
    C --> I[XAI]
```

Evaluation provides the quantitative evidence that the segmentation model is performing correctly before its predictions are used by downstream modules.

---

## 19. Why This Module Matters

```mermaid
flowchart LR
    A[Predicted Mask]
    --> B[Quantitative Evaluation]
    --> C[Model Performance]
    --> D[Model Validation]
    --> E[Research Results]
```

Without evaluation:

```text
CNN → Prediction
```

With evaluation:

```text
CNN
 ↓
Prediction
 ↓
Ground Truth Comparison
 ↓
Confusion Matrix
 ↓
Accuracy / Precision / Recall
F1 / IoU / Dice
 ↓
Inference Performance
```

---

## 20. Research-Paper View

```mermaid
flowchart TD
    A[Trained Segmentation Model]
    --> B[Evaluation Dataset]
    --> C[Pixel-Level Prediction]
    --> D[Confusion Matrix]
    --> E[Segmentation Metrics]
    --> F[Inference-Time Analysis]
    --> G[Model Performance Assessment]
```

### Key Technical Points

| Component | Current implementation |
|---|---|
| Task | Binary road-defect segmentation |
| Evaluation level | Pixel level |
| Prediction conversion | Sigmoid + threshold |
| Default threshold | 0.5 |
| Confusion matrix | TP / TN / FP / FN |
| Metrics | Accuracy, Precision, Recall, F1, IoU, Dice |
| Inference metric | Average time per sample |
| Timing | Excludes data-loading time |
| Model mode | `eval()` |
| Gradient calculation | Disabled |
| Model comparison | Sort by selected metric |
| Result storage | JSON |
| Prediction storage | PyTorch tensor file |

> **Interpretation:** IoU and Dice are especially useful for evaluating segmentation overlap, while precision and recall show false-positive and missed-defect behavior.
