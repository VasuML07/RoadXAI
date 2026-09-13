# RoadXAI — CNN Segmentation Module

## 1. Module in One Diagram

```mermaid
flowchart LR
    A[Road Image] --> B[RoadXAI U-Net CNN]
    B --> C[Pixel-wise Defect Prediction]
    C --> D[Binary Defect Mask]
    D --> E[Engineering Analysis]
    D --> F[3D Visualization]
    C --> G[XAI / Explainability]
```

**Goal:** detect and segment road defects at the pixel level.

---

## 2. CNN Architecture

```mermaid
flowchart TD
    A[RGB Image<br/>3 Channels] --> B[Encoder 1<br/>32]
    B --> C[Encoder 2<br/>64]
    C --> D[Encoder 3<br/>128]
    D --> E[Encoder 4<br/>256]
    E --> F[Bottleneck<br/>512]

    F --> G[Decoder 4<br/>256]
    G --> H[Decoder 3<br/>128]
    H --> I[Decoder 2<br/>64]
    I --> J[Decoder 1<br/>32]
    J --> K[1×1 Conv]
    K --> L[Defect Logits<br/>1 Channel]
```

---

## 3. U-Net Structure

```mermaid
flowchart LR
    A[Input] --> B[Encoder]
    B --> C[Bottleneck]
    C --> D[Decoder]
    D --> E[Output]

    B -. Skip Connections .-> D
```

### Why skip connections?

```text
Encoder
  ↓
captures deeper / contextual features

Skip connection
  ↓
preserves fine spatial details

Decoder
  ↓
reconstructs precise defect boundaries
```

This is important because road defects can be small, thin, irregular, or have unclear boundaries.

---

## 4. Encoder

```mermaid
flowchart TD
    A[Feature Map] --> B[Conv 3×3]
    B --> C[BatchNorm]
    C --> D[ReLU]
    D --> E[Dropout]
    E --> F[Conv 3×3]
    F --> G[BatchNorm]
    G --> H[ReLU]
    H --> I[MaxPool 2×2]
```

Each encoder stage:

```text
Convolution
     ↓
Feature extraction
     ↓
Max pooling
     ↓
Smaller spatial size
     +
More feature channels
```

Channels progress:

```text
32 → 64 → 128 → 256 → 512
```

---

## 5. Bottleneck

```mermaid
flowchart LR
    A[Deep Encoder Features] --> B[Conv Block]
    B --> C[512 Feature Representation]
    C --> D[Decoder]
```

The bottleneck provides the deepest feature representation before reconstruction.

---

## 6. Decoder

```mermaid
flowchart TD
    A[Deep Feature Map] --> B[Transposed Convolution]
    B --> C[Upsampling]
    C --> D[Concatenate Skip Features]
    D --> E[Conv Block]
    E --> F[Reconstructed Feature Map]
```

```text
Low resolution
      ↓
Upsample
      ↓
Combine encoder detail
      ↓
Refine
      ↓
Higher resolution
```

---

## 7. Final Prediction

```mermaid
flowchart LR
    A[Decoder Features] --> B[1×1 Convolution]
    B --> C[1 Output Channel]
    C --> D[Pixel-wise Logits]
    D --> E[Sigmoid]
    E --> F[Pixel Probability]
    F --> G[Threshold 0.5]
    G --> H[Binary Mask]
```

Output:

```text
0 → background / normal road
1 → detected defect
```

The model internally returns **logits**. Sigmoid is applied for probability/inference.

---

## 8. Training Loss

```mermaid
flowchart LR
    A[Predicted Logits] --> B[BCE Loss]
    A --> C[Dice Loss]
    D[Ground Truth Mask] --> B
    D --> C
    B --> E[0.5 × BCE]
    C --> F[0.5 × Dice]
    E --> G[Combined Loss]
    F --> G
```

### Why two losses?

```text
BCE
 ↓
pixel-level classification

Dice
 ↓
segmentation overlap

BCE + Dice
 ↓
better segmentation supervision
```

The current implementation uses equal BCE and Dice weights by default.

---

## 9. Training Pipeline

```mermaid
flowchart TD
    A[Training Image + Ground Truth Mask]
    A --> B[Forward Pass]
    B --> C[Predicted Logits]
    C --> D[BCEDice Loss]
    D --> E[Backpropagation]
    E --> F[AdamW]
    F --> G[Updated CNN]
    G --> B

    G --> H[Validation]
    H --> I[Validation Loss]
    I --> J[ReduceLROnPlateau]
    J --> F
```

---

## 10. Evaluation

```mermaid
flowchart LR
    A[Prediction] --> B[Threshold]
    B --> C[Binary Mask]
    C --> D[IoU]
    C --> E[Dice]
    F[Ground Truth] --> D
    F --> E
```

### Metrics

```text
IoU  = Intersection / Union

Dice = 2 × Intersection
       ───────────────────
       Prediction + Target
```

Both measure how closely the predicted defect region overlaps the ground-truth region.

---

## 11. Model → RoadXAI Modules

```mermaid
flowchart LR
    A[Road Image] --> B[CNN]
    B --> C[Defect Logits]
    C --> D[Probability Map]
    C --> E[Binary Segmentation Mask]

    E --> F[Engineering Analysis]
    E --> G[3D Visualization]
    D --> H[AI Confidence Visualization]
    B --> I[XAI]
```

The CNN is therefore the **core defect-segmentation stage** feeding downstream RoadXAI components.

---

## 12. Training Utilities

```mermaid
flowchart TD
    A[CNN Model] --> B[Device Selection]
    B --> C[CPU / CUDA]

    A --> D[Optimizer]
    D --> E[AdamW]

    E --> F[Learning Rate Scheduler]
    F --> G[ReduceLROnPlateau]

    A --> H[Checkpoint]
    H --> I[Save / Load]

    A --> J[Parameter Statistics]
```

The implementation also validates image/mask tensors and supports checkpointing of model, optimizer, scheduler, epoch, and loss.

---

## 13. Research-Paper View

```mermaid
flowchart LR
    A[RGB Road Image] --> B[U-Net CNN]
    B --> C[Hierarchical Feature Extraction]
    C --> D[Skip-Connected Reconstruction]
    D --> E[Pixel-wise Defect Logits]
    E --> F[Binary Defect Mask]
    F --> G[Road Damage Analysis]
```

### Core contribution

> **A U-Net-style CNN performs pixel-level road-defect segmentation, producing masks that drive the downstream engineering, explainability, and 3D visualization stages.**

---

## 14. Key Architecture Summary

```text
Input          : 3-channel RGB image
Architecture   : U-Net style CNN
Encoder        : 32 → 64 → 128 → 256
Bottleneck     : 512
Decoder        : 256 → 128 → 64 → 32
Output         : 1-channel defect logits
Activation     : Sigmoid for probability
Loss           : 0.5 BCE + 0.5 Dice
Optimizer      : AdamW
Scheduler      : ReduceLROnPlateau
Metrics        : IoU + Dice
```

## 15. Important Interpretation

```text
RGB Image
   ↓
CNN prediction
   ↓
Probability / Logits
   ↓
Threshold
   ↓
Defect Mask
   ↓
┌──────────────┬──────────────┐
↓              ↓              ↓
Engineering   3D View        XAI
Analysis      Visualization   Analysis
```

The CNN identifies **where the defect is**.  
The downstream RoadXAI modules use that result to determine and visualize **what the defect means**.
