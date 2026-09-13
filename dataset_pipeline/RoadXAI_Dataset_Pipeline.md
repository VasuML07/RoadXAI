# RoadXAI Dataset Pipeline

## 1. Dataset Pipeline — One View

```mermaid
flowchart LR
    A[Road Datasets] --> B[Find Images]
    B --> C[Find Annotation]
    C --> D{Annotation Type}
    D -->|Mask PNG| E[Load Mask]
    D -->|YOLO TXT| F[Convert to Pixel Mask]
    E --> G[Resize]
    F --> G
    G --> H[Augmentation]
    H --> I[Normalize]
    I --> J[PyTorch Tensors]
    J --> K[DataLoader]
    K --> L[CNN Segmentation Model]
```

**Purpose:** convert different road-defect dataset formats into a common input format for the CNN.

---

## 2. Dataset → Model Interface

```mermaid
flowchart LR
    A[Dataset] --> B["Image Tensor
[B, 3, 512, 512]"]
    A --> C["Mask Tensor
[B, 1, 512, 512]"]

    B --> D[CNN]
    C --> E[Loss / Evaluation]
```

The pipeline standardizes the data to:

- **Image:** `float32`, 3-channel RGB
- **Mask:** `float32`, 1-channel binary
- **Size:** `512 × 512`

---

## 3. Dataset Discovery

```mermaid
flowchart TD
    A[Dataset Root] --> B[Recursive Image Search]
    B --> C[".jpg / .jpeg / .png
.bmp / .webp"]
    C --> D[Load Image]

    D --> E{Find Annotation}
    E -->|Matching PNG| F[Segmentation Mask]
    E -->|YOLO TXT| G[YOLO Annotation]
    E -->|None| H[Reject Sample]

    F --> I[Valid Sample]
    G --> I
```

The pipeline searches through folders and keeps only images with a usable annotation.

---

## 4. Annotation Handling

```mermaid
flowchart TD
    A[Image] --> B{Annotation}

    B -->|PNG Mask| C[Read Grayscale Mask]
    B -->|YOLO TXT| D[Read YOLO Labels]

    D --> E{Label Format}
    E -->|Polygon| F[Scale Coordinates]
    E -->|Bounding Box| G[Convert Box to Pixels]

    F --> H[Fill Polygon]
    G --> I[Fill Rectangle]

    C --> J[Pixel Mask]
    H --> J
    I --> J
```

YOLO annotations are converted into the same pixel-mask representation used by the segmentation model.

---

## 5. Train / Validation Split

```mermaid
flowchart LR
    A[All Valid Samples] --> B{Existing train/val folders?}

    B -->|Yes| C[Use Existing Splits]
    B -->|No| D[Shuffle with Seed 42]

    D --> E["20% Validation"]
    D --> F["80% Training"]

    C --> G[Train Dataset]
    C --> H[Validation Dataset]

    E --> H
    F --> G
```

If predefined `train` and `val` folders exist, they are used directly.

Otherwise, the dataset is shuffled with a fixed seed and split using `val_ratio=0.2`.

---

## 6. Image Preprocessing

```mermaid
flowchart LR
    A[OpenCV Image] --> B[BGR → RGB]
    B --> C["Resize
512 × 512"]
    C --> D["float32"]
    D --> E["Divide by 255"]
    E --> F["Range 0–1"]
    F --> G["H × W × 3"]
    G --> H["3 × H × W"]
```

---

## 7. Mask Preprocessing

```mermaid
flowchart LR
    A[Raw Mask] --> B["Resize
512 × 512"]
    B --> C["Nearest-neighbor"]
    C --> D["Threshold > 127"]
    D --> E["0 / 1"]
    E --> F["Add Channel Dimension"]
    F --> G["1 × 512 × 512"]
```

Nearest-neighbor interpolation preserves the discrete mask labels during resizing.

---

## 8. Training Augmentation

```mermaid
flowchart TD
    A[Training Image + Mask]
    A --> B{Horizontal Flip}
    B -->|50% probability| C[Flip Image + Mask]
    B -->|No| D[Keep]

    C --> E{Vertical Flip}
    D --> E

    E -->|20% probability| F[Flip Image + Mask]
    E -->|No| G[Keep]

    F --> H[Training Sample]
    G --> H
```

Validation data uses **no augmentation**.

---

## 9. Tensor Conversion

```mermaid
flowchart LR
    A["RGB Image
H × W × 3"] --> B["PyTorch Tensor"]
    B --> C["Permute"]
    C --> D["3 × 512 × 512"]

    E["Binary Mask
H × W"] --> F["PyTorch Tensor"]
    F --> G["Unsqueeze"]
    G --> H["1 × 512 × 512"]
```

---

## 10. DataLoader

```mermaid
flowchart LR
    A[RoadDataset] --> B[DataLoader]

    B --> C["Training Loader"]
    B --> D["Validation Loader"]

    C --> E["shuffle=True
augmentation=True"]
    D --> F["shuffle=False
augmentation=False"]

    E --> G[CNN Training]
    F --> H[CNN Validation]
```

Default configuration:

```text
image_size = 512
batch_size = 8
num_workers = 0
val_ratio = 0.2
seed = 42
```

---

## 11. Complete Internal Workflow

```mermaid
flowchart TD
    A[Dataset Root]
    --> B[Recursive Image Discovery]

    B --> C[Find PNG Mask]
    C --> D{Mask Found?}

    D -->|Yes| E[Use Mask]
    D -->|No| F[Find YOLO TXT]

    F --> G{YOLO Label Found?}
    G -->|Yes| H[Generate Pixel Mask]
    G -->|No| I[Discard Sample]

    E --> J[Train / Validation Split]
    H --> J

    J --> K[Read Image]
    K --> L[BGR → RGB]

    L --> M[Resize Image + Mask]
    M --> N[Training Augmentation]

    N --> O[Normalize Image]
    O --> P[Binary Mask]

    P --> Q[PyTorch Tensors]
    Q --> R[DataLoader]
    R --> S[CNN]
```

---

## 12. Dataset Pipeline → RoadXAI

```mermaid
flowchart LR
    A[CRACK500] --> D[Dataset Pipeline]
    B[Pothole Dataset] --> D
    C[Other Supported Dataset] --> D

    D --> E[Common RGB + Binary Mask Format]

    E --> F[CNN Segmentation]
    F --> G[Engineering Analysis]
    F --> H[3D Visualization]
    F --> I[XAI]
```

The dataset pipeline is the **data standardization layer** connecting heterogeneous datasets to the rest of RoadXAI.

---

## 13. Dataset Verification

```mermaid
flowchart TD
    A[Dataset] --> B[Create Train/Val Loaders]
    B --> C[Read First Training Batch]

    C --> D{Validation Checks}

    D --> E["Image: 4D"]
    D --> F["Mask: 4D"]
    D --> G["Image: 3 × 512 × 512"]
    D --> H["Mask: 1 × 512 × 512"]
    D --> I["float32"]
    D --> J["Finite Values"]
    D --> K["Mask ∈ {0,1}"]

    E --> L[PASS / FAIL]
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L
```

`test_dataset.py` verifies the loader output before model training.

---

## 14. Why This Module Matters

```mermaid
flowchart LR
    A[Different Dataset Formats]
    --> B[One Standard Pipeline]
    --> C[Consistent Model Input]
    --> D[Reliable Training]
    --> E[Comparable Evaluation]
```

**Main contribution:** it makes different road-defect datasets usable by the same segmentation pipeline.

---

## 15. Research-Paper View

```mermaid
flowchart TD
    A[Raw Road-Defect Datasets]
    --> B[Annotation Parsing]
    --> C[Preprocessing]
    --> D[Augmentation]
    --> E[Tensor Standardization]
    --> F[Train / Validation DataLoaders]
    --> G[U-Net Segmentation]
```

### Key Technical Points

| Component | Current implementation |
|---|---|
| Image formats | JPG, JPEG, PNG, BMP, WEBP |
| Annotation | PNG mask or YOLO TXT |
| YOLO polygons | Converted to pixel masks |
| YOLO boxes | Converted to filled pixel masks |
| Image format | RGB |
| Image range | 0–1 |
| Image size | 512 × 512 |
| Mask | Binary, 1 channel |
| Training augmentation | Horizontal + vertical flips |
| Validation augmentation | None |
| Default validation ratio | 20% |
| Default batch size | 8 |
| Framework | PyTorch |

> **Important:** the pipeline standardizes annotations for segmentation; YOLO bounding boxes become filled rectangular masks, so they do not provide the same boundary precision as true pixel-level segmentation masks.
