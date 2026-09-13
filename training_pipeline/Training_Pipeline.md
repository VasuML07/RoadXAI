# RoadXAI Training Pipeline


## 1. One View

```mermaid
flowchart LR
    A[Training DataLoader] --> B[Batch Preparation]
    B --> C[Move to Device]
    C --> D[Model Forward Pass]
    D --> E[Segmentation Loss]
    E --> F[Backpropagation]
    F --> G[Gradient Clipping]
    G --> H[Optimizer Step]
    H --> I[Dice + IoU]
    I --> J[Epoch Results]

    K[Validation DataLoader] --> L[Validation Batch]
    L --> M[Model Evaluation]
    M --> N[Validation Loss]
    M --> O[Validation Dice + IoU]
    N --> P[Validation Result]
    O --> P

    J --> Q[Training History]
    P --> Q
    Q --> R[Scheduler]
    Q --> S[Best Checkpoint]
    Q --> T[Early Stopping]
    Q --> U[TensorBoard]
```

The training pipeline trains the RoadXAI CNN segmentation model, validates it after every epoch, tracks loss/Dice/IoU, updates the learning-rate scheduler, saves checkpoints, logs to TensorBoard, and optionally stops early when validation loss stops improving.

---

## 2. Complete Training Workflow

```mermaid
flowchart TD
    A[fit()] --> B[Validate Configuration]
    B --> C[Select CUDA or CPU]
    C --> D[Move Model to Device]
    D --> E[Create Checkpoint Directory]
    E --> F[Create TensorBoard SummaryWriter]
    F --> G[Enable AMP on CUDA]
    G --> H[Initialize TrainingHistory]

    H --> I[Epoch Loop]

    I --> J[train_one_epoch()]
    J --> K[validate_one_epoch()]
    K --> L[Read Learning Rate]
    L --> M[Step Scheduler]

    M --> N[Append Epoch Metrics]
    N --> O[TensorBoard Logging]
    O --> P{Validation Loss Improved?}

    P -->|Yes| Q[Update Best Loss/Epoch]
    Q --> R[Save best.pt]
    P -->|No| S[Increment No-Improvement Counter]

    R --> T{Early Stopping?}
    S --> T

    T -->|No| I
    T -->|Yes| U[Stop Training]

    I -->|All Epochs Finished| V[Close TensorBoard Writer]
    U --> V
    V --> W[Return TrainingHistory]
```

---

## 3. Configuration Validation

```mermaid
flowchart TD
    A[fit()] --> B{epochs > 0?}
    B -->|No| C[Raise ValueError]
    B -->|Yes| D{threshold between 0 and 1?}
    D -->|No| C
    D -->|Yes| E{patience valid?}
    E -->|No| C
    E -->|Yes| F[Select Device]
    F --> G[Continue Training]
```

The pipeline validates:
- `epochs > 0`
- `0 < threshold < 1`
- `early_stopping_patience` is non-negative when supplied

The device is automatically selected as CUDA when available, otherwise CPU.

---

## 4. Batch Preparation

```mermaid
flowchart TD
    A[DataLoader Batch] --> B{Batch Type}
    B -->|tuple/list| C[images = batch 0<br/>masks = batch 1]
    B -->|dict| D{Supported Keys}
    D -->|image/mask| E[Extract image + mask]
    D -->|images/masks| F[Extract images + masks]
    D -->|Other| G[Raise KeyError]

    C --> H[Convert to Tensor if Needed]
    E --> H
    F --> H

    H --> I[Move to Device]
    I --> J[Convert to float]
    J --> K{Mask is 3D?}
    K -->|Yes| L[Unsqueeze Channel Dimension]
    K -->|No| M[Keep Shape]
    L --> N[Prepared Images + Masks]
    M --> N
```

Supported batch formats:

```text
(images, masks)
(image, mask)
(images, masks)
```

For dictionary batches, the accepted key pairs are:
- `image` + `mask`
- `images` + `masks`

Images and masks are moved using non-blocking device transfers and converted to floating point tensors.

---

## 5. Training Epoch

```mermaid
flowchart TD
    A[train_one_epoch()] --> B[model.train()]
    B --> C[Start Timer]
    C --> D[Initialize Metric Accumulators]
    D --> E[Iterate DataLoader]

    E --> F[Prepare Batch]
    F --> G[optimizer.zero_grad]
    G --> H[Autocast if CUDA AMP Enabled]
    H --> I[Model(images)]
    I --> J[Criterion(logits, masks)]
    J --> K{AMP Scaler Active?}

    K -->|Yes| L[Scaler Backward]
    L --> M[Unscale]
    M --> N[Clip Gradients]
    N --> O[Scaler Step + Update]

    K -->|No| P[Loss Backward]
    P --> Q[Clip Gradients]
    Q --> R[Optimizer Step]

    O --> S[Calculate Batch Dice]
    R --> S
    S --> T[Calculate Batch IoU]
    T --> U[Accumulate Loss + Metrics]
    U --> E

    E -->|Finished| V[Calculate Epoch Averages]
    V --> W[Return EpochResult]
```

### Training features

| Feature | Implementation |
|---|---|
| Training mode | `model.train()` |
| Loss | User-provided segmentation criterion |
| Optimizer | User-provided PyTorch optimizer |
| AMP | Enabled only when requested and CUDA is available |
| Gradient clipping | Default maximum norm `1.0` |
| Dice threshold | Default `0.5` |
| IoU threshold | Default `0.5` |
| Timing | `time.perf_counter()` |
| Empty loader protection | Raises `ValueError` |

---

## 6. Mixed Precision

```mermaid
flowchart LR
    A[use_amp=True] --> B{Device is CUDA?}
    B -->|No| C[AMP Disabled]
    B -->|Yes| D[AMP Enabled]
    D --> E[torch.autocast]
    E --> F[GradScaler]
    F --> G[Backward]
    G --> H[Unscale]
    H --> I[Gradient Clipping]
    I --> J[Optimizer Step]
    J --> K[Scaler Update]
```

Automatic mixed precision is conditional:

```text
amp_enabled = use_amp and device.type == "cuda"
```

On CUDA, `torch.autocast` and `torch.amp.GradScaler` are used.

On CPU, the scaler is not created and standard training is used.

---

## 7. Validation Epoch

```mermaid
flowchart TD
    A[validate_one_epoch()] --> B[model.eval()]
    B --> C[@torch.no_grad]
    C --> D[Iterate Validation DataLoader]
    D --> E[Prepare Batch]
    E --> F[Autocast if CUDA AMP Enabled]
    F --> G[Model Forward Pass]
    G --> H[Validation Loss]
    H --> I[Batch Dice]
    I --> J[Batch IoU]
    J --> K[Accumulate Metrics]
    K --> D
    D -->|Finished| L[Calculate Averages]
    L --> M[Return EpochResult]
```

Validation does not perform:
- Backpropagation
- Gradient updates
- Optimizer steps

It uses `model.eval()` and `@torch.no_grad()`.

---

## 8. Dice Metric

```mermaid
flowchart LR
    A[Logits] --> B[Sigmoid]
    B --> C[Probability]
    C --> D[Threshold >= 0.5]
    D --> E[Predicted Mask]

    F[Target Mask] --> G[Threshold >= 0.5]
    G --> H[Binary Target]

    E --> I[Flatten Per Sample]
    H --> I
    I --> J[Intersection]
    I --> K[Predicted Area]
    I --> L[Target Area]

    J --> M[Dice Formula]
    K --> M
    L --> M
    M --> N[Mean Batch Dice]
```

The implemented metric is:

```text
Dice = (2 × Intersection + smooth)
       / (Prediction Area + Target Area + smooth)
```

The default smoothing value is `1e-7`.

---

## 9. IoU Metric

```mermaid
flowchart LR
    A[Logits] --> B[Sigmoid]
    B --> C[Threshold]
    C --> D[Predicted Mask]

    E[Target] --> F[Binary Mask]

    D --> G[Intersection]
    F --> G

    D --> H[Union]
    F --> H

    G --> I[IoU Formula]
    H --> I
    I --> J[Mean Batch IoU]
```

The implemented metric is:

```text
IoU = (Intersection + smooth)
      / (Union + smooth)
```

where:

```text
Union = Prediction + Target - Intersection
```

---

## 10. EpochResult

```mermaid
classDiagram
    class EpochResult {
        +float loss
        +float dice
        +float iou
        +int samples
        +float duration_seconds
    }
```

Each training and validation epoch returns:
- loss
- Dice
- IoU
- number of processed samples
- epoch duration

---

## 11. TrainingHistory

```mermaid
classDiagram
    class TrainingHistory {
        +list train_loss
        +list train_dice
        +list train_iou
        +list val_loss
        +list val_dice
        +list val_iou
        +list learning_rates
        +float best_val_loss
        +int best_epoch
    }
```

The history stores the complete metric trajectory of the training run.

Initial state:

```text
best_val_loss = infinity
best_epoch = -1
```

---

## 12. Learning-Rate Scheduling

```mermaid
flowchart TD
    A[Validation Epoch Complete] --> B[Read Validation Loss]
    B --> C{Scheduler Type}
    C -->|ReduceLROnPlateau| D[scheduler.step(val_loss)]
    C -->|Other Scheduler| E[scheduler.step()]
    D --> F[Next Epoch]
    E --> F
```

The pipeline supports both:
- `ReduceLROnPlateau`, stepped using validation loss
- Other PyTorch schedulers, stepped without arguments

The learning rate from the first optimizer parameter group is stored in `TrainingHistory`.

---

## 13. Best Checkpoint Selection

```mermaid
flowchart TD
    A[Validation Loss] --> B{val_loss < best_val_loss?}
    B -->|Yes| C[Update Best Loss]
    C --> D[Update Best Epoch]
    D --> E[Reset No-Improvement Counter]
    E --> F[Save checkpoints/best.pt]

    B -->|No| G[Increment Counter]
```

The best model is selected using **minimum validation loss**, not maximum Dice or IoU.

The best checkpoint contains:

```text
epoch
model_state_dict
optimizer_state_dict
train_loss
train_dice
train_iou
val_loss
val_dice
val_iou
```

When available, it also stores:
- scheduler state
- AMP scaler state
- training history

---

## 14. Early Stopping

```mermaid
flowchart TD
    A[Epoch Complete] --> B{Validation Loss Improved?}
    B -->|Yes| C[Counter = 0]
    B -->|No| D[Counter += 1]

    C --> E{Patience Enabled?}
    D --> E

    E -->|No| F[Continue]
    E -->|Yes| G{Counter >= Patience?}
    G -->|No| F
    G -->|Yes| H[Stop Training]
```

Default:

```text
early_stopping_patience = 5
```

Setting:

```text
early_stopping_patience = None
```

disables early stopping.

---

## 15. TensorBoard Logging

```mermaid
flowchart TD
    A[Epoch Results] --> B[SummaryWriter]
    B --> C[Loss/Train]
    B --> D[Loss/Validation]
    B --> E[Metrics/Train_Dice]
    B --> F[Metrics/Validation_Dice]
    B --> G[Metrics/Train_IoU]
    B --> H[Metrics/Validation_IoU]
    B --> I[Learning_Rate]
```

The default log directory is:

```text
runs/roadxai
```

Logged quantities per epoch:

```text
Loss/Train
Loss/Validation
Metrics/Train_Dice
Metrics/Validation_Dice
Metrics/Train_IoU
Metrics/Validation_IoU
Learning_Rate
```

The `SummaryWriter` is closed in a `finally` block, including when training exits because of an exception.

---

## 16. Checkpoint Loading

```mermaid
flowchart TD
    A[load_training_checkpoint()] --> B[Check File]
    B -->|Missing| C[FileNotFoundError]
    B -->|Exists| D[torch.load]
    D --> E{model_state_dict Present?}
    E -->|No| F[KeyError]
    E -->|Yes| G[Load Model State]

    G --> H{Optimizer Supplied?}
    H -->|Yes + State Exists| I[Load Optimizer]
    H -->|No| J[Skip]

    I --> K{Scheduler Supplied?}
    J --> K
    K -->|Yes + State Exists| L[Load Scheduler]
    K -->|No| M[Skip]

    L --> N{Scaler Supplied?}
    M --> N
    N -->|Yes + State Exists| O[Load Scaler]
    N -->|No| P[Return Checkpoint]

    O --> P
```

Checkpoint loading supports resuming:
- model weights
- optimizer state
- scheduler state
- AMP scaler state

The checkpoint is loaded with the selected device as `map_location`.

---

## 17. Checkpoint Modes

```mermaid
flowchart TD
    A[fit()] --> B{save_best_only?}
    B -->|True| C[Save only best.pt]
    B -->|False| D[Save best.pt]
    D --> E[Save epoch_001.pt]
    D --> F[Save epoch_002.pt]
    D --> G[...]
    D --> H[Save epoch_NNN.pt]
```

Default:

```text
save_best_only = True
```

When disabled, a checkpoint is also saved after every epoch:

```text
epoch_001.pt
epoch_002.pt
...
epoch_NNN.pt
```

---

## 18. Reproducibility Utilities

```mermaid
flowchart LR
    A[set_seed] --> B[Python random]
    A --> C[NumPy]
    A --> D[PyTorch CPU]
    A --> E[PyTorch CUDA]
    A --> F[cuDNN Deterministic]
```

`set_seed()` controls:
- Python random seed
- NumPy seed
- PyTorch CPU seed
- CUDA seeds when available

cuDNN is configured as:

```text
deterministic = True
benchmark = False
```

This prioritizes reproducibility over cuDNN benchmarking performance.

---

## 19. Training Utilities

```mermaid
flowchart TD
    A[Training Utilities] --> B[Reproducibility]
    A --> C[Device]
    A --> D[Model Statistics]
    A --> E[Metrics]
    A --> F[Learning Rate]
    A --> G[History Persistence]
    A --> H[DataLoader Validation]
    A --> I[Time Formatting]
    A --> J[Checkpoint Utilities]

    B --> B1[set_seed]
    C --> C1[get_device]
    D --> D1[count_parameters]
    E --> E1[calculate_dice]
    E --> E2[calculate_iou]
    F --> F1[get_current_learning_rate]
    G --> G1[save_json]
    G --> G2[load_json]
    G --> G3[save_history]
    H --> H1[validate_dataloader]
    I --> I1[format_seconds]
    J --> J1[checkpoint_exists]
    J --> J2[get_checkpoint_size_mb]
```

Utility functions also provide:
- parameter counting
- learning-rate inspection
- JSON persistence
- training-history persistence
- DataLoader validation
- elapsed-time formatting
- checkpoint existence checks
- checkpoint size inspection

---

## 20. Parameter Accounting

```mermaid
flowchart LR
    A[PyTorch Model] --> B[All Parameters]
    A --> C[Trainable Parameters]
    B --> D[Total Count]
    C --> E[Trainable Count]
    D --> F[Non-Trainable = Total - Trainable]
```

`count_parameters()` returns:

```text
total
trainable
non_trainable
```

This provides a direct model-complexity summary.

---

## 21. Training Pipeline API

```mermaid
flowchart TD
    A[fit] --> B[train_one_epoch]
    A --> C[validate_one_epoch]
    A --> D[save_training_checkpoint]
    A --> E[Scheduler]
    A --> F[TensorBoard]
    A --> G[TrainingHistory]

    H[load_training_checkpoint] --> I[Model]
    H --> J[Optimizer]
    H --> K[Scheduler]
    H --> L[Scaler]
```

Main public components:

| Component | Role |
|---|---|
| `EpochResult` | Stores one epoch's metrics |
| `TrainingHistory` | Stores complete training history |
| `train_one_epoch()` | Runs model training for one epoch |
| `validate_one_epoch()` | Runs validation for one epoch |
| `save_training_checkpoint()` | Saves training state |
| `load_training_checkpoint()` | Restores training state |
| `fit()` | Orchestrates the complete training pipeline |

---

## 22. Integration with RoadXAI

```mermaid
flowchart LR
    A[Dataset Pipeline] --> B[Train DataLoader]
    A --> C[Validation DataLoader]

    B --> D[RoadXAI Training Pipeline]
    C --> D

    D --> E[CNN Segmentation Model]
    D --> F[Best Checkpoint]
    D --> G[Training History]
    D --> H[TensorBoard Logs]

    F --> I[Evaluation]
    F --> J[Streamlit Application]
    F --> K[XAI]
    F --> L[Engineering Analysis]
```

The training pipeline is the bridge between the dataset pipeline and downstream RoadXAI inference/evaluation.

The resulting `best.pt` checkpoint can be consumed by the later application and analysis stages.

---

## 23. End-to-End Research View

```mermaid
flowchart LR
    A[Road Images + Masks]
    --> B[Dataset Pipeline]
    --> C[DataLoaders]
    --> D[U-Net Segmentation Model]
    --> E[Training Pipeline]

    E --> F[Validation]
    E --> G[Best Checkpoint]
    E --> H[Training History]
    E --> I[TensorBoard]

    G --> J[Evaluation]
    G --> K[Streamlit Inference]
    K --> L[Engineering Analysis]
    K --> M[Grad-CAM XAI]
    K --> N[3D Visualization]
    K --> O[Report Generation]
```

---

## 24. Technical Configuration

| Parameter | Default |
|---|---:|
| Epochs | `10` |
| Probability threshold | `0.5` |
| Early stopping patience | `5` |
| AMP | `True` |
| Gradient clipping max norm | `1.0` |
| Checkpoint directory | `checkpoints` |
| TensorBoard directory | `runs/roadxai` |
| Save best only | `True` |
| Device | CUDA if available, otherwise CPU |

---

## 25. Scientific / Engineering Limitations

### Metric limitation

Dice and IoU are calculated from thresholded binary masks at the configured threshold. They therefore evaluate segmentation overlap rather than physical road-condition correctness.

### Checkpoint criterion

The implementation chooses the best checkpoint using validation loss. A model with the best Dice or IoU is not necessarily selected if its validation loss is not the minimum.

### Dataset dependence

Training quality depends on the quality, diversity, and annotation consistency of the DataLoaders supplied by the dataset pipeline.

### Physical measurement limitation

The training pipeline learns segmentation masks. It does not itself establish physical dimensions such as meters or millimeters.

### Reproducibility

The pipeline explicitly configures random seeds and cuDNN deterministic behavior, but exact reproducibility can still depend on the hardware and software execution environment.

---

## 26. Final RoadXAI Training Architecture

```mermaid
flowchart TD
    A[Dataset Pipeline] --> B[Train / Validation DataLoaders]

    B --> C[fit]

    C --> D[Batch Preparation]
    D --> E[Model Forward]
    E --> F[Segmentation Loss]
    F --> G[Backward]
    G --> H[Gradient Clipping]
    H --> I[Optimizer]

    I --> J[Train Dice / IoU]
    E --> K[Validation]
    K --> L[Validation Loss / Dice / IoU]

    L --> M[Learning-Rate Scheduler]
    L --> N{Best Validation Loss?}

    N -->|Yes| O[best.pt]
    N -->|No| P[Early-Stopping Counter]

    C --> Q[TrainingHistory]
    C --> R[TensorBoard]

    O --> S[RoadXAI Evaluation]
    O --> T[Streamlit Inference]
    T --> U[Engineering]
    T --> V[XAI]
    T --> W[3D]
    T --> X[Reports]
```

**Training Pipeline role:** optimize the segmentation model, measure training/validation performance, preserve the best model state, and provide reproducible training artifacts for the rest of RoadXAI.
