# RoadXAI Training Pipeline

## 1. Purpose

The RoadXAI Training Pipeline trains the CNN segmentation model using paired road images and defect masks.

```mermaid
flowchart TD
    N0["DATASET PIPELINE"]
    N1["Train DataLoader"]
    N2["Validation DataLoader"]
    N3["fit()"]
    N4["Training Orchestrator"]
    N5["v           v"]
    N6["TRAINING    VALIDATION"]
    N7["Loss / Dice / IoU"]
    N8["LR Scheduler"]
    N9["Early Stopping"]
    N10["Checkpointing"]
    N11["TensorBoard"]
    N12["best.pt"]
    N13["Evaluation"]
    N14["Streamlit"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N7
    N7 --> N8
    N8 --> N9
    N9 --> N10
    N10 --> N11
    N11 --> N12
    N12 --> N13
    N13 --> N14
```

The package contains the training loop, validation loop, checkpointing, TensorBoard logging, and mixed-precision training components. fileciteturn12file3L1-L14

---

## 2. Training Package Structure

```mermaid
flowchart TD
    A[training/] --> B[__init__.py]
    A --> C[module.py]
    A --> D[utils.py]

    C --> C1[EpochResult]
    C --> C2[TrainingHistory]
    C --> C3[train_one_epoch]
    C --> C4[validate_one_epoch]
    C --> C5[save_training_checkpoint]
    C --> C6[load_training_checkpoint]
    C --> C7[fit]

    D --> D1[set_seed]
    D --> D2[get_device]
    D --> D3[count_parameters]
    D --> D4[calculate_dice]
    D --> D5[calculate_iou]
    D --> D6[History / JSON helpers]
    D --> D7[DataLoader validation]
    D --> D8[Checkpoint helpers]
```

---

## 3. Complete Training Flow

```mermaid
flowchart TD
    A[Road Images + Masks] --> B[DataLoader]
    B --> C[_prepare_batch]
    C --> C1[Extract image / mask]
    C1 --> C2[Convert to Tensor]
    C2 --> C3[Move to Device]
    C3 --> C4[Convert to Float]
    C4 --> C5[Fix Mask Dimensions]
    C5 --> D[Model Forward]
    D --> E[Logits]
    E --> F[BCE + Dice Loss]
    F --> G[Backpropagation]
    G --> H[Gradient Clipping]
    H --> I[Optimizer Step]
    I --> J[Dice + IoU Metrics]
    J --> K[EpochResult]
```

---

## 4. Configuration Validation

`fit()` validates the main training configuration before starting the run.

```mermaid
flowchart TD
    N0["fit()"]
    N1["epochs  0 ?"]
    N2["NO  ValueError"]
    N3["0 < threshold < 1 ?"]
    N4["early_stopping_patience valid ?"]
    N5["select CUDA / CPU"]
    N6["training"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
```

The implementation requires `epochs > 0`, requires the threshold to be strictly between 0 and 1, and validates the supplied early-stopping patience. fileciteturn12file5L1-L39

---

## 5. Batch Preparation

```mermaid
flowchart TD
    N0["DataLoader Batch"]
    N1["tuple/list                   dict"]
    N2["batch[0], batch[1]"]
    N3["v             v"]
    N4["image  mask    images  masks"]
    N5["\             /"]
    N6["\           /"]
    N7["v         v"]
    N8["Tensor Conversion"]
    N9["Move to Device"]
    N10["float32"]
    N11["Mask ndim == 3?"]
    N12["/           \"]
    N13["YES            NO"]
    N14["v              v"]
    N15["unsqueeze(1)      keep"]
    N16["\              /"]
    N17["\            /"]
    N18["v          v"]
    N19["Prepared Batch"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N7
    N7 --> N8
    N8 --> N9
    N9 --> N10
    N10 --> N11
    N11 --> N12
    N12 --> N13
    N13 --> N14
    N14 --> N15
    N15 --> N16
    N16 --> N17
    N17 --> N18
    N18 --> N19
```

Supported inputs are:
- tuple/list containing image and mask
- dictionary with `image` + `mask`
- dictionary with `images` + `masks`

Images and masks are transferred with `non_blocking=True` and converted to floating-point tensors. A 3D mask receives a channel dimension. fileciteturn12file8L1-L68

---

## 6. Training Epoch

```mermaid
flowchart TD
    A[train_one_epoch] --> B[model.train]
    B --> C[Start Timer]
    C --> D[Iterate DataLoader]
    D --> E[_prepare_batch]
    E --> F[optimizer.zero_grad]
    F --> G{CUDA AMP enabled?}
    G -->|Yes| H[autocast]
    G -->|No| I[Normal Forward]
    H --> J[model images]
    I --> J
    J --> K[criterion]
    K --> L[Loss]
    L --> M{AMP?}
    M -->|Yes| N[Scaler Backward]
    M -->|No| O[loss.backward]
    N --> P[Gradient Clipping]
    O --> P
    P --> Q[Optimizer Step]
    Q --> R[Dice + IoU]
    R --> S[Weighted Accumulation]
    S --> T[Epoch Averages]
    T --> U[EpochResult]
```

The training implementation uses `model.train()`, records elapsed time, accumulates loss/Dice/IoU by sample count, protects against an empty DataLoader, and returns an `EpochResult`. fileciteturn12file18L1-L73

---

## 7. Mixed-Precision Training

AMP is conditional:

```mermaid
flowchart TD
    N0["amp_enabled = use_amp AND device.type == 'cuda'"]
    N1["use_amp"]
    N2["CUDA device?"]
    N3["/        \"]
    N4["NO          YES"]
    N5["v            v"]
    N6["AMP disabled   AMP enabled"]
    N7["torch.autocast"]
    N8["GradScaler"]
    N9["backward"]
    N10["scaler.unscale_()"]
    N11["gradient clipping"]
    N12["scaler.step()"]
    N13["scaler.update()"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N7
    N7 --> N8
    N8 --> N9
    N9 --> N10
    N10 --> N11
    N11 --> N12
    N12 --> N13
```

On CUDA, the training loop uses autocasting and a gradient scaler. On CPU, it falls back to normal training. Gradient clipping is performed after unscaling when AMP is active. fileciteturn12file18L65-L73

---

## 8. Validation Epoch

```mermaid
flowchart TD
    A[validate_one_epoch] --> B[model.eval]
    B --> C[torch.no_grad]
    C --> D[Validation DataLoader]
    D --> E[_prepare_batch]
    E --> F[model images]
    F --> G[Validation Loss]
    G --> H[Dice]
    G --> I[IoU]
    H --> J[Accumulate by Samples]
    I --> J
    J --> K[Epoch Averages]
    K --> L[EpochResult]
```

Validation is measurement-only. It does not perform backpropagation, optimizer updates, or gradient-based training.

---

## 9. Dice Metric

```mermaid
flowchart TD
    A[Logits] --> B[Sigmoid]
    B --> C[Probability]
    C --> D[Threshold >= 0.5]
    D --> E[Predicted Binary Mask]
    E --> F[Flatten per Sample]
    G[Target Mask] --> F
    F --> H[Intersection]
    F --> I[Predicted Area]
    F --> J[Target Area]
    H --> K[Dice Score]
    I --> K
    J --> K
```

The implemented formula is:

```text
Dice =
(2 * Intersection + smooth)
--------------------------------
(Prediction Area + Target Area + smooth)
```

with default `smooth = 1e-7` and default threshold `0.5`. The utility implementation also applies sigmoid to logits before thresholding. fileciteturn12file6L42-L90

---

## 10. IoU Metric

```mermaid
flowchart TD
    A[Logits] --> B[Sigmoid]
    B --> C[Threshold]
    C --> D[Predicted Mask]
    D --> E[Intersection]
    F[Target Mask] --> E
    D --> G[Union]
    F --> G
    E --> H[IoU]
    G --> H
```

The implemented union is:

```text
Union = Prediction + Target - Intersection
```

and:

```text
IoU =
(Intersection + smooth)
-----------------------
(Union + smooth)
```

The default threshold is `0.5` and the default smoothing value is `1e-7`. fileciteturn12file17L69-L105

---

## 11. EpochResult

Each training or validation epoch returns:

```mermaid
flowchart TD
    N0["EpochResult"]
    N1["loss"]
    N2["dice"]
    N3["iou"]
    N4["samples"]
    N5["duration_seconds"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
```

This gives the training orchestrator a consistent representation of epoch-level results. fileciteturn12file8L16-L38

---

## 12. TrainingHistory

```mermaid
flowchart TD
    A[TrainingHistory]
    A --> B[train_loss]
    A --> C[train_dice]
    A --> D[train_iou]
    A --> E[val_loss]
    A --> F[val_dice]
    A --> G[val_iou]
    A --> H[learning_rates]
    A --> I[best_val_loss]
    A --> J[best_epoch]
```

The history object stores the complete training trajectory and identifies the epoch associated with the lowest validation loss. fileciteturn12file8L28-L45

---

## 13. Epoch-Level Orchestration

```mermaid
flowchart TD
    A[fit] --> B[Setup Device]
    B --> C[Setup TensorBoard Writer]
    C --> D[Setup AMP Scaler if Applicable]
    D --> E[Epoch Loop]
    E --> F[train_one_epoch]
    F --> G[validate_one_epoch]
    G --> H[Append TrainingHistory]
    H --> I[TensorBoard Logging]
    I --> J[Scheduler Step]
    J --> K{Validation Loss Improved?}
    K -->|Yes| L[Save best.pt]
    K -->|No| M[Increment Counter]
    L --> N[Optional Epoch Checkpoint]
    M --> N
    N --> O{Early Stopping?}
    O -->|No| E
    O -->|Yes| P[Close TensorBoard Writer]
    P --> Q[Return TrainingHistory]
```

The implementation logs training and validation metrics, records the learning rate, saves `best.pt` when validation loss improves, optionally saves every epoch, and stops when the configured patience is reached. fileciteturn12file9L1-L61

---

## 14. Learning-Rate Scheduling

The CNN module supplies a `ReduceLROnPlateau` scheduler based on validation loss.

```mermaid
flowchart TD
    N0["Validation Loss"]
    N1["Scheduler"]
    N2["Has loss plateaued?"]
    N3["/          \"]
    N4["YES           NO"]
    N5["v             v"]
    N6["Reduce LR     Keep LR"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
```

The scheduler builder uses:
- factor: `0.5`
- patience: `3`
- minimum learning rate: `1e-6`

The training loop supports both `ReduceLROnPlateau` and other scheduler APIs. For `ReduceLROnPlateau`, validation loss is passed to `scheduler.step()`. fileciteturn12file14L57-L91 fileciteturn12file5L1-L25

---

## 15. Checkpoint Selection

```mermaid
flowchart TD
    N0["Validation Loss"]
    N1["Better than best?"]
    N2["/          \"]
    N3["YES           NO"]
    N4["v             v"]
    N5["best_val_loss      counter"]
    N6["updated"]
    N7["best.pt"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N7
```

The best checkpoint is selected using **minimum validation loss**, not maximum Dice or IoU. fileciteturn12file9L15-L34

---

## 16. Checkpoint Contents

The checkpoint loader expects `model_state_dict` and can restore the available training state.

```mermaid
flowchart TD
    A[Checkpoint]
    A --> B[model_state_dict]
    A --> C[optimizer_state_dict]
    A --> D[scheduler_state_dict]
    A --> E[scaler_state_dict]
    A --> F[epoch / training metadata]
```

The loader:
1. verifies that the checkpoint file exists
2. loads it with the selected device as `map_location`
3. requires `model_state_dict`
4. restores optimizer, scheduler, and scaler state when both the object and corresponding state are available. fileciteturn12file5L1-L25

---

## 17. Checkpoint Modes

```mermaid
flowchart TD
    A{save_best_only?}
    A -->|True| B[best.pt]
    A -->|False| C[best.pt]
    A --> D[epoch_001.pt]
    A --> D2[epoch_002.pt]
    A --> D3[...]
    A --> D4[epoch_NNN.pt]
```

With `save_best_only=True`, only the best checkpoint is saved by the epoch-level checkpoint logic.

With `save_best_only=False`, the implementation additionally saves a checkpoint after each epoch. fileciteturn12file9L27-L42

---

## 18. Reproducibility

```mermaid
flowchart TD
    A[set_seed seed] --> B[Python random]
    A --> C[NumPy]
    A --> D[PyTorch CPU]
    A --> E[PyTorch CUDA]
    A --> F[cuDNN deterministic = True]
    A --> G[cuDNN benchmark = False]
```

The default seed is `42`. The utility configures deterministic cuDNN behavior to prioritize reproducibility. fileciteturn12file6L1-L39

---

## 19. Training Utilities

```mermaid
flowchart TD
    A[Training Utilities]
    A --> B[Reproducibility]
    B --> B1[set_seed]
    A --> C[Hardware]
    C --> C1[get_device]
    A --> D[Model Statistics]
    D --> D1[count_parameters]
    A --> E[Metrics]
    E --> E1[calculate_dice]
    E --> E2[calculate_iou]
    A --> F[Learning Rate]
    F --> F1[get_current_learning_rate]
    A --> G[Persistence]
    G --> G1[save_json]
    G --> G2[load_json]
    G --> G3[save_history]
    A --> H[Validation]
    H --> H1[validate_dataloader]
    A --> I[Checkpoints]
    I --> I1[checkpoint_exists]
    I --> I2[get_checkpoint_size_mb]
    A --> J[Timing]
    J --> J1[format_seconds]
```

These utilities support experiment tracking, validation, reproducibility, model accounting, and checkpoint inspection. fileciteturn12file2L1-L29

---

## 20. Model, Loss, Optimizer and Scheduler

The training pipeline receives the model, loss, optimizer, and optional scheduler from the CNN model package.

```mermaid
flowchart TD
    A[CNN Model Package] --> B[Model]
    A --> C[Loss]
    A --> D[Optimizer]
    B --> E[fit]
    C --> E
    D --> E
    E --> F[Scheduler]
```

Current CNN builders include:

```text
build_model()
build_loss()
build_optimizer()
build_scheduler()
```

The implemented defaults are:
- U-Net-style binary segmentation model
- BCE + Dice loss with equal default weights
- AdamW with learning rate `1e-3`
- AdamW weight decay `1e-4`
- ReduceLROnPlateau scheduler. fileciteturn12file14L1-L91

---

## 21. Parameter Accounting

```mermaid
flowchart TD
    A[PyTorch Model] --> B[All Parameters]
    A --> C[Trainable Parameters]
    B --> D[Total Count]
    C --> E[Trainable Count]
    D --> F[Non-trainable = Total - Trainable]
```

`count_parameters()` reports:

```text
total
trainable
non_trainable
```

This provides a model-complexity summary. fileciteturn12file6L25-L39

---

## 22. Public Training API

| Component | Purpose |
|---|---|
| `EpochResult` | Stores one epoch's metrics |
| `TrainingHistory` | Stores complete run history |
| `train_one_epoch()` | Trains for one epoch |
| `validate_one_epoch()` | Validates for one epoch |
| `save_training_checkpoint()` | Saves training state |
| `load_training_checkpoint()` | Restores training state |
| `fit()` | Runs the complete training process |

These are the public training components exported by the training module. fileciteturn12file9L45-L61

---

## 23. RoadXAI Integration

```mermaid
flowchart TD
    N0["Dataset Pipeline"]
    N1["Train / Val Loaders"]
    N2["Training Pipeline"]
    N3["fit()"]
    N4["v           v                v"]
    N5["best.pt    History         TensorBoard"]
    N6["v        v        v"]
    N7["Evaluation  Streamlit  Inference"]
    N8["Engineering"]
    N9["XAI"]
    N10["3D"]
    N11["Reports"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N7
    N7 --> N8
    N8 --> N9
    N9 --> N10
    N10 --> N11
```

The training pipeline is the bridge from dataset DataLoaders to the trained segmentation checkpoint used by downstream RoadXAI components.

---

## 24. Technical Configuration

| Parameter | `fit()` Default |
|---|---:|
| Epochs | `10` |
| Probability threshold | `0.5` |
| Early-stopping patience | `5` |
| AMP | `True` |
| Gradient clipping max norm | `1.0` |
| Checkpoint directory | `checkpoints` |
| TensorBoard directory | `runs/roadxai` |
| Save best only | `True` |
| Device | CUDA if available, otherwise CPU |

These values come directly from the `fit()` signature. fileciteturn12file5L25-L39

---

## 25. Actual Training Objective

The CNN package uses a combined BCE + Dice loss:

```mermaid
flowchart TD
    N0["Total Loss"]
    N1["v                  v"]
    N2["BCE Loss          Dice Loss"]
    N3["0.5 x              0.5 x"]
    N4["\                  /"]
    N5["\                /"]
    N6["Training Loss"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
```

The default loss weights are:

```text
BCE weight  = 0.5
Dice weight = 0.5
```

The combined loss operates on model logits and target masks. fileciteturn12file14L1-L29

---

## 26. What the Pipeline Produces

```mermaid
flowchart TD
    N0["Training Run"]
    N1["Epoch metrics"]
    N2["Train Loss"]
    N3["Train Dice"]
    N4["Train IoU"]
    N5["Val Loss"]
    N6["Val Dice"]
    N7["Val IoU"]
    N8["Learning Rate"]
    N9["TrainingHistory"]
    N10["TensorBoard logs"]
    N11["best.pt"]
    N12["optional epoch_NNN.pt"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N7
    N7 --> N8
    N8 --> N9
    N9 --> N10
    N10 --> N11
    N11 --> N12
```

The outputs are intended to support model selection, experiment analysis, checkpoint restoration, and downstream inference.

---

## 27. Scientific / Engineering Limitations

### Segmentation metrics

Dice and IoU measure mask overlap. They do not directly establish physical road-condition correctness.

### Checkpoint criterion

The selected `best.pt` is based on minimum validation loss. It is not necessarily the epoch with the highest validation Dice or IoU.

### Dataset dependence

Training performance depends on the quality, diversity, and annotation consistency of the supplied DataLoaders.

### Physical measurements

The training pipeline learns segmentation masks. It does not itself establish physical dimensions in meters or millimeters.

### Reproducibility

Seed and deterministic settings improve reproducibility, but exact results can still depend on the software, hardware, and execution environment.

---

## 28. Final Training Architecture

```mermaid
flowchart TD
    A[Dataset Pipeline] --> B[Train / Validation DataLoaders]
    B --> C[fit]
    C --> D[train_one_epoch]
    C --> E[validate_one_epoch]

    D --> D1[model.train]
    D1 --> D2[Forward + Loss]
    D2 --> D3[Backward]
    D3 --> D4[Gradient Clipping]
    D4 --> D5[Optimizer Step]
    D5 --> F[Epoch Metrics]

    E --> E1[model.eval]
    E1 --> E2[Forward + Loss]
    E2 --> E3[Dice / IoU]
    E3 --> F

    F --> G[LR Scheduler]
    F --> H[Best Checkpoint]
    F --> I[Early Stop]
    H --> J[best.pt]

    J --> K[Evaluation]
    J --> L[Streamlit]
    J --> M[Deployment]

    L --> N[Engineering]
    L --> O[XAI]
    L --> P[3D]
    L --> Q[Reports]
```

**Training Pipeline role:** optimize the segmentation model, measure training and validation performance, preserve the best model state, and provide training artifacts for the rest of RoadXAI.
