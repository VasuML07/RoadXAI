# RoadXAI Training Pipeline

> Training, validation, checkpointing, metrics, mixed precision, scheduling, early stopping, and TensorBoard logging for RoadXAI.

## 1. Pipeline Overview

```text
TRAINING DATA
     |
     v
+------------------+
| Batch Preparation|
+--------+---------+
         |
         v
+------------------+
| Move to Device   |
+--------+---------+
         |
         v
+------------------+
| Model Forward    |
+--------+---------+
         |
         v
+------------------+
| Segmentation Loss|
+--------+---------+
         |
         v
+------------------+
| Backpropagation  |
+--------+---------+
         |
         v
+------------------+
| Gradient Clip    |
+--------+---------+
         |
         v
+------------------+
| Optimizer Step   |
+--------+---------+
         |
         v
+------------------+
| Dice / IoU       |
+--------+---------+
         |
         v
+------------------+
| Epoch Results    |
+------------------+

VALIDATION
     |
     v
+------------------+
| Model Evaluation |
+--------+---------+
         |
         v
+------------------+
| Val Loss / Dice  |
| / IoU            |
+--------+---------+
         |
         +--------------------+
         |                    |
         v                    v
   Scheduler             Best Checkpoint
                              |
                              v
                           best.pt
```

The pipeline trains the RoadXAI CNN segmentation model and evaluates it after every epoch.

---

## 2. Complete Training Cycle

```text
fit()
 |
 +--> Validate configuration
 |
 +--> Select CUDA or CPU
 |
 +--> Move model to device
 |
 +--> Create checkpoint directory
 |
 +--> Create TensorBoard writer
 |
 +--> Enable AMP when CUDA is available
 |
 +--> Initialize TrainingHistory
 |
 +--> Epoch loop
       |
       +--> train_one_epoch()
       |
       +--> validate_one_epoch()
       |
       +--> Read learning rate
       |
       +--> Step scheduler
       |
       +--> Store metrics
       |
       +--> Write TensorBoard metrics
       |
       +--> Compare validation loss
       |      |
       |      +--> Improved -> save best.pt
       |      |
       |      +--> Not improved -> increment counter
       |
       +--> Check early stopping
 |
 +--> Close TensorBoard writer
 |
 +--> Return TrainingHistory
```

---

## 3. Configuration Validation

`fit()` validates the main training parameters before starting.

| Parameter | Requirement |
|---|---|
| `epochs` | Greater than `0` |
| `threshold` | Between `0` and `1` |
| `early_stopping_patience` | `None` or non-negative |
| `device` | Automatically selected when omitted |

Device selection:

```text
CUDA available?
     |
   +---+
   |   |
  YES  NO
   |   |
   v   v
 CUDA CPU
```

---

## 4. Batch Preparation

The training pipeline accepts these batch structures:

```text
(images, masks)
```

or

```text
(image, mask)
```

or dictionaries containing:

```text
{"image": ..., "mask": ...}
```

or:

```text
{"images": ..., "masks": ...}
```

Processing sequence:

```text
DataLoader batch
      |
      v
Extract images and masks
      |
      v
Convert to Tensor if necessary
      |
      v
Move to selected device
      |
      v
Convert to float
      |
      v
If mask is 3D:
add channel dimension
      |
      v
Prepared batch
```

Images and masks use non-blocking device transfers.

---

## 5. Training Epoch

`train_one_epoch()` performs one complete optimization pass.

```text
model.train()
     |
     v
For every batch
     |
     +--> Prepare images and masks
     |
     +--> optimizer.zero_grad()
     |
     +--> Forward pass
     |
     +--> Calculate loss
     |
     +--> Backward pass
     |
     +--> Gradient clipping
     |
     +--> Optimizer step
     |
     +--> Calculate Dice
     |
     +--> Calculate IoU
     |
     +--> Accumulate metrics
     |
     v
Calculate epoch averages
     |
     v
Return EpochResult
```

### Training features

| Feature | Implementation |
|---|---|
| Training mode | `model.train()` |
| Loss | User-provided criterion |
| Optimizer | User-provided optimizer |
| AMP | CUDA only when enabled |
| Gradient clipping | Default max norm `1.0` |
| Dice threshold | `0.5` |
| IoU threshold | `0.5` |
| Timing | `time.perf_counter()` |
| Empty loader | Raises `ValueError` |

---

## 6. Mixed Precision

AMP is enabled only when both conditions are satisfied:

```text
use_amp == True
        AND
device.type == "cuda"
```

CUDA path:

```text
Forward pass
     |
     v
torch.autocast
     |
     v
Loss
     |
     v
GradScaler
     |
     +--> Backward
     |
     +--> Unscale
     |
     +--> Gradient clipping
     |
     +--> Optimizer step
     |
     +--> Scaler update
```

CPU path:

```text
Forward
  |
  v
Loss
  |
  v
Backward
  |
  v
Gradient clipping
  |
  v
Optimizer step
```

---

## 7. Validation Epoch

Validation runs without parameter updates.

```text
model.eval()
     |
     v
torch.no_grad()
     |
     v
Validation DataLoader
     |
     +--> Prepare batch
     |
     +--> Forward pass
     |
     +--> Validation loss
     |
     +--> Dice
     |
     +--> IoU
     |
     +--> Accumulate metrics
     |
     v
Return EpochResult
```

Validation does not perform:

- Backpropagation
- Gradient updates
- Optimizer steps

---

## 8. Dice Metric

The implemented Dice calculation is:

```text
Logits
  |
  v
Sigmoid
  |
  v
Probabilities
  |
  v
Threshold at 0.5
  |
  v
Predicted binary mask
  |
  +------------------+
                     |
Target mask ---------+
                     |
                     v
               Flatten per sample
                     |
                     v
                Intersection
                     |
          +----------+----------+
          |                     |
          v                     v
   Prediction area         Target area
          \                     /
           \                   /
            v                 v
             +---------------+
             | Dice formula  |
             +---------------+
                     |
                     v
                Mean Dice
```

Formula:

```text
Dice = (2 * Intersection + smooth)
       / (Prediction Area + Target Area + smooth)
```

Default:

```text
smooth = 1e-7
threshold = 0.5
```

---

## 9. IoU Metric

The implemented IoU calculation is:

```text
Logits
  |
  v
Sigmoid
  |
  v
Threshold
  |
  v
Predicted mask
  |
  +------------------+
                     |
Target mask ---------+
                     |
                     v
                Intersection
                     |
                     v
                   Union
                     |
                     v
                 IoU score
```

Formula:

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

`EpochResult` stores the result of one training or validation epoch.

| Field | Meaning |
|---|---|
| `loss` | Mean epoch loss |
| `dice` | Mean Dice score |
| `iou` | Mean IoU score |
| `samples` | Number of processed samples |
| `duration_seconds` | Epoch execution time |

Conceptually:

```text
EpochResult
 |
 +-- loss
 +-- dice
 +-- iou
 +-- samples
 +-- duration_seconds
```

---

## 11. TrainingHistory

`TrainingHistory` stores the complete training trajectory.

```text
TrainingHistory
 |
 +-- train_loss
 +-- train_dice
 +-- train_iou
 |
 +-- val_loss
 +-- val_dice
 +-- val_iou
 |
 +-- learning_rates
 |
 +-- best_val_loss
 +-- best_epoch
```

Initial values:

```text
best_val_loss = infinity
best_epoch = -1
```

---

## 12. Learning-Rate Scheduling

After validation:

```text
Validation complete
       |
       v
Read validation loss
       |
       v
+-------------------------+
| Which scheduler?        |
+------------+------------+
             |
       +-----+-----+
       |           |
       v           v
ReduceLROnPlateau  Other
       |           |
       v           v
step(val_loss)   step()
       |           |
       +-----+-----+
             |
             v
        Next epoch
```

Supported behavior:

```text
ReduceLROnPlateau
    -> scheduler.step(validation_loss)

Other scheduler
    -> scheduler.step()
```

The learning rate of the first optimizer parameter group is stored in `TrainingHistory`.

---

## 13. Best Checkpoint

The best model is selected using:

```text
minimum validation loss
```

Decision:

```text
Current validation loss
          |
          v
Is it lower than best_val_loss?
          |
       +--+--+
       |     |
      YES    NO
       |     |
       v     v
Update     Increment
best       no-improvement
       |
       v
Save best.pt
```

The best checkpoint stores:

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

```text
scheduler_state_dict
scaler_state_dict
history
```

---

## 14. Early Stopping

Default:

```text
early_stopping_patience = 5
```

Logic:

```text
Epoch complete
     |
     v
Validation loss improved?
     |
   +---+---+
   |       |
  YES      NO
   |       |
   v       v
Counter   Counter + 1
= 0
   |       |
   +---+---+
       |
       v
Patience reached?
       |
    +--+--+
    |     |
   NO    YES
    |     |
    v     v
Continue Stop
```

Disable early stopping with:

```python
early_stopping_patience = None
```

---

## 15. TensorBoard

Default log directory:

```text
runs/roadxai
```

The following values are logged every epoch:

```text
Loss/Train
Loss/Validation

Metrics/Train_Dice
Metrics/Validation_Dice

Metrics/Train_IoU
Metrics/Validation_IoU

Learning_Rate
```

Flow:

```text
Epoch results
     |
     v
SummaryWriter
     |
     +--> Train loss
     +--> Validation loss
     +--> Train Dice
     +--> Validation Dice
     +--> Train IoU
     +--> Validation IoU
     +--> Learning rate
```

The writer is closed in a `finally` block.

---

## 16. Checkpoint Saving

Default checkpoint directory:

```text
checkpoints/
```

With:

```text
save_best_only = True
```

the main output is:

```text
checkpoints/
└── best.pt
```

With:

```text
save_best_only = False
```

the pipeline additionally saves:

```text
checkpoints/
├── best.pt
├── epoch_001.pt
├── epoch_002.pt
├── ...
└── epoch_NNN.pt
```

---

## 17. Checkpoint Loading

`load_training_checkpoint()` can restore:

```text
Checkpoint
    |
    +--> Model state
    |
    +--> Optimizer state
    |
    +--> Scheduler state
    |
    +--> AMP scaler state
```

Only supplied components with corresponding checkpoint state are restored.

The checkpoint is loaded using the selected device as `map_location`.

---

## 18. Reproducibility

`set_seed()` controls:

```text
Python random
     |
     +--> seed

NumPy
     |
     +--> seed

PyTorch CPU
     |
     +--> seed

PyTorch CUDA
     |
     +--> seed
```

When CUDA is available:

```text
torch.cuda.manual_seed()
torch.cuda.manual_seed_all()
```

cuDNN configuration:

```text
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This prioritizes reproducibility over cuDNN benchmarking performance.

---

## 19. Training Utilities

| Function | Purpose |
|---|---|
| `set_seed()` | Reproducible random state |
| `get_device()` | Select CUDA or CPU |
| `count_parameters()` | Count model parameters |
| `get_current_learning_rate()` | Read optimizer learning rate |
| `calculate_dice()` | Calculate Dice |
| `calculate_iou()` | Calculate IoU |
| `save_json()` | Save JSON |
| `load_json()` | Load JSON |
| `save_history()` | Save training history |
| `validate_dataloader()` | Validate DataLoader |
| `format_seconds()` | Format elapsed time |
| `checkpoint_exists()` | Check checkpoint existence |
| `get_checkpoint_size_mb()` | Get checkpoint size |

---

## 20. Parameter Accounting

`count_parameters()` returns:

```text
total
trainable
non_trainable
```

Relationship:

```text
Total parameters
       |
       +----------------------+
       |                      |
       v                      v
Trainable parameters    Non-trainable parameters
                              |
                              v
                 Total - Trainable
```

---

## 21. Public Training API

```text
training/
 |
 +-- EpochResult
 |
 +-- TrainingHistory
 |
 +-- train_one_epoch()
 |
 +-- validate_one_epoch()
 |
 +-- save_training_checkpoint()
 |
 +-- load_training_checkpoint()
 |
 +-- fit()
```

### Main responsibilities

| Component | Responsibility |
|---|---|
| `EpochResult` | One epoch's results |
| `TrainingHistory` | Complete run history |
| `train_one_epoch()` | Training |
| `validate_one_epoch()` | Validation |
| `save_training_checkpoint()` | Save state |
| `load_training_checkpoint()` | Restore state |
| `fit()` | Full training orchestration |

---

## 22. RoadXAI Integration

```text
DATASET PIPELINE
       |
       +--> Train DataLoader
       |
       +--> Validation DataLoader
                    |
                    v
           TRAINING PIPELINE
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Model     History   TensorBoard
          |
          v
      best.pt
          |
    +-----+------+----------------+
    |            |                |
    v            v                v
Evaluation   Streamlit          XAI
                 |
          +------+------+
          |             |
          v             v
     Engineering       3D
          |
          v
       Reports
```

The training pipeline is the hand-off point between dataset preparation and downstream RoadXAI inference/evaluation.

---

## 23. End-to-End Research Flow

```text
Road Images + Masks
        |
        v
Dataset Pipeline
        |
        v
DataLoaders
        |
        v
U-Net Segmentation Model
        |
        v
Training Pipeline
        |
        +--> Validation
        |
        +--> Training History
        |
        +--> TensorBoard
        |
        +--> Best Checkpoint
                    |
          +---------+---------+
          |         |         |
          v         v         v
      Evaluation Streamlit   XAI
                    |
              +-----+-----+
              |           |
              v           v
        Engineering      3D
              |
              v
           Reports
```

---

## 24. Default Configuration

| Parameter | Default |
|---|---|
| `epochs` | `10` |
| `threshold` | `0.5` |
| `early_stopping_patience` | `5` |
| `use_amp` | `True` |
| `max_grad_norm` | `1.0` |
| `checkpoint_dir` | `checkpoints` |
| `log_dir` | `runs/roadxai` |
| `save_best_only` | `True` |
| Device | CUDA if available, otherwise CPU |

---

## 25. Scientific and Engineering Boundaries

### Segmentation metrics

Dice and IoU measure segmentation overlap after thresholding. They do not directly measure road-condition quality.

### Checkpoint selection

The best checkpoint is selected using minimum validation loss, not maximum Dice or IoU.

### Dataset dependence

Model performance depends on the supplied training and validation data and their annotations.

### Physical measurements

The training pipeline learns segmentation masks. It does not itself establish physical dimensions such as meters or millimeters.

### Explainability

Grad-CAM-family outputs describe model activation behavior. They are not ground truth and do not prove causality.

### 3D visualization

The 3D representation is visual geometry unless validated depth information is supplied.

### Field validation

Engineering decisions should be validated through appropriate field inspection and calibration.

---

## 26. Final Architecture

```text
+----------------------+
|   Dataset Pipeline   |
+----------+-----------+
           |
           v
+----------------------+
| Train / Val Loaders  |
+----------+-----------+
           |
           v
+----------------------+
|        fit()         |
+----------+-----------+
           |
           v
+----------------------+
| Batch Preparation    |
+----------+-----------+
           |
           v
+----------------------+
| Model Forward Pass   |
+----------+-----------+
           |
           +--------------------+
           |                    |
           v                    v
+----------------------+  +----------------------+
| Training Loss        |  | Validation           |
+----------+-----------+  | Loss / Dice / IoU   |
           |              +----------+-----------+
           v                         |
+----------------------+             v
| Backward + Optimizer |     +-------------------+
+----------+-----------+     | Scheduler         |
           |                 +-------------------+
           v
+----------------------+
| Train Dice / IoU     |
+----------+-----------+
           |
           +--------------------+
                                |
                                v
                       +----------------------+
                       | Best Validation Loss|
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | checkpoints/best.pt |
                       +----------+-----------+
                                  |
                 +----------------+----------------+
                 |                |                |
                 v                v                v
             Evaluation      Streamlit           XAI
                                  |
                            +-----+-----+
                            |           |
                            v           v
                       Engineering     3D
                            |
                            v
                         Reports
```

## Training Pipeline Role

The training pipeline:

```text
OPTIMIZE
   +
VALIDATE
   +
MEASURE
   +
LOG
   +
CHECKPOINT
   +
STOP WHEN APPROPRIATE
        |
        v
READY FOR DOWNSTREAM ROADXAI STAGES
```

It provides the trained model state and training artifacts required by the rest of the RoadXAI system.
