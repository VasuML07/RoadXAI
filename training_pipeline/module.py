"""
RoadXAI Training Pipeline.

Provides:
    - Training loop
    - Validation loop
    - Mixed-precision training
    - Metric tracking
    - Checkpointing
    - Early stopping
    - Learning-rate scheduling
    - TensorBoard logging
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class EpochResult:
    """Results collected from one training or validation epoch."""

    loss: float
    dice: float
    iou: float
    samples: int
    duration_seconds: float


@dataclass
class TrainingHistory:
    """Complete history of a training run."""

    train_loss: list[float] = field(default_factory=list)
    train_dice: list[float] = field(default_factory=list)
    train_iou: list[float] = field(default_factory=list)

    val_loss: list[float] = field(default_factory=list)
    val_dice: list[float] = field(default_factory=list)
    val_iou: list[float] = field(default_factory=list)

    learning_rates: list[float] = field(default_factory=list)

    best_val_loss: float = float("inf")
    best_epoch: int = -1


def _prepare_batch(
    batch: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract images and masks from a DataLoader batch.

    Supports:
        tuple/list: (images, masks)
        dict: {"image": ..., "mask": ...}
        dict: {"images": ..., "masks": ...}
    """

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(
                "A batch tuple/list must contain images and masks."
            )

        images = batch[0]
        masks = batch[1]

    elif isinstance(batch, dict):
        if "image" in batch and "mask" in batch:
            images = batch["image"]
            masks = batch["mask"]

        elif "images" in batch and "masks" in batch:
            images = batch["images"]
            masks = batch["masks"]

        else:
            raise KeyError(
                "Batch dictionary must contain "
                "'image'/'mask' or 'images'/'masks'."
            )

    else:
        raise TypeError(
            "Unsupported batch type. Expected tuple, list, or dict."
        )

    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)

    if not isinstance(masks, torch.Tensor):
        masks = torch.as_tensor(masks)

    images = images.to(
        device=device,
        non_blocking=True,
    )

    masks = masks.to(
        device=device,
        non_blocking=True,
    )

    images = images.float()
    masks = masks.float()

    if masks.ndim == 3:
        masks = masks.unsqueeze(1)

    return images, masks


def _batch_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> torch.Tensor:
    """Calculate Dice score for a batch."""

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    targets = (targets >= 0.5).float()

    predictions = predictions.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return dice.mean()


def _batch_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> torch.Tensor:
    """Calculate Intersection over Union for a batch."""

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    targets = (targets >= 0.5).float()

    predictions = predictions.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    threshold: float = 0.5,
    max_grad_norm: float | None = 1.0,
) -> EpochResult:
    """
    Train the model for one complete epoch.

    Mixed precision is used when:
        - use_amp=True
        - CUDA is available
    """

    model.train()

    start_time = time.perf_counter()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    amp_enabled = use_amp and device.type == "cuda"

    for batch in dataloader:
        images, masks = _prepare_batch(
            batch=batch,
            device=device,
        )

        batch_size = images.shape[0]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

            optimizer.step()

        with torch.no_grad():
            dice = _batch_dice(
                logits=logits,
                targets=masks,
                threshold=threshold,
            )

            iou = _batch_iou(
                logits=logits,
                targets=masks,
                threshold=threshold,
            )

        total_loss += loss.item() * batch_size
        total_dice += dice.item() * batch_size
        total_iou += iou.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader contains no samples.")

    duration = time.perf_counter() - start_time

    return EpochResult(
        loss=total_loss / total_samples,
        dice=total_dice / total_samples,
        iou=total_iou / total_samples,
        samples=total_samples,
        duration_seconds=duration,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> EpochResult:
    """Evaluate the model on a validation dataset."""

    model.eval()

    start_time = time.perf_counter()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    amp_enabled = use_amp and device.type == "cuda"

    for batch in dataloader:
        images, masks = _prepare_batch(
            batch=batch,
            device=device,
        )

        batch_size = images.shape[0]

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        dice = _batch_dice(
            logits=logits,
            targets=masks,
            threshold=threshold,
        )

        iou = _batch_iou(
            logits=logits,
            targets=masks,
            threshold=threshold,
        )

        total_loss += loss.item() * batch_size
        total_dice += dice.item() * batch_size
        total_iou += iou.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation DataLoader contains no samples.")

    duration = time.perf_counter() - start_time

    return EpochResult(
        loss=total_loss / total_samples,
        dice=total_dice / total_samples,
        iou=total_iou / total_samples,
        samples=total_samples,
        duration_seconds=duration,
    )


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str | Path,
    train_result: EpochResult,
    val_result: EpochResult,
    scheduler: Any = None,
    scaler: torch.amp.GradScaler | None = None,
    history: TrainingHistory | None = None,
) -> Path:
    """
    Save a complete training checkpoint.
    """

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_result.loss,
        "train_dice": train_result.dice,
        "train_iou": train_result.iou,
        "val_loss": val_result.loss,
        "val_dice": val_result.dice,
        "val_iou": val_result.iou,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if history is not None:
        checkpoint["history"] = history

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    return checkpoint_path


def load_training_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: torch.amp.GradScaler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load a complete training checkpoint."""

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if (
        scaler is not None
        and "scaler_state_dict" in checkpoint
    ):
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    return checkpoint


def _step_scheduler(
    scheduler: Any,
    validation_loss: float,
) -> None:
    """Step a scheduler while supporting different scheduler APIs."""

    if scheduler is None:
        return

    if isinstance(
        scheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ):
        scheduler.step(validation_loss)
    else:
        scheduler.step()


def _get_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> float:
    """Get the first parameter group's learning rate."""

    if not optimizer.param_groups:
        raise ValueError(
            "Optimizer has no parameter groups."
        )

    return float(
        optimizer.param_groups[0]["lr"]
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | None = None,
    scheduler: Any = None,
    epochs: int = 10,
    checkpoint_dir: str | Path = "checkpoints",
    log_dir: str | Path = "runs/roadxai",
    use_amp: bool = True,
    threshold: float = 0.5,
    early_stopping_patience: int | None = 5,
    save_best_only: bool = True,
    max_grad_norm: float | None = 1.0,
) -> TrainingHistory:
    """
    Run the complete RoadXAI training pipeline.

    Args:
        model:
            CNN segmentation model.

        train_loader:
            Training DataLoader.

        val_loader:
            Validation DataLoader.

        criterion:
            Segmentation loss function.

        optimizer:
            Optimizer.

        device:
            CUDA or CPU. Automatically selected when omitted.

        scheduler:
            Optional learning-rate scheduler.

        epochs:
            Number of training epochs.

        checkpoint_dir:
            Directory for model checkpoints.

        log_dir:
            TensorBoard log directory.

        use_amp:
            Enable automatic mixed precision on CUDA.

        threshold:
            Probability threshold for Dice/IoU.

        early_stopping_patience:
            Number of epochs without validation improvement
            before stopping. None disables early stopping.

        save_best_only:
            If True, only the best checkpoint is saved.

        max_grad_norm:
            Maximum gradient norm. None disables clipping.
    """

    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if (
        early_stopping_patience is not None
        and early_stopping_patience < 0
    ):
        raise ValueError(
            "early_stopping_patience cannot be negative."
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)

    checkpoint_directory = Path(checkpoint_dir)
    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = SummaryWriter(
        log_dir=str(log_dir)
    )

    amp_enabled = use_amp and device.type == "cuda"

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )
        if device.type == "cuda"
        else None
    )

    history = TrainingHistory()

    epochs_without_improvement = 0

    try:
        for epoch in range(1, epochs + 1):
            train_result = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                threshold=threshold,
                max_grad_norm=max_grad_norm,
            )

            val_result = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                threshold=threshold,
                use_amp=use_amp,
            )

            learning_rate = _get_learning_rate(
                optimizer
            )

            _step_scheduler(
                scheduler=scheduler,
                validation_loss=val_result.loss,
            )

            history.train_loss.append(
                train_result.loss
            )
            history.train_dice.append(
                train_result.dice
            )
            history.train_iou.append(
                train_result.iou
            )

            history.val_loss.append(
                val_result.loss
            )
            history.val_dice.append(
                val_result.dice
            )
            history.val_iou.append(
                val_result.iou
            )

            history.learning_rates.append(
                learning_rate
            )

            writer.add_scalar(
                "Loss/Train",
                train_result.loss,
                epoch,
            )

            writer.add_scalar(
                "Loss/Validation",
                val_result.loss,
                epoch,
            )

            writer.add_scalar(
                "Metrics/Train_Dice",
                train_result.dice,
                epoch,
            )

            writer.add_scalar(
                "Metrics/Validation_Dice",
                val_result.dice,
                epoch,
            )

            writer.add_scalar(
                "Metrics/Train_IoU",
                train_result.iou,
                epoch,
            )

            writer.add_scalar(
                "Metrics/Validation_IoU",
                val_result.iou,
                epoch,
            )

            writer.add_scalar(
                "Learning_Rate",
                learning_rate,
                epoch,
            )

            is_best = (
                val_result.loss < history.best_val_loss
            )

            if is_best:
                history.best_val_loss = val_result.loss
                history.best_epoch = epoch
                epochs_without_improvement = 0

                save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    path=checkpoint_directory / "best.pt",
                    train_result=train_result,
                    val_result=val_result,
                    scheduler=scheduler,
                    scaler=scaler,
                    history=history,
                )

            else:
                epochs_without_improvement += 1

            if not save_best_only:
                save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    path=checkpoint_directory / f"epoch_{epoch:03d}.pt",
                    train_result=train_result,
                    val_result=val_result,
                    scheduler=scheduler,
                    scaler=scaler,
                    history=history,
                )

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {train_result.loss:.4f} | "
                f"Val Loss: {val_result.loss:.4f} | "
                f"Train Dice: {train_result.dice:.4f} | "
                f"Val Dice: {val_result.dice:.4f} | "
                f"Train IoU: {train_result.iou:.4f} | "
                f"Val IoU: {val_result.iou:.4f} | "
                f"LR: {learning_rate:.2e}"
            )

            if (
                early_stopping_patience is not None
                and epochs_without_improvement
                >= early_stopping_patience
            ):
                print(
                    "Early stopping triggered at "
                    f"epoch {epoch}."
                )
                break

    finally:
        writer.close()

    return history


__all__ = [
    "EpochResult",
    "TrainingHistory",
    "train_one_epoch",
    "validate_one_epoch",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "fit",
]