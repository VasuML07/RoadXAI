"""
RoadXAI Evaluation Module.

Provides evaluation metrics for binary road-defect segmentation:
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - IoU
    - Dice score
    - Confusion matrix
    - Inference time
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class ConfusionMatrix:
    """Binary segmentation confusion-matrix counts."""

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    iou: float
    dice: float
    inference_time_ms: float
    samples: int
    confusion_matrix: ConfusionMatrix


def _prepare_batch(
    batch: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and prepare images and masks from a batch."""

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(
                "Batch must contain images and masks."
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
            "Batch must be a tuple, list, or dictionary."
        )

    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)

    if not isinstance(masks, torch.Tensor):
        masks = torch.as_tensor(masks)

    images = images.to(
        device=device,
        non_blocking=True,
    ).float()

    masks = masks.to(
        device=device,
        non_blocking=True,
    ).float()

    if masks.ndim == 3:
        masks = masks.unsqueeze(1)

    return images, masks


def _binary_predictions(
    logits: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Convert model logits into binary predictions."""

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = torch.sigmoid(logits)

    return (
        probabilities >= threshold
    ).bool()


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> ConfusionMatrix:
    """
    Calculate binary pixel-level confusion-matrix counts.
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have the same shape."
        )

    predicted = _binary_predictions(
        predictions,
        threshold,
    )

    actual = targets >= 0.5

    true_positive = (
        predicted & actual
    ).sum().item()

    true_negative = (
        (~predicted) & (~actual)
    ).sum().item()

    false_positive = (
        predicted & (~actual)
    ).sum().item()

    false_negative = (
        (~predicted) & actual
    ).sum().item()

    return ConfusionMatrix(
        true_positive=int(true_positive),
        true_negative=int(true_negative),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
    )


def calculate_accuracy(
    cm: ConfusionMatrix,
) -> float:
    """Calculate binary accuracy."""

    total = (
        cm.true_positive
        + cm.true_negative
        + cm.false_positive
        + cm.false_negative
    )

    if total == 0:
        return 0.0

    return (
        cm.true_positive
        + cm.true_negative
    ) / total


def calculate_precision(
    cm: ConfusionMatrix,
) -> float:
    """Calculate binary precision."""

    denominator = (
        cm.true_positive
        + cm.false_positive
    )

    if denominator == 0:
        return 0.0

    return cm.true_positive / denominator


def calculate_recall(
    cm: ConfusionMatrix,
) -> float:
    """Calculate binary recall."""

    denominator = (
        cm.true_positive
        + cm.false_negative
    )

    if denominator == 0:
        return 0.0

    return cm.true_positive / denominator


def calculate_f1(
    precision: float,
    recall: float,
) -> float:
    """Calculate F1 score from precision and recall."""

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / denominator
    )


def calculate_iou(
    cm: ConfusionMatrix,
) -> float:
    """Calculate Intersection over Union."""

    denominator = (
        cm.true_positive
        + cm.false_positive
        + cm.false_negative
    )

    if denominator == 0:
        return 0.0

    return cm.true_positive / denominator


def calculate_dice(
    cm: ConfusionMatrix,
) -> float:
    """Calculate Dice/F1 overlap score."""

    denominator = (
        2 * cm.true_positive
        + cm.false_positive
        + cm.false_negative
    )

    if denominator == 0:
        return 0.0

    return (
        2 * cm.true_positive
    ) / denominator


def merge_confusion_matrices(
    first: ConfusionMatrix,
    second: ConfusionMatrix,
) -> ConfusionMatrix:
    """Combine two binary confusion matrices."""

    return ConfusionMatrix(
        true_positive=(
            first.true_positive
            + second.true_positive
        ),
        true_negative=(
            first.true_negative
            + second.true_negative
        ),
        false_positive=(
            first.false_positive
            + second.false_positive
        ),
        false_negative=(
            first.false_negative
            + second.false_negative
        ),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> EvaluationResult:
    """
    Evaluate a segmentation model on a complete dataset.

    Metrics are calculated at the pixel level.
    Inference time excludes data-loading time.
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(device)
    model.eval()

    total_cm = ConfusionMatrix(
        true_positive=0,
        true_negative=0,
        false_positive=0,
        false_negative=0,
    )

    total_samples = 0
    total_inference_seconds = 0.0

    for batch in dataloader:
        images, masks = _prepare_batch(
            batch,
            device,
        )

        batch_size = images.shape[0]

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        batch_cm = confusion_matrix(
            predictions=logits,
            targets=masks,
            threshold=threshold,
        )

        total_cm = merge_confusion_matrices(
            total_cm,
            batch_cm,
        )

        total_samples += batch_size
        total_inference_seconds += elapsed

    if total_samples == 0:
        raise ValueError(
            "Evaluation DataLoader contains no samples."
        )

    accuracy = calculate_accuracy(total_cm)
    precision = calculate_precision(total_cm)
    recall = calculate_recall(total_cm)
    f1 = calculate_f1(
        precision,
        recall,
    )
    iou = calculate_iou(total_cm)
    dice = calculate_dice(total_cm)

    total_pixels = (
        total_cm.true_positive
        + total_cm.true_negative
        + total_cm.false_positive
        + total_cm.false_negative
    )

    if total_pixels == 0:
        inference_time_ms = 0.0
    else:
        inference_time_ms = (
            total_inference_seconds
            / total_samples
            * 1000.0
        )

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
        dice=dice,
        inference_time_ms=inference_time_ms,
        samples=total_samples,
        confusion_matrix=total_cm,
    )


def result_to_dict(
    result: EvaluationResult,
) -> dict[str, Any]:
    """Convert evaluation results to a serializable dictionary."""

    cm = result.confusion_matrix

    return {
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "iou": result.iou,
        "dice": result.dice,
        "inference_time_ms": result.inference_time_ms,
        "samples": result.samples,
        "confusion_matrix": {
            "true_positive": cm.true_positive,
            "true_negative": cm.true_negative,
            "false_positive": cm.false_positive,
            "false_negative": cm.false_negative,
        },
    }


__all__ = [
    "ConfusionMatrix",
    "EvaluationResult",
    "confusion_matrix",
    "calculate_accuracy",
    "calculate_precision",
    "calculate_recall",
    "calculate_f1",
    "calculate_iou",
    "calculate_dice",
    "merge_confusion_matrices",
    "evaluate_model",
    "result_to_dict",
]