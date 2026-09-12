"""
Utility functions for the RoadXAI evaluation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_evaluation_result(
    result: dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Save evaluation results as formatted JSON.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def load_evaluation_result(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load evaluation results from a JSON file.
    """

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Evaluation result not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        result = json.load(file)

    if not isinstance(result, dict):
        raise ValueError(
            "Evaluation result must contain a JSON object."
        )

    return result


def calculate_binary_metrics(
    true_positive: int,
    true_negative: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    """
    Calculate standard binary classification metrics.

    Metrics:
        Accuracy
        Precision
        Recall
        F1
        IoU
        Dice
    """

    if min(
        true_positive,
        true_negative,
        false_positive,
        false_negative,
    ) < 0:
        raise ValueError(
            "Confusion-matrix counts cannot be negative."
        )

    total = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )

    accuracy = (
        (true_positive + true_negative) / total
        if total > 0
        else 0.0
    )

    precision_denominator = (
        true_positive + false_positive
    )

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall_denominator = (
        true_positive + false_negative
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = precision + recall

    f1 = (
        2.0 * precision * recall / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    iou = (
        true_positive / iou_denominator
        if iou_denominator > 0
        else 0.0
    )

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    dice = (
        2 * true_positive / dice_denominator
        if dice_denominator > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
    }


def format_metrics(
    metrics: dict[str, float],
    decimals: int = 4,
) -> str:
    """
    Format evaluation metrics for terminal output.
    """

    if decimals < 0:
        raise ValueError(
            "decimals cannot be negative."
        )

    preferred_order = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "iou",
        "dice",
    ]

    lines: list[str] = []

    for name in preferred_order:
        if name in metrics:
            lines.append(
                f"{name.capitalize():<10}: "
                f"{metrics[name]:.{decimals}f}"
            )

    for name, value in metrics.items():
        if name not in preferred_order:
            lines.append(
                f"{name.capitalize():<10}: "
                f"{value:.{decimals}f}"
            )

    return "\n".join(lines)


def validate_metric_range(
    metrics: dict[str, float],
) -> None:
    """
    Validate that metric values are in the [0, 1] range.
    """

    for name, value in metrics.items():
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Metric '{name}' must be numeric."
            )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"Metric '{name}' must be between 0 and 1."
            )


def binary_mask_from_logits(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert segmentation logits into binary masks.
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            "logits must be a torch.Tensor."
        )

    if logits.ndim != 4:
        raise ValueError(
            "Expected logits with shape [B, C, H, W]."
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = torch.sigmoid(logits)

    return (
        probabilities >= threshold
    ).float()


def calculate_pixel_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, int]:
    """
    Calculate pixel-level confusion-matrix counts.
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have identical shapes."
        )

    predicted_masks = binary_mask_from_logits(
        predictions,
        threshold,
    )

    target_masks = (
        targets >= 0.5
    ).float()

    true_positive = (
        (predicted_masks == 1)
        & (target_masks == 1)
    ).sum().item()

    true_negative = (
        (predicted_masks == 0)
        & (target_masks == 0)
    ).sum().item()

    false_positive = (
        (predicted_masks == 1)
        & (target_masks == 0)
    ).sum().item()

    false_negative = (
        (predicted_masks == 0)
        & (target_masks == 1)
    ).sum().item()

    return {
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device | None = None,
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> dict[str, float]:
    """
    Measure model inference latency.

    Returns:
        Average, minimum and maximum latency in milliseconds.
    """

    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs cannot be negative."
        )

    if timed_runs <= 0:
        raise ValueError(
            "timed_runs must be greater than zero."
        )

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(device)
    model.eval()

    images = images.to(
        device=device,
        non_blocking=True,
    )

    for _ in range(warmup_runs):
        _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    timings: list[float] = []

    for _ in range(timed_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_event = (
            torch.cuda.Event(enable_timing=True)
            if device.type == "cuda"
            else None
        )

        end_event = (
            torch.cuda.Event(enable_timing=True)
            if device.type == "cuda"
            else None
        )

        if start_event is not None and end_event is not None:
            start_event.record()

            _ = model(images)

            end_event.record()

            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(
                end_event
            )

        else:
            import time

            start_time = time.perf_counter()

            _ = model(images)

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

        timings.append(float(elapsed_ms))

    return {
        "average_ms": sum(timings) / len(timings),
        "minimum_ms": min(timings),
        "maximum_ms": max(timings),
    }


def compare_models(
    results: dict[str, dict[str, float]],
    metric: str = "f1",
) -> str:
    """
    Produce a simple model comparison table.

    Args:
        results:
            Mapping of model name -> metrics.

        metric:
            Metric used to determine the best model.
    """

    if not results:
        raise ValueError(
            "No model results were provided."
        )

    for model_name, metrics in results.items():
        if metric not in metrics:
            raise KeyError(
                f"Metric '{metric}' missing for model "
                f"'{model_name}'."
            )

    ranked = sorted(
        results.items(),
        key=lambda item: item[1][metric],
        reverse=True,
    )

    lines = [
        f"{'Rank':<6}"
        f"{'Model':<25}"
        f"{metric.upper():<12}"
    ]

    lines.append("-" * 43)

    for rank, (model_name, metrics) in enumerate(
        ranked,
        start=1,
    ):
        lines.append(
            f"{rank:<6}"
            f"{model_name:<25}"
            f"{metrics[metric]:<12.4f}"
        )

    return "\n".join(lines)


def save_predictions(
    predictions: torch.Tensor,
    path: str | Path,
) -> Path:
    """
    Save prediction tensor to a PyTorch file.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        predictions.detach().cpu(),
        output_path,
    )

    return output_path


def load_predictions(
    path: str | Path,
) -> torch.Tensor:
    """Load predictions saved by save_predictions()."""

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {input_path}"
        )

    predictions = torch.load(
        input_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(predictions, torch.Tensor):
        raise TypeError(
            "Saved prediction file does not contain a tensor."
        )

    return predictions


__all__ = [
    "save_evaluation_result",
    "load_evaluation_result",
    "calculate_binary_metrics",
    "format_metrics",
    "validate_metric_range",
    "binary_mask_from_logits",
    "calculate_pixel_counts",
    "measure_inference_time",
    "compare_models",
    "save_predictions",
    "load_predictions",
]