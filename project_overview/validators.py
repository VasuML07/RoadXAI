"""
RoadXAI Project Validators
==========================

Validation utilities for project-level configuration and metadata.
"""

from pathlib import Path
from typing import Iterable


def validate_non_empty_string(value: str, field_name: str) -> None:
    """Validate that a value is a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty.")


def validate_positive_integer(value: int, field_name: str) -> None:
    """Validate that a value is a positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")


def validate_probability(value: float, field_name: str) -> None:
    """Validate that a value lies in the range (0, 1)."""

    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")

    if not 0 < value < 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")


def validate_string_collection(
    values: Iterable[str],
    field_name: str,
) -> None:
    """Validate a collection containing non-empty strings."""

    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple.")

    if not values:
        raise ValueError(f"{field_name} cannot be empty.")

    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name}[{index}] must be a string."
            )

        if not value.strip():
            raise ValueError(
                f"{field_name}[{index}] cannot be empty."
            )


def validate_path(path: Path, field_name: str) -> None:
    """Validate that a path is a pathlib.Path instance."""

    if not isinstance(path, Path):
        raise TypeError(f"{field_name} must be a pathlib.Path.")


def validate_split_configuration(
    train_split: float,
    validation_split: float,
) -> None:
    """Validate train/validation split proportions."""

    validate_probability(train_split, "train_split")
    validate_probability(validation_split, "validation_split")

    if abs((train_split + validation_split) - 1.0) > 1e-6:
        raise ValueError(
            "train_split and validation_split must sum to 1.0."
        )


def validate_image_size(
    width: int,
    height: int,
) -> None:
    """Validate image dimensions."""

    validate_positive_integer(width, "width")
    validate_positive_integer(height, "height")


if __name__ == "__main__":
    validate_non_empty_string("RoadXAI", "project_name")
    validate_positive_integer(512, "image_size")
    validate_probability(0.8, "train_split")
    validate_string_collection(
        ["crack", "pothole"],
        "damage_classes",
    )
    validate_image_size(512, 512)
    validate_split_configuration(0.8, 0.2)

    print("RoadXAI Validators: PASS")