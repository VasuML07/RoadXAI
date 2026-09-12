"""
RoadXAI Deployment Utilities.

Helper functions for deployment validation, environment inspection,
model artifact management, and production readiness checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import torch


def get_project_root(
    current_file: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve the RoadXAI project root.

    Parameters
    ----------
    current_file:
        Optional file path used as the starting point.
        Defaults to this module's location.
    """
    if current_file is None:
        current = Path(__file__).resolve()
    else:
        current = Path(current_file).resolve()

    path = current

    for _ in range(5):
        if (
            (path / "cnn_model").exists()
            or (path / "roadxai").exists()
            or (path / "requirements.txt").exists()
            or (path / "pyproject.toml").exists()
        ):
            return path

        path = path.parent

    return current.parent


def ensure_directory(
    directory: Union[str, Path],
) -> Path:
    """
    Create a directory if it does not already exist.
    """
    path = Path(directory)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def verify_file(
    file_path: Union[str, Path],
    allowed_extensions: Optional[
        Iterable[str]
    ] = None,
) -> Path:
    """
    Verify that a required file exists and is a regular file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file, received: {path}"
        )

    if allowed_extensions is not None:
        allowed = {
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in allowed_extensions
        }

        if path.suffix.lower() not in allowed:
            raise ValueError(
                f"Unsupported file extension: "
                f"{path.suffix}"
            )

    return path


def get_file_size(
    file_path: Union[str, Path],
) -> int:
    """
    Return a file's size in bytes.
    """
    path = verify_file(file_path)

    return path.stat().st_size


def format_file_size(
    size_bytes: int,
    decimals: int = 2,
) -> str:
    """
    Convert bytes into a human-readable file size.
    """
    if size_bytes < 0:
        raise ValueError(
            "size_bytes cannot be negative."
        )

    if size_bytes == 0:
        return "0 B"

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    size = float(size_bytes)
    index = 0

    while size >= 1024.0 and index < len(units) - 1:
        size /= 1024.0
        index += 1

    return (
        f"{size:.{decimals}f} "
        f"{units[index]}"
    )


def calculate_file_hash(
    file_path: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate a cryptographic hash for a file.
    """
    path = verify_file(file_path)

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive."
        )

    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported hashing algorithm: {algorithm}"
        ) from exc

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def get_model_parameter_count(
    model: torch.nn.Module,
) -> Dict[str, int]:
    """
    Calculate total and trainable model parameters.
    """
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(
            total - trainable
        ),
    }


def estimate_model_size_mb(
    model: torch.nn.Module,
) -> float:
    """
    Estimate model parameter memory in megabytes.
    """
    total_bytes = 0

    for parameter in model.parameters():
        total_bytes += (
            parameter.numel()
            * parameter.element_size()
        )

    for buffer in model.buffers():
        total_bytes += (
            buffer.numel()
            * buffer.element_size()
        )

    return total_bytes / (
        1024.0 * 1024.0
    )


def inspect_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Inspect the structure and metadata of a PyTorch checkpoint.
    """
    path = verify_file(
        checkpoint_path,
        allowed_extensions=[
            ".pt",
            ".pth",
            ".ckpt",
        ],
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    information: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": get_file_size(path),
        "size": format_file_size(
            get_file_size(path)
        ),
        "sha256": calculate_file_hash(path),
        "type": type(checkpoint).__name__,
    }

    if isinstance(checkpoint, dict):
        information["keys"] = list(
            checkpoint.keys()
        )

        for key in [
            "epoch",
            "best_metric",
            "best_val_loss",
            "learning_rate",
        ]:
            if key in checkpoint:
                value = checkpoint[key]

                if hasattr(value, "item"):
                    value = value.item()

                information[key] = value

        state_dict = checkpoint.get(
            "model_state_dict"
        )

        if state_dict is None:
            state_dict = checkpoint.get(
                "state_dict"
            )

        if state_dict is not None:
            information["parameter_tensors"] = len(
                state_dict
            )

    return information


def copy_model_artifact(
    source_path: Union[str, Path],
    destination_directory: Union[str, Path],
    destination_name: Optional[str] = None,
) -> Path:
    """
    Copy a trained model artifact into a deployment directory.
    """
    source = verify_file(
        source_path
    )

    destination_dir = ensure_directory(
        destination_directory
    )

    filename = (
        destination_name
        if destination_name
        else source.name
    )

    destination = (
        destination_dir / filename
    )

    shutil.copy2(
        source,
        destination,
    )

    return destination


def create_required_directories(
    base_directory: Union[str, Path],
    directories: Optional[
        Iterable[str]
    ] = None,
) -> Dict[str, Path]:
    """
    Create standard deployment directories.
    """
    base = ensure_directory(
        base_directory
    )

    if directories is None:
        directories = [
            "models",
            "logs",
            "reports",
            "outputs",
            "cache",
        ]

    result: Dict[str, Path] = {}

    for directory in directories:
        path = ensure_directory(
            base / directory
        )

        result[str(directory)] = path

    return result


def get_environment_variables(
    prefix: Optional[str] = None,
) -> Dict[str, str]:
    """
    Return environment variables, optionally filtered by prefix.
    """
    if prefix is None:
        return dict(os.environ)

    prefix = str(prefix)

    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith(prefix)
    }


def check_disk_space(
    directory: Union[str, Path],
    minimum_free_gb: float = 1.0,
) -> Dict[str, Any]:
    """
    Check available disk space.
    """
    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(
            f"Directory not found: {path}"
        )

    usage = shutil.disk_usage(
        path
    )

    total_gb = (
        usage.total
        / (1024 ** 3)
    )

    free_gb = (
        usage.free
        / (1024 ** 3)
    )

    used_gb = (
        usage.used
        / (1024 ** 3)
    )

    return {
        "path": str(path),
        "total_gb": round(
            total_gb,
            3,
        ),
        "used_gb": round(
            used_gb,
            3,
        ),
        "free_gb": round(
            free_gb,
            3,
        ),
        "minimum_required_gb": float(
            minimum_free_gb
        ),
        "sufficient": (
            free_gb >= minimum_free_gb
        ),
    }


def validate_model_output(
    output: torch.Tensor,
    expected_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validate the output of the RoadXAI segmentation model.
    """
    if not isinstance(
        output,
        torch.Tensor,
    ):
        raise TypeError(
            "Model output must be a torch.Tensor."
        )

    if output.ndim != 4:
        raise ValueError(
            "Expected model output shape "
            "[B, C, H, W]."
        )

    if output.shape[1] != 1:
        raise ValueError(
            "RoadXAI segmentation output "
            "must contain one channel."
        )

    if expected_batch_size is not None:
        if output.shape[0] != expected_batch_size:
            raise ValueError(
                f"Expected batch size "
                f"{expected_batch_size}, "
                f"got {output.shape[0]}."
            )

    if not torch.isfinite(
        output
    ).all():
        raise ValueError(
            "Model output contains NaN or infinite values."
        )

    return {
        "valid": True,
        "shape": list(output.shape),
        "batch_size": int(output.shape[0]),
        "channels": int(output.shape[1]),
        "height": int(output.shape[2]),
        "width": int(output.shape[3]),
        "device": str(output.device),
        "dtype": str(output.dtype),
    }


def create_runtime_manifest(
    model_path: Optional[
        Union[str, Path]
    ] = None,
    application_version: str = "1.0.0",
    model_name: str = "RoadXAIUNet",
) -> Dict[str, Any]:
    """
    Create deployment metadata for a production artifact.
    """
    import platform
    import sys

    manifest: Dict[str, Any] = {
        "application": "RoadXAI",
        "application_version": application_version,
        "model_name": model_name,
        "python_version": (
            sys.version.split()[0]
        ),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": (
            torch.cuda.is_available()
        ),
    }

    if torch.cuda.is_available():
        manifest["cuda_version"] = (
            torch.version.cuda
        )

        manifest["gpu_count"] = (
            torch.cuda.device_count()
        )

        manifest["gpu_names"] = [
            torch.cuda.get_device_name(index)
            for index in range(
                torch.cuda.device_count()
            )
        ]

    if model_path is not None:
        path = verify_file(
            model_path
        )

        manifest["model"] = {
            "path": str(path),
            "size_bytes": get_file_size(path),
            "sha256": calculate_file_hash(
                path
            ),
        }

    return manifest


def save_runtime_manifest(
    manifest: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Save runtime metadata as JSON.
    """
    path = Path(output_path)

    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            manifest,
            indent=4,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return path


def check_model_artifact(
    model_path: Union[str, Path],
    minimum_size_bytes: int = 1,
) -> Dict[str, Any]:
    """
    Perform basic integrity checks on a model artifact.
    """
    path = verify_file(
        model_path,
        allowed_extensions=[
            ".pt",
            ".pth",
            ".ckpt",
        ],
    )

    size = get_file_size(path)

    return {
        "valid_path": True,
        "exists": True,
        "is_file": True,
        "size_bytes": size,
        "size_valid": (
            size >= minimum_size_bytes
        ),
        "sha256": calculate_file_hash(
            path
        ),
    }


def production_readiness_check(
    model_path: Optional[
        Union[str, Path]
    ] = None,
    deployment_directory: Optional[
        Union[str, Path]
    ] = None,
) -> Dict[str, Any]:
    """
    Run a basic production-readiness assessment.
    """
    checks: Dict[str, Any] = {}

    checks["torch_available"] = True

    if model_path is not None:
        try:
            checks["model"] = (
                check_model_artifact(
                    model_path
                )
            )
        except Exception as exc:
            checks["model"] = {
                "valid": False,
                "error": str(exc),
            }

    if deployment_directory is not None:
        try:
            checks["disk"] = (
                check_disk_space(
                    deployment_directory
                )
            )
        except Exception as exc:
            checks["disk"] = {
                "valid": False,
                "error": str(exc),
            }

    model_valid = True

    if "model" in checks:
        model_valid = checks["model"].get(
            "valid",
            checks["model"].get(
                "size_valid",
                False,
            ),
        )

    disk_valid = True

    if "disk" in checks:
        disk_valid = checks["disk"].get(
            "sufficient",
            False,
        )

    checks["ready"] = bool(
        model_valid and disk_valid
    )

    return checks