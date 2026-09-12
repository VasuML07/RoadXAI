"""
RoadXAI Deployment Module.

Provides production-oriented utilities for loading models,
checking runtime capabilities, managing inference configuration,
and preparing RoadXAI components for deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import json
import platform
import sys
import time

import torch


@dataclass
class DeploymentConfig:
    """Configuration used by a RoadXAI deployment."""

    model_path: Optional[str] = None
    device: str = "auto"
    image_size: int = 512
    threshold: float = 0.5
    max_batch_size: int = 1
    enable_xai: bool = True
    enable_3d: bool = True


@dataclass
class RuntimeInfo:
    """Information about the current deployment runtime."""

    python_version: str
    platform: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    device: str


@dataclass
class InferenceTiming:
    """Inference timing information."""

    elapsed_seconds: float
    images_per_second: float


def resolve_device(
    device: str = "auto",
) -> torch.device:
    """
    Resolve the deployment device.

    Supported values:
        auto
        cpu
        cuda
        cuda:0
        mps
    """
    normalized = str(device).lower().strip()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device(normalized)

    if normalized == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS was requested but is not available."
            )

        return torch.device("mps")

    if normalized == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported deployment device: {device}"
    )


def get_runtime_info(
    device: str = "auto",
) -> RuntimeInfo:
    """
    Collect runtime information for deployment diagnostics.
    """
    resolved_device = resolve_device(device)

    return RuntimeInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        device=str(resolved_device),
    )


def validate_deployment_config(
    config: DeploymentConfig,
) -> DeploymentConfig:
    """
    Validate deployment configuration.
    """
    if not isinstance(
        config,
        DeploymentConfig,
    ):
        raise TypeError(
            "config must be a DeploymentConfig."
        )

    if config.image_size <= 0:
        raise ValueError(
            "image_size must be positive."
        )

    if not 0.0 <= config.threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if config.max_batch_size <= 0:
        raise ValueError(
            "max_batch_size must be positive."
        )

    resolve_device(config.device)

    if config.model_path is not None:
        path = Path(config.model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}"
            )

    return config


def load_model(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    device: str = "auto",
    strict: bool = True,
) -> torch.nn.Module:
    """
    Load a trained PyTorch model for deployment.
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    target_device = resolve_device(device)

    checkpoint = torch.load(
        path,
        map_location=target_device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict"
        )

        if state_dict is None:
            state_dict = checkpoint.get(
                "state_dict"
            )

        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=strict,
    )

    model.to(target_device)
    model.eval()

    return model


def prepare_model_for_inference(
    model: torch.nn.Module,
    device: str = "auto",
) -> torch.nn.Module:
    """
    Move a model to the deployment device and enable evaluation mode.
    """
    target_device = resolve_device(device)

    model = model.to(target_device)
    model.eval()

    return model


@torch.inference_mode()
def run_model_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str = "auto",
) -> torch.Tensor:
    """
    Execute model inference on a deployment device.
    """
    if not isinstance(
        input_tensor,
        torch.Tensor,
    ):
        raise TypeError(
            "input_tensor must be a torch.Tensor."
        )

    target_device = resolve_device(device)

    model = model.to(target_device)
    model.eval()

    input_tensor = input_tensor.to(
        target_device,
        non_blocking=True,
    )

    return model(input_tensor)


def benchmark_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str = "auto",
    iterations: int = 10,
    warmup_iterations: int = 3,
) -> InferenceTiming:
    """
    Benchmark model inference latency.
    """
    if iterations <= 0:
        raise ValueError(
            "iterations must be positive."
        )

    if warmup_iterations < 0:
        raise ValueError(
            "warmup_iterations cannot be negative."
        )

    target_device = resolve_device(device)

    model = model.to(target_device)
    model.eval()

    input_tensor = input_tensor.to(
        target_device,
        non_blocking=True,
    )

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(input_tensor)

        if target_device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(iterations):
            model(input_tensor)

        if target_device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter() - start
        )

    total_images = (
        iterations * input_tensor.shape[0]
    )

    images_per_second = (
        total_images / elapsed
        if elapsed > 0
        else 0.0
    )

    return InferenceTiming(
        elapsed_seconds=elapsed,
        images_per_second=images_per_second,
    )


def optimize_model(
    model: torch.nn.Module,
    device: str = "auto",
) -> torch.nn.Module:
    """
    Apply lightweight inference optimizations.

    CUDA models use channels-last memory format when possible.
    """
    target_device = resolve_device(device)

    model = model.to(target_device)
    model.eval()

    if target_device.type == "cuda":
        model = model.to(
            memory_format=torch.channels_last
        )

    return model


def validate_input_tensor(
    input_tensor: torch.Tensor,
    expected_channels: int = 3,
    image_size: Optional[int] = 512,
) -> None:
    """
    Validate a model input tensor.
    """
    if not isinstance(
        input_tensor,
        torch.Tensor,
    ):
        raise TypeError(
            "input_tensor must be a torch.Tensor."
        )

    if input_tensor.ndim != 4:
        raise ValueError(
            "Expected input shape [B,C,H,W]."
        )

    if input_tensor.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} input channels, "
            f"got {input_tensor.shape[1]}."
        )

    if image_size is not None:
        if (
            input_tensor.shape[2] != image_size
            or input_tensor.shape[3] != image_size
        ):
            raise ValueError(
                f"Expected image size "
                f"{image_size}x{image_size}, "
                f"got "
                f"{input_tensor.shape[2]}x"
                f"{input_tensor.shape[3]}."
            )

    if not torch.isfinite(
        input_tensor
    ).all():
        raise ValueError(
            "Input tensor contains NaN or infinite values."
        )


def save_deployment_config(
    config: DeploymentConfig,
    output_path: Union[str, Path],
) -> Path:
    """
    Save deployment configuration as JSON.
    """
    validate_deployment_config(config)

    path = Path(output_path)

    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(config),
            indent=4,
        ),
        encoding="utf-8",
    )

    return path


def load_deployment_config(
    input_path: Union[str, Path],
) -> DeploymentConfig:
    """
    Load deployment configuration from JSON.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    config = DeploymentConfig(
        **data
    )

    return validate_deployment_config(
        config
    )


def runtime_info_to_dict(
    runtime_info: RuntimeInfo,
) -> Dict[str, Any]:
    """
    Convert runtime information to a dictionary.
    """
    return asdict(runtime_info)


def timing_to_dict(
    timing: InferenceTiming,
) -> Dict[str, float]:
    """
    Convert inference timing to a dictionary.
    """
    return asdict(timing)


def create_deployment_manifest(
    config: DeploymentConfig,
    model_name: str = "RoadXAIUNet",
) -> Dict[str, Any]:
    """
    Create a deployment manifest containing model and runtime metadata.
    """
    validate_deployment_config(config)

    runtime = get_runtime_info(
        config.device
    )

    return {
        "application": "RoadXAI",
        "model": model_name,
        "version": "1.0.0",
        "configuration": asdict(config),
        "runtime": runtime_info_to_dict(
            runtime
        ),
    }


def save_deployment_manifest(
    manifest: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Save a deployment manifest as JSON.
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
        ),
        encoding="utf-8",
    )

    return path


def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of optional deployment dependencies.
    """
    dependencies = [
        "torch",
        "numpy",
        "PIL",
        "streamlit",
        "plotly",
        "cv2",
        "reportlab",
    ]

    result: Dict[str, bool] = {}

    for dependency in dependencies:
        try:
            __import__(dependency)
            result[dependency] = True
        except ImportError:
            result[dependency] = False

    return result


def deployment_health_check(
    model: Optional[torch.nn.Module] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Run a basic deployment health check.
    """
    resolved_device = resolve_device(device)

    health: Dict[str, Any] = {
        "status": "healthy",
        "device": str(resolved_device),
        "torch_version": torch.__version__,
        "dependencies": check_dependencies(),
    }

    if model is not None:
        try:
            model_device = next(
                model.parameters()
            ).device

            health["model_device"] = str(
                model_device
            )

            health["model_training"] = bool(
                model.training
            )

            health["model_loaded"] = True

        except StopIteration:
            health["model_device"] = "none"
            health["model_training"] = False
            health["model_loaded"] = True

    else:
        health["model_loaded"] = False

    missing_dependencies = [
        name
        for name, available
        in health["dependencies"].items()
        if not available
    ]

    health["missing_optional_dependencies"] = (
        missing_dependencies
    )

    return health