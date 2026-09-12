"""
RoadXAI Project Overview
========================

Defines the public interface for the project overview module.

RoadXAI is an explainable road infrastructure intelligence system
designed to analyze road images, identify road damage, localize the
damage using segmentation, estimate engineering-related properties,
assess severity, and produce structured inspection information.

This package contains project-level metadata and configuration only.
It does not contain model training, dataset loading, or inference logic.
"""

from .module import (
    PROJECT_NAME,
    PROJECT_VERSION,
    PROJECT_DESCRIPTION,
    PROJECT_OBJECTIVES,
    DAMAGE_CLASSES,
)

__all__ = [
    "PROJECT_NAME",
    "PROJECT_VERSION",
    "PROJECT_DESCRIPTION",
    "PROJECT_OBJECTIVES",
    "DAMAGE_CLASSES",
]