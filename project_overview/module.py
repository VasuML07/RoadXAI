"""
RoadXAI Project Overview

Core project metadata, objectives, and supported road-damage classes.

This module contains only static project-level information.
It does not perform model training, inference, dataset processing,
segmentation, explainability, or engineering calculations.
"""

# Project Identity

PROJECT_NAME = "RoadXAI"
PROJECT_VERSION = "0.1.0"

PROJECT_DESCRIPTION = (
    "An explainable AI-based road infrastructure intelligence system "
    "for detecting, segmenting, analyzing, and reporting road damage "
    "from images."
)

# Project Objectives

PROJECT_OBJECTIVES = [
    "Detect road damage from road images.",
    "Classify different types of road damage.",
    "Localize road damage using image segmentation.",
    "Estimate damage severity.",
    "Provide explainable AI outputs for model predictions.",
    "Extract structured information from visual road inspections.",
    "Support engineering-oriented road condition assessment.",
    "Generate consistent and interpretable inspection reports.",
]



# Supported Damage Classes

DAMAGE_CLASSES = [
    "crack",
    "pothole",
]


# Public Metadata

PROJECT_METADATA = {
    "name": PROJECT_NAME,
    "version": PROJECT_VERSION,
    "description": PROJECT_DESCRIPTION,
    "objectives": PROJECT_OBJECTIVES,
    "damage_classes": DAMAGE_CLASSES,
}


def get_project_info() -> dict:
    """
    Return the complete RoadXAI project metadata.

    Returns
    -------
    dict
        Project name, version, description, objectives,
        and supported damage classes.
    """
    return PROJECT_METADATA.copy()


def get_damage_classes() -> list[str]:
    """
    Return the supported road-damage classes.

    Returns
    -------
    list[str]
        List of supported damage class names.
    """
    return DAMAGE_CLASSES.copy()


def get_project_objectives() -> list[str]:
    """
    Return the project's objectives.

    Returns
    -------
    list[str]
        List of project objectives.
    """
    return PROJECT_OBJECTIVES.copy()


if __name__ == "__main__":
    print(f"Project: {PROJECT_NAME}")
    print(f"Version: {PROJECT_VERSION}")
    print(f"Description: {PROJECT_DESCRIPTION}")
    print("\nDamage Classes:")

    for index, damage_class in enumerate(DAMAGE_CLASSES, start=1):
        print(f"{index}. {damage_class}")

    print("\nObjectives:")

    for index, objective in enumerate(PROJECT_OBJECTIVES, start=1):
        print(f"{index}. {objective}")