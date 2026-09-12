"""
RoadXAI Project Schema
======================

Defines the standard structure of project-level information used
throughout RoadXAI.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ProjectSchema:
    """
    Immutable representation of the RoadXAI project configuration.
    """

    name: str
    version: str
    description: str
    damage_classes: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)


def create_project_schema() -> ProjectSchema:
    """
    Create the standard RoadXAI project schema.

    Returns
    -------
    ProjectSchema
        Structured project information.
    """

    return ProjectSchema(
        name="RoadXAI",
        version="0.1.0",
        description=(
            "An explainable AI-based road infrastructure intelligence "
            "system for detecting, segmenting, analyzing, and reporting "
            "road damage from images."
        ),
        damage_classes=[
            "crack",
            "pothole",
        ],
        objectives=[
            "Detect road damage from road images.",
            "Classify different types of road damage.",
            "Localize road damage using image segmentation.",
            "Estimate damage severity.",
            "Provide explainable AI outputs for model predictions.",
            "Extract structured information from visual road inspections.",
            "Support engineering-oriented road condition assessment.",
            "Generate consistent and interpretable inspection reports.",
        ],
    )


def validate_project_schema(schema: ProjectSchema) -> bool:
    """
    Validate a ProjectSchema instance.

    Parameters
    ----------
    schema : ProjectSchema
        Project schema to validate.

    Returns
    -------
    bool
        True if the schema is valid.
    """

    if not schema.name.strip():
        return False

    if not schema.version.strip():
        return False

    if not schema.description.strip():
        return False

    if not schema.damage_classes:
        return False

    if not schema.objectives:
        return False

    if any(not item.strip() for item in schema.damage_classes):
        return False

    if any(not item.strip() for item in schema.objectives):
        return False

    return True


if __name__ == "__main__":
    project = create_project_schema()

    print(f"Project: {project.name}")
    print(f"Version: {project.version}")
    print(f"Classes: {project.damage_classes}")
    print(f"Valid: {validate_project_schema(project)}")