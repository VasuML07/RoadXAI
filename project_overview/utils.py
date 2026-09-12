"""
RoadXAI Project Overview Utilities

Utility functions for displaying and validating RoadXAI project
metadata.

This module does not contain ML, dataset, training, inference,
segmentation, or engineering-analysis logic.
"""

from .module import (
    DAMAGE_CLASSES,
    PROJECT_DESCRIPTION,
    PROJECT_NAME,
    PROJECT_OBJECTIVES,
    PROJECT_VERSION,
)


def print_project_overview() -> None:
    
    #Print a readable overview of the RoadXAI project.

    print("=" * 70)
    print(f"{PROJECT_NAME} - PROJECT OVERVIEW")
    print("=" * 70)

    print(f"\nVersion: {PROJECT_VERSION}")

    print("\nDescription:")
    print(PROJECT_DESCRIPTION)

    print("\nDamage Classes:")
    for index, damage_class in enumerate(DAMAGE_CLASSES, start=1):
        print(f"  {index}. {damage_class}")

    print("\nObjectives:")
    for index, objective in enumerate(PROJECT_OBJECTIVES, start=1):
        print(f"  {index}. {objective}")

    print("=" * 70)


def validate_project_metadata() -> bool:
    
    #Validate the basic project metadata.

    if not PROJECT_NAME.strip():
        return False

    if not PROJECT_VERSION.strip():
        return False

    if not PROJECT_DESCRIPTION.strip():
        return False

    if not PROJECT_OBJECTIVES:
        return False

    if not DAMAGE_CLASSES:
        return False

    if any(not objective.strip() for objective in PROJECT_OBJECTIVES):
        return False

    if any(not damage_class.strip() for damage_class in DAMAGE_CLASSES):
        return False

    return True


def get_project_summary() -> dict:

    return {
        "name": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "damage_class_count": len(DAMAGE_CLASSES),
        "objective_count": len(PROJECT_OBJECTIVES),
    }


if __name__ == "__main__":
    print_project_overview()

    print("\nMetadata Validation:")

    if validate_project_metadata():
        print("PASS")
    else:
        print("FAIL")