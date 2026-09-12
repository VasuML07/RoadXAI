"""
RoadXAI GitHub Module.

Utilities for validating the repository structure and preparing
project metadata for GitHub publication.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import json


@dataclass
class RepositoryCheck:
    """Result of a repository structure check."""

    path: str
    exists: bool
    is_directory: bool


@dataclass
class RepositorySummary:
    """Summary of the RoadXAI repository."""

    project_root: str
    required_directories: Dict[str, bool]
    required_files: Dict[str, bool]
    python_file_count: int
    total_file_count: int
    ready: bool


REQUIRED_DIRECTORIES = [
    "dataset_pipeline",
    "cnn_model",
    "training_pipeline",
    "evaluation",
    "explainable_ai",
    "engineering_analysis",
    "3d_visualization",
    "streamlit_app",
    "report_generation",
    "deployment",
    "github",
]

REQUIRED_FILES = [
    "requirements.txt",
    ".gitignore",
]


def find_project_root(
    start_path: Optional[str | Path] = None,
) -> Path:
    """
    Find the RoadXAI project root.

    Searches upward for a directory containing the core
    RoadXAI project folders.
    """
    if start_path is None:
        current = Path.cwd()
    else:
        current = Path(start_path).resolve()

    if current.is_file():
        current = current.parent

    markers = {
        "cnn_model",
        "dataset_pipeline",
        "training_pipeline",
    }

    for path in [current, *current.parents]:
        existing = {
            item.name
            for item in path.iterdir()
            if item.is_dir()
        }

        if len(markers.intersection(existing)) >= 2:
            return path

    return current


def check_path(
    path: str | Path,
) -> RepositoryCheck:
    """
    Check whether a repository path exists.
    """
    target = Path(path)

    return RepositoryCheck(
        path=str(target),
        exists=target.exists(),
        is_directory=target.is_dir(),
    )


def check_repository_structure(
    project_root: str | Path,
) -> Dict[str, List[RepositoryCheck]]:
    """
    Check required RoadXAI directories and files.
    """
    root = Path(project_root)

    directories = [
        check_path(root / directory)
        for directory in REQUIRED_DIRECTORIES
    ]

    files = [
        check_path(root / file)
        for file in REQUIRED_FILES
    ]

    return {
        "directories": directories,
        "files": files,
    }


def count_files(
    project_root: str | Path,
    extension: Optional[str] = None,
) -> int:
    """
    Count files inside the repository.

    Hidden Git metadata and virtual environments are excluded.
    """
    root = Path(project_root)

    excluded = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "datasets",
    }

    count = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(
            part in excluded
            for part in path.parts
        ):
            continue

        if extension is not None:
            normalized = extension.lower()

            if not normalized.startswith("."):
                normalized = "." + normalized

            if path.suffix.lower() != normalized:
                continue

        count += 1

    return count


def create_repository_summary(
    project_root: str | Path,
) -> RepositorySummary:
    """
    Create a complete repository structure summary.
    """
    root = Path(project_root)

    directory_status = {
        directory: (
            root / directory
        ).is_dir()
        for directory in REQUIRED_DIRECTORIES
    }

    file_status = {
        file: (
            root / file
        ).is_file()
        for file in REQUIRED_FILES
    }

    directories_ready = all(
        directory_status.values()
    )

    files_ready = all(
        file_status.values()
    )

    return RepositorySummary(
        project_root=str(root),
        required_directories=directory_status,
        required_files=file_status,
        python_file_count=count_files(
            root,
            ".py",
        ),
        total_file_count=count_files(
            root,
        ),
        ready=(
            directories_ready
            and files_ready
        ),
    )


def validate_python_packages(
    project_root: str | Path,
) -> Dict[str, bool]:
    """
    Verify that each RoadXAI package contains __init__.py.
    """
    root = Path(project_root)

    result: Dict[str, bool] = {}

    for package in REQUIRED_DIRECTORIES:
        package_path = root / package

        result[package] = (
            package_path.is_dir()
            and (
                package_path / "__init__.py"
            ).is_file()
        )

    return result


def find_python_files(
    project_root: str | Path,
) -> List[Path]:
    """
    Return all tracked-style Python source files.
    """
    root = Path(project_root)

    excluded = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "datasets",
    }

    files = []

    for path in root.rglob("*.py"):
        if any(
            part in excluded
            for part in path.parts
        ):
            continue

        files.append(path)

    return sorted(files)


def check_required_source_files(
    project_root: str | Path,
) -> Dict[str, bool]:
    """
    Check the core source files expected in each module.
    """
    root = Path(project_root)

    result: Dict[str, bool] = {}

    for package in REQUIRED_DIRECTORIES:
        package_path = root / package

        for filename in [
            "__init__.py",
            "module.py",
            "utils.py",
        ]:
            key = (
                f"{package}/{filename}"
            )

            result[key] = (
                package_path / filename
            ).is_file()

    return result


def repository_readiness(
    project_root: str | Path,
) -> Dict[str, object]:
    """
    Perform a complete repository readiness check.
    """
    root = Path(project_root)

    structure = check_repository_structure(
        root
    )

    packages = validate_python_packages(
        root
    )

    source_files = check_required_source_files(
        root
    )

    summary = create_repository_summary(
        root
    )

    return {
        "ready": (
            summary.ready
            and all(packages.values())
            and all(source_files.values())
        ),
        "summary": asdict(summary),
        "packages": packages,
        "source_files": source_files,
    }


def create_repository_metadata(
    project_root: str | Path,
    project_name: str = "RoadXAI",
    version: str = "1.0.0",
) -> Dict[str, object]:
    """
    Create repository metadata suitable for local project tooling.
    """
    root = Path(project_root)

    summary = create_repository_summary(
        root
    )

    return {
        "name": project_name,
        "version": version,
        "root": str(root),
        "python_files": summary.python_file_count,
        "total_files": summary.total_file_count,
        "ready": summary.ready,
    }


def save_repository_metadata(
    metadata: Dict[str, object],
    output_path: str | Path,
) -> Path:
    """
    Save repository metadata as JSON.
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
            metadata,
            indent=4,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return path


def get_git_directory(
    project_root: str | Path,
) -> Path:
    """
    Return the local .git directory path.
    """
    return Path(project_root) / ".git"


def is_git_repository(
    project_root: str | Path,
) -> bool:
    """
    Check whether the project is already initialized as Git.
    """
    return get_git_directory(
        project_root
    ).is_dir()


def get_repository_status(
    project_root: str | Path,
) -> Dict[str, object]:
    """
    Return basic local Git repository status information.
    """
    root = Path(project_root)

    return {
        "project_root": str(root),
        "git_initialized": is_git_repository(
            root
        ),
        "git_directory": str(
            get_git_directory(root)
        ),
        "gitignore_exists": (
            root / ".gitignore"
        ).is_file(),
        "requirements_exists": (
            root / "requirements.txt"
        ).is_file(),
    }