"""
RoadXAI GitHub Utilities.

Helper functions for repository validation, Git configuration,
source-file inspection, and GitHub publication preparation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union


def run_git_command(
    project_root: Union[str, Path],
    *arguments: str,
) -> str:
    """
    Execute a Git command inside the project repository.
    """
    root = Path(project_root)

    if not root.exists():
        raise FileNotFoundError(
            f"Project directory not found: {root}"
        )

    command = ["git", *arguments]

    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Git is not installed or is not available in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Git command failed: {' '.join(command)}\n"
            f"{exc.stderr.strip()}"
        ) from exc

    return result.stdout.strip()


def git_is_initialized(
    project_root: Union[str, Path],
) -> bool:
    """
    Check whether a directory is a Git repository.
    """
    root = Path(project_root)

    if not (root / ".git").exists():
        return False

    try:
        run_git_command(
            root,
            "rev-parse",
            "--is-inside-work-tree",
        )
        return True
    except RuntimeError:
        return False


def initialize_git_repository(
    project_root: Union[str, Path],
) -> str:
    """
    Initialize a Git repository if necessary.
    """
    root = Path(project_root)

    if not root.exists():
        raise FileNotFoundError(
            f"Project directory not found: {root}"
        )

    if git_is_initialized(root):
        return "Git repository already initialized."

    return run_git_command(
        root,
        "init",
    )


def get_git_status(
    project_root: Union[str, Path],
) -> Dict[str, object]:
    """
    Return basic Git repository status information.
    """
    root = Path(project_root)

    initialized = git_is_initialized(root)

    if not initialized:
        return {
            "initialized": False,
            "branch": None,
            "clean": False,
            "status": [],
        }

    branch = run_git_command(
        root,
        "branch",
        "--show-current",
    )

    status_output = run_git_command(
        root,
        "status",
        "--short",
    )

    status_lines = (
        status_output.splitlines()
        if status_output
        else []
    )

    return {
        "initialized": True,
        "branch": branch or None,
        "clean": len(status_lines) == 0,
        "status": status_lines,
    }


def get_current_branch(
    project_root: Union[str, Path],
) -> Optional[str]:
    """
    Return the currently checked-out Git branch.
    """
    if not git_is_initialized(project_root):
        return None

    branch = run_git_command(
        project_root,
        "branch",
        "--show-current",
    )

    return branch or None


def create_branch(
    project_root: Union[str, Path],
    branch_name: str,
) -> str:
    """
    Create a new Git branch.
    """
    branch_name = branch_name.strip()

    if not branch_name:
        raise ValueError(
            "branch_name cannot be empty."
        )

    return run_git_command(
        project_root,
        "switch",
        "-c",
        branch_name,
    )


def stage_files(
    project_root: Union[str, Path],
    files: Optional[List[str]] = None,
) -> str:
    """
    Stage selected files or all project changes.
    """
    if files is None:
        return run_git_command(
            project_root,
            "add",
            ".",
        )

    if not files:
        raise ValueError(
            "files cannot be empty."
        )

    return run_git_command(
        project_root,
        "add",
        "--",
        *files,
    )


def create_commit(
    project_root: Union[str, Path],
    message: str,
) -> str:
    """
    Create a Git commit.
    """
    message = message.strip()

    if not message:
        raise ValueError(
            "Commit message cannot be empty."
        )

    return run_git_command(
        project_root,
        "commit",
        "-m",
        message,
    )


def get_remote_urls(
    project_root: Union[str, Path],
) -> List[str]:
    """
    Return configured Git remote URLs.
    """
    if not git_is_initialized(project_root):
        return []

    output = run_git_command(
        project_root,
        "remote",
        "-v",
    )

    urls = []

    for line in output.splitlines():
        parts = line.split()

        if len(parts) >= 2:
            url = parts[1]

            if url not in urls:
                urls.append(url)

    return urls


def add_remote(
    project_root: Union[str, Path],
    remote_name: str,
    remote_url: str,
) -> str:
    """
    Add a Git remote.
    """
    remote_name = remote_name.strip()
    remote_url = remote_url.strip()

    if not remote_name:
        raise ValueError(
            "remote_name cannot be empty."
        )

    if not remote_url:
        raise ValueError(
            "remote_url cannot be empty."
        )

    return run_git_command(
        project_root,
        "remote",
        "add",
        remote_name,
        remote_url,
    )


def set_git_config(
    project_root: Union[str, Path],
    key: str,
    value: str,
    global_config: bool = False,
) -> str:
    """
    Set a Git configuration value.
    """
    key = key.strip()
    value = value.strip()

    if not key:
        raise ValueError(
            "Git config key cannot be empty."
        )

    if global_config:
        return run_git_command(
            project_root,
            "config",
            "--global",
            key,
            value,
        )

    return run_git_command(
        project_root,
        "config",
        key,
        value,
    )


def check_required_repository_files(
    project_root: Union[str, Path],
) -> Dict[str, bool]:
    """
    Check files required for publishing the RoadXAI repository.
    """
    root = Path(project_root)

    required_files = [
        "requirements.txt",
        ".gitignore",
    ]

    return {
        filename: (
            root / filename
        ).is_file()
        for filename in required_files
    }


def check_package_structure(
    project_root: Union[str, Path],
    packages: Optional[List[str]] = None,
) -> Dict[str, Dict[str, bool]]:
    """
    Check the standard RoadXAI package structure.
    """
    root = Path(project_root)

    if packages is None:
        packages = [
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

    result: Dict[str, Dict[str, bool]] = {}

    for package in packages:
        package_root = root / package

        result[package] = {
            "__init__.py": (
                package_root / "__init__.py"
            ).is_file(),
            "module.py": (
                package_root / "module.py"
            ).is_file(),
            "utils.py": (
                package_root / "utils.py"
            ).is_file(),
        }

    return result


def repository_validation_report(
    project_root: Union[str, Path],
) -> Dict[str, object]:
    """
    Create a complete GitHub publication-readiness report.
    """
    required_files = (
        check_required_repository_files(
            project_root
        )
    )

    package_structure = (
        check_package_structure(
            project_root
        )
    )

    git_status = get_git_status(
        project_root
    )

    files_ready = all(
        required_files.values()
    )

    packages_ready = all(
        all(package.values())
        for package in package_structure.values()
    )

    return {
        "required_files": required_files,
        "package_structure": package_structure,
        "git": git_status,
        "files_ready": files_ready,
        "packages_ready": packages_ready,
        "ready": (
            files_ready
            and packages_ready
        ),
    }


def get_tracked_files(
    project_root: Union[str, Path],
) -> List[str]:
    """
    Return files currently tracked by Git.
    """
    if not git_is_initialized(project_root):
        return []

    output = run_git_command(
        project_root,
        "ls-files",
    )

    if not output:
        return []

    return output.splitlines()


def get_untracked_files(
    project_root: Union[str, Path],
) -> List[str]:
    """
    Return untracked files that are not ignored.
    """
    if not git_is_initialized(project_root):
        return []

    output = run_git_command(
        project_root,
        "ls-files",
        "--others",
        "--exclude-standard",
    )

    if not output:
        return []

    return output.splitlines()


def check_gitignore(
    project_root: Union[str, Path],
) -> bool:
    """
    Check whether .gitignore exists and is non-empty.
    """
    path = Path(project_root) / ".gitignore"

    if not path.is_file():
        return False

    return bool(
        path.read_text(
            encoding="utf-8"
        ).strip()
    )


def repository_statistics(
    project_root: Union[str, Path],
) -> Dict[str, int]:
    """
    Calculate basic Git repository statistics.
    """
    tracked = get_tracked_files(
        project_root
    )

    untracked = get_untracked_files(
        project_root
    )

    python_files = [
        file
        for file in tracked
        if file.endswith(".py")
    ]

    return {
        "tracked_files": len(tracked),
        "untracked_files": len(untracked),
        "tracked_python_files": len(
            python_files
        ),
    }


def create_github_checklist(
    project_root: Union[str, Path],
) -> Dict[str, bool]:
    """
    Create a final GitHub publication checklist.
    """
    root = Path(project_root)

    return {
        "requirements.txt": (
            root / "requirements.txt"
        ).is_file(),
        ".gitignore": (
            root / ".gitignore"
        ).is_file(),
        "git_initialized": (
            git_is_initialized(root)
        ),
        "package_structure": all(
            all(package.values())
            for package in check_package_structure(
                root
            ).values()
        ),
        "gitignore_nonempty": (
            check_gitignore(root)
        ),
    }