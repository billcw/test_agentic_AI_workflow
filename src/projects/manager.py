"""
projects/manager.py - Project workspace manager.

A "project" is just a named workspace folder under workspaces_root.
Each project has its own ChromaDB vectors, BM25 index, SQLite metadata,
and chat history — completely isolated from other projects.

This module handles: listing existing projects, creating new ones,
and checking whether a project exists. Think of it like a file manager
for project folders, but one that also initializes the right structure.
"""

from pathlib import Path
from src.config import PATHS


def get_workspaces_root() -> Path:
    """Return the root folder where all project workspaces live."""
    root = Path(PATHS["workspaces_root"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_projects() -> list[dict]:
    """
    List all existing project workspaces.

    Returns a list of dicts with project name and basic stats.
    A valid project folder must exist under workspaces_root.
    """
    root = get_workspaces_root()
    projects = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        # Count ingested documents via metadata.db presence
        has_metadata = (folder / "metadata.db").exists()
        has_vectors = (folder / "vectors").exists()
        has_bm25 = (folder / "bm25_index").exists()

        projects.append({
            "name": folder.name,
            "path": str(folder),
            "has_metadata": has_metadata,
            "has_vectors": has_vectors,
            "has_bm25": has_bm25,
            "ready": has_metadata and has_vectors and has_bm25
        })

    return projects


def project_exists(project_name: str) -> bool:
    """Check whether a project workspace folder exists."""
    root = get_workspaces_root()
    return (root / project_name).is_dir()


def create_project(project_name: str) -> dict:
    """
    Create a new project workspace with the required folder structure.

    Does nothing if the project already exists — safe to call repeatedly.

    Returns a dict with status and the project path.
    """
    # Sanitize name: lowercase, hyphens only, no spaces or special chars
    safe_name = project_name.strip().lower()
    safe_name = "".join(c if c.isalnum() or c == "-" else "-"
                        for c in safe_name)
    safe_name = safe_name.strip("-")

    if not safe_name:
        return {"status": "error", "message": "Invalid project name."}

    root = get_workspaces_root()
    project_path = root / safe_name

    if project_path.exists():
        return {
            "status": "exists",
            "message": f"Project '{safe_name}' already exists.",
            "path": str(project_path),
            "name": safe_name
        }

    # Create the required subfolders
    (project_path / "vectors").mkdir(parents=True, exist_ok=True)
    (project_path / "bm25_index").mkdir(parents=True, exist_ok=True)
    (project_path / "documents").mkdir(parents=True, exist_ok=True)

    return {
        "status": "created",
        "message": f"Project '{safe_name}' created successfully.",
        "path": str(project_path),
        "name": safe_name
    }


def delete_project(project_name: str) -> dict:
    """
    Delete a project workspace and all its data.
    This is irreversible — use with caution.
    """
    import shutil
    root = get_workspaces_root()
    project_path = root / project_name

    if not project_path.exists():
        return {"status": "error",
                "message": f"Project '{project_name}' not found."}

    shutil.rmtree(str(project_path))
    return {"status": "deleted",
            "message": f"Project '{project_name}' deleted."}
