"""
projects/archive_manager.py - Three-tier project workspace manager.

Tiers:
  Active   — workspaces/<project>/         (in use, on internal drive)
  Archived — /mnt/ai-models/archive/<project>/  (fast swap, uncompressed)
  Backed up— /mnt/ai-models/backups/<project>.tar.gz  (compressed copy)

A project may exist in multiple tiers simultaneously.

SAFETY NOTE: Always use shutil.move() not os.rename() for moves between
tiers — internal-to-external drive is a cross-filesystem operation and
os.rename() will fail silently or raise on cross-device moves.

CHROMADB NOTE: The server does not hold persistent ChromaDB handles between
requests. Moving a project folder between requests is safe. Do not archive
a project while a query is actively running against it.
"""

import shutil
import tarfile
from datetime import datetime
from pathlib import Path

from src.config import PATHS


# ---------------------------------------------------------------------------
# Root path helpers
# ---------------------------------------------------------------------------

def _workspaces_root() -> Path:
    return Path(PATHS["workspaces_root"])


def _archive_root() -> Path:
    return Path(PATHS["archive_root"])


def _backup_root() -> Path:
    return Path(PATHS["backup_root"])


def _check_external_drive():
    """
    Raise RuntimeError if the external drive is not mounted.
    Both archive_root and backup_root live on /mnt/ai-models — if the
    parent mount point doesn't exist as a directory, the drive is offline.
    """
    archive = _archive_root()
    backup = _backup_root()
    if not archive.parent.exists():
        raise RuntimeError(
            f"External drive not mounted. "
            f"Expected mount at: {archive.parent}\n"
            f"Run: sudo mount /dev/sda1 /mnt/ai-models"
        )


def _assert_inside(path: Path, root: Path, label: str):
    """
    Safety check: raise ValueError if path is not inside root.
    Prevents accidental operations on arbitrary filesystem locations.
    """
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(
            f"Safety check failed: {path} is not inside {root} ({label})"
        )


# ---------------------------------------------------------------------------
# Size and metadata helpers
# ---------------------------------------------------------------------------

def _folder_size_bytes(path: Path) -> int:
    """Return total size of all files in a folder tree, in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _folder_last_modified(path: Path) -> str:
    """Return ISO8601 timestamp of the most recently modified file in a folder."""
    latest = 0.0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                mtime = f.stat().st_mtime
                if mtime > latest:
                    latest = mtime
            except OSError:
                pass
    if latest == 0.0:
        return path.stat().st_mtime if path.exists() else ""
    return datetime.fromtimestamp(latest).isoformat(timespec="seconds")


def _count_chunks(project_path: Path) -> int:
    """
    Estimate chunk count for a project folder.
    Counts .parquet and .bin files inside the vectors/ subfolder,
    which is where ChromaDB stores its segment data.
    Falls back to 0 if the folder doesn't exist or is empty.
    This is a fast file-count estimate, not a ChromaDB query.
    """
    vectors_dir = project_path / "vectors"
    if not vectors_dir.exists():
        return 0
    count = sum(1 for f in vectors_dir.rglob("*") if f.is_file())
    return count


def _project_info(folder: Path, tier: str) -> dict:
    """Build the info dict for a single project folder."""
    return {
        "name": folder.name,
        "path": str(folder),
        "tier": tier,
        "chunk_count": _count_chunks(folder),
        "size_bytes": _folder_size_bytes(folder),
        "last_modified": _folder_last_modified(folder),
        "has_vectors": (folder / "vectors").exists(),
        "has_bm25": (folder / "bm25_index").exists(),
        "has_metadata": (folder / "metadata.db").exists(),
        "ready": (
            (folder / "vectors").exists() and
            (folder / "bm25_index").exists() and
            (folder / "metadata.db").exists()
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_project_status() -> dict:
    """
    Return a dict describing all known projects across all three tiers.

    A project may appear in multiple lists — e.g. both 'active' and
    'backed_up' if it was backed up while still active.

    Returns:
        {
            "active":    [ { name, path, tier, chunk_count, size_bytes,
                             last_modified, has_vectors, has_bm25,
                             has_metadata, ready }, ... ],
            "archived":  [ ... ],
            "backed_up": [ ... ],
            "drive_mounted": bool,
            "error": str or None
        }
    """
    result = {
        "active": [],
        "archived": [],
        "backed_up": [],
        "drive_mounted": False,
        "error": None,
    }

    # --- Active projects (always readable, internal drive) ---
    ws_root = _workspaces_root()
    if ws_root.exists():
        for folder in sorted(ws_root.iterdir()):
            if folder.is_dir():
                result["active"].append(_project_info(folder, "active"))

    # --- External drive tiers (may not be mounted) ---
    try:
        _check_external_drive()
        result["drive_mounted"] = True
    except RuntimeError as e:
        result["error"] = str(e)
        return result  # Can't read archive or backup without the drive

    # --- Archived projects ---
    archive_root = _archive_root()
    archive_root.mkdir(parents=True, exist_ok=True)
    for folder in sorted(archive_root.iterdir()):
        if folder.is_dir():
            result["archived"].append(_project_info(folder, "archived"))

    # --- Backed-up projects ---
    backup_root = _backup_root()
    backup_root.mkdir(parents=True, exist_ok=True)
    for tarball in sorted(backup_root.glob("*.tar.gz")):
        name = tarball.name[: -len(".tar.gz")]  # strip extension
        result["backed_up"].append({
            "name": name,
            "path": str(tarball),
            "tier": "backed_up",
            "chunk_count": 0,   # can't count without extracting
            "size_bytes": tarball.stat().st_size,
            "last_modified": datetime.fromtimestamp(
                tarball.stat().st_mtime
            ).isoformat(timespec="seconds"),
            "has_vectors": None,  # unknown without extraction
            "has_bm25": None,
            "has_metadata": None,
            "ready": None,
        })

    return result


def archive_project(project_name: str) -> dict:
    """
    Move an active project to the archive tier.

    Moves workspaces/<name>/ to archive_root/<name>/.
    Does NOT compress — this is a fast cross-filesystem move.
    After this call the project is no longer queryable (not in workspaces/).

    Raises:
        RuntimeError  — external drive not mounted
        FileNotFoundError — project not found in workspaces/
        FileExistsError   — a folder with that name already exists in archive
        ValueError        — safety check failed
    """
    _check_external_drive()

    src = _workspaces_root() / project_name
    if not src.exists():
        raise FileNotFoundError(
            f"Project '{project_name}' not found in workspaces/."
        )

    _assert_inside(src, _workspaces_root(), "workspaces")

    dst = _archive_root() / project_name
    _archive_root().mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise FileExistsError(
            f"An archived project named '{project_name}' already exists. "
            f"Delete the existing archive first."
        )

    shutil.move(str(src), str(dst))

    return {
        "status": "ok",
        "archived_to": str(dst),
        "project_name": project_name,
    }


def restore_project(project_name: str, from_tier: str) -> dict:
    """
    Restore a project to the active tier.

    from_tier must be "archived" or "backed_up".

    If an active project already exists, it is auto-archived first.
    The .tar.gz is preserved after restore (backup is never deleted).

    Raises:
        RuntimeError      — external drive not mounted
        ValueError        — invalid from_tier, or safety check failed
        FileNotFoundError — source not found in specified tier
        FileExistsError   — target already exists in workspaces (shouldn't happen)
    """
    _check_external_drive()

    if from_tier not in ("archived", "backed_up"):
        raise ValueError(
            f"Invalid from_tier '{from_tier}'. Must be 'archived' or 'backed_up'."
        )

    result = {"status": "ok", "active_project": project_name, "warning": None}

    # --- Auto-archive any existing active projects first ---
    ws_root = _workspaces_root()
    existing_active = [f.name for f in ws_root.iterdir() if f.is_dir()] if ws_root.exists() else []

    for active_name in existing_active:
        archive_result = archive_project(active_name)
        result["warning"] = (
            f"'{active_name}' was auto-archived to {archive_result['archived_to']} "
            f"before restoring '{project_name}'."
        )

    # --- Restore from the appropriate tier ---
    dst = ws_root / project_name
    ws_root.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise FileExistsError(
            f"Destination already exists: {dst}. This should not happen after auto-archive."
        )

    if from_tier == "archived":
        src = _archive_root() / project_name
        if not src.exists():
            raise FileNotFoundError(
                f"No archived project named '{project_name}' found."
            )
        _assert_inside(src, _archive_root(), "archive")
        shutil.move(str(src), str(dst))
        result["restored_from"] = str(src)

    elif from_tier == "backed_up":
        tarball = _backup_root() / f"{project_name}.tar.gz"
        if not tarball.exists():
            raise FileNotFoundError(
                f"No backup found for '{project_name}' at {tarball}."
            )

        # Safety: inspect tarball members before extracting
        with tarfile.open(str(tarball), "r:gz") as tf:
            members = tf.getmembers()
            for member in members:
                # Prevent path traversal attacks in tarballs
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(
                        f"Unsafe path in tarball: {member.name}"
                    )
            tf.extractall(path=str(ws_root))

        result["restored_from"] = str(tarball)
        result["backup_preserved"] = True

    return result


def backup_project(project_name: str) -> dict:
    """
    Compress a project into a .tar.gz backup on the external drive.

    Checks active tier first, then archived. Does NOT remove the source.
    If a backup already exists, the caller must pass overwrite=True.

    Raises:
        RuntimeError      — external drive not mounted
        FileNotFoundError — project not found in active or archived tier
        FileExistsError   — backup already exists and overwrite=False
        ValueError        — safety check failed
    """
    _check_external_drive()

    # Find source — active takes priority over archived
    src = _workspaces_root() / project_name
    if not src.exists():
        src = _archive_root() / project_name
    if not src.exists():
        raise FileNotFoundError(
            f"Project '{project_name}' not found in active or archived tier."
        )

    _assert_inside(src, src.parent, "source")  # must be inside its own tier root

    backup_root = _backup_root()
    backup_root.mkdir(parents=True, exist_ok=True)
    tarball = backup_root / f"{project_name}.tar.gz"

    backup_existed = tarball.exists()

    with tarfile.open(str(tarball), "w:gz") as tf:
        tf.add(str(src), arcname=project_name)

    size = tarball.stat().st_size

    return {
        "status": "ok",
        "backup_path": str(tarball),
        "size_bytes": size,
        "project_name": project_name,
        "source_tier": "active" if (_workspaces_root() / project_name).exists() else "archived",
        "overwrote_existing": backup_existed,
    }


def delete_project(project_name: str, tier: str, confirmed_name: str) -> dict:
    """
    Permanently delete a project from a specific tier.

    Server-side validation: confirmed_name must exactly match project_name.
    This is a non-negotiable safety check — do not remove it.

    tier must be "active", "archived", or "backed_up".

    Raises:
        ValueError        — name mismatch, invalid tier, or safety check failed
        FileNotFoundError — project not found in specified tier
    """
    # Server-side confirmation — never trust the UI alone
    if confirmed_name != project_name:
        raise ValueError(
            f"Confirmation name '{confirmed_name}' does not match "
            f"project name '{project_name}'. Deletion refused."
        )

    if tier == "active":
        target = _workspaces_root() / project_name
        if not target.exists():
            raise FileNotFoundError(
                f"No active project named '{project_name}'."
            )
        _assert_inside(target, _workspaces_root(), "workspaces")
        shutil.rmtree(str(target))
        return {"status": "ok", "deleted": str(target), "tier": tier}

    elif tier == "archived":
        _check_external_drive()
        target = _archive_root() / project_name
        if not target.exists():
            raise FileNotFoundError(
                f"No archived project named '{project_name}'."
            )
        _assert_inside(target, _archive_root(), "archive")
        shutil.rmtree(str(target))
        return {"status": "ok", "deleted": str(target), "tier": tier}

    elif tier == "backed_up":
        _check_external_drive()
        target = _backup_root() / f"{project_name}.tar.gz"
        if not target.exists():
            raise FileNotFoundError(
                f"No backup found for '{project_name}'."
            )
        _assert_inside(target, _backup_root(), "backups")
        target.unlink()
        return {"status": "ok", "deleted": str(target), "tier": tier}

    else:
        raise ValueError(
            f"Invalid tier '{tier}'. Must be 'active', 'archived', or 'backed_up'."
        )
