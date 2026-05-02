"""
projects/drive_sync.py - Workspace sync between internal NVMe and external drives.

Philosophy:
  The external ai-models drive (/mnt/ai-models/workspaces/) is the VAULT —
  permanent authoritative storage for all project workspaces.
  The internal NVMe (workspaces/ under project root) is the DESK —
  fast working copy of whichever project is currently active.

  On project switch:
    1. rsync current workspace DESK -> VAULT (save it back)
    2. rsync incoming workspace VAULT -> DESK (pull it in)

  On backup:
    rsync current workspace DESK -> /mnt/backup-drive/workspaces/ (on-demand snapshot)

  rsync is used for all transfers:
    - Only changed files are copied (delta sync — fast after first push)
    - Transfer is atomic per-file — a failed sync never corrupts the destination
    - --delete keeps destination in sync (removes files deleted from source)

Drive mount points (must match your fstab / manual mount commands):
  Primary (ai-models): /mnt/ai-models
  Backup:              /mnt/backup-drive
"""

import subprocess
from pathlib import Path
from src.config import PATHS


# --- Constants ---

PRIMARY_DRIVE_MOUNT = Path("/mnt/ai-models")
BACKUP_DRIVE_MOUNT  = Path("/mnt/backup-drive")

PRIMARY_WORKSPACES  = PRIMARY_DRIVE_MOUNT / "workspaces"
BACKUP_WORKSPACES   = BACKUP_DRIVE_MOUNT  / "workspaces"


# --- Mount Checking ---

def is_mounted(mount_point: Path) -> bool:
    """
    Check whether a drive is actually mounted at the given path.

    We check two things:
      1. The directory exists
      2. It is a mount point (i.e. a real device is mounted there,
         not just an empty folder sitting on the NVMe)

    Why not just check if the folder exists?
    Because /mnt/ai-models is always present as a directory —
    it was created when you set up the mount point. An empty folder
    existing does NOT mean the drive is plugged in and mounted.
    Path.is_mount() is the correct check.
    """
    return mount_point.exists() and mount_point.is_mount()


def check_primary_drive() -> dict:
    """
    Return status of the ai-models (primary) drive.
    Raises a clear error dict if not mounted.
    """
    if not is_mounted(PRIMARY_DRIVE_MOUNT):
        return {
            "ok": False,
            "error": (
                f"Primary drive not mounted at {PRIMARY_DRIVE_MOUNT}. "
                f"Run: sudo mount /dev/sda1 /mnt/ai-models"
            )
        }
    return {"ok": True}


def check_backup_drive() -> dict:
    """
    Return status of the backup drive.
    Raises a clear error dict if not mounted.
    """
    if not is_mounted(BACKUP_DRIVE_MOUNT):
        return {
            "ok": False,
            "error": (
                f"Backup drive not mounted at {BACKUP_DRIVE_MOUNT}. "
                f"Run: sudo mount /dev/sdc1 /mnt/backup-drive"
            )
        }
    return {"ok": True}


# --- Core rsync Helper ---

def _rsync(source: Path, destination: Path) -> dict:
    """
    Run rsync from source to destination directory.

    Flags used:
      -a  archive mode: preserves permissions, timestamps, symlinks,
          owner, group — equivalent to -rlptgoD
      -v  verbose: logs each file transferred
      --delete  removes files in destination that no longer exist in source
                (keeps the two copies truly in sync, not just additive)

    The trailing slash on source/ is intentional and critical:
      rsync source/  dest/   -> copies CONTENTS of source into dest
      rsync source   dest/   -> copies source FOLDER ITSELF into dest
                                (creates dest/source/ — NOT what we want)

    Think of it like: "copy everything inside source into destination."
    """
    destination.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync",
        "-av",
        "--delete",
        f"{source}/",       # trailing slash = copy contents, not folder
        f"{destination}/"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600       # same timeout philosophy as the rest of the project
        )

        if result.returncode != 0:
            return {
                "ok": False,
                "error": f"rsync failed (exit {result.returncode}): {result.stderr.strip()}"
            }

        # Count transferred files from rsync output
        lines = [l for l in result.stdout.splitlines()
                 if l and not l.startswith("sending") and not l.startswith("sent")]
        transferred = len([l for l in lines if not l.startswith(".") and "/" not in l or True])

        return {
            "ok": True,
            "source": str(source),
            "destination": str(destination),
            "output": result.stdout.strip()
        }

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "rsync timed out after 3600 seconds."}
    except FileNotFoundError:
        return {"ok": False, "error": "rsync not found. Install with: sudo apt install rsync"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# --- Public API ---

def save_workspace_to_primary(project_name: str) -> dict:
    """
    Push the local (NVMe) workspace to the primary ai-models drive.

    DESK -> VAULT

    Called automatically before switching away from a project.
    Safe to call manually at any time.
    """
    drive_check = check_primary_drive()
    if not drive_check["ok"]:
        return drive_check

    local_path = Path(PATHS["workspaces_root"]) / project_name
    if not local_path.exists():
        return {
            "ok": False,
            "error": f"Local workspace '{project_name}' not found at {local_path}"
        }

    remote_path = PRIMARY_WORKSPACES / project_name
    result = _rsync(local_path, remote_path)
    if result["ok"]:
        result["message"] = (
            f"Workspace '{project_name}' saved to primary drive."
        )
    return result


def load_workspace_from_primary(project_name: str) -> dict:
    """
    Pull a workspace from the primary ai-models drive to local NVMe.

    VAULT -> DESK

    Called automatically when switching to a project.
    If the workspace doesn't exist on the drive yet, returns an error.
    """
    drive_check = check_primary_drive()
    if not drive_check["ok"]:
        return drive_check

    remote_path = PRIMARY_WORKSPACES / project_name
    if not remote_path.exists():
        return {
            "ok": False,
            "error": (
                f"Workspace '{project_name}' not found on primary drive "
                f"at {remote_path}. It may not have been saved there yet."
            )
        }

    local_path = Path(PATHS["workspaces_root"]) / project_name
    result = _rsync(remote_path, local_path)
    if result["ok"]:
        result["message"] = (
            f"Workspace '{project_name}' loaded from primary drive."
        )
    return result


def backup_workspace_to_backup_drive(project_name: str) -> dict:
    """
    Copy the local (NVMe) workspace to the backup drive.

    DESK -> BACKUP

    On-demand only. Called by the Backup button in the UI.
    Does NOT require a project switch — backs up whatever is active right now.
    """
    drive_check = check_backup_drive()
    if not drive_check["ok"]:
        return drive_check

    local_path = Path(PATHS["workspaces_root"]) / project_name
    if not local_path.exists():
        return {
            "ok": False,
            "error": f"Local workspace '{project_name}' not found at {local_path}"
        }

    backup_path = BACKUP_WORKSPACES / project_name
    result = _rsync(local_path, backup_path)
    if result["ok"]:
        result["message"] = (
            f"Workspace '{project_name}' backed up to backup drive."
        )
    return result


def sync_on_switch(from_project: str, to_project: str) -> dict:
    """
    Full project switch sync:
      1. Save 'from_project' workspace to primary drive
      2. Load 'to_project' workspace from primary drive

    Both steps must succeed. If step 1 fails, the switch is blocked —
    we never abandon unsaved work.

    If 'from_project' is None or empty (first load at startup), skip step 1.
    If 'to_project' does not exist on the drive yet (new project),
    skip step 2 — it will be pushed on the next switch away.

    Returns a dict with 'ok', 'save_result', and 'load_result' keys.
    """
    save_result = None
    load_result = None

    # Step 1: Save current project (skip if no current project)
    if from_project:
        save_result = save_workspace_to_primary(from_project)
        if not save_result["ok"]:
            return {
                "ok": False,
                "step": "save",
                "save_result": save_result,
                "load_result": None,
                "error": f"Could not save '{from_project}' before switching: {save_result['error']}"
            }

    # Step 2: Load incoming project (skip if not on drive yet — new project)
    remote_path = PRIMARY_WORKSPACES / to_project
    if remote_path.exists():
        load_result = load_workspace_from_primary(to_project)
        if not load_result["ok"]:
            return {
                "ok": False,
                "step": "load",
                "save_result": save_result,
                "load_result": load_result,
                "error": f"Could not load '{to_project}' from drive: {load_result['error']}"
            }
    else:
        load_result = {
            "ok": True,
            "message": f"'{to_project}' is new — no drive copy yet. Will be saved on next switch."
        }

    return {
        "ok": True,
        "save_result": save_result,
        "load_result": load_result,
        "message": f"Switched from '{from_project}' to '{to_project}' successfully."
    }


def list_drive_workspaces() -> list[str]:
    """
    Return names of all workspaces currently stored on the primary drive.
    Used by the UI to show which projects have a drive copy.
    """
    if not is_mounted(PRIMARY_DRIVE_MOUNT):
        return []
    if not PRIMARY_WORKSPACES.exists():
        return []
    return sorted([
        d.name for d in PRIMARY_WORKSPACES.iterdir() if d.is_dir()
    ])
