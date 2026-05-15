#!/usr/bin/env python3
"""
reingest_pst.py
Re-ingests selected PST files into test-project with force=True.

force=True replaces existing chunk records so the new email_date,
email_sender, and email_subject columns get populated. Without force,
already-ingested files would be skipped and the columns would remain NULL.

Files deliberately excluded:
  - outlook_inbox_backup_102021-002.pst  (likely duplicate of -102021.pst)
  - back-sdcwin21h2p0228.pst             (265KB, likely small exported folder)
  - backup weekly report up to 12-30-24.pst  (265KB, likely small exported folder)
"""

from pathlib import Path
from src.ingestion.pipeline import ingest_file

PROJECT = "test-project"
PST_DIR = Path("/home/bill/local-ai-doc-assistant/docs/pst/pst")

FILES = [
    "4-20-26-backup.pst",
    "4-20-26-backup2.pst",
    "4-20-26-backup3.pst",
    "online_archive_050726-003.pst",
    "Online archive backup from outlook-001.pst",
    "outlook_inbox_backup_102021.pst",
]

def main():
    total = len(FILES)
    ingested = skipped = errors = 0

    print(f"\nRe-ingesting {total} PST files into project '{PROJECT}' with force=True")
    print(f"This will take several hours for large archives.\n")

    for i, filename in enumerate(FILES, 1):
        file_path = PST_DIR / filename

        if not file_path.exists():
            print(f"[{i}/{total}] MISSING: {filename}")
            errors += 1
            continue

        size_gb = file_path.stat().st_size / (1024 ** 3)
        print(f"[{i}/{total}] Starting: {filename} ({size_gb:.1f} GB)...")

        result = ingest_file(PROJECT, file_path, force=True)

        if result["status"] == "ingested":
            ingested += 1
            print(f"[{i}/{total}] OK: {filename} — {result['chunks']} chunks")
        elif result["status"] == "skipped":
            skipped += 1
            print(f"[{i}/{total}] SKIPPED: {filename} — {result['message']}")
        else:
            errors += 1
            print(f"[{i}/{total}] ERROR: {filename} — {result['message']}")

    print(f"\nRe-ingestion complete.")
    print(f"  Ingested: {ingested}")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {errors}")

if __name__ == "__main__":
    main()
