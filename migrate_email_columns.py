#!/usr/bin/env python3
"""
migrate_email_columns.py
Adds email_date, email_sender, email_subject columns to the chunks
table in every existing project database.

Safe to run multiple times — uses ALTER TABLE only if the column
does not already exist. Existing data is never touched.

Why we need this:
initialize_db() uses CREATE TABLE IF NOT EXISTS, which means it
never modifies a table that already exists. So existing databases
don't get the new columns automatically — we have to add them
with ALTER TABLE.

Run this once after deploying the metadata_db.py patch.
"""

import sqlite3
from pathlib import Path
from src.config import PATHS


def migrate_project(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Find out which columns already exist
    cursor.execute("PRAGMA table_info(chunks)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    new_columns = {
        "email_date":    "TEXT",
        "email_sender":  "TEXT",
        "email_subject": "TEXT",
    }

    added = []
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_type}")
            added.append(col_name)

    conn.commit()
    conn.close()

    if added:
        print(f"  {db_path.parent.name}: added columns {added}")
    else:
        print(f"  {db_path.parent.name}: already up to date — no changes needed")


def main():
    workspaces_root = Path(PATHS["workspaces_root"])

    if not workspaces_root.exists():
        print(f"Workspaces root not found: {workspaces_root}")
        return

    db_files = list(workspaces_root.rglob("metadata.db"))

    if not db_files:
        print("No metadata.db files found — nothing to migrate.")
        return

    print(f"Found {len(db_files)} database(s) to migrate:\n")
    for db_path in db_files:
        migrate_project(db_path)

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
