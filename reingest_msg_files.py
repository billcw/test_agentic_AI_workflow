"""
reingest_msg_files.py - Clean duplicate document rows and re-ingest
all .msg files in docs/temp/ so email metadata is populated correctly.

Why this is needed:
    .msg files were ingested before purge_document() existed. Some have
    duplicate rows in the documents table (from INSERT OR REPLACE creating
    new ids while old chunk rows remained under the original id). The
    pipeline's force=True purge logic now handles this correctly for
    future ingestions, but existing duplicates need manual cleanup first.

What this script does per .msg file:
    1. Find all document rows for this filename in SQLite
    2. Delete all associated chunk rows
    3. Delete all document rows for this filename
    4. Re-ingest via pipeline.ingest_file() with force=True

Run from the project root with the virtualenv active:
    python reingest_msg_files.py

The server must NOT be running during this script.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.pipeline import ingest_file
from src.storage.vector_store import get_or_create_collection

PROJECT_NAME   = "test-project"
WORKSPACE_ROOT = Path("workspaces")
DB_PATH        = WORKSPACE_ROOT / PROJECT_NAME / "metadata.db"
MSG_DIR        = Path("docs/temp")
CHROMA_BATCH   = 500


def clean_msg_from_sqlite(conn: sqlite3.Connection,
                          filename: str) -> list[str]:
    """
    Delete all document rows and chunk rows for a .msg filename.
    Returns list of chunk_ids deleted (for ChromaDB cleanup).
    """
    cur = conn.cursor()

    # Find all document ids for this filename (may be duplicates)
    cur.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
    doc_ids = [r[0] for r in cur.fetchall()]

    if not doc_ids:
        return []

    # Collect all chunk_ids across all document versions
    placeholders = ",".join("?" * len(doc_ids))
    cur.execute(
        f"SELECT chunk_id FROM chunks WHERE document_id IN ({placeholders})",
        doc_ids
    )
    chunk_ids = [r[0] for r in cur.fetchall()]

    # Also catch orphaned chunks by source name (safety net)
    cur.execute(
        "SELECT chunk_id FROM chunks WHERE source = ? AND chunk_id NOT IN "
        f"(SELECT chunk_id FROM chunks WHERE document_id IN ({placeholders}))",
        [filename] + doc_ids
    )
    orphans = [r[0] for r in cur.fetchall()]
    chunk_ids.extend(orphans)

    # Delete everything
    cur.execute(
        f"DELETE FROM chunks WHERE document_id IN ({placeholders})",
        doc_ids
    )
    cur.execute("DELETE FROM chunks WHERE source = ?", (filename,))
    cur.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()

    return chunk_ids


def delete_from_chroma(collection, chunk_ids: list[str]) -> None:
    """Delete chunk_ids from ChromaDB in batches."""
    if not chunk_ids:
        return
    for i in range(0, len(chunk_ids), CHROMA_BATCH):
        batch = chunk_ids[i:i + CHROMA_BATCH]
        try:
            collection.delete(ids=batch)
        except Exception as e:
            print(f"    WARNING: ChromaDB delete failed for batch: {e}")


def main():
    print("=== MSG File Re-ingest: Email Metadata Fix ===")
    print(f"Project:  {PROJECT_NAME}")
    print(f"MSG dir:  {MSG_DIR}")
    print()

    # Find all .msg files
    msg_files = sorted(MSG_DIR.glob("*.msg"))
    print(f"Found {len(msg_files)} .msg files")
    print()

    if not msg_files:
        print("No .msg files found — nothing to do.")
        return

    # Open connections
    conn = sqlite3.connect(str(DB_PATH))
    collection = get_or_create_collection(PROJECT_NAME)

    ingested = 0
    errors   = 0

    for i, msg_path in enumerate(msg_files, 1):
        filename = msg_path.name
        print(f"[{i}/{len(msg_files)}] {filename}")

        # Step 1: Clean SQLite (handles duplicates)
        chunk_ids = clean_msg_from_sqlite(conn, filename)
        if chunk_ids:
            print(f"    Cleaned {len(chunk_ids)} chunks from SQLite")

        # Step 2: Clean ChromaDB
        delete_from_chroma(collection, chunk_ids)

        # Step 3: Re-ingest via pipeline
        result = ingest_file(PROJECT_NAME, msg_path, force=True)
        status = result.get("status")
        chunks = result.get("chunks", 0)
        message = result.get("message", "")

        if status == "ingested":
            print(f"    OK: {chunks} chunks ingested")
            ingested += 1
        else:
            print(f"    ERROR: {message}")
            errors += 1

    conn.close()
    print()
    print("=== Complete ===")
    print(f"  Ingested: {ingested}")
    print(f"  Errors:   {errors}")
    print()

    # Spot-check
    print("=== Spot-check: first 3 .msg files in SQLite ===")
    conn2 = sqlite3.connect(str(DB_PATH))
    conn2.row_factory = sqlite3.Row
    cur = conn2.cursor()
    cur.execute("""
        SELECT source, email_date, email_sender
        FROM chunks
        WHERE source LIKE '%.msg'
        GROUP BY source
        LIMIT 3
    """)
    for row in cur.fetchall():
        print(f"  {row['source']}")
        print(f"    date:   {row['email_date']}")
        print(f"    sender: {row['email_sender']}")
    conn2.close()

    print()
    print("Run patch_chroma_email_metadata.py next to sync the new")
    print("SQLite metadata into ChromaDB for these chunks.")


if __name__ == "__main__":
    main()
