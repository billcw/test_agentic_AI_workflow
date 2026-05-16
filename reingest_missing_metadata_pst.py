"""
reingest_missing_metadata_pst.py - Delete and re-ingest 5 PST files
that were ingested before email metadata fields were added to ChromaDB.

Why delete first?
    add_chunks() skips chunks that already exist in ChromaDB (by chunk_id).
    force=True only bypasses the "already ingested" document-level check --
    it does NOT delete existing ChromaDB chunks. So re-ingesting without
    deleting first silently does nothing for already-indexed chunks.

What this script does per PST file:
    1. Find the document record in SQLite by filename
    2. Collect all chunk_ids for that document from SQLite
    3. Delete those chunk_ids from ChromaDB (no embeddings lost permanently
       -- they will be regenerated fresh on re-ingest)
    4. Delete chunk rows from SQLite chunks table
    5. Delete the document row from SQLite documents table
    6. Re-ingest the file using the normal pipeline (which now writes
       email_date, email_sender, email_subject into ChromaDB metadata)

Run from the project root with the virtualenv active:
    python reingest_missing_metadata_pst.py

The server must NOT be running during this script -- ChromaDB does not
support concurrent writers.
"""

import sqlite3
import chromadb
import sys
from pathlib import Path

# Must run from project root so src imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.pipeline import ingest_file

# --- Configuration ---
PROJECT_NAME   = "test-project"
WORKSPACE_ROOT = Path("workspaces")
DB_PATH        = WORKSPACE_ROOT / PROJECT_NAME / "metadata.db"
VECTORS_PATH   = WORKSPACE_ROOT / PROJECT_NAME / "vectors"
CHROMA_BATCH   = 500

# The 5 PST files with missing email metadata.
# Using the project docs paths as canonical source.
PST_FILES = [
    Path("docs/Documents_for_ingest_first_project/outlook_inbox_backup_102021.pst"),
    Path("docs/pst/pst/4-20-26-backup.pst"),
    Path("docs/pst/pst/4-20-26-backup2.pst"),
    Path("docs/pst/pst/4-20-26-backup3.pst"),
    Path("docs/pst/pst/4-20-26-backup4.pst"),
]


def delete_document_from_sqlite(conn: sqlite3.Connection,
                                filename: str) -> tuple[int, list[str]]:
    """
    Delete a document and all its chunks from SQLite.
    Returns (document_id, list_of_chunk_ids) so we can clean ChromaDB too.
    Returns (0, []) if the document is not found.
    """
    cur = conn.cursor()

    # Find document row by filename (basename match)
    cur.execute(
        "SELECT id FROM documents WHERE filename = ?",
        (filename,)
    )
    row = cur.fetchone()
    if not row:
        print(f"    WARNING: '{filename}' not found in documents table — skipping SQLite delete")
        return 0, []

    doc_id = row[0]

    # Collect all chunk_ids for this document
    cur.execute(
        "SELECT chunk_id FROM chunks WHERE document_id = ?",
        (doc_id,)
    )
    chunk_ids = [r[0] for r in cur.fetchall()]
    print(f"    SQLite: found {len(chunk_ids):,} chunks for document_id={doc_id}")

    # Delete chunks first (foreign key child), then document (parent)
    cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    print(f"    SQLite: deleted {len(chunk_ids):,} chunks and document record")

    return doc_id, chunk_ids


def delete_chunks_from_chroma(collection, chunk_ids: list[str]):
    """
    Delete chunk_ids from ChromaDB in batches of CHROMA_BATCH.
    ChromaDB delete() accepts a list of ids directly.
    """
    if not chunk_ids:
        print(f"    ChromaDB: no chunk_ids to delete")
        return

    total_deleted = 0
    for i in range(0, len(chunk_ids), CHROMA_BATCH):
        batch = chunk_ids[i:i + CHROMA_BATCH]
        collection.delete(ids=batch)
        total_deleted += len(batch)

    print(f"    ChromaDB: deleted {total_deleted:,} chunks")


def main():
    print("=== PST Re-ingest: Missing Email Metadata Fix ===")
    print(f"Project:  {PROJECT_NAME}")
    print(f"SQLite:   {DB_PATH}")
    print(f"ChromaDB: {VECTORS_PATH}")
    print()

    # Verify all PST files exist before starting
    print("Verifying PST files exist...")
    missing = [p for p in PST_FILES if not p.exists()]
    if missing:
        print("ERROR: The following PST files were not found:")
        for p in missing:
            print(f"  {p}")
        print("Aborting — fix paths before running.")
        sys.exit(1)
    print(f"  All {len(PST_FILES)} PST files found.")
    print()

    # Open SQLite and ChromaDB connections
    conn = sqlite3.connect(str(DB_PATH))
    client = chromadb.PersistentClient(path=str(VECTORS_PATH))
    collection = client.get_collection(name=PROJECT_NAME)

    print(f"ChromaDB chunk count before: {collection.count():,}")
    print()

    # Process each PST file
    for pst_path in PST_FILES:
        filename = pst_path.name
        print(f"--- Processing: {filename} ---")

        # Step 1+2: Delete from SQLite, collect chunk_ids
        doc_id, chunk_ids = delete_document_from_sqlite(conn, filename)

        # Step 3: Delete from ChromaDB
        delete_chunks_from_chroma(collection, chunk_ids)

        # Step 4: Re-ingest using the normal pipeline
        print(f"    Re-ingesting {filename}...")
        result = ingest_file(
            project_name=PROJECT_NAME,
            file_path=pst_path,
            force=True
        )
        status = result.get("status", "unknown")
        chunks = result.get("chunks", 0)
        message = result.get("message", "")
        print(f"    Ingest result: status={status}, chunks={chunks:,}")
        if message:
            print(f"    Message: {message}")
        print()

    conn.close()

    print(f"ChromaDB chunk count after: {collection.count():,}")
    print()

    # Spot-check: verify email metadata now populated for each file
    print("=== Spot-check: verifying email metadata in SQLite ===")
    conn2 = sqlite3.connect(str(DB_PATH))
    cur = conn2.cursor()
    for pst_path in PST_FILES:
        filename = pst_path.name
        cur.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN email_date != '' AND email_date IS NOT NULL
                            THEN 1 ELSE 0 END) as has_date,
                   SUM(CASE WHEN email_sender != '' AND email_sender IS NOT NULL
                            THEN 1 ELSE 0 END) as has_sender
            FROM chunks
            WHERE source = ?
        """, (filename,))
        row = cur.fetchone()
        total, has_date, has_sender = row
        print(f"  {filename}")
        print(f"    total={total:,}  has_date={has_date:,}  has_sender={has_sender:,}")
    conn2.close()

    print()
    print("=== Done ===")
    print("Run patch_chroma_email_metadata.py next to sync the new")
    print("SQLite metadata into ChromaDB for the re-ingested chunks.")


if __name__ == "__main__":
    main()
