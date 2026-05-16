"""
patch_chroma_email_metadata.py - One-time migration script.

Patches existing ChromaDB chunks with email metadata (email_date,
email_sender, email_subject) sourced from SQLite metadata.db.

Why this is needed:
    add_chunks() previously stored only source, page, chunk_index,
    and method in ChromaDB metadata. Email fields existed in SQLite
    but were never written to ChromaDB, making date-range where
    filters impossible.

What this script does:
    1. Reads all chunk_ids + email metadata from SQLite
    2. Opens ChromaDB for test-project
    3. Calls collection.update() in batches of 500 to patch metadata
       on existing chunks -- no embeddings are touched, no re-ingestion

Non-email chunks receive empty strings for the three email fields.
ChromaDB requires identical metadata keys across all documents in a
collection, so empty strings are used rather than omitting the keys.

Run once from the project root with the virtualenv active:
    python patch_chroma_email_metadata.py

Safe to re-run -- update() is idempotent.
"""

import sqlite3
import chromadb
from pathlib import Path

# --- Configuration ---
PROJECT_NAME  = "test-project"
WORKSPACE_ROOT = Path("workspaces")
DB_PATH       = WORKSPACE_ROOT / PROJECT_NAME / "metadata.db"
VECTORS_PATH  = WORKSPACE_ROOT / PROJECT_NAME / "vectors"
BATCH_SIZE    = 500


def main():
    print(f"=== ChromaDB Email Metadata Migration ===")
    print(f"Project:  {PROJECT_NAME}")
    print(f"SQLite:   {DB_PATH}")
    print(f"ChromaDB: {VECTORS_PATH}")
    print()

    # --- Step 1: Load all chunks from SQLite ---
    print("Step 1: Reading chunks from SQLite...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT chunk_id, source, page, chunk_index, method,
               email_date, email_sender, email_subject
        FROM chunks
        ORDER BY id
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"  Loaded {len(rows):,} chunks from SQLite")

    if not rows:
        print("  No chunks found — nothing to migrate.")
        return

    # --- Step 2: Open ChromaDB ---
    print("Step 2: Opening ChromaDB collection...")
    client = chromadb.PersistentClient(path=str(VECTORS_PATH))
    collection = client.get_collection(name=PROJECT_NAME)
    chroma_count = collection.count()
    print(f"  ChromaDB has {chroma_count:,} chunks")

    # --- Step 3: Patch in batches ---
    print(f"Step 3: Patching metadata in batches of {BATCH_SIZE}...")
    print()

    total_patched = 0
    total_skipped = 0
    batch_ids      = []
    batch_metadata = []

    for i, row in enumerate(rows):
        batch_ids.append(row["chunk_id"])
        batch_metadata.append({
            "source":        row["source"]        or "",
            "page":          row["page"]           or 0,
            "chunk_index":   row["chunk_index"]    or 0,
            "method":        row["method"]         or "",
            "email_date":    row["email_date"]     or "",
            "email_sender":  row["email_sender"]   or "",
            "email_subject": row["email_subject"]  or "",
        })

        # Flush batch
        if len(batch_ids) >= BATCH_SIZE:
            try:
                collection.update(
                    ids=batch_ids,
                    metadatas=batch_metadata
                )
                total_patched += len(batch_ids)
            except Exception as e:
                print(f"  WARNING: Batch update failed: {e}")
                total_skipped += len(batch_ids)
            batch_ids      = []
            batch_metadata = []

            if total_patched % 10000 == 0 and total_patched > 0:
                pct = (total_patched / len(rows)) * 100
                print(f"  Progress: {total_patched:,} / {len(rows):,} "
                      f"({pct:.1f}%)")

    # Flush remaining
    if batch_ids:
        try:
            collection.update(
                ids=batch_ids,
                metadatas=batch_metadata
            )
            total_patched += len(batch_ids)
        except Exception as e:
            print(f"  WARNING: Final batch update failed: {e}")
            total_skipped += len(batch_ids)

    print()
    print("=== Migration Complete ===")
    print(f"  Patched: {total_patched:,}")
    print(f"  Skipped: {total_skipped:,}")
    print()

    # --- Step 4: Spot-check a few email chunks ---
    print("Step 4: Spot-checking email chunks in ChromaDB...")
    sample = collection.get(
        where={"method": {"$eq": "email"}},
        limit=5,
        include=["metadatas"]
    )
    if sample["ids"]:
        for cid, meta in zip(sample["ids"], sample["metadatas"]):
            print(f"  chunk_id: {cid}")
            print(f"    email_date:    {meta.get('email_date', '(missing)')}")
            print(f"    email_sender:  {meta.get('email_sender', '(missing)')}")
            print(f"    email_subject: {meta.get('email_subject', '(missing)')}")
            print()
    else:
        print("  No email chunks found in ChromaDB for spot-check.")

    print("Done. ChromaDB is ready for email metadata filtering.")


if __name__ == "__main__":
    main()
