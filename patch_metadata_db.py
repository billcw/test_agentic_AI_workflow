#!/usr/bin/env python3
"""
patch_metadata_db.py
Two changes to metadata_db.py:

1. initialize_db() — adds email_date, email_sender, email_subject
   columns to the chunks CREATE TABLE statement.
   These are NULL for non-email documents (PDFs, Word, etc.)
   and populated only for email chunks.

2. record_chunks() — reads those three fields out of the chunk's
   metadata dict and stores them in the database.

Why nullable: Not every chunk comes from an email. PDFs, Word docs,
spreadsheets, etc. have no sender or date — NULL is the correct
value for those rows, not an empty string.
"""

from pathlib import Path

TARGET = Path("/home/bill/local-ai-doc-assistant/src/storage/metadata_db.py")

# --- Patch 1: initialize_db() CREATE TABLE for chunks ---
OLD_SCHEMA = '''    # Table for individual chunks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL UNIQUE,
            document_id INTEGER,
            source TEXT NOT NULL,
            page INTEGER,
            chunk_index INTEGER,
            method TEXT,
            text_preview TEXT,
            ingested_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)'''

NEW_SCHEMA = '''    # Table for individual chunks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL UNIQUE,
            document_id INTEGER,
            source TEXT NOT NULL,
            page INTEGER,
            chunk_index INTEGER,
            method TEXT,
            text_preview TEXT,
            ingested_at TEXT NOT NULL,
            email_date TEXT,
            email_sender TEXT,
            email_subject TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)'''

# --- Patch 2: record_chunks() INSERT statement ---
OLD_RECORD = '''    for chunk in chunks:
        cursor.execute("""
            INSERT OR IGNORE INTO chunks
            (chunk_id, document_id, source, page, chunk_index, method, text_preview, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["chunk_id"],
            document_id,
            chunk["source"],
            chunk["page"],
            chunk["chunk_index"],
            chunk["method"],
            chunk["text"][:200],  # Store first 200 chars as preview
            datetime.now().isoformat()
        ))'''

NEW_RECORD = '''    for chunk in chunks:
        # Pull email metadata if present (None for non-email documents)
        meta = chunk.get("metadata", {})
        email_date    = meta.get("email_date")    or meta.get("date")
        email_sender  = meta.get("email_sender")  or meta.get("sender")
        email_subject = meta.get("email_subject") or meta.get("subject")

        cursor.execute("""
            INSERT OR IGNORE INTO chunks
            (chunk_id, document_id, source, page, chunk_index, method,
             text_preview, ingested_at, email_date, email_sender, email_subject)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["chunk_id"],
            document_id,
            chunk["source"],
            chunk["page"],
            chunk["chunk_index"],
            chunk["method"],
            chunk["text"][:200],  # Store first 200 chars as preview
            datetime.now().isoformat(),
            email_date,
            email_sender,
            email_subject,
        ))'''

text = TARGET.read_text()

assert text.count(OLD_SCHEMA) == 1, (
    f"Schema block: expected 1 match, found {text.count(OLD_SCHEMA)}"
)
assert text.count(OLD_RECORD) == 1, (
    f"Record block: expected 1 match, found {text.count(OLD_RECORD)}"
)

patched = text.replace(OLD_SCHEMA, NEW_SCHEMA)
patched = patched.replace(OLD_RECORD, NEW_RECORD)
TARGET.write_text(patched)
print("metadata_db.py patched successfully.")
