"""
metadata_db.py - SQLite metadata store for chunks and source documents.
Tracks what has been ingested, when, and from where.

Why we need this alongside ChromaDB:
- ChromaDB stores vectors and text but isn't great for relational queries
- SQLite lets us ask: "What documents are in this project?"
- SQLite lets us check: "Has this file already been ingested?"
- SQLite stores ingestion timestamps, file sizes, page counts
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from src.config import PATHS


def get_db_path(project_name: str) -> Path:
    """Get the SQLite database path for a project."""
    db_dir = Path(PATHS["workspaces_root"]) / project_name
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "metadata.db"


def get_connection(project_name: str) -> sqlite3.Connection:
    """Get a SQLite connection for a project."""
    db_path = get_db_path(project_name)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    return conn


def initialize_db(project_name: str) -> None:
    """
    Create the database tables if they don't exist.
    Safe to call multiple times - won't overwrite existing data.
    """
    conn = get_connection(project_name)
    cursor = conn.cursor()

    # Table for source documents
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL UNIQUE,
            file_type TEXT NOT NULL,
            file_size_bytes INTEGER,
            page_count INTEGER,
            chunk_count INTEGER,
            ingested_at TEXT NOT NULL,
            status TEXT DEFAULT 'ingested'
        )
    """)

    # Table for individual chunks
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
    """)

    conn.commit()
    conn.close()


def document_already_ingested(project_name: str, file_path: str) -> bool:
    """Check if a document has already been ingested."""
    conn = get_connection(project_name)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM documents WHERE file_path = ?",
        (str(file_path),)
    )
    result = cursor.fetchone()
    conn.close()
    return result is not None


def record_document(project_name: str, file_path: str, file_type: str,
                    page_count: int, chunk_count: int) -> int:
    """
    Record a successfully ingested document.
    Returns the document ID.
    """
    conn = get_connection(project_name)
    cursor = conn.cursor()

    path = Path(file_path)
    cursor.execute("""
        INSERT OR REPLACE INTO documents
        (filename, file_path, file_type, file_size_bytes, page_count, chunk_count, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        path.name,
        str(file_path),
        file_type,
        path.stat().st_size if path.exists() else 0,
        page_count,
        chunk_count,
        datetime.now().isoformat()
    ))

    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def record_chunks(project_name: str, chunks: list[dict], document_id: int) -> None:
    """Record all chunks for a document in the metadata database."""
    conn = get_connection(project_name)
    cursor = conn.cursor()

    for chunk in chunks:
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
        ))

    conn.commit()
    conn.close()


def list_documents(project_name: str) -> list[dict]:
    """List all ingested documents for a project."""
    conn = get_connection(project_name)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, file_type, page_count, chunk_count, ingested_at
        FROM documents
        ORDER BY ingested_at DESC
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_project_stats(project_name: str) -> dict:
    """Get summary statistics for a project."""
    conn = get_connection(project_name)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as doc_count FROM documents")
    doc_count = cursor.fetchone()["doc_count"]

    cursor.execute("SELECT COUNT(*) as chunk_count FROM chunks")
    chunk_count = cursor.fetchone()["chunk_count"]

    conn.close()
    return {
        "project": project_name,
        "documents": doc_count,
        "chunks": chunk_count
    }
