"""
pipeline.py - Document ingestion pipeline.
Orchestrates the full flow: detect -> read -> chunk -> index -> record.
"""

from pathlib import Path
from src.ingestion.detector import detect_file_type
from src.ingestion.pdf_reader import read_pdf
from src.ingestion.image_reader import read_image
from src.ingestion.docx_reader import read_docx
from src.ingestion.email_reader import read_email
from src.ingestion.excel_reader import read_excel
from src.ingestion.text_reader import read_text
from src.ingestion.pst_reader import read_pst
from src.ingestion.mbox_reader import read_mbox
from src.ingestion.chunker import chunk_pages
from src.storage.vector_store import add_chunks, get_or_create_collection
from src.storage.keyword_store import load_index, save_index
from src.storage.metadata_db import (
    initialize_db,
    document_already_ingested,
    purge_document,
    record_document,
    record_chunks,
)

READERS = {
    "pdf":      read_pdf,
    "image":    read_image,
    "docx":     read_docx,
    "email":    read_email,
    "excel":    read_excel,
    "text":     read_text,
    "pst":      read_pst,
    "mbox":     read_mbox,
}

CHROMA_BATCH = 500


def _purge_from_chroma(project_name: str, chunk_ids: list[str]) -> None:
    """
    Delete a list of chunk_ids from ChromaDB in batches.

    Called by ingest_file() when force=True, after purge_document()
    has cleaned SQLite. Together they ensure a complete clean slate
    before re-ingestion so no stale data (e.g. empty email metadata)
    survives from a prior ingest run.
    """
    if not chunk_ids:
        return
    collection = get_or_create_collection(project_name)
    for i in range(0, len(chunk_ids), CHROMA_BATCH):
        batch = chunk_ids[i:i + CHROMA_BATCH]
        collection.delete(ids=batch)


def ingest_file(project_name: str, file_path: str | Path,
                force: bool = False) -> dict:
    """
    Ingest a single document into the project index.

    Args:
        project_name: The project workspace to ingest into
        file_path: Path to the document
        force: If True, purge existing chunks and re-ingest from scratch.
               This guarantees stale metadata (e.g. empty email fields
               from a pre-metadata-column ingest) is replaced with fresh
               data. Without purging first, INSERT OR IGNORE in SQLite
               and the skip-if-exists check in ChromaDB would silently
               retain the old empty records.

    Returns a result dict:
    {
        "status": "ingested" | "skipped" | "error",
        "file": "filename.pdf",
        "chunks": 12,
        "message": "..."
    }
    """
    file_path = Path(file_path)

    # Step 1: Detect file type
    file_type = detect_file_type(file_path)
    if file_type == "unsupported":
        return {
            "status": "skipped",
            "file": file_path.name,
            "chunks": 0,
            "message": f"Unsupported file type: {file_path.suffix}"
        }

    # Step 2: Initialize DB for this project
    initialize_db(project_name)

    # Step 3: Skip if already ingested (unless forced)
    if not force and document_already_ingested(project_name, str(file_path)):
        return {
            "status": "skipped",
            "file": file_path.name,
            "chunks": 0,
            "message": "Already ingested - use force=True to re-ingest"
        }

    # Step 3b: If forced, purge existing records so re-ingest is clean.
    # purge_document() removes chunk rows and the document row from SQLite
    # and returns the old chunk_ids so we can also clean ChromaDB.
    # Without this, INSERT OR IGNORE (SQLite) and the exists-check
    # (ChromaDB add_chunks) silently skip already-present chunk_ids,
    # leaving stale data in place even after a force re-ingest.
    if force:
        old_chunk_ids = purge_document(project_name, str(file_path))
        if old_chunk_ids:
            print(f"  [Pipeline] Purged {len(old_chunk_ids)} stale chunks "
                  f"for {file_path.name}")
        _purge_from_chroma(project_name, old_chunk_ids)

    try:
        # Step 4: Read the document
        reader = READERS[file_type]
        pages = reader(file_path)
        if not pages:
            return {
                "status": "error",
                "file": file_path.name,
                "chunks": 0,
                "message": "No text extracted from document"
            }

        # Step 5: Chunk the pages
        chunks = chunk_pages(pages)
        if not chunks:
            return {
                "status": "error",
                "file": file_path.name,
                "chunks": 0,
                "message": "No chunks produced from document"
            }

        # Step 6: Add to vector store (ChromaDB)
        added = add_chunks(project_name, chunks)

        # Step 7: Update BM25 index
        # Load existing chunks, merge with new ones, deduplicate by chunk_id.
        # This prevents duplicate entries if a file is re-ingested.
        _, existing_chunks = load_index(project_name)
        existing_by_id = {c["chunk_id"]: c for c in (existing_chunks or [])}
        for chunk in chunks:
            existing_by_id[chunk["chunk_id"]] = chunk
        all_chunks = list(existing_by_id.values())
        save_index(project_name, all_chunks)

        # Step 8: Record in metadata DB
        doc_id = record_document(
            project_name=project_name,
            file_path=str(file_path),
            file_type=file_type,
            page_count=len(pages),
            chunk_count=len(chunks)
        )
        record_chunks(project_name, chunks, doc_id)

        return {
            "status": "ingested",
            "file": file_path.name,
            "chunks": len(chunks),
            "message": f"Successfully ingested {len(pages)} pages into {len(chunks)} chunks"
        }

    except Exception as e:
        return {
            "status": "error",
            "file": file_path.name,
            "chunks": 0,
            "message": f"Error during ingestion: {str(e)}"
        }


def ingest_directory(project_name: str, directory: str | Path,
                     force: bool = False) -> dict:
    """
    Ingest all supported documents in a directory (recursively).

    Args:
        project_name: The project workspace to ingest into
        directory: Path to the directory to scan
        force: If True, re-ingest files even if already indexed

    Returns a summary dict:
    {
        "ingested": 5,
        "skipped": 2,
        "errors": 1,
        "total": 8,
        "results": [...]
    }
    """
    directory = Path(directory)
    if not directory.exists():
        return {
            "ingested": 0,
            "skipped": 0,
            "errors": 1,
            "total": 0,
            "results": [{"status": "error", "message": f"Directory not found: {directory}"}]
        }

    results = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            result = ingest_file(project_name, file_path, force=force)
            results.append(result)

    ingested = sum(1 for r in results if r["status"] == "ingested")
    skipped  = sum(1 for r in results if r["status"] == "skipped")
    errors   = sum(1 for r in results if r["status"] == "error")

    return {
        "ingested": ingested,
        "skipped":  skipped,
        "errors":   errors,
        "total":    len(results),
        "results":  results
    }
