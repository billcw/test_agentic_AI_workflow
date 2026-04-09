"""
pipeline.py - Full document ingestion orchestrator.
Ties together: detector -> reader -> chunker -> vector store -> BM25 -> metadata DB.

This is the single entry point for ingesting documents.
Call ingest_file() for one file, or ingest_directory() for a whole folder.
"""

from pathlib import Path
from src.ingestion.detector import detect_file_type, scan_directory
from src.ingestion.pdf_reader import read_pdf
from src.ingestion.docx_reader import read_docx
from src.ingestion.email_reader import read_email
from src.ingestion.excel_reader import read_excel
from src.ingestion.text_reader import read_text
from src.ingestion.image_reader import read_image
from src.ingestion.pst_reader import read_pst
from src.ingestion.chunker import chunk_pages
from src.storage.vector_store import add_chunks
from src.storage.keyword_store import save_index, load_index
from src.storage.metadata_db import (
    initialize_db,
    document_already_ingested,
    record_document,
    record_chunks,
    get_project_stats
)


# Map file types to their reader functions
READERS = {
    "pdf":   read_pdf,
    "docx":  read_docx,
    "email": read_email,
    "excel": read_excel,
    "text":  read_text,
    "image": read_image,
    "pst":   read_pst,
}


def ingest_file(project_name: str, file_path: str | Path,
                force: bool = False) -> dict:
    """
    Ingest a single document into the project index.

    Args:
        project_name: The project workspace to ingest into
        file_path: Path to the document
        force: If True, re-ingest even if already indexed

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
    Ingest all supported documents from a directory recursively.

    Returns a summary dict:
    {
        "ingested": 10,
        "skipped": 2,
        "errors": 1,
        "total": 13,
        "results": [...]
    }
    """
    directory = Path(directory)
    print(f"\nScanning {directory} for documents...")

    # Scan directory for all files
    file_groups = scan_directory(directory)

    # Collect all supported files
    all_files = []
    for file_type, files in file_groups.items():
        if file_type != "unsupported":
            all_files.extend(files)

    unsupported = file_groups.get("unsupported", [])

    print(f"Found {len(all_files)} supported files, "
          f"{len(unsupported)} unsupported files")

    # Ingest each file
    results = []
    ingested = skipped = errors = 0

    for i, file_path in enumerate(all_files):
        print(f"  [{i+1}/{len(all_files)}] {file_path.name}...", end=" ")
        result = ingest_file(project_name, file_path, force=force)
        results.append(result)

        if result["status"] == "ingested":
            ingested += 1
            print(f"OK ({result['chunks']} chunks)")
        elif result["status"] == "skipped":
            skipped += 1
            print("SKIPPED")
        else:
            errors += 1
            print(f"ERROR: {result['message']}")

    # Print final stats
    stats = get_project_stats(project_name)
    print(f"\nIngestion complete:")
    print(f"  Ingested: {ingested}")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {errors}")
    print(f"  Project total: {stats['documents']} docs, {stats['chunks']} chunks")

    return {
        "ingested": ingested,
        "skipped": skipped,
        "errors": errors,
        "total": len(all_files),
        "results": results
    }
