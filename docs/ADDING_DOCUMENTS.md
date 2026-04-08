# Adding Documents to Your Projects

## Supported File Types

| File Type | Extensions | Method |
|-----------|-----------|--------|
| PDF (digital) | .pdf | pymupdf direct extraction |
| PDF (scanned) | .pdf | pymupdf + Tesseract OCR |
| Word Document | .docx | python-docx |
| Excel Spreadsheet | .xlsx, .xls | openpyxl |
| Email | .eml | Python email library |
| Plain Text | .txt, .md | Direct read |
| CSV | .csv | Direct read |
| Images | .jpg, .png, .tiff | Tesseract OCR |

---

## Method 1 — Upload Files via Web UI

1. Open `http://localhost:8000` in your browser
2. Select your project from the dropdown
3. Click **⬆ Upload Files** in the top bar
4. Select one or more files — any supported type, any mix
5. Files are ingested one by one — watch for the confirmation:
   `✓ 3 files ingested`

The documents are immediately searchable after upload.

---

## Method 2 — Ingest a Folder via Web UI (Recommended for Bulk)

For ingesting an entire folder of documents already on your machine:

1. Open `http://localhost:8000` in your browser
2. Select your project from the dropdown
3. In the **📁 Ingest folder** bar below the top bar, type the
   absolute path to your folder:
   `/home/yourname/documents/scada-manuals`
4. Optional: check **re-ingest already indexed** to force re-processing
   of files that have already been ingested
5. Click **Ingest Folder**
6. Watch the status — it will show:
   `✓ 16 ingested, 2 skipped`

The server reads the documents directly from disk — no uploading
required. This is the fastest method for large document collections.

**Note:** The path must be an absolute path accessible to the server
process. If the server is running on a remote machine, the path must
exist on that machine.

---

## Method 3 — Python Script (Advanced)

For scripted or automated ingestion, use the pipeline directly:
```python
from src.ingestion.pipeline import ingest_file, ingest_directory

# Ingest a single file
result = ingest_file("my-project", "/path/to/document.pdf")
print(result)
# {"status": "ingested", "file": "document.pdf", "chunks": 42}

# Ingest an entire folder recursively
result = ingest_directory("my-project", "/path/to/documents/")
print(f"Ingested: {result['ingested']}, Skipped: {result['skipped']}, Errors: {result['errors']}")
```

Run from the project root with the virtualenv active:
```bash
source venv/bin/activate
python3 -c "
from src.ingestion.pipeline import ingest_directory
result = ingest_directory('my-project', '/path/to/documents/')
print(f\"Ingested {result['ingested']} of {result['total']} files\")
"
```

---

## Re-ingesting Documents

The pipeline deduplicates by `chunk_id`. If you ingest the same file
twice, existing chunks are updated rather than duplicated. It is safe
to re-run ingestion on a folder after adding new files — only new
files will be processed.

To force re-processing of already-indexed files, use the
**re-ingest already indexed** checkbox in the UI, or pass `force=True`
in the Python API.

---

## Checking What Has Been Ingested
```python
from src.storage.metadata_db import get_project_stats, list_documents

# Summary stats
stats = get_project_stats("my-project")
print(stats)
# {"project": "my-project", "documents": 42, "chunks": 1847}

# List all documents
docs = list_documents("my-project")
for doc in docs:
    print(doc["filename"], doc["chunks"])
```

---

## Tips for Best Results

**PDF quality matters.** Digital PDFs (text-based) produce much better
results than scanned PDFs. If you have the choice, use digital PDFs.

**Scanned documents need good scan quality.** Tesseract OCR works best
on clean, high-contrast scans at 300 DPI or higher.

**Chunk size affects retrieval.** The default 1000-character chunks work
well for most documents. For very dense technical manuals, you may want
to reduce `chunk_size` in `config.yaml` to 600-800.

**Organize by project.** Keep related documents in the same project
workspace. The hybrid search works across all documents in a project,
so grouping related content together improves answer quality.

**Email archives.** The system ingests .eml files directly. Export
emails from your mail client as .eml files and place them in a folder,
then use the Ingest Folder feature to bulk-ingest them.

**Large folders.** Ingesting hundreds of PDFs takes time — expect
roughly 30-90 seconds per large PDF depending on your hardware. The
server terminal shows per-file progress during ingestion.
