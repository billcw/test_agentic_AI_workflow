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

## Method 1 — Web UI Upload (Recommended)

1. Open `http://localhost:8000` in your browser
2. Select your project from the dropdown
3. Click **Upload Doc**
4. Select a file — any supported type
5. Wait for the confirmation: `✓ filename.pdf (N chunks)`

The document is immediately searchable after upload.

---

## Method 2 — Python Script (Bulk Ingestion)

For ingesting many documents at once, use the pipeline directly:
```python
from src.ingestion.pipeline import ingest_file, ingest_folder

# Ingest a single file
result = ingest_file("my-project", "/path/to/document.pdf")
print(result)

# Ingest an entire folder
results = ingest_folder("my-project", "/path/to/documents/")
for r in results:
    print(r["status"], r.get("file", ""))
```

Run from the project root with the virtualenv active:
```bash
source venv/bin/activate
python3 -c "
from src.ingestion.pipeline import ingest_folder
results = ingest_folder('my-project', '/path/to/documents/')
success = sum(1 for r in results if r['status'] == 'ingested')
print(f'Ingested {success} of {len(results)} files')
"
```

---

## Re-ingesting Documents

The pipeline deduplicates by `chunk_id`. If you ingest the same file
twice, existing chunks are updated rather than duplicated. It is safe
to re-run ingestion on a folder after adding new files.

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
emails from your mail client as .eml files and ingest the folder.
