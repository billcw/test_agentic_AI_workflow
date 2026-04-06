"""
detector.py - File type detection for the ingestion pipeline.
Every document entering the system is identified here first.
"""

from pathlib import Path


# Supported file types and their categories
EXTENSION_MAP = {
    # PDFs - could be digital or scanned
    ".pdf": "pdf",

    # Word documents
    ".docx": "docx",
    ".doc": "docx",

    # Email files
    ".eml": "email",
    ".msg": "email",

    # Excel spreadsheets
    ".xlsx": "excel",
    ".xls": "excel",

    # Plain text formats
    ".txt": "text",
    ".md": "text",
    ".csv": "text",

    # Images (for OCR)
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".tiff": "image",
    ".tif": "image",
}


def detect_file_type(file_path: str | Path) -> str:
    """
    Detect the type of a file based on its extension.

    Returns a type string like 'pdf', 'docx', 'email', etc.
    Returns 'unsupported' if the file type is not recognized.
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    return EXTENSION_MAP.get(extension, "unsupported")


def is_supported(file_path: str | Path) -> bool:
    """Return True if the file type is supported."""
    return detect_file_type(file_path) != "unsupported"


def scan_directory(directory: str | Path) -> dict:
    """
    Scan a directory recursively and group files by type.

    Returns a dict like:
    {
        'pdf': [Path(...), Path(...)],
        'docx': [Path(...)],
        'unsupported': [Path(...)]
    }
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    results = {}
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            file_type = detect_file_type(file_path)
            if file_type not in results:
                results[file_type] = []
            results[file_type].append(file_path)

    return results
