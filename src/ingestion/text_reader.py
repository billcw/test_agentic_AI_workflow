"""
text_reader.py - Plain text file extraction (.txt, .md, .csv).
"""

from pathlib import Path


def read_text(file_path: str | Path) -> list[dict]:
    """
    Read a plain text file (.txt, .md, .csv).

    Returns a list with a single dict:
    [
        {
            "page": 1,
            "text": "file contents...",
            "method": "text",
            "source": "filename.txt"
        }
    ]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    # Try UTF-8 first, fall back to latin-1 for older files
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")

    return [{
        "page": 1,
        "text": text.strip(),
        "method": "text",
        "source": file_path.name
    }]
