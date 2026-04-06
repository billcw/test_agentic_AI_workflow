"""
docx_reader.py - Microsoft Word document text extraction.
"""

from pathlib import Path
from docx import Document


def read_docx(file_path: str | Path) -> list[dict]:
    """
    Extract text from a Word document (.docx).
    Groups content into sections by heading structure.
    Falls back to paragraph chunks if no headings exist.

    Returns a list of dicts:
    [
        {
            "page": 1,
            "text": "extracted text...",
            "method": "docx",
            "source": "filename.docx"
        },
        ...
    ]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Word document not found: {file_path}")

    doc = Document(str(file_path))
    sections = []
    current_section = []
    section_num = 1

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Start a new section at each heading
        if para.style.name.startswith("Heading"):
            if current_section:
                sections.append({
                    "page": section_num,
                    "text": "\n".join(current_section),
                    "method": "docx",
                    "source": file_path.name
                })
                section_num += 1
                current_section = []
        current_section.append(text)

    # Don't forget the last section
    if current_section:
        sections.append({
            "page": section_num,
            "text": "\n".join(current_section),
            "method": "docx",
            "source": file_path.name
        })

    return sections
