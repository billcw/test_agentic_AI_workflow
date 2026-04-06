"""
pdf_reader.py - PDF text extraction with automatic OCR fallback.
Handles both digital PDFs (direct text) and scanned PDFs (OCR).
"""

import pymupdf
import pytesseract
from PIL import Image
from pathlib import Path
from src.config import PATHS, OCR


def is_page_scanned(page) -> bool:
    """
    Determine if a PDF page is scanned (image) or digital (text).
    A page is considered scanned if it has very little extractable text
    but contains images.
    """
    text = page.get_text().strip()
    image_list = page.get_images()
    # If less than 50 characters of text but has images, likely scanned
    return len(text) < 50 and len(image_list) > 0


def extract_page_with_ocr(page) -> str:
    """
    Extract text from a scanned PDF page using Tesseract OCR.
    Renders the page as an image first, then runs OCR on it.
    """
    # Render page to image at 300 DPI for good OCR quality
    mat = pymupdf.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)

    # Convert to PIL Image for Tesseract
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Run Tesseract OCR
    pytesseract.pytesseract.tesseract_cmd = PATHS["tesseract_path"]
    text = pytesseract.image_to_string(img, lang=OCR["language"])
    return text.strip()


def read_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF file, page by page.
    Automatically uses OCR for scanned pages.

    Returns a list of dicts, one per page:
    [
        {
            "page": 1,
            "text": "extracted text...",
            "method": "digital" or "ocr",
            "source": "filename.pdf"
        },
        ...
    ]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages = []
    doc = pymupdf.open(str(file_path))

    for page_num in range(len(doc)):
        page = doc[page_num]

        if is_page_scanned(page):
            # Scanned page - use OCR
            text = extract_page_with_ocr(page)
            method = "ocr"
        else:
            # Digital page - extract text directly
            text = page.get_text().strip()
            method = "digital"

        if text:  # Only include pages with actual content
            pages.append({
                "page": page_num + 1,
                "text": text,
                "method": method,
                "source": file_path.name
            })

    doc.close()
    return pages
