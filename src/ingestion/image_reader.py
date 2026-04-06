"""
image_reader.py - Standalone image OCR (.jpg, .png, .tiff).
"""

import pytesseract
from PIL import Image
from pathlib import Path
from src.config import PATHS, OCR


def read_image(file_path: str | Path) -> list[dict]:
    """
    Extract text from an image file using Tesseract OCR.

    Returns a list with a single dict:
    [
        {
            "page": 1,
            "text": "ocr extracted text...",
            "method": "ocr",
            "source": "filename.jpg"
        }
    ]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    pytesseract.pytesseract.tesseract_cmd = PATHS["tesseract_path"]
    img = Image.open(str(file_path))
    text = pytesseract.image_to_string(img, lang=OCR["language"])

    return [{
        "page": 1,
        "text": text.strip(),
        "method": "ocr",
        "source": file_path.name
    }]
