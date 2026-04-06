"""
chunker.py - Smart text chunking for the ingestion pipeline.
Splits document pages into overlapping chunks for indexing.

Why chunking matters:
- LLMs have context limits - we can't feed them entire documents
- Smaller chunks = more precise retrieval
- Overlap between chunks = no information lost at boundaries
"""

from src.config import RETRIEVAL


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk
        chunk_size: Max characters per chunk (default from config)
        chunk_overlap: Characters of overlap between chunks (default from config)

    Returns:
        List of text chunks

    Example with chunk_size=100, chunk_overlap=20:
        "AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC"
        Chunk 1: "AAAAAAAAAA BBBBBBBBBB"
        Chunk 2: "BBBBBBBBBB CCCCCCCCCC"  <- overlaps with chunk 1
    """
    if chunk_size is None:
        chunk_size = RETRIEVAL["chunk_size"]
    if chunk_overlap is None:
        chunk_overlap = RETRIEVAL["chunk_overlap"]

    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size

        # If not at the end, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence boundary (. ! ?) within last 200 chars of chunk
            boundary = -1
            for punct in [". ", "! ", "? ", "\n\n"]:
                pos = text.rfind(punct, start + chunk_size - 200, end)
                if pos > boundary:
                    boundary = pos

            if boundary > start:
                end = boundary + 1  # Include the punctuation

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by chunk_size minus overlap
        start = start + chunk_size - chunk_overlap

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Take the output of any reader and chunk all pages into
    indexed, metadata-rich chunk dicts ready for indexing.

    Input: list of page dicts from any reader
    Output: list of chunk dicts with full metadata

    Each chunk dict:
    {
        "chunk_id": "filename.pdf_p1_c0",
        "text": "chunk text...",
        "source": "filename.pdf",
        "page": 1,
        "chunk_index": 0,
        "method": "digital"
    }
    """
    all_chunks = []

    for page in pages:
        text = page.get("text", "")
        source = page.get("source", "unknown")
        page_num = page.get("page", 1)
        method = page.get("method", "unknown")

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}_p{page_num}_c{i}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "source": source,
                "page": page_num,
                "chunk_index": i,
                "method": method
            })

    return all_chunks
