#!/usr/bin/env python3
"""
patch_chunker.py
Patches chunk_pages() in chunker.py to pass the metadata dict
from each page through to every chunk it produces.

Why: pst_reader.py already extracts email_date, email_sender,
email_subject into a metadata dict on each page. But chunk_pages()
currently ignores that dict, so the metadata is lost before it
ever reaches the database. This patch fixes that.
"""

from pathlib import Path

TARGET = Path("/home/bill/local-ai-doc-assistant/src/ingestion/chunker.py")

OLD = '''        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}_p{page_num}_c{i}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "source": source,
                "page": page_num,
                "chunk_index": i,
                "method": method
            })'''

NEW = '''        # Carry any metadata the reader attached to this page
        # (e.g. email_date, email_sender, email_subject from pst_reader)
        page_metadata = page.get("metadata", {})

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}_p{page_num}_c{i}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "source": source,
                "page": page_num,
                "chunk_index": i,
                "method": method,
                "metadata": page_metadata,
            })'''

text = TARGET.read_text()
assert text.count(OLD) == 1, f"Expected exactly 1 match, found {text.count(OLD)}"
patched = text.replace(OLD, NEW)
TARGET.write_text(patched)
print("chunker.py patched successfully.")
