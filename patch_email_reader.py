#!/usr/bin/env python3
"""
patch_email_reader.py
Adds a metadata dict to the page dicts returned by _read_eml()
and _read_msg() in email_reader.py.

Why: chunker.py now passes page["metadata"] through to every chunk,
but email_reader.py never populated that dict. Without this patch,
.eml and .msg emails would still have no email_date/sender/subject
in the database even after the chunker fix.
"""

from pathlib import Path

TARGET = Path("/home/bill/local-ai-doc-assistant/src/ingestion/email_reader.py")

# --- Patch 1: _read_eml() return value ---
OLD_EML = '''    return [{
        "page": 1,
        "text": full_text,
        "method": "email",
        "source": file_path.name
    }]'''

NEW_EML = '''    return [{
        "page": 1,
        "text": full_text,
        "method": "email",
        "source": file_path.name,
        "metadata": {
            "email_date":    date,
            "email_sender":  sender,
            "email_subject": subject,
            "source_type":   "email",
        },
    }]'''

# --- Patch 2: _read_msg() return value ---
OLD_MSG = '''    return [{
        "page": 1,
        "text": full_text,
        "method": "email",
        "source": file_path.name
    }]'''

NEW_MSG = '''    return [{
        "page": 1,
        "text": full_text,
        "method": "email",
        "source": file_path.name,
        "metadata": {
            "email_date":    date,
            "email_sender":  sender,
            "email_subject": subject,
            "source_type":   "email",
        },
    }]'''

text = TARGET.read_text()

# Both return blocks are identical text, so we must patch them
# individually by asserting count == 2 first, then replacing both.
assert text.count(OLD_EML) == 2, (
    f"Expected exactly 2 matches for the return block, found {text.count(OLD_EML)}"
)

# Replace both occurrences (eml first, msg second — same text, same replacement)
patched = text.replace(OLD_EML, NEW_EML)
TARGET.write_text(patched)
print("email_reader.py patched successfully.")
