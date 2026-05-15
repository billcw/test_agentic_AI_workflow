#!/usr/bin/env python3
"""
patch_mbox_wiring.py
Two changes to wire mbox_reader into the pipeline:

1. detector.py — adds ".mbox" to EXTENSION_MAP so the detector
   recognizes MBOX files instead of returning "unsupported".

2. pipeline.py — adds the mbox import and adds "mbox" to the
   READERS dict so ingest_file() knows which reader to call.
"""

from pathlib import Path

# --- Patch 1: detector.py ---
DETECTOR = Path("/home/bill/local-ai-doc-assistant/src/ingestion/detector.py")

OLD_DETECTOR = '''    # Outlook archive files
    ".pst": "pst",
    ".ost": "pst",'''

NEW_DETECTOR = '''    # Outlook archive files
    ".pst": "pst",
    ".ost": "pst",
    # MBOX email archives (Gmail, Yahoo, Thunderbird, Apple Mail)
    ".mbox": "mbox",'''

text = DETECTOR.read_text()
assert text.count(OLD_DETECTOR) == 1, (
    f"detector.py: expected 1 match, found {text.count(OLD_DETECTOR)}"
)
DETECTOR.write_text(text.replace(OLD_DETECTOR, NEW_DETECTOR))
print("detector.py patched successfully.")

# --- Patch 2: pipeline.py — add import ---
PIPELINE = Path("/home/bill/local-ai-doc-assistant/src/ingestion/pipeline.py")

OLD_IMPORT = "from src.ingestion.pst_reader import read_pst"
NEW_IMPORT = ("from src.ingestion.pst_reader import read_pst\n"
              "from src.ingestion.mbox_reader import read_mbox")

text = PIPELINE.read_text()
assert text.count(OLD_IMPORT) == 1, (
    f"pipeline.py import: expected 1 match, found {text.count(OLD_IMPORT)}"
)
text = text.replace(OLD_IMPORT, NEW_IMPORT)

# --- Patch 3: pipeline.py — add to READERS dict ---
OLD_READERS = '''    "pst":   read_pst,
}'''

NEW_READERS = '''    "pst":   read_pst,
    "mbox":  read_mbox,
}'''

assert text.count(OLD_READERS) == 1, (
    f"pipeline.py READERS: expected 1 match, found {text.count(OLD_READERS)}"
)
text = text.replace(OLD_READERS, NEW_READERS)

PIPELINE.write_text(text)
print("pipeline.py patched successfully.")
