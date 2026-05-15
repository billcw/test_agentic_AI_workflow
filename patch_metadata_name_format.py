#!/usr/bin/env python3
"""
patch_metadata_name_format.py
Fixes the SQL generation prompt in metadata_agent.py to handle the
"Last, First" name storage format used by Outlook/PST archives.

The problem: when a user asks "emails from Joe Deluna", the LLM was
generating LIKE '%Joe Deluna%' which never matches because the database
stores names as "Deluna, Joe" or "Deluna, Joe [Contractor]" or even
as an email address like "Scott.Larson@osii.com".

The fix: tell the LLM to always search by last name only using LIKE,
which catches all variants: "Deluna, Joe", "Deluna, Joe [Contractor]",
and email addresses containing the last name.
"""

from pathlib import Path

TARGET = Path("/home/bill/local-ai-doc-assistant/src/agents/metadata_agent.py")

OLD_SCHEMA = '''Important notes:
- Each email may produce MULTIPLE chunks (rows) with the same email_date/email_sender/email_subject.
  Use DISTINCT or GROUP BY when counting emails, not rows.
- email_date, email_sender, email_subject are NULL for non-email documents (PDFs, Word files, etc.)
  Always filter with WHERE email_date IS NOT NULL to restrict to emails only.
- email_date is stored as text in 'YYYY-MM-DD HH:MM' format — use LIKE for partial date matches.
  Example: WHERE email_date LIKE '2024-03%' finds all emails from March 2024.
- email_sender may contain full name and address like 'John Smith <john@example.com>'
  Use LIKE '%john%' for case-insensitive partial matching.'''

NEW_SCHEMA = '''Important notes:
- Each email may produce MULTIPLE chunks (rows) with the same email_date/email_sender/email_subject.
  Use DISTINCT or GROUP BY when counting emails, not rows.
- email_date, email_sender, email_subject are NULL for non-email documents (PDFs, Word files, etc.)
  Always filter with WHERE email_date IS NOT NULL to restrict to emails only.
- email_date is stored as text in 'YYYY-MM-DD HH:MM' format — use LIKE for partial date matches.
  Example: WHERE email_date LIKE '2024-03%' finds all emails from March 2024.
- CRITICAL — NAME FORMAT: email_sender is stored in "Last, First" format (e.g. "Deluna, Joe")
  or as an email address (e.g. "Scott.Larson@osii.com") or with suffixes (e.g. "Deluna, Joe [Contractor]").
  NEVER search for "First Last" format — it will never match.
  ALWAYS search by last name only using LIKE '%lastname%' to catch all variants.
  Example: user asks for "Joe Deluna" → use WHERE email_sender LIKE '%Deluna%'
  Example: user asks for "Scott Larson" → use WHERE email_sender LIKE '%Larson%'
  Example: user asks for "emails from John" → use WHERE email_sender LIKE '%John%''''

text = TARGET.read_text()
assert text.count(OLD_SCHEMA) == 1, (
    f"Schema block: expected 1 match, found {text.count(OLD_SCHEMA)}"
)
patched = text.replace(OLD_SCHEMA, NEW_SCHEMA)
TARGET.write_text(patched)
print("metadata_agent.py patched successfully.")
