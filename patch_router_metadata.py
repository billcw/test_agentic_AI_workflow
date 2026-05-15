#!/usr/bin/env python3
"""
patch_router_metadata.py
Adds "metadata" as a sixth intent to router.py.

Two changes:
1. Add INTENT_METADATA constant and add it to VALID_INTENTS
2. Add "metadata" as a category in the INTENT_PROMPT

The prompt language is carefully written to distinguish metadata queries
(database facts: counts, dates, senders) from content queries
(what did the emails SAY about X — those belong in RAG, not SQL).
"""

from pathlib import Path

TARGET = Path("/home/bill/local-ai-doc-assistant/src/agents/router.py")

# --- Patch 1: constants block ---
OLD_CONSTANTS = """INTENT_TEACH = "teach"
INTENT_TROUBLESHOOT = "troubleshoot"
INTENT_CHECK = "check"
INTENT_LOOKUP = "lookup"
INTENT_SENTIMENT = "sentiment"
VALID_INTENTS = {INTENT_TEACH, INTENT_TROUBLESHOOT, INTENT_CHECK, INTENT_LOOKUP, INTENT_SENTIMENT}"""

NEW_CONSTANTS = """INTENT_TEACH = "teach"
INTENT_TROUBLESHOOT = "troubleshoot"
INTENT_CHECK = "check"
INTENT_LOOKUP = "lookup"
INTENT_SENTIMENT = "sentiment"
INTENT_METADATA = "metadata"
VALID_INTENTS = {INTENT_TEACH, INTENT_TROUBLESHOOT, INTENT_CHECK, INTENT_LOOKUP, INTENT_SENTIMENT, INTENT_METADATA}"""

# --- Patch 2: add metadata category to INTENT_PROMPT ---
OLD_PROMPT_TAIL = """- lookup: User wants a specific fact, definition, value, or quick
  reference. Everything else that does not fit the above four.
  Examples: What is..., What does X mean?, Who is..., When did...

Respond with ONLY the single word category. No explanation. No punctuation.
Just one of: teach, troubleshoot, check, sentiment, lookup\""""

NEW_PROMPT_TAIL = """- lookup: User wants a specific fact, definition, value, or quick
  reference. Everything else that does not fit the above five.
  Examples: What is..., What does X mean?, Who is..., When did...

- metadata: User is asking about EMAIL DATABASE STATISTICS or properties
  that require counting, listing, or sorting emails by date/sender/subject.
  These are questions about the EMAIL ARCHIVE ITSELF, not about what the
  emails contain. Examples: How many emails do I have?, What is the oldest
  email?, Who sent the most emails?, Show me emails from John, What emails
  came in March 2024?, List senders in my archive.
  IMPORTANT: If the user asks what emails SAY about a topic (e.g. "What do
  emails say about the outage?") that is NOT metadata — classify it as the
  appropriate content intent (troubleshoot, lookup, etc.) instead.

Respond with ONLY the single word category. No explanation. No punctuation.
Just one of: teach, troubleshoot, check, sentiment, lookup, metadata\""""

text = TARGET.read_text()

assert text.count(OLD_CONSTANTS) == 1, (
    f"Constants block: expected 1 match, found {text.count(OLD_CONSTANTS)}"
)
assert text.count(OLD_PROMPT_TAIL) == 1, (
    f"Prompt tail: expected 1 match, found {text.count(OLD_PROMPT_TAIL)}"
)

patched = text.replace(OLD_CONSTANTS, NEW_CONSTANTS)
patched = patched.replace(OLD_PROMPT_TAIL, NEW_PROMPT_TAIL)
TARGET.write_text(patched)
print("router.py patched successfully.")
