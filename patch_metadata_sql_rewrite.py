#!/usr/bin/env python3
"""
patch_metadata_sql_rewrite.py
Adds a Python post-processing step to metadata_agent.py that rewrites
"First Last" name patterns in generated SQL to "Last" only.

Why Python instead of prompt instructions:
gemma4:e4b reliably ignores nuanced prompt instructions about name format.
Python string manipulation is deterministic and always works.

How it works:
After the LLM generates SQL like:
    WHERE email_sender LIKE '%Joe Deluna%'
The rewriter detects the two-word name pattern inside LIKE '%%' and
rewrites it to use only the last word (last name):
    WHERE email_sender LIKE '%Deluna%'

This catches "First Last", "First Last [Contractor]", etc.
It does not touch single-word LIKE patterns (already correct).
"""

from pathlib import Path

TARGET = Path("/home/bill/local-ai-doc-assistant/src/agents/metadata_agent.py")

# Insert the rewriter function before _generate_sql
OLD_GENERATE = "def _generate_sql(question: str, model: str = None) -> str:"

NEW_REWRITER = '''def _rewrite_name_likes(sql: str) -> str:
    """
    Post-process generated SQL to fix "First Last" name patterns in LIKE clauses.

    Outlook/PST stores names as "Last, First" format. When the LLM generates
    LIKE '%Joe Deluna%' it never matches. We rewrite it to LIKE '%Deluna%'
    (last name only) which matches "Deluna, Joe", "Deluna, Joe [Contractor]",
    and email addresses containing the last name.

    Only rewrites two-word name patterns inside LIKE '...%Name%...' clauses.
    Single-word patterns are left alone (already correct).
    """
    import re

    def replace_name(match):
        prefix = match.group(1)   # everything before the name
        name   = match.group(2)   # the name portion e.g. "Joe Deluna"
        suffix = match.group(3)   # everything after the name
        words = name.strip().split()
        if len(words) >= 2:
            # Use last word only (the surname)
            last_name = words[-1]
            print(f"  [Metadata] Name rewrite: '{name}' -> '{last_name}'")
            return f"{prefix}{last_name}{suffix}"
        return match.group(0)  # single word, leave alone

    # Match LIKE '%...Name...' patterns — handles quoted names with spaces
    pattern = r"(LIKE\s+'%?)([A-Za-z][A-Za-z\s\-']{2,30}?)(%')"
    return re.sub(pattern, replace_name, sql, flags=re.IGNORECASE)


def _generate_sql(question: str, model: str = None) -> str:'''

# Add the rewriter call inside _generate_sql, after the sql is cleaned
OLD_RETURN_SQL = '''        return sql

    except Exception as e:
        print(f"  [Metadata] SQL generation error: {e}")
        return ""'''

NEW_RETURN_SQL = '''        # Fix "First Last" name patterns → "Last" only
        # (Outlook stores names as "Last, First" format)
        sql = _rewrite_name_likes(sql)

        return sql

    except Exception as e:
        print(f"  [Metadata] SQL generation error: {e}")
        return ""'''

text = TARGET.read_text()

assert text.count(OLD_GENERATE) == 1, (
    f"_generate_sql def: expected 1 match, found {text.count(OLD_GENERATE)}"
)
assert text.count(OLD_RETURN_SQL) == 1, (
    f"return sql block: expected 1 match, found {text.count(OLD_RETURN_SQL)}"
)

text = text.replace(OLD_GENERATE, NEW_REWRITER)
text = text.replace(OLD_RETURN_SQL, NEW_RETURN_SQL)
TARGET.write_text(text)
print("metadata_agent.py patched successfully.")
