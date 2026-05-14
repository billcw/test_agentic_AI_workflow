"""
metadata_agent.py - Metadata query agent for the Local AI Document Assistant.

Handles queries about EMAIL METADATA — facts that live in the database,
not in the document content itself. Examples:

    "What is the oldest email?"         → SQL on email_date column
    "How many emails do I have?"        → SQL COUNT on chunks table
    "Who sent the most emails?"         → SQL GROUP BY email_sender
    "Show me emails from John"          → SQL WHERE email_sender LIKE '%John%'
    "What emails came in March 2024?"   → SQL WHERE email_date LIKE '2024-03%'

These are NOT content questions — they cannot be answered by RAG retrieval.
They are database questions answered by running SQL against metadata.db.

Why a dedicated agent for this?
RAG works by finding chunks of text that are semantically similar to the
query. But "how many emails" has no semantically similar chunk — the answer
is a COUNT() that lives only in the database. Sending metadata queries
through the RAG pipeline produces hallucinated or empty answers.

The agent flow:
1. LLM (gemma4:e4b) translates natural language → SQL
2. SQL runs against the project's metadata.db chunks table
3. LLM (gemma4:e4b) formats the raw result into a plain-English answer
4. Returns answer with no sources (it came from the database, not documents)

Safety: Only SELECT statements are allowed. Any attempt to run INSERT,
UPDATE, DELETE, DROP, or any other mutating statement is refused.
"""

import re
import sqlite3
import requests
from pathlib import Path

from src.config import OLLAMA, MODELS, PATHS


# --- Schema description fed to the LLM so it knows what columns exist ---

SCHEMA_DESCRIPTION = """
The SQLite table is named 'chunks'. It has these columns relevant to email metadata:

    chunk_id      TEXT    -- unique ID for each chunk
    source        TEXT    -- filename the chunk came from (e.g. 'inbox.pst')
    email_date    TEXT    -- date the email was sent, format 'YYYY-MM-DD HH:MM' (NULL for non-emails)
    email_sender  TEXT    -- From: address/name of the email sender (NULL for non-emails)
    email_subject TEXT    -- Subject: line of the email (NULL for non-emails)
    ingested_at   TEXT    -- when WE processed the file (not the email sent date)

Important notes:
- Each email may produce MULTIPLE chunks (rows) with the same email_date/email_sender/email_subject.
  Use DISTINCT or GROUP BY when counting emails, not rows.
- email_date, email_sender, email_subject are NULL for non-email documents (PDFs, Word files, etc.)
  Always filter with WHERE email_date IS NOT NULL to restrict to emails only.
- email_date is stored as text in 'YYYY-MM-DD HH:MM' format — use LIKE for partial date matches.
  Example: WHERE email_date LIKE '2024-03%' finds all emails from March 2024.
- email_sender may contain full name and address like 'John Smith <john@example.com>'
  Use LIKE '%john%' for case-insensitive partial matching.
"""

SQL_GENERATION_PROMPT = """You are a SQL query generator for a SQLite database.
Your job is to translate a natural language question into a single valid SQL SELECT statement.

{schema}

Rules:
- Write ONLY a single SQL SELECT statement. Nothing else.
- No explanation, no markdown, no backticks, no comments.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, or any mutating statement.
- Always include WHERE email_date IS NOT NULL unless the user is explicitly asking about non-email documents.
- Use DISTINCT when counting unique emails to avoid counting multiple chunks from the same email.
- Keep the query simple and correct. Do not over-engineer it.
- If the question cannot be answered with the available columns, write:
  SELECT 'This question cannot be answered from email metadata alone.' as answer

User question: {question}

SQL:"""

ANSWER_FORMAT_PROMPT = """You are a helpful assistant explaining database query results.
The user asked: {question}

The SQL query returned this result:
{result}

Write a clear, direct, plain-English answer to the user's question based on this result.
Be concise. If the result is empty, say no matching emails were found.
Do not mention SQL or databases in your answer.
"""


def _generate_sql(question: str, model: str = None) -> str:
    """
    Ask the LLM to translate a natural language question into SQL.
    Returns the raw SQL string.
    """
    prompt = SQL_GENERATION_PROMPT.format(
        schema=SCHEMA_DESCRIPTION,
        question=question
    )

    try:
        response = requests.post(
            OLLAMA["base_url"] + "/api/generate",
            json={
                "model": model or MODELS["router_llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 500,
                }
            },
            timeout=OLLAMA["timeout_seconds"]
        )
        response.raise_for_status()
        sql = response.json()["response"].strip()

        # Strip markdown fences if the model wrapped the SQL anyway
        sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*", "", sql)
        sql = sql.strip()

        return sql

    except Exception as e:
        print(f"  [Metadata] SQL generation error: {e}")
        return ""


def _is_safe_sql(sql: str) -> bool:
    """
    Safety check: only allow SELECT statements.
    Refuses any mutating SQL regardless of how it was generated.
    """
    if not sql:
        return False
    # Strip leading whitespace/comments and check first keyword
    first_word = sql.strip().split()[0].upper()
    return first_word == "SELECT"


def _run_sql(project_name: str, sql: str) -> list[dict]:
    """
    Execute the SQL against the project's metadata.db.
    Returns a list of row dicts.
    """
    db_path = Path(PATHS["workspaces_root"]) / project_name / "metadata.db"

    if not db_path.exists():
        raise FileNotFoundError(f"metadata.db not found for project '{project_name}'")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()


def _format_answer(question: str, rows: list[dict], model: str = None) -> str:
    """
    Ask the LLM to turn raw SQL results into a plain-English answer.
    Falls back to a simple formatted result if the LLM call fails.
    """
    if not rows:
        result_text = "(no results found)"
    else:
        # Format rows as a simple readable table
        lines = []
        for row in rows[:50]:  # Cap at 50 rows to avoid token overload
            lines.append("  " + " | ".join(f"{k}: {v}" for k, v in row.items()))
        result_text = "\n".join(lines)
        if len(rows) > 50:
            result_text += f"\n  ... and {len(rows) - 50} more rows"

    prompt = ANSWER_FORMAT_PROMPT.format(
        question=question,
        result=result_text
    )

    try:
        response = requests.post(
            OLLAMA["base_url"] + "/api/generate",
            json={
                "model": model or MODELS["router_llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 500,
                }
            },
            timeout=OLLAMA["timeout_seconds"]
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()
        return answer if answer else result_text

    except Exception as e:
        print(f"  [Metadata] Answer formatting error: {e}")
        # Fall back to raw result if LLM call fails
        return f"Query results:\n{result_text}"


def run_metadata_query(project_name: str, query: str,
                       model: str = None) -> dict:
    """
    Main entry point for the metadata agent.

    Translates a natural language query into SQL, runs it against
    the project database, and returns a plain-English answer.

    Returns:
    {
        "answer":      "There are 1,247 emails in the archive...",
        "intent":      "metadata",
        "sources":     [],
        "chunks_used": 0,
        "confidence":  4,
        "sql":         "SELECT COUNT(DISTINCT source)..."   # for debugging
    }
    """
    print(f"  [Metadata] Generating SQL for: '{query[:60]}'")

    # Step 1: Generate SQL
    sql = _generate_sql(query, model=model)
    print(f"  [Metadata] Generated SQL: {sql[:120]}")

    # Step 2: Safety check
    if not _is_safe_sql(sql):
        return {
            "answer": "I was unable to generate a safe database query for that question. "
                      "Please try rephrasing, or ask a content question instead.",
            "intent": "metadata",
            "sources": [],
            "chunks_used": 0,
            "confidence": 1,
            "sql": sql
        }

    # Step 3: Run the SQL
    try:
        rows = _run_sql(project_name, sql)
        print(f"  [Metadata] Query returned {len(rows)} row(s)")
    except Exception as e:
        print(f"  [Metadata] SQL execution error: {e}")
        return {
            "answer": f"The database query failed: {str(e)}",
            "intent": "metadata",
            "sources": [],
            "chunks_used": 0,
            "confidence": 1,
            "sql": sql
        }

    # Step 4: Format the answer
    answer = _format_answer(query, rows, model=model)

    return {
        "answer": answer,
        "intent": "metadata",
        "sources": [],
        "chunks_used": 0,
        "confidence": 4,
        "sql": sql  # Included for debugging — not shown in UI
    }
