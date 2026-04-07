"""
history.py - Per-project chat history using SQLite.

Each project gets its own chat_history.db stored in its workspace folder.
This lets the agent remember what was said earlier in a conversation,
and lets users scroll back through past sessions.

Analogy: Think of this as a notebook per project. Every question and
answer gets written down with a timestamp. The agent reads the last
N entries before answering so it has conversation context.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from src.config import PATHS


def _get_db_path(project_name: str) -> Path:
    """Return the path to the chat history database for a project."""
    workspace = Path(PATHS["workspaces_root"]) / project_name
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace / "chat_history.db"


def _get_connection(project_name: str) -> sqlite3.Connection:
    """Open a connection to the project's chat history database."""
    db_path = _get_db_path(project_name)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_history(project_name: str) -> None:
    """
    Create the chat history table if it doesn't exist.
    Called automatically on first use.
    """
    conn = _get_connection(project_name)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                intent      TEXT,
                sources     TEXT,
                timestamp   TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session
            ON chat_history (session_id, timestamp)
        """)
        conn.commit()
    finally:
        conn.close()


def save_turn(project_name: str, session_id: str,
              user_message: str, assistant_reply: str,
              intent: str = None, sources: list = None) -> None:
    """
    Save one complete conversation turn (user + assistant) to history.

    Args:
        project_name: Which project workspace
        session_id: Unique ID for this conversation session
        user_message: What the user said
        assistant_reply: What the agent replied
        intent: Which agent handled it (teach/troubleshoot/check/lookup)
        sources: List of source documents used
    """
    init_history(project_name)
    conn = _get_connection(project_name)
    timestamp = datetime.utcnow().isoformat()
    sources_json = json.dumps(sources or [])

    try:
        conn.execute(
            """INSERT INTO chat_history
               (session_id, role, content, intent, sources, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, "user", user_message, intent, sources_json, timestamp)
        )
        conn.execute(
            """INSERT INTO chat_history
               (session_id, role, content, intent, sources, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, "assistant", assistant_reply, intent, sources_json, timestamp)
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_history(project_name: str, session_id: str,
                       max_turns: int = 6) -> list[dict]:
    """
    Retrieve the most recent N turns of a conversation.
    Returns a list of {role, content} dicts suitable for passing
    to an agent as chat_history.

    max_turns=6 means the last 6 messages (3 user + 3 assistant).
    This keeps context window usage reasonable.
    """
    init_history(project_name)
    conn = _get_connection(project_name)
    try:
        rows = conn.execute(
            """SELECT role, content FROM chat_history
               WHERE session_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (session_id, max_turns)
        ).fetchall()
        # Reverse so oldest is first (chronological order for the agent)
        return [{"role": r["role"], "content": r["content"]}
                for r in reversed(rows)]
    finally:
        conn.close()


def get_session_history(project_name: str,
                        session_id: str) -> list[dict]:
    """
    Retrieve the complete history for a session.
    Used for displaying chat history in the UI.
    """
    init_history(project_name)
    conn = _get_connection(project_name)
    try:
        rows = conn.execute(
            """SELECT role, content, intent, sources, timestamp
               FROM chat_history
               WHERE session_id = ?
               ORDER BY timestamp ASC""",
            (session_id,)
        ).fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "intent": r["intent"],
                "sources": json.loads(r["sources"] or "[]"),
                "timestamp": r["timestamp"]
            }
            for r in rows
        ]
    finally:
        conn.close()


def list_sessions(project_name: str) -> list[dict]:
    """
    List all conversation sessions for a project, newest first.
    Returns session_id and the first user message as a preview.
    """
    init_history(project_name)
    conn = _get_connection(project_name)
    try:
        rows = conn.execute(
            """SELECT session_id,
                      MIN(timestamp) as started,
                      COUNT(*) as message_count
               FROM chat_history
               WHERE role = 'user'
               GROUP BY session_id
               ORDER BY started DESC""",
        ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "started": r["started"],
                "message_count": r["message_count"]
            }
            for r in rows
        ]
    finally:
        conn.close()
