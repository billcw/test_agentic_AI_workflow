"""
keyword_search.py - Keyword search wrapper for the retrieval layer.

This is a thin pass-through to keyword_store.py. Having it here keeps
the retrieval layer self-contained — agents import from src.retrieval,
not directly from src.storage.
"""

from src.storage.keyword_store import keyword_search as _keyword_search
from src.config import RETRIEVAL


def keyword_search(project_name: str, query: str, top_k: int = None) -> list[dict]:
    """
    Search for chunks matching the query using BM25 keyword search.

    Returns list of dicts:
    [
        {
            "chunk_id": "...",
            "text": "...",
            "source": "...",
            "page": 1,
            "score": 0.85   # 0.0 to 1.0, higher is better
        },
        ...
    ]
    """
    if top_k is None:
        top_k = RETRIEVAL["top_k"]
    return _keyword_search(project_name, query, top_k)
