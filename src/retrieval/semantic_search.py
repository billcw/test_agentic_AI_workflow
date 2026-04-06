"""
semantic_search.py - Semantic search wrapper for the retrieval layer.

This is a thin pass-through to vector_store.py. Having it here keeps
the retrieval layer self-contained — agents import from src.retrieval,
not directly from src.storage.
"""

from src.storage.vector_store import semantic_search as _semantic_search
from src.config import RETRIEVAL


def semantic_search(project_name: str, query: str, top_k: int = None) -> list[dict]:
    """
    Search for chunks semantically similar to the query using ChromaDB.

    Returns list of dicts:
    [
        {
            "chunk_id": "...",
            "text": "...",
            "source": "...",
            "page": 1,
            "score": 0.92   # 0.0 to 1.0, higher is better
        },
        ...
    ]
    """
    if top_k is None:
        top_k = RETRIEVAL["top_k"]
    return _semantic_search(project_name, query, top_k)
