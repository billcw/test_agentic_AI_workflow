"""
keyword_search.py - Keyword search wrapper for the retrieval layer.
This is a thin pass-through to keyword_store.py. Having it here keeps
the retrieval layer self-contained -- agents import from src.retrieval,
not directly from src.storage.
"""
from src.storage.keyword_store import keyword_search as _keyword_search
from src.config import RETRIEVAL

SCOPE_TO_METHOD = {
    "email": ["email"],
    "document": ["digital", "ocr"],
    "all": None
}

def keyword_search(project_name: str, query: str, top_k: int = None,
                   scope: str = "all") -> list[dict]:
    """
    Search for chunks matching the query using BM25 keyword search.
    scope: email, document, or all -- filters by document method.
    Returns list of dicts with chunk_id, text, source, page, score.
    """
    if top_k is None:
        top_k = RETRIEVAL["top_k"]
    method_filter = SCOPE_TO_METHOD.get(scope, None)
    return _keyword_search(project_name, query, top_k, method_filter=method_filter)
