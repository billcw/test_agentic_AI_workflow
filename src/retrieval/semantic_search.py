"""
semantic_search.py - Semantic search wrapper for the retrieval layer.
This is a thin pass-through to vector_store.py. Having it here keeps
the retrieval layer self-contained -- agents import from src.retrieval,
not directly from src.storage.
"""
from src.storage.vector_store import semantic_search as _semantic_search
from src.config import RETRIEVAL

def scope_to_where(scope: str):
    """
    Convert a scope string to a ChromaDB where filter dict.
    email    -> filter to method == email
    document -> filter to method in [digital, ocr]
    all      -> no filter (None)
    """
    if scope == "email":
        return {"method": {"$eq": "email"}}
    elif scope == "document":
        return {"method": {"$in": ["digital", "ocr"]}}
    else:
        return None

def semantic_search(project_name: str, query: str, top_k: int = None,
                    scope: str = "all") -> list[dict]:
    """
    Search for chunks semantically similar to the query using ChromaDB.
    scope: email, document, or all -- filters by document method.
    Returns list of dicts with chunk_id, text, source, page, score.
    """
    if top_k is None:
        top_k = RETRIEVAL["top_k"]
    where = scope_to_where(scope)
    return _semantic_search(project_name, query, top_k, where=where)
