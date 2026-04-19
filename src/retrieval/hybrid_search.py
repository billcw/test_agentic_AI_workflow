"""
hybrid_search.py - Combines semantic and keyword search results.
Why hybrid search?
- Semantic search (ChromaDB) finds conceptually similar chunks even if
  they don't use the exact same words as the query.
- Keyword search (BM25) finds chunks with exact technical term matches -
  critical for SCADA where terms like alrm_server or lpmd must be
  found precisely.
- Combining both gives better results than either alone.
The hybrid_weight config setting controls the balance:
  hybrid_weight = 0.7 means 70% semantic, 30% keyword (default)
  hybrid_weight = 1.0 means semantic only
  hybrid_weight = 0.0 means keyword only
MMR (Maximum Marginal Relevance):
- After scoring, MMR penalizes chunks from already-selected sources.
- mmr_diversity = 0.0 means no penalty (pure score ranking)
- mmr_diversity = 1.0 means maximum diversity enforcement
- Default 0.3 applies a 30% penalty to already-represented sources.
"""
from src.retrieval.semantic_search import semantic_search
from src.retrieval.keyword_search import keyword_search
from src.config import RETRIEVAL


def hybrid_search(project_name: str, query: str,
                  top_k: int = None,
                  hybrid_weight: float = None,
                  mmr_diversity: float = 0.3,
                  scope: str = "all") -> list[dict]:
    """
    Search using both semantic and keyword search, combining scores.
    Each search returns scores normalized to 0.0-1.0.
    Final score = (semantic_score * weight) + (keyword_score * (1 - weight))

    MMR pass after scoring penalizes chunks from already-selected sources
    by multiplying their score by (1 - mmr_diversity) before selection.
    This enforces source diversity without discarding relevant content.

    Returns list of dicts sorted by MMR-adjusted selection order.
    """
    if top_k is None:
        top_k = RETRIEVAL["top_k"]
    if hybrid_weight is None:
        hybrid_weight = RETRIEVAL["hybrid_weight"]

    semantic_results = semantic_search(project_name, query, top_k=top_k, scope=scope)
    keyword_results = keyword_search(project_name, query, top_k=top_k, scope=scope)

    semantic_map = {r["chunk_id"]: r for r in semantic_results}
    keyword_map = {r["chunk_id"]: r for r in keyword_results}

    all_chunk_ids = set(semantic_map.keys()) | set(keyword_map.keys())

    combined = []
    for chunk_id in all_chunk_ids:
        sem = semantic_map.get(chunk_id)
        kw = keyword_map.get(chunk_id)
        semantic_score = sem["score"] if sem else 0.0
        keyword_score = kw["score"] if kw else 0.0
        combined_score = (semantic_score * hybrid_weight) + \
                         (keyword_score * (1 - hybrid_weight))
        source = sem or kw
        combined.append({
            "chunk_id": chunk_id,
            "text": source["text"],
            "source": source["source"],
            "page": source["page"],
            "score": round(combined_score, 4),
            "semantic_score": round(semantic_score, 4),
            "keyword_score": round(keyword_score, 4)
        })

    # MMR selection: iteratively pick best chunk with diversity penalty
    combined.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    selected_sources = set()
    candidates = list(combined)

    while candidates and len(selected) < top_k:
        best_idx = None
        best_effective_score = -1.0
        for i, chunk in enumerate(candidates):
            if chunk["source"] in selected_sources:
                effective = chunk["score"] * (1.0 - mmr_diversity)
            else:
                effective = chunk["score"]
            if effective > best_effective_score:
                best_effective_score = effective
                best_idx = i
        chosen = candidates.pop(best_idx)
        selected.append(chosen)
        selected_sources.add(chosen["source"])

    return selected
