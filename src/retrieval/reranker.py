"""
reranker.py - Filters hybrid search results down to the best chunks.

Why a re-ranker?
Hybrid search returns up to top_k results (default 10). The re-ranker
trims that down to top_k_final (default 5) — the chunks actually sent
to the agent. This keeps the agent's context window focused on the
most relevant content rather than padding it with marginal results.

This is a score-based re-ranker (no second model needed). A future
upgrade could use a cross-encoder model for deeper re-ranking, but
for this use case score filtering is fast and effective.
"""

from src.config import RETRIEVAL


def rerank(results: list[dict], top_k_final: int = None) -> list[dict]:
    """
    Take hybrid search results and return only the top N by score.

    Args:
        results: Output from hybrid_search() - already sorted by score
        top_k_final: How many chunks to return (default from config)

    Returns the top N chunks, preserving all fields from hybrid_search.
    """
    if top_k_final is None:
        top_k_final = RETRIEVAL["top_k_final"]

    # Results are already sorted by score from hybrid_search.
    # We just trim to top_k_final and filter out anything with score 0.
    filtered = [r for r in results if r["score"] > 0]
    return filtered[:top_k_final]
