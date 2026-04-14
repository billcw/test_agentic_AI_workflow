"""
multi_turn.py - Multi-turn retrieval helper for agentic-v2.

Why multi-turn retrieval?
MMR (hybrid_search.py) enforces diversity when multiple sources score
competitively in a single pass. But when one source dominates the score
rankings entirely, MMR cannot help -- the second source never appears
in the candidate pool at all.

Multi-turn retrieval detects this situation and fires a second search
with a rephrased query to surface documents that the first pass missed.

How it works:
1. First pass: standard hybrid search + rerank
2. Diversity check: count unique source documents in results
3. If only 1 source represented: run second pass with refined query
4. Merge second pass results into first pass pool
5. Re-rerank the merged pool to get the best final set

The refined query adds context words to shift the semantic embedding
away from the dominant source and toward related documentation.

Max 2 passes total -- we never loop more than once.
"""

from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank
from src.config import RETRIEVAL


def run_retrieval(project_name: str, query: str,
                  top_k: int = None,
                  top_k_final: int = None,
                  hybrid_weight: float = None,
                  scope: str = "all") -> list[dict]:
    """
    Single-pass hybrid search + rerank.
    Returns the final chunks ready for an agent to use.
    scope: 'email', 'document', or 'all' -- filters retrieval by source type.
    """
    if top_k is None:
        top_k = RETRIEVAL["top_k"]
    if top_k_final is None:
        top_k_final = RETRIEVAL["top_k_final"]

    raw = hybrid_search(project_name, query, top_k=top_k,
                        hybrid_weight=hybrid_weight, scope=scope)
    return rerank(raw, top_k_final=top_k_final)


def check_diversity(chunks: list[dict]) -> bool:
    """
    Return True if all chunks come from a single source document.
    This is the trigger condition for a second retrieval pass.
    """
    if not chunks:
        return False
    sources = {chunk.get("source", "") for chunk in chunks}
    return len(sources) == 1


def refine_query(query: str) -> str:
    """
    Rephrase the query to broaden the second search pass.
    Adds context words that shift semantic search toward
    related documentation beyond the dominant source.
    """
    return f"detailed procedure background reference {query}"


def multi_turn_retrieve(project_name: str, query: str,
                        top_k: int = None,
                        top_k_final: int = None,
                        hybrid_weight: float = None,
                        scope: str = "all") -> tuple[list[dict], bool]:
    """
    Full multi-turn retrieval pipeline.

    Runs first pass, checks source diversity, and if only one source
    is represented fires a second pass with a refined query. Merges
    both passes and re-reranks to produce the final chunk set.

    Args:
        project_name: Which workspace to search
        query: The user's original query
        top_k: Candidates per search pass (default from config)
        top_k_final: Final chunks after reranking (default from config)

    Returns:
        (chunks, second_pass_fired)
        chunks: Final list of retrieved chunks for the agent
        second_pass_fired: True if a second retrieval pass was needed
    """
    if top_k is None:
        top_k = RETRIEVAL["top_k"]
    if top_k_final is None:
        top_k_final = RETRIEVAL["top_k_final"]

    # First pass
    first_chunks = run_retrieval(project_name, query,
                                 top_k=top_k, top_k_final=top_k_final,
                                 hybrid_weight=hybrid_weight, scope=scope)

    print(f"  [Retrieval] First pass: {len(first_chunks)} chunks from "
          f"{len({c.get('source') for c in first_chunks})} source(s)")

    # Diversity check
    if not check_diversity(first_chunks):
        # Multiple sources already represented -- no second pass needed
        return first_chunks, False

    # Second pass with refined query
    dominant_source = first_chunks[0].get("source", "unknown") if first_chunks else "unknown"
    print(f"  [Retrieval] Single source dominant: {dominant_source}")
    print(f"  [Retrieval] Firing second pass with refined query...")

    refined = refine_query(query)
    second_raw = hybrid_search(project_name, refined, top_k=top_k,
                               hybrid_weight=hybrid_weight, scope=scope)

    # Merge: combine first pass raw results with second pass raw results
    # Use chunk_id as key to deduplicate -- first pass score wins on conflict
    merged_map = {}
    for chunk in first_chunks:
        merged_map[chunk["chunk_id"]] = chunk
    for chunk in second_raw:
        if chunk["chunk_id"] not in merged_map:
            merged_map[chunk["chunk_id"]] = chunk

    merged = list(merged_map.values())
    merged.sort(key=lambda x: x["score"], reverse=True)

    # Re-rerank the merged pool
    final_chunks = rerank(merged, top_k_final=top_k_final)

    second_sources = {c.get("source") for c in final_chunks}
    print(f"  [Retrieval] After second pass: {len(final_chunks)} chunks from "
          f"{len(second_sources)} source(s)")

    return final_chunks, True
