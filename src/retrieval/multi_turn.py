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
1. Query rewriting: gemma4:e4b generates 3 alternative phrasings
2. First pass: hybrid search on original + all 3 rewrites (4 total)
3. Merge all results by chunk_id (deduplication), re-rerank merged pool
4. Diversity check: count unique source documents in results
5. If only 1 source represented: run second pass with refined query
6. Merge second pass results into pool, re-rerank final set

Query rewriting improves retrieval when the user's phrasing differs
from how the source documents are written (e.g. "reset AGC" vs
"restore automatic generation control to service"). Running all
phrasings through hybrid search and merging by chunk_id means we
surface chunks that any phrasing would find, without duplicates.

Max 2 passes total for diversity -- we never loop more than once.
"""

import re
import requests
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank
from src.config import RETRIEVAL

OLLAMA_URL = "http://localhost:11434/api/generate"
REWRITE_MODEL = "gemma4:e4b"


def rewrite_query(query: str) -> list[str]:
    """
    Use gemma4:e4b to generate 3 alternative phrasings of the query.

    Why alternative phrasings?
    Users phrase questions in natural conversational language. Source
    documents use formal technical language. The gap between the two
    causes BM25 to miss relevant chunks entirely (no word overlap)
    and weakens semantic search too. Generating alternatives that
    sound more like documentation language bridges this gap.

    Returns a list of up to 3 alternative query strings.
    If the LLM call fails for any reason, returns an empty list
    so the caller can fall back to the original query alone.
    """
    prompt = (
            "Output exactly 3 alternative phrasings of the query below. "
            "Each alternative must be a single plain sentence. "
            "Output only the 3 sentences, one per line, numbered 1. 2. 3. "
            "No headers, no labels, no explanations, no preamble.\n\n"
            f"Query: {query}\n\n"
            "1."
        )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": REWRITE_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 1000
                }
            },
            timeout=60
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        # Parse numbered lines: "1. ...", "2. ...", "3. ..."
        alternatives = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading number and punctuation: "1.", "1)", "1 -", etc.
            cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
            if cleaned and len(cleaned) > 10:
                alternatives.append(cleaned)
            if len(alternatives) == 3:
                break

        print(f"  [QueryRewrite] Generated {len(alternatives)} alternatives")
        for i, alt in enumerate(alternatives, 1):
            print(f"  [QueryRewrite] {i}: {alt}")

        return alternatives

    except Exception as e:
        print(f"  [QueryRewrite] Failed: {e} — using original query only")
        return []


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


def merge_by_chunk_id(primary: list[dict], *others: list[dict]) -> list[dict]:
    """
    Merge multiple result lists by chunk_id.
    Primary list scores win on conflict -- later lists only contribute
    chunks not already present in the primary pool.
    Returns a flat list sorted by score descending.
    """
    merged_map = {}
    for chunk in primary:
        merged_map[chunk["chunk_id"]] = chunk
    for result_list in others:
        for chunk in result_list:
            if chunk["chunk_id"] not in merged_map:
                merged_map[chunk["chunk_id"]] = chunk
    merged = list(merged_map.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


def multi_turn_retrieve(project_name: str, query: str,
                        top_k: int = None,
                        top_k_final: int = None,
                        hybrid_weight: float = None,
                        scope: str = "all") -> tuple[list[dict], bool]:
    """
    Full multi-turn retrieval pipeline with query rewriting.

    Step 1 — Query rewriting:
        gemma4:e4b generates 3 alternative phrasings of the query.
        hybrid_search() runs on all 4 queries (original + 3 rewrites).
        Results are merged by chunk_id and re-ranked once.

    Step 2 — Diversity check:
        If the merged pool comes from only 1 source document, fire a
        second pass with a broadened query. Merge again, re-rank again.

    Args:
        project_name: Which workspace to search
        query: The user's original query
        top_k: Candidates per search pass (default from config)
        top_k_final: Final chunks after reranking (default from config)
        hybrid_weight: Semantic/keyword balance (default from config)
        scope: 'email', 'document', or 'all'

    Returns:
        (chunks, second_pass_fired)
        chunks: Final list of retrieved chunks for the agent
        second_pass_fired: True if a diversity second pass was fired
    """
    if top_k is None:
        top_k = RETRIEVAL["top_k"]
    if top_k_final is None:
        top_k_final = RETRIEVAL["top_k_final"]

    # --- Step 1: Query rewriting ---
    print(f"  [Retrieval] Starting query rewriting...")
    alternatives = rewrite_query(query)
    all_queries = [query] + alternatives

    # Run hybrid_search on all queries, merge by chunk_id
    all_raw = []
    for i, q in enumerate(all_queries):
        label = "original" if i == 0 else f"rewrite {i}"
        print(f"  [Retrieval] Searching with {label}: '{q[:70]}'")
        results = hybrid_search(project_name, q, top_k=top_k,
                                hybrid_weight=hybrid_weight, scope=scope)
        all_raw.append(results)

    # Original query results are primary -- their scores win on conflict
    primary = all_raw[0]
    rewrites = all_raw[1:]
    merged_raw = merge_by_chunk_id(primary, *rewrites)

    print(f"  [Retrieval] Merged pool: {len(merged_raw)} unique chunks "
          f"from {len({c.get('source') for c in merged_raw})} source(s)")

    # Re-rank the merged pool
    first_chunks = rerank(merged_raw, top_k_final=top_k_final)

    print(f"  [Retrieval] After rerank: {len(first_chunks)} chunks from "
          f"{len({c.get('source') for c in first_chunks})} source(s)")

    # --- Step 2: Diversity check ---
    if not check_diversity(first_chunks):
        return first_chunks, False

    # Single source dominant -- fire second pass with broadened query
    dominant_source = first_chunks[0].get("source", "unknown") if first_chunks else "unknown"
    print(f"  [Retrieval] Single source dominant: {dominant_source}")
    print(f"  [Retrieval] Firing second pass with refined query...")

    refined = refine_query(query)
    second_raw = hybrid_search(project_name, refined, top_k=top_k,
                               hybrid_weight=hybrid_weight, scope=scope)

    # Merge second pass into existing pool -- first pass scores win
    final_merged = merge_by_chunk_id(first_chunks, second_raw)
    final_chunks = rerank(final_merged, top_k_final=top_k_final)

    second_sources = {c.get("source") for c in final_chunks}
    print(f"  [Retrieval] After second pass: {len(final_chunks)} chunks from "
          f"{len(second_sources)} source(s)")

    return final_chunks, True
