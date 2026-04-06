"""
keyword_store.py - BM25 keyword index for exact term matching.

Why BM25Plus instead of BM25Okapi:
BM25Okapi's IDF formula can produce 0.0 scores for small corpora (fewer
than ~10 documents) because its epsilon floor causes terms that appear in
all documents to score zero. BM25Plus uses a lower-bound IDF formula that
always produces positive scores, making it reliable at any corpus size.
"""

import json
from pathlib import Path
from rank_bm25 import BM25Plus
from src.config import PATHS


def get_index_path(project_name: str) -> Path:
    index_dir = Path(PATHS["workspaces_root"]) / project_name / "bm25_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def save_index(project_name: str, chunks: list[dict]) -> None:
    index_path = get_index_path(project_name)
    with open(index_path / "chunks.json", "w") as f:
        json.dump(chunks, f)
    print(f"  BM25 index saved: {len(chunks)} chunks indexed")


def load_index(project_name: str):
    index_path = get_index_path(project_name)
    chunks_file = index_path / "chunks.json"
    if not chunks_file.exists():
        return None, None
    with open(chunks_file, "r") as f:
        chunks = json.load(f)
    if not chunks:
        return None, None
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Plus(tokenized_corpus)
    return bm25, chunks


def keyword_search(project_name: str, query: str, top_k: int = 10) -> list[dict]:
    bm25, chunks = load_index(project_name)
    if bm25 is None or not chunks:
        return []
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]
    results = []
    max_score = max(scores) if max(scores) > 0 else 1
    for idx in top_indices:
        if scores[idx] > 0:
            chunk = chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(scores[idx] / max_score)
            })
    return results
