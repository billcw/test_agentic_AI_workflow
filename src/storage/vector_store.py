"""
vector_store.py - ChromaDB wrapper for semantic vector storage.
Handles embedding generation via Ollama and vector storage/retrieval.
"""

import chromadb
from pathlib import Path
from src.config import PATHS, MODELS, OLLAMA, RETRIEVAL
import ollama as ollama_client


def get_chroma_client(project_name: str):
    """
    Get a ChromaDB client for a specific project workspace.
    Each project gets its own isolated vector database.
    """
    db_path = Path(PATHS["workspaces_root"]) / project_name / "vectors"
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def get_or_create_collection(project_name: str):
    """
    Get or create a ChromaDB collection for a project.
    Think of a collection like a table in a database.
    """
    client = get_chroma_client(project_name)
    return client.get_or_create_collection(
        name=project_name,
        metadata={"hnsw:space": "cosine"}  # Cosine similarity for text
    )


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for a piece of text using Ollama.
    An embedding is a list of numbers that represents the meaning of text.
    Similar texts produce similar vectors.
    """
    response = ollama_client.embeddings(
        model=MODELS["embeddings"],
        prompt=text
    )
    return response["embedding"]


def add_chunks(project_name: str, chunks: list[dict]) -> int:
    """
    Add a list of chunks to the vector store.
    Generates embeddings for each chunk and stores them.

    Returns the number of chunks added.
    """
    if not chunks:
        return 0

    collection = get_or_create_collection(project_name)

    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        # Skip if already indexed
        existing = collection.get(ids=[chunk_id])
        if existing["ids"]:
            continue

        # Generate embedding
        embedding = embed_text(chunk["text"])

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "method": chunk["method"]
        })

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    return len(ids)


def semantic_search(project_name: str, query: str, top_k: int = None) -> list[dict]:
    """
    Search the vector store for chunks semantically similar to the query.

    Returns a list of result dicts:
    [
        {
            "chunk_id": "...",
            "text": "...",
            "source": "...",
            "page": 1,
            "score": 0.92
        },
        ...
    ]
    """
    if top_k is None:
        top_k = RETRIEVAL["top_k"]

    collection = get_or_create_collection(project_name)

    # Check collection has documents
    if collection.count() == 0:
        return []

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "score": 1 - results["distances"][0][i]  # Convert distance to similarity
        })

    return output
