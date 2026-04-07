"""
teacher.py - Teaching agent.

Receives a teaching request and relevant document chunks, then uses
Gemma 4 31B to generate a clear step-by-step explanation with
source citations.

Why citations are non-negotiable:
For SCADA/EMS work, operators need to know WHICH document and section
a procedure came from so they can verify it themselves. An answer
without a source is just the AI guessing.

Why /api/generate instead of /api/chat:
Gemma 4 models use /api/chat to trigger extended thinking mode, which
consumes all available tokens on internal reasoning and returns empty
content. /api/generate bypasses this and returns actual answers.
"""

import requests
from src.config import OLLAMA, MODELS, RETRIEVAL
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank


TEACHER_SYSTEM_PROMPT = """You are a technical training assistant for SCADA/EMS operations.
Your job is to teach procedures clearly and accurately based ONLY on the
provided document excerpts.

Rules you must always follow:
1. Base your answer ONLY on the provided document excerpts. Do not invent steps.
2. Cite your source for each major step or claim, like this: [Source: filename.pdf, p.2]
3. If the documents don't contain enough information to answer fully, say so explicitly.
4. Structure your response as numbered steps when explaining a procedure.
5. If you see conflicting information between documents, flag it explicitly:
   WARNING: Document A says X but Document B says Y. Verify before proceeding.
6. End with a summary of which documents were used."""


def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.
    Each chunk is labeled with its source so the model can cite it.
    """
    if not chunks:
        return "No relevant documents found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        parts.append(f"[Excerpt {i} — Source: {source}, Page {page}]\n{text}")

    return "\n\n---\n\n".join(parts)


def teach(project_name: str, query: str,
          chat_history: list[dict] = None) -> dict:
    """
    Answer a teaching request using retrieved document chunks.

    Args:
        project_name: Which workspace to search
        query: The user's question or request
        chat_history: Optional list of prior messages for context
                      [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        {
            "answer": "Step-by-step explanation...",
            "sources": ["file1.pdf", "file2.txt"],
            "chunks_used": 3,
            "intent": "teach"
        }
    """
    # Step 1: Retrieve relevant chunks via hybrid search
    raw_results = hybrid_search(project_name, query,
                                top_k=RETRIEVAL["top_k"])
    chunks = rerank(raw_results, top_k_final=RETRIEVAL["top_k_final"])

    if not chunks:
        return {
            "answer": ("I could not find relevant information in the "
                       "project documents to answer your question. "
                       "Please ensure the relevant documents have been ingested."),
            "sources": [],
            "chunks_used": 0,
            "intent": "teach"
        }

    # Step 2: Build context block from retrieved chunks
    context = build_context(chunks)

    # Step 3: Build a single prompt string
    # /api/generate takes a prompt string, not a messages list.
    # We bake the system prompt, chat history, context, and query together.
    history_text = ""
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "user").capitalize()
            history_text += f"{role}: {msg.get('content', '')}\n"

    prompt = f"""{TEACHER_SYSTEM_PROMPT}

{history_text}
--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Question: {query}

Provide a clear, step-by-step explanation with citations."""

    # Step 4: Call Gemma 4 31B via /api/generate
    try:
        response = requests.post(
            f"{OLLAMA['base_url']}/api/generate",
            json={
                "model": MODELS["llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2048,
                    "num_ctx": 8192,
                }
            },
            timeout=OLLAMA["timeout_seconds"]
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()

    except Exception as e:
        answer = f"Error contacting language model: {str(e)}"

    # Step 5: Collect unique source filenames
    sources = list({chunk.get("source", "unknown") for chunk in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "intent": "teach"
    }
