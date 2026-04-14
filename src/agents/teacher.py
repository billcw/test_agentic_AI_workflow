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

Chain of Thought + Confidence:
The system prompt instructs the model to reason step-by-step before
answering, and to append a CONFIDENCE line at the end. We parse that
line out and return it as a separate integer field (1-5) so the UI
can display a warning badge when confidence is low.
"""

import re
import requests
from src.config import OLLAMA, MODELS, RETRIEVAL
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank


TEACHER_SYSTEM_PROMPT = """You are a technical training assistant for SCADA/EMS operations.
Your job is to teach procedures clearly and accurately based ONLY on the
provided document excerpts.

Rules you must always follow:
1. Before giving your final answer, briefly think through what the documents
   say about this topic and what the key points are. Show this reasoning.
2. Base your answer ONLY on the provided document excerpts. Do not invent steps.
3. Cite your source for each major step or claim, like this: [Source: filename.pdf, p.2]
4. If the documents don't contain enough information to answer fully, say so explicitly.
5. Structure your response as numbered steps when explaining a procedure.
6. If you see conflicting information between documents, flag it explicitly:
   WARNING: Document A says X but Document B says Y. Verify before proceeding.
7. End with a summary of which documents were used.
8. END with exactly: CONFIDENCE: X/5
   Where X is 1 (very uncertain) to 5 (fully supported by documents)."""


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


def parse_confidence(answer: str) -> tuple[str, int]:
    """
    Extract the CONFIDENCE line from the end of the model's response.

    Returns:
        (cleaned_answer, confidence_score)
        cleaned_answer: the answer with the CONFIDENCE line removed
        confidence_score: integer 1-5, defaults to 3 if not parseable

    Why we strip it from the answer:
        Confidence is metadata for the UI, not part of the explanation
        the operator reads. Keeping them separate makes both cleaner.
    """
    # Match "CONFIDENCE: X/5 — anything" at end of response
    pattern = r'[\n\r]?\s*CONFIDENCE:\s*([1-5])/5[^\n]*$'
    match = re.search(pattern, answer.strip(), re.IGNORECASE)

    if match:
        score = int(match.group(1))
        cleaned = answer[:match.start()].strip()
        return cleaned, score

    # If the model didn't follow the format, default to 3 (neutral)
    return answer.strip(), 3


def teach(project_name: str, query: str,
          chat_history: list[dict] = None,
          model: str = None,
          top_k: int = None,
          top_k_final: int = None,
          chunks: list = None) -> dict:
    """
    Answer a teaching request using retrieved document chunks.

    Args:
        project_name: Which workspace to search
        query: The user's question or request
        chat_history: Optional list of prior messages for context
                      [{"role": "user"|"assistant", "content": "..."}]
        chunks: Pre-retrieved chunks from retrieval_node. If provided
                and non-empty, skips internal retrieval entirely.

    Returns:
        {
            "answer": "Step-by-step explanation...",
            "sources": ["file1.pdf", "file2.txt"],
            "chunks_used": 3,
            "confidence": 4,
            "intent": "teach"
        }
    """
    # Step 1: Use pre-retrieved chunks if provided, else retrieve now
    if not chunks:
        raw_results = hybrid_search(project_name, query,
                                    top_k=top_k or RETRIEVAL["top_k"])
        chunks = rerank(raw_results, top_k_final=top_k_final or RETRIEVAL["top_k_final"])

    if not chunks:
        return {
            "answer": ("I could not find relevant information in the "
                       "project documents to answer your question. "
                       "Please ensure the relevant documents have been ingested."),
            "sources": [],
            "chunks_used": 0,
            "confidence": 1,
            "intent": "teach"
        }

    # Step 2: Build context block from retrieved chunks
    context = build_context(chunks)

    # Step 3: Build a single prompt string
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

Think through what the documents say, then provide a clear, step-by-step
explanation with citations. End with your CONFIDENCE rating."""

    # Step 4: Call Gemma 4 31B via /api/generate
    try:
        response = requests.post(
            f"{OLLAMA['base_url']}/api/generate",
            json={
                "model": model or MODELS["llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 3000,
                    "num_ctx": 8192,
                }
            },
            timeout=OLLAMA["timeout_seconds"]
        )
        response.raise_for_status()
        raw_answer = response.json()["response"].strip()

    except Exception as e:
        return {
            "answer": f"Error contacting language model: {str(e)}",
            "sources": [],
            "chunks_used": len(chunks),
            "confidence": 1,
            "intent": "teach"
        }

    # Step 5: Parse confidence score out of the answer
    answer, confidence = parse_confidence(raw_answer)

    # Step 6: Collect unique source filenames
    sources = list({chunk.get("source", "unknown") for chunk in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "confidence": confidence,
        "intent": "teach"
    }
