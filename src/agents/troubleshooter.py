"""
troubleshooter.py - Diagnosis agent.

Receives a troubleshooting request and relevant document chunks, then
uses Gemma 4 to diagnose the problem and suggest corrective actions.

Key difference from teacher.py:
The troubleshooter actively looks for contradictions between sources
and flags them explicitly. In SCADA/EMS work, conflicting procedures
are a safety issue -- the operator must know when documents disagree.
"""

import requests
from src.config import OLLAMA, MODELS, RETRIEVAL
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank
from src.agents.teacher import build_context


TROUBLESHOOTER_SYSTEM_PROMPT = """You are a SCADA/EMS troubleshooting assistant.
Your job is to help diagnose problems and suggest corrective actions
based ONLY on the provided document excerpts.

Rules you must always follow:
1. Base your diagnosis ONLY on the provided document excerpts.
2. Cite your source for each step or claim: [Source: filename, p.X]
3. Structure your response as:
   - LIKELY CAUSE: What is probably wrong
   - DIAGNOSTIC STEPS: How to confirm the cause
   - CORRECTIVE ACTIONS: What to do to fix it
   - ESCALATION: When to call for help
4. If documents contradict each other, flag it explicitly:
   *** CONFLICT: Document A says X but Document B says Y. Verify before acting. ***
5. If you are uncertain, say so explicitly. Never guess on safety-critical steps.
6. End with which documents were referenced."""


def troubleshoot(project_name: str, query: str,
                 chat_history: list[dict] = None) -> dict:
    """
    Diagnose a problem using retrieved document chunks.

    Args:
        project_name: Which workspace to search
        query: Description of the problem or error
        chat_history: Optional prior conversation context

    Returns:
        {
            "answer": "Diagnosis and corrective actions...",
            "sources": ["manual.pdf", "emails.txt"],
            "chunks_used": 4,
            "intent": "troubleshoot"
        }
    """
    # Step 1: Retrieve relevant chunks via hybrid search
    raw_results = hybrid_search(project_name, query,
                                top_k=RETRIEVAL["top_k"])
    chunks = rerank(raw_results, top_k_final=RETRIEVAL["top_k_final"])

    if not chunks:
        return {
            "answer": ("I could not find relevant troubleshooting information "
                       "in the project documents. Please ensure the relevant "
                       "manuals and notes have been ingested."),
            "sources": [],
            "chunks_used": 0,
            "intent": "troubleshoot"
        }

    # Step 2: Build context block
    context = build_context(chunks)

    # Step 3: Build prompt
    history_text = ""
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "user").capitalize()
            history_text += f"{role}: {msg.get('content', '')}\n"

    prompt = f"""{TROUBLESHOOTER_SYSTEM_PROMPT}

{history_text}
--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Problem reported: {query}

Provide a structured diagnosis with cited corrective actions."""

    # Step 4: Call the LLM via /api/generate
    try:
        response = requests.post(
            f"{OLLAMA['base_url']}/api/generate",
            json={
                "model": MODELS["llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                    "num_ctx": 4096,
                }
            },
            timeout=OLLAMA["timeout_seconds"]
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()

    except Exception as e:
        answer = f"Error contacting language model: {str(e)}"

    # Step 5: Collect unique sources
    sources = list({chunk.get("source", "unknown") for chunk in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "intent": "troubleshoot"
    }
