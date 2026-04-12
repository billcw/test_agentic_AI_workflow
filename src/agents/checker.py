"""
checker.py - Work verification agent.

The user describes what they did or plan to do. This agent retrieves
the official procedure and compares the two, flagging any steps that
were missed, done out of order, or done incorrectly.

This is the safety net agent -- it is deliberately conservative and
always flags its own uncertainty rather than silently approving work.

Chain of Thought + Confidence:
Same pattern as teacher.py -- model reasons before answering and
appends a CONFIDENCE line that we parse out for the UI.
"""

import requests
from src.config import OLLAMA, MODELS, RETRIEVAL
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank
from src.agents.teacher import build_context, parse_confidence


CHECKER_SYSTEM_PROMPT = """You are a SCADA/EMS work verification assistant.
Your job is to compare what an operator did against the official procedure
from the provided document excerpts.

Rules you must always follow:
1. Before giving your verdict, briefly think through what the official
   procedure requires and how it compares to what the operator reported.
   Show this reasoning before your structured answer.
2. Base your verification ONLY on the provided document excerpts.
3. Cite the official procedure source for every comparison: [Source: filename, p.X]
4. Structure your response as:
   - STEPS COMPLETED CORRECTLY: What was done right
   - STEPS MISSED: Required steps not mentioned by the operator
   - STEPS OUT OF ORDER: Correct actions done in wrong sequence
   - POTENTIAL RISKS: Safety concerns based on what was reported
   - OVERALL VERDICT: APPROVED / NEEDS REVIEW / UNSAFE
5. If you cannot find the official procedure in the documents, say so explicitly.
6. If you are uncertain about any comparison, flag it:
   UNCERTAIN: I cannot confirm this step from the available documents.
7. Never approve work you cannot verify against a document.
8. On the very last line of your response, write your confidence rating in
   this exact format (nothing else on that line):
   CONFIDENCE: X/5 — brief reason
   Where X is 1 (very uncertain) to 5 (fully supported by documents)."""


def check(project_name: str, query: str,
          chat_history: list[dict] = None,
          model: str = None,
          top_k: int = None,
          top_k_final: int = None,
          chunks: list = None) -> dict:
    """
    Verify operator work against official procedures.

    Args:
        project_name: Which workspace to search
        query: Description of what the operator did or plans to do
        chat_history: Optional prior conversation context
        chunks: Pre-retrieved chunks from retrieval_node. If provided
                and non-empty, skips internal retrieval entirely.

    Returns:
        {
            "answer": "Verification result with citations...",
            "sources": ["procedure.pdf"],
            "chunks_used": 3,
            "confidence": 4,
            "intent": "check"
        }
    """
    # Step 1: Use pre-retrieved chunks if provided, else retrieve now
    if not chunks:
        raw_results = hybrid_search(project_name, query,
                                    top_k=top_k or RETRIEVAL["top_k"])
        chunks = rerank(raw_results, top_k_final=top_k_final or RETRIEVAL["top_k_final"])

    if not chunks:
        return {
            "answer": ("I could not find the relevant official procedure "
                       "in the project documents. I cannot verify this work "
                       "without a source document to compare against."),
            "sources": [],
            "chunks_used": 0,
            "confidence": 1,
            "intent": "check"
        }

    # Step 2: Build context block
    context = build_context(chunks)

    # Step 3: Build prompt
    history_text = ""
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "user").capitalize()
            history_text += f"{role}: {msg.get('content', '')}\n"

    prompt = f"""{CHECKER_SYSTEM_PROMPT}

{history_text}
--- OFFICIAL PROCEDURE EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Operator's reported actions: {query}

Think through how the operator's actions compare to the official procedure,
then provide a structured verification with citations. End with your
CONFIDENCE rating."""

    # Step 4: Call the LLM via /api/generate
    try:
        response = requests.post(
            f"{OLLAMA['base_url']}/api/generate",
            json={
                "model": model or MODELS["llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2500,
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
            "intent": "check"
        }

    # Step 5: Parse confidence score out of the answer
    answer, confidence = parse_confidence(raw_answer)

    # Step 6: Collect unique sources
    sources = list({chunk.get("source", "unknown") for chunk in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "confidence": confidence,
        "intent": "check"
    }
