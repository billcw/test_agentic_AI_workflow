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
Your job is to teach procedures accurately based ONLY on the provided document excerpts.

CRITICAL RULES — violating these is worse than giving no answer:
1. THINK FIRST: Before answering, briefly note what each excerpt actually says.
   Do not summarize what you know about the topic — only what the excerpts say.
2. ONLY WHAT IS WRITTEN: Every step you provide must be directly stated in an excerpt.
   If a step is implied but not explicitly written, DO NOT include it.
   Do not use your general knowledge to fill gaps between steps.
3. CITE EVERY STEP: After each step, cite its source like this: [Source: filename.pdf, p.2]
   If you cannot cite a step, remove it from your answer.
4. INCOMPLETE IS HONEST: If the excerpts only cover part of a procedure, say explicitly:
   "The provided documents cover steps X through Y only. Steps Z onward are not in the excerpts."
   Do NOT complete the procedure from memory.
5. NEVER SYNTHESIZE: Do not combine fragments from different excerpts into a procedure
   that no single document states end-to-end. If the full procedure is not in one source,
   present only what each source explicitly states, separately.
6. CONTRADICTIONS: If documents conflict, flag it:
   WARNING: [source A] says X but [source B] says Y. Verify before proceeding.
7. STRUCTURE: Use numbered steps for procedures. Plain paragraphs for explanations.
8. CLOSE WITH: A one-line list of documents used, then exactly: CONFIDENCE: X/5
   Where X reflects how completely the excerpts support your answer (not your general knowledge)."""


def _clean_chunk_text(text: str, max_chars: int = 400) -> str:
    """
    Clean a single chunk's text before feeding it to the LLM.

    Why this matters:
    PST email archives contain large amounts of content that is useless
    to the LLM and consumes precious context window tokens:
    - urldefense.com encoded URLs: long base64-like strings that carry
      no semantic meaning (e.g. urldefense.com/v3/__https://u1107...)
    - mailto: encoded link fragments scattered throughout reply chains
    - Lines of pure alphanumeric noise (base64, encoded tokens, etc.)
    - Excessive blank lines from email formatting

    With num_ctx=8192, feeding 10 chunks of raw noisy email text can
    overflow the context window, causing the model to loop and repeat
    phrases endlessly. Capping at 400 chars per chunk after cleaning
    keeps total context well within safe limits while preserving the
    meaningful content the LLM actually needs.

    Args:
        text: Raw chunk text from ChromaDB
        max_chars: Maximum characters to keep after cleaning (default 400)

    Returns:
        Cleaned, truncated text. Empty string if nothing useful remains.
    """
    if not text:
        return ""

    # Remove urldefense.com encoded URLs entirely — these are never
    # useful to the LLM. They look like:
    # https://urldefense.com/v3/__https://u11074663.ct.sendgrid...
    text = re.sub(r'https?://urldefense\.com\S+', '[URL removed]', text)

    # Remove mailto: encoded links — fragments like:
    # <mailto:Victoria.Robinson@lge-ku.com>
    text = re.sub(r'<mailto:[^>]+>', '', text)

    # Remove bare mailto: references without angle brackets
    text = re.sub(r'mailto:\S+', '', text)

    # Remove lines that are pure encoded noise — lines with no spaces
    # and longer than 40 characters are almost always base64/token garbage
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep the line if it has spaces (real text) or is short
        if ' ' in stripped or len(stripped) <= 40:
            cleaned_lines.append(line)
        # Otherwise it's likely an encoded token — skip it
    text = '\n'.join(cleaned_lines)

    # Collapse runs of 3+ blank lines down to a single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse runs of whitespace within lines
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Cap at max_chars — truncate with an ellipsis so the LLM knows
    # the text continues beyond what it can see
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    return text


def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.
    Each chunk is labeled with its source so the model can cite it.

    Chunks are cleaned before assembly to remove URL-encoded noise
    and capped at 400 characters each to prevent context overflow.
    Chunks that are empty after cleaning are skipped entirely.
    """
    if not chunks:
        return "No relevant documents found."

    parts = []
    skipped = 0
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")

        cleaned = _clean_chunk_text(text, max_chars=400)

        if not cleaned or len(cleaned) < 20:
            skipped += 1
            continue

        parts.append(f"[Excerpt {i} — Source: {source}, Page {page}]\n{cleaned}")

    if skipped > 0:
        print(f"  [Context] Skipped {skipped} empty/noise chunks after cleaning")

    if not parts:
        return "No usable document content found after filtering noise."

    print(f"  [Context] Built context from {len(parts)} chunks, "
          f"~{sum(len(p) for p in parts)} chars total")

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
