"""
sentiment.py - Sentiment analysis agent for agentic-v2.

Analyzes emotional tone, urgency, and mood in retrieved documents.
Uses the 31B reasoning model for deep analysis, with a keyword-based
fallback if the LLM call fails.

Why LLM-first for sentiment?
Keyword counting is brittle -- "not urgent" contains the word "urgent"
and would score as urgent. The LLM understands context and nuance that
simple word matching cannot.
"""
import requests
from src.config import OLLAMA, MODELS, RETRIEVAL
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank
from src.agents.teacher import build_context, parse_confidence

SENTIMENT_PROMPT = """You are a sentiment and tone analysis specialist.
Analyze the emotional tone, urgency, and mood expressed in the provided documents.

Your analysis must include:
1. Overall sentiment: Positive, Negative, Neutral, Mixed, or Urgent
2. Key emotional themes detected (frustration, satisfaction, urgency, concern, etc.)
3. Notable examples from the text that support your assessment
4. Any communications that require immediate attention or escalation

Always cite the source document for each observation.
Be specific -- quote or closely paraphrase the relevant passages.

End your response with:
CONFIDENCE: X/5 -- brief reason

Where X is your confidence that the retrieved documents are sufficient
to answer the user query (1=insufficient, 5=comprehensive coverage)."""


def _keyword_fallback(chunks: list) -> tuple[str, int]:
    """
    Simple keyword-based fallback if LLM call fails.
    Returns (summary_text, confidence_score).
    """
    positive_words = ["good", "great", "excellent", "happy", "pleased", "thank", "resolved"]
    negative_words = ["bad", "terrible", "frustrated", "angry", "problem", "issue", "failed"]
    urgent_words = ["urgent", "asap", "immediately", "emergency", "critical", "escalate"]

    total_text = " ".join([chunk.get("text", "") for chunk in chunks]).lower()

    pos_count = sum(1 for word in positive_words if word in total_text)
    neg_count = sum(1 for word in negative_words if word in total_text)
    urg_count = sum(1 for word in urgent_words if word in total_text)

    if urg_count > 0:
        primary = "Urgent"
    elif neg_count > pos_count:
        primary = "Negative"
    elif pos_count > 0:
        primary = "Positive"
    else:
        primary = "Neutral"

    summary = (f"Keyword-based sentiment analysis of {len(chunks)} documents:\n\n"
               f"Primary sentiment: {primary}\n"
               f"Indicators: {pos_count} positive, {neg_count} negative, {urg_count} urgent")
    return summary, 2


def analyze_sentiment(project_name: str, query: str,
                      chat_history: list = None,
                      model: str = None,
                      top_k: int = None,
                      top_k_final: int = None,
                      chunks: list = None,
                      email_max_chars: int = None,
                      doc_max_chars: int = None) -> dict:
    """
    Analyze sentiment and emotional tone in retrieved documents.

    Args:
        project_name: Which workspace to search
        query: The user's sentiment analysis request
        chat_history: Optional prior conversation context
        chunks: Pre-retrieved chunks from retrieval_node. If provided
                and non-empty, skips internal retrieval entirely.
        email_max_chars: Character cap for email chunks passed to
                         build_context(). If None, uses default (600).
        doc_max_chars: Character cap for document chunks passed to
                       build_context(). If None, uses default (600).
    """
    if not chunks:
        raw_results = hybrid_search(project_name, query, top_k=top_k or 15)
        chunks = rerank(raw_results, top_k_final=top_k_final or 10)

    if not chunks:
        return {
            "answer": "No documents found for sentiment analysis.",
            "sources": [],
            "chunks_used": 0,
            "confidence": 1,
            "intent": "sentiment"
        }

    # Build context block, passing through UI-supplied chunk caps
    build_kwargs = {}
    if email_max_chars is not None:
        build_kwargs["email_max_chars"] = email_max_chars
    if doc_max_chars is not None:
        build_kwargs["doc_max_chars"] = doc_max_chars

    context = build_context(chunks, **build_kwargs)

    history_text = ""
    if chat_history:
        for turn in chat_history[-3:]:
            role = turn.get("role", "user")
            text = turn.get("content", "")
            history_text += f"{role.capitalize()}: {text}\n"

    prompt = SENTIMENT_PROMPT
    if history_text:
        prompt += f"\n\nConversation history:\n{history_text}"
    prompt += f"\n\nDocuments to analyze:\n{context}"
    prompt += f"\n\nUser query: {query}"
    prompt += "\n\nProvide your sentiment analysis with specific citations:"

    try:
        response = requests.post(
            OLLAMA["base_url"] + "/api/generate",
            json={
                "model": model or MODELS["llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": MODELS.get("num_predict", 3000),
                }
            },
            timeout=OLLAMA["timeout_seconds"]
        )
        response.raise_for_status()
        raw_answer = response.json()["response"].strip()

        if not raw_answer:
            print("  [Sentiment] LLM returned empty response, using keyword fallback")
            answer, confidence = _keyword_fallback(chunks)
        else:
            answer, confidence = parse_confidence(raw_answer)

    except Exception as e:
        print(f"  [Sentiment] LLM call failed: {e}, using keyword fallback")
        answer, confidence = _keyword_fallback(chunks)

    sources = list({chunk.get("source", "unknown") for chunk in chunks})
    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "confidence": confidence,
        "intent": "sentiment"
    }
