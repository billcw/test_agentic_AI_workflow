"""
sentiment.py - Sentiment analysis agent for agentic-v2.

Analyzes emotional tone, mood, and sentiment in documents, especially
emails and social media posts. Uses LLM analysis with bag-of-words
fallback if the model fails to respond.

Use cases:
- Find emails by emotional tone: "Show me angry emails from Q3"
- Analyze document sentiment: "What's the mood in these posts?"
- Search by emotional markers: "Find frustrated communications"
"""

import requests
from src.config import OLLAMA, MODELS, RETRIEVAL
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank
from src.agents.teacher import build_context, parse_confidence


SENTIMENT_PROMPT = """You are a sentiment analysis assistant.
Analyze the emotional tone of the provided document excerpts.

Categories to identify:
- Positive: happy, excited, satisfied, grateful, optimistic
- Negative: angry, frustrated, disappointed, worried, upset
- Neutral: factual, professional, informational
- Urgent: time-sensitive, emergency, immediate action needed
- Humorous: funny, joking, sarcastic, witty

For each document excerpt, identify the primary sentiment category
and provide specific examples of emotional language.

End with: CONFIDENCE: X/5
Where X is your confidence in the sentiment analysis."""


def analyze_sentiment(project_name: str, query: str,
                     chat_history: list = None,
                     model: str = None,
                     top_k: int = None,
                     top_k_final: int = None,
                     chunks: list = None) -> dict:
    """
    Analyze sentiment in documents matching the query.

    Args:
        project_name: Which workspace to search
        query: Sentiment search query (e.g., "Find angry emails")
        chunks: Pre-retrieved chunks from retrieval_node

    Returns:
        {
            "answer": "Sentiment analysis with examples...",
            "sources": ["email1.pst", "posts.txt"],
            "chunks_used": 5,
            "confidence": 4,
            "intent": "sentiment"
        }
    """
    # Step 1: Use pre-retrieved chunks if provided, else retrieve now
    if not chunks:
        raw_results = hybrid_search(project_name, query,
                                    top_k=top_k or RETRIEVAL["top_k"])
        chunks = rerank(raw_results, top_k_final=top_k_final or RETRIEVAL["top_k_final"])

    if not chunks:
        return {
            "answer": ("I could not find relevant documents to analyze for sentiment. "
                       "Please ensure documents with emotional content have been ingested."),
            "sources": [],
            "chunks_used": 0,
            "confidence": 1,
            "intent": "sentiment"
        }

    # Step 2: Build context block from retrieved chunks
    context = build_context(chunks)

    # Step 3: Build prompt
    history_text = ""
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "user").capitalize()
            history_text += f"{role}: {msg.get('content', '')}\n"
"

    prompt = f"""{SENTIMENT_PROMPT}

{history_text}
--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Query: {query}

Analyze the sentiment in these documents. Identify emotional tone and provide
specific examples. End with CONFIDENCE rating."""

    # Step 4: Try LLM analysis (simplified to avoid empty responses)
    try:
        response = requests.post(
            f"{OLLAMA['base_url']}/api/generate",
            json={
                "model": model or MODELS["llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1500,
                    "num_ctx": 8192,
                }
            },
            timeout=OLLAMA["timeout_seconds"]
        )
        response.raise_for_status()
        raw_answer = response.json()["response"].strip()

        if raw_answer:  # LLM returned content
            answer, confidence = parse_confidence(raw_answer)
            sources = list({chunk.get("source", "unknown") for chunk in chunks})
            
            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": len(chunks),
                "confidence": confidence,
                "intent": "sentiment"
            }

    except Exception as e:
        print(f"  [Sentiment] LLM error: {e}, using simplified analysis")

    # Step 5: Simplified text analysis fallback
    print(f"  [Sentiment] Using simplified sentiment analysis")
    
    # Basic emotional keyword detection
    positive_words = ["good", "great", "excellent", "happy", "pleased", "thank"]
    negative_words = ["bad", "terrible", "frustrated", "angry", "problem", "issue"]
    urgent_words = ["urgent", "asap", "immediately", "emergency", "critical"]
    
    total_text = " ".join([chunk.get("text", "") for chunk in chunks]).lower()
    
    pos_count = sum(1 for word in positive_words if word in total_text)
    neg_count = sum(1 for word in negative_words if word in total_text) 
    urg_count = sum(1 for word in urgent_words if word in total_text)
    
    if neg_count > pos_count:
        primary_sentiment = "Negative"
    elif pos_count > 0:
        primary_sentiment = "Positive"
    elif urg_count > 0:
        primary_sentiment = "Urgent"
    else:
        primary_sentiment = "Neutral"
    
    answer = f"Sentiment analysis of {len(chunks)} documents:\n\n"
    answer += f"Primary sentiment detected: **{primary_sentiment}**\n\n"
    answer += f"Analyzed {len(chunks)} chunks from {len(set(c.get('source') for c in chunks))} sources.\n"
    answer += f"Positive indicators: {pos_count}, Negative: {neg_count}, Urgent: {urg_count}"

    sources = list({chunk.get("source", "unknown") for chunk in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "confidence": 3,
        "intent": "sentiment"
    }
