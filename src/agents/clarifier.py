"""
clarifier.py - Ambiguity detection and clarifying question generation.

Uses gemma4:e4b (fast router model) to decide whether a user query
is too ambiguous to retrieve good results. If ambiguous, generates
2-3 targeted clarifying questions.

Design principles:
- Fast: uses e4b not 31b — this runs BEFORE retrieval, so latency matters
- Conservative: only fires on genuinely ambiguous queries, not just vague ones
- Zero-cost on clear queries: a simple YES/NO check first, questions only if YES
- Same /api/generate pattern as router and query rewriter

When does it fire?
  AMBIGUOUS: "What are the steps?" (steps for what?)
             "How do I fix this?" (fix what?)
             "What should I check?" (check for what purpose?)
  NOT AMBIGUOUS: "What is the AGC setpoint range?"
                 "How do I perform a manual trip on breaker 47?"
                 "Show me emails about the July outage"

The second query in those NOT AMBIGUOUS examples is specific enough that
retrieval will find good results. The clarifier should not fire on it.
"""

import re
import requests

from src.config import OLLAMA, MODELS

OLLAMA_URL = OLLAMA["base_url"]
ROUTER_MODEL = MODELS.get("router_llm", "gemma4:e4b")


def _call_llm(prompt: str, model: str, num_predict: int = 50) -> str:
    """
    Call Ollama /api/generate and return the response text.
    Same pattern used everywhere in this codebase.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
        }
    }
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=OLLAMA.get("timeout_seconds", 3600)
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [Clarifier] LLM call failed: {e}")
        return ""


def is_ambiguous(query: str, model: str = None) -> bool:
    """
    Ask the router model whether this query is too ambiguous to answer well.

    Returns True if the query needs clarification before retrieval.
    Returns False if retrieval should proceed immediately.

    We use a tight YES/NO prompt with temperature 0.0 so the answer
    is deterministic and parseable without fragile string matching.
    """
    m = model or ROUTER_MODEL

    prompt = (
        "You are a query ambiguity classifier for a technical document search system.\n"
        "The document collection contains SCADA/EMS system manuals and work emails.\n\n"
        "Determine if the following query is TOO AMBIGUOUS to search effectively.\n"
        "A query is ambiguous if it is missing critical context that would determine\n"
        "WHICH documents or procedures to retrieve (e.g. missing equipment name,\n"
        "system name, error type, or action to perform).\n\n"
        "A query is NOT ambiguous if it names a specific system, equipment, procedure,\n"
        "error, or topic — even if it is short or informal.\n\n"
        "Respond with only YES (ambiguous) or NO (not ambiguous). Nothing else.\n\n"
        f"Query: {query}\n\n"
        "Answer:"
    )

    response = _call_llm(prompt, m, num_predict=500)

    # Be conservative: only treat as ambiguous on a clear YES
    # If the model returns anything other than a clear YES, proceed with retrieval
    first_word = response.strip().upper().split()[0] if response.strip() else "NO"
    return first_word == "YES"


def generate_questions(query: str, model: str = None) -> list[str]:
    """
    Generate 2-3 targeted clarifying questions for an ambiguous query.

    Returns a list of question strings.
    Returns empty list on any failure (fail-open: pipeline proceeds normally).

    Prompt seeds with "1." to get clean numbered output — same technique
    used successfully in the query rewriter.
    """
    m = model or ROUTER_MODEL

    prompt = (
        "You are an assistant helping users search a technical document system.\n"
        "The document collection contains SCADA/EMS system manuals and work emails.\n\n"
        "The user asked a question that needs clarification before you can search effectively.\n"
        "Generate exactly 2 or 3 short, specific clarifying questions that would help\n"
        "narrow down which documents or procedures are relevant.\n\n"
        "Rules:\n"
        "- Each question must be on its own numbered line\n"
        "- Questions must be short and specific (one sentence each)\n"
        "- Do not include any preamble, explanation, or closing text\n"
        "- Do not repeat the user's query back to them\n\n"
        f"User query: {query}\n\n"
        "Clarifying questions:\n"
        "1."
    )

    response = _call_llm(prompt, m, num_predict=1000)

    # The prompt seeds with "1." so the response starts mid-line-1
    # Prepend it back so parsing works cleanly
    full_text = "1. " + response if not response.lstrip().startswith(("1.", "1)")) else response

    questions = []
    for line in full_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match lines starting with a number and period/dot: "1.", "2.", "3."
        match = re.match(r"^\d+[.)]\s*(.+)", line)
        if match:
            q = match.group(1).strip()
            if len(q) > 10:  # ignore fragments
                questions.append(q)

    return questions[:3]  # cap at 3


def clarify(query: str, model: str = None) -> dict:
    """
    Main entry point called by the orchestrator clarifier_node.

    Returns:
        {
            "needs_clarification": bool,
            "clarifying_questions": list[str]   # empty if not ambiguous
        }

    Fail-open design: any exception returns needs_clarification=False
    so the pipeline always proceeds even if this step errors.
    """
    try:
        if not is_ambiguous(query, model=model):
            return {"needs_clarification": False, "clarifying_questions": []}

        questions = generate_questions(query, model=model)

        if not questions:
            # Model returned nothing usable — fail open
            print("  [Clarifier] No questions generated, proceeding with retrieval")
            return {"needs_clarification": False, "clarifying_questions": []}

        print(f"  [Clarifier] Query ambiguous — generated {len(questions)} questions")
        return {"needs_clarification": True, "clarifying_questions": questions}

    except Exception as e:
        print(f"  [Clarifier] Error in clarify(): {e} — failing open")
        return {"needs_clarification": False, "clarifying_questions": []}
