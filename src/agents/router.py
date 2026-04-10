"""
router.py - Intent classification agent.

Uses Gemma 4 E4B (the fast/small model) to classify every incoming
user message into one of four intent categories before routing to
the appropriate specialist agent.

Why E4B here instead of 31B?
Routing is a simple classification task — it doesn't need deep
reasoning. E4B is much faster and keeps latency low for this
first-hop decision. Think of it as a traffic cop, not a detective.
"""

import requests
from src.config import OLLAMA, MODELS


# The four intent categories the router can assign
INTENT_TEACH = "teach"
INTENT_TROUBLESHOOT = "troubleshoot"
INTENT_CHECK = "check"
INTENT_LOOKUP = "lookup"

VALID_INTENTS = {INTENT_TEACH, INTENT_TROUBLESHOOT, INTENT_CHECK, INTENT_LOOKUP}

ROUTER_PROMPT = """You are an intent classifier for a document assistant system.
Classify the user's message into exactly one of these four categories:

- teach: User wants to learn how to do something, understand a procedure,
  or get step-by-step instructions. Examples: "How do I...", "Walk me
  through...", "Explain how to...", "What are the steps for..."

- troubleshoot: User is reporting a problem, error, alarm, or failure and
  wants help diagnosing or fixing it. Examples: "Why is X failing?",
  "I'm getting this alarm...", "Something is wrong with...", "Error: ..."

- check: User wants to verify, review, or confirm work against a procedure.
  The word "check" or "verify" often appears directly in the message.
  Examples: "Check my work:", "Verify this procedure:", "Did I do this
  right?", "Check this against the manual:", "Is this correct?",
  "Review what I did:", "check my steps", "verify my work"
  IMPORTANT: If the user says "check" followed by their work or steps,
  always classify as check - not troubleshoot.

- lookup: User wants a specific fact, definition, value, or quick
  reference. Everything else that doesn't fit the above three.
  Examples: "What is...", "What does X mean?", "Who is...", "When did..."

Respond with ONLY the single word category. No explanation. No punctuation.
Just one of: teach, troubleshoot, check, lookup"""


def classify_intent(user_message: str) -> str:
    """
    Classify the intent of a user message using the E4B router model.

    Returns one of: 'teach', 'troubleshoot', 'check', 'lookup'
    Falls back to 'lookup' if the model returns something unexpected.
    """
    try:
        prompt = f"{ROUTER_PROMPT}\n\nUser message: {user_message}"
        response = requests.post(
            f"{OLLAMA['base_url']}/api/generate",
            json={
                "model": MODELS["router_llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,   # Zero temp = deterministic classification
                    "num_predict": 10,    # We only need one word back
                }
            },
            timeout=OLLAMA["timeout_seconds"]
        )
        response.raise_for_status()
        raw = response.json()["response"].strip().lower()

        # Clean up in case the model adds punctuation or extra words
        intent = raw.split()[0].rstrip(".,!?") if raw else "lookup"

        if intent in VALID_INTENTS:
            return intent
        else:
            # Model returned something unexpected — default to lookup
            print(f"  [Router] Unexpected intent '{intent}', defaulting to 'lookup'")
            return INTENT_LOOKUP

    except Exception as e:
        print(f"  [Router] Error during classification: {e}")
        return INTENT_LOOKUP  # Safe fallback
