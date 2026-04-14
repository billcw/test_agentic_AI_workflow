"""
router.py - Intent classification agent.
Uses Gemma 4 E4B (the fast/small model) to classify every incoming
user message into one of four intent categories before routing to
the appropriate specialist agent.

Why E4B here instead of 31B?
Routing is a simple classification task -- it does not need deep
reasoning. E4B is much faster and keeps latency low for this
first-hop decision. Think of it as a traffic cop, not a detective.
"""
import requests
from src.config import OLLAMA, MODELS

INTENT_TEACH = "teach"
INTENT_TROUBLESHOOT = "troubleshoot"
INTENT_CHECK = "check"
INTENT_LOOKUP = "lookup"
INTENT_SENTIMENT = "sentiment"
VALID_INTENTS = {INTENT_TEACH, INTENT_TROUBLESHOOT, INTENT_CHECK, INTENT_LOOKUP, INTENT_SENTIMENT}

ROUTER_PROMPT = """You are an intent classifier for a document assistant system.
Classify the user message into exactly one of these five categories:

- teach: User wants to learn how to do something, understand a procedure,
  or get step-by-step instructions. Examples: How do I..., Walk me
  through..., Explain how to..., What are the steps for...

- troubleshoot: User is reporting a problem, error, alarm, or failure and
  wants help diagnosing or fixing it. Examples: Why is X failing?,
  I am getting this alarm..., Something is wrong with..., Error: ...

- check: User wants to verify, review, or confirm work against a procedure.
  The word check or verify often appears directly in the message.
  Examples: Check my work, Verify this procedure, Did I do this right?,
  Check this against the manual, Is this correct?, Review what I did.
  IMPORTANT: If the user says check followed by their work or steps,
  always classify as check - not troubleshoot.

- sentiment: User wants to analyze the emotional tone, mood, urgency, or
  attitude expressed in emails, messages, or documents. Examples:
  What is the sentiment in these emails?, Are there urgent messages?,
  Analyze the tone of communications about X, What is the mood of
  the team based on emails?, Find frustrated or angry messages,
  Are there any critical or emergency communications?

- lookup: User wants a specific fact, definition, value, or quick
  reference. Everything else that does not fit the above four.
  Examples: What is..., What does X mean?, Who is..., When did...

Respond with ONLY the single word category. No explanation. No punctuation.
Just one of: teach, troubleshoot, check, sentiment, lookup"""


def classify_intent(user_message: str, model: str = None) -> str:
    """
    Classify the intent of a user message using the E4B router model.
    Returns one of: teach, troubleshoot, check, sentiment, lookup
    Falls back to lookup if the model returns something unexpected.
    """
    try:
        prompt = ROUTER_PROMPT + "\n\nUser message: " + user_message
        response = requests.post(
            OLLAMA["base_url"] + "/api/generate",
            json={
                "model": model or MODELS["router_llm"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 10,
                }
            },
            timeout=OLLAMA["timeout_seconds"]
        )
        response.raise_for_status()
        raw = response.json()["response"].strip().lower()
        intent = raw.split()[0].rstrip(".,!?") if raw else "lookup"
        if intent in VALID_INTENTS:
            return intent
        else:
            print("  [Router] Unexpected intent: " + intent + ", defaulting to lookup")
            return INTENT_LOOKUP
    except Exception as e:
        print("  [Router] Error during classification: " + str(e))
        return INTENT_LOOKUP
