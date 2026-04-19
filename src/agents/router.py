"""
router.py - Intent classification agent.
Uses Gemma 4 E4B (the fast/small model) to classify every incoming
user message into intent and scope before routing to the appropriate
specialist agent.

Why two separate calls?
Each call is a single focused classification task. Asking the model
to return two things at once increases parsing complexity and reduces
reliability. Two simple calls beats one complex call every time.

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

SCOPE_EMAIL = "email"
SCOPE_DOCUMENT = "document"
SCOPE_ALL = "all"
VALID_SCOPES = {SCOPE_EMAIL, SCOPE_DOCUMENT, SCOPE_ALL}

INTENT_PROMPT = """You are an intent classifier for a document assistant system.
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

SCOPE_PROMPT = """You are a search scope classifier for a document assistant system.
The system has two types of documents: emails and technical manuals/documents.

Classify the user message into exactly one of these three search scopes:

- email: User is asking about communications, messages, inbox, emails, or
  anything that would be found in an email archive. Examples:
  Show me urgent communications, What emails mention the outage?,
  Find messages about the alarm, Are there any emails from John?,
  Show me communications about the FEP issue, urgent messages,
  What did the team say about X?, correspondence about Y.

- document: User is asking about procedures, manuals, configurations,
  technical specifications, or anything found in technical documents.
  Examples: How do I configure X?, What does the manual say about Y?,
  Show me the procedure for Z, What are the alarm setpoints?,
  Find the configuration steps, technical specifications for X.

- all: User is asking broadly across all sources, or the scope is
  ambiguous and could apply to either emails or documents.
  Examples: What do we know about X?, Find everything about Y,
  Search all sources for Z, What information exists about X?

Respond with ONLY the single word scope. No explanation. No punctuation.
Just one of: email, document, all"""


def _call_router(prompt: str, valid_values: set, default: str, model: str = None) -> str:
    """
    Internal helper: make a single focused classification call to the router model.
    Returns one of the valid_values, or default if the model returns something unexpected.
    """
    try:
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
        result = raw.split()[0].rstrip(".,!?") if raw else default
        if result in valid_values:
            return result
        else:
            print("  [Router] Unexpected value: " + result + ", defaulting to " + default)
            return default
    except Exception as e:
        print("  [Router] Error during classification: " + str(e))
        return default


def classify_intent(user_message: str, model: str = None) -> str:
    """
    Classify the intent of a user message using the E4B router model.
    Returns one of: teach, troubleshoot, check, sentiment, lookup
    Falls back to lookup if the model returns something unexpected.
    """
    prompt = INTENT_PROMPT + "\n\nUser message: " + user_message
    intent = _call_router(prompt, VALID_INTENTS, INTENT_LOOKUP, model)
    print("  [Router] Intent: " + intent)
    return intent


def classify_scope(user_message: str, model: str = None) -> str:
    """
    Classify the search scope of a user message using the E4B router model.
    Returns one of: email, document, all
    Falls back to all if the model returns something unexpected.
    """
    prompt = SCOPE_PROMPT + "\n\nUser message: " + user_message
    scope = _call_router(prompt, VALID_SCOPES, SCOPE_ALL, model)
    print("  [Router] Scope: " + scope)
    return scope
