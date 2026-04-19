"""
critic.py - Critic agent for agentic-v2.

The Critic evaluates specialist agent responses before they reach the user.
It checks whether the answer is adequately supported by the retrieved chunks
and whether obvious gaps exist.

Why a Critic agent?
Specialist agents generate answers from whatever chunks they receive.
If retrieval missed key documents, the specialist doesn't know -- it
just works with what it has. The Critic has visibility into both the
chunks AND the answer, so it can spot mismatches: "the question asked
about FEP controls but all chunks are from SCADA docs."

Why gemma4:e4b instead of 31B?
The Critic does structured evaluation, not deep reasoning. E4B is fast
and cheap for this task. We don't want the Critic to double the response
time -- it should add only a few seconds.

PASS/REJECT semantics:
- PASS: answer is adequately supported, send it to the user
- REJECT: answer has gaps or wrong sources, flag for iterative refinement
  (Step 4 of agentic-v2 will use the feedback to re-retrieve and retry)

Fail-safe: if the model returns an unparseable response, default to PASS.
A broken Critic must never block a valid answer from reaching the user.
"""

import requests
from src.config import OLLAMA, MODELS


CRITIC_PROMPT = """Evaluate if the answer addresses the question using the provided sources.

Format:
VERDICT: PASS
FEEDBACK: Short explanation

Or:
VERDICT: REJECT
FEEDBACK: Short explanation"""


def parse_critic_response(raw: str) -> tuple[str, str]:
    """
    Parse the critic model response into verdict and feedback.

    Returns:
        (verdict, feedback)
        verdict: "PASS" or "REJECT"
        feedback: explanation string

    Defaults to PASS if response is unparseable -- broken critic
    must never block a valid answer.
    """
    verdict = "PASS"
    feedback = "Critic response unparseable — defaulting to PASS."

    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in ("PASS", "REJECT"):
                verdict = val
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return verdict, feedback


def critique(query: str,
             chunks: list,
             answer: str,
             model: str = None) -> dict:
    """
    Evaluate a specialist agent response for quality and source relevance.
    
    TEMPORARILY DISABLED: Always returns PASS until Gemma 4 model issues resolved.
    """
    print(f"  [Critic] DISABLED - always returning PASS")
    return {
        "verdict": "PASS",
        "feedback": "Critic temporarily disabled - model compatibility issue."
    }