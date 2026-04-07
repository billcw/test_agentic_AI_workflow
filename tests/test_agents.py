"""
test_agents.py - Tests for the agent layer.

Run with: pytest tests/test_agents.py -v

These tests verify that the router correctly classifies intent.
Note: These tests call Ollama (E4B model) so require Ollama running.
"""

import pytest


ROUTER_TEST_CASES = [
    ("How do I reset a breaker?", "teach"),
    ("Walk me through the shutdown procedure", "teach"),
    ("Explain how to restart the alarm server", "teach"),
    ("Why is my alarm server failing?", "troubleshoot"),
    ("I am getting this error: connection refused", "troubleshoot"),
    ("Something is wrong with the lpmd service", "troubleshoot"),
    ("I just completed the shutdown procedure, did I do it right?", "check"),
    ("Check my work: I disconnected L1 then L2 then L3", "check"),
    ("Verify this procedure I followed", "check"),
    ("What does LPMD stand for?", "lookup"),
    ("What is the alrm_server?", "lookup"),
    ("What is the escalation phone number?", "lookup"),
]


def test_router_is_importable():
    """Router module should import without errors."""
    from src.agents.router import classify_intent
    assert callable(classify_intent)


@pytest.mark.parametrize("query,expected_intent", ROUTER_TEST_CASES)
def test_router_classifies_intent(query, expected_intent):
    """Router should correctly classify each query type."""
    from src.agents.router import classify_intent
    result = classify_intent(query)
    assert result == expected_intent, (
        f"Query: {query!r}\n"
        f"Expected: {expected_intent!r}\n"
        f"Got: {result!r}"
    )


def test_router_returns_valid_intent_for_unknown():
    """Router should return a valid intent even for unusual input."""
    from src.agents.router import classify_intent, VALID_INTENTS
    result = classify_intent("xyzzy blorp flibbertigibbet")
    assert result in VALID_INTENTS, f"Got invalid intent: {result!r}"


def test_orchestrator_is_importable():
    """Orchestrator module should import and expose run_agent."""
    from src.agents.orchestrator import run_agent
    assert callable(run_agent)
