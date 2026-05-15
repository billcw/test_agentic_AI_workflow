#!/usr/bin/env python3
"""
patch_orchestrator_metadata.py
Wires the metadata agent into the LangGraph orchestrator.

Four changes:
1. Import run_metadata_query from metadata_agent
2. Add metadata_result to AgentState
3. Add metadata_node function
4. Add metadata node + edge to the graph, and include "metadata"
   in route_to_agent so the conditional edge can reach it

Routing design:
  metadata queries skip the clarifier entirely and go straight
  to the metadata node after the router. The metadata node
  bypasses retrieval, all specialist agents, refinement, and
  the critic — it goes directly to END.

Why skip clarifier for metadata?
  Metadata queries are typically very concrete ("how many emails",
  "oldest email") and don't benefit from clarification. The SQL
  generator handles ambiguity by producing a best-effort query.

Why skip refinement and critic?
  The answer comes from a deterministic SQL query, not LLM
  synthesis. There is nothing to refine or critique — the result
  is either correct or the SQL was wrong.
"""

from pathlib import Path

TARGET = Path("/home/bill/local-ai-doc-assistant/src/agents/orchestrator.py")

# --- Patch 1: add import ---
OLD_IMPORT = "from src.agents.sentiment import analyze_sentiment"
NEW_IMPORT = ("from src.agents.sentiment import analyze_sentiment\n"
              "from src.agents.metadata_agent import run_metadata_query")

# --- Patch 2: add metadata_result to AgentState ---
OLD_STATE = """    needs_clarification: bool
    clarifying_questions: list"""

NEW_STATE = """    needs_clarification: bool
    clarifying_questions: list
    metadata_result: dict"""

# --- Patch 3: add metadata_node function (insert before router_node) ---
OLD_NODE_START = "def router_node(state: AgentState) -> dict:"

NEW_METADATA_NODE = '''def metadata_node(state: AgentState) -> dict:
    """
    Handle metadata queries — questions about the email archive itself
    (counts, dates, senders) answered via SQL, not RAG retrieval.

    Bypasses clarifier, retrieval, all specialist agents, refinement,
    and critic. Goes directly to END after this node.
    """
    print(f"  [Metadata] Running SQL metadata query...")
    result = run_metadata_query(
        project_name=state["project_name"],
        query=state["query"],
        model=state.get("router_model") or None,
    )
    return {
        "answer":          result["answer"],
        "sources":         result["sources"],
        "chunks_used":     result["chunks_used"],
        "confidence":      result["confidence"],
        "metadata_result": result,
    }


def router_node(state: AgentState) -> dict:'''

# --- Patch 4: add route_after_router conditional edge function ---
OLD_ROUTE_CLARIFIER = '''def route_after_clarifier(state: AgentState) -> str:'''

NEW_ROUTE_AFTER_ROUTER = '''def route_after_router(state: AgentState) -> str:
    """
    Conditional edge after router_node.
    Metadata queries bypass clarifier and retrieval entirely.
    All other intents proceed to clarifier as normal.
    """
    if state.get("intent") == "metadata":
        return "metadata"
    return "clarifier"


def route_after_clarifier(state: AgentState) -> str:'''

# --- Patch 5: add "metadata" to route_to_agent routes dict ---
OLD_ROUTES = '''    routes = {
        "teach": "teach",
        "troubleshoot": "troubleshoot",
        "check": "check",
        "sentiment": "sentiment",
        "lookup": "lookup",
    }'''

NEW_ROUTES = '''    routes = {
        "teach": "teach",
        "troubleshoot": "troubleshoot",
        "check": "check",
        "sentiment": "sentiment",
        "lookup": "lookup",
        "metadata": "metadata",
    }'''

# --- Patch 6: wire metadata node into the graph ---
OLD_GRAPH_NODES = '''    graph.add_node("router", router_node)
    graph.add_node("clarifier", clarifier_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("teach", teach_node)
    graph.add_node("troubleshoot", troubleshoot_node)
    graph.add_node("check", check_node)
    graph.add_node("lookup", lookup_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("refinement", refinement_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "clarifier")

    # After clarifier: either exit early with questions, or continue to retrieval
    graph.add_conditional_edges(
        "clarifier",
        route_after_clarifier,
        {
            "end_early": END,
            "retrieval": "retrieval",
        }
    )'''

NEW_GRAPH_NODES = '''    graph.add_node("router", router_node)
    graph.add_node("clarifier", clarifier_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("teach", teach_node)
    graph.add_node("troubleshoot", troubleshoot_node)
    graph.add_node("check", check_node)
    graph.add_node("lookup", lookup_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("metadata", metadata_node)
    graph.add_node("refinement", refinement_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("router")

    # After router: metadata queries go directly to metadata node,
    # all others continue to clarifier as normal
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "metadata": "metadata",
            "clarifier": "clarifier",
        }
    )

    # Metadata node exits immediately — no retrieval, refinement, or critic
    graph.add_edge("metadata", END)

    # After clarifier: either exit early with questions, or continue to retrieval
    graph.add_conditional_edges(
        "clarifier",
        route_after_clarifier,
        {
            "end_early": END,
            "retrieval": "retrieval",
        }
    )'''

# --- Patch 7: add metadata_result to initial_state in run_agent ---
OLD_INITIAL_STATE = '''        needs_clarification=False,
        clarifying_questions=[]
    )'''

NEW_INITIAL_STATE = '''        needs_clarification=False,
        clarifying_questions=[],
        metadata_result={}
    )'''

# --- Apply all patches ---
text = TARGET.read_text()

patches = [
    ("import",          OLD_IMPORT,          NEW_IMPORT),
    ("AgentState",      OLD_STATE,           NEW_STATE),
    ("metadata_node",   OLD_NODE_START,      NEW_METADATA_NODE),
    ("route_after_router", OLD_ROUTE_CLARIFIER, NEW_ROUTE_AFTER_ROUTER),
    ("routes dict",     OLD_ROUTES,          NEW_ROUTES),
    ("graph nodes",     OLD_GRAPH_NODES,     NEW_GRAPH_NODES),
    ("initial_state",   OLD_INITIAL_STATE,   NEW_INITIAL_STATE),
]

for name, old, new in patches:
    count = text.count(old)
    assert count == 1, f"Patch '{name}': expected 1 match, found {count}"
    text = text.replace(old, new)
    print(f"  Patch '{name}': OK")

TARGET.write_text(text)
print("\norchestrator.py patched successfully.")
