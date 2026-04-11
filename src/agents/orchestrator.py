"""
orchestrator.py - LangGraph workflow orchestrator.

This is the single entry point for the entire agentic system.
It wires together: Router -> appropriate specialist agent -> response.

How LangGraph works here (simple analogy):
Think of it as a flowchart with named states. Each state is a function
that does some work and returns the name of the next state to go to.
The graph defines which states exist and which transitions are allowed.

Our graph is simple:
  START -> route -> [teach | troubleshoot | check | lookup] -> END
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.agents.router import classify_intent
from src.agents.teacher import teach
from src.agents.troubleshooter import troubleshoot
from src.agents.checker import check


# --- State Definition ---
# This is the data that flows through the graph at every step.
# TypedDict makes it explicit what fields exist and their types.

class AgentState(TypedDict):
    project_name: str          # Which workspace to search
    query: str                 # The user's message
    intent: str                # Filled in by the router node
    answer: str                # Filled in by the specialist node
    sources: list              # Documents used
    chunks_used: int           # How many chunks were retrieved
    confidence: int            # 1-5 score parsed from model response
    chat_history: list         # Prior conversation turns
    router_model: str          # Model to use for routing
    reasoning_model: str       # Model to use for specialist agents
    top_k: int                 # Candidates from hybrid search
    top_k_final: int           # Chunks sent to agent after reranking


# --- Node Functions ---
# Each node takes the full state, does its work, and returns
# a dict of fields to update in the state.

def router_node(state: AgentState) -> dict:
    """Classify the user's intent and store it in state."""
    print(f"  [Router] Classifying: '{state['query'][:60]}...'")
    intent = classify_intent(state["query"], model=state.get("router_model"))
    print(f"  [Router] Intent: {intent}")
    return {"intent": intent}


def teach_node(state: AgentState) -> dict:
    """Handle teaching requests."""
    print(f"  [Teacher] Answering teaching request...")
    result = teach(
        project_name=state["project_name"],
        query=state["query"],
        chat_history=state.get("chat_history", []),
        model=state.get("reasoning_model"),
        top_k=state.get("top_k"),
        top_k_final=state.get("top_k_final")
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"],
        "confidence": result.get("confidence", 3)
    }


def troubleshoot_node(state: AgentState) -> dict:
    """Handle troubleshooting requests."""
    print(f"  [Troubleshooter] Diagnosing problem...")
    result = troubleshoot(
        project_name=state["project_name"],
        query=state["query"],
        chat_history=state.get("chat_history", []),
        model=state.get("reasoning_model"),
        top_k=state.get("top_k"),
        top_k_final=state.get("top_k_final")
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"],
        "confidence": result.get("confidence", 3)
    }


def check_node(state: AgentState) -> dict:
    """Handle work verification requests."""
    print(f"  [Checker] Verifying work...")
    result = check(
        project_name=state["project_name"],
        query=state["query"],
        chat_history=state.get("chat_history", []),
        model=state.get("reasoning_model"),
        top_k=state.get("top_k"),
        top_k_final=state.get("top_k_final")
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"],
        "confidence": result.get("confidence", 3)
    }


def lookup_node(state: AgentState) -> dict:
    """Handle simple lookup requests using the teacher agent."""
    print(f"  [Lookup] Searching for information...")
    result = teach(
        project_name=state["project_name"],
        query=state["query"],
        chat_history=state.get("chat_history", []),
        model=state.get("reasoning_model"),
        top_k=state.get("top_k"),
        top_k_final=state.get("top_k_final")
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"],
        "confidence": result.get("confidence", 3)
    }


def route_to_agent(state: AgentState) -> str:
    """
    Conditional edge function — tells LangGraph which node to go to next
    based on the intent stored in state by the router node.
    """
    intent = state.get("intent", "lookup")
    routes = {
        "teach": "teach",
        "troubleshoot": "troubleshoot",
        "check": "check",
        "lookup": "lookup",
    }
    return routes.get(intent, "lookup")


# --- Build the Graph ---

def build_graph():
    """
    Construct and compile the LangGraph workflow.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("teach", teach_node)
    graph.add_node("troubleshoot", troubleshoot_node)
    graph.add_node("check", check_node)
    graph.add_node("lookup", lookup_node)

    # Set entry point
    graph.set_entry_point("router")

    # Add conditional edge from router to specialist agents
    graph.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "teach": "teach",
            "troubleshoot": "troubleshoot",
            "check": "check",
            "lookup": "lookup",
        }
    )

    # All specialist nodes lead to END
    graph.add_edge("teach", END)
    graph.add_edge("troubleshoot", END)
    graph.add_edge("check", END)
    graph.add_edge("lookup", END)

    return graph.compile()


# Compile once at module load time
agent_graph = build_graph()


# --- Public Entry Point ---

def run_agent(project_name: str, query: str,
              chat_history: list = None,
              router_model: str = None,
              reasoning_model: str = None,
              top_k: int = None,
              top_k_final: int = None) -> dict:
    """
    Run the full agentic pipeline for a user query.

    Args:
        project_name: Which workspace to search
        query: The user's message
        chat_history: Optional prior conversation turns

    Returns:
        {
            "answer": "...",
            "intent": "teach|troubleshoot|check|lookup",
            "sources": ["file.pdf"],
            "chunks_used": 3,
            "confidence": 4
        }
    """
    initial_state = AgentState(
        project_name=project_name,
        query=query,
        intent="",
        answer="",
        sources=[],
        chunks_used=0,
        confidence=3,
        chat_history=chat_history or [],
        router_model=router_model or "",
        reasoning_model=reasoning_model or "",
        top_k=top_k or 0,
        top_k_final=top_k_final or 0
    )

    final_state = agent_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "intent": final_state["intent"],
        "sources": final_state["sources"],
        "chunks_used": final_state["chunks_used"],
        "confidence": final_state.get("confidence", 3)
    }
