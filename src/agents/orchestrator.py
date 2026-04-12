"""
orchestrator.py - LangGraph workflow orchestrator.

This is the single entry point for the entire agentic system.
It wires together: Router -> Retrieval -> appropriate specialist agent -> response.

How LangGraph works here (simple analogy):
Think of it as a flowchart with named states. Each state is a function
that does some work and returns the name of the next state to go to.
The graph defines which states exist and which transitions are allowed.

Our graph (agentic-v2):
  START -> route -> retrieval -> [teach | troubleshoot | check | lookup] -> END

The retrieval node runs BEFORE the specialist agents. It performs
multi-turn retrieval (with source diversity checking) and stores the
chunks in state so specialists use them directly without re-retrieving.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.agents.router import classify_intent
from src.agents.teacher import teach
from src.agents.troubleshooter import troubleshoot
from src.agents.checker import check
from src.retrieval.multi_turn import multi_turn_retrieve
from src.agents.critic import critique
from src.agents.critic import critique


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
    retrieved_chunks: list     # Pre-retrieved chunks from retrieval node
    second_pass_fired: bool    # Whether multi-turn second pass was needed
    hybrid_weight: float       # Semantic/keyword balance (0.0-1.0)
    critic_verdict: str        # PASS or REJECT from critic agent
    critic_feedback: str       # Critic explanation
    refinement_attempted: bool # Whether refinement retry was attempted
    critic_verdict: str        # PASS or REJECT from critic agent
    critic_feedback: str       # Critic explanation
    refinement_attempted: bool # Whether refinement retry was attempted


# --- Node Functions ---
# Each node takes the full state, does its work, and returns
# a dict of fields to update in the state.

def router_node(state: AgentState) -> dict:
    """Classify the user's intent and store it in state."""
    print(f"  [Router] Classifying: '{state['query'][:60]}...'")
    intent = classify_intent(state["query"], model=state.get("router_model"))
    print(f"  [Router] Intent: {intent}")
    return {"intent": intent}


def retrieval_node(state: AgentState) -> dict:
    """
    Run multi-turn retrieval before handing off to specialist agents.

    This node performs source-diversity-aware retrieval so specialists
    receive a pre-built, diverse chunk set rather than each running
    their own single-pass retrieval independently.
    """
    print(f"  [Retrieval] Starting multi-turn retrieval...")
    chunks, second_pass = multi_turn_retrieve(
        project_name=state["project_name"],
        query=state["query"],
        top_k=state.get("top_k") or None,
        top_k_final=state.get("top_k_final") or None,
        hybrid_weight=state.get("hybrid_weight") or None
    )
    if second_pass:
        print(f"  [Retrieval] Second pass fired — source diversity enforced")
    return {
        "retrieved_chunks": chunks,
        "second_pass_fired": second_pass
    }


def teach_node(state: AgentState) -> dict:
    """Handle teaching requests."""
    print(f"  [Teacher] Answering teaching request...")
    result = teach(
        project_name=state["project_name"],
        query=state["query"],
        chat_history=state.get("chat_history", []),
        model=state.get("reasoning_model"),
        top_k=state.get("top_k"),
        top_k_final=state.get("top_k_final"),
        chunks=state.get("retrieved_chunks", [])
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
        top_k_final=state.get("top_k_final"),
        chunks=state.get("retrieved_chunks", [])
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
        top_k_final=state.get("top_k_final"),
        chunks=state.get("retrieved_chunks", [])
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
        top_k_final=state.get("top_k_final"),
        chunks=state.get("retrieved_chunks", [])
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"],
        "confidence": result.get("confidence", 3)
    }


def refinement_node(state: AgentState) -> dict:
    """
    Check if the specialist response needs refinement based on confidence score.
    
    If confidence is 1-2 and no refinement attempted yet, trigger retry with
    different retrieval strategy. Otherwise pass through to critic.
    """
    confidence = state.get("confidence", 3)
    already_attempted = state.get("refinement_attempted", False)
    
    if confidence <= 2 and not already_attempted:
        print(f"  [Refinement] Low confidence ({confidence}) - triggering retry...")
        
        # Try different hybrid weight for retry (more keyword-heavy)
        retry_weight = max(0.1, (state.get("hybrid_weight", 0.5) - 0.3))
        print(f"  [Refinement] Retry with hybrid_weight={retry_weight}")
        
        # Re-run retrieval with different weight
        from src.retrieval.multi_turn import multi_turn_retrieve
        chunks, second_pass = multi_turn_retrieve(
            project_name=state["project_name"],
            query=state["query"],
            top_k=state.get("top_k") or None,
            top_k_final=state.get("top_k_final") or None,
            hybrid_weight=retry_weight
        )
        
        # Re-run the same specialist with new chunks
        intent = state["intent"]
        if intent == "teach":
            from src.agents.teacher import teach
            result = teach(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks
            )
        elif intent == "troubleshoot":
            from src.agents.troubleshooter import troubleshoot
            result = troubleshoot(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks
            )
        elif intent == "check":
            from src.agents.checker import check
            result = check(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks
            )
        else:  # lookup
            from src.agents.teacher import teach
            result = teach(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks
            )
        
        retry_confidence = result.get("confidence", 3)
        print(f"  [Refinement] Retry confidence: {retry_confidence}")
        
        # Use better result (original vs retry)
        if retry_confidence > confidence:
            print(f"  [Refinement] Retry improved confidence {confidence} -> {retry_confidence}")
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "chunks_used": result["chunks_used"],
                "confidence": retry_confidence,
                "retrieved_chunks": chunks,
                "refinement_attempted": True
            }
        else:
            print(f"  [Refinement] Retry did not improve, keeping original")
            return {"refinement_attempted": True}
    else:
        if already_attempted:
            print(f"  [Refinement] Already attempted, passing through")
        else:
            print(f"  [Refinement] Confidence {confidence} acceptable, passing through")
        return {"refinement_attempted": state.get("refinement_attempted", False)}


def critic_node(state: AgentState) -> dict:
    """
    Evaluate the specialist agent response before sending to user.
    
    The Critic checks whether the answer adequately addresses the query
    given the retrieved chunks. Returns PASS/REJECT with feedback.
    For now, all responses pass through regardless of verdict -- 
    Step 4 (iterative refinement) will use REJECT to trigger retry.
    """
    print(f"  [Critic] Evaluating specialist response...")
    result = critique(
        query=state["query"],
        chunks=state.get("retrieved_chunks", []),
        answer=state["answer"],
        model=state.get("router_model") or None
    )
    return {
        "critic_verdict": result["verdict"],
        "critic_feedback": result["feedback"]
    }


def refinement_node(state: AgentState) -> dict:
    """
    Check if the specialist response needs refinement based on confidence score.
    
    If confidence is 1-2 and no refinement attempted yet, trigger retry with
    different retrieval strategy. Otherwise pass through to critic.
    """
    confidence = state.get("confidence", 3)
    already_attempted = state.get("refinement_attempted", False)
    
    if confidence <= 2 and not already_attempted:
        print(f"  [Refinement] Low confidence ({confidence}) - triggering retry...")
        
        # Try different hybrid weight for retry (more keyword-heavy)
        retry_weight = max(0.1, (state.get("hybrid_weight", 0.5) - 0.3))
        print(f"  [Refinement] Retry with hybrid_weight={retry_weight}")
        
        # Re-run retrieval with different weight
        from src.retrieval.multi_turn import multi_turn_retrieve
        chunks, second_pass = multi_turn_retrieve(
            project_name=state["project_name"],
            query=state["query"],
            top_k=state.get("top_k") or None,
            top_k_final=state.get("top_k_final") or None,
            hybrid_weight=retry_weight
        )
        
        # Re-run the same specialist with new chunks
        intent = state["intent"]
        if intent == "teach":
            from src.agents.teacher import teach
            result = teach(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks
            )
        elif intent == "troubleshoot":
            from src.agents.troubleshooter import troubleshoot
            result = troubleshoot(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks
            )
        elif intent == "check":
            from src.agents.checker import check
            result = check(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks
            )
        else:  # lookup
            from src.agents.teacher import teach
            result = teach(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks
            )
        
        retry_confidence = result.get("confidence", 3)
        print(f"  [Refinement] Retry confidence: {retry_confidence}")
        
        # Use better result (original vs retry)
        if retry_confidence > confidence:
            print(f"  [Refinement] Retry improved confidence {confidence} -> {retry_confidence}")
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "chunks_used": result["chunks_used"],
                "confidence": retry_confidence,
                "retrieved_chunks": chunks,
                "refinement_attempted": True
            }
        else:
            print(f"  [Refinement] Retry did not improve, keeping original")
            return {"refinement_attempted": True}
    else:
        if already_attempted:
            print(f"  [Refinement] Already attempted, passing through")
        else:
            print(f"  [Refinement] Confidence {confidence} acceptable, passing through")
        return {"refinement_attempted": state.get("refinement_attempted", False)}


def critic_node(state: AgentState) -> dict:
    """
    Evaluate the specialist agent response before sending to user.
    
    The Critic checks whether the answer adequately addresses the query
    given the retrieved chunks. Returns PASS/REJECT with feedback.
    For now, all responses pass through regardless of verdict -- 
    Step 4 (iterative refinement) will use REJECT to trigger retry.
    """
    print(f"  [Critic] Evaluating specialist response...")
    result = critique(
        query=state["query"],
        chunks=state.get("retrieved_chunks", []),
        answer=state["answer"],
        model=state.get("router_model") or None
    )
    return {
        "critic_verdict": result["verdict"],
        "critic_feedback": result["feedback"]
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

    Graph flow (agentic-v2):
      router -> retrieval -> [teach | troubleshoot | check | lookup] -> END
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("teach", teach_node)
    graph.add_node("troubleshoot", troubleshoot_node)
    graph.add_node("check", check_node)
    graph.add_node("lookup", lookup_node)
    graph.add_node("refinement", refinement_node)
    graph.add_node("critic", critic_node)

    # Set entry point
    graph.set_entry_point("router")

    # Router always goes to retrieval
    graph.add_edge("router", "retrieval")

    # Retrieval fans out to specialist agents based on intent
    graph.add_conditional_edges(
        "retrieval",
        route_to_agent,
        {
            "teach": "teach",
            "troubleshoot": "troubleshoot",
            "check": "check",
            "lookup": "lookup",
        }
    )

    # All specialist nodes lead to refinement, refinement -> critic -> END
    graph.add_edge("teach", "refinement")
    graph.add_edge("troubleshoot", "refinement")
    graph.add_edge("check", "refinement")
    graph.add_edge("lookup", "refinement")
    graph.add_edge("refinement", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


# Compile once at module load time
agent_graph = build_graph()


# --- Public Entry Point ---

def run_agent(project_name: str, query: str,
              chat_history: list = None,
              router_model: str = None,
              reasoning_model: str = None,
              top_k: int = None,
              top_k_final: int = None,
              hybrid_weight: float = None) -> dict:
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
            "confidence": 4,
            "second_pass_fired": True/False
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
        top_k_final=top_k_final or 0,
        retrieved_chunks=[],
        second_pass_fired=False,
        hybrid_weight=hybrid_weight or 0.0,
        critic_verdict="",
        critic_feedback="",
        refinement_attempted=False
    )

    final_state = agent_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "intent": final_state["intent"],
        "sources": final_state["sources"],
        "chunks_used": final_state["chunks_used"],
        "confidence": final_state.get("confidence", 3),
        "second_pass_fired": final_state.get("second_pass_fired", False),
        "critic_verdict": final_state.get("critic_verdict", ""),
        "critic_feedback": final_state.get("critic_feedback", "")
    }
