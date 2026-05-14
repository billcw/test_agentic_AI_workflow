"""
orchestrator.py - LangGraph workflow orchestrator.

This is the single entry point for the entire agentic system.
It wires together: Router -> Clarifier -> Retrieval -> specialist agent -> Refinement -> Critic -> END.

How LangGraph works here (simple analogy):
Think of it as a flowchart with named states. Each state is a function
that does some work and returns the name of the next state to go to.
The graph defines which states exist and which transitions are allowed.

Our graph:
  START -> router -> clarifier -> [END_with_questions | retrieval]
       -> [teach|troubleshoot|check|sentiment|lookup]
       -> refinement -> critic -> END

Clarifying questions flow:
  If the clarifier decides the query is ambiguous, the pipeline exits
  early and returns needs_clarification=True with the questions list.
  The UI renders the questions, collects user answers, and re-submits
  a new enriched query. On the second call the clarifier passes through
  because the enriched query is specific enough.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.agents.router import classify_intent, classify_scope
from src.agents.clarifier import clarify
from src.agents.teacher import teach
from src.agents.troubleshooter import troubleshoot
from src.agents.checker import check
from src.retrieval.multi_turn import multi_turn_retrieve
from src.agents.critic import critique
from src.agents.sentiment import analyze_sentiment
from src.agents.metadata_agent import run_metadata_query


# --- State Definition ---

class AgentState(TypedDict):
    project_name: str
    query: str
    intent: str
    answer: str
    sources: list
    chunks_used: int
    confidence: int
    chat_history: list
    router_model: str
    reasoning_model: str
    top_k: int
    top_k_final: int
    retrieved_chunks: list
    second_pass_fired: bool
    hybrid_weight: float
    critic_verdict: str
    critic_feedback: str
    refinement_attempted: bool
    scope: str
    email_max_chars: int
    doc_max_chars: int
    needs_clarification: bool
    clarifying_questions: list
    metadata_result: dict


# --- Node Functions ---

def metadata_node(state: AgentState) -> dict:
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


def router_node(state: AgentState) -> dict:
    """Classify the user intent and scope, store both in state."""
    print(f"  [Router] Classifying: '{state['query'][:60]}...'")
    intent = classify_intent(state["query"], model=state.get("router_model"))
    scope = classify_scope(state["query"], model=state.get("router_model"))
    return {"intent": intent, "scope": scope}


def clarifier_node(state: AgentState) -> dict:
    """
    Check whether the query is too ambiguous to retrieve good results.

    If ambiguous: sets needs_clarification=True and clarifying_questions=[...].
    The conditional edge after this node will route to END immediately,
    returning the questions to the UI without running retrieval.

    If not ambiguous: sets needs_clarification=False and the pipeline
    continues normally to retrieval. Zero latency cost on clear queries
    beyond the fast e4b classification call.

    Fail-open: any error in the clarifier returns needs_clarification=False
    so the pipeline always proceeds even if this step errors.
    """
    print(f"  [Clarifier] Checking query ambiguity...")
    result = clarify(state["query"], model=state.get("router_model"))
    needs = result["needs_clarification"]
    questions = result["clarifying_questions"]
    if needs:
        print(f"  [Clarifier] Ambiguous — returning {len(questions)} questions to UI")
    else:
        print(f"  [Clarifier] Clear — proceeding to retrieval")
    return {
        "needs_clarification": needs,
        "clarifying_questions": questions
    }


def retrieval_node(state: AgentState) -> dict:
    """
    Run multi-turn retrieval before handing off to specialist agents.
    Performs source-diversity-aware retrieval so specialists receive
    a pre-built diverse chunk set.
    """
    print(f"  [Retrieval] Starting multi-turn retrieval...")
    chunks, second_pass = multi_turn_retrieve(
        project_name=state["project_name"],
        query=state["query"],
        top_k=state.get("top_k") or None,
        top_k_final=state.get("top_k_final") or None,
        hybrid_weight=state.get("hybrid_weight") or None,
        scope=state.get("scope", "all")
    )
    if second_pass:
        print(f"  [Retrieval] Second pass fired - source diversity enforced")
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
        chunks=state.get("retrieved_chunks", []),
        email_max_chars=state.get("email_max_chars"),
        doc_max_chars=state.get("doc_max_chars")
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
        chunks=state.get("retrieved_chunks", []),
        email_max_chars=state.get("email_max_chars"),
        doc_max_chars=state.get("doc_max_chars")
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
        chunks=state.get("retrieved_chunks", []),
        email_max_chars=state.get("email_max_chars"),
        doc_max_chars=state.get("doc_max_chars")
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
        chunks=state.get("retrieved_chunks", []),
        email_max_chars=state.get("email_max_chars"),
        doc_max_chars=state.get("doc_max_chars")
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"],
        "confidence": result.get("confidence", 3)
    }


def sentiment_node(state: AgentState) -> dict:
    """Handle sentiment analysis requests."""
    print(f"  [Sentiment] Analyzing emotional tone...")
    result = analyze_sentiment(
        project_name=state["project_name"],
        query=state["query"],
        chat_history=state.get("chat_history", []),
        model=state.get("reasoning_model"),
        top_k=state.get("top_k"),
        top_k_final=state.get("top_k_final"),
        chunks=state.get("retrieved_chunks", []),
        email_max_chars=state.get("email_max_chars"),
        doc_max_chars=state.get("doc_max_chars")
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
    If confidence is 1-2 and no refinement attempted yet, retry with
    different retrieval strategy. One retry max to prevent loops.
    """
    confidence = state.get("confidence", 3)
    already_attempted = state.get("refinement_attempted", False)

    if confidence <= 2 and not already_attempted:
        print(f"  [Refinement] Low confidence ({confidence}) - triggering retry...")
        retry_weight = max(0.1, (state.get("hybrid_weight", 0.5) - 0.3))
        print(f"  [Refinement] Retry with hybrid_weight={retry_weight}")

        chunks, second_pass = multi_turn_retrieve(
            project_name=state["project_name"],
            query=state["query"],
            top_k=state.get("top_k") or None,
            top_k_final=state.get("top_k_final") or None,
            hybrid_weight=retry_weight,
            scope=state.get("scope", "all")
        )

        intent = state["intent"]
        if intent == "teach":
            result = teach(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks,
                email_max_chars=state.get("email_max_chars"),
                doc_max_chars=state.get("doc_max_chars")
            )
        elif intent == "troubleshoot":
            result = troubleshoot(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks,
                email_max_chars=state.get("email_max_chars"),
                doc_max_chars=state.get("doc_max_chars")
            )
        elif intent == "check":
            result = check(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks,
                email_max_chars=state.get("email_max_chars"),
                doc_max_chars=state.get("doc_max_chars")
            )
        elif intent == "sentiment":
            result = analyze_sentiment(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks,
                email_max_chars=state.get("email_max_chars"),
                doc_max_chars=state.get("doc_max_chars")
            )
        else:
            result = teach(
                project_name=state["project_name"],
                query=state["query"],
                chat_history=state.get("chat_history", []),
                model=state.get("reasoning_model"),
                chunks=chunks,
                email_max_chars=state.get("email_max_chars"),
                doc_max_chars=state.get("doc_max_chars")
            )

        retry_confidence = result.get("confidence", 3)
        print(f"  [Refinement] Retry confidence: {retry_confidence}")

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
    Returns PASS/REJECT with feedback. Currently passes all responses through.
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


def route_after_router(state: AgentState) -> str:
    """
    Conditional edge after router_node.
    Metadata queries bypass clarifier and retrieval entirely.
    All other intents proceed to clarifier as normal.
    """
    if state.get("intent") == "metadata":
        return "metadata"
    return "clarifier"


def route_after_clarifier(state: AgentState) -> str:
    """
    Conditional edge after clarifier_node.
    If the query needs clarification: exit immediately (return questions to UI).
    If the query is clear: proceed to retrieval as normal.
    """
    if state.get("needs_clarification", False):
        return "end_early"
    return "retrieval"


def route_to_agent(state: AgentState) -> str:
    """Conditional edge — routes to specialist node based on intent."""
    intent = state.get("intent", "lookup")
    routes = {
        "teach": "teach",
        "troubleshoot": "troubleshoot",
        "check": "check",
        "sentiment": "sentiment",
        "lookup": "lookup",
        "metadata": "metadata",
    }
    return routes.get(intent, "lookup")


# --- Build the Graph ---

def build_graph():
    """
    Construct and compile the LangGraph workflow.
    Flow: router -> clarifier -> [END (questions) | retrieval -> specialist -> refinement -> critic -> END]
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
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
    )

    graph.add_conditional_edges(
        "retrieval",
        route_to_agent,
        {
            "teach": "teach",
            "troubleshoot": "troubleshoot",
            "check": "check",
            "sentiment": "sentiment",
            "lookup": "lookup",
        }
    )

    graph.add_edge("teach", "refinement")
    graph.add_edge("troubleshoot", "refinement")
    graph.add_edge("check", "refinement")
    graph.add_edge("sentiment", "refinement")
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
              hybrid_weight: float = None,
              email_max_chars: int = None,
              doc_max_chars: int = None) -> dict:
    """
    Run the full agentic pipeline for a user query.

    Returns answer, intent, sources, chunks_used, confidence,
    second_pass_fired, critic_verdict, critic_feedback.

    When the clarifier fires, returns early with:
        needs_clarification=True
        clarifying_questions=[list of question strings]
        answer="" (no answer yet)
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
        refinement_attempted=False,
        scope="all",
        email_max_chars=email_max_chars or 0,
        doc_max_chars=doc_max_chars or 0,
        needs_clarification=False,
        clarifying_questions=[],
        metadata_result={}
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
        "critic_feedback": final_state.get("critic_feedback", ""),
        "needs_clarification": final_state.get("needs_clarification", False),
        "clarifying_questions": final_state.get("clarifying_questions", [])
    }
