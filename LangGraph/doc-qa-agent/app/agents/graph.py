"""
LangGraph StateGraph: Doc QA Agent

Routing logic:
  - has doc in session  → live_context path
  - no doc in session   → RAG path
  - poor grade + retries left → rewrite → retry
  - poor grade + retries exhausted → fallback
"""
from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.nodes import (
    analyze_query,
    check_session,
    answer_from_live_context,
    answer_from_rag,
    grade_answer,
    rewrite_query,
    store_to_vectorstore,
    fallback_response,
    format_response,
)
from app.config.settings import get_settings

settings = get_settings()


# ── Conditional edge functions ────────────────────────────────────────────────

def route_by_session(state: AgentState) -> str:
    """Route to live context or RAG based on session doc availability."""
    return state["retrieval_path"]  # "live_context" or "rag"


def route_by_grade(state: AgentState) -> str:
    """After grading, decide: format, retry, or fallback."""
    if state.get("fallback"):
        return "fallback"
    if state["grading_score"] == "good":
        return "good"
    # Poor grade — check retry budget
    if state.get("retry_count", 0) < settings.max_rewrite_retries:
        return "retry"
    return "fallback"


def route_after_rewrite(state: AgentState) -> str:
    """After rewriting, go back to the same retrieval path."""
    return state["retrieval_path"]


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("check_session", check_session)
    graph.add_node("answer_from_live_context", answer_from_live_context)
    graph.add_node("answer_from_rag", answer_from_rag)
    graph.add_node("grade_answer", grade_answer)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("store_to_vectorstore", store_to_vectorstore)
    graph.add_node("fallback_response", fallback_response)
    graph.add_node("format_response", format_response)

    # Entry point
    graph.set_entry_point("analyze_query")

    # Linear edges
    graph.add_edge("analyze_query", "check_session")

    # Branch: live context vs RAG
    graph.add_conditional_edges(
        "check_session",
        route_by_session,
        {
            "live_context": "answer_from_live_context",
            "rag": "answer_from_rag",
        },
    )

    # Both paths converge at grade_answer
    graph.add_edge("answer_from_live_context", "grade_answer")
    graph.add_edge("answer_from_rag", "grade_answer")

    # Grade routing
    graph.add_conditional_edges(
        "grade_answer",
        route_by_grade,
        {
            "good": "store_to_vectorstore",     # live path: store then format
            "retry": "rewrite_query",
            "fallback": "fallback_response",
        },
    )

    # store → format (only on live_context good path)
    # RAG good path: store_to_vectorstore is a no-op (no doc_payload), goes to format
    graph.add_edge("store_to_vectorstore", "format_response")

    # Retry loop: rewrite → back to same retrieval path
    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "live_context": "answer_from_live_context",
            "rag": "answer_from_rag",
        },
    )

    # Terminal edges
    graph.add_edge("fallback_response", END)
    graph.add_edge("format_response", END)

    return graph.compile()


# Compiled graph — import this in routes
qa_graph = build_graph()
