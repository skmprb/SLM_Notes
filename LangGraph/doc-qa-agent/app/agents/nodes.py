"""
Agent Nodes: all node functions for the LangGraph QA graph.

Flow:
  analyze_query → check_session
      ├── has doc  → answer_from_live_context → grade_answer
      └── no doc   → answer_from_rag          → grade_answer
                                                    ├── good → store_to_vectorstore → format_response
                                                    └── poor → rewrite_query → (retry) or fallback_response
"""
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.memory.session_store import get_doc_payload, has_doc
from app.memory import vector_store
from app.observability.tracer import track_node, get_logger
from app.config.settings import get_settings

settings = get_settings()
logger = get_logger("nodes")

# ── LLM clients ──────────────────────────────────────────────────────────────

def _llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        max_tokens=settings.openai_max_tokens,
        openai_api_key=settings.openai_api_key,
    )


# ── Structured output schemas ─────────────────────────────────────────────────

class QueryAnalysis(BaseModel):
    query_type: Literal["factual", "summary", "table", "image", "general"]
    refined_query: str = Field(description="Cleaned, specific version of the user query")


class GradeResult(BaseModel):
    score: Literal["good", "poor"]
    reason: str


# ── Node 1: analyze_query ─────────────────────────────────────────────────────

@track_node("analyze_query")
async def analyze_query(state: AgentState) -> dict:
    """Classify query type and clean the query for better retrieval."""
    result: QueryAnalysis = await _llm().with_structured_output(QueryAnalysis).ainvoke([
        SystemMessage(content=(
            "Classify the user query into one of: factual, summary, table, image, general. "
            "Also return a refined, specific version of the query."
        )),
        HumanMessage(content=state["query"]),
    ])
    return {
        "query_type": result.query_type,
        "rewritten_query": result.refined_query,
    }


# ── Node 2: check_session ─────────────────────────────────────────────────────

@track_node("check_session")
async def check_session(state: AgentState) -> dict:
    """Load doc payload from Redis if available. Sets retrieval_path."""
    doc_in_session = await has_doc(state["user_id"], state["session_id"])

    if doc_in_session:
        payload = await get_doc_payload(state["user_id"], state["session_id"])
        return {"doc_payload": payload, "retrieval_path": "live_context"}

    return {"doc_payload": None, "retrieval_path": "rag"}


# ── Node 3a: answer_from_live_context ─────────────────────────────────────────

@track_node("answer_from_live_context")
async def answer_from_live_context(state: AgentState) -> dict:
    """
    Answer using the full document loaded directly into LLM context.
    Multimodal: sends text + tables + base64 images in one call.
    Token-optimized: images only sent for image/graph query types.
    """
    payload = state["doc_payload"]
    query = state["rewritten_query"] or state["query"]

    # Build message content
    content: list = [
        {
            "type": "text",
            "text": (
                f"Document: {payload['doc_name']} ({payload['page_count']} pages)\n\n"
                f"{payload['text']}\n\n"
                f"Conversation history:\n"
                + "\n".join(f"{h['role']}: {h['content']}" for h in state.get("history", []))
                + f"\n\nQuestion: {query}"
            ),
        }
    ]

    # Only attach images for image/graph queries — saves tokens for text queries
    if state["query_type"] in ("image", "general") and payload["images"]:
        for b64 in payload["images"][:5]:   # cap at 5 images per call
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "auto"},
            })

    system = (
        "You are a precise document analyst. Answer ONLY from the provided document. "
        "Cite page references where possible. If the answer is not in the document, say so."
    )

    response = await _llm().ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=content),
    ])

    return {"raw_answer": response.content}


# ── Node 3b: answer_from_rag ──────────────────────────────────────────────────

@track_node("answer_from_rag")
async def answer_from_rag(state: AgentState) -> dict:
    """
    Answer using RAG — retrieves from ChromaDB filtered strictly by user_id.
    Falls back gracefully if no chunks found.
    """
    query = state["rewritten_query"] or state["query"]

    chunks = await vector_store.retrieve(
        user_id=state["user_id"],
        query=query,
        doc_id=state.get("doc_id"),
    )

    if not chunks:
        return {
            "raw_answer": "",
            "retrieved_chunks": [],
            "fallback": True,
        }

    context = "\n\n---\n\n".join(
        f"[{c['metadata']['type'].upper()} | doc: {c['metadata']['doc_name']}]\n{c['content']}"
        for c in chunks
    )

    system = (
        "You are a precise document analyst. Answer ONLY from the provided context. "
        "Cite the document name and chunk type. If the answer is not in the context, say so."
    )
    prompt = (
        f"Context:\n{context}\n\n"
        f"History:\n"
        + "\n".join(f"{h['role']}: {h['content']}" for h in state.get("history", []))
        + f"\n\nQuestion: {query}"
    )

    response = await _llm().ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])

    sources = [
        {
            "doc_name": c["metadata"]["doc_name"],
            "type": c["metadata"]["type"],
            "score": c["score"],
        }
        for c in chunks
    ]

    return {
        "raw_answer": response.content,
        "retrieved_chunks": [c["content"] for c in chunks],
        "sources": sources,
    }


# ── Node 4: grade_answer ──────────────────────────────────────────────────────

@track_node("grade_answer")
async def grade_answer(state: AgentState) -> dict:
    """
    Grade whether the answer is grounded and sufficient.
    Uses structured output for reliable binary scoring.
    """
    if not state.get("raw_answer"):
        return {"grading_score": "poor"}

    context = (
        state["doc_payload"]["text"][:3000]     # first 3k chars for grading context
        if state["retrieval_path"] == "live_context" and state.get("doc_payload")
        else "\n".join(state.get("retrieved_chunks", []))[:3000]
    )

    result: GradeResult = await _llm().with_structured_output(GradeResult).ainvoke([
        SystemMessage(content=(
            "Grade whether the answer is grounded in the provided context. "
            "Score 'good' if the answer is supported by the context. "
            "Score 'poor' if the answer is vague, hallucinated, or says it cannot answer."
        )),
        HumanMessage(content=(
            f"Context (excerpt):\n{context}\n\n"
            f"Question: {state['query']}\n\n"
            f"Answer: {state['raw_answer']}"
        )),
    ])

    return {"grading_score": result.score}


# ── Node 5: rewrite_query ─────────────────────────────────────────────────────

@track_node("rewrite_query")
async def rewrite_query(state: AgentState) -> dict:
    """Reformulate the query to improve retrieval on retry."""
    response = await _llm().ainvoke([
        SystemMessage(content=(
            "Rewrite the following question to be more specific and retrieval-friendly. "
            "Return only the rewritten question, nothing else."
        )),
        HumanMessage(content=state["rewritten_query"] or state["query"]),
    ])
    return {
        "rewritten_query": response.content.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ── Node 6: store_to_vectorstore ──────────────────────────────────────────────

@track_node("store_to_vectorstore")
async def store_to_vectorstore(state: AgentState) -> dict:
    """
    Background-style: store the doc payload to ChromaDB after live-context answer.
    Only runs on live_context path to build up the vector store for future RAG.
    Skipped if doc_id already indexed (idempotent).
    """
    if state.get("doc_payload") and state.get("doc_id"):
        await vector_store.ingest_document(
            user_id=state["user_id"],
            doc_id=state["doc_id"],
            payload=state["doc_payload"],
        )
    return {}


# ── Node 7: fallback_response ─────────────────────────────────────────────────

@track_node("fallback_response")
async def fallback_response(state: AgentState) -> dict:
    """Return a graceful fallback when retries are exhausted."""
    return {
        "final_answer": (
            "I was unable to find a confident answer in your document. "
            "Please try rephrasing your question or upload the document again."
        ),
        "sources": [],
        "confidence": 0.0,
        "fallback": True,
    }


# ── Node 8: format_response ───────────────────────────────────────────────────

@track_node("format_response")
async def format_response(state: AgentState) -> dict:
    """Package the final answer with sources and confidence score."""
    sources = state.get("sources", [])
    confidence = (
        0.9 if state["retrieval_path"] == "live_context"
        else (max((s["score"] for s in sources), default=0.5) if sources else 0.5)
    )
    return {
        "final_answer": state["raw_answer"],
        "sources": sources,
        "confidence": round(confidence, 2),
    }
