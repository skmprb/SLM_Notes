"""
API Routes:
  POST /upload              → ingest doc, store in session + background vectorization
  POST /query               → run QA agent, return JSON
  GET  /stream/{session_id} → SSE streaming answer
  GET  /sessions            → list user's active sessions
  DELETE /sessions/{id}     → clear a session
"""
import uuid
import asyncio
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.auth import get_current_user
from app.ingestion.extractor import extract
from app.memory.session_store import (
    store_doc_payload, get_doc_payload, delete_doc_payload,
    append_history, get_history, has_doc,
)
from app.memory import vector_store
from app.agents.graph import qa_graph
from app.agents.state import AgentState
from app.observability.tracer import get_logger, trace_span
from app.config.settings import get_settings

router = APIRouter()
logger = get_logger("routes")
settings = get_settings()

ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "image/png", "image/jpeg", "image/tiff",
}


# ── Request / Response schemas ────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    doc_id: str
    doc_name: str
    page_count: int
    message: str


class QueryRequest(BaseModel):
    query: str
    session_id: str
    doc_id: str | None = None   # optional: scope RAG to a specific doc


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float
    retrieval_path: str
    session_id: str


# ── Background task: vectorize after upload ───────────────────────────────────

async def _vectorize_background(user_id: str, doc_id: str, session_id: str) -> None:
    """Runs after upload response is sent — no user waiting on this."""
    payload = await get_doc_payload(user_id, session_id)
    if payload:
        with trace_span("background_vectorize", {"user_id": user_id, "doc_id": doc_id}):
            await vector_store.ingest_document(user_id, doc_id, payload)
        logger.info("vectorization_complete", extra={"extra": {"doc_id": doc_id, "user_id": user_id}})


# ── POST /upload ──────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
) -> UploadResponse:
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    # Validate file size
    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    # Save temp file
    session_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    upload_path = Path(settings.upload_dir) / f"{doc_id}_{file.filename}"
    upload_path.write_bytes(content)

    # Extract document content
    with trace_span("doc_extraction", {"user_id": user_id, "doc_id": doc_id}):
        payload = extract(upload_path)

    # Store in Redis session
    await store_doc_payload(user_id, session_id, payload)

    # Clean up temp file
    upload_path.unlink(missing_ok=True)

    # Vectorize in background — user doesn't wait
    background_tasks.add_task(_vectorize_background, user_id, doc_id, session_id)

    logger.info("upload_complete", extra={"extra": {
        "user_id": user_id, "doc_id": doc_id, "session_id": session_id,
        "doc_name": payload["doc_name"], "pages": payload["page_count"],
    }})

    return UploadResponse(
        session_id=session_id,
        doc_id=doc_id,
        doc_name=payload["doc_name"],
        page_count=payload["page_count"],
        message="Document uploaded. You can now ask questions.",
    )


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_document(
    req: QueryRequest,
    user_id: str = Depends(get_current_user),
) -> QueryResponse:
    history = await get_history(user_id, req.session_id)

    initial_state: AgentState = {
        "user_id": user_id,                    # from auth — never from client
        "session_id": req.session_id,
        "doc_id": req.doc_id,
        "query": req.query,
        "rewritten_query": None,
        "retry_count": 0,
        "query_type": "general",
        "retrieval_path": "",
        "doc_payload": None,
        "retrieved_chunks": [],
        "raw_answer": "",
        "grading_score": "",
        "final_answer": "",
        "sources": [],
        "confidence": 0.0,
        "history": history,
        "fallback": False,
    }

    with trace_span("agent_invoke", {"user_id": user_id, "session_id": req.session_id}):
        result: AgentState = await qa_graph.ainvoke(initial_state)

    # Persist conversation turn
    await append_history(user_id, req.session_id, "user", req.query)
    await append_history(user_id, req.session_id, "assistant", result["final_answer"])

    return QueryResponse(
        answer=result["final_answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        retrieval_path=result["retrieval_path"],
        session_id=req.session_id,
    )


# ── GET /stream/{session_id} ──────────────────────────────────────────────────

@router.get("/stream/{session_id}")
async def stream_query(
    session_id: str,
    query: str,
    user_id: str = Depends(get_current_user),
    doc_id: str | None = None,
):
    """SSE streaming endpoint — streams answer tokens as they arrive."""
    history = await get_history(user_id, session_id)

    initial_state: AgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "doc_id": doc_id,
        "query": query,
        "rewritten_query": None,
        "retry_count": 0,
        "query_type": "general",
        "retrieval_path": "",
        "doc_payload": None,
        "retrieved_chunks": [],
        "raw_answer": "",
        "grading_score": "",
        "final_answer": "",
        "sources": [],
        "confidence": 0.0,
        "history": history,
        "fallback": False,
    }

    async def event_generator():
        async for event in qa_graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            # Stream LLM tokens from answer nodes only
            if kind == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node", "")
                if node in ("answer_from_live_context", "answer_from_rag"):
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── GET /sessions ─────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/status")
async def session_status(
    session_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    doc_present = await has_doc(user_id, session_id)
    history = await get_history(user_id, session_id)
    return {
        "session_id": session_id,
        "has_document": doc_present,
        "history_turns": len(history),
    }


# ── DELETE /sessions/{id} ─────────────────────────────────────────────────────

@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    await delete_doc_payload(user_id, session_id)
    return {"message": "Session cleared", "session_id": session_id}
