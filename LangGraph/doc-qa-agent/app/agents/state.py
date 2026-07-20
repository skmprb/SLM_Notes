from typing import TypedDict, Annotated, Literal
from operator import add


class DocPayload(TypedDict):
    text: str                       # full extracted text
    tables: list[str]               # each table as markdown string
    images: list[str]               # base64 encoded images/graphs
    page_count: int
    doc_name: str


class AgentState(TypedDict):
    # Identity — injected from auth, never from client
    user_id: str
    session_id: str
    doc_id: str | None

    # Input
    query: str
    rewritten_query: str | None
    retry_count: int

    # Routing
    query_type: Literal["factual", "summary", "table", "image", "general"]
    retrieval_path: Literal["live_context", "rag", ""] 

    # Doc context (only when doc is in session)
    doc_payload: DocPayload | None

    # Retrieved content
    retrieved_chunks: Annotated[list[str], add]     # accumulates across retries

    # Answer
    raw_answer: str
    grading_score: Literal["good", "poor", ""]
    final_answer: str
    sources: list[dict]                             # [{page, type, excerpt}]
    confidence: float

    # Conversation history (last N turns stored in Redis, passed here)
    history: list[dict]                             # [{"role": "user/assistant", "content": "..."}]

    # Control
    fallback: bool                                  # True when retries exhausted
