# Doc QA Agent

Agentic document Q&A system built with LangGraph. Upload any document (PDF, DOCX, images) and ask questions — agents answer from live document context or fall back to vector store RAG for previously uploaded docs.

---

## Architecture

```
POST /upload
  └── extract (text + tables + images)
  └── store in Redis session (live context)
  └── background: chunk → embed → ChromaDB (per user, isolated)

POST /query
  └── LangGraph agent graph:
        analyze_query
            └── check_session
                  ├── doc in session → answer_from_live_context (GPT-4o multimodal)
                  └── no doc        → answer_from_rag (ChromaDB, user-isolated)
                        └── grade_answer
                              ├── good  → store_to_vectorstore → format_response
                              ├── retry → rewrite_query → retry retrieval (max 2x)
                              └── fail  → fallback_response
```

### User Isolation
- Redis keys: `session:{user_id}:{session_id}:*` — namespaced by user
- ChromaDB: single collection, every query filtered `WHERE user_id = {uid}` server-side
- `user_id` always comes from auth header — never trusted from client payload

---

## Quickstart

### 1. Clone and configure
```bash
cp .env.example .env
# Fill in OPENAI_API_KEY
```

### 2. Run with Docker (recommended)
```bash
docker-compose up --build
```

### 3. Run locally
```bash
pip install -r requirements.txt
# Start Redis and ChromaDB separately, then:
uvicorn app.main:app --reload --port 8000
```

---

## API Usage

### Upload a document
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -H "X-User-ID: user_123" \
  -F "file=@report.pdf"

# Response:
# { "session_id": "abc-123", "doc_id": "xyz-456", "doc_name": "report.pdf", "page_count": 12 }
```

### Ask a question (doc in session → live context)
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-User-ID: user_123" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "session_id": "abc-123"}'
```

### Ask a question (no doc in session → RAG from vector store)
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-User-ID: user_123" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the Q3 report", "session_id": "new-session-id"}'
```

### Stream answer (SSE)
```bash
curl -N "http://localhost:8000/api/v1/stream/abc-123?query=What+is+the+revenue&X-User-ID=user_123"
```

### Check session status
```bash
curl http://localhost:8000/api/v1/sessions/abc-123/status \
  -H "X-User-ID: user_123"
```

### Clear session
```bash
curl -X DELETE http://localhost:8000/api/v1/sessions/abc-123 \
  -H "X-User-ID: user_123"
```

---

## Project Structure

```
doc-qa-agent/
├── app/
│   ├── api/
│   │   ├── auth.py              # X-User-ID extraction (swap for JWT)
│   │   └── routes.py            # All API endpoints
│   ├── agents/
│   │   ├── state.py             # AgentState TypedDict
│   │   ├── nodes.py             # All 8 node functions
│   │   └── graph.py             # LangGraph StateGraph
│   ├── ingestion/
│   │   └── extractor.py         # PDF/DOCX/image → DocPayload
│   ├── memory/
│   │   ├── session_store.py     # Redis: doc payload + history
│   │   └── vector_store.py      # ChromaDB: user-isolated retrieval
│   ├── observability/
│   │   └── tracer.py            # Structured logs + LangSmith tracing
│   ├── config/
│   │   └── settings.py          # Pydantic BaseSettings
│   └── main.py                  # FastAPI app
├── uploads/                     # Temp storage, auto-cleaned post-extraction
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Observability

Every agent node emits a structured JSON log:
```json
{
  "ts": "2024-01-01T00:00:00",
  "level": "INFO",
  "node": "grade_answer",
  "user_id": "user_123",
  "session_id": "abc-123",
  "retrieval_path": "live_context",
  "retry_count": 0,
  "latency_ms": 342.5,
  "grading_score": "good"
}
```

Enable LangSmith tracing by setting `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`.

---

## Token Optimization

| Strategy | Detail |
|---|---|
| Images only when needed | Base64 images sent to GPT-4o only for `image`/`general` query types |
| Image resize | Images capped at 1024px wide before encoding |
| Context cap for grading | Only first 3000 chars used for grade_answer node |
| History window | Last 20 turns kept in Redis, older turns dropped |
| Embedding model | `text-embedding-3-small` — 5x cheaper than large |
| Chunk size | 512 tokens with 64 overlap — minimal redundancy |

---

## Production Swap Points

| Component | Current | Production Swap |
|---|---|---|
| Auth | `X-User-ID` header | JWT decode in `app/api/auth.py` |
| Vector store | ChromaDB (local) | Qdrant / Pinecone (managed) |
| Session store | Redis (single) | Redis Cluster / ElastiCache |
| LLM | GPT-4o | Any OpenAI-compatible endpoint |
| File storage | Local `uploads/` | S3 / GCS pre-signed URLs |
