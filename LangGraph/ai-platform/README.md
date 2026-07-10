# Enterprise AI Platform

A production-grade GenAI platform built incrementally, one production concern at a time.

## Phase 1: Simplest Working Product ✅

### Architecture

```
Client (POST /api/v1/chat)
    │
    ▼
API Route (routes.py)
    │  - Request validation (Pydantic)
    │  - Error handling
    │  - Logging
    ▼
ChatService (chat_service.py)
    │  - Business logic
    │  - Message building
    │  - Orchestration
    ▼
LLMService (llm_service.py)
    │  - Direct OpenAI communication
    │  - Latency tracking
    │  - Token counting
    ▼
OpenAI API
```

### What You Learn in Phase 1

| Concept | Where |
|---------|-------|
| Project structure | `app/` folder layout |
| Service layer pattern | ChatService → LLMService |
| Dependency Injection | Constructor injection in services |
| Configuration | pydantic-settings + .env |
| Structured logging | Custom logger with consistent format |
| Request/Response models | Pydantic schemas |
| API versioning | `/api/v1/` prefix |
| Health checks | `/api/v1/health` |

### Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file and add your API key
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run
uvicorn app.main:app --reload
```

### Test

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is dependency injection?"}'
```

### API Docs

FastAPI auto-generates docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Roadmap

- [x] Phase 1: Simplest Working Product
- [ ] Phase 2: LLM Abstraction Layer (multiple providers)
- [ ] Phase 3: Model Router
- [ ] Phase 4: Retry & Fallback
- [ ] Phase 5: Streaming
- [ ] Phase 6: Conversation Management
- [ ] Phase 7: Tool Calling
- [ ] Phase 8: Agent Framework
- [ ] Phase 9: Graph Engine (LangGraph)
- [ ] Phase 10: RAG
- [ ] Phase 11: Memory
- [ ] Phase 12: Observability
- [ ] Phase 13: Queue System
- [ ] Phase 14: Production Readiness
- [ ] Phase 15: Kubernetes Deployment
