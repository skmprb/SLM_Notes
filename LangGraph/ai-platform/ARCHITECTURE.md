# Enterprise AI Platform — Architecture (v1.0.0)

> 15 phases, 40 production concerns, ONE evolving codebase.

---

## Platform Overview

```
                    Client (API / UI)
                         │
                    ┌────┴────┐
                    │ FastAPI │  ← Phase 1: API Layer
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
     Rate Limiter   Guardrails   Auth
     (Phase 14)    (Phase 12)   (future)
              │          │          │
              └──────────┼──────────┘
                         │
                  ┌──────┴──────┐
                  │ ChatService │  ← Orchestrator
                  └──────┬──────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
   Conversation     Model Router     Tool System
   (Phase 6)        (Phase 3)        (Phase 7)
         │               │               │
    Session DB      ┌────┴────┐     ToolRegistry
    Messages        │Strategies│    ToolExecutor
    Context         │Rule/Cost │    Built-in tools
                    │/LLM-based│
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
         LLM Service  Cache     Structured
         (Phase 2+4)  (Phase 11) (Phase 8)
              │
    ┌─────────┼─────────┐
    │         │         │
  Retry    Circuit    Fallback
  Handler  Breaker   Manager
    │         │         │
    └─────────┼─────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
  OpenAI  Anthropic  Gemini
    │         │         │
    └─────────┼─────────┘
              │
         ┌────┴────┐
         │Observability│  ← Phase 13
         │Tracing/Metrics│
         │Cost Tracking │
         └─────────────┘
```

---

## Phase Summary

| Phase | Module | Production Concern | Key Classes |
|-------|--------|-------------------|-------------|
| 1 | `app/` | API Layer + Config + Logging | FastAPI, Settings, Logger |
| 2 | `app/llm/` | LLM Abstraction (multi-provider) | BaseLLM, Factory, Providers |
| 3 | `app/llm/router/` | Model Routing (3 strategies) | RuleBasedRouter, CostAwareRouter, LLMRouter |
| 4 | `app/llm/resilience/` | Retry + Circuit Breaker + Fallback | RetryHandler, CircuitBreaker, FallbackManager |
| 5 | `app/services/streaming_service.py` | Streaming (SSE) | StreamingService, StreamChunk |
| 6 | `app/storage/` + `app/services/conversation_service.py` | Conversation Management | Session, Message, ConversationStore |
| 7 | `app/tools/` | Tool Calling (registry + execution) | ToolRegistry, ToolExecutor, ToolDefinition |
| 8 | `app/structured/` | Structured Output (JSON enforcement) | OutputSchema, StructuredOutputService |
| 9 | `app/prompts/` | Prompt Management (templates + versioning) | PromptTemplate, PromptRegistry |
| 10 | `app/rag/` | RAG (chunk + embed + retrieve) | Chunker, EmbeddingService, VectorStore |
| 11 | `app/cache/` | Response Caching (LRU + TTL) | ResponseCache, InMemoryLRUCache |
| 12 | `app/guardrails/` | Input/Output Guardrails | PIIDetector, InjectionDetector, Pipeline |
| 13 | `app/observability/` | Tracing + Metrics + Cost Tracking | Tracer, MetricsCollector, CostTracker |
| 14 | `app/middleware/` | Rate Limiting + Multi-tenancy | RateLimiter, TenantManager, TenantConfig |
| 15 | `app/evaluation/` | Evaluation Framework | TestCase, EvaluationRunner, EvalSuiteResult |

---

## Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| Abstract Base Class | BaseLLM, BaseRouter, BaseCache, BaseVectorStore | Swap implementations without changing callers |
| Factory | create_llm(), create_router() | Decouple creation from usage |
| Registry | ToolRegistry, PromptRegistry | Central lookup, dynamic registration |
| Strategy | Routing strategies, Structured output strategies | Multiple algorithms, same interface |
| Circuit Breaker | CircuitBreaker | Prevent cascading failures |
| Token Bucket | RateLimiter | Smooth rate limiting with burst support |
| Pipeline | GuardrailPipeline | Chain of responsibility for checks |
| Observer | MetricsCollector, CostTracker | Decouple metric recording from business logic |
| Adapter | Provider classes | Normalize different API shapes |
| Template Method | BaseLLM.generate_with_tools() default | Override in subclasses, fallback in base |

---

## API Endpoints

| Method | Path | Phase | Description |
|--------|------|-------|-------------|
| POST | /chat | 1-7 | Chat with memory + tools + routing |
| POST | /chat/stream | 5 | Streaming chat via SSE |
| POST | /chat/structured | 8 | Structured output (JSON schema) |
| GET | /sessions | 6 | List conversation sessions |
| POST | /sessions | 6 | Create new session |
| GET | /sessions/{id} | 6 | Get session with messages |
| DELETE | /sessions/{id} | 6 | Delete session |
| GET | /tools | 7 | List available tools |
| POST | /route | 3 | Test routing decision |
| GET | /circuits | 4 | Circuit breaker status |
| POST | /circuits/reset | 4 | Reset circuit breakers |
| GET | /metrics | 13 | Platform metrics & costs |
| GET | /prompts | 9 | List prompt templates |
| POST | /guardrails/test | 12 | Test guardrail checks |
| GET | /health | 1 | Health check |

---

## LangChain/LangGraph Mapping (Complete)

| Our Platform | LangChain/LangGraph |
|-------------|---------------------|
| BaseLLM | BaseChatModel |
| create_llm() | init_chat_model() |
| generate() | model.invoke() |
| stream() | model.stream() |
| generate_with_tools() | model.bind_tools().invoke() |
| ToolRegistry + ToolExecutor | ToolNode / create_react_agent() |
| StructuredOutputService | model.with_structured_output() |
| PromptTemplate | ChatPromptTemplate |
| ConversationService | InMemoryChatMessageHistory / checkpointer |
| RetrievalService | create_retrieval_chain() |
| ResponseCache | InMemoryCache / set_llm_cache() |
| FallbackManager | model.with_fallbacks() |
| RetryHandler | model.with_retry() |
| ModelRouter | Custom routing in LangGraph |
| GuardrailPipeline | NeMo Guardrails |
| Tracer | LangSmith tracing |
| CostTracker | get_openai_callback() |
| EvaluationRunner | LangSmith evaluate() |
| TenantManager | LangGraph Platform |

---

## Key Insight

> LangChain/LangGraph occupy ONE layer of this stack.
> The surrounding 80-90% — reliability, scalability, observability,
> security, cost management, and operational excellence — is what
> turns an LLM prototype into a production-grade AI platform.
> 
> By building each layer from scratch, you now understand what
> every framework does under the hood.
