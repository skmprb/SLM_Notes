"""
API Routes - Phase 15 (v1.0.0): All production concerns integrated.

Endpoints:
- POST /chat: Chat with memory + tools + routing + resilience
- POST /chat/stream: Streaming chat via SSE
- POST /chat/structured: Structured output (JSON schema enforcement)
- GET/POST/DELETE /sessions: Session management
- GET /tools: List available tools
- POST /route: Test routing decisions
- GET /circuits: Circuit breaker status
- GET /metrics: Platform metrics & cost tracking
- GET /guardrails/test: Test guardrail checks
- GET /prompts: List registered prompts
- GET /health: Health check
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from app.models.schemas import ChatRequest, ChatResponse, StreamRequest, HealthResponse, StructuredRequest, StructuredResponse
from app.services.chat_service import ChatService
from app.services.streaming_service import StreamingService
from app.services.conversation_service import ConversationService
from app.tools import get_tool_registry
from app.structured import StructuredOutputService, OutputSchema, StructuredOutputStrategy
from app.guardrails import get_guardrail_pipeline
from app.observability import get_metrics, get_cost_tracker
from app.prompts import get_prompt_registry
from app.middleware import get_tenant_manager
from app.llm.router import create_router, RoutingContext
from app.config.settings import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()
chat_service = ChatService()
streaming_service = StreamingService()
conversation_service = ConversationService()
structured_service = StructuredOutputService()


class RouteTestRequest(BaseModel):
    message: str = Field(..., min_length=1)
    has_images: bool = False
    estimated_tokens: int = 0


class RouteTestResponse(BaseModel):
    provider: str
    model: str
    reason: str
    estimated_cost: Optional[float] = None


# ============================================================
# CHAT ENDPOINTS
# ============================================================

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat with conversation memory.

    - First request: omit conversation_id → creates new session
    - Subsequent requests: pass conversation_id → continues conversation
    - Response includes conversation_id for continuing

    Example flow:
        1. POST /chat {"message": "What is Python?"}
           → {"conversation_id": "abc-123", "message": "Python is..."}

        2. POST /chat {"message": "What about Java?", "conversation_id": "abc-123"}
           → {"conversation_id": "abc-123", "message": "Java is... (knows you asked about Python)"}
    """
    logger.info(f"POST /chat | session={request.conversation_id or 'new'} | len={len(request.message)}")

    try:
        response = chat_service.chat(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat failed | error={type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred. Please try again.")


@router.post("/chat/stream")
def chat_stream(request: StreamRequest):
    """Streaming chat with SSE."""
    logger.info(f"POST /chat/stream | session={request.conversation_id or 'new'} | len={len(request.message)}")

    # Build messages with conversation history
    session = conversation_service.get_or_create_session(request.conversation_id)
    messages = conversation_service.prepare_messages(session.id, request.message)

    # Route
    provider = request.provider
    model = request.model
    if not provider:
        settings = get_settings()
        model_router = create_router(settings.routing_strategy)
        context = RoutingContext(message=request.message, estimated_tokens=len(request.message) // 4)
        decision = model_router.route(context)
        provider = decision.provider
        model = decision.model

    return StreamingResponse(
        streaming_service.stream_chat(messages, provider=provider, model=model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ============================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================

@router.post("/chat/structured", response_model=StructuredResponse)
def chat_structured(request: StructuredRequest):
    """
    Generate structured output matching a JSON schema.

    The LLM is forced to return data matching your schema.

    Example:
        POST /chat/structured {
            "message": "John is 30 years old and lives in NYC",
            "output_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"}
                },
                "required": ["name", "age", "city"]
            }
        }
        → {"data": {"name": "John", "age": 30, "city": "NYC"}, "success": true}
    """
    logger.info(f"POST /chat/structured | schema={request.schema_name} | strategy={request.strategy}")

    try:
        schema = OutputSchema.from_json_schema(
            name=request.schema_name,
            schema=request.output_schema,
        )
        strategy = StructuredOutputStrategy(request.strategy) if request.strategy else None

        result = structured_service.generate(
            messages=[{"role": "user", "content": request.message}],
            schema=schema,
            provider_name=request.provider,
            strategy=strategy,
        )

        # Convert Pydantic model to dict if needed
        data = result.data
        if hasattr(data, "model_dump"):
            data = data.model_dump()

        return StructuredResponse(
            data=data if isinstance(data, dict) else None,
            raw_text=result.raw_text,
            success=result.success,
            strategy_used=result.strategy_used.value,
            model=result.model,
            provider=result.provider,
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms,
            validation_error=result.validation_error,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Structured output failed | {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Structured output generation failed.")


@router.get("/sessions")
def list_sessions(limit: int = 20, offset: int = 0):
    """List all conversation sessions."""
    sessions = conversation_service.list_sessions(limit, offset)
    return {"sessions": sessions, "count": len(sessions)}


@router.post("/sessions")
def create_session():
    """Create a new conversation session."""
    session = conversation_service.create_session()
    return {"id": session.id, "created_at": session.created_at.isoformat()}


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get session details with all messages."""
    session = conversation_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = conversation_service.get_session_messages(session_id)
    return {
        "id": session.id,
        "title": session.title,
        "message_count": session.message_count,
        "total_tokens": session.total_tokens,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "messages": messages,
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a session and all its messages."""
    deleted = conversation_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ============================================================
# ROUTING & RESILIENCE ENDPOINTS
# ============================================================

@router.get("/tools")
def list_tools():
    """List all available tools the LLM can use."""
    registry = get_tool_registry()
    return {"tools": registry.list_tools(), "count": registry.tool_count}


@router.post("/route", response_model=RouteTestResponse)
def test_route(request: RouteTestRequest):
    """Test routing decision without calling LLM."""
    settings = get_settings()
    model_router = create_router(settings.routing_strategy)
    context = RoutingContext(
        message=request.message,
        has_images=request.has_images,
        estimated_tokens=request.estimated_tokens or len(request.message) // 4,
    )
    decision = model_router.route(context)
    return RouteTestResponse(
        provider=decision.provider, model=decision.model,
        reason=decision.reason, estimated_cost=decision.estimated_cost,
    )


@router.get("/circuits")
def get_circuit_status():
    """View circuit breaker status."""
    return {"circuits": chat_service.get_circuit_status()}


@router.post("/circuits/reset")
def reset_circuits():
    """Reset all circuit breakers."""
    chat_service.resilient_llm.fallback_manager.reset_circuits()
    return {"status": "All circuit breakers reset"}


@router.get("/health", response_model=HealthResponse)
def health():
    """Health check."""
    settings = get_settings()
    return HealthResponse(
        status="healthy", version=settings.app_version,
        environment=settings.app_env, default_provider=settings.default_provider,
        routing_strategy=settings.routing_strategy,
    )


# ============================================================
# OBSERVABILITY & PLATFORM ENDPOINTS
# ============================================================

@router.get("/metrics")
def get_platform_metrics():
    """Get platform metrics, cost tracking, and performance stats."""
    return {
        "metrics": get_metrics().get_summary(),
        "costs": get_cost_tracker().get_summary(),
    }


@router.get("/prompts")
def list_prompts():
    """List all registered prompt templates."""
    registry = get_prompt_registry()
    return {"prompts": registry.list_prompts()}


@router.post("/guardrails/test")
def test_guardrails(content: str):
    """Test input guardrails against a string."""
    pipeline = get_guardrail_pipeline()
    result = pipeline.check_input(content)
    return {
        "passed": result.passed,
        "action": result.action.value,
        "reason": result.reason,
        "violations": result.violations,
        "redacted": result.redacted_content,
    }
