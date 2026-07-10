from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Session ID to continue a conversation")
    provider: Optional[str] = Field(default=None, description="Override LLM provider")
    model: Optional[str] = Field(default=None, description="Override model name")


class ChatResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: Optional[str] = None
    message: str
    model: str
    provider: str = "openai"
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    routing_reason: Optional[str] = None
    was_fallback: bool = False
    attempts: int = 1
    fallback_reason: Optional[str] = None
    tools_used: Optional[list[dict]] = None  # [{name, success}] if tools were called
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class StructuredRequest(BaseModel):
    """Request for structured output generation."""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    output_schema: dict = Field(..., description="JSON Schema for expected output")
    schema_name: str = Field(default="output", description="Name for the schema")
    provider: Optional[str] = Field(default=None, description="Override LLM provider")
    strategy: Optional[str] = Field(default=None, description="json_schema | function_calling | json_mode")


class StructuredResponse(BaseModel):
    """Response from structured output generation."""
    data: Optional[dict] = None
    raw_text: str = ""
    success: bool = True
    strategy_used: str = ""
    model: str = ""
    provider: str = ""
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    validation_error: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    title: Optional[str] = None
    message_count: int = 0
    total_tokens: int = 0
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    environment: str
    default_provider: str = ""
    routing_strategy: str = ""
