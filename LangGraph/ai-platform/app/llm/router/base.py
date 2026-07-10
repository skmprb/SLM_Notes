"""
Model Router - Base Interface

A Model Router decides WHICH model to use for a given request.
This is different from LangChain's multi-agent router (which routes to agents).

Why route?
    - Cost: Simple questions don't need GPT-4o ($$$), use GPT-4o-mini ($)
    - Capability: Vision tasks need Gemini/GPT-4o, not text-only models
    - Latency: Real-time chat needs fast models, batch jobs can use slow/cheap ones
    - Load: If one provider is overloaded, route to another

Routing strategies:
    1. Rule-based (keywords, message length, content type)
    2. Cost-aware (cheapest model that can handle the task)
    3. LLM-based (use a small/fast model to classify, then route to the right model)
    4. Capability-based (match task requirements to model capabilities)

LangChain equivalent:
    - ChatLiteLLMRouter (external routing via LiteLLM)
    - Configurable models with runtime switching
    - Middleware-based dynamic model selection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class TaskType(Enum):
    """Classification of task types for routing."""
    SIMPLE_QA = "simple_qa"
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    VISION = "vision"
    LONG_CONTEXT = "long_context"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"


@dataclass
class RoutingContext:
    """All information the router needs to make a decision."""
    message: str
    task_type: Optional[TaskType] = None
    has_images: bool = False
    estimated_tokens: int = 0
    priority: str = "normal"  # low, normal, high
    max_cost_per_token: Optional[float] = None
    max_latency_ms: Optional[float] = None
    required_capabilities: list[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """The router's output - which provider and model to use."""
    provider: str
    model: str
    reason: str
    estimated_cost: Optional[float] = None
    estimated_latency_ms: Optional[float] = None


class BaseRouter(ABC):
    """Abstract base for all routing strategies."""

    @abstractmethod
    def route(self, context: RoutingContext) -> RoutingDecision:
        """Decide which model to use based on context."""
        ...
