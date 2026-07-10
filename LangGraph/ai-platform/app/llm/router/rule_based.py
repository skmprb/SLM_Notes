"""
Rule-Based Router - Routes based on keywords, patterns, and message properties.

This is the MOST COMMON router in production because:
    - Zero latency (no LLM call needed to route)
    - Deterministic (same input → same route)
    - Easy to debug and explain
    - No additional cost

Real-world examples:
    - Stripe: Routes simple billing questions to cheap models
    - GitHub Copilot: Routes code completion vs explanation to different models
    - Customer support: Routes "cancel subscription" to human, rest to AI

When to use:
    - You have clear, well-defined categories
    - Latency matters (can't afford an extra LLM call to classify)
    - You need deterministic behavior
"""

import re
from app.llm.router.base import BaseRouter, RoutingContext, RoutingDecision, TaskType
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Cost per 1M tokens (approximate, for routing decisions)
MODEL_COSTS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}


class RuleBasedRouter(BaseRouter):
    """
    Routes requests based on configurable rules.

    Rule priority (first match wins):
        1. Has images → vision-capable model
        2. Long context → large context window model
        3. Coding keywords → coding-optimized model
        4. Reasoning keywords → reasoning model
        5. Default → cheapest capable model
    """

    def __init__(self):
        # Configurable routing rules
        self.coding_keywords = [
            "code", "function", "class", "debug", "error", "bug",
            "python", "javascript", "typescript", "java", "sql",
            "implement", "refactor", "algorithm", "api", "endpoint",
        ]
        self.reasoning_keywords = [
            "analyze", "compare", "evaluate", "reason", "think",
            "step by step", "pros and cons", "trade-off", "why",
            "explain the logic", "what would happen if",
        ]
        self.creative_keywords = [
            "write a story", "poem", "creative", "imagine",
            "brainstorm", "generate ideas",
        ]

    def route(self, context: RoutingContext) -> RoutingDecision:
        """Apply rules in priority order."""
        message_lower = context.message.lower()

        # Rule 1: Vision tasks
        if context.has_images or self._needs_vision(message_lower):
            return RoutingDecision(
                provider="openai",
                model="gpt-4o",
                reason="Vision capability required",
                estimated_cost=MODEL_COSTS["gpt-4o"]["input"],
            )

        # Rule 2: Long context (>4000 tokens estimated)
        if context.estimated_tokens > 4000 or len(context.message) > 16000:
            return RoutingDecision(
                provider="google",
                model="gemini-1.5-pro",
                reason="Long context - Gemini has 1M token window",
                estimated_cost=MODEL_COSTS["gemini-1.5-pro"]["input"],
            )

        # Rule 3: Coding tasks
        if self._is_coding_task(message_lower):
            return RoutingDecision(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                reason="Coding task - Claude excels at code",
                estimated_cost=MODEL_COSTS["claude-3-5-sonnet-20241022"]["input"],
            )

        # Rule 4: Complex reasoning
        if self._is_reasoning_task(message_lower):
            return RoutingDecision(
                provider="openai",
                model="gpt-4o",
                reason="Complex reasoning task",
                estimated_cost=MODEL_COSTS["gpt-4o"]["input"],
            )

        # Rule 5: Creative tasks
        if self._is_creative_task(message_lower):
            return RoutingDecision(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                reason="Creative writing task",
                estimated_cost=MODEL_COSTS["claude-3-5-sonnet-20241022"]["input"],
            )

        # Default: Cheapest model for simple tasks
        return RoutingDecision(
            provider="google",
            model="gemini-1.5-flash",
            reason="Simple task - using cheapest model",
            estimated_cost=MODEL_COSTS["gemini-1.5-flash"]["input"],
        )

    def _needs_vision(self, message: str) -> bool:
        vision_patterns = ["image", "picture", "photo", "screenshot", "diagram", "look at this"]
        return any(p in message for p in vision_patterns)

    def _is_coding_task(self, message: str) -> bool:
        # Check for code blocks
        if "```" in message:
            return True
        # Check for coding keywords
        return sum(1 for kw in self.coding_keywords if kw in message) >= 2

    def _is_reasoning_task(self, message: str) -> bool:
        return sum(1 for kw in self.reasoning_keywords if kw in message) >= 2

    def _is_creative_task(self, message: str) -> bool:
        return any(kw in message for kw in self.creative_keywords)
