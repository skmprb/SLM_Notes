"""
Cost-Aware Router - Picks the cheapest model that meets the task requirements.

Production use case:
    - You have a budget of $X/day
    - Simple questions should use the cheapest model
    - Only escalate to expensive models when needed
    - Track spend and route to cheaper models when approaching budget

This is how companies like Stripe, Notion, and Intercom manage LLM costs.
They don't use GPT-4o for everything — they use tiered routing.

Model tiers (approximate cost per 1M tokens):
    Tier 1 (Cheapest):  Gemini Flash ($0.075), GPT-4o-mini ($0.15)
    Tier 2 (Mid):       Claude Haiku ($0.80), Gemini Pro ($1.25)
    Tier 3 (Premium):   GPT-4o ($2.50), Claude Sonnet ($3.00)
"""

from dataclasses import dataclass
from app.llm.router.base import BaseRouter, RoutingContext, RoutingDecision, TaskType
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ModelTier:
    """A model with its capabilities and cost."""
    provider: str
    model: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    max_context: int
    supports_vision: bool = False
    supports_code: bool = True
    quality_score: int = 5  # 1-10, higher = better quality


# Model registry sorted by cost (cheapest first)
MODEL_TIERS = [
    ModelTier("google", "gemini-1.5-flash", 0.075, 0.30, 1_000_000, True, True, 6),
    ModelTier("openai", "gpt-4o-mini", 0.15, 0.60, 128_000, True, True, 7),
    ModelTier("anthropic", "claude-3-5-haiku-20241022", 0.80, 4.00, 200_000, False, True, 7),
    ModelTier("google", "gemini-1.5-pro", 1.25, 5.00, 1_000_000, True, True, 8),
    ModelTier("openai", "gpt-4o", 2.50, 10.00, 128_000, True, True, 9),
    ModelTier("anthropic", "claude-3-5-sonnet-20241022", 3.00, 15.00, 200_000, False, True, 9),
]


class CostAwareRouter(BaseRouter):
    """
    Routes to the cheapest model that satisfies the request requirements.

    Logic:
        1. Determine minimum quality needed for the task
        2. Filter models that meet the requirements
        3. Pick the cheapest one
    """

    def __init__(self, max_cost_per_request: float | None = None):
        self.max_cost_per_request = max_cost_per_request  # Budget cap

    def route(self, context: RoutingContext) -> RoutingDecision:
        min_quality = self._determine_min_quality(context)
        candidates = self._filter_candidates(context, min_quality)

        if not candidates:
            # Fallback to cheapest available
            candidates = MODEL_TIERS

        # Pick cheapest from candidates
        best = candidates[0]

        estimated_cost = self._estimate_cost(best, context.estimated_tokens)

        decision = RoutingDecision(
            provider=best.provider,
            model=best.model,
            reason=f"Cheapest model meeting quality={min_quality} | est_cost=${estimated_cost:.4f}",
            estimated_cost=estimated_cost,
        )

        logger.info(
            f"CostRouter | quality_needed={min_quality} | "
            f"candidates={len(candidates)} | selected={best.model} | cost=${estimated_cost:.4f}"
        )
        return decision

    def _determine_min_quality(self, context: RoutingContext) -> int:
        """Determine minimum quality score needed."""
        if context.task_type in (TaskType.REASONING, TaskType.CODING):
            return 8
        if context.task_type in (TaskType.CREATIVE, TaskType.LONG_CONTEXT):
            return 7
        if context.priority == "high":
            return 8
        if context.priority == "low":
            return 5
        return 6  # Default: mid-tier is fine

    def _filter_candidates(self, context: RoutingContext, min_quality: int) -> list[ModelTier]:
        """Filter models that meet all requirements."""
        candidates = []
        for model in MODEL_TIERS:
            if model.quality_score < min_quality:
                continue
            if context.has_images and not model.supports_vision:
                continue
            if context.estimated_tokens > model.max_context:
                continue
            if self.max_cost_per_request:
                est_cost = self._estimate_cost(model, context.estimated_tokens)
                if est_cost > self.max_cost_per_request:
                    continue
            candidates.append(model)
        return candidates

    def _estimate_cost(self, model: ModelTier, estimated_tokens: int) -> float:
        """Estimate cost for this request."""
        tokens = max(estimated_tokens, 500)  # Minimum estimate
        input_cost = (tokens / 1_000_000) * model.cost_per_1m_input
        output_cost = (tokens / 1_000_000) * model.cost_per_1m_output
        return input_cost + output_cost
