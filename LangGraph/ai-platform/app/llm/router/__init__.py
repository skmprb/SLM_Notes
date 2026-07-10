"""
Model Router Package

Three routing strategies available:
    - RuleBasedRouter: Fast, deterministic, zero-cost (default)
    - CostAwareRouter: Optimizes for budget
    - LLMRouter: Most accurate, uses classifier LLM

Usage:
    from app.llm.router import create_router, RoutingContext

    router = create_router("rule_based")
    context = RoutingContext(message="Write a Python function to sort a list")
    decision = router.route(context)
    # → provider="anthropic", model="claude-3-5-sonnet", reason="Coding task"
"""

from app.llm.router.base import BaseRouter, RoutingContext, RoutingDecision, TaskType
from app.llm.router.rule_based import RuleBasedRouter
from app.llm.router.cost_aware import CostAwareRouter
from app.llm.router.llm_router import LLMRouter

_ROUTER_REGISTRY = {
    "rule_based": RuleBasedRouter,
    "cost_aware": CostAwareRouter,
    "llm": LLMRouter,
}


def create_router(strategy: str = "rule_based", **kwargs) -> BaseRouter:
    """Factory to create a router by strategy name."""
    if strategy not in _ROUTER_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(_ROUTER_REGISTRY.keys())}")
    return _ROUTER_REGISTRY[strategy](**kwargs)


__all__ = [
    "BaseRouter", "RoutingContext", "RoutingDecision", "TaskType",
    "RuleBasedRouter", "CostAwareRouter", "LLMRouter",
    "create_router",
]
