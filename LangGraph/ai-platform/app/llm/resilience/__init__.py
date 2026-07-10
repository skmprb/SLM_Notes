"""
Resilience Package - Retry, Circuit Breaker, and Fallback for LLM calls.

Components:
    - RetryHandler: Exponential backoff with jitter
    - CircuitBreaker: Prevents hammering failing providers
    - FallbackManager: Orchestrates retry + circuit breaker + provider switching
    - Exceptions: Typed error hierarchy for intelligent error handling

Usage:
    from app.llm.resilience import FallbackManager, FallbackConfig, RetryConfig

    manager = FallbackManager(FallbackConfig(
        fallback_chain=["anthropic", "openai", "google"],
        retry_config=RetryConfig(max_retries=3, initial_delay=1.0),
        enable_circuit_breaker=True,
    ))

    result = manager.call(messages, config)
    print(f"Used: {result.provider_used}, Fallback: {result.was_fallback}")
"""

from app.llm.resilience.exceptions import (
    LLMError, RetryableError, NonRetryableError,
    RateLimitError, TimeoutError, ServerError, ConnectionError,
    AuthenticationError, InvalidRequestError, ContentPolicyError,
    QuotaExhaustedError, ProviderUnavailableError, ModelNotFoundError,
    CircuitOpenError, classify_error,
)
from app.llm.resilience.retry_handler import RetryHandler, RetryConfig, RetryResult
from app.llm.resilience.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry, CircuitState,
)
from app.llm.resilience.fallback_manager import FallbackManager, FallbackConfig, ResilientResponse

__all__ = [
    # Exceptions
    "LLMError", "RetryableError", "NonRetryableError",
    "RateLimitError", "TimeoutError", "ServerError", "ConnectionError",
    "AuthenticationError", "InvalidRequestError", "ContentPolicyError",
    "QuotaExhaustedError", "ProviderUnavailableError", "ModelNotFoundError",
    "CircuitOpenError", "classify_error",
    # Retry
    "RetryHandler", "RetryConfig", "RetryResult",
    # Circuit Breaker
    "CircuitBreaker", "CircuitBreakerConfig", "CircuitBreakerRegistry", "CircuitState",
    # Fallback
    "FallbackManager", "FallbackConfig", "ResilientResponse",
]
