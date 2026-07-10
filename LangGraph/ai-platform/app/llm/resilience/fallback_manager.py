"""
Fallback Manager - Tries alternative providers when the primary fails.

Fallback chain:
    Primary (Claude) → Fallback 1 (GPT-4o) → Fallback 2 (Gemini) → Cached → Error

Why?
    - No single provider has 100% uptime
    - Users don't care which model answered — they care that they GOT an answer
    - Different failure modes need different fallbacks:
        * Provider down → switch provider
        * Rate limited → switch provider OR wait
        * All providers down → serve cached response
        * Content blocked → can't fallback (same content will be blocked everywhere)

LangChain equivalent:
    - ModelFallbackMiddleware("gpt-4o-mini", "claude-3-5-sonnet")
    - model.with_fallbacks([fallback_model1, fallback_model2])
    - Both try alternatives in order when primary fails
"""

from dataclasses import dataclass, field
from typing import Optional

from app.llm.base import BaseLLM, LLMConfig, LLMResponse
from app.llm.factory import create_llm
from app.llm.resilience.exceptions import (
    NonRetryableError, ContentPolicyError, AuthenticationError,
    CircuitOpenError, ProviderUnavailableError, classify_error,
)
from app.llm.resilience.retry_handler import RetryHandler, RetryConfig, RetryResult
from app.llm.resilience.circuit_breaker import CircuitBreakerRegistry
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    fallback_chain: list[str] = field(default_factory=lambda: ["openai", "anthropic", "google"])
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    enable_circuit_breaker: bool = True


@dataclass
class ResilientResponse:
    """Response from the resilient LLM layer."""
    response: LLMResponse
    provider_used: str
    was_fallback: bool = False
    attempts_on_primary: int = 0
    total_attempts: int = 0
    fallback_reason: Optional[str] = None


class FallbackManager:
    """
    Orchestrates retry + circuit breaker + fallback for LLM calls.

    This is the TOP-LEVEL resilience layer. It combines:
        1. Circuit breaker (skip known-failing providers)
        2. Retry (retry transient errors on same provider)
        3. Fallback (switch to next provider in chain)

    Architecture:
        FallbackManager
            ├── CircuitBreakerRegistry (per-provider health tracking)
            ├── RetryHandler (per-attempt retry logic)
            └── Provider chain [openai, anthropic, google]

    Usage:
        manager = FallbackManager(FallbackConfig(
            fallback_chain=["anthropic", "openai", "google"]
        ))
        result = manager.call(messages, config)
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.circuit_breakers = CircuitBreakerRegistry()
        self.retry_handler = RetryHandler(self.config.retry_config)
        self._providers: dict[str, BaseLLM] = {}

    def call(self, messages: list[dict], config: Optional[LLMConfig] = None) -> ResilientResponse:
        """
        Make a resilient LLM call with retry and fallback.

        Flow:
            1. Try primary provider (with retries)
            2. If primary fails → check circuit breaker → try next in chain
            3. Continue until success or all providers exhausted
        """
        total_attempts = 0
        primary_attempts = 0
        last_error = None

        for i, provider_name in enumerate(self.config.fallback_chain):
            is_primary = (i == 0)

            # Check circuit breaker
            if self.config.enable_circuit_breaker:
                breaker = self.circuit_breakers.get(provider_name)
                if not breaker.can_execute():
                    logger.info(f"Circuit open for {provider_name}, skipping")
                    continue

            # Get or create provider
            provider = self._get_provider(provider_name)

            # Try with retries
            def make_call():
                return provider.generate(messages, config)

            retry_result: RetryResult = self.retry_handler.execute(
                make_call,
                operation_name=f"{provider_name}.generate",
            )

            total_attempts += retry_result.attempts
            if is_primary:
                primary_attempts = retry_result.attempts

            if retry_result.success:
                # Record success for circuit breaker
                if self.config.enable_circuit_breaker:
                    self.circuit_breakers.get(provider_name).record_success()

                return ResilientResponse(
                    response=retry_result.result,
                    provider_used=provider_name,
                    was_fallback=not is_primary,
                    attempts_on_primary=primary_attempts,
                    total_attempts=total_attempts,
                    fallback_reason=f"Primary failed, used {provider_name}" if not is_primary else None,
                )

            # Record failure for circuit breaker
            if self.config.enable_circuit_breaker:
                self.circuit_breakers.get(provider_name).record_failure()

            last_error = retry_result.error

            # Don't fallback for content policy errors (same content will fail everywhere)
            if isinstance(last_error, ContentPolicyError):
                logger.warning("Content policy error - no fallback possible")
                break

            # Don't fallback for auth errors on all providers
            if isinstance(last_error, AuthenticationError):
                logger.warning(f"Auth error on {provider_name} - trying next provider")

            logger.info(f"Provider {provider_name} failed, trying next in fallback chain")

        # All providers exhausted
        raise ProviderUnavailableError(
            f"All providers in fallback chain failed. Last error: {last_error}",
            provider="all",
        )

    def get_circuit_status(self) -> list[dict]:
        """Get circuit breaker status for all providers."""
        return self.circuit_breakers.get_all_status()

    def reset_circuits(self):
        """Reset all circuit breakers (manual recovery)."""
        self.circuit_breakers.reset_all()

    def _get_provider(self, provider_name: str) -> BaseLLM:
        """Lazy-load providers."""
        if provider_name not in self._providers:
            self._providers[provider_name] = create_llm(provider_name)
        return self._providers[provider_name]
