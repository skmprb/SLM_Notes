"""
LLMService - Phase 4: Now with resilience (retry + fallback + circuit breaker).

Architecture:
    ChatService → LLMService → FallbackManager
                                    ├── CircuitBreaker (per provider)
                                    ├── RetryHandler (per attempt)
                                    └── Provider chain [primary, fallback1, fallback2]

Two modes:
    1. Direct mode: Use a specific provider (no fallback)
    2. Resilient mode: Use FallbackManager with full retry/fallback chain

The caller decides which mode based on whether they specify a provider.
"""

from typing import Optional

from app.llm.base import BaseLLM, LLMConfig, LLMResponse
from app.llm.factory import create_llm
from app.llm.resilience import FallbackManager, FallbackConfig, RetryConfig, ResilientResponse
from app.config.settings import Settings, get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMService:
    """
    Service layer for LLM interactions with built-in resilience.

    Usage:
        # Resilient mode (default) - uses fallback chain
        service = LLMService()
        result = service.generate(messages)

        # Direct mode - specific provider, no fallback
        service = LLMService(provider=create_llm("anthropic"))
        result = service.generate(messages)
    """

    def __init__(
        self,
        provider: Optional[BaseLLM] = None,
        settings: Optional[Settings] = None,
        fallback_config: Optional[FallbackConfig] = None,
    ):
        self.settings = settings or get_settings()

        if provider:
            # Direct mode: use this specific provider
            self.provider = provider
            self.fallback_manager = None
            logger.info(f"LLMService (direct) | provider={provider.provider_name}")
        else:
            # Resilient mode: use FallbackManager
            self.provider = None
            config = fallback_config or FallbackConfig(
                fallback_chain=self.settings.fallback_chain.split(","),
                retry_config=RetryConfig(
                    max_retries=self.settings.max_retries,
                    initial_delay=self.settings.retry_initial_delay,
                    backoff_factor=self.settings.retry_backoff_factor,
                ),
                enable_circuit_breaker=self.settings.enable_circuit_breaker,
            )
            self.fallback_manager = FallbackManager(config)
            logger.info(f"LLMService (resilient) | chain={config.fallback_chain}")

    def generate(self, messages: list[dict], **kwargs) -> dict:
        """
        Generate a response with resilience.

        Returns dict with extra fields:
            - was_fallback: bool
            - attempts: int
            - fallback_reason: str | None
        """
        config = LLMConfig(
            model=kwargs.get("model", ""),
            temperature=kwargs.get("temperature", self.settings.openai_temperature),
            max_tokens=kwargs.get("max_tokens", self.settings.openai_max_tokens),
        )

        if self.provider:
            # Direct mode: call provider directly (no resilience)
            response: LLMResponse = self.provider.generate(messages, config)
            return self._format_response(response, was_fallback=False, attempts=1)
        else:
            # Resilient mode: use FallbackManager
            resilient: ResilientResponse = self.fallback_manager.call(messages, config)
            return self._format_response(
                resilient.response,
                was_fallback=resilient.was_fallback,
                attempts=resilient.total_attempts,
                fallback_reason=resilient.fallback_reason,
            )

    def get_circuit_status(self) -> list[dict]:
        """Get circuit breaker status (only in resilient mode)."""
        if self.fallback_manager:
            return self.fallback_manager.get_circuit_status()
        return []

    def _format_response(
        self,
        response: LLMResponse,
        was_fallback: bool = False,
        attempts: int = 1,
        fallback_reason: str | None = None,
    ) -> dict:
        return {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "tokens_used": response.tokens_used,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "was_fallback": was_fallback,
            "attempts": attempts,
            "fallback_reason": fallback_reason,
        }
