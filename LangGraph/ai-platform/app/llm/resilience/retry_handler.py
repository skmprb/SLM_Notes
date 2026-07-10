"""
Retry Handler - Exponential backoff with jitter.

Why not just use tenacity?
    - Understanding the pattern is more important than the library
    - We need custom logic (different backoff for rate limits vs timeouts)
    - We need integration with our error hierarchy
    - In production, you might use tenacity AND this (tenacity for decoration, this for logic)

Key concepts:
    - Exponential backoff: 1s → 2s → 4s → 8s (doubles each time)
    - Jitter: Add randomness to avoid thundering herd
    - Max retries: Don't retry forever
    - Retry budget: Don't retry non-retryable errors
    - Respect Retry-After: If provider says "wait 30s", wait 30s

LangChain equivalent:
    - Built-in: max_retries=6 on all chat models
    - Middleware: ModelRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0)
    - LangGraph: RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0)
"""

import time
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.llm.resilience.exceptions import (
    LLMError, RetryableError, NonRetryableError,
    RateLimitError, TimeoutError, ServerError,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0       # Seconds before first retry
    backoff_factor: float = 2.0      # Multiplier each retry
    max_delay: float = 60.0          # Cap on delay
    jitter: bool = True              # Add randomness
    jitter_range: float = 0.25       # ±25% jitter


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: any = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_delay: float = 0.0
    errors: list[Exception] = field(default_factory=list)


class RetryHandler:
    """
    Handles retry logic with exponential backoff and jitter.

    Usage:
        handler = RetryHandler(RetryConfig(max_retries=3))
        result = handler.execute(lambda: llm.generate(messages))

        if result.success:
            print(result.result)
        else:
            print(f"Failed after {result.attempts} attempts: {result.error}")
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def execute(self, operation: Callable, operation_name: str = "LLM call") -> RetryResult:
        """
        Execute an operation with retry logic.

        Args:
            operation: Callable that performs the LLM call
            operation_name: For logging

        Returns:
            RetryResult with success/failure info
        """
        errors = []
        total_delay = 0.0

        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                result = operation()
                if attempt > 0:
                    logger.info(f"Retry succeeded | op={operation_name} | attempt={attempt + 1}")
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_delay=total_delay,
                    errors=errors,
                )

            except NonRetryableError as e:
                # Don't retry these - fail immediately
                logger.warning(f"Non-retryable error | op={operation_name} | error={type(e).__name__}: {e}")
                return RetryResult(
                    success=False,
                    error=e,
                    attempts=attempt + 1,
                    total_delay=total_delay,
                    errors=[e],
                )

            except (RetryableError, Exception) as e:
                errors.append(e)

                if attempt >= self.config.max_retries:
                    # Exhausted all retries
                    logger.error(
                        f"All retries exhausted | op={operation_name} | "
                        f"attempts={attempt + 1} | last_error={type(e).__name__}: {e}"
                    )
                    return RetryResult(
                        success=False,
                        error=e,
                        attempts=attempt + 1,
                        total_delay=total_delay,
                        errors=errors,
                    )

                # Calculate delay
                delay = self._calculate_delay(attempt, e)
                total_delay += delay

                logger.warning(
                    f"Retrying | op={operation_name} | attempt={attempt + 1}/{self.config.max_retries + 1} | "
                    f"error={type(e).__name__} | delay={delay:.2f}s"
                )

                time.sleep(delay)

        # Should never reach here, but just in case
        return RetryResult(success=False, error=errors[-1] if errors else None, attempts=self.config.max_retries + 1)

    def _calculate_delay(self, attempt: int, error: Exception) -> float:
        """
        Calculate delay before next retry.

        Special cases:
            - RateLimitError with retry_after: use that value
            - Normal: exponential backoff with jitter
        """
        # Respect Retry-After header from rate limit responses
        if isinstance(error, RateLimitError) and error.retry_after:
            return min(error.retry_after, self.config.max_delay)

        # Exponential backoff: initial_delay * (backoff_factor ^ attempt)
        delay = self.config.initial_delay * (self.config.backoff_factor ** attempt)

        # Cap at max_delay
        delay = min(delay, self.config.max_delay)

        # Add jitter to avoid thundering herd
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Never negative or too small

        return delay
