"""
Circuit Breaker - Prevents calling a provider that's known to be failing.

The pattern:
    CLOSED (normal) → failures exceed threshold → OPEN (reject all calls)
    OPEN → after timeout → HALF-OPEN (allow one test call)
    HALF-OPEN → test succeeds → CLOSED
    HALF-OPEN → test fails → OPEN (reset timeout)

Why?
    - If OpenAI is down, don't waste 30s * 3 retries * 1000 requests = 90,000 seconds of user wait time
    - Fail fast, switch to fallback immediately
    - Give the failing provider time to recover
    - Reduce load on a struggling service (be a good citizen)

Real-world example:
    - Claude API returns 503 for 5 consecutive requests
    - Circuit opens → all Claude requests immediately fail with CircuitOpenError
    - After 60 seconds, one test request is allowed through
    - If it succeeds → circuit closes, Claude is back
    - If it fails → circuit stays open for another 60 seconds

LangChain equivalent:
    - No built-in circuit breaker
    - ModelFallbackMiddleware handles the "switch provider" part
    - But doesn't prevent hammering the failing provider
    - In production, you'd combine our circuit breaker WITH LangChain's fallback
"""

import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Rejecting all calls
    HALF_OPEN = "half_open" # Testing with one call


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5       # Failures before opening
    recovery_timeout: float = 60.0   # Seconds before trying again (OPEN → HALF_OPEN)
    success_threshold: int = 2       # Successes in HALF_OPEN before closing
    half_open_max_calls: int = 1     # Max concurrent calls in HALF_OPEN


class CircuitBreaker:
    """
    Circuit breaker for an individual provider.

    Usage:
        breaker = CircuitBreaker("openai")

        if breaker.can_execute():
            try:
                result = call_openai()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
        else:
            raise CircuitOpenError("openai circuit is open")
    """

    def __init__(self, provider_name: str, config: Optional[CircuitBreakerConfig] = None):
        self.provider_name = provider_name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()

    def can_execute(self) -> bool:
        """Check if a call is allowed through the circuit."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._recovery_timeout_elapsed():
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return True

        return False

    def record_success(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open → back to open
            self._transition_to(CircuitState.OPEN)

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        return {
            "provider": self.provider_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time,
            "time_in_state": time.time() - self.last_state_change,
        }

    def reset(self):
        """Manually reset the circuit breaker."""
        self._transition_to(CircuitState.CLOSED)
        self.failure_count = 0
        self.success_count = 0

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()

        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0

        logger.info(
            f"Circuit breaker | provider={self.provider_name} | "
            f"{old_state.value} → {new_state.value} | failures={self.failure_count}"
        )

    def _recovery_timeout_elapsed(self) -> bool:
        """Check if enough time has passed since last failure."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout


class CircuitBreakerRegistry:
    """
    Manages circuit breakers for all providers.

    Usage:
        registry = CircuitBreakerRegistry()
        breaker = registry.get("openai")
        if breaker.can_execute():
            ...
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, provider_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider."""
        if provider_name not in self._breakers:
            self._breakers[provider_name] = CircuitBreaker(provider_name, self.config)
        return self._breakers[provider_name]

    def get_all_status(self) -> list[dict]:
        """Get status of all circuit breakers."""
        return [breaker.get_status() for breaker in self._breakers.values()]

    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()
