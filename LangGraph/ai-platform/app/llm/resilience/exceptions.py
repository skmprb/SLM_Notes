"""
Custom Exception Hierarchy for LLM operations.

Why a hierarchy?
    - Different errors need different handling
    - Retry logic needs to know: is this transient or permanent?
    - Fallback logic needs to know: is this provider-specific or global?
    - Monitoring needs to categorize errors

Rule of thumb:
    - Retryable: Network issues, rate limits, server errors (5xx)
    - Non-retryable: Auth errors, invalid input, content policy violations
    - Fallback-worthy: Provider down, model deprecated, quota exhausted
"""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    def __init__(self, message: str, provider: str = "", model: str = "", status_code: int | None = None):
        self.provider = provider
        self.model = model
        self.status_code = status_code
        super().__init__(message)


# ============================================================
# RETRYABLE ERRORS (transient, will likely succeed on retry)
# ============================================================

class RetryableError(LLMError):
    """Base for errors that should be retried."""
    pass


class RateLimitError(RetryableError):
    """429 - Too many requests. Retry after backoff."""

    def __init__(self, message: str, retry_after: float | None = None, **kwargs):
        self.retry_after = retry_after  # Seconds to wait (from Retry-After header)
        super().__init__(message, **kwargs)


class TimeoutError(RetryableError):
    """Request timed out. Retry with same or different provider."""
    pass


class ServerError(RetryableError):
    """5xx from provider. Their problem, retry."""
    pass


class ConnectionError(RetryableError):
    """Network connectivity issue. Retry."""
    pass


# ============================================================
# NON-RETRYABLE ERRORS (permanent, don't waste retries)
# ============================================================

class NonRetryableError(LLMError):
    """Base for errors that should NOT be retried."""
    pass


class AuthenticationError(NonRetryableError):
    """401/403 - Invalid API key or permissions."""
    pass


class InvalidRequestError(NonRetryableError):
    """400 - Bad request (invalid model, too many tokens, etc.)."""
    pass


class ContentPolicyError(NonRetryableError):
    """Content was blocked by safety filters."""
    pass


class QuotaExhaustedError(NonRetryableError):
    """Account quota/budget exhausted. Need human intervention."""
    pass


# ============================================================
# FALLBACK-WORTHY ERRORS (try a different provider)
# ============================================================

class ProviderUnavailableError(LLMError):
    """Provider is completely down. Switch to fallback."""
    pass


class ModelNotFoundError(LLMError):
    """Model doesn't exist or was deprecated."""
    pass


class CircuitOpenError(LLMError):
    """Circuit breaker is open - provider is known to be failing."""
    pass


def classify_error(error: Exception, provider: str = "", model: str = "") -> LLMError:
    """
    Classify a raw exception into our hierarchy.

    This is called by providers to translate SDK-specific errors
    into our standard error types.
    """
    error_str = str(error).lower()
    status_code = getattr(error, "status_code", None)

    # Rate limit
    if status_code == 429 or "rate limit" in error_str or "too many requests" in error_str:
        retry_after = getattr(error, "retry_after", None)
        return RateLimitError(str(error), retry_after=retry_after, provider=provider, model=model, status_code=429)

    # Auth
    if status_code in (401, 403) or "unauthorized" in error_str or "invalid api key" in error_str:
        return AuthenticationError(str(error), provider=provider, model=model, status_code=status_code)

    # Server error
    if status_code and 500 <= status_code < 600:
        return ServerError(str(error), provider=provider, model=model, status_code=status_code)

    # Timeout
    if "timeout" in error_str or "timed out" in error_str:
        return TimeoutError(str(error), provider=provider, model=model)

    # Connection
    if "connection" in error_str or "network" in error_str:
        return ConnectionError(str(error), provider=provider, model=model)

    # Content policy
    if "content policy" in error_str or "safety" in error_str or "blocked" in error_str:
        return ContentPolicyError(str(error), provider=provider, model=model)

    # Invalid request
    if status_code == 400 or "invalid" in error_str:
        return InvalidRequestError(str(error), provider=provider, model=model, status_code=400)

    # Default: treat as retryable (optimistic)
    return RetryableError(str(error), provider=provider, model=model, status_code=status_code)
