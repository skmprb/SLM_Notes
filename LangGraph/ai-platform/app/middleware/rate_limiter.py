"""
Rate Limiting & Multi-tenancy (Phase 14)

Rate limiting strategies:
    - Token bucket (smooth, allows bursts)
    - Sliding window (strict per-minute/hour limits)
    - Per-tenant limits (different tiers get different rates)

Multi-tenancy:
    - Tenant isolation (separate configs, budgets, rate limits)
    - Tenant context propagation through the request lifecycle
    - Per-tenant model/tool access control

LangChain equivalent:
    - No built-in rate limiting (handled at infra level)
    - LangGraph Platform has per-assistant rate limits
    - Custom middleware pattern
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================
# RATE LIMITER (Token Bucket)
# ============================================================

@dataclass
class TokenBucket:
    """Token bucket rate limiter for a single key."""
    capacity: float          # Max tokens (burst size)
    refill_rate: float       # Tokens added per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self):
        self.tokens = self.capacity

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


class RateLimiter:
    """Per-key rate limiter using token bucket algorithm."""

    def __init__(self, default_rpm: int = 60, default_burst: int = 10):
        self._buckets: dict[str, TokenBucket] = {}
        self._default_rpm = default_rpm
        self._default_burst = default_burst

    def is_allowed(self, key: str, tokens: int = 1) -> bool:
        """Check if request is allowed for this key."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=self._default_burst,
                refill_rate=self._default_rpm / 60.0,
            )
        allowed = self._buckets[key].consume(tokens)
        if not allowed:
            logger.warning(f"Rate limited: {key}")
        return allowed

    def configure_key(self, key: str, rpm: int, burst: int) -> None:
        """Set custom rate limits for a specific key."""
        self._buckets[key] = TokenBucket(capacity=burst, refill_rate=rpm / 60.0)


# ============================================================
# MULTI-TENANCY
# ============================================================

@dataclass
class TenantConfig:
    """Configuration for a single tenant."""
    tenant_id: str
    name: str
    tier: str = "free"  # free | pro | enterprise
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10
    max_tokens_per_request: int = 4096
    allowed_providers: list[str] = field(default_factory=lambda: ["openai"])
    allowed_models: list[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    monthly_budget_usd: float = 10.0
    tools_enabled: bool = True


# Default tier configurations
TIER_DEFAULTS = {
    "free": TenantConfig(
        tenant_id="", name="", tier="free",
        rate_limit_rpm=20, rate_limit_burst=5,
        max_tokens_per_request=2048,
        allowed_providers=["openai"],
        allowed_models=["gpt-4o-mini"],
        monthly_budget_usd=5.0,
        tools_enabled=False,
    ),
    "pro": TenantConfig(
        tenant_id="", name="", tier="pro",
        rate_limit_rpm=120, rate_limit_burst=20,
        max_tokens_per_request=8192,
        allowed_providers=["openai", "anthropic"],
        allowed_models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"],
        monthly_budget_usd=100.0,
        tools_enabled=True,
    ),
    "enterprise": TenantConfig(
        tenant_id="", name="", tier="enterprise",
        rate_limit_rpm=600, rate_limit_burst=50,
        max_tokens_per_request=32768,
        allowed_providers=["openai", "anthropic", "google"],
        allowed_models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"],
        monthly_budget_usd=1000.0,
        tools_enabled=True,
    ),
}


class TenantManager:
    """Manages tenant configurations and access control."""

    def __init__(self):
        self._tenants: dict[str, TenantConfig] = {}
        self._rate_limiter = RateLimiter()
        self._usage: dict[str, float] = defaultdict(float)  # tenant_id → cost spent

    def register_tenant(self, tenant_id: str, name: str, tier: str = "free") -> TenantConfig:
        """Register a new tenant with tier-based defaults."""
        defaults = TIER_DEFAULTS.get(tier, TIER_DEFAULTS["free"])
        config = TenantConfig(
            tenant_id=tenant_id, name=name, tier=tier,
            rate_limit_rpm=defaults.rate_limit_rpm,
            rate_limit_burst=defaults.rate_limit_burst,
            max_tokens_per_request=defaults.max_tokens_per_request,
            allowed_providers=defaults.allowed_providers.copy(),
            allowed_models=defaults.allowed_models.copy(),
            monthly_budget_usd=defaults.monthly_budget_usd,
            tools_enabled=defaults.tools_enabled,
        )
        self._tenants[tenant_id] = config
        self._rate_limiter.configure_key(tenant_id, config.rate_limit_rpm, config.rate_limit_burst)
        logger.info(f"Tenant registered: {tenant_id} | tier={tier}")
        return config

    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        return self._tenants.get(tenant_id)

    def check_access(self, tenant_id: str, provider: str = "", model: str = "") -> tuple[bool, str]:
        """Check if tenant can make this request. Returns (allowed, reason)."""
        config = self._tenants.get(tenant_id)
        if not config:
            return False, "Unknown tenant"

        # Rate limit check
        if not self._rate_limiter.is_allowed(tenant_id):
            return False, "Rate limit exceeded"

        # Budget check
        if self._usage[tenant_id] >= config.monthly_budget_usd:
            return False, "Monthly budget exceeded"

        # Provider check
        if provider and provider not in config.allowed_providers:
            return False, f"Provider '{provider}' not allowed for tier '{config.tier}'"

        # Model check
        if model and model not in config.allowed_models:
            return False, f"Model '{model}' not allowed for tier '{config.tier}'"

        return True, "allowed"

    def record_usage(self, tenant_id: str, cost: float) -> None:
        self._usage[tenant_id] += cost

    def get_usage(self, tenant_id: str) -> dict:
        config = self._tenants.get(tenant_id)
        spent = self._usage.get(tenant_id, 0.0)
        return {
            "tenant_id": tenant_id,
            "tier": config.tier if config else "unknown",
            "spent_usd": round(spent, 4),
            "budget_usd": config.monthly_budget_usd if config else 0,
            "remaining_usd": round((config.monthly_budget_usd - spent), 4) if config else 0,
        }


_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager
