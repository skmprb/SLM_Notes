"""
Caching Layer (Phase 11) - Exact match + semantic cache for LLM responses.

Two cache strategies:
    1. Exact cache: Hash of (messages + model) → cached response
    2. Semantic cache: Embed query → find similar cached query → return if close enough

Production concerns:
    - TTL (time-to-live) for cache entries
    - Cache invalidation strategy
    - Max cache size (LRU eviction)
    - Cache hit/miss metrics
    - Separate caches per tenant (multi-tenancy)

LangChain equivalent:
    - InMemoryCache / SQLiteCache / RedisCache
    - GPTCache for semantic caching
    - set_llm_cache() global cache
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import OrderedDict

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CacheEntry:
    """A cached LLM response."""
    key: str
    response: dict  # The full LLM response dict
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0  # 1 hour default
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class BaseCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[dict]: ...

    @abstractmethod
    def set(self, key: str, response: dict, ttl: Optional[float] = None) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def stats(self) -> dict: ...


class InMemoryLRUCache(BaseCache):
    """LRU cache with TTL eviction."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[dict]:
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1
        logger.debug(f"Cache HIT: {key[:16]}...")
        return entry.response

    def set(self, key: str, response: dict, ttl: Optional[float] = None) -> None:
        # Evict LRU if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = CacheEntry(
            key=key, response=response, ttl_seconds=ttl or self._default_ttl
        )

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }


class ResponseCache:
    """
    High-level cache for LLM responses.

    Generates cache keys from messages + model config.
    Wraps the low-level cache with LLM-specific logic.
    """

    def __init__(self, cache: Optional[BaseCache] = None, enabled: bool = True):
        self._cache = cache or InMemoryLRUCache()
        self.enabled = enabled

    def get(self, messages: list[dict], model: str = "", **kwargs) -> Optional[dict]:
        """Look up cached response for these messages."""
        if not self.enabled:
            return None
        key = self._make_key(messages, model, **kwargs)
        return self._cache.get(key)

    def set(self, messages: list[dict], model: str, response: dict, ttl: Optional[float] = None) -> None:
        """Cache a response."""
        if not self.enabled:
            return
        key = self._make_key(messages, model)
        self._cache.set(key, response, ttl)
        logger.debug(f"Cache SET: {key[:16]}... | model={model}")

    def stats(self) -> dict:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()

    def _make_key(self, messages: list[dict], model: str = "", **kwargs) -> str:
        """Generate deterministic cache key from messages + config."""
        payload = json.dumps({"messages": messages, "model": model, **kwargs}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


# Global cache instance
_response_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache
