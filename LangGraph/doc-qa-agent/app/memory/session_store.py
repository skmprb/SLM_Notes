"""
Session Store: Redis-backed per-user-session state.
Keys are namespaced by user_id to guarantee isolation.
"""
import json
import redis.asyncio as aioredis

from app.agents.state import DocPayload
from app.config.settings import get_settings

settings = get_settings()

_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _doc_key(user_id: str, session_id: str) -> str:
    return f"session:{user_id}:{session_id}:doc"


def _history_key(user_id: str, session_id: str) -> str:
    return f"session:{user_id}:{session_id}:history"


async def store_doc_payload(user_id: str, session_id: str, payload: DocPayload) -> None:
    r = get_redis()
    await r.setex(
        _doc_key(user_id, session_id),
        settings.session_ttl_seconds,
        json.dumps(payload),
    )


async def get_doc_payload(user_id: str, session_id: str) -> DocPayload | None:
    r = get_redis()
    raw = await r.get(_doc_key(user_id, session_id))
    if raw is None:
        return None
    return json.loads(raw)


async def delete_doc_payload(user_id: str, session_id: str) -> None:
    r = get_redis()
    await r.delete(_doc_key(user_id, session_id))


async def append_history(user_id: str, session_id: str, role: str, content: str) -> None:
    r = get_redis()
    key = _history_key(user_id, session_id)
    history = await get_history(user_id, session_id)
    history.append({"role": role, "content": content})
    # Keep last 20 turns to bound token usage
    history = history[-20:]
    await r.setex(key, settings.session_ttl_seconds, json.dumps(history))


async def get_history(user_id: str, session_id: str) -> list[dict]:
    r = get_redis()
    raw = await r.get(_history_key(user_id, session_id))
    return json.loads(raw) if raw else []


async def has_doc(user_id: str, session_id: str) -> bool:
    r = get_redis()
    return bool(await r.exists(_doc_key(user_id, session_id)))
