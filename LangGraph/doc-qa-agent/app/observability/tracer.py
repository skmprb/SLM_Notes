"""
Observability: structured logging + LangSmith tracing + token/latency tracking.
Every agent node calls track_node() to emit a structured event.
"""
import time
import logging
import json
import os
from contextlib import contextmanager
from functools import wraps

from app.config.settings import get_settings

settings = get_settings()

# Configure LangSmith if enabled
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project


# Structured JSON logger
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log.update(record.extra)
        return json.dumps(log)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    return logger


agent_logger = get_logger("agent")


def track_node(node_name: str):
    """
    Decorator for agent nodes.
    Logs: node_name, user_id, session_id, latency_ms, tokens (if available).
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(state: dict, *args, **kwargs):
            start = time.perf_counter()
            result = await fn(state, *args, **kwargs)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            agent_logger.info(
                f"node:{node_name}",
                extra={
                    "extra": {
                        "node": node_name,
                        "user_id": state.get("user_id"),
                        "session_id": state.get("session_id"),
                        "doc_id": state.get("doc_id"),
                        "retrieval_path": state.get("retrieval_path", ""),
                        "retry_count": state.get("retry_count", 0),
                        "latency_ms": latency_ms,
                        "grading_score": result.get("grading_score", "") if result else "",
                    }
                },
            )
            return result
        return wrapper
    return decorator


@contextmanager
def trace_span(name: str, metadata: dict | None = None):
    """Lightweight context manager for timing arbitrary spans."""
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = round((time.perf_counter() - start) * 1000, 2)
        agent_logger.info(
            f"span:{name}",
            extra={"extra": {"span": name, "latency_ms": ms, **(metadata or {})}},
        )
