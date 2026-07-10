"""
StreamingService - Handles streaming LLM responses via Server-Sent Events (SSE).

SSE Protocol:
    - Client opens a long-lived HTTP connection
    - Server pushes events as they arrive
    - Each event: "data: {json}\n\n"
    - Final event: "data: [DONE]\n\n"

This is the same protocol used by:
    - OpenAI's /chat/completions (stream=true)
    - ChatGPT's web interface
    - Claude's web interface
    - Every modern AI chat application

Why SSE over WebSocket?
    - Simpler (HTTP, no upgrade needed)
    - Works through proxies/CDNs
    - Auto-reconnect built into browser EventSource API
    - Sufficient for server→client streaming (chat is one-directional per message)
    - WebSocket is better for bidirectional (real-time collaboration, gaming)

LangChain equivalent:
    - model.stream("Hello") → yields AIMessageChunk objects
    - In FastAPI: StreamingResponse with model.stream()
"""

import json
import time
from typing import Generator, Optional

from app.llm.base import BaseLLM, LLMConfig, StreamChunk
from app.llm.factory import create_llm
from app.config.settings import Settings, get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class StreamingService:
    """
    Handles streaming LLM responses formatted as SSE events.

    Usage:
        service = StreamingService()
        for sse_event in service.stream_chat(messages, provider="openai"):
            yield sse_event  # Send to client
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def stream_chat(
        self,
        messages: list[dict],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a chat response as SSE-formatted events.

        Yields strings in SSE format:
            data: {"type": "token", "content": "Hello"}

            data: {"type": "token", "content": " world"}

            data: {"type": "done", "model": "gpt-4o-mini", "tokens": 42}

            data: [DONE]

        Args:
            messages: Standard message format
            provider: Which provider to use
            model: Which model to use
        """
        provider_name = provider or self.settings.default_provider
        llm: BaseLLM = create_llm(provider_name)

        config = LLMConfig(
            model=model or "",
            temperature=self.settings.openai_temperature,
            max_tokens=self.settings.openai_max_tokens,
        )

        start_time = time.time()
        total_content = ""
        input_tokens = None
        output_tokens = None

        try:
            # Yield start event
            yield self._format_sse({
                "type": "start",
                "provider": provider_name,
                "model": config.model or llm.provider_name,
            })

            # Stream tokens
            for chunk in llm.stream(messages, config):
                if chunk.content:
                    total_content += chunk.content
                    yield self._format_sse({
                        "type": "token",
                        "content": chunk.content,
                    })

                # Capture usage from final chunk
                if chunk.input_tokens:
                    input_tokens = chunk.input_tokens
                if chunk.output_tokens:
                    output_tokens = chunk.output_tokens

            # Yield completion event with metadata
            latency_ms = (time.time() - start_time) * 1000
            yield self._format_sse({
                "type": "done",
                "model": chunk.model if chunk else config.model,
                "provider": provider_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": (input_tokens or 0) + (output_tokens or 0) or None,
                "latency_ms": round(latency_ms, 2),
                "content_length": len(total_content),
            })

            # SSE termination signal
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error | provider={provider_name} | error={e}")
            yield self._format_sse({
                "type": "error",
                "error": str(e),
                "provider": provider_name,
            })
            yield "data: [DONE]\n\n"

    def _format_sse(self, data: dict) -> str:
        """Format a dict as an SSE event string."""
        return f"data: {json.dumps(data)}\n\n"
