"""
Anthropic Provider - Phase 5: Added streaming support.

Anthropic streaming uses message_stream context manager.
Events: message_start, content_block_delta, message_stop
"""

import time
from typing import Optional, Generator

from app.llm.base import BaseLLM, LLMConfig, LLMResponse, StreamChunk
from app.config.settings import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AnthropicProvider(BaseLLM):
    """Anthropic/Claude implementation with streaming."""

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.default_model = default_model or settings.anthropic_model

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"AnthropicProvider initialized | model={self.default_model}")
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def generate(self, messages: list[dict], config: Optional[LLMConfig] = None) -> LLMResponse:
        model = config.model if config and config.model else self.default_model
        temperature = config.temperature if config else 0.7
        max_tokens = config.max_tokens if config else 1024

        system_message, chat_messages = self._split_messages(messages)

        start_time = time.time()

        response = self.client.messages.create(
            model=model,
            system=system_message,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider=self.provider_name,
            tokens_used=(response.usage.input_tokens + response.usage.output_tokens),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=round(latency_ms, 2),
        )

    def stream(self, messages: list[dict], config: Optional[LLMConfig] = None) -> Generator[StreamChunk, None, None]:
        """
        Stream tokens from Anthropic.

        Anthropic streaming uses events:
            - message_start: Contains input token count
            - content_block_delta: Contains text tokens
            - message_delta: Contains output token count + stop reason
            - message_stop: Stream complete
        """
        model = config.model if config and config.model else self.default_model
        temperature = config.temperature if config else 0.7
        max_tokens = config.max_tokens if config else 1024

        system_message, chat_messages = self._split_messages(messages)

        with self.client.messages.stream(
            model=model,
            system=system_message,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield StreamChunk(
                    content=text,
                    model=model,
                    provider=self.provider_name,
                    finish_reason=None,
                )

            # Final chunk with usage
            final_message = stream.get_final_message()
            yield StreamChunk(
                content="",
                model=model,
                provider=self.provider_name,
                finish_reason="stop",
                input_tokens=final_message.usage.input_tokens,
                output_tokens=final_message.usage.output_tokens,
            )

    def _split_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Split system message from chat messages (Anthropic requirement)."""
        system_message = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)
        return system_message, chat_messages
