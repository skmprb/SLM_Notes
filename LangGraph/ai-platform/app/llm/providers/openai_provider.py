"""
OpenAI Provider - Phase 7: Added tool calling support.

OpenAI tool calling:
    - Pass tools=[{"type": "function", "function": {...}}] to the API
    - Response may have message.tool_calls instead of message.content
    - Each tool_call has: id, function.name, function.arguments (JSON string)
"""

import time
from typing import Optional, Generator
from openai import OpenAI

import json
from app.llm.base import BaseLLM, LLMConfig, LLMResponse, StreamChunk, ToolCallResponse
from app.config.settings import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIProvider(BaseLLM):
    """OpenAI implementation with streaming."""

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.default_model = default_model or settings.openai_model
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"OpenAIProvider initialized | model={self.default_model}")

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(self, messages: list[dict], config: Optional[LLMConfig] = None) -> LLMResponse:
        model = config.model if config and config.model else self.default_model
        temperature = config.temperature if config else 0.7
        max_tokens = config.max_tokens if config else 1024

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = (time.time() - start_time) * 1000
        usage = response.usage

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider=self.provider_name,
            tokens_used=usage.total_tokens if usage else None,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_ms=round(latency_ms, 2),
        )

    def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        config: Optional[LLMConfig] = None,
    ) -> ToolCallResponse:
        """Generate with tool calling support."""
        model = config.model if config and config.model else self.default_model
        temperature = config.temperature if config else 0.7
        max_tokens = config.max_tokens if config else 1024

        start_time = time.time()

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        latency_ms = (time.time() - start_time) * 1000
        usage = response.usage
        message = response.choices[0].message

        # Parse tool calls if present
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return ToolCallResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            model=response.model,
            provider=self.provider_name,
            tokens_used=usage.total_tokens if usage else None,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_ms=round(latency_ms, 2),
        )

    def stream(self, messages: list[dict], config: Optional[LLMConfig] = None) -> Generator[StreamChunk, None, None]:
        """
        Stream tokens from OpenAI.

        OpenAI streaming response format:
            chunk.choices[0].delta.content = "token"
            chunk.choices[0].finish_reason = None | "stop"
        """
        model = config.model if config and config.model else self.default_model
        temperature = config.temperature if config else 0.7
        max_tokens = config.max_tokens if config else 1024

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},  # Get token counts at end
        )

        for chunk in response:
            if not chunk.choices:
                # Last chunk with usage info
                if chunk.usage:
                    yield StreamChunk(
                        content="",
                        model=chunk.model or model,
                        provider=self.provider_name,
                        finish_reason="stop",
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta and delta.content:
                yield StreamChunk(
                    content=delta.content,
                    model=chunk.model or model,
                    provider=self.provider_name,
                    finish_reason=choice.finish_reason,
                )
            elif choice.finish_reason:
                yield StreamChunk(
                    content="",
                    model=chunk.model or model,
                    provider=self.provider_name,
                    finish_reason=choice.finish_reason,
                )
