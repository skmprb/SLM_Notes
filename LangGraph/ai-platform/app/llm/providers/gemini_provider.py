"""
Google Gemini Provider - Phase 5: Added streaming support.

Gemini streaming uses generate_content with stream=True.
"""

import time
from typing import Optional, Generator

from app.llm.base import BaseLLM, LLMConfig, LLMResponse, StreamChunk
from app.config.settings import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class GeminiProvider(BaseLLM):
    """Google Gemini implementation with streaming."""

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        self.default_model = default_model or settings.google_model

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            logger.info(f"GeminiProvider initialized | model={self.default_model}")
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    @property
    def provider_name(self) -> str:
        return "google"

    def generate(self, messages: list[dict], config: Optional[LLMConfig] = None) -> LLMResponse:
        model_name = config.model if config and config.model else self.default_model
        temperature = config.temperature if config else 0.7
        max_tokens = config.max_tokens if config else 1024

        system_instruction, gemini_messages = self._convert_messages(messages)

        model = self.genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction or None,
            generation_config=self.genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        start_time = time.time()
        response = model.generate_content(gemini_messages)
        latency_ms = (time.time() - start_time) * 1000

        input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else None
        output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else None

        return LLMResponse(
            content=response.text,
            model=model_name,
            provider=self.provider_name,
            tokens_used=(input_tokens or 0) + (output_tokens or 0) if input_tokens else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
        )

    def stream(self, messages: list[dict], config: Optional[LLMConfig] = None) -> Generator[StreamChunk, None, None]:
        """
        Stream tokens from Gemini.

        Gemini streaming yields response chunks with .text attribute.
        """
        model_name = config.model if config and config.model else self.default_model
        temperature = config.temperature if config else 0.7
        max_tokens = config.max_tokens if config else 1024

        system_instruction, gemini_messages = self._convert_messages(messages)

        model = self.genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction or None,
            generation_config=self.genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        response = model.generate_content(gemini_messages, stream=True)

        for chunk in response:
            if chunk.text:
                yield StreamChunk(
                    content=chunk.text,
                    model=model_name,
                    provider=self.provider_name,
                    finish_reason=None,
                )

        # Final chunk
        yield StreamChunk(
            content="",
            model=model_name,
            provider=self.provider_name,
            finish_reason="stop",
            input_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else None,
            output_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else None,
        )

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert standard messages to Gemini format."""
        system_instruction = ""
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})
        return system_instruction, gemini_messages
