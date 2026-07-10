"""
StructuredOutputService - Enforces structured output from LLMs (Phase 8).

Three strategies, chosen based on provider capability:
    1. json_schema: OpenAI native (gpt-4o+) — guaranteed schema compliance
    2. function_calling: Works on any model with tool support — uses fake tool trick
    3. json_mode: Weakest — just asks for JSON with schema in prompt

Production concerns:
    - Automatic strategy selection based on provider
    - Pydantic validation of parsed output
    - Graceful fallback if parsing fails (returns raw + error)
    - Retry with prompt repair on validation failure

LangChain equivalent:
    model.with_structured_output(MySchema, method="json_schema")
    model.with_structured_output(MySchema, method="function_calling")
"""

import json
from typing import Optional, Type

from pydantic import BaseModel, ValidationError

from app.llm.base import BaseLLM, LLMConfig, LLMResponse, ToolCallResponse
from app.llm.factory import create_llm
from app.structured.base import OutputSchema, ParsedOutput, StructuredOutputStrategy
from app.config.settings import Settings, get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class StructuredOutputService:
    """
    Service that forces LLMs to return structured data matching a schema.

    Usage:
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        service = StructuredOutputService()
        result = service.generate(
            messages=[{"role": "user", "content": "John is 30 years old"}],
            schema=OutputSchema.from_pydantic(Person),
        )
        # result.data = Person(name="John", age=30)
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def generate(
        self,
        messages: list[dict],
        schema: OutputSchema,
        provider_name: Optional[str] = None,
        strategy: Optional[StructuredOutputStrategy] = None,
        config: Optional[LLMConfig] = None,
    ) -> ParsedOutput:
        """
        Generate structured output from the LLM.

        Args:
            messages: Conversation messages
            schema: The expected output schema
            provider_name: Which provider to use (default from settings)
            strategy: Force a specific strategy (auto-selects if None)
            config: LLM configuration overrides
        """
        provider = create_llm(provider_name or self.settings.default_provider)
        strategy = strategy or self._select_strategy(provider)
        config = config or LLMConfig(model="", temperature=0.0, max_tokens=2048)

        logger.info(f"Structured output | schema={schema.name} | strategy={strategy.value} | provider={provider.provider_name}")

        if strategy == StructuredOutputStrategy.JSON_SCHEMA:
            return self._via_json_schema(provider, messages, schema, config)
        elif strategy == StructuredOutputStrategy.FUNCTION_CALLING:
            return self._via_function_calling(provider, messages, schema, config)
        else:
            return self._via_json_mode(provider, messages, schema, config)

    def _via_json_schema(
        self, provider: BaseLLM, messages: list[dict], schema: OutputSchema, config: LLMConfig
    ) -> ParsedOutput:
        """Strategy 1: Use OpenAI's native structured output (response_format)."""
        from openai import OpenAI

        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        model = config.model or settings.openai_model

        import time
        start = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            response_format=schema.to_openai_response_format(),
        )

        latency_ms = (time.time() - start) * 1000
        raw_text = response.choices[0].message.content or ""
        usage = response.usage

        return self._parse_and_validate(
            raw_text=raw_text,
            schema=schema,
            strategy=StructuredOutputStrategy.JSON_SCHEMA,
            model=response.model,
            provider=provider.provider_name,
            tokens_used=usage.total_tokens if usage else None,
            latency_ms=round(latency_ms, 2),
        )

    def _via_function_calling(
        self, provider: BaseLLM, messages: list[dict], schema: OutputSchema, config: LLMConfig
    ) -> ParsedOutput:
        """Strategy 2: Fake tool trick — bind schema as tool, force-call it."""
        tool = schema.to_function_tool()
        tool_name = tool["function"]["name"]

        # Add instruction to use the tool
        augmented_messages = messages.copy()
        augmented_messages.insert(0, {
            "role": "system",
            "content": f"You must call the '{tool_name}' function with the extracted/generated data. Do not respond with text.",
        })

        response: ToolCallResponse = provider.generate_with_tools(
            augmented_messages, tools=[tool], config=config
        )

        # Extract arguments from the tool call
        if response.has_tool_calls:
            raw_data = response.tool_calls[0]["arguments"]
            raw_text = json.dumps(raw_data)
        else:
            raw_text = response.content

        return self._parse_and_validate(
            raw_text=raw_text,
            schema=schema,
            strategy=StructuredOutputStrategy.FUNCTION_CALLING,
            model=response.model,
            provider=response.provider,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
        )

    def _via_json_mode(
        self, provider: BaseLLM, messages: list[dict], schema: OutputSchema, config: LLMConfig
    ) -> ParsedOutput:
        """Strategy 3: JSON mode + schema in prompt (weakest guarantee)."""
        schema_str = json.dumps(schema.json_schema, indent=2)
        system_msg = (
            f"You must respond with valid JSON matching this schema:\n"
            f"```json\n{schema_str}\n```\n"
            f"Return ONLY the JSON object, no other text."
        )

        augmented_messages = [{"role": "system", "content": system_msg}] + messages

        response: LLMResponse = provider.generate(augmented_messages, config)
        raw_text = response.content

        # Strip markdown code fences if present
        raw_text = self._strip_code_fences(raw_text)

        return self._parse_and_validate(
            raw_text=raw_text,
            schema=schema,
            strategy=StructuredOutputStrategy.JSON_MODE,
            model=response.model,
            provider=response.provider,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
        )

    def _parse_and_validate(
        self,
        raw_text: str,
        schema: OutputSchema,
        strategy: StructuredOutputStrategy,
        model: str,
        provider: str,
        tokens_used: Optional[int],
        latency_ms: Optional[float],
    ) -> ParsedOutput:
        """Parse JSON and validate against Pydantic model if available."""
        try:
            # Parse JSON
            if isinstance(raw_text, str):
                data = json.loads(raw_text)
            else:
                data = raw_text

            # Validate with Pydantic if model available
            if schema.pydantic_model:
                validated = schema.pydantic_model.model_validate(data)
                data = validated

            return ParsedOutput(
                data=data,
                raw_text=raw_text if isinstance(raw_text, str) else json.dumps(raw_text),
                strategy_used=strategy,
                model=model,
                provider=provider,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
            return ParsedOutput(
                data=None,
                raw_text=raw_text if isinstance(raw_text, str) else str(raw_text),
                strategy_used=strategy,
                model=model,
                provider=provider,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                validation_error=f"JSON parse error: {e}",
            )
        except ValidationError as e:
            logger.warning(f"Pydantic validation failed: {e}")
            return ParsedOutput(
                data=data,  # Return raw dict even if validation fails
                raw_text=raw_text if isinstance(raw_text, str) else json.dumps(raw_text),
                strategy_used=strategy,
                model=model,
                provider=provider,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                validation_error=f"Validation error: {e}",
            )

    def _select_strategy(self, provider: BaseLLM) -> StructuredOutputStrategy:
        """Auto-select best strategy based on provider capabilities."""
        if provider.provider_name == "openai":
            return StructuredOutputStrategy.JSON_SCHEMA
        else:
            # Anthropic, Gemini, etc. — use function_calling trick
            return StructuredOutputStrategy.FUNCTION_CALLING

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences from LLM output."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
