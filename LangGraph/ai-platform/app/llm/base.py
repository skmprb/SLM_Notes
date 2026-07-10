"""
BaseLLM - Phase 7: Added tool calling support.

Three generation modes:
    - generate(): Returns complete response (blocking)
    - stream(): Yields tokens as they arrive (non-blocking UX)
    - generate_with_tools(): Returns response OR tool_calls (for tool execution loop)

Tool calling flow:
    1. Send messages + tool schemas → LLM
    2. LLM returns either text OR tool_calls
    3. If tool_calls: execute tools, append results, call LLM again
    4. Repeat until LLM returns text (final answer)

LangChain equivalent:
    - model.invoke() → our generate()
    - model.stream() → our stream()
    - model.bind_tools(tools).invoke() → our generate_with_tools()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Generator


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


@dataclass
class LLMConfig:
    """Configuration for an LLM call."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    stop: Optional[list[str]] = None


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""
    content: str           # The token/text fragment
    model: str = ""
    provider: str = ""
    finish_reason: Optional[str] = None  # None = more coming, "stop" = done
    input_tokens: Optional[int] = None   # Only available on first/last chunk
    output_tokens: Optional[int] = None


@dataclass
class ToolCallResponse:
    """
    Response from generate_with_tools() — either text OR tool calls.

    When the LLM wants to call tools:
        - content = "" (or partial reasoning)
        - tool_calls = [ToolCallData(...)]

    When the LLM gives a final answer:
        - content = "The answer is..."
        - tool_calls = []
    """
    content: str
    tool_calls: list[dict] = field(default_factory=list)  # [{id, name, arguments}]
    model: str = ""
    provider: str = ""
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.

    Every provider MUST implement:
        - generate(): Synchronous full generation
        - stream(): Synchronous streaming generation
        - provider_name: Property returning the provider identifier
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier."""
        ...

    @abstractmethod
    def generate(self, messages: list[dict], config: Optional[LLMConfig] = None) -> LLMResponse:
        """Generate a complete response (blocking)."""
        ...

    @abstractmethod
    def stream(self, messages: list[dict], config: Optional[LLMConfig] = None) -> Generator[StreamChunk, None, None]:
        """
        Stream response tokens as they arrive.

        Yields:
            StreamChunk objects, one per token/fragment.
            Last chunk has finish_reason="stop".

        Usage:
            for chunk in provider.stream(messages):
                print(chunk.content, end="", flush=True)
        """
        ...

    def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        config: Optional[LLMConfig] = None,
    ) -> ToolCallResponse:
        """
        Generate a response with tool-calling capability.

        Args:
            messages: Conversation messages
            tools: Tool schemas in OpenAI format
            config: LLM configuration

        Returns:
            ToolCallResponse — check .has_tool_calls to see if tools were requested

        Default implementation: falls back to regular generate() (no tool support).
        Providers that support tool calling should override this.
        """
        response = self.generate(messages, config)
        return ToolCallResponse(
            content=response.content,
            tool_calls=[],
            model=response.model,
            provider=response.provider,
            tokens_used=response.tokens_used,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )

    def health_check(self) -> bool:
        """Check if the provider is reachable."""
        try:
            response = self.generate(
                messages=[{"role": "user", "content": "ping"}],
                config=LLMConfig(model="", max_tokens=5),
            )
            return bool(response.content)
        except Exception:
            return False
