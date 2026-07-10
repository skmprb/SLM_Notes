"""
Tool Calling - Base Models (Phase 7)

Core data structures for the tool system:
    - ToolDefinition: Schema that tells the LLM what a tool does + its parameters
    - ToolCall: What the LLM returns when it wants to use a tool
    - ToolResult: What we send back after executing the tool

The flow:
    1. We send ToolDefinition schemas to the LLM (so it knows what's available)
    2. LLM returns ToolCall (name + arguments it chose)
    3. We execute and return ToolResult (success/error + output)

LangChain equivalent:
    - ToolDefinition → tool schema passed to bind_tools()
    - ToolCall → response.tool_calls[0] (TypedDict with name, args, id)
    - ToolResult → ToolMessage (content + tool_call_id)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class ToolParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """A single parameter in a tool's input schema."""
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list[str]] = None


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool the LLM can call.

    This gets converted to the provider-specific format:
    - OpenAI: {"type": "function", "function": {"name": ..., "parameters": ...}}
    - Anthropic: {"name": ..., "input_schema": ...}
    """
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None  # The actual function to execute
    timeout_seconds: float = 30.0
    max_result_length: int = 4000  # Truncate results beyond this

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type.value, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolCall:
    """What the LLM returns when it wants to call a tool."""
    id: str              # Unique ID for this call (from the LLM)
    name: str            # Tool name to execute
    arguments: dict      # Parsed arguments


@dataclass
class ToolResult:
    """Result of executing a tool, sent back to the LLM."""
    tool_call_id: str    # Matches the ToolCall.id
    name: str            # Tool name (for logging)
    content: str         # The result (stringified)
    success: bool = True
    error: Optional[str] = None
