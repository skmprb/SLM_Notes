"""
Structured Output - Base Models (Phase 8)

Core data structures:
    - OutputSchema: Defines the expected output shape (from Pydantic model or JSON schema)
    - ParsedOutput: The validated result after parsing LLM response

The problem this solves:
    LLM returns: "The person's name is John and they are 30 years old"
    We need:     {"name": "John", "age": 30}

Three enforcement strategies:
    1. json_schema: Provider-native (OpenAI structured outputs) — guaranteed schema match
    2. function_calling: Trick the LLM into "calling a tool" whose args ARE the output
    3. json_mode: Ask for JSON + describe schema in prompt — weakest guarantee

LangChain equivalent:
    - OutputSchema → Pydantic model passed to with_structured_output()
    - ParsedOutput → The validated Pydantic instance returned
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Type
from enum import Enum

from pydantic import BaseModel


class StructuredOutputStrategy(str, Enum):
    """How to enforce structured output from the LLM."""
    JSON_SCHEMA = "json_schema"           # OpenAI native structured outputs
    FUNCTION_CALLING = "function_calling"  # Fake tool trick (broad compatibility)
    JSON_MODE = "json_mode"               # response_format=json_object + prompt


@dataclass
class OutputSchema:
    """
    Defines the expected output structure.

    Can be created from:
        - A Pydantic model class (preferred — auto-generates JSON schema)
        - A raw JSON schema dict (for dynamic schemas)
    """
    name: str
    description: str
    json_schema: dict  # JSON Schema format
    pydantic_model: Optional[Type[BaseModel]] = None  # For validation

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel]) -> "OutputSchema":
        """Create OutputSchema from a Pydantic model class."""
        schema = model.model_json_schema()
        return cls(
            name=model.__name__,
            description=model.__doc__ or model.__name__,
            json_schema=schema,
            pydantic_model=model,
        )

    @classmethod
    def from_json_schema(cls, name: str, schema: dict, description: str = "") -> "OutputSchema":
        """Create OutputSchema from a raw JSON schema dict."""
        return cls(
            name=name,
            description=description or name,
            json_schema=schema,
            pydantic_model=None,
        )

    def to_openai_response_format(self) -> dict:
        """Convert to OpenAI's response_format parameter for json_schema strategy."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "description": self.description,
                "schema": self.json_schema,
                "strict": True,
            },
        }

    def to_function_tool(self) -> dict:
        """Convert to a fake tool definition for function_calling strategy."""
        return {
            "type": "function",
            "function": {
                "name": f"output_{self.name}",
                "description": f"Return the structured output: {self.description}",
                "parameters": self.json_schema,
            },
        }


@dataclass
class ParsedOutput:
    """Result of structured output parsing."""
    data: Any                          # The parsed data (dict or Pydantic instance)
    raw_text: str                      # Raw LLM response text
    strategy_used: StructuredOutputStrategy
    model: str = ""
    provider: str = ""
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    validation_error: Optional[str] = None  # Set if parsing/validation failed

    @property
    def success(self) -> bool:
        return self.validation_error is None
