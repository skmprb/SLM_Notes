"""
ToolRegistry - Central registry for all available tools (Phase 7).

Responsibilities:
    - Register tools with their handlers
    - Provide schemas to send to the LLM
    - Look up tools by name for execution
    - List available tools for API introspection

LangChain equivalent:
    - model.bind_tools([tool1, tool2]) — binds tool schemas to the model
    - In LangGraph: ToolNode takes a list of tools and auto-dispatches

Production concerns:
    - Duplicate name detection (fail fast)
    - Schema validation at registration time
    - Tool enable/disable without code changes
"""

from typing import Optional
from app.tools.base import ToolDefinition
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ToolRegistry:
    """Registry of tools available to the LLM."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool. Raises if duplicate name."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        if not tool.handler:
            raise ValueError(f"Tool '{tool.name}' has no handler function")
        self._tools[tool.name] = tool
        logger.info(f"Tool registered: {tool.name}")

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def get_openai_schemas(self) -> list[dict]:
        """Get all tool schemas in OpenAI format (for sending to LLM)."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def list_tools(self) -> list[dict]:
        """List all registered tools (for API introspection)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": [
                    {"name": p.name, "type": p.type.value, "required": p.required}
                    for p in t.parameters
                ],
            }
            for t in self._tools.values()
        ]

    @property
    def tool_count(self) -> int:
        return len(self._tools)


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _register_builtin_tools(_registry)
    return _registry


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Register built-in tools on first access."""
    from app.tools.builtin import get_builtin_tools
    for tool in get_builtin_tools():
        registry.register(tool)
