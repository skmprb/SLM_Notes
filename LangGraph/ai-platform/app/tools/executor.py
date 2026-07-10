"""
ToolExecutor - Runs tools safely with timeout and error isolation (Phase 7).

Production concerns handled:
    - Timeout: Tools can't run forever (configurable per tool)
    - Error isolation: A failing tool returns an error message, doesn't crash the system
    - Result truncation: Oversized results are trimmed to fit context windows
    - Argument validation: Missing required args caught before execution

LangChain equivalent:
    - ToolNode in LangGraph (auto-executes tool calls from model output)
    - ToolExecutor class in langchain.agents (deprecated, replaced by LangGraph)
"""

import json
import concurrent.futures
from typing import Optional

from app.tools.base import ToolCall, ToolResult, ToolDefinition
from app.tools.registry import ToolRegistry, get_tool_registry
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ToolExecutor:
    """Executes tool calls safely with timeout and error handling."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or get_tool_registry()

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call with safety guarantees."""
        tool = self.registry.get(tool_call.name)

        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Tool '{tool_call.name}' not found",
                success=False,
                error="tool_not_found",
            )

        # Validate required arguments
        missing = self._check_required_args(tool, tool_call.arguments)
        if missing:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Missing required arguments: {missing}",
                success=False,
                error="missing_arguments",
            )

        # Execute with timeout
        try:
            result = self._run_with_timeout(tool, tool_call.arguments)
            content = self._format_result(result, tool.max_result_length)
            logger.info(f"Tool executed: {tool_call.name} | len={len(content)}")
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=content,
                success=True,
            )
        except concurrent.futures.TimeoutError:
            logger.warning(f"Tool timeout: {tool_call.name} | limit={tool.timeout_seconds}s")
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Tool '{tool_call.name}' timed out after {tool.timeout_seconds}s",
                success=False,
                error="timeout",
            )
        except Exception as e:
            logger.error(f"Tool error: {tool_call.name} | {type(e).__name__}: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error executing '{tool_call.name}': {type(e).__name__}: {str(e)}",
                success=False,
                error=str(e),
            )

    def execute_batch(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls (sequential for now)."""
        return [self.execute(tc) for tc in tool_calls]

    def _run_with_timeout(self, tool: ToolDefinition, arguments: dict):
        """Run tool handler with a timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool.handler, **arguments)
            return future.result(timeout=tool.timeout_seconds)

    def _check_required_args(self, tool: ToolDefinition, arguments: dict) -> list[str]:
        """Return list of missing required arguments."""
        required = [p.name for p in tool.parameters if p.required]
        return [r for r in required if r not in arguments]

    def _format_result(self, result, max_length: int) -> str:
        """Convert result to string and truncate if needed."""
        if isinstance(result, str):
            content = result
        elif isinstance(result, (dict, list)):
            content = json.dumps(result, indent=2, default=str)
        else:
            content = str(result)

        if len(content) > max_length:
            content = content[:max_length] + f"\n... [truncated, {len(content)} chars total]"
        return content
