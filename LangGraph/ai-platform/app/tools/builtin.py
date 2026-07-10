"""
Built-in Tools (Phase 7) - Sample tools that ship with the platform.

These demonstrate the tool system and are useful for testing.
In production, you'd add domain-specific tools (DB queries, API calls, etc.)

Tools included:
    - calculator: Evaluate math expressions safely
    - get_current_time: Get current date/time in any timezone
    - web_search: Mock web search (replace with real API in production)
"""

import math
from datetime import datetime, timezone, timedelta

from app.tools.base import ToolDefinition, ToolParameter, ToolParameterType


# ============================================================
# CALCULATOR TOOL
# ============================================================

def _calculator_handler(expression: str) -> str:
    """Safely evaluate a math expression."""
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10,
    }
    try:
        # Only allow safe math operations
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


calculator_tool = ToolDefinition(
    name="calculator",
    description="Evaluate mathematical expressions. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e. Use this for any math calculations.",
    parameters=[
        ToolParameter(
            name="expression",
            type=ToolParameterType.STRING,
            description="The math expression to evaluate, e.g. '25 * 47 + 13' or 'sqrt(144)'",
        ),
    ],
    handler=_calculator_handler,
    timeout_seconds=5.0,
)


# ============================================================
# DATETIME TOOL
# ============================================================

def _datetime_handler(timezone_offset: int = 0) -> str:
    """Get current date and time."""
    tz = timezone(timedelta(hours=timezone_offset))
    now = datetime.now(tz)
    return now.strftime(f"%Y-%m-%d %H:%M:%S (UTC{timezone_offset:+d})")


datetime_tool = ToolDefinition(
    name="get_current_time",
    description="Get the current date and time. Optionally specify a UTC offset for timezone.",
    parameters=[
        ToolParameter(
            name="timezone_offset",
            type=ToolParameterType.INTEGER,
            description="UTC offset in hours (e.g., -5 for EST, +5.5 for IST). Default 0 (UTC).",
            required=False,
            default=0,
        ),
    ],
    handler=_datetime_handler,
    timeout_seconds=5.0,
)


# ============================================================
# WEB SEARCH TOOL (Mock)
# ============================================================

def _web_search_handler(query: str, num_results: int = 3) -> str:
    """Mock web search - replace with real API (Tavily, Serper, etc.) in production."""
    # In production: call Tavily/Serper/Google Search API
    mock_results = [
        {"title": f"Result {i+1} for '{query}'", "snippet": f"This is a mock search result about {query}.", "url": f"https://example.com/{i+1}"}
        for i in range(min(num_results, 5))
    ]
    lines = []
    for r in mock_results:
        lines.append(f"- [{r['title']}]({r['url']}): {r['snippet']}")
    return "\n".join(lines)


web_search_tool = ToolDefinition(
    name="web_search",
    description="Search the web for current information. Use this when you need up-to-date facts, news, or information you don't have.",
    parameters=[
        ToolParameter(
            name="query",
            type=ToolParameterType.STRING,
            description="The search query",
        ),
        ToolParameter(
            name="num_results",
            type=ToolParameterType.INTEGER,
            description="Number of results to return (1-5)",
            required=False,
            default=3,
        ),
    ],
    handler=_web_search_handler,
    timeout_seconds=10.0,
)


# ============================================================
# REGISTRY HELPER
# ============================================================

def get_builtin_tools() -> list[ToolDefinition]:
    """Return all built-in tools for registration."""
    return [calculator_tool, datetime_tool, web_search_tool]
