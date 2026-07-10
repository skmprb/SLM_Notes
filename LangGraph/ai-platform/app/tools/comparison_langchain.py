"""
Phase 7: Tool Calling — LangChain/LangGraph Comparison

Shows how our custom tool system maps to LangChain's tool calling patterns.
"""

# ============================================================
# 1. TOOL DEFINITION
# ============================================================

# --- OUR CODE ---
from app.tools.base import ToolDefinition, ToolParameter, ToolParameterType

calculator = ToolDefinition(
    name="calculator",
    description="Evaluate math expressions",
    parameters=[
        ToolParameter(name="expression", type=ToolParameterType.STRING, description="Math expression"),
    ],
    handler=lambda expression: str(eval(expression)),
)

# --- LANGCHAIN EQUIVALENT ---
# from langchain.tools import tool
#
# @tool
# def calculator(expression: str) -> str:
#     """Evaluate math expressions"""
#     return str(eval(expression))
#
# # Or with Pydantic schema:
# from pydantic import BaseModel, Field
# class CalculatorInput(BaseModel):
#     expression: str = Field(description="Math expression")
#
# @tool(args_schema=CalculatorInput)
# def calculator(expression: str) -> str:
#     """Evaluate math expressions"""
#     return str(eval(expression))


# ============================================================
# 2. BINDING TOOLS TO MODEL
# ============================================================

# --- OUR CODE ---
# tool_schemas = registry.get_openai_schemas()
# response = provider.generate_with_tools(messages, tools=tool_schemas)

# --- LANGCHAIN EQUIVALENT ---
# model_with_tools = model.bind_tools([calculator, web_search])
# response = model_with_tools.invoke(messages)
# response.tool_calls  # [{'name': 'calculator', 'args': {...}, 'id': '...'}]


# ============================================================
# 3. TOOL EXECUTION LOOP (ReAct Pattern)
# ============================================================

# --- OUR CODE ---
# for iteration in range(MAX_TOOL_ITERATIONS):
#     response = provider.generate_with_tools(messages, tool_schemas)
#     if not response.has_tool_calls:
#         break
#     for tc in response.tool_calls:
#         result = tool_executor.execute(ToolCall(...))
#         messages.append({"role": "tool", "tool_call_id": ..., "content": ...})

# --- LANGCHAIN/LANGGRAPH EQUIVALENT ---
# from langgraph.prebuilt import create_react_agent
#
# agent = create_react_agent(model, tools=[calculator, web_search])
# result = agent.invoke({"messages": [("user", "What is 25 * 47?")]})
#
# # Under the hood, LangGraph does the same loop:
# # 1. Call model with tools bound
# # 2. If tool_calls → ToolNode executes them
# # 3. Results fed back as ToolMessages
# # 4. Loop until model returns text


# ============================================================
# 4. TOOL REGISTRY
# ============================================================

# --- OUR CODE ---
# registry = get_tool_registry()
# registry.register(my_tool)
# schemas = registry.get_openai_schemas()

# --- LANGCHAIN EQUIVALENT ---
# tools = [calculator, web_search, datetime_tool]
# model.bind_tools(tools)  # Schemas auto-generated from @tool decorator
#
# # In LangGraph:
# from langgraph.prebuilt import ToolNode
# tool_node = ToolNode(tools)  # Auto-dispatches by tool name


# ============================================================
# 5. TOOL EXECUTOR WITH ERROR HANDLING
# ============================================================

# --- OUR CODE ---
# class ToolExecutor:
#     def execute(self, tool_call) -> ToolResult:
#         # timeout, error isolation, result truncation
#         ...

# --- LANGCHAIN EQUIVALENT ---
# ToolNode handles execution automatically
# For custom error handling:
# from langchain_core.tools import ToolException
#
# @tool
# def my_tool(x: str) -> str:
#     """My tool."""
#     if not x:
#         raise ToolException("Input required")  # Graceful error to LLM
#     return f"Result: {x}"
#
# my_tool.handle_tool_error = True  # Returns error as ToolMessage instead of crashing


# ============================================================
# 6. KEY DIFFERENCES
# ============================================================

# | Aspect              | Our Code                    | LangChain/LangGraph          |
# |---------------------|-----------------------------|------------------------------|
# | Tool definition     | ToolDefinition dataclass    | @tool decorator              |
# | Schema format       | Manual ToolParameter list   | Auto from type hints         |
# | Binding             | Pass schemas to API call    | model.bind_tools()           |
# | Execution loop      | Manual for-loop             | create_react_agent()         |
# | Error handling      | ToolResult.success=False    | ToolException + handle_error |
# | Timeout             | ThreadPoolExecutor          | Not built-in (custom)        |
# | Max iterations      | MAX_TOOL_ITERATIONS const   | recursion_limit config       |
# | Result truncation   | max_result_length           | Not built-in (custom)        |

# Our approach gives us:
# ✅ Full control over execution (timeouts, truncation)
# ✅ Understanding of what LangGraph does under the hood
# ✅ Easy to add custom middleware (logging, billing per tool call)
# ✅ Provider-agnostic (same loop works for OpenAI, Anthropic, etc.)
