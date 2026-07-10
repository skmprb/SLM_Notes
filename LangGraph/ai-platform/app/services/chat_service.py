"""
ChatService - Phase 7: Tool calling integration.

Now supports:
    - Tool execution loop (ReAct pattern)
    - LLM decides when to call tools vs respond directly
    - Max iterations to prevent infinite loops
    - Tool results fed back to LLM for final answer

Flow (with tools):
    1. User sends message
    2. ConversationService loads history
    3. LLM called with tool schemas attached
    4. If LLM returns tool_calls → execute tools → feed results back → repeat
    5. If LLM returns text → save and return

LangChain equivalent:
    - This is what create_react_agent() does in LangGraph
    - The loop is the "agent executor" pattern
"""

from typing import Optional

from app.llm.base import BaseLLM, ToolCallResponse
from app.llm.factory import create_llm
from app.llm.router import create_router, BaseRouter, RoutingContext
from app.services.llm_service import LLMService
from app.services.conversation_service import ConversationService
from app.tools import ToolExecutor, ToolCall, get_tool_registry
from app.models.schemas import ChatRequest, ChatResponse
from app.config.settings import Settings, get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

MAX_TOOL_ITERATIONS = 5  # Prevent infinite tool-calling loops


class ChatService:
    """Orchestrates chat with conversation memory + routing + resilience + tools."""

    def __init__(self, settings: Optional[Settings] = None, router: Optional[BaseRouter] = None):
        self.settings = settings or get_settings()
        self.router = router or create_router(self.settings.routing_strategy)
        self.resilient_llm = LLMService()
        self.conversations = ConversationService()
        self.tool_executor = ToolExecutor()
        self.tool_registry = get_tool_registry()
        logger.info(f"ChatService initialized | routing={self.settings.routing_strategy} | tools={self.tool_registry.tool_count}")

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request with conversation memory and tool calling."""

        # 1. Get or create session, prepare messages with history
        session = self.conversations.get_or_create_session(request.conversation_id)
        messages = self.conversations.prepare_messages(session.id, request.message)

        # 2. Route to best model
        if request.provider:
            provider = create_llm(request.provider)
            model_override = request.model
            routing_reason = f"User specified provider={request.provider}"
        else:
            context = RoutingContext(
                message=request.message,
                has_images=False,
                estimated_tokens=sum(len(m["content"]) for m in messages) // 4,
            )
            decision = self.router.route(context)
            provider = create_llm(decision.provider)
            model_override = decision.model
            routing_reason = decision.reason

        # 3. Tool execution loop
        from app.llm.base import LLMConfig
        config = LLMConfig(
            model=model_override or "",
            temperature=self.settings.openai_temperature,
            max_tokens=self.settings.openai_max_tokens,
        )

        tool_schemas = self.tool_registry.get_openai_schemas()
        tools_used = []
        total_tokens = 0
        total_input = 0
        total_output = 0

        for iteration in range(MAX_TOOL_ITERATIONS):
            response: ToolCallResponse = provider.generate_with_tools(messages, tool_schemas, config)
            total_tokens += response.tokens_used or 0
            total_input += response.input_tokens or 0
            total_output += response.output_tokens or 0

            if not response.has_tool_calls:
                # LLM gave a final text answer
                break

            # LLM wants to call tools — execute them
            logger.info(f"Tool calls (iter {iteration+1}): {[tc['name'] for tc in response.tool_calls]}")

            # Add assistant message with tool calls to conversation
            messages.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": str(tc["arguments"])}}
                    for tc in response.tool_calls
                ],
            })

            # Execute each tool and append results
            for tc in response.tool_calls:
                tool_call = ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                result = self.tool_executor.execute(tool_call)
                tools_used.append({"name": tc["name"], "success": result.success})

                messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                })
        else:
            # Hit max iterations — force a response
            logger.warning(f"Max tool iterations ({MAX_TOOL_ITERATIONS}) reached")
            response = ToolCallResponse(
                content="I've reached the maximum number of tool calls. Here's what I found so far based on the tool results above.",
                model=response.model,
                provider=response.provider,
            )

        # 4. Save assistant response to session
        self.conversations.add_assistant_message(
            session_id=session.id,
            content=response.content,
            model=response.model,
            provider=response.provider,
            tokens_used=total_tokens,
            latency_ms=response.latency_ms,
        )

        # 5. Return response
        return ChatResponse(
            message=response.content,
            model=response.model,
            provider=response.provider,
            tokens_used=total_tokens or None,
            input_tokens=total_input or None,
            output_tokens=total_output or None,
            latency_ms=response.latency_ms,
            routing_reason=routing_reason,
            was_fallback=False,
            attempts=1,
            conversation_id=session.id,
            tools_used=tools_used if tools_used else None,
        )

    def get_circuit_status(self) -> list[dict]:
        return self.resilient_llm.get_circuit_status()
