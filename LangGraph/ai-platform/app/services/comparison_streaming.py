"""
COMPARISON: Our Streaming vs LangChain/LangGraph Streaming

Streaming is where LangChain really shines — they've built a sophisticated
event system. But understanding the fundamentals helps you debug and extend it.
"""

# ============================================================
# APPROACH 1: Our Custom Streaming
# ============================================================

def our_streaming():
    """
    Our streaming architecture:

    Client (EventSource/fetch)
        ↕ SSE (text/event-stream)
    FastAPI StreamingResponse
        ↕ Generator
    StreamingService
        ↕ Generator
    Provider.stream()
        ↕ Provider SDK streaming
    LLM API

    SSE Event format:
        data: {"type": "start", "provider": "openai"}
        data: {"type": "token", "content": "Hello"}
        data: {"type": "token", "content": " world"}
        data: {"type": "done", "tokens": 42, "latency_ms": 1234}
        data: [DONE]
    """
    pass


# ============================================================
# APPROACH 2: LangChain model.stream()
# ============================================================

def langchain_stream():
    """
    LangChain's basic streaming:

    from langchain.chat_models import init_chat_model

    model = init_chat_model("openai:gpt-4o-mini")

    # Yields AIMessageChunk objects
    for chunk in model.stream("Hello"):
        print(chunk.text, end="", flush=True)

    # Each chunk has:
    #   chunk.content → the token text
    #   chunk.response_metadata → provider-specific metadata
    #   chunk.usage_metadata → token counts (on last chunk)

    # Chunks can be accumulated:
    full = None
    for chunk in model.stream("Hello"):
        full = chunk if full is None else full + chunk
    # full is now a complete AIMessage
    """
    pass


# ============================================================
# APPROACH 3: LangChain astream_events (Advanced)
# ============================================================

def langchain_astream_events():
    """
    LangChain's event streaming — most powerful pattern.

    async for event in model.astream_events("Hello"):
        if event["event"] == "on_chat_model_start":
            print(f"Input: {event['data']['input']}")
        elif event["event"] == "on_chat_model_stream":
            print(f"Token: {event['data']['chunk'].text}")
        elif event["event"] == "on_chat_model_end":
            print(f"Full: {event['data']['output'].text}")

    This gives you:
    - Start/end events for every component
    - Token-level streaming
    - Tool call streaming
    - Metadata at each step

    Useful for complex chains where you need to know
    WHICH component is producing output.
    """
    pass


# ============================================================
# APPROACH 4: LangGraph Streaming
# ============================================================

def langgraph_streaming():
    """
    LangGraph streaming — graph-level streaming with multiple modes.

    # Stream modes:
    # "values" → full state after each node
    # "updates" → only the changes from each node
    # "messages" → token-level LLM output
    # "custom" → custom events from nodes

    for chunk in graph.stream(
        {"topic": "AI"},
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        if chunk["type"] == "updates":
            print(f"Node updated: {chunk['data']}")
        elif chunk["type"] == "messages":
            print(f"Token: {chunk['data']}")

    # Event streaming (most granular):
    stream = graph.stream_events(input, version="v3")
    for message in stream.messages:
        print(message.text)  # Token-by-token
    """
    pass


# ============================================================
# APPROACH 5: FastAPI + LangChain Streaming (Production Pattern)
# ============================================================

def fastapi_langchain_streaming():
    """
    The production pattern: FastAPI SSE + LangChain streaming.

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from langchain.chat_models import init_chat_model

    app = FastAPI()
    model = init_chat_model("openai:gpt-4o-mini")

    @app.post("/chat/stream")
    async def stream_chat(message: str):
        async def generate():
            async for chunk in model.astream(message):
                yield f"data: {json.dumps({'content': chunk.text})}\\n\\n"
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    This is essentially what we built, but using LangChain's model.astream()
    instead of our custom provider.stream().
    """
    pass


# ============================================================
# MAPPING TABLE
# ============================================================

"""
┌──────────────────────────────┬──────────────────────────────────────────────┐
│ Our Code                     │ LangChain/LangGraph Equivalent               │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ BaseLLM.stream()             │ BaseChatModel.stream() / .astream()          │
│ StreamChunk                  │ AIMessageChunk                               │
│ StreamingService             │ No equivalent (you build this yourself)      │
│ SSE formatting               │ No equivalent (framework-agnostic)           │
│ StreamingResponse            │ Same (FastAPI)                               │
│ "type": "token"             │ "on_chat_model_stream" event                 │
│ "type": "done"              │ "on_chat_model_end" event                    │
│ finish_reason                │ chunk.response_metadata["finish_reason"]     │
│ Token counting in stream     │ chunk.usage_metadata (last chunk)            │
└──────────────────────────────┴──────────────────────────────────────────────┘

Key differences:
    1. LangChain streams AIMessageChunk objects (rich, typed)
       We stream simple dicts over SSE (lightweight, protocol-agnostic)

    2. LangChain has "auto-streaming" (invoke() internally streams if context allows)
       We explicitly choose stream() vs generate()

    3. LangGraph streams at GRAPH level (node updates, custom events)
       We stream at LLM level only (for now)

    4. LangChain's astream_events gives you events from ALL components
       We only stream the final LLM output (simpler, but less visibility)
"""
