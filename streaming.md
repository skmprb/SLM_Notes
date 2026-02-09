Streaming Response Explanation
The streaming response implementation uses Server-Sent Events (SSE) to enable real-time communication between the backend and frontend. Here's how it works:

Backend (FastAPI) - /agentcore/chat/stream endpoint
Key Components:

StreamingResponse with async generator:

return StreamingResponse(
    generate_response(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", ...}
)

Event Format - Uses SSE format (data: {json}\n\n):

yield f"data: {json.dumps({'text': text})}\n\n"
yield f"data: {json.dumps({'tool_start': tool_name})}\n\n"
yield f"data: {json.dumps({'tool_result': result})}\n\n"


Event Types Streamed:

session_id - Session identifier

text - Incremental text chunks from AI response

message_end - Marks end of a message

tool_start - Tool execution begins (with tool_input)

tool_result - Tool execution result

tool_end - Tool execution completes

done - Entire conversation turn complete

error - Error occurred

AWS Bedrock AgentCore Integration:

Calls invoke_agent_runtime() with agent ID

Processes streaming response from Bedrock

Parses events like contentBlockStart, contentBlockDelta, contentBlockStop

Frontend (React) - AgentCoreWorkflowPage.tsx
Key Components:

Fetch API with ReadableStream:

const response = await fetch(`${API_URL}/agentcore/chat/stream`, {
  method: 'POST',
  body: JSON.stringify({ prompt, session_id, agent_id })
});
const reader = response.body?.getReader();

Copy

Insert at cursor
typescript
Stream Processing Loop:

while (true) {
  const { done, value } = await reader.read();
  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split(/\r?\n/);
  // Process each line starting with "data: "
}

Copy

Insert at cursor
typescript
Event Handling with Reducer:

parsed.text → Appends text to current message

parsed.tool_start → Creates tool message UI component

parsed.tool_result → Updates tool with result

parsed.tool_end → Marks tool as completed (green)

parsed.message_end → Resets message ID for next message

CloudWatch Logs Streaming (Secondary Stream):

Uses EventSource API for SSE

Connects to /agentcore/cloudwatch/logs/stream

Displays real-time execution logs in terminal panel

Flow Diagram:
User sends message
    ↓
Frontend: POST /agentcore/chat/stream
    ↓
Backend: invoke_agent_runtime() → AWS Bedrock AgentCore
    ↓
Backend: yield events as SSE (data: {...}\n\n)
    ↓
Frontend: ReadableStream reader processes chunks
    ↓
Frontend: Parse JSON events → Update UI via reducer
    ↓
Display: Text streams word-by-word, tools show progress

Copy

Insert at cursor
Key Benefits:
Real-time updates - Text appears as it's generated

Tool visibility - Users see when tools execute

Non-blocking - UI remains responsive

Efficient - Only sends incremental data


Perfect! Now I have all three files. Let me provide a comprehensive explanation of the streaming response architecture:

Complete Streaming Response Architecture
Three-Layer Streaming System
1. Agent Layer (Python - Strands Framework)
File: harnesspipeline_agent.py

Purpose: AWS Bedrock AgentCore agent that processes user requests and streams responses.

Key Components:

@app.entrypoint
async def agent_invocation(payload, context):
    agent = create_strands_agent(session_id)
    async for chunk in agent.stream_async(prompt):
        yield chunk  # Streams chunks to Bedrock AgentCore runtime

Copy

Insert at cursor
python
What it does:

Creates Strands Agent with Claude 3.5 Sonnet model

Processes user prompt with MCP tools

Streams response chunks asynchronously using agent.stream_async()

Each chunk contains text, tool invocations, or tool results

Stores session in S3 (qbddassistant-strands/harnesspipelinebuilder/)

Streaming Format: Bedrock AgentCore native format with events like:

contentBlockStart - Message/tool begins

contentBlockDelta - Incremental text/data

contentBlockStop - Message/tool ends

2. Backend Layer (FastAPI)
File: backend/pdlc/routers/agentcore.py

Purpose: Bridges frontend and AWS Bedrock AgentCore, transforms streaming format to SSE.

Key Function: /agentcore/chat/stream

async def generate_response():
    # Invoke Bedrock AgentCore agent
    response = await invoke_agent(agentcore_runtime_id, payload, session_id)
    
    # Process streaming response
    for line in streaming_body.iter_lines():
        data = json.loads(line_text[6:])  # Parse "data: {...}"
        
        # Transform events to frontend-friendly format
        if 'contentBlockDelta' in event:
            yield f"data: {json.dumps({'text': delta['text']})}\n\n"
        
        if 'contentBlockStart' in event and 'toolUse' in start:
            yield f"data: {json.dumps({'tool_start': tool_name, 'tool_input': tool_input})}\n\n"
        
        if 'toolResult' in start:
            yield f"data: {json.dumps({'tool_result': parsed_result})}\n\n"

return StreamingResponse(generate_response(), media_type="text/event-stream")


Copy

Insert at cursor
python
Event Transformation:

Bedrock Event	Backend SSE Event	Purpose
contentBlockDelta.delta.text	{text: "..."}	Stream text chunks
contentBlockStart.toolUse	{tool_start: "name", tool_input: {...}}	Tool execution begins
contentBlockStart.toolResult	{tool_result: {...}}	Tool result available
contentBlockStop	{tool_end: "name"}	Tool completes
messageStop	{message_end: true}	Message boundary
Headers (Critical for streaming):

headers={
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # Disable nginx buffering
    "Content-Type": "text/event-stream"
}

Copy

Insert at cursor
python
3. Frontend Layer (React/TypeScript)
File: frontend/apps/pdlc/src/components/pages/AgentCoreWorkflowPage.tsx

Purpose: Consumes SSE stream and updates UI in real-time.

Streaming Consumer:

const reader = response.body?.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || '';  // Keep incomplete line in buffer
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const parsed = JSON.parse(line.slice(6));
            
            // Handle different event types
            if (parsed.text) {
                dispatchMsg({ type: 'APPEND_TEXT', id: messageId, text: parsed.text });
            }
            if (parsed.tool_start) {
                dispatchMsg({ type: 'ADD', payload: { isTool: true, toolName: parsed.tool_start, isCompleted: false } });
            }
            if (parsed.tool_result) {
                dispatchMsg({ type: 'UPDATE', toolName: lastToolName, updates: { toolResult: parsed.tool_result } });
            }
            if (parsed.tool_end) {
                dispatchMsg({ type: 'UPDATE', toolName: parsed.tool_end, updates: { isCompleted: true } });
            }
        }
    }
}


Copy

Insert at cursor
typescript
UI Updates via Reducer:

const messagesReducer = (state, action) => {
    switch (action.type) {
        case 'APPEND_TEXT': 
            return state.map(m => m.id === action.id ? { ...m, text: m.text + action.text } : m);
        case 'UPDATE':
            return state.map(m => m.toolName === action.toolName ? { ...m, ...action.updates } : m);
    }
};

Copy

Insert at cursor
typescript
Complete Flow Diagram
┌─────────────────────────────────────────────────────────────────┐
│ USER: "Create Harness pipeline"                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND (React)                                                │
│ - POST /agentcore/chat/stream                                   │
│ - ReadableStream reader                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND (FastAPI)                                               │
│ - invoke_agent_runtime(agentcore_runtime_id)                    │
│ - Transform Bedrock events → SSE format                         │
│ - yield f"data: {json.dumps(event)}\n\n"                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ AWS BEDROCK AGENTCORE RUNTIME                                   │
│ - Executes harnesspipeline_agent.py                             │
│ - agent.stream_async(prompt)                                    │
│ - Streams: contentBlockDelta, toolUse, toolResult               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STRANDS AGENT (Python)                                          │
│ - Claude 3.5 Sonnet model                                       │
│ - MCP tools (invoke_harness_agent, check_agent_status)          │
│ - async for chunk in agent.stream_async()                       │
│ - S3 session storage                                            │
└─────────────────────────────────────────────────────────────────┘


Copy

Insert at cursor
Key Streaming Concepts
Chunked Transfer Encoding: Data sent in small pieces without knowing total size upfront

Server-Sent Events (SSE): One-way server→client streaming with data: {...}\n\n format

Async Generators: Python yield and TypeScript async for enable non-blocking streaming

Buffer Management: Frontend accumulates partial lines until complete \n\n received

Event-Driven UI: Reducer pattern updates UI incrementally as events arrive

This architecture enables real-time AI responses with tool execution visibility and CloudWatch log streaming!