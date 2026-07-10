# Enterprise AI Platform — Complete Learning Notes

> A phase-by-phase guide that teaches Python concepts, design patterns,
> and LangChain/LangGraph equivalents through building a production AI platform.

---

## How to Read This Guide

Each phase covers:
1. **What we're building** — the production concern
2. **Python concepts used** — language features you need to understand
3. **Code walkthrough** — key code explained line by line
4. **Design patterns** — architectural decisions and why
5. **LangChain mapping** — how frameworks do the same thing

---


# PHASE 1: API Layer + Configuration + Logging

## What We're Building

The skeleton of the platform — a FastAPI server with centralized configuration
and structured logging. Every production system starts here.

## Python Concepts

### 1. Type Hints / Annotations

Python is dynamically typed, but type hints make code self-documenting:

```python
# Without hints — what does this return? What's `name`?
def greet(name):
    return f"Hello {name}"

# With hints — crystal clear
def greet(name: str) -> str:
    return f"Hello {name}"
```

In our code, you'll see:
```python
app_name: str = "Enterprise AI Platform"   # Variable annotation
def health() -> HealthResponse:            # Return type annotation
def chat(request: ChatRequest):            # Parameter annotation
```

**Why it matters in production:** IDEs catch bugs before runtime. New team members
understand the code instantly. Tools like mypy enforce correctness.

### 2. Pydantic BaseModel

Pydantic validates data at runtime using type hints:

```python
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    provider: Optional[str] = None
```

- `...` means "required, no default"
- `Field()` adds validation rules
- If someone sends `{"message": ""}` → Pydantic raises ValidationError automatically
- If someone sends `{"message": 123}` → Pydantic coerces or rejects

**LangChain equivalent:** LangChain uses Pydantic everywhere — tool schemas,
output parsers, config objects are all BaseModel subclasses.

### 3. Pydantic Settings (pydantic-settings)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = ""
    app_debug: bool = True

    class Config:
        env_file = ".env"
```

This automatically:
- Reads from environment variables (OPENAI_API_KEY)
- Falls back to .env file
- Falls back to default values
- Validates types (if env var is "true", converts to bool True)

**Why not just os.getenv()?** No validation, no type conversion, no defaults,
scattered across codebase, easy to misspell variable names.

### 4. @lru_cache — Singleton Pattern via Caching

```python
from functools import lru_cache

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

`@lru_cache()` memoizes the function — first call creates Settings,
subsequent calls return the same instance. This is Python's idiomatic singleton.

**Why:** Settings reads .env file and validates — expensive to do on every request.
Do it once, reuse everywhere.

### 5. FastAPI Decorators

```python
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    ...
```

- `@router.post("/chat")` — registers this function as a POST endpoint
- `response_model=ChatResponse` — FastAPI auto-serializes return value to this schema
- `request: ChatRequest` — FastAPI auto-deserializes request body into this model

**LangChain equivalent:** LangServe wraps LangChain chains as FastAPI endpoints
using the same pattern.

## Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app creation, router mounting |
| `app/config/settings.py` | All configuration in one place |
| `app/utils/logger.py` | Structured logging setup |
| `app/api/routes.py` | All HTTP endpoints |
| `app/models/schemas.py` | Request/response Pydantic models |

## Design Pattern: Layered Architecture

```
Routes (HTTP) → Services (Business Logic) → Providers (External APIs)
```

Each layer only talks to the one below it. Routes never call OpenAI directly.
Services don't know about HTTP. This separation lets you test each layer independently.

---


# PHASE 2: LLM Abstraction Layer

## What We're Building

A provider-agnostic interface so the rest of the code never knows (or cares)
whether it's talking to OpenAI, Anthropic, or Gemini.

## Python Concepts

### 1. Abstract Base Classes (ABC)

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def generate(self, messages: list[dict], config: Optional[LLMConfig] = None) -> LLMResponse: ...
```

- `ABC` = Abstract Base Class. You CANNOT instantiate it directly.
- `@abstractmethod` = subclasses MUST implement this. If they don't → TypeError at class creation.
- `...` (Ellipsis) = placeholder body for abstract methods.

**Why ABC and not just a regular class?**
- Enforces the contract at class definition time (not at runtime when it's too late)
- Documents the interface explicitly
- IDE shows you what to implement

**LangChain equivalent:** `BaseChatModel` is their ABC. All providers (ChatOpenAI,
ChatAnthropic, etc.) inherit from it and implement `_generate()`.

### 2. @property Decorator

```python
class OpenAIProvider(BaseLLM):
    @property
    def provider_name(self) -> str:
        return "openai"
```

`@property` makes a method behave like an attribute:
```python
provider = OpenAIProvider()
print(provider.provider_name)  # "openai" — no parentheses needed
```

Combined with `@abstractmethod`, it forces subclasses to define a read-only attribute.

### 3. Dataclasses

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
```

`@dataclass` auto-generates:
- `__init__()` — constructor from fields
- `__repr__()` — readable string representation
- `__eq__()` — comparison by field values

**Why not a dict?** Dicts have no structure — `response["conten"]` (typo) fails silently.
Dataclasses give you autocomplete, type checking, and clear documentation.

**Why not Pydantic?** Dataclasses are lighter — no validation overhead. Use Pydantic
for external data (API requests), dataclasses for internal data structures.

### 4. Optional Type

```python
from typing import Optional

tokens_used: Optional[int] = None
# Equivalent to: Union[int, None]
# Means: this field can be an int OR None
```

**Why:** Not all providers return token counts. Rather than inventing a sentinel
value (-1), we use None to mean "not available."

### 5. Factory Pattern

```python
# app/llm/factory.py
PROVIDER_REGISTRY = {
    "openai": "app.llm.providers.openai_provider.OpenAIProvider",
    "anthropic": "app.llm.providers.anthropic_provider.AnthropicProvider",
    "google": "app.llm.providers.gemini_provider.GeminiProvider",
}

def create_llm(provider: str) -> BaseLLM:
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider}")
    module_path, class_name = PROVIDER_REGISTRY[provider].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()
```

**Lazy imports:** We don't import all providers at startup. If you only use OpenAI,
Anthropic's SDK is never loaded. This speeds up startup and avoids import errors
for uninstalled packages.

**LangChain equivalent:** `init_chat_model("openai:gpt-4o")` — same factory pattern,
parses provider from the model string.

## Design Pattern: Adapter Pattern

Each provider adapts a different API shape to our uniform interface:

```
OpenAI API:  client.chat.completions.create() → response.choices[0].message.content
Anthropic:   client.messages.create()         → response.content[0].text
Gemini:      model.generate_content()         → response.text

All adapted to → LLMResponse(content=..., model=..., provider=...)
```

The rest of our code only sees `LLMResponse`. Provider differences are hidden.

---


# PHASE 3: Model Routing

## What We're Building

Intelligent routing that picks the best model for each request based on
task complexity, cost constraints, or capability requirements.

## Python Concepts

### 1. Inheritance and Polymorphism

```python
class BaseRouter(ABC):
    @abstractmethod
    def route(self, context: RoutingContext) -> RoutingDecision: ...

class RuleBasedRouter(BaseRouter):
    def route(self, context: RoutingContext) -> RoutingDecision:
        # keyword matching logic
        ...

class CostAwareRouter(BaseRouter):
    def route(self, context: RoutingContext) -> RoutingDecision:
        # cheapest model that meets quality threshold
        ...
```

**Polymorphism** = same method name (`route`), different behavior depending on
which class you're using. The caller doesn't need to know which router it has:

```python
router: BaseRouter = create_router("cost_aware")  # Could be any router
decision = router.route(context)  # Works regardless of which implementation
```

### 2. Dataclass with Default Factory

```python
from dataclasses import dataclass, field

@dataclass
class RoutingContext:
    message: str
    has_images: bool = False
    estimated_tokens: int = 0
    metadata: dict = field(default_factory=dict)
```

**Why `field(default_factory=dict)` instead of `metadata: dict = {}`?**

The mutable default trap:
```python
# WRONG — all instances share the SAME dict object!
@dataclass
class Bad:
    items: list = []

a = Bad()
b = Bad()
a.items.append(1)
print(b.items)  # [1] — b is affected! Bug!

# CORRECT — each instance gets a NEW dict
@dataclass
class Good:
    items: list = field(default_factory=list)
```

`default_factory` is called fresh for each instance.

### 3. Enum for Fixed Choices

```python
# In the router, strategies are a fixed set:
# "rule_based" | "cost_aware" | "llm"
```

While we use strings here for simplicity, production code often uses Enums:
```python
from enum import Enum

class RoutingStrategy(str, Enum):
    RULE_BASED = "rule_based"
    COST_AWARE = "cost_aware"
    LLM = "llm"
```

`str, Enum` means it's both a string AND an enum — can be used in JSON serialization.

## Design Pattern: Strategy Pattern

The Strategy pattern lets you swap algorithms at runtime:

```
┌─────────────┐         ┌──────────────────┐
│ ChatService │────────→│ BaseRouter (ABC)  │
│             │         └──────────────────┘
│ self.router │                  △
└─────────────┘         ┌────────┼────────┐
                        │        │        │
                   RuleBased  CostAware  LLMRouter
```

Changing routing strategy = changing one config value. Zero code changes.

**LangChain equivalent:** LangChain doesn't have built-in routing, but LangGraph
uses conditional edges (`should_continue`) to route between nodes — same concept.

---


# PHASE 4: Resilience (Retry + Circuit Breaker + Fallback)

## What We're Building

LLMs fail. Networks timeout. APIs rate-limit you. This phase ensures the
system recovers gracefully instead of crashing.

## Python Concepts

### 1. Exception Hierarchy (Custom Exceptions)

```python
class LLMException(Exception):
    """Base exception for all LLM errors."""
    pass

class RetryableError(LLMException):
    """Errors worth retrying (timeout, rate limit, 5xx)."""
    pass

class NonRetryableError(LLMException):
    """Errors that will never succeed on retry (auth, invalid input)."""
    pass
```

**Why custom exceptions?**
- `except RetryableError` — retry it
- `except NonRetryableError` — fail immediately, don't waste time retrying
- `except LLMException` — catch all LLM errors but not unrelated bugs

Python's exception hierarchy:
```
BaseException
  └── Exception
        └── LLMException (ours)
              ├── RetryableError
              └── NonRetryableError
```

### 2. try/except/finally

```python
try:
    response = provider.generate(messages)
except RateLimitError as e:
    # Specific handling — wait and retry
    time.sleep(e.retry_after)
except TimeoutError:
    # Different handling — try next provider
    raise RetryableError("Timeout")
except Exception as e:
    # Catch-all — log and re-raise
    logger.error(f"Unexpected: {e}")
    raise
finally:
    # ALWAYS runs — cleanup
    metrics.record_attempt()
```

**`as e`** captures the exception object for inspection.
**`raise`** without arguments re-raises the current exception.
**`finally`** runs whether or not an exception occurred.

### 3. time.sleep() and Exponential Backoff

```python
import time
import random

def exponential_backoff(attempt: int, base_delay: float = 1.0, factor: float = 2.0) -> float:
    delay = base_delay * (factor ** attempt)
    jitter = random.uniform(0, delay * 0.1)  # Add randomness
    return delay + jitter

# attempt 0: ~1.0s
# attempt 1: ~2.0s
# attempt 2: ~4.0s
# attempt 3: ~8.0s
```

**Why jitter?** If 1000 clients all retry at exactly 2 seconds, the server gets
hammered again. Random jitter spreads retries over time.

**LangChain equivalent:** `model.with_retry(stop_after_attempt=3)` — uses tenacity
library internally with exponential backoff.

### 4. State Machine (Circuit Breaker)

```python
class CircuitState(Enum):
    CLOSED = "closed"      # Normal — requests flow through
    OPEN = "open"          # Broken — reject immediately
    HALF_OPEN = "half_open"  # Testing — allow one request

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0.0
```

State transitions:
```
CLOSED ──(5 failures)──→ OPEN ──(60s passes)──→ HALF_OPEN
   △                                                │
   └──────────(success)────────────────────────────┘
                                                    │
   OPEN ←──────────(failure)───────────────────────┘
```

**Why:** If OpenAI is down, don't keep sending requests (wastes time, gets rate-limited).
Skip it for 60 seconds, then test with one request.

### 5. The Fallback Chain

```python
class FallbackManager:
    def call(self, messages, config) -> ResilientResponse:
        for provider_name in self.fallback_chain:  # ["openai", "anthropic", "google"]
            if self.circuit_breakers[provider_name].is_open:
                continue  # Skip broken providers
            try:
                response = self.retry_handler.call_with_retry(provider, messages, config)
                return ResilientResponse(response=response, was_fallback=...)
            except RetryableError:
                continue  # Try next provider
        raise AllProvidersFailedError()
```

**LangChain equivalent:** `model.with_fallbacks([fallback1, fallback2])` — same
chain concept, tries each model in order.

## Design Pattern: Circuit Breaker Pattern

From Michael Nygard's "Release It!" — prevents cascading failures in distributed systems.
Used by Netflix, Amazon, and every serious microservice architecture.

---


# PHASE 5: Streaming (Server-Sent Events)

## What We're Building

Instead of waiting 10-20 seconds for a complete response, stream tokens
to the user as they're generated. Time-to-first-token (TTFT) is the key UX metric.

## Python Concepts

### 1. Generators (yield)

```python
def count_up(n: int):
    """A generator function — uses yield instead of return."""
    for i in range(n):
        yield i  # Pauses here, returns value, resumes on next iteration

# Usage:
for num in count_up(5):
    print(num)  # 0, 1, 2, 3, 4
```

**Key difference from return:**
- `return` — function ends, all values computed at once (memory-heavy)
- `yield` — function pauses, produces one value at a time (memory-efficient)

**Why generators for streaming?**
- LLM produces tokens one at a time
- We don't want to buffer the entire response in memory
- We want to send each token to the client immediately

### 2. Generator Type Annotation

```python
from typing import Generator

def stream(self, messages: list[dict]) -> Generator[StreamChunk, None, None]:
    ...
    yield StreamChunk(content="Hello")
    yield StreamChunk(content=" world")
```

`Generator[YieldType, SendType, ReturnType]`:
- YieldType = what it yields (StreamChunk)
- SendType = what you can send into it (None — we don't use this)
- ReturnType = what it returns when done (None)

### 3. Server-Sent Events (SSE) Format

```python
def format_sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"
```

SSE is a simple text protocol:
```
event: token
data: {"content": "Hello"}

event: token
data: {"content": " world"}

event: done
data: [DONE]
```

Each message is separated by `\n\n`. The browser/client reads them as they arrive.

### 4. FastAPI StreamingResponse

```python
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
def chat_stream(request: StreamRequest):
    return StreamingResponse(
        streaming_service.stream_chat(messages),  # Generator!
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
```

FastAPI iterates over the generator and sends each yielded string to the client
as it's produced. The connection stays open until the generator is exhausted.

### 5. flush=True (Real-time Output)

```python
for chunk in provider.stream(messages):
    print(chunk.content, end="", flush=True)
```

`flush=True` forces Python to write to stdout immediately instead of buffering.
Without it, you'd see nothing until the buffer fills up (defeats the purpose of streaming).

**LangChain equivalent:**
```python
for chunk in model.stream("Hello"):
    print(chunk.content, end="", flush=True)
```
Same pattern — LangChain's `.stream()` returns a generator of `AIMessageChunk`.

---


# PHASE 6: Conversation Management

## What We're Building

Persistent conversations — the LLM remembers what you said earlier in the chat.
Without this, every message is independent (the LLM has no memory).

## Python Concepts

### 1. Enum (MessageRole)

```python
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
```

`str, Enum` — inherits from both str and Enum. This means:
- `MessageRole.USER == "user"` → True (can compare with strings)
- `MessageRole.USER.value` → "user"
- JSON serializable without custom encoder

**Why not just strings?** Typos. `"assitant"` (misspelled) would silently break things.
Enums catch this at definition time.

### 2. Abstract Base Class with Multiple Implementations

```python
class BaseConversationStore(ABC):
    @abstractmethod
    def save_session(self, session: Session) -> None: ...
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]: ...

class InMemoryStore(BaseConversationStore):
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def save_session(self, session: Session) -> None:
        self._sessions[session.id] = session

class SQLiteStore(BaseConversationStore):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
```

**Dependency Inversion Principle:** High-level code (ConversationService) depends on
the abstraction (BaseConversationStore), not the concrete implementation.
Swap SQLite for PostgreSQL by changing one config value.

### 3. Context Window Management (List Slicing)

```python
def prepare_messages(self, session_id: str, new_message: str) -> list[dict]:
    messages = self.get_session_messages(session_id)
    # Keep only last 20 messages (context window limit)
    recent = messages[-20:]
    recent.append({"role": "user", "content": new_message})
    return recent
```

`messages[-20:]` — Python negative slicing. Takes the last 20 items.
- `[-1]` = last item
- `[-20:]` = last 20 items (or all if fewer than 20)

**Why trim?** LLMs have token limits (128K for GPT-4o). Sending the entire
conversation history would exceed the limit and cost a fortune.

### 4. UUID for Session IDs

```python
import uuid

session_id = str(uuid.uuid4())
# "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

UUID4 = random 128-bit number. Collision probability is astronomically low
(you'd need to generate 1 billion UUIDs per second for 85 years to have a 50% chance).

**Why not auto-increment integers?** UUIDs are:
- Globally unique (no coordination needed between servers)
- Not guessable (security — can't enumerate other users' sessions)
- Generated client-side (no database round-trip needed)

### 5. SQLite (Embedded Database)

```python
import sqlite3

conn = sqlite3.connect("conversations.db")
cursor = conn.execute("INSERT INTO sessions (id, title) VALUES (?, ?)", (id, title))
conn.commit()
```

SQLite is a file-based database — no server needed. Perfect for:
- Development and testing
- Single-server deployments
- Prototyping before migrating to PostgreSQL

**LangChain equivalent:**
- `InMemoryChatMessageHistory` → our InMemoryStore
- `SQLChatMessageHistory` → our SQLiteStore
- LangGraph `MemorySaver` / `SqliteSaver` → checkpointing (saves full graph state)

---


# PHASE 7: Tool Calling

## What We're Building

The LLM can now call external functions (calculator, search, APIs) and use
the results in its response. This is what turns a chatbot into an agent.

## Python Concepts

### 1. Callable Type (Functions as First-Class Objects)

```python
from typing import Callable

@dataclass
class ToolDefinition:
    name: str
    handler: Callable  # A function stored as data!
```

In Python, functions are objects. You can:
```python
def add(a, b): return a + b

# Store in a variable
my_func = add

# Store in a dict
tools = {"add": add}

# Pass as argument
def execute(fn, *args):
    return fn(*args)

execute(add, 2, 3)  # 5
```

**Why this matters for tools:** We store the handler function in the ToolDefinition,
then call it later when the LLM requests it. The registry doesn't know or care
what the function does — it just calls it.

### 2. **kwargs (Keyword Argument Unpacking)

```python
def _calculator_handler(expression: str) -> str:
    return str(eval(expression))

# When the LLM says: call calculator with {"expression": "25 * 47"}
# We do:
arguments = {"expression": "25 * 47"}
result = handler(**arguments)  # Same as: handler(expression="25 * 47")
```

`**arguments` unpacks a dict into keyword arguments. This is how we dynamically
call tool functions with arguments the LLM chose.

### 3. concurrent.futures (Timeout Execution)

```python
import concurrent.futures

def _run_with_timeout(self, tool, arguments):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tool.handler, **arguments)
        return future.result(timeout=tool.timeout_seconds)
```

- `ThreadPoolExecutor` — runs code in a separate thread
- `executor.submit()` — starts execution, returns a Future
- `future.result(timeout=30)` — waits up to 30s, raises TimeoutError if exceeded

**Why:** A buggy tool (infinite loop, hanging network call) shouldn't freeze
the entire platform. Timeout kills it after N seconds.

### 4. The Tool Execution Loop (ReAct Pattern)

```python
for iteration in range(MAX_TOOL_ITERATIONS):
    response = provider.generate_with_tools(messages, tool_schemas)

    if not response.has_tool_calls:
        break  # LLM gave final answer

    # Execute tools, append results
    for tc in response.tool_calls:
        result = tool_executor.execute(ToolCall(...))
        messages.append({"role": "tool", "content": result.content})
```

This is the **ReAct** (Reason + Act) loop:
1. LLM reasons about what to do
2. LLM acts (calls a tool)
3. Tool result fed back
4. LLM reasons again with new information
5. Repeat until LLM has enough info to answer

**MAX_TOOL_ITERATIONS = 5** prevents infinite loops (LLM keeps calling tools forever).

**LangChain equivalent:**
```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model, tools=[calculator, search])
# This creates the same loop internally
```

### 5. JSON Schema for Tool Definitions

```python
def to_openai_schema(self) -> dict:
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...],
            },
        },
    }
```

The LLM needs to know what tools are available and what arguments they accept.
We describe this as JSON Schema — a standard format all providers understand.

---


# PHASE 8: Structured Output

## What We're Building

Force the LLM to return JSON matching a specific schema instead of free-form text.
Critical for APIs that need predictable, parseable responses.

## Python Concepts

### 1. Type[BaseModel] — Passing Classes as Arguments

```python
from typing import Type
from pydantic import BaseModel

def from_pydantic(cls, model: Type[BaseModel]) -> "OutputSchema":
    schema = model.model_json_schema()
    return cls(name=model.__name__, json_schema=schema, pydantic_model=model)
```

`Type[BaseModel]` means "a class that IS or inherits from BaseModel" (not an instance).

```python
class Person(BaseModel):
    name: str
    age: int

# Passing the CLASS itself, not an instance
schema = OutputSchema.from_pydantic(Person)  # Person, not Person()
```

**Why:** We need the class to:
- Generate its JSON schema (`model_json_schema()`)
- Validate data later (`model.model_validate(data)`)

### 2. @classmethod — Alternative Constructors

```python
class OutputSchema:
    @classmethod
    def from_pydantic(cls, model: Type[BaseModel]) -> "OutputSchema":
        ...

    @classmethod
    def from_json_schema(cls, name: str, schema: dict) -> "OutputSchema":
        ...
```

`@classmethod` receives the class (`cls`) as first argument instead of an instance (`self`).
Used for alternative ways to create an object:

```python
# Multiple ways to create the same thing:
schema1 = OutputSchema.from_pydantic(Person)
schema2 = OutputSchema.from_json_schema("person", {...})
```

**Real-world analogy:** `datetime.now()`, `datetime.fromtimestamp()`, `datetime.fromisoformat()`
— all classmethods that create datetime objects differently.

### 3. model_json_schema() — Pydantic Schema Generation

```python
class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")

Person.model_json_schema()
# {
#   "type": "object",
#   "properties": {
#     "name": {"type": "string", "description": "Full name"},
#     "age": {"type": "integer", "description": "Age in years"}
#   },
#   "required": ["name", "age"]
# }
```

Pydantic auto-generates JSON Schema from your Python class. This schema is what
we send to the LLM to tell it the expected output format.

### 4. model_validate() — Runtime Validation

```python
data = {"name": "John", "age": 30}
person = Person.model_validate(data)  # Returns Person instance
# person.name == "John", person.age == 30

bad_data = {"name": "John", "age": "not a number"}
Person.model_validate(bad_data)  # Raises ValidationError!
```

This is the validation step — even if the LLM returns valid JSON, we verify
it matches our schema exactly.

### 5. Three Strategies (Why Multiple Approaches?)

```python
class StructuredOutputStrategy(str, Enum):
    JSON_SCHEMA = "json_schema"           # Best: provider guarantees schema match
    FUNCTION_CALLING = "function_calling"  # Good: trick via fake tool
    JSON_MODE = "json_mode"               # Weak: just asks nicely
```

**json_schema** (OpenAI only): The API itself enforces the schema. Output is
guaranteed to be valid. Uses constrained decoding internally.

**function_calling** (any provider with tools): We create a fake tool whose
parameters ARE our output schema, then force the LLM to "call" it.
The "arguments" it generates = our structured output. Clever trick!

**json_mode**: We put the schema in the system prompt and say "respond in JSON."
Weakest guarantee — LLM might not follow instructions perfectly.

**LangChain equivalent:**
```python
model.with_structured_output(Person, method="json_schema")
model.with_structured_output(Person, method="function_calling")
model.with_structured_output(Person, method="json_mode")
```

---


# PHASE 9: Prompt Management

## What We're Building

A system to manage prompts as external files (not hardcoded strings),
with variable substitution and versioning support.

## Python Concepts

### 1. pathlib.Path — Modern File Paths

```python
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
# __file__ = current file's path
# .parent = directory containing this file
# / "templates" = join path (works on Windows AND Linux!)
```

`Path` is better than string concatenation:
```python
# Old way (breaks on different OS):
path = os.path.join(os.path.dirname(__file__), "templates")

# New way (cross-platform, readable):
path = Path(__file__).parent / "templates"
```

Useful methods:
```python
path.exists()           # Does it exist?
path.glob("*.md")       # Find all .md files
path.read_text()        # Read entire file as string
path.stem               # Filename without extension ("system" from "system.md")
```

### 2. Regular Expressions (re module)

```python
import re

content = "Hello {user_name}, the time is {current_time}"
variables = re.findall(r"\{(\w+)\}", content)
# variables = ["user_name", "current_time"]
```

Breaking down the regex `\{(\w+)\}`:
- `\{` — literal `{` (escaped because `{` is special in regex)
- `(\w+)` — capture group: one or more word characters (letters, digits, underscore)
- `\}` — literal `}`

`re.findall()` returns all matches of the capture group.

### 3. String .replace() for Template Rendering

```python
def render(self, **kwargs) -> str:
    result = self.content
    for var in self.variables:
        placeholder = "{" + var + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(kwargs[var]))
    return result

# Usage:
template.render(user_name="John", current_time="2024-01-01")
```

Simple but effective. For production, you might use Jinja2 for complex templates
(loops, conditionals), but string replacement works for most prompt templates.

### 4. Global Singleton with Lazy Initialization

```python
_registry: Optional[PromptRegistry] = None

def get_prompt_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()  # Created only on first call
    return _registry
```

**`global` keyword:** Tells Python "I want to modify the module-level variable,
not create a local one." Without `global`, `_registry = PromptRegistry()` would
create a new local variable and the module-level one stays None.

**Why lazy?** The registry loads files from disk. Don't do that at import time —
do it when first needed.

**LangChain equivalent:**
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are {role}. Current time: {time}"),
    ("user", "{input}"),
])
formatted = prompt.invoke({"role": "assistant", "time": "now", "input": "hi"})
```

---


# PHASE 10: RAG (Retrieval-Augmented Generation)

## What We're Building

Give the LLM access to your private data by retrieving relevant documents
and injecting them into the prompt. The LLM answers based on YOUR data,
not just its training data.

## Python Concepts

### 1. Hashing for Document IDs

```python
import hashlib

doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
# "a1b2c3d4e5f6"
```

- `.encode()` — converts string to bytes (hashlib needs bytes)
- `.hexdigest()` — returns hash as hex string
- `[:12]` — take first 12 chars (shorter ID, still unique enough)

**Why hash?** Same content always produces same ID. Useful for deduplication —
if you ingest the same document twice, it gets the same ID.

### 2. List Comprehensions with zip()

```python
embeddings = self.embedding.embed(texts)
for chunk, emb in zip(chunks, embeddings):
    chunk.embedding = emb
```

`zip(chunks, embeddings)` pairs up elements from two lists:
```python
chunks     = [chunk1, chunk2, chunk3]
embeddings = [emb1,   emb2,   emb3]
zip(...)   = [(chunk1, emb1), (chunk2, emb2), (chunk3, emb3)]
```

### 3. Cosine Similarity (Vector Math)

```python
def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)
```

Cosine similarity measures how "similar" two vectors are (0 = unrelated, 1 = identical).

**How embeddings work:**
1. Text → Embedding model → Vector of 1536 floats (for OpenAI)
2. Similar texts produce similar vectors
3. "Python programming" and "coding in Python" → vectors close together
4. "Python programming" and "chocolate cake recipe" → vectors far apart

**The RAG pipeline:**
```
User query: "How do I deploy to AWS?"
    ↓ embed
Query vector: [0.1, 0.3, -0.2, ...]
    ↓ cosine similarity against all stored vectors
Top 3 most similar documents retrieved
    ↓ inject into prompt
"Based on these documents: [doc1, doc2, doc3], answer: How do I deploy to AWS?"
    ↓ LLM generates answer grounded in YOUR docs
```

### 4. Chunking with Overlap

```python
def chunk(self, text: str, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap  # Overlap!
```

**Why overlap?** If a sentence spans two chunks, overlap ensures it appears
in at least one chunk completely. Without overlap, you might split
"The capital of France is Paris" into "The capital of France" and "is Paris" —
neither chunk has the full fact.

**LangChain equivalent:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_text(text)
vectorstore = FAISS.from_texts(docs, OpenAIEmbeddings())
results = vectorstore.similarity_search("query", k=5)
```

---


# PHASE 11: Caching

## What We're Building

Cache LLM responses so identical (or similar) questions don't cost money
and time to re-generate. A cache hit returns in <1ms vs 2-10 seconds for an LLM call.

## Python Concepts

### 1. OrderedDict (LRU Cache Implementation)

```python
from collections import OrderedDict

class InMemoryLRUCache:
    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str):
        entry = self._cache.get(key)
        if entry:
            self._cache.move_to_end(key)  # Mark as recently used
        return entry

    def set(self, key: str, value):
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # Remove OLDEST (least recently used)
        self._cache[key] = value
```

**LRU = Least Recently Used.** When cache is full, evict the item that hasn't
been accessed for the longest time.

- `OrderedDict` remembers insertion order
- `move_to_end(key)` — moves item to the end (most recent)
- `popitem(last=False)` — removes from the beginning (oldest)

### 2. hashlib for Cache Keys

```python
import hashlib, json

def _make_key(self, messages: list[dict], model: str) -> str:
    payload = json.dumps({"messages": messages, "model": model}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

**Why hash?**
- Messages can be very long (thousands of chars) — bad as dict keys
- SHA-256 produces a fixed 64-char string regardless of input size
- `sort_keys=True` ensures same data always produces same hash
  (dict ordering shouldn't affect the key)

### 3. time.time() for TTL (Time-To-Live)

```python
@dataclass
class CacheEntry:
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds
```

`time.time()` returns seconds since epoch (Jan 1, 1970) as a float.
Subtracting two timestamps gives elapsed seconds.

**Why TTL?** LLM knowledge can become stale. A cached answer about "current weather"
should expire quickly. A cached answer about "what is Python" can live longer.

### 4. Cache Hit Rate Metric

```python
def stats(self) -> dict:
    total = self._hits + self._misses
    return {
        "hit_rate": self._hits / total if total > 0 else 0.0,
    }
```

**Hit rate** = what percentage of requests were served from cache.
- 0% = cache is useless (all misses)
- 80%+ = great (saving 80% of LLM calls)

**LangChain equivalent:**
```python
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
# Now all LLM calls are automatically cached
```

---


# PHASE 12: Guardrails

## What We're Building

Safety checks that run BEFORE the LLM (input guardrails) and AFTER the LLM
(output guardrails) to prevent PII leakage, prompt injection, and harmful content.

## Python Concepts

### 1. Regular Expressions for Pattern Matching

```python
import re

PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}

matches = re.findall(PATTERNS["email"], text)
redacted = re.sub(PATTERNS["email"], "[REDACTED_EMAIL]", text)
```

Key regex syntax:
- `\b` — word boundary (prevents matching inside longer numbers)
- `\d{3}` — exactly 3 digits
- `[-.]?` — optional dash or dot
- `[a-zA-Z]{2,}` — 2 or more letters
- `+` — one or more of the preceding

`re.findall()` — returns all matches as a list
`re.sub()` — replaces all matches with replacement string

### 2. Pipeline Pattern (Chain of Responsibility)

```python
class GuardrailPipeline:
    def __init__(self):
        self.input_guards = [
            PIIDetector(),
            PromptInjectionDetector(),
            ContentLengthGuard(),
        ]

    def check_input(self, content: str) -> GuardrailResult:
        for guard in self.input_guards:
            result = guard.check(content)
            if not result.passed:
                return result  # Short-circuit on first failure
        return GuardrailResult(passed=True)
```

Each guardrail in the pipeline:
1. Checks the content
2. If it fails → stop immediately, return the failure
3. If it passes → continue to next guardrail

**Short-circuit evaluation:** We don't run all guards if the first one blocks.
This saves processing time and follows the "fail fast" principle.

### 3. Enum for Actions

```python
class GuardrailAction(str, Enum):
    ALLOW = "allow"    # Let it through
    BLOCK = "block"    # Reject entirely
    REDACT = "redact"  # Clean it and continue
    FLAG = "flag"      # Allow but mark for review
```

Different violations need different responses:
- PII in input → REDACT (replace with [REDACTED], still process the request)
- Prompt injection → BLOCK (reject the request entirely)
- Mild policy violation → FLAG (process but alert a human)

### 4. re.search() vs re.findall()

```python
# re.search() — finds FIRST match, returns Match object or None
if re.search(r"ignore\s+previous\s+instructions", text):
    # Injection detected!

# re.findall() — finds ALL matches, returns list of strings
emails = re.findall(r"[\w.]+@[\w.]+", text)
# ["john@example.com", "jane@test.org"]
```

Use `search()` when you just need to know IF a pattern exists.
Use `findall()` when you need to extract ALL occurrences.

**LangChain equivalent:**
- No built-in guardrails in LangChain core
- NeMo Guardrails (NVIDIA) — separate library
- Guardrails AI — another option
- Custom `RunnablePassthrough` with validation logic

---


# PHASE 13: Observability (Tracing + Metrics + Cost Tracking)

## What We're Building

The ability to see what's happening inside the system — how long things take,
how much they cost, where errors occur, and how to trace a request through
all components.

## Python Concepts

### 1. Context Managers (with statement)

```python
from contextlib import contextmanager

class Tracer:
    @contextmanager
    def span(self, name: str, trace_id: str = None):
        s = Span(name=name, trace_id=trace_id)
        try:
            yield s          # Code inside 'with' block runs here
        except Exception as e:
            s.finish(status="error")
            raise            # Re-raise so caller sees the error
        else:
            s.finish(status="ok")
```

Usage:
```python
tracer = get_tracer()
with tracer.span("llm_call", trace_id="req-123") as span:
    response = provider.generate(messages)
    span.metadata["model"] = response.model
# span.finish() called automatically when exiting 'with' block
```

**Why context managers?**
- Guaranteed cleanup (finish is called even if exception occurs)
- Clean syntax (no try/finally boilerplate everywhere)
- Pairs well with tracing (start/end of an operation)

`@contextmanager` turns a generator function into a context manager:
- Code before `yield` = setup (start timing)
- `yield` = the 'with' block runs
- Code after `yield` = teardown (stop timing)

### 2. collections.defaultdict

```python
from collections import defaultdict

# Regular dict — KeyError if key doesn't exist
regular = {}
regular["count"] += 1  # KeyError!

# defaultdict — auto-creates missing keys
counters = defaultdict(int)    # Missing keys default to 0
counters["requests"] += 1      # Works! Creates key with 0, then adds 1

histograms = defaultdict(list)  # Missing keys default to []
histograms["latency"].append(150.5)  # Works!
```

**Why:** Metrics collection adds to many keys. Without defaultdict, you'd need
`if key not in dict: dict[key] = 0` everywhere.

### 3. Percentile Calculation

```python
def get_summary(self):
    sorted_v = sorted(values)
    n = len(sorted_v)
    return {
        "p50": sorted_v[int(n * 0.5)],   # Median
        "p95": sorted_v[int(n * 0.95)],  # 95th percentile
        "p99": sorted_v[int(n * 0.99)],  # 99th percentile
    }
```

**P50 (median):** Half of requests are faster than this.
**P95:** 95% of requests are faster. The remaining 5% are your "slow tail."
**P99:** Only 1% of requests are slower. This is what your worst users experience.

**Why P95/P99 matter more than average:**
- Average latency: 200ms (looks great!)
- P99 latency: 5000ms (1% of users wait 5 seconds — terrible!)

### 4. Cost Tracking with Pricing Tables

```python
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},   # per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 2.0})
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    return cost
```

`1_000_000` — Python allows underscores in numbers for readability.
Same as `1000000` but much easier to read.

**LangChain equivalent:**
```python
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = model.invoke("Hello")
    print(f"Cost: ${cb.total_cost}")
    print(f"Tokens: {cb.total_tokens}")
```

LangSmith provides full tracing with a web UI for exploring traces.

---


# PHASE 14: Rate Limiting & Multi-tenancy

## What We're Building

Protect the system from abuse (rate limiting) and support multiple customers
with different access levels (multi-tenancy).

## Python Concepts

### 1. Token Bucket Algorithm

```python
@dataclass
class TokenBucket:
    capacity: float       # Max tokens (burst size)
    refill_rate: float    # Tokens added per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)

    def consume(self, tokens: int = 1) -> bool:
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True   # Allowed
        return False      # Rate limited

    def _refill(self):
        elapsed = time.time() - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = time.time()
```

**How it works (analogy):**
- Imagine a bucket that holds 10 tokens (capacity)
- 1 token is added every second (refill_rate = 1/sec = 60 RPM)
- Each request consumes 1 token
- If bucket is empty → request rejected
- Bucket can accumulate up to 10 tokens → allows bursts of 10 rapid requests

**Why token bucket over simple counting?**
- Simple counter: "60 requests per minute" — but what if all 60 come in the first second?
- Token bucket: allows bursts (up to `capacity`) but smooths out over time

### 2. Dataclass with __post_init__

```python
@dataclass
class TokenBucket:
    capacity: float
    refill_rate: float
    tokens: float = 0.0

    def __post_init__(self):
        self.tokens = self.capacity  # Start with a full bucket
```

`__post_init__` runs after the auto-generated `__init__`. Use it for:
- Derived values (set tokens = capacity)
- Validation (raise if capacity < 0)
- Side effects (log creation)

### 3. Tuple Return for Multiple Values

```python
def check_access(self, tenant_id: str, provider: str) -> tuple[bool, str]:
    if not self._rate_limiter.is_allowed(tenant_id):
        return False, "Rate limit exceeded"
    if self._usage[tenant_id] >= config.monthly_budget_usd:
        return False, "Monthly budget exceeded"
    return True, "allowed"

# Usage:
allowed, reason = tenant_manager.check_access("tenant-123", "openai")
if not allowed:
    raise HTTPException(status_code=429, detail=reason)
```

Returning `tuple[bool, str]` gives both the decision AND the reason.
The caller can destructure with `allowed, reason = ...`.

### 4. Tier-Based Configuration (Data-Driven Design)

```python
TIER_DEFAULTS = {
    "free": TenantConfig(rate_limit_rpm=20, monthly_budget_usd=5.0, ...),
    "pro": TenantConfig(rate_limit_rpm=120, monthly_budget_usd=100.0, ...),
    "enterprise": TenantConfig(rate_limit_rpm=600, monthly_budget_usd=1000.0, ...),
}
```

**Data-driven design:** Instead of if/elif chains for each tier, store configs
in a dict. Adding a new tier = adding one dict entry, not modifying code.

### 5. .copy() for Mutable Defaults

```python
def register_tenant(self, tenant_id: str, tier: str):
    defaults = TIER_DEFAULTS[tier]
    config = TenantConfig(
        allowed_providers=defaults.allowed_providers.copy(),  # .copy()!
        allowed_models=defaults.allowed_models.copy(),
    )
```

**Why .copy()?** Without it, all tenants of the same tier would share the SAME list.
Modifying one tenant's allowed_providers would affect all others.

```python
# Without copy:
a = defaults.allowed_providers  # Same object as defaults!
a.append("new_provider")        # Modifies defaults too! Bug!

# With copy:
a = defaults.allowed_providers.copy()  # New independent list
a.append("new_provider")               # Only affects this tenant
```

**LangChain equivalent:** No built-in rate limiting or multi-tenancy.
This is typically handled at the infrastructure level (API Gateway, Kong, nginx)
or by LangGraph Platform (which has per-assistant limits).

---


# PHASE 15: Evaluation Framework

## What We're Building

Automated quality testing for the AI platform — run test cases, measure
correctness, latency, and cost, detect regressions when prompts change.

## Python Concepts

### 1. Callable as Parameter (Higher-Order Functions)

```python
class EvaluationRunner:
    def __init__(self, generate_fn: Optional[Callable] = None):
        self._generate_fn = generate_fn

    def _run_single(self, test_case: TestCase):
        output = self._generate_fn(test_case.input_message)
```

**Higher-order function:** A function that takes another function as an argument.
This lets the EvaluationRunner work with ANY generation function:

```python
# Test with the real platform:
runner = EvaluationRunner(generate_fn=lambda msg: chat_service.chat(msg).message)

# Test with a mock (for unit tests):
runner = EvaluationRunner(generate_fn=lambda msg: "mocked response")

# Test with a different model:
runner = EvaluationRunner(generate_fn=lambda msg: anthropic.generate(msg))
```

### 2. List Comprehensions with Conditions

```python
passed = sum(1 for r in results if r.passed)
latencies = [r.latency_ms for r in results if r.latency_ms > 0]
```

Breaking this down:
```python
# Generator expression (lazy, memory-efficient):
sum(1 for r in results if r.passed)
# Equivalent to:
count = 0
for r in results:
    if r.passed:
        count += 1

# List comprehension (creates a list):
[r.latency_ms for r in results if r.latency_ms > 0]
# Equivalent to:
latencies = []
for r in results:
    if r.latency_ms > 0:
        latencies.append(r.latency_ms)
```

### 3. Dict Comprehension for Aggregation

```python
avg_scores = {k: round(sum(v) / len(v), 3) for k, v in all_scores.items()}
```

Creates a new dict by transforming each key-value pair:
```python
# Input:  {"relevance": [0.8, 0.9, 1.0], "latency": [0.5, 0.7]}
# Output: {"relevance": 0.9, "latency": 0.6}
```

### 4. .setdefault() for Accumulating

```python
all_scores: dict[str, list[float]] = {}
for r in results:
    for metric, score in r.scores.items():
        all_scores.setdefault(metric, []).append(score)
```

`dict.setdefault(key, default)`:
- If key exists → return its value
- If key doesn't exist → set it to default, then return it

This is a one-liner for "create list if not exists, then append."

### 5. @property for Computed Values

```python
@dataclass
class EvalSuiteResult:
    total: int = 0
    passed: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0
```

`pass_rate` is computed from other fields — it's not stored, it's calculated
on access. Using `@property` makes it look like a regular attribute:

```python
result.pass_rate  # 0.85 (not result.pass_rate())
```

**Why not just a method?** Semantically, pass_rate IS a property of the result,
not an action. Properties express "what it is", methods express "what it does."

**LangChain equivalent:**
```python
from langsmith import evaluate

results = evaluate(
    my_agent,
    data="my-dataset",
    evaluators=[correctness_evaluator, relevance_evaluator],
)
# Returns experiment results with pass rates, scores, etc.
```

---


---

# SUMMARY: Python Concepts Covered

| Concept | Phase | Example |
|---------|-------|---------|
| Type hints / annotations | 1 | `def chat(request: ChatRequest) -> ChatResponse` |
| Pydantic BaseModel | 1 | Request/response validation |
| pydantic-settings | 1 | Environment-based configuration |
| @lru_cache | 1 | Singleton settings instance |
| Decorators (@router.post) | 1 | FastAPI endpoint registration |
| ABC / @abstractmethod | 2 | BaseLLM interface |
| @property | 2 | provider_name as read-only attribute |
| @dataclass | 2 | LLMResponse, LLMConfig |
| Optional[T] | 2 | Nullable fields |
| Factory pattern (importlib) | 2 | create_llm() dynamic imports |
| Inheritance / polymorphism | 3 | Multiple router implementations |
| field(default_factory) | 3 | Mutable defaults in dataclasses |
| Enum (str, Enum) | 3,6,8,12 | Fixed choice sets |
| Custom exceptions | 4 | RetryableError vs NonRetryableError |
| try/except/finally | 4 | Error handling with cleanup |
| time.sleep() + backoff | 4 | Exponential retry delays |
| State machines | 4 | Circuit breaker states |
| Generators (yield) | 5 | Streaming tokens |
| Generator[Y, S, R] type | 5 | Type-annotated generators |
| UUID generation | 6 | Session IDs |
| List slicing ([-20:]) | 6 | Context window trimming |
| sqlite3 module | 6 | Embedded database |
| Callable type | 7 | Functions as data (tool handlers) |
| **kwargs unpacking | 7 | Dynamic function calls |
| concurrent.futures | 7 | Timeout execution |
| Type[BaseModel] | 8 | Passing classes as arguments |
| @classmethod | 8 | Alternative constructors |
| model_json_schema() | 8 | Auto schema generation |
| model_validate() | 8 | Runtime validation |
| pathlib.Path | 9 | Cross-platform file paths |
| re (regex) | 9,12 | Pattern matching and extraction |
| global keyword | 9 | Module-level singletons |
| hashlib | 10,11 | Deterministic IDs and cache keys |
| zip() | 10 | Pairing parallel lists |
| Math (cosine similarity) | 10 | Vector search |
| OrderedDict | 11 | LRU cache implementation |
| time.time() for TTL | 11 | Cache expiration |
| Underscore in numbers | 13 | 1_000_000 readability |
| @contextmanager | 13 | Tracing spans |
| defaultdict | 13 | Auto-initializing counters |
| Percentile calculation | 13 | P50/P95/P99 metrics |
| __post_init__ | 14 | Post-construction logic |
| tuple returns | 14 | Multiple return values |
| .copy() for safety | 14 | Avoiding shared mutable state |
| Higher-order functions | 15 | Passing functions as arguments |
| List/dict comprehensions | 15 | Data transformation |
| .setdefault() | 15 | Accumulating into dicts |

---

# SUMMARY: Design Patterns Used

| Pattern | Phase | Purpose |
|---------|-------|---------|
| Layered Architecture | 1 | Separation of concerns (routes → services → providers) |
| Adapter | 2 | Normalize different provider APIs |
| Factory | 2 | Decouple object creation from usage |
| Strategy | 3 | Swap algorithms at runtime (routing) |
| Circuit Breaker | 4 | Prevent cascading failures |
| Observer | 5,13 | Decouple event production from consumption |
| Repository | 6 | Abstract data persistence |
| Registry | 7,9 | Central lookup for tools/prompts |
| Template Method | 7 | Default behavior in base, override in subclass |
| Pipeline / Chain of Responsibility | 12 | Sequential checks with short-circuit |
| Token Bucket | 14 | Smooth rate limiting |
| Data-Driven Design | 14 | Config in data structures, not code |

---

# SUMMARY: LangChain/LangGraph Complete Mapping

```
┌─────────────────────────────────────────────────────────────────┐
│                    OUR PLATFORM                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  BaseLLM ─────────────────────→ BaseChatModel                    │
│  create_llm("openai") ────────→ init_chat_model("openai:gpt-4o")│
│  provider.generate() ─────────→ model.invoke()                   │
│  provider.stream() ───────────→ model.stream()                   │
│  generate_with_tools() ───────→ model.bind_tools().invoke()      │
│                                                                   │
│  ToolRegistry + ToolExecutor ─→ ToolNode / create_react_agent()  │
│  Tool execution loop ─────────→ LangGraph agent loop             │
│  MAX_TOOL_ITERATIONS ─────────→ recursion_limit                  │
│                                                                   │
│  StructuredOutputService ─────→ model.with_structured_output()   │
│  OutputSchema.from_pydantic() → Pass Pydantic class directly     │
│                                                                   │
│  PromptTemplate ──────────────→ ChatPromptTemplate               │
│  PromptRegistry ──────────────→ LangChain Hub                    │
│                                                                   │
│  ConversationService ─────────→ InMemoryChatMessageHistory       │
│  SQLiteStore ─────────────────→ SqliteSaver (LangGraph)          │
│  Context window trimming ─────→ trim_messages()                  │
│                                                                   │
│  RetrievalService ────────────→ create_retrieval_chain()         │
│  Chunker ─────────────────────→ RecursiveCharacterTextSplitter   │
│  InMemoryVectorStore ─────────→ FAISS / Chroma                   │
│                                                                   │
│  ResponseCache ───────────────→ set_llm_cache(InMemoryCache())   │
│  FallbackManager ─────────────→ model.with_fallbacks()           │
│  RetryHandler ────────────────→ model.with_retry()               │
│                                                                   │
│  GuardrailPipeline ───────────→ NeMo Guardrails                  │
│  Tracer ──────────────────────→ LangSmith tracing                │
│  CostTracker ─────────────────→ get_openai_callback()            │
│  EvaluationRunner ────────────→ langsmith.evaluate()             │
│  TenantManager ───────────────→ LangGraph Platform               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

# Key Takeaway

> **LangChain/LangGraph is ONE layer in a production AI system.**
> 
> It handles: model abstraction, tool calling, chains, agents, memory.
> 
> It does NOT handle: rate limiting, circuit breakers, caching, guardrails,
> cost tracking, multi-tenancy, evaluation, deployment, monitoring.
> 
> By building all 15 phases from scratch, you now understand:
> 1. What every framework does under the hood
> 2. When to use a framework vs build custom
> 3. How to architect a system that can swap ANY component
> 4. The 80-90% of engineering that surrounds the LLM call

```
Junior:  "It works in my notebook"
Mid:     "It works with error handling and retries"
Senior:  "It works at scale, recovers from failures,
          is observable, testable, cost-efficient,
          and I can swap any component without
          touching the rest"
```
