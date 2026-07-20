# Learning Notes — Doc QA Agent
### For beginners who want to learn Python + LangChain + LangGraph through this project

---

## SECTION 1 — How to Read This Project

This project is a real production-grade AI system. If you are a beginner, do not try to read
all files at once. Follow this order:

```
1. app/config/settings.py       ← simplest file, learn: classes, types, env vars
2. app/agents/state.py          ← learn: TypedDict, type hints, data modeling
3. app/ingestion/extractor.py   ← learn: functions, imports, file handling
4. app/memory/session_store.py  ← learn: async/await, Redis, global variables
5. app/memory/vector_store.py   ← learn: embeddings, ChromaDB, chunking
6. app/observability/tracer.py  ← learn: decorators, logging, context managers
7. app/agents/nodes.py          ← learn: LangChain, LLM calls, Pydantic models
8. app/agents/graph.py          ← learn: LangGraph, StateGraph, conditional edges
9. app/api/auth.py              ← learn: FastAPI dependencies
10. app/api/routes.py           ← learn: REST API, background tasks, streaming
11. app/main.py                 ← learn: app startup, middleware, lifespan
```

### What this project does (plain English)
- User uploads a PDF or Word document
- The system reads everything in it — text, tables, images, graphs
- User asks a question about the document
- An AI agent answers it
- If the document is still in the session (uploaded recently) → answer directly from the doc
- If the session expired → answer from the vector store (long-term memory)
- Multiple users can use this at the same time, each seeing only their own data

---

## SECTION 2 — Python Basics: Variables, Types, Type Hints

### Variables
A variable is just a name that holds a value.

```python
name = "Alice"          # str  — text
age = 30                # int  — whole number
score = 0.95            # float — decimal number
is_active = True        # bool — True or False
```

### Type Hints (used everywhere in this project)
Python does not force you to declare types, but you CAN annotate them.
This makes code readable and catches bugs early with tools like mypy.

```python
# Without type hints
def greet(name):
    return "Hello " + name

# With type hints — much clearer
def greet(name: str) -> str:
    return "Hello " + name
```

In this project you will see this pattern constantly:

```python
# from app/config/settings.py
openai_model: str = "gpt-4o"          # type: str, default: "gpt-4o"
openai_max_tokens: int = 2048         # type: int, default: 2048
langsmith_tracing: bool = False       # type: bool, default: False
```

### None and Optional
`None` means "no value". `str | None` means "either a string or nothing".

```python
# from app/agents/state.py
doc_id: str | None       # doc_id can be a string OR None (not yet set)
rewritten_query: str | None = None
```

### Why type hints matter in production
- Your editor (VS Code) shows errors before you run the code
- Other developers instantly know what a function expects
- Pydantic (used heavily here) uses them to validate data automatically

---

## SECTION 3 — Python Classes and Objects

### What is a class?
A class is a blueprint. An object is a thing built from that blueprint.

```python
# Blueprint
class Dog:
    def __init__(self, name: str):   # __init__ runs when you create the object
        self.name = name             # self.name stores the value on the object

    def bark(self) -> str:
        return f"{self.name} says woof!"

# Object (instance of the class)
dog = Dog("Rex")
print(dog.bark())    # Rex says woof!
```

### Classes in this project

#### 1. Settings class (app/config/settings.py)
```python
class Settings(BaseSettings):        # inherits from BaseSettings (Pydantic)
    openai_api_key: str              # required — no default, must be in .env
    openai_model: str = "gpt-4o"    # optional — has a default value
    debug: bool = False
```
`BaseSettings` is a class from the `pydantic-settings` library.
By inheriting from it, our `Settings` class automatically reads values from the `.env` file.
This is called **inheritance** — your class gets all the features of the parent class for free.

#### 2. Pydantic BaseModel (app/agents/nodes.py)
```python
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    query_type: Literal["factual", "summary", "table", "image", "general"]
    refined_query: str = Field(description="Cleaned version of the user query")
```
`BaseModel` gives you:
- Automatic data validation (wrong type → error immediately)
- JSON serialization for free
- Used here to force the LLM to return structured output

#### 3. JSONFormatter class (app/observability/tracer.py)
```python
class JSONFormatter(logging.Formatter):    # inherits from logging.Formatter
    def format(self, record: logging.LogRecord) -> str:
        log = {"ts": ..., "level": ..., "msg": ...}
        return json.dumps(log)             # override the parent's format method
```
This overrides the `format` method from the parent class to output JSON instead of plain text.
Overriding a parent method is called **polymorphism**.

### Key OOP concepts used in this project

| Concept | Where used | What it does |
|---|---|---|
| Inheritance | `Settings(BaseSettings)` | Gets env-reading for free |
| Inheritance | `JSONFormatter(logging.Formatter)` | Gets logging infrastructure for free |
| Inheritance | `QueryAnalysis(BaseModel)` | Gets validation + serialization for free |
| Override | `JSONFormatter.format()` | Replaces parent behavior with custom JSON output |
| Instance | `settings = get_settings()` | Creates one Settings object, reused everywhere |

---

## SECTION 4 — TypedDict, Literal, Annotated

These three are used in `app/agents/state.py` to model the agent's state.

### TypedDict — a dictionary with known keys and types

A regular Python dict can hold anything:
```python
data = {"name": "Alice", "age": 30}   # no type safety
data["age"] = "thirty"                # Python allows this — bug waiting to happen
```

A `TypedDict` enforces what keys exist and what types they hold:
```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

p: Person = {"name": "Alice", "age": 30}   # correct
p: Person = {"name": "Alice", "age": "30"} # type checker flags this as error
```

In this project, `AgentState` is a `TypedDict` — it is the single object that flows
through every node in the LangGraph graph. Every node reads from it and writes back to it.

```python
# from app/agents/state.py
class AgentState(TypedDict):
    user_id: str
    query: str
    retrieval_path: Literal["live_context", "rag", ""]
    retry_count: int
    fallback: bool
    # ... more fields
```

### Literal — restrict a value to specific options only

```python
from typing import Literal

# This field can ONLY be one of these three strings
retrieval_path: Literal["live_context", "rag", ""]

# This field can ONLY be one of these two strings
grading_score: Literal["good", "poor", ""]
```

If you try to set `retrieval_path = "something_else"`, the type checker will catch it.
This prevents entire categories of bugs where a wrong string silently causes wrong behavior.

### Annotated — attach extra metadata to a type

```python
from typing import Annotated
from operator import add

# This tells LangGraph: when merging state updates,
# ADD new items to the list instead of replacing it
retrieved_chunks: Annotated[list[str], add]
```

This is a LangGraph-specific pattern. When multiple nodes update `retrieved_chunks`,
LangGraph uses the `add` function to merge them (list concatenation) instead of overwriting.
Without `Annotated[list[str], add]`, the second node's update would erase the first node's chunks.

### Why model state this way?
- Every node in the graph receives the same `AgentState` dict
- Nodes return only the keys they changed — LangGraph merges the rest automatically
- TypedDict makes it impossible to accidentally read a key that doesn't exist
- The whole graph's data flow is visible in one file (`state.py`)

---

## SECTION 5 — Functions, async/await, Coroutines

### Regular functions
```python
def add(a: int, b: int) -> int:
    return a + b

result = add(2, 3)   # blocks here until done, then moves on
```

### The problem with blocking code in a web server
Imagine 100 users upload documents at the same time.
Each upload takes 3 seconds to process.
With regular (blocking) functions, user 2 waits for user 1 to finish, user 3 waits for user 2...
Total wait for user 100 = 300 seconds. That is unusable.

### async/await — non-blocking execution
`async def` defines a coroutine — a function that can pause and let other things run while waiting.

```python
import asyncio

async def fetch_data() -> str:
    await asyncio.sleep(3)    # "await" = pause here, let other coroutines run
    return "data"

async def main():
    result = await fetch_data()
    print(result)
```

### How this project uses async

Every node function is `async def`:
```python
# from app/agents/nodes.py
async def analyze_query(state: AgentState) -> dict:
    result = await _llm().with_structured_output(QueryAnalysis).ainvoke([...])
    #        ^^^^^ wait for OpenAI API response without blocking other users
    return {"query_type": result.query_type}
```

Every Redis and ChromaDB call is awaited:
```python
# from app/memory/session_store.py
async def get_doc_payload(user_id: str, session_id: str) -> DocPayload | None:
    r = get_redis()
    raw = await r.get(_doc_key(user_id, session_id))   # non-blocking Redis read
    return json.loads(raw) if raw else None
```

### The `await` keyword
- You can only use `await` inside an `async def` function
- `await` means: "start this operation, but don't block — let other coroutines run while waiting"
- When the operation finishes, come back here and continue

### async vs sync — when to use which
| Use `async def` | Use regular `def` |
|---|---|
| Calling external APIs (OpenAI, Redis, ChromaDB) | Pure computation (math, string manipulation) |
| Reading/writing files | Building data structures |
| Database queries | Conditional logic |
| Anything that waits for I/O | Helper/utility functions |

In this project, `extract()` in `extractor.py` is a regular `def` because it does
CPU-bound file parsing, not I/O waiting. Everything that touches a network is `async`.

---

## SECTION 6 — Decorators

Decorators are one of the most powerful Python features. They look intimidating at first
but the idea is simple: a decorator wraps a function to add behavior before/after it runs.

### The simplest decorator
```python
def shout(fn):                      # decorator takes a function as input
    def wrapper(*args, **kwargs):   # wrapper replaces the original function
        result = fn(*args, **kwargs)
        print("DONE!")              # extra behavior added after
        return result
    return wrapper                  # return the new wrapped function

@shout                              # apply the decorator
def greet(name: str) -> str:
    return f"Hello {name}"

greet("Alice")
# Hello Alice
# DONE!
```

`@shout` above `greet` is exactly the same as writing `greet = shout(greet)`.
The `@` syntax is just cleaner.

### Decorator with arguments
When a decorator needs its own arguments, you add one more layer:

```python
def repeat(times: int):             # outer function takes decorator args
    def decorator(fn):              # middle function takes the function
        def wrapper(*args, **kwargs):
            for _ in range(times):
                fn(*args, **kwargs)
        return wrapper
    return decorator

@repeat(times=3)
def say_hi():
    print("hi")

say_hi()   # prints "hi" three times
```

### The `track_node` decorator in this project

This is the real decorator from `app/observability/tracer.py`:

```python
def track_node(node_name: str):          # takes node_name as argument
    def decorator(fn):                   # receives the actual node function
        @wraps(fn)                       # preserves original function's name/docs
        async def wrapper(state: dict, *args, **kwargs):
            start = time.perf_counter()          # record start time
            result = await fn(state, *args, **kwargs)  # run the actual node
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            agent_logger.info(f"node:{node_name}", extra={
                "extra": {
                    "node": node_name,
                    "user_id": state.get("user_id"),
                    "latency_ms": latency_ms,
                }
            })
            return result
        return wrapper
    return decorator
```

Used on every node like this:
```python
@track_node("analyze_query")           # wraps the function with timing + logging
async def analyze_query(state: AgentState) -> dict:
    ...
```

Every time `analyze_query` runs, the decorator automatically:
1. Records the start time
2. Runs the actual node function
3. Calculates how long it took
4. Logs the node name, user_id, session_id, latency — without the node function knowing

This is the power of decorators: **add cross-cutting behavior without touching the original code**.

### `@wraps(fn)` — why it matters
Without `@wraps`, the wrapped function loses its original name and docstring.
```python
print(analyze_query.__name__)   # without @wraps → "wrapper"
print(analyze_query.__name__)   # with @wraps    → "analyze_query"
```
Always use `@wraps` when writing decorators.

### `@lru_cache` — another decorator in this project
```python
# from app/config/settings.py
@lru_cache
def get_settings() -> Settings:
    return Settings()
```
`lru_cache` is a built-in Python decorator. It caches the return value of the function.
The first call creates the `Settings` object. Every call after that returns the same cached object.
This means `.env` is only read once, no matter how many times `get_settings()` is called.

### `@asynccontextmanager` — decorator for context managers
```python
# from app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.upload_dir).mkdir(exist_ok=True)   # runs on startup
    yield                                             # app runs here
    logger.info("shutdown")                           # runs on shutdown
```
This decorator turns a generator function into a context manager.
The code before `yield` runs when the app starts. The code after `yield` runs when it stops.

---

## SECTION 7 — Imports, Modules, Packages

### What is a module?
Every `.py` file is a module. You import from it using its file path with dots.

```
app/
├── config/
│   └── settings.py      ← module: app.config.settings
├── agents/
│   └── nodes.py         ← module: app.agents.nodes
└── memory/
    └── vector_store.py  ← module: app.memory.vector_store
```

### Import styles

```python
# Import the whole module
import json
json.dumps({"key": "value"})

# Import specific names from a module
from pathlib import Path
from typing import TypedDict, Literal

# Import with alias (rename to avoid conflicts or shorten)
import redis.asyncio as aioredis
from PIL import Image as PILImage    # PIL's Image conflicts with unstructured's Image
```

### Why `PIL.Image as PILImage`?
In `extractor.py` you see:
```python
from unstructured.documents.elements import Image   # unstructured's Image element
from PIL import Image as PILImage                   # PIL's Image for processing
```
Both libraries have a class called `Image`. Aliasing one avoids the name collision.

### `__init__.py` — what is it?
Every folder with an `__init__.py` file is a **package** — Python treats it as importable.
Without `__init__.py`, Python cannot import from that folder.

```
app/
├── __init__.py          ← makes "app" a package
├── agents/
│   ├── __init__.py      ← makes "app.agents" a package
│   └── nodes.py
```

The `__init__.py` files in this project are empty — they just signal "this is a package".
In larger projects, `__init__.py` can re-export things to simplify imports.

### Relative vs absolute imports
This project uses absolute imports throughout — always starting from `app`:
```python
from app.agents.state import AgentState          # absolute — always works
from app.config.settings import get_settings     # absolute — clear and explicit
```

Relative imports (using dots) also exist but are harder to read:
```python
from ..config.settings import get_settings       # relative — avoid in production code
```

### Circular imports — a common beginner mistake
If `nodes.py` imports from `graph.py` AND `graph.py` imports from `nodes.py`, Python crashes.
This project avoids this by having a clear one-way dependency chain:

```
settings.py  ←  state.py  ←  nodes.py  ←  graph.py  ←  routes.py  ←  main.py
```
Each file only imports from files to its left. Never circular.

---

## SECTION 8 — Global Variables, Singleton Pattern, Lazy Initialization

### Global variables
A variable defined at the top of a module (outside any function or class) is a global variable.
It is shared across all code that imports that module.

```python
# from app/memory/session_store.py
_redis: aioredis.Redis | None = None    # global, starts as None
```

The underscore prefix `_redis` is a Python convention meaning "private to this module —
don't import this directly from outside".

### Lazy initialization — create only when first needed
```python
def get_redis() -> aioredis.Redis:
    global _redis                        # tell Python we mean the module-level variable
    if _redis is None:                   # first call: create it
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis                        # all calls: return the existing one
```

This pattern is called **lazy initialization**:
- The Redis connection is NOT created when the module loads
- It is created on the first call to `get_redis()`
- Every call after that reuses the same connection

### Why not just create it at module level?
```python
# BAD — runs at import time, before settings are loaded
_redis = aioredis.from_url(settings.redis_url)
```
If `settings.redis_url` is not yet available when the module is imported, this crashes.
Lazy initialization defers creation until the value is actually needed.

### Singleton pattern
The same pattern is used for ChromaDB and embeddings:
```python
# from app/memory/vector_store.py
_chroma_client: chromadb.AsyncHttpClient | None = None
_embeddings: OpenAIEmbeddings | None = None

def get_chroma() -> chromadb.AsyncHttpClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.AsyncHttpClient(...)
    return _chroma_client
```

This ensures only ONE ChromaDB client exists for the entire application lifetime.
Creating a new client on every request would be wasteful and slow.

### `@lru_cache` as a cleaner singleton
```python
# from app/config/settings.py
@lru_cache
def get_settings() -> Settings:
    return Settings()
```
`@lru_cache` does the same thing as the `if None` pattern but in one line.
The first call runs `Settings()`. Every subsequent call returns the cached result.
This is the preferred way for simple singletons in Python.

---

## SECTION 9 — Context Managers and the `with` Statement

### What is a context manager?
A context manager is an object that sets something up before a block of code runs
and tears it down after — even if an error occurs.

The most common example is file handling:
```python
# Without context manager — risky, file may not close if error occurs
f = open("file.txt")
data = f.read()
f.close()           # what if read() raises an exception? close() never runs!

# With context manager — file always closes, even on error
with open("file.txt") as f:
    data = f.read()
# file is automatically closed here
```

### How context managers work internally
A class becomes a context manager by implementing `__enter__` and `__exit__`:
```python
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):          # runs even if an exception occurred
        elapsed = time.perf_counter() - self.start
        print(f"Took {elapsed:.2f}s")

with Timer():
    time.sleep(1)
# Took 1.00s
```

### `@contextmanager` — create one from a generator function
Instead of writing a class, you can use the `@contextmanager` decorator:

```python
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    try:
        yield                           # code inside `with` block runs here
    finally:
        elapsed = time.perf_counter() - start
        print(f"Took {elapsed:.2f}s")

with timer():
    time.sleep(1)
```

### `trace_span` in this project
```python
# from app/observability/tracer.py
@contextmanager
def trace_span(name: str, metadata: dict | None = None):
    start = time.perf_counter()
    try:
        yield                           # the wrapped code runs here
    finally:
        ms = round((time.perf_counter() - start) * 1000, 2)
        agent_logger.info(f"span:{name}", extra={"extra": {"span": name, "latency_ms": ms}})
```

Used in routes.py like this:
```python
with trace_span("doc_extraction", {"user_id": user_id}):
    payload = extract(upload_path)      # this runs between start and finally
# after this line, latency is logged automatically
```

The `finally` block guarantees the log is written even if `extract()` raises an exception.

### `@asynccontextmanager` — async version
```python
# from app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.upload_dir).mkdir(exist_ok=True)   # startup
    yield                                             # app runs
    logger.info("shutdown")                           # shutdown
```
Same idea as `@contextmanager` but for async code. Used by FastAPI to manage app lifecycle.

---

## SECTION 10 — List Comprehensions, Generators, zip, filter

These are used throughout the project to process data concisely.

### List comprehension — build a list in one line
```python
# Regular loop
squares = []
for i in range(5):
    squares.append(i * i)

# List comprehension — same result, one line
squares = [i * i for i in range(5)]   # [0, 1, 4, 9, 16]
```

With a condition:
```python
evens = [i for i in range(10) if i % 2 == 0]   # [0, 2, 4, 6, 8]
```

### In this project — building sources list (nodes.py)
```python
sources = [
    {
        "doc_name": c["metadata"]["doc_name"],
        "type": c["metadata"]["type"],
        "score": c["score"],
    }
    for c in chunks          # for each chunk in the retrieved chunks list
]
```
This builds a list of dicts from a list of chunks in one expression.

### Generator expression — lazy list comprehension
A generator does NOT build the whole list in memory. It produces values one at a time.
```python
# List — all values in memory at once
names = [c["metadata"]["doc_name"] for c in chunks]

# Generator — values produced one at a time (memory efficient)
names = (c["metadata"]["doc_name"] for c in chunks)
```

Used in `format_response` node:
```python
confidence = max((s["score"] for s in sources), default=0.5)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                generator expression passed to max()
```

### `filter()` — keep only items that pass a condition
```python
# from app/ingestion/extractor.py
"\n\n".join(filter(None, text_parts))
```
`filter(None, text_parts)` removes all falsy values (empty strings, None) from `text_parts`.
Then `join` combines the remaining parts with double newlines between them.

### `zip()` — iterate multiple lists together
```python
# from app/memory/vector_store.py
for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    chunks.append({"content": doc, "metadata": meta, "score": round(1 - dist, 4)})
```
`zip` pairs up items from three lists by index:
- `results["documents"][0][0]` pairs with `results["metadatas"][0][0]` and `results["distances"][0][0]`
- `results["documents"][0][1]` pairs with `results["metadatas"][0][1]` and `results["distances"][0][1]`
- ...and so on

### `enumerate()` — loop with index
```python
# from app/memory/vector_store.py
for i, chunk in enumerate(_chunk_text(payload["text"])):
    metadatas.append({"chunk_index": i, ...})
```
`enumerate` gives you both the index `i` and the value `chunk` in each iteration.

---

## SECTION 11 — LangChain: LLMs, Messages, Structured Output, Embeddings

LangChain is a framework that makes it easy to work with LLMs (Large Language Models like GPT-4o).
Without LangChain, you would write raw HTTP requests to OpenAI's API.
LangChain wraps all of that into clean Python objects.

### ChatOpenAI — the LLM client
```python
# from app/agents/nodes.py
from langchain_openai import ChatOpenAI

def _llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,      # 0 = deterministic, 1 = creative/random
        max_tokens=2048,
    )
```
`ChatOpenAI` is a LangChain class that wraps the OpenAI API.
`temperature=0.0` means the model always picks the most likely next token — good for factual answers.

### Messages — how you talk to the LLM
LLMs work with a conversation format. Every message has a role:

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),   # sets behavior
    HumanMessage(content="What is the capital of France?"),  # user's question
]

response = await llm.ainvoke(messages)
print(response.content)   # "Paris"
```

- `SystemMessage` — instructions to the model (how to behave)
- `HumanMessage` — the user's input
- `AIMessage` — the model's response (returned by `ainvoke`)

### Multimodal messages — text + images
GPT-4o can understand images. You pass them as a list inside `HumanMessage`:

```python
# from app/agents/nodes.py
content = [
    {"type": "text", "text": "What does this chart show?"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
]
response = await _llm().ainvoke([
    SystemMessage(content="You are a document analyst."),
    HumanMessage(content=content),   # content is a list, not a string
])
```

### Structured output — force the LLM to return a specific format
Without structured output, the LLM returns free text. You cannot reliably parse it.
With structured output, you define a Pydantic model and the LLM fills it in:

```python
from pydantic import BaseModel
from typing import Literal

class GradeResult(BaseModel):
    score: Literal["good", "poor"]
    reason: str

# .with_structured_output() tells LangChain to enforce this schema
result: GradeResult = await _llm().with_structured_output(GradeResult).ainvoke([
    SystemMessage(content="Grade this answer..."),
    HumanMessage(content="..."),
])

print(result.score)    # always "good" or "poor" — never anything else
print(result.reason)   # always a string
```

Under the hood, LangChain uses OpenAI's function calling / JSON mode to enforce the schema.

### Embeddings — turning text into numbers
An embedding is a list of numbers (a vector) that represents the meaning of a text.
Texts with similar meaning have vectors that are close together in space.

```python
# from app/memory/vector_store.py
from langchain_openai import OpenAIEmbeddings

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Convert text to a vector
vector = await embedder.aembed_query("What is the revenue?")
# vector = [0.023, -0.145, 0.891, ...]  (1536 numbers)
```

This is how semantic search works:
1. At upload time: embed every chunk of the document → store vectors in ChromaDB
2. At query time: embed the user's question → find the closest vectors in ChromaDB
3. Return the chunks whose vectors are closest to the question vector

### Why LangChain instead of raw OpenAI SDK?
| Raw OpenAI SDK | LangChain |
|---|---|
| Manual HTTP requests | Clean Python objects |
| Manual message formatting | `HumanMessage`, `SystemMessage` classes |
| Manual JSON parsing for structured output | `.with_structured_output(MyModel)` |
| No built-in retry/fallback | Built-in retry, fallback, streaming |
| No tracing | Native LangSmith integration |
| Vendor-locked to OpenAI | Swap to Anthropic/Gemini by changing one line |

---

## SECTION 12 — LangGraph: StateGraph, Nodes, Edges, Conditional Edges

LangGraph is built on top of LangChain. It lets you build AI agents as graphs —
where each step is a node and the connections between steps are edges.

### Why a graph instead of just calling functions in order?
With plain functions:
```python
result1 = analyze_query(state)
result2 = check_session(state)
# ... but what if check_session needs to branch?
# ... what if grade_answer needs to loop back?
# ... what if you need retries?
```
You end up with complex if/else chains that are hard to read, test, and visualize.

A graph makes the flow explicit, visual, and easy to modify.

### The three building blocks

#### 1. State — the shared data object
```python
# from app/agents/state.py
class AgentState(TypedDict):
    query: str
    retrieval_path: str
    raw_answer: str
    grading_score: str
    # ...
```
Every node receives the full state and returns only the keys it changed.
LangGraph merges the changes automatically.

#### 2. Nodes — the steps
```python
# from app/agents/graph.py
graph = StateGraph(AgentState)

graph.add_node("analyze_query", analyze_query)       # name → function
graph.add_node("check_session", check_session)
graph.add_node("grade_answer", grade_answer)
```
Each node is just an async function that takes state and returns a dict of updates.

#### 3. Edges — the connections
```python
# Simple edge: always go from A to B
graph.add_edge("analyze_query", "check_session")

# Conditional edge: go to different nodes based on state
graph.add_conditional_edges(
    "check_session",          # from this node
    route_by_session,         # call this function to decide where to go
    {
        "live_context": "answer_from_live_context",   # if returns "live_context"
        "rag": "answer_from_rag",                     # if returns "rag"
    },
)
```

The routing function just reads the state and returns a string:
```python
def route_by_session(state: AgentState) -> str:
    return state["retrieval_path"]   # "live_context" or "rag"
```

### The full graph in this project
```
START
  │
  ▼
analyze_query ──────────────────────────────► check_session
                                                    │
                              ┌─────────────────────┴──────────────────────┐
                              │ "live_context"                  "rag"       │
                              ▼                                  ▼          │
                  answer_from_live_context          answer_from_rag         │
                              │                                  │          │
                              └──────────────┬───────────────────┘          │
                                             ▼                              │
                                       grade_answer                         │
                                             │                              │
                          ┌──────────────────┼──────────────────┐          │
                          │ "good"           │ "retry"          │ "fallback"│
                          ▼                  ▼                  ▼          │
                  store_to_vectorstore  rewrite_query    fallback_response  │
                          │                  │                  │          │
                          ▼                  └──► (back to      ▼          │
                  format_response             retrieval)       END         │
                          │                                                 │
                         END                                                │
```

### `graph.compile()` — what does it do?
```python
qa_graph = graph.compile()
```
Compiling validates the graph (checks for missing nodes, unreachable nodes, etc.)
and returns a `CompiledGraph` object that you can invoke.

### Invoking the graph
```python
# from app/api/routes.py

# Synchronous invoke — waits for full result
result = await qa_graph.ainvoke(initial_state)

# Streaming invoke — yields events as they happen
async for event in qa_graph.astream_events(initial_state, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content)
```

### Why LangGraph instead of plain Python?
| Plain Python | LangGraph |
|---|---|
| Manual if/else branching | Declarative conditional edges |
| Hard to add retries | Built-in retry loops via graph cycles |
| No visualization | Graph can be visualized with `.get_graph().draw_mermaid()` |
| No streaming per-node | `astream_events` gives per-node token streaming |
| No checkpointing | Built-in state persistence across sessions |
| Hard to debug | LangSmith shows every node's input/output |

---

## SECTION 13 — FastAPI: Routes, Depends, Background Tasks, Streaming

FastAPI is the web framework that exposes the agent as an HTTP API.

### What is a web framework?
When a user sends a request to `POST /upload`, something needs to:
1. Receive the HTTP request
2. Parse the file from the request body
3. Call your Python code
4. Return an HTTP response

FastAPI handles all of that. You just write the Python function.

### Defining a route
```python
# from app/api/routes.py
from fastapi import APIRouter
router = APIRouter()

@router.post("/upload")                    # HTTP POST to /upload
async def upload_document(...):
    ...
    return UploadResponse(...)
```

`@router.post("/upload")` is a decorator that registers this function as the handler
for `POST /upload` requests.

### Pydantic models for request/response validation
```python
class QueryRequest(BaseModel):
    query: str
    session_id: str
    doc_id: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float
    retrieval_path: str
    session_id: str

@router.post("/query", response_model=QueryResponse)
async def query_document(req: QueryRequest, ...) -> QueryResponse:
    # FastAPI automatically:
    # - parses the JSON request body into QueryRequest
    # - validates all fields (wrong type → 422 error automatically)
    # - serializes the return value using QueryResponse schema
```

### `Depends` — dependency injection
```python
# from app/api/routes.py
from fastapi import Depends

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),   # ← dependency injection
):
```

`Depends(get_current_user)` tells FastAPI: "before calling this function,
call `get_current_user` and pass its return value as `user_id`".

`get_current_user` reads the `X-User-ID` header and returns the user ID.
If the header is missing, it raises an exception and the route never runs.

This pattern means auth logic is written once and reused across all routes.

### Background tasks — respond immediately, work later
```python
# from app/api/routes.py
from fastapi import BackgroundTasks

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    ...
):
    # ... extract document, store in Redis ...

    # Schedule vectorization to run AFTER the response is sent
    background_tasks.add_task(_vectorize_background, user_id, doc_id, session_id)

    return UploadResponse(...)   # user gets this immediately
    # _vectorize_background runs after this return
```

The user gets their `session_id` and `doc_id` instantly.
ChromaDB ingestion happens in the background — they don't wait for it.

### Streaming responses — SSE (Server-Sent Events)
```python
# from app/api/routes.py
from fastapi.responses import StreamingResponse

@router.get("/stream/{session_id}")
async def stream_query(...):
    async def event_generator():
        async for event in qa_graph.astream_events(initial_state, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"data: {chunk}\n\n"   # SSE format
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

SSE (Server-Sent Events) is a protocol where the server keeps the connection open
and pushes data as it becomes available. The client sees tokens appearing word by word,
like ChatGPT's streaming effect.

The `async def event_generator()` is an **async generator** — a function that uses
`yield` inside an `async def`. Each `yield` sends one chunk to the client.

### HTTPException — return error responses
```python
from fastapi import HTTPException

if file.content_type not in ALLOWED_TYPES:
    raise HTTPException(status_code=415, detail="Unsupported file type")
```
FastAPI catches `HTTPException` and converts it to a proper HTTP error response with
the status code and detail message as JSON.

---

## SECTION 14 — Redis and ChromaDB: Why These Tools?

### Redis — fast key-value store (session memory)

#### What is Redis?
Redis stores data as key → value pairs in memory (RAM), making it extremely fast.
Think of it as a Python dictionary that lives outside your application,
survives restarts, and can be shared across multiple server instances.

```python
# Conceptually, Redis is like this:
store = {
    "session:user_123:abc-456:doc": '{"text": "...", "tables": [...]}',
    "session:user_123:abc-456:history": '[{"role": "user", "content": "..."}]',
}
```

#### Why Redis for sessions?
| Option | Problem |
|---|---|
| Python dict in memory | Lost on server restart, not shared across multiple servers |
| Database (PostgreSQL) | Too slow for frequent reads/writes, overkill for temporary data |
| Redis | Fast, supports TTL (auto-expiry), shared across servers |

#### TTL — Time To Live
```python
await r.setex(
    key,
    3600,          # TTL: 3600 seconds = 1 hour
    json.dumps(payload),
)
```
After 1 hour, Redis automatically deletes the key. No cleanup code needed.
This is how session expiry works — the document disappears from the session after 1 hour.

#### Key namespacing for isolation
```python
def _doc_key(user_id: str, session_id: str) -> str:
    return f"session:{user_id}:{session_id}:doc"
```
User A's key: `session:user_A:abc:doc`
User B's key: `session:user_B:xyz:doc`
They can never collide. User A cannot accidentally read User B's data.

---

### ChromaDB — vector database (long-term memory)

#### What is a vector database?
A regular database searches by exact match: `WHERE name = "Alice"`.
A vector database searches by similarity: "find documents whose meaning is closest to this query".

It stores vectors (lists of numbers representing meaning) and finds the nearest ones efficiently.

#### Why ChromaDB?
| Option | When to use |
|---|---|
| ChromaDB | Local development, small-medium scale, easy Docker setup |
| Qdrant | Production, high performance, more filtering options |
| Pinecone | Fully managed cloud, no infrastructure to manage |
| pgvector | Already using PostgreSQL, want vectors in same DB |

ChromaDB is chosen here because it runs locally with zero configuration,
has a simple Python API, and can be swapped for Qdrant/Pinecone by changing one file.

#### User isolation in ChromaDB
ChromaDB uses a single collection but every document is tagged with `user_id`:
```python
metadatas.append({
    "user_id": user_id,     # ← tagged at write time
    "doc_id": doc_id,
    "type": "text",
})
```

At query time, the filter is always injected server-side:
```python
where = {"user_id": {"$eq": user_id}}   # user_id comes from JWT, not from client
results = await collection.query(
    query_embeddings=[query_vector],
    where=where,            # ← enforced filter
)
```
The client never controls the `user_id` filter. It comes from the auth token.
This is the security guarantee: User A's query can never return User B's documents.

#### Cosine similarity — how vector search works
```python
"score": round(1 - dist, 4)    # convert distance to similarity
```
ChromaDB returns distances (lower = more similar).
`1 - distance` converts to similarity (higher = more similar).
A score of `0.95` means the chunk is very relevant to the query.
A score of `0.3` means it is barely related.

---

## SECTION 15 — Observability: Logging, Tracing, Why It Matters

### What is observability?
Observability is the ability to understand what your system is doing from the outside.
In production, you cannot attach a debugger. You need logs and traces to understand:
- Which node ran?
- How long did it take?
- Did it succeed or fail?
- Which user triggered it?
- How many tokens were used?

Without observability, debugging a production issue is guesswork.

### Structured logging — logs as JSON
Plain text logs are hard to search and filter:
```
2024-01-01 INFO analyze_query ran in 342ms for user user_123
```

JSON logs can be queried by any log aggregation tool (Datadog, CloudWatch, Grafana):
```json
{
  "ts": "2024-01-01T00:00:00",
  "level": "INFO",
  "node": "analyze_query",
  "user_id": "user_123",
  "session_id": "abc-456",
  "latency_ms": 342.5,
  "retrieval_path": "live_context"
}
```

You can now query: "show me all requests where latency_ms > 1000" or
"show me all requests for user_123 in the last hour".

### The JSONFormatter class
```python
# from app/observability/tracer.py
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log.update(record.extra)    # merge in extra fields
        return json.dumps(log)
```

`logging.Formatter` is Python's built-in log formatter.
We override `format()` to output JSON instead of plain text.

### How every node gets logged automatically
The `@track_node` decorator wraps every node function:
```python
@track_node("grade_answer")
async def grade_answer(state: AgentState) -> dict:
    ...
```

The decorator runs before and after the function:
```python
start = time.perf_counter()
result = await fn(state)                    # actual node runs here
latency_ms = (time.perf_counter() - start) * 1000

agent_logger.info("node:grade_answer", extra={
    "extra": {
        "node": "grade_answer",
        "user_id": state.get("user_id"),
        "latency_ms": latency_ms,
        "grading_score": result.get("grading_score", ""),
    }
})
```

The node function itself has zero logging code. The decorator handles it all.
This is called **separation of concerns** — the node does its job, the decorator observes it.

### LangSmith — tracing for LLM applications
LangSmith is a platform by LangChain that records every LLM call with:
- Full input messages sent to the model
- Full output received
- Token count (input + output)
- Latency
- Cost estimate
- Which node in the graph triggered it

Enable it with two env vars:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
```

Then go to smith.langchain.com to see a visual trace of every agent run:
```
analyze_query (120ms, 45 tokens)
  └── check_session (8ms)
        └── answer_from_live_context (1840ms, 1203 tokens)
              └── grade_answer (340ms, 89 tokens)
                    └── store_to_vectorstore (210ms)
                          └── format_response (2ms)
```

### `time.perf_counter()` — high-precision timing
```python
start = time.perf_counter()    # returns float in seconds, very precise
# ... do work ...
elapsed = time.perf_counter() - start
ms = round(elapsed * 1000, 2)  # convert to milliseconds
```
`perf_counter` is more precise than `time.time()` for measuring short durations.

---

## SECTION 16 — Document Extraction: unstructured, pathlib, base64, io

### The `unstructured` library — why it exists
A PDF is not just text. It contains:
- Paragraphs of text
- Tables (rows and columns)
- Images and graphs
- Headers, footers, captions
- Different fonts, layouts, columns

Reading a PDF with a simple text extractor loses all structure.
`unstructured` is a library that understands document structure and extracts each element
with its type (Table, Image, NarrativeText, Title, etc.).

```python
from unstructured.partition.auto import partition

elements = partition(
    filename="report.pdf",
    strategy="hi_res",              # use high-resolution model for better accuracy
    infer_table_structure=True,     # detect and parse tables
    extract_images_in_pdf=True,     # extract embedded images
)

for el in elements:
    print(type(el).__name__, str(el)[:50])
# NarrativeText  "The company reported revenue of $4.2 billion..."
# Table          "| Quarter | Revenue | Growth |..."
# Image          "Figure 1: Revenue by region"
```

### `isinstance()` — check what type an element is
```python
from unstructured.documents.elements import Table, Image, NarrativeText

for el in elements:
    if isinstance(el, Table):
        # handle table
    elif isinstance(el, Image):
        # handle image
    elif isinstance(el, NarrativeText):
        # handle text paragraph
```
`isinstance(el, Table)` returns `True` if `el` is a `Table` object (or a subclass of it).

### `pathlib.Path` — modern file path handling
```python
from pathlib import Path

# Old way (string concatenation — error-prone)
path = "uploads/" + doc_id + "_" + filename

# New way (pathlib — clean and cross-platform)
path = Path("uploads") / f"{doc_id}_{filename}"

path.write_bytes(content)          # write bytes to file
path.read_bytes()                  # read bytes from file
path.unlink(missing_ok=True)       # delete file (no error if already gone)
path.name                          # just the filename: "report.pdf"
path.suffix                        # just the extension: ".pdf"
Path("uploads").mkdir(exist_ok=True)  # create directory if it doesn't exist
```

`pathlib` handles Windows (`\`) and Unix (`/`) path separators automatically.
Always use `pathlib` instead of string concatenation for file paths.

### `base64` — encoding binary data as text
Images are binary data (bytes). JSON and HTTP are text-based.
Base64 encoding converts binary → text so images can be sent in JSON or HTTP requests.

```python
import base64

# Encode bytes to base64 string
image_bytes = Path("image.png").read_bytes()
b64_string = base64.b64encode(image_bytes).decode("utf-8")
# b64_string = "iVBORw0KGgoAAAANSUhEUgAA..."

# Decode back
original_bytes = base64.b64decode(b64_string)
```

In this project, images are base64-encoded and sent directly to GPT-4o:
```python
{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
```

### `io.BytesIO` — in-memory file
`BytesIO` is a file-like object that lives in memory instead of on disk.
Used to avoid writing temporary files:

```python
import io
from PIL import Image as PILImage

# Open image from bytes without saving to disk
img = PILImage.open(io.BytesIO(image_bytes))

# Save resized image to memory buffer
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
```

### `hasattr()` — safely check if an attribute exists
```python
if hasattr(el, "metadata") and el.metadata.page_number:
    page_numbers.add(el.metadata.page_number)
```
`hasattr(el, "metadata")` returns `True` if `el` has a `metadata` attribute.
Without this check, accessing `el.metadata` on an element that doesn't have it would crash.

---

## SECTION 17 — RAG: Retrieval Augmented Generation

### The problem RAG solves
LLMs like GPT-4o are trained on data up to a certain date.
They do not know about your private documents, internal reports, or recent data.

If you ask GPT-4o "What was our Q3 revenue?", it cannot answer — it has never seen your report.

### Option 1: Put the whole document in the prompt
```python
prompt = f"Document:\n{entire_document}\n\nQuestion: {query}"
response = llm.invoke(prompt)
```
This works for small documents. But:
- GPT-4o has a context limit (~128k tokens ≈ ~100 pages)
- Sending 100 pages costs a lot of tokens on every query
- For very large documents, it simply does not fit

### Option 2: RAG — only send the relevant parts
1. At upload time: split the document into small chunks, embed each chunk, store in vector DB
2. At query time: embed the query, find the most similar chunks, send only those to the LLM

```
Document (100 pages)
    │
    ▼ chunk
[chunk 1] [chunk 2] [chunk 3] ... [chunk 200]
    │
    ▼ embed
[vector 1] [vector 2] [vector 3] ... [vector 200]
    │
    ▼ store
ChromaDB

Query: "What was Q3 revenue?"
    │
    ▼ embed query
[query_vector]
    │
    ▼ find nearest vectors in ChromaDB
[chunk 47] [chunk 48] [chunk 51]   ← most relevant chunks
    │
    ▼ send to LLM
"Based on these excerpts, Q3 revenue was $4.2B"
```

### How this project uses RAG
RAG is the fallback path when the document is NOT in the session:

```python
# from app/agents/nodes.py
async def answer_from_rag(state: AgentState) -> dict:
    chunks = await vector_store.retrieve(
        user_id=state["user_id"],    # only this user's documents
        query=state["rewritten_query"],
    )
    context = "\n\n---\n\n".join(c["content"] for c in chunks)
    response = await _llm().ainvoke([
        SystemMessage(content="Answer only from the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])
```

### Live context vs RAG — the tradeoff
| | Live Context | RAG |
|---|---|---|
| When | Doc in Redis session | Doc expired from session |
| What is sent to LLM | Full document | Top-K relevant chunks only |
| Accuracy | Higher (full context) | Lower (may miss context) |
| Token cost | Higher | Lower |
| Speed | Slower (more tokens) | Faster |
| Works for large docs | No (context limit) | Yes |

This project uses live context first (better accuracy) and falls back to RAG
(cheaper, works for large docs, works when session expires).

### Text splitting — why chunks need overlap
```python
# from app/memory/vector_store.py
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # each chunk is ~512 tokens
    chunk_overlap=64,     # last 64 tokens of chunk N = first 64 tokens of chunk N+1
)
```

Without overlap, a sentence that spans a chunk boundary gets split:
```
Chunk 1: "...The Q3 revenue was"
Chunk 2: "$4.2 billion, up 12%..."
```
Neither chunk alone answers "What was Q3 revenue?".

With overlap, both chunks contain the full sentence, so at least one will be retrieved.

---

## SECTION 18 — Agentic Patterns: Why Agents, Grading, Self-Correction

### What makes something "agentic"?
A simple LLM call is not an agent:
```python
response = llm.invoke("What is the revenue?")   # one call, done
```

An agent is a system that:
- Decides what to do next based on the current state
- Can take multiple steps
- Can retry or change strategy when something fails
- Has tools it can use (retrieval, memory, etc.)

### The grading pattern — self-evaluation
After generating an answer, the agent grades its own answer:

```python
# from app/agents/nodes.py
async def grade_answer(state: AgentState) -> dict:
    result = await _llm().with_structured_output(GradeResult).ainvoke([
        SystemMessage(content="Grade whether this answer is grounded in the context."),
        HumanMessage(content=f"Context: {context}\nAnswer: {state['raw_answer']}"),
    ])
    return {"grading_score": result.score}   # "good" or "poor"
```

Why grade? LLMs can hallucinate — confidently state things that are not in the document.
The grader catches this and triggers a retry instead of returning a bad answer.

### The rewrite pattern — query reformulation
If the answer is poor, the agent rewrites the query and tries again:

```python
async def rewrite_query(state: AgentState) -> dict:
    response = await _llm().ainvoke([
        SystemMessage(content="Rewrite this question to be more specific."),
        HumanMessage(content=state["rewritten_query"]),
    ])
    return {
        "rewritten_query": response.content.strip(),
        "retry_count": state["retry_count"] + 1,
    }
```

Example:
- Original: "What happened in Q3?"
- Rewritten: "What were the Q3 2024 financial results including revenue, profit, and growth?"

The rewritten query is more specific → better retrieval → better answer.

### Retry budget — preventing infinite loops
```python
# from app/agents/graph.py
def route_by_grade(state: AgentState) -> str:
    if state["grading_score"] == "good":
        return "good"
    if state.get("retry_count", 0) < settings.max_rewrite_retries:   # max 2 retries
        return "retry"
    return "fallback"   # give up gracefully after 2 retries
```

Without a retry budget, a poor document could cause infinite loops.
`max_rewrite_retries=2` means at most 3 total attempts (1 original + 2 retries).

### The fallback pattern — graceful degradation
```python
async def fallback_response(state: AgentState) -> dict:
    return {
        "final_answer": (
            "I was unable to find a confident answer in your document. "
            "Please try rephrasing your question or upload the document again."
        ),
        "confidence": 0.0,
    }
```

Never crash or return an empty response. Always give the user something useful,
even if it is just an explanation of why the answer could not be found.

### Why this is better than a single LLM call
| Single LLM call | Agentic approach |
|---|---|
| May hallucinate | Grader catches hallucinations |
| Bad query = bad answer | Query rewriter improves retrieval |
| No retry | Up to 2 retries with improved queries |
| Crashes on failure | Graceful fallback always returns something |
| No observability | Every step logged with latency |

---

## SECTION 19 — Token Optimization

Tokens are the units LLMs charge by. Every word ≈ 1 token. Every API call costs money.
In production with many users, token waste adds up fast.

### What is a token?
```
"Hello world"  →  2 tokens
"The quarterly revenue report for Q3 2024"  →  8 tokens
A typical PDF page  →  ~500-800 tokens
```

GPT-4o pricing (approximate): $5 per 1 million input tokens, $15 per 1 million output tokens.
A 50-page document ≈ 30,000 tokens. Sending it on every query = expensive.

### Strategy 1: Only send images when needed
```python
# from app/agents/nodes.py
if state["query_type"] in ("image", "general") and payload["images"]:
    for b64 in payload["images"][:5]:   # max 5 images
        content.append({"type": "image_url", ...})
```
Images are expensive — a single image can cost 500-1000 tokens.
For a text query like "What is the revenue?", images are irrelevant.
Only attach images when the query type suggests visual content is needed.

### Strategy 2: Resize images before encoding
```python
# from app/ingestion/extractor.py
if img.width > 1024:
    ratio = 1024 / img.width
    img = img.resize((1024, int(img.height * ratio)))
```
GPT-4o charges more for larger images. Resizing to max 1024px wide
reduces token cost while keeping enough detail for the model to understand.

### Strategy 3: Cap images per call
```python
for b64 in payload["images"][:5]:   # never send more than 5 images
```
A document might have 50 images. Sending all 50 would be extremely expensive.
Cap at 5 — enough for the model to understand visual content.

### Strategy 4: Truncate context for grading
```python
# from app/agents/nodes.py
context = state["doc_payload"]["text"][:3000]   # first 3000 chars only
```
The grader does not need the full document to check if an answer is grounded.
3000 characters (≈ 750 tokens) is enough context for grading.
Sending the full document to the grader would waste tokens.

### Strategy 5: Conversation history window
```python
# from app/memory/session_store.py
history = history[-20:]   # keep only last 20 turns
```
Conversation history grows with every turn. Sending the full history on every query
would eventually exceed the context limit and cost a lot.
Keeping only the last 20 turns bounds the token cost.

### Strategy 6: Cheap embedding model
```python
openai_embedding_model: str = "text-embedding-3-small"
```
`text-embedding-3-small` costs ~20x less than `text-embedding-3-large`
with only a small accuracy difference. For most RAG use cases, small is sufficient.

### Strategy 7: RAG instead of full doc for repeat queries
When the session expires, RAG sends only the top-6 relevant chunks (≈ 3000 tokens)
instead of the full document (≈ 30,000 tokens for a 50-page doc).
10x token reduction for repeat queries.

### Token cost comparison per query
| Scenario | Approx tokens | Approx cost |
|---|---|---|
| Live context, text query, 10-page doc | 6,000 | $0.03 |
| Live context, image query, 10-page doc + 5 images | 11,000 | $0.055 |
| RAG query (top-6 chunks) | 3,500 | $0.017 |
| Naive: full doc every query, 50-page doc | 35,000 | $0.175 |

---

## SECTION 20 — Key Mental Models and What to Build Next

### The 5 mental models that explain this entire project

#### 1. State flows through nodes
```
State → Node 1 → updated State → Node 2 → updated State → ...
```
Every node receives the full state, does one job, returns only what changed.
LangGraph merges the changes. No node needs to know about other nodes.

#### 2. Decorators add behavior without changing functions
```
@track_node("analyze_query")   ← adds timing + logging
async def analyze_query(state):
    ...                        ← just does its job, knows nothing about logging
```
Cross-cutting concerns (logging, timing, auth) live in decorators, not in business logic.

#### 3. Dependency injection via `Depends`
```
Route function ← FastAPI injects → user_id (from auth)
                                 → settings (from cache)
                                 → db connection (from factory)
```
Functions declare what they need. The framework provides it. Functions stay testable.

#### 4. Isolation by namespacing
```
Redis key:    session:{user_id}:{session_id}:doc
ChromaDB filter: WHERE user_id = {user_id}
```
User data is isolated not by separate databases but by consistent namespacing.
The `user_id` always comes from auth, never from the client.

#### 5. Async for I/O, sync for CPU
```
async def → Redis, ChromaDB, OpenAI API (waiting for network)
def       → extract(), _chunk_text(), _to_markdown_table() (CPU work)
```

---

### Common beginner mistakes to avoid

| Mistake | Correct approach |
|---|---|
| `import *` from a module | Always import specific names |
| Mutable default arguments `def f(x=[])` | Use `def f(x=None)` and set inside |
| Blocking calls inside async functions | Always `await` async operations |
| Trusting client input for auth | Always extract user_id from JWT/header server-side |
| No error handling | Use `try/except` and `HTTPException` |
| Creating DB connections on every request | Use singleton pattern with lazy init |
| Sending full document on every query | Use RAG for repeat queries |

---

### What to build next (learning path)

1. Add a `/docs/{doc_id}` endpoint to list all documents a user has uploaded
2. Add support for `.txt` and `.csv` files in `extractor.py`
3. Add a confidence threshold — if confidence < 0.5, always fall back to RAG
4. Replace `X-User-ID` header with real JWT auth using `python-jose`
5. Add a `/metrics` endpoint that returns average latency per node from logs
6. Swap ChromaDB for Qdrant — only `vector_store.py` needs to change
7. Add rate limiting per user using Redis counters
8. Write tests for each node function — they are just async functions, easy to test

---

### File-by-file concept map

| File | Python concepts | AI/infra concepts |
|---|---|---|
| `config/settings.py` | Class, inheritance, `@lru_cache`, type hints | Env vars, configuration management |
| `agents/state.py` | TypedDict, Literal, Annotated | Agent state, data modeling |
| `ingestion/extractor.py` | Functions, pathlib, base64, io, isinstance | Document parsing, multimodal |
| `memory/session_store.py` | async/await, global vars, lazy init, json | Redis, TTL, session management |
| `memory/vector_store.py` | async, list comprehension, enumerate, zip | Embeddings, RAG, vector search, chunking |
| `observability/tracer.py` | Decorators, `@wraps`, context managers, logging | Structured logging, tracing, LangSmith |
| `agents/nodes.py` | async functions, Pydantic models, f-strings | LangChain, LLM calls, structured output |
| `agents/graph.py` | Functions, conditional logic, module-level vars | LangGraph, StateGraph, edges, routing |
| `api/auth.py` | Function, type hints, exceptions | Auth, dependency injection |
| `api/routes.py` | Classes, async generators, `Depends`, decorators | FastAPI, REST API, SSE streaming, background tasks |
| `main.py` | `@asynccontextmanager`, middleware, app factory | FastAPI app lifecycle, CORS |

---
