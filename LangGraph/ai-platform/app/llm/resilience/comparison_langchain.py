"""
COMPARISON: Our Resilience Layer vs LangChain/LangGraph

This file maps our retry/fallback/circuit-breaker patterns to framework equivalents.
"""

# ============================================================
# APPROACH 1: Our Custom Resilience
# ============================================================

def our_approach():
    """
    Our resilience stack:

    FallbackManager
        ├── CircuitBreakerRegistry
        │       ├── CircuitBreaker("openai")    → CLOSED/OPEN/HALF_OPEN
        │       ├── CircuitBreaker("anthropic")
        │       └── CircuitBreaker("google")
        ├── RetryHandler
        │       └── Exponential backoff + jitter
        └── Provider chain
                └── [openai, anthropic, google]

    Flow:
        1. Check circuit breaker for primary provider
        2. If open → skip to next provider
        3. If closed → try with retries
        4. If all retries fail → record failure, try next provider
        5. If all providers fail → raise ProviderUnavailableError
    """
    from app.llm.resilience import FallbackManager, FallbackConfig, RetryConfig

    manager = FallbackManager(FallbackConfig(
        fallback_chain=["openai", "anthropic", "google"],
        retry_config=RetryConfig(max_retries=3, initial_delay=1.0, backoff_factor=2.0),
        enable_circuit_breaker=True,
    ))

    messages = [{"role": "user", "content": "Hello"}]
    result = manager.call(messages)

    print(f"Provider: {result.provider_used}")
    print(f"Fallback: {result.was_fallback}")
    print(f"Attempts: {result.total_attempts}")


# ============================================================
# APPROACH 2: LangChain Built-in Retry
# ============================================================

def langchain_builtin_retry():
    """
    LangChain chat models have built-in retry (max_retries=6 by default).

    from langchain.chat_models import init_chat_model

    model = init_chat_model(
        "openai:gpt-4o",
        max_retries=10,     # Increase for unreliable networks
        timeout=120,        # Seconds
    )

    # Automatically retries on 429, 5xx, network errors
    # Does NOT retry on 401, 404
    response = model.invoke("Hello")
    """
    pass


# ============================================================
# APPROACH 3: LangChain Middleware (Retry + Fallback)
# ============================================================

def langchain_middleware():
    """
    LangChain's middleware approach combines retry + fallback.

    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        ModelRetryMiddleware,
        ModelFallbackMiddleware,
    )

    agent = create_agent(
        model="openai:gpt-4o",
        middleware=[
            # Retry with exponential backoff
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
                jitter=True,
                retry_on=(TimeoutError, ConnectionError),  # Specific errors
                on_failure="continue",  # Return error message instead of raising
            ),
            # Fallback to alternative models
            ModelFallbackMiddleware(
                "gpt-4o-mini",
                "claude-3-5-sonnet-20241022",
            ),
        ],
    )

    Note: LangChain's fallback is simpler than ours:
    - No circuit breaker
    - No per-provider health tracking
    - No configurable retry per fallback
    """
    pass


# ============================================================
# APPROACH 4: LangGraph RetryPolicy (Node-level)
# ============================================================

def langgraph_retry_policy():
    """
    LangGraph has retry at the GRAPH NODE level.

    from langgraph.graph import StateGraph
    from langgraph.pregel import RetryPolicy

    graph = StateGraph(State)
    graph.add_node(
        "call_llm",
        call_llm_node,
        retry=RetryPolicy(
            max_attempts=3,
            initial_interval=0.5,
            backoff_factor=2.0,
            max_interval=128.0,
            jitter=True,
            retry_on=lambda e: isinstance(e, (TimeoutError, RateLimitError)),
        ),
    )

    This retries the ENTIRE NODE, not just the LLM call.
    Useful when the node does multiple things (retrieve + generate).
    """
    pass


# ============================================================
# APPROACH 5: LangChain with_fallbacks (Legacy)
# ============================================================

def langchain_with_fallbacks():
    """
    Older LangChain pattern using .with_fallbacks().

    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic

    primary = ChatOpenAI(model="gpt-4o")
    fallback1 = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    fallback2 = ChatOpenAI(model="gpt-4o-mini")

    model_with_fallbacks = primary.with_fallbacks([fallback1, fallback2])

    # If primary fails, tries fallback1, then fallback2
    response = model_with_fallbacks.invoke("Hello")

    Limitations:
    - No circuit breaker
    - No configurable retry per model
    - No health tracking
    - Simple sequential fallback only
    """
    pass


# ============================================================
# MAPPING TABLE
# ============================================================

"""
┌─────────────────────────────┬──────────────────────────────────────────────────┐
│ Our Code                    │ LangChain/LangGraph Equivalent                   │
├─────────────────────────────┼──────────────────────────────────────────────────┤
│ RetryHandler                │ ModelRetryMiddleware / max_retries param          │
│ RetryConfig                 │ ModelRetryMiddleware params / RetryPolicy         │
│ CircuitBreaker              │ No equivalent (must build yourself)               │
│ CircuitBreakerRegistry      │ No equivalent                                    │
│ FallbackManager             │ ModelFallbackMiddleware / .with_fallbacks()       │
│ FallbackConfig              │ Middleware constructor params                     │
│ ResilientResponse           │ No equivalent (metadata not exposed)              │
│ LLMError hierarchy          │ No standard hierarchy (each provider has own)     │
│ classify_error()            │ No equivalent (each provider handles differently) │
│ /circuits endpoint          │ No equivalent (use external monitoring)           │
└─────────────────────────────┴──────────────────────────────────────────────────┘

What we have that LangChain doesn't:
    1. Circuit breaker (prevents hammering failing providers)
    2. Per-provider health tracking
    3. Unified error classification
    4. Resilience metadata in response (was_fallback, attempts)
    5. Manual circuit reset endpoint
    6. Configurable retry per fallback level

What LangChain has that we don't (yet):
    1. Async retry (we're sync for now)
    2. Tool-level retry (ToolRetryMiddleware)
    3. on_failure="continue" (return error as message to agent)
    4. Integration with LangSmith tracing
"""
