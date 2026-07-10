"""
COMPARISON: Our Model Router vs LangChain/LangGraph Routing

This file shows how our routing patterns map to the framework equivalents.

Key insight:
    Model routing (which MODEL to use) is different from
    Agent routing (which AGENT to use).

    Our Phase 3 = Model routing (cost/capability optimization)
    LangGraph's router = Agent routing (task delegation)

    Both are important in production. We'll add agent routing later.
"""

# ============================================================
# APPROACH 1: Our Custom Router
# ============================================================

def our_rule_based_router():
    """Our rule-based routing."""
    from app.llm.router import create_router, RoutingContext

    router = create_router("rule_based")

    # Test different messages
    test_cases = [
        "What is 2+2?",                          # → gemini-flash (simple)
        "Write a Python function to sort a list", # → claude (coding)
        "Analyze the pros and cons of microservices vs monolith",  # → gpt-4o (reasoning)
        "Write a poem about the ocean",           # → claude (creative)
    ]

    for msg in test_cases:
        context = RoutingContext(message=msg, estimated_tokens=len(msg) // 4)
        decision = router.route(context)
        print(f"  '{msg[:40]}...' → {decision.provider}:{decision.model}")
        print(f"    Reason: {decision.reason}\n")


# ============================================================
# APPROACH 2: LangChain Configurable Model (Runtime Switching)
# ============================================================

def langchain_configurable_model():
    """
    LangChain's approach to runtime model switching.

    from langchain.chat_models import init_chat_model

    # Create a configurable model (no default)
    model = init_chat_model(temperature=0)

    # Switch at runtime via config
    model.invoke("simple question", config={"configurable": {"model": "gpt-4o-mini"}})
    model.invoke("complex code", config={"configurable": {"model": "claude-3-5-sonnet-20241022"}})

    The routing decision is EXTERNAL to LangChain.
    You still need your own router to decide WHICH model to pass in config.
    """
    pass


# ============================================================
# APPROACH 3: LangChain Middleware (Dynamic Model Selection)
# ============================================================

def langchain_middleware_routing():
    """
    LangChain's middleware approach for dynamic model selection.

    This is the closest to our LLMRouter pattern.

    from langchain import createAgent, createMiddleware
    from langchain_openai import ChatOpenAI

    basic_model = ChatOpenAI(model="gpt-4o-mini")
    advanced_model = ChatOpenAI(model="gpt-4o")

    dynamic_routing = createMiddleware(
        name="DynamicModelSelection",
        wrapModelCall=lambda request, handler: handler({
            ...request,
            model=advanced_model if len(request.messages) > 10 else basic_model,
        })
    )

    Note: This is JavaScript-focused in LangChain docs.
    In Python, you'd use LangGraph conditional edges.
    """
    pass


# ============================================================
# APPROACH 4: LangGraph Conditional Routing
# ============================================================

def langgraph_conditional_routing():
    """
    LangGraph's approach: conditional edges in a graph.

    This is the most powerful pattern for complex routing.

    from langgraph.graph import StateGraph
    from langgraph.types import Command

    def classify_and_route(state):
        # Classify the task
        task_type = classify(state["message"])

        # Route to different nodes based on classification
        if task_type == "coding":
            return Command(goto="coding_agent")
        elif task_type == "reasoning":
            return Command(goto="reasoning_agent")
        else:
            return Command(goto="simple_agent")

    graph = StateGraph(State)
    graph.add_node("router", classify_and_route)
    graph.add_node("coding_agent", coding_node)
    graph.add_node("reasoning_agent", reasoning_node)
    graph.add_node("simple_agent", simple_node)
    graph.set_entry_point("router")

    This routes to different AGENTS (not just models).
    We'll build this in Phase 8-9.
    """
    pass


# ============================================================
# APPROACH 5: LiteLLM Router (External Service)
# ============================================================

def litellm_router():
    """
    LiteLLM provides a production-ready router as a service.

    from langchain.chat_models import init_chat_model

    # LiteLLM handles routing, fallbacks, load balancing
    model = init_chat_model("litellm:gpt-4o")

    Features:
    - Automatic fallbacks
    - Load balancing across providers
    - Cost tracking
    - Rate limit handling
    - Retry logic

    This is what you'd use in production if you don't want to build your own.
    But building your own teaches you the patterns.
    """
    pass


# ============================================================
# MAPPING TABLE
# ============================================================

"""
┌──────────────────────────┬─────────────────────────────────────────────┐
│ Our Code                 │ LangChain/LangGraph Equivalent              │
├──────────────────────────┼─────────────────────────────────────────────┤
│ BaseRouter               │ No direct equivalent (custom pattern)       │
│ RuleBasedRouter          │ Custom logic before init_chat_model()       │
│ CostAwareRouter          │ LiteLLM Router / OpenRouter                 │
│ LLMRouter                │ Middleware wrapModelCall / LangGraph node    │
│ RoutingContext           │ State in LangGraph                          │
│ RoutingDecision          │ Command(goto=...) in LangGraph              │
│ create_router()          │ No equivalent (you build this yourself)     │
│ /route test endpoint     │ LangSmith tracing (see why model was used)  │
└──────────────────────────┴─────────────────────────────────────────────┘

Key takeaway:
    LangChain/LangGraph don't have a built-in "model router" because
    routing logic is highly application-specific.

    They provide:
    - Configurable models (switch at runtime)
    - Middleware (intercept and modify model selection)
    - Conditional edges (route in a graph)
    - External routers (LiteLLM, OpenRouter)

    But the DECISION LOGIC (which model for which task) is always yours to build.
    That's exactly what our router package does.
"""


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Our Rule-Based Router")
    print("=" * 60)
    our_rule_based_router()
