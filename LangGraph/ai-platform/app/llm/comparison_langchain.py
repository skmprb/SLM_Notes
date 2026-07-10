"""
COMPARISON: Our Custom Abstraction vs LangChain

This file is for LEARNING ONLY - it shows how our patterns map to LangChain.
Run this to see both approaches side by side.

Key insight:
    LangChain's init_chat_model() is essentially our create_llm() factory.
    LangChain's BaseChatModel is our BaseLLM interface.
    LangChain's ChatOpenAI/ChatAnthropic are our OpenAIProvider/AnthropicProvider.

The difference:
    - LangChain gives you this for free (battle-tested, maintained)
    - Building it yourself teaches you WHY it exists
    - In production, you'll likely use LangChain's version
    - But you'll UNDERSTAND what's happening under the hood
"""

# ============================================================
# APPROACH 1: Our Custom Abstraction (what we just built)
# ============================================================

def our_approach():
    """How our custom abstraction works."""
    from app.llm.factory import create_llm
    from app.llm.base import LLMConfig

    # Factory creates the right provider from config
    llm = create_llm("openai")

    # Same interface regardless of provider
    response = llm.generate(
        messages=[{"role": "user", "content": "Hello"}],
        config=LLMConfig(model="gpt-4o-mini", temperature=0.7),
    )

    print(f"[Our Abstraction] {response.provider}: {response.content[:50]}...")

    # Switch provider - SAME code, different provider
    llm_claude = create_llm("anthropic")
    response = llm_claude.generate(
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(f"[Our Abstraction] {response.provider}: {response.content[:50]}...")


# ============================================================
# APPROACH 2: LangChain's Abstraction
# ============================================================

def langchain_approach():
    """
    How LangChain does the same thing.

    LangChain's init_chat_model is their factory.
    It returns a BaseChatModel instance (their interface).

    Usage:
        from langchain.chat_models import init_chat_model

        # Factory - creates the right provider
        model = init_chat_model("openai:gpt-4o-mini")

        # Same interface regardless of provider
        response = model.invoke("Hello")
        print(response.content)

        # Switch provider - same code
        model = init_chat_model("anthropic:claude-3-5-sonnet-20241022")
        response = model.invoke("Hello")
        print(response.content)

    Configurable model (runtime switching):
        model = init_chat_model(temperature=0)

        # Same model instance, different provider at runtime
        model.invoke("Hello", config={"configurable": {"model": "gpt-4o-mini"}})
        model.invoke("Hello", config={"configurable": {"model": "claude-3-5-sonnet-20241022"}})
    """
    from langchain.chat_models import init_chat_model

    # LangChain's factory - "provider:model" format
    model = init_chat_model("openai:gpt-4o-mini")
    response = model.invoke("Hello")
    print(f"[LangChain] {response.content[:50]}...")

    # Switch provider
    model = init_chat_model("anthropic:claude-3-5-sonnet-20241022")
    response = model.invoke("Hello")
    print(f"[LangChain] {response.content[:50]}...")


# ============================================================
# MAPPING: Our Code → LangChain Equivalent
# ============================================================

"""
┌─────────────────────────┬──────────────────────────────────────┐
│ Our Code                │ LangChain Equivalent                 │
├─────────────────────────┼──────────────────────────────────────┤
│ BaseLLM                 │ BaseChatModel                        │
│ LLMResponse             │ AIMessage                            │
│ LLMConfig               │ Model constructor params             │
│ create_llm("openai")    │ init_chat_model("openai:gpt-4o")     │
│ OpenAIProvider          │ ChatOpenAI                           │
│ AnthropicProvider       │ ChatAnthropic                        │
│ GeminiProvider          │ ChatGoogleGenerativeAI               │
│ provider.generate()     │ model.invoke()                       │
│ provider.provider_name  │ model._llm_type                      │
└─────────────────────────┴──────────────────────────────────────┘

Why build our own first?
1. You understand the Adapter Pattern
2. You understand the Factory Pattern
3. You understand Dependency Inversion
4. You can debug LangChain issues because you know what's underneath
5. You can extend LangChain's classes when needed
6. You can build custom providers that LangChain doesn't support
"""


if __name__ == "__main__":
    print("=" * 60)
    print("Our Custom Abstraction")
    print("=" * 60)
    our_approach()

    print("\n" + "=" * 60)
    print("LangChain Equivalent")
    print("=" * 60)
    langchain_approach()
