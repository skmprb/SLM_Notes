"""
LLM Factory - Creates the correct provider based on configuration.

This is the Factory Pattern:
- Caller doesn't need to know which concrete class to instantiate
- Adding a new provider = add one entry to the registry
- Configuration drives which provider is used

LangChain equivalent: init_chat_model("openai:gpt-4o")
    - LangChain uses the "provider:model" string format
    - We use a config-driven approach

Usage:
    from app.llm.factory import create_llm

    llm = create_llm("openai")       # Returns OpenAIProvider
    llm = create_llm("anthropic")    # Returns AnthropicProvider
    llm = create_llm("google")       # Returns GeminiProvider
    llm = create_llm()               # Returns default from config
"""

from app.llm.base import BaseLLM
from app.config.settings import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Provider registry - maps string names to provider classes
# This is the only place you need to modify when adding a new provider
_PROVIDER_REGISTRY: dict[str, type] = {}


def _register_providers():
    """Lazy registration to avoid import errors for uninstalled packages."""
    global _PROVIDER_REGISTRY
    if _PROVIDER_REGISTRY:
        return

    _PROVIDER_REGISTRY = {
        "openai": "app.llm.providers.openai_provider.OpenAIProvider",
        "anthropic": "app.llm.providers.anthropic_provider.AnthropicProvider",
        "google": "app.llm.providers.gemini_provider.GeminiProvider",
    }


def create_llm(provider: str | None = None) -> BaseLLM:
    """
    Factory function to create an LLM provider instance.

    Args:
        provider: Provider name ("openai", "anthropic", "google").
                  If None, uses the default from settings.

    Returns:
        An instance of BaseLLM (the concrete provider)

    Raises:
        ValueError: If provider is not supported
    """
    _register_providers()

    settings = get_settings()
    provider = provider or settings.default_provider

    if provider not in _PROVIDER_REGISTRY:
        available = list(_PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")

    # Dynamic import to avoid requiring all provider packages
    module_path = _PROVIDER_REGISTRY[provider]
    module_name, class_name = module_path.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_name)
    provider_class = getattr(module, class_name)

    instance = provider_class()
    logger.info(f"Created LLM provider | provider={provider} | class={class_name}")
    return instance
