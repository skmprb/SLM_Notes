from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    app_name: str = "Enterprise AI Platform"
    app_version: str = "1.0.0"
    app_env: str = "development"
    app_debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Default LLM Provider
    default_provider: str = "openai"

    # Routing Strategy: rule_based | cost_aware | llm
    routing_strategy: str = "rule_based"

    # Resilience: Fallback chain
    fallback_chain: str = "openai,anthropic,google"
    max_retries: int = 3
    retry_initial_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    enable_circuit_breaker: bool = True

    # Conversation Store: memory | sqlite
    conversation_store: str = "memory"
    sqlite_db_path: str = "conversations.db"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 1024

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Google
    google_api_key: str = ""
    google_model: str = "gemini-1.5-flash"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance - loaded once, reused everywhere."""
    return Settings()
