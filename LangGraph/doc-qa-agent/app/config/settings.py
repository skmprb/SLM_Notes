from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Doc QA Agent"
    app_env: str = "development"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"                    # vision-capable for images/graphs
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 2048

    # Redis (session store)
    redis_url: str = "redis://localhost:6379"
    session_ttl_seconds: int = 3600                 # 1hr session expiry

    # ChromaDB (vector store)
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "documents"

    # Ingestion
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 50
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Agent
    max_rewrite_retries: int = 2
    retrieval_top_k: int = 6

    # Observability
    langsmith_api_key: str = ""
    langsmith_project: str = "doc-qa-agent"
    langsmith_tracing: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
