"""
Enterprise AI Platform - Main Application Entry Point

Phase 1: Simplest working product
- FastAPI with service layer
- Direct OpenAI integration
- Structured logging
- Configuration management
- Health check

Run:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from app.api.routes import router
from app.config.settings import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise AI Platform - Production-grade GenAI system",
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def startup():
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} | "
        f"env={settings.app_env} | model={settings.openai_model}"
    )


@app.on_event("shutdown")
def shutdown():
    logger.info("Shutting down...")
