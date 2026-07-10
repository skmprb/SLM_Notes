import logging
import sys
from app.config.settings import get_settings


def setup_logger(name: str) -> logging.Logger:
    """Create a structured logger with consistent formatting."""
    settings = get_settings()

    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
