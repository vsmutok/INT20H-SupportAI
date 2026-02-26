"""
Centralized logger configuration for the project.

Usage:
    from src.config.logger import logger

The logger is pre-configured with:
- Console output with colored formatting
- File output (logs/app.log) with rotation
- Separate error log (logs/errors.log)
"""

import sys
from pathlib import Path

from loguru import logger

# Remove default loguru handler
logger.remove()

# Log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Console handler
# Rich, readable format with step indicators
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)

# File handler — all logs
logger.add(
    LOG_DIR / "app.log",
    format=("{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} | {message}"),
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
)

# File handler — errors only
logger.add(
    LOG_DIR / "errors.log",
    format=("{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} | {message}"),
    level="WARNING",
    rotation="5 MB",
    retention="30 days",
    encoding="utf-8",
)

__all__ = ["logger"]
