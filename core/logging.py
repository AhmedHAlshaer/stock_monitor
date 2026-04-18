"""
JSON structured logging (structlog) for ``core/`` and ``jobs/`` — Railway-friendly.

Import :func:`get_logger` after calling :func:`configure_structlog` once at process entry.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def configure_structlog() -> None:
    """Emit one JSON object per line to stdout. Respect ``LOG_LEVEL`` (default INFO)."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Structured logger (JSON lines)."""
    return structlog.get_logger(name)
