"""Structured JSON logging for the FraudGuard API.

One line of JSON per event, which is what log aggregators actually want. The
application logger is configured independently of Uvicorn's own handlers so the
two do not duplicate each other's output.

Privacy rule enforced by usage (see ``app/main.py``): callers pass only derived
values as structured extras — decision, probability, latency. Raw transaction
feature values are never logged, and neither are secrets.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Final

#: Standard ``LogRecord`` attributes, excluded from the structured extras.
_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)

_LOGGER_NAME: Final[str] = "fraudguard.api"
_configured = False


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure the application logger once. Safe to call repeatedly."""
    global _configured

    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        logger.setLevel(level.upper())
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())

    logger.handlers.clear()
    logger.addHandler(handler)
    # Keep our JSON lines separate from Uvicorn's default text handlers.
    logger.propagate = False

    try:
        logger.setLevel(level.upper())
    except (ValueError, TypeError):
        logger.setLevel(logging.INFO)

    _configured = True


def get_logger() -> logging.Logger:
    """Return the application logger."""
    return logging.getLogger(_LOGGER_NAME)


__all__ = ["JsonFormatter", "configure_logging", "get_logger"]
