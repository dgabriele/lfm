"""Logging utilities for the LFM framework.

Provides a pre-configured logger factory that emits structured, consistent log
output across all LFM components.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track which loggers have already been configured to avoid duplicate handlers.
_configured_loggers: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with structured format.

    Loggers are configured once with a ``StreamHandler`` writing to *stderr*
    using a uniform format.  Subsequent calls with the same *name* return the
    same logger without adding extra handlers.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)

    if name not in _configured_loggers:
        _configured_loggers.add(name)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Prevent messages from propagating to the root logger and being
        # printed twice when a root handler is also present.
        logger.propagate = False

    return logger
