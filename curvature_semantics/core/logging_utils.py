"""Structured logging with rich formatting and optional W&B sink."""

from __future__ import annotations

import logging
import sys
from typing import Any

try:
    from rich.logging import RichHandler
    _RICH = True
except ImportError:
    _RICH = False


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _RICH:
        handler: logging.Handler = RichHandler(rich_tracebacks=True, markup=True)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_config(logger: logging.Logger, config: dict[str, Any]) -> None:
    logger.info("Experiment config:")
    for k, v in config.items():
        if not k.startswith("_"):
            logger.info("  %s: %s", k, v)
