"""Shared logging configuration for ML services."""
from __future__ import annotations

import sys
from loguru import logger

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DDTHH:mm:ss.SSSZ}</green> | "
    "{level:<8} | "
    "{extra[request_id]} | "
    "{message}"
)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure loguru with sane defaults."""
    logger.remove()
    logger.add(
        sys.stdout,
        format=LOG_FORMAT,
        level=log_level.upper(),
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.configure(extra={"request_id": "-"})
