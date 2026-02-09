"""Language identification placeholder for the TTS pipeline.

TODO: Replace with FastText/Compact LID model integration.
"""
from __future__ import annotations

from loguru import logger


def detect_language(text: str, hint: str | None) -> str:
    """Return a best-effort language selection."""
    if hint:
        logger.debug("Using provided language hint: {}", hint)
        return hint
    logger.debug("Detecting language for text length {}", len(text))
    return "en-US"
