"""Truecasing placeholder.

TODO: Implement neural truecasing for sentence-level polishing.
"""
from __future__ import annotations

from loguru import logger


def apply_truecase(text: str, language: str) -> str:
    """Capitalize sentences."""
    logger.debug("Applying truecasing for language {}", language)
    return text.capitalize()
