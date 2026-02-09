"""Language model rescoring placeholder.

TODO: Wire to domain-specific LM for contextual biasing.
"""
from __future__ import annotations

from loguru import logger


def refine_transcript(text: str, language: str) -> str:
    """Apply simple heuristics to improve transcript."""
    logger.debug("Refining transcript with language model for {}", language)
    return text.strip()
