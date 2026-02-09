"""Inverse text normalization placeholder.

TODO: Connect NeMo/ITN grammars for high fidelity textualization.
"""
from __future__ import annotations

from loguru import logger


def apply_itn(text: str, language: str) -> str:
    """Normalize numerals and entities."""
    logger.debug("Applying ITN for language {}", language)
    return text
