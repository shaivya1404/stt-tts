"""Emotion and style conditioning placeholder.

TODO: Replace with GST/style-token model or diffusion-based prosody control.
"""
from __future__ import annotations

from loguru import logger


def apply_style_emotion(emotion: str | None, language: str) -> dict:
    """Return style parameters used by downstream models."""
    style = {
        "emotion": emotion or "neutral",
        "language": language,
        "energy": 0.5,
    }
    logger.debug("Applying style/emotion: {}", style)
    return style
