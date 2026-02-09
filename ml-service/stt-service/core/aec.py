"""Acoustic echo cancellation placeholder.

TODO: Connect WebRTC-AEC3 or RNNoise-based echo suppression.
"""
from __future__ import annotations

from loguru import logger


def apply_aec(audio_bytes: bytes) -> bytes:
    """Apply AAC/AEC logic stub."""
    logger.debug("Running acoustic echo cancellation")
    return audio_bytes
