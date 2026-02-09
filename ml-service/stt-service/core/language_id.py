"""Language identification for the STT pipeline.

Note: Faster-Whisper includes built-in language detection which is more accurate.
This module provides a fallback or can be replaced with a dedicated language ID model.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from loguru import logger


LANGUAGE_FALLBACK = "en"


def detect_language(audio_data: Union[np.ndarray, bytes], hint: str | None) -> str:
    """Detect or return language for the audio.

    Args:
        audio_data: Audio as numpy array or bytes (for backward compatibility)
        hint: Optional language hint from user

    Returns:
        ISO language code (e.g., "en", "hi", "ta")
    """
    if hint:
        # Normalize language hint (strip region codes for Whisper compatibility)
        normalized = hint.split("-")[0].lower()
        logger.debug("Using provided language hint: %s (normalized from %s)", normalized, hint)
        return normalized

    # For actual language detection, we rely on Whisper's built-in detection
    # This is just a placeholder fallback
    if isinstance(audio_data, np.ndarray):
        fingerprint = len(audio_data) % 4
    else:
        fingerprint = len(audio_data) % 4

    prediction = {
        0: "en",
        1: "hi",
        2: "ta",
        3: "te",
    }.get(fingerprint, LANGUAGE_FALLBACK)

    logger.debug("Language detection placeholder: %s", prediction)
    return prediction
