"""HiFi-GAN vocoder pass-through.

XTTS v2 includes an integrated vocoder, so this module serves as a pass-through
for backward compatibility with the pipeline architecture.

In the future, this could be replaced with a separate HiFi-GAN vocoder for
custom acoustic models that output mel spectrograms instead of waveforms.
"""
from __future__ import annotations

from loguru import logger


def vocode(audio_data: bytes, language: str) -> bytes:
    """Pass through audio data (vocoding is integrated in XTTS v2).

    Args:
        audio_data: Audio waveform bytes (already vocoded by XTTS v2)
        language: Language code (unused, kept for API compatibility)

    Returns:
        The same audio data unchanged
    """
    logger.debug("Vocoder pass-through (XTTS v2 includes integrated vocoder)")
    return audio_data
