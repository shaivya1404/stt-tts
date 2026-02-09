"""Primary ASR module using Faster-Whisper.

This module serves as the primary ASR engine. In production, this could be
replaced with a custom Conformer-RNNT model. Currently, it delegates to
the Faster-Whisper implementation for consistent high-quality results.

The fallback mechanism in pipeline.py will only trigger if confidence < 0.7,
but since both primary and fallback use the same model, results are consistent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger

# Import the shared Whisper implementation
from . import asr_whisper_fallback

# Model identifier for this "primary" path
PRIMARY_MODEL_ID = "whisper_large-v3:primary"


def transcribe(
    audio_data: np.ndarray,
    language: str | None,
) -> Tuple[str, float, List[Dict[str, Any]], str]:
    """Transcribe audio using the primary ASR model.

    Currently delegates to Faster-Whisper. In the future, this could be
    replaced with a custom Conformer-RNNT model trained on Indic languages.

    Args:
        audio_data: Numpy array of audio samples (float32, 16kHz mono)
        language: Optional language hint (e.g., "en", "hi", "ta")

    Returns:
        Tuple of (text, confidence, timestamps, model_used):
            - text: Full transcribed text
            - confidence: Average confidence score (0.0 to 1.0)
            - timestamps: List of word-level timestamps
            - model_used: Model identifier string
    """
    logger.debug(
        "Primary ASR (Conformer-RNNT wrapper): delegating to Faster-Whisper"
    )

    # Delegate to Faster-Whisper
    text, confidence, timestamps, _ = asr_whisper_fallback.transcribe(
        audio_data=audio_data,
        language=language,
    )

    # Return with primary model identifier
    return text, confidence, timestamps, PRIMARY_MODEL_ID


def get_model_info() -> Dict[str, Any]:
    """Get information about the primary ASR model.

    Returns:
        Dict with model name, status, and capabilities
    """
    whisper_info = asr_whisper_fallback.get_model_info()
    return {
        "name": "conformer_rnnt_wrapper",
        "id": PRIMARY_MODEL_ID,
        "backend": whisper_info,
        "note": "Currently uses Faster-Whisper as backend. Replace with custom Conformer-RNNT for production.",
    }
